from __future__ import annotations

import concurrent.futures
import json
import re
import time
from pathlib import Path

import requests

from .core import digest, save, seed

ACTOR_SYSTEM = """Complete the household task by interacting with a text environment. You receive observations and the currently available commands. Choose one available command at each turn. Return ONLY that command on a single line, without explanation, quotation marks, or markdown. The task is complete only when the environment reports success. Past experience, when provided, may concern different objects or rooms: use it only where applicable to the current observations."""


def make_env(game):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    # No expert, policy commands, hidden facts, or ground truth is exposed.
    return textworld.start(str(game), request_infos=textworld.EnvInfos(
        won=True, admissible_commands=True), wrappers=[AlfredDemangler(shuffle=False)])


def clean_command(text, available):
    text = text.strip()
    if text in available:
        return text
    lines = [re.sub(r"^(?:action|command)\s*:\s*", "", l.strip(), flags=re.I).strip("`\"'")
             for l in text.splitlines() if l.strip()]
    matched = [x for x in lines if x in available]
    return matched[-1] if matched else (lines[-1] if lines else "")


class Actor:
    def __init__(self, plan):
        self.plan = plan
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    def generate(self, messages, random_seed):
        payload = {"model": "frozen-actor", "messages": messages,
                   "temperature": self.plan["actor_temperature"], "top_p": 1.0, "top_k": -1,
                   "max_tokens": self.plan["actor_max_tokens"], "seed": random_seed}
        started = time.monotonic()
        r = requests.post(self.plan["actor_url"] + "/v1/chat/completions", json=payload, timeout=240)
        r.raise_for_status()
        value = r.json()
        return {"text": value["choices"][0]["message"]["content"],
                "finish_reason": value["choices"][0]["finish_reason"],
                "usage": value.get("usage", {}), "seed": random_seed,
                "prompt_sha256": digest(json.dumps(messages, ensure_ascii=False)),
                "seconds": time.monotonic() - started}

    def run_many(self, jobs):
        """Interleave independent official environments; every job gets a fresh reset."""
        active, results = [], [None] * len(jobs)
        try:
            for i, job in enumerate(jobs):
                path = Path(job["output"])
                if (path / "episode.json").exists():
                    raise FileExistsError(path)
                game = Path(self.plan["data_root"]) / job["game"]
                env = make_env(game)
                state = env.reset()
                initial = str(state["feedback"])
                available = list(state["admissible_commands"])
                system = ACTOR_SYSTEM
                if job["memory"]:
                    system += "\n\nPast experience:\n" + job["memory"]
                messages = [{"role": "system", "content": system}]
                item = {"i": i, "job": job, "env": env, "state": state, "messages": messages,
                        "episode": {"game": job["game"], "memory": job["memory"],
                        "memory_sha256": digest(job["memory"]), "seed": job["seed"],
                        "initial_observation": initial, "initial_commands_sha256": digest(json.dumps(available)),
                        "trajectory": [], "generations": [], "reward": 0.0, "steps": 0,
                        "actor_adapter_enabled": False}}
                active.append(item)
            for turn in range(self.plan["max_steps"]):
                if not active:
                    break
                pending = []
                for item in active:
                    feedback = str(item["state"]["feedback"])
                    available = list(item["state"]["admissible_commands"])
                    item["messages"].append({"role": "user", "content": feedback +
                        "\nAvailable commands:\n" + "\n".join(available)})
                    pending.append(self.pool.submit(self.generate, item["messages"],
                                   seed(item["job"]["seed"], turn)))
                remaining = []
                for item, future in zip(active, pending):
                    completion = future.result()  # Infrastructure failure is not a task reward.
                    command = clean_command(completion["text"], item["state"]["admissible_commands"])
                    was_valid = command in item["state"]["admissible_commands"]
                    state, score, done = item["env"].step(command)
                    item["messages"].append({"role": "assistant", "content": completion["text"]})
                    ep = item["episode"]
                    ep["generations"].append(completion)
                    ep["trajectory"].append({"action": command, "observation": str(state["feedback"]),
                                             "valid_command": was_valid})
                    ep.update(reward=float(bool(state["won"])), steps=turn + 1)
                    item["state"] = state
                    if done or state["won"] or turn + 1 == self.plan["max_steps"]:
                        ep["status"] = "complete"
                        ep["termination"] = "success" if state["won"] else "budget_or_environment_done"
                        save(Path(item["job"]["output"]) / "episode.json", ep)
                        results[item["i"]] = ep
                        item["env"].close()
                    else:
                        remaining.append(item)
                active = remaining
        finally:
            for item in active:
                item["env"].close()
        if any(x is None for x in results):
            raise RuntimeError("Incomplete actor execution")
        return results
