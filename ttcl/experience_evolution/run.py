from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .core import read, save, append, seed, writer_messages, reward_signal, prepare


def job(plan, seq, position, memory, random_seed, output):
    return {"game": seq["games"][position], "memory": memory, "seed": random_seed,
            "output": str(output)}


def check_pair(with_ep, without_ep):
    for key in ["game", "seed", "initial_observation", "initial_commands_sha256"]:
        if with_ep[key] != without_ep[key]:
            raise AssertionError(f"Mismatched paired reset: {key}")
    assert not without_ep["memory"]


def train(root, mode):
    from .environment import Actor
    from .writer import Writer
    plan = read(root / "plan.json")
    out = root / "training" / mode
    out.mkdir(parents=True, exist_ok=False)
    save(out / "status.json", {"phase": "loading_writer"})
    writer, actor = Writer(plan, train=True), Actor(plan)
    totals = []
    for batch_i in range(plan["train_updates"]):
        sequences = plan["training"][batch_i*4:(batch_i+1)*4]
        directory = out / f"batch_{batch_i:03d}"
        save(out / "status.json", {"phase": "rollout", "batch": batch_i, "total": plan["train_updates"]})
        states = [""] * len(sequences)
        current = actor.run_many([job(plan, seq, 0, "", seed(722, seq["id"], 0),
                    directory / f"seq_{k}" / "task_0") for k,seq in enumerate(sequences)])
        samples = []
        for position in range(1, plan["sequence_length"]):
            dirs = [directory / f"seq_{k}" / f"update_{position-1}" for k in range(len(sequences))]
            messages = [writer_messages(m, ep) for m,ep in zip(states, current)]
            candidates = writer.generate(messages, seed(722, "writer", batch_i, position), dirs)
            tasks = []
            for k,(seq, sample) in enumerate(zip(sequences, candidates)):
                random_seed = seed(722, seq["id"], position)
                tasks.extend([job(plan, seq, position, sample["text"], random_seed,
                                   directory / f"seq_{k}" / f"task_{position}"),
                              job(plan, seq, position, "", random_seed,
                                   directory / f"seq_{k}" / f"baseline_{position}")])
            outcomes = actor.run_many(tasks)
            current = []
            for k,sample in enumerate(candidates):
                positive, baseline = outcomes[2*k:2*k+2]
                check_pair(positive, baseline)
                sample.update(advantage=reward_signal(positive["reward"], baseline["reward"], mode),
                              delta=positive["reward"]-baseline["reward"],
                              with_reward=positive["reward"], without_reward=baseline["reward"],
                              sequence=sequences[k]["id"], position=position)
                save(dirs[k] / "label.json", {a: sample[a] for a in ["advantage", "delta", "with_reward",
                     "without_reward", "sequence", "position", "prompt_sha256"]})
                samples.append(sample)
                # Only the actual memory-guided path is carried into the next update.
                states[k] = sample["text"]
                current.append(positive)
        save(out / "status.json", {"phase": "gradient_update", "batch": batch_i})
        update = writer.update(samples)
        update.update(batch=batch_i, examples=len(samples), time=time.time())
        totals.append(update)
        append(out / "training.jsonl", update)
        print(json.dumps(update), flush=True)
    audit = writer.save(out)
    result = {"phase": "complete", "mode": mode, "batches": len(totals),
              "writer_examples": sum(x["examples"] for x in totals),
              "positive": sum(x["positive"] for x in totals),
              "negative": sum(x["negative"] for x in totals),
              "zero": sum(x["zero"] for x in totals), "audit": audit}
    save(out / "status.json", result)


def evaluate(root, arm):
    from .environment import Actor
    from .writer import Writer
    plan = read(root / "plan.json")
    out = root / "evaluation" / arm
    out.mkdir(parents=True, exist_ok=False)
    save(out / "status.json", {"phase": "loading_writer"})
    adapter = root / "training" / ("delta" if arm == "delta_reset" else arm) / "adapter"
    writer = None if arm == "none" else Writer(plan,
               adapter=None if arm == "untrained" else adapter, train=False)
    actor, rows = Actor(plan), []
    for repeat in plan["eval_seeds"]:
        for start in range(0, len(plan["evaluation"]), 4):
            sequences = plan["evaluation"][start:start+4]
            states = [""] * len(sequences)
            current = None
            for position in range(plan["sequence_length"]):
                if position and writer:
                    messages = [writer_messages("" if arm == "delta_reset" else m, ep)
                                for m,ep in zip(states,current)]
                    dirs = [out / str(repeat) / seq["id"].replace(":", "_") / f"update_{position-1}"
                            for seq in sequences]
                    generated = writer.generate(messages, seed(repeat, "eval_writer", start, position), dirs)
                    states = [s["text"] for s in generated]
                jobs = [job(plan, seq, position, states[k], seed(repeat, seq["id"], position),
                        out / str(repeat) / seq["id"].replace(":", "_") / f"task_{position}")
                        for k,seq in enumerate(sequences)]
                current = actor.run_many(jobs)
                for seq,ep in zip(sequences,current):
                    row = {"sequence": seq["id"], "family": seq["family"], "repeat": repeat,
                           "position": position, "arm": arm, "game": ep["game"], "seed": ep["seed"],
                           "reward": ep["reward"], "steps": ep["steps"], "memory_sha256": ep["memory_sha256"],
                           "initial_sha256": __import__("hashlib").sha256(ep["initial_observation"].encode()).hexdigest()}
                    rows.append(row)
                    append(out / "scores.jsonl", row)
                save(out / "status.json", {"phase": "evaluating", "completed": len(rows),
                     "total": len(plan["evaluation"])*len(plan["eval_seeds"])*plan["sequence_length"]})
    save(out / "status.json", {"phase": "complete", "completed": len(rows)})


def calibration(root):
    from .environment import Actor, make_env
    plan = read(root / "plan.json")
    jobs = [{"game": r["game"], "memory": "", "seed": seed(722, "calibration", i),
             "output": str(root / "calibration" / str(i))} for i,r in enumerate(plan["calibration"])]
    # Check deterministic reset without looking at hidden state.
    for j in jobs:
        env = make_env(Path(plan["data_root"]) / j["game"])
        a = env.reset()
        first = (str(a["feedback"]), list(a["admissible_commands"]))
        b = env.reset()
        assert first == (str(b["feedback"]), list(b["admissible_commands"]))
        env.close()
    rows = Actor(plan).run_many(jobs)
    save(root / "calibration.json", {"episodes": len(rows), "successes": sum(r["reward"] for r in rows),
        "steps": [r["steps"] for r in rows], "reset_check_passed": True,
        "invalid_commands": sum(not t["valid_command"] for r in rows for t in r["trajectory"])})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "calibrate", "train", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--arm")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.root)
        return
    os.environ["ALFWORLD_DATA"] = read(args.root / "plan.json")["data_root"]
    try:
        if args.command == "calibrate":
            calibration(args.root)
        elif args.command == "train":
            train(args.root, args.arm)
        else:
            evaluate(args.root, args.arm)
    except Exception:
        save(args.root / f"failure_{args.command}_{args.arm}.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
