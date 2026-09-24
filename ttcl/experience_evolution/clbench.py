"""Frozen ALFWorld writer transfer to existing CLBench development tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import traceback
from types import SimpleNamespace

from .core import WRITER_SYSTEM, digest, save, seed, workspace_root, python_executable

WORKSPACE = workspace_root()
BENCH = WORKSPACE / "current_work/continual-learning-bench"
PYTHON = str(python_executable(WORKSPACE))
ORIGIN = WORKSPACE / "ttcl/results/experience_evolution/alfworld_delta_20260922"
ARMS = ["none", "untrained", "delta", "delta_reset"]


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def writer_messages(memory, episode):
    """Only completed public interaction and the declared terminal scalar."""
    public = {
        "initial_observation": {
            "brief": episode["public_task_brief"],
            "query": episode["initial_public_query"],
            "response_schemas": episode["response_schemas"],
        },
        "trajectory": episode["steps"],
        "reward": episode["reward"],
        "steps": len(episode["steps"]),
        "completed": episode["completed"],
        "format_failures": episode["format_failures"],
    }
    return [
        {"role": "system", "content": WRITER_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"previous_experience": memory, "completed_interaction": public},
                ensure_ascii=False,
            ),
        },
    ]


def prepare(root, port):
    root.mkdir(parents=True, exist_ok=False)
    prior = read(
        WORKSPACE
        / "ttcl/results/experience_training/next_reward_sft_20260921/plan.json"
    )
    plan = {
        "created_unix": time.time(),
        "origin": str(ORIGIN),
        "tasks": ["database_exploration", "cohort_studies"],
        "indices": list(range(12, 20)),
        "repeats": [303, 404],
        "arms": ARMS,
        "environment_seed": 42,
        "model": prior["model"],
        "adapter": str(root / "adapter"),
        "api_url": f"http://127.0.0.1:{port}",
        "actor_name": "frozen-actor",
        "writer_name": "delta-writer",
        "writer_tokens": 384,
        "writer_temperature": 1.0,
        "writer_top_p": 1.0,
        "context_limit": 65536,
        "expected_cells": 128,
        "reward_visibility": "Completed-episode scalar supplied to writer, as in prior local experiments; this is not the strict score-hidden cohort protocol.",
        "scope": "Existing CLBench development environments; no CLBench parameter training or checkpoint selection.",
        "backend": "vLLM 0.9.1 for every arm; all controls rerun. Actor always requests base model, only trained writers request LoRA.",
        "reuse": "Identical actor context at same task/index/seed and identical writer input/model/seed reused explicitly within each worker.",
    }
    shutil.copytree(ORIGIN / "training/delta/adapter", root / "adapter")
    package = root / "source/ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    shutil.copy2(WORKSPACE / "ttcl/paths.py", package / "paths.py")
    for name in ["common", "llm_memory", "structured_memory", "experience_evolution"]:
        shutil.copytree(
            WORKSPACE / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    save(root / "plan.json", plan)
    (root / "WRITER_PROMPT.txt").write_text(WRITER_SYSTEM)
    protocol = """# ALFWorld Delta writer → CLBench

Frozen existing Delta adapter; no training or checkpoint selection on CLBench.
Database multi_group and Cohort, canonical indices 12..19, environment seed 42,
actor sampling repeats 303/404. Every task/repeat/arm begins with empty memory.
Arms: none, base writer, Delta cumulative writer, Delta latest-trajectory-only writer.
All arms use the existing official run_episode task/score/format-retry implementation.
Actor: frozen base Qwen3-4B, temperature .7, top_p .9, output 4096, context 65536.
Writer: exact ALFWorld system prompt, temperature 1, top_p 1, output 384.
Writer context raised from 12288 to 65536 to admit full CLBench public trajectories;
no silent truncation. The only cross-episode state is the generated text.
Writer receives completed public actions/observations and the official terminal scalar,
including Cohort, matching the existing local reward-visible experimental variant;
not a standard score-hidden Cohort leaderboard run. No hidden labels or scorer metadata.
All parameters frozen; actor requests base endpoint, trained writers request LoRA.
Same first-episode executions and otherwise identical inputs explicitly reused.
The last episode has no writer update, since no subsequent task consumes it.
Infrastructure failures abort a worker. Invalid final-format task failures are retained
as missing scores, never changed to zero; their visible trajectories may be summarized.
Report each task separately, full and excluding first episode, common and pairwise
complete outcomes, failures, writer truncation and cost. This is a small development
transfer test, not an unseen CLBench environment or statistically stable ranking.
"""
    (root / "PROTOCOL.md").write_text(protocol)
    paths = [
        p
        for sub in ["adapter", "source"]
        for p in (root / sub).rglob("*")
        if p.is_file()
    ]
    paths += [root / "plan.json", root / "WRITER_PROMPT.txt", root / "PROTOCOL.md"]
    paths += list((BENCH / "src").rglob("*.py"))
    save(root / "input_hashes.json", {str(p): file_hash(p) for p in paths})
    save(root / "status.json", {"phase": "prepared", "expected_cells": 128})


class RemoteModel:
    def __init__(self, plan, repeat):
        import requests
        from transformers import AutoTokenizer

        self.plan, self.repeat = plan, repeat
        self.tokenizer = AutoTokenizer.from_pretrained(
            plan["model"]["model"], local_files_only=True
        )
        self.session = requests.Session()
        self.session.trust_env = False

    def complete(self, messages, random_seed, trained, tokens, temperature, top_p):
        from ttcl.common.local_qwen import check_context

        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        check_context(count, tokens, self.plan["context_limit"])
        model = self.plan["writer_name"] if trained else self.plan["actor_name"]
        # Raw rendered completions avoid any server-side chat-template divergence.
        payload = {
            "model": model,
            "prompt": rendered,
            "seed": random_seed % (2**32),
            "max_tokens": tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "add_special_tokens": False,
        }
        result = self.session.post(
            self.plan["api_url"] + "/v1/completions", json=payload, timeout=900
        )
        result.raise_for_status()
        data = result.json()
        if data["usage"]["prompt_tokens"] != count:
            raise ValueError("Server/client tokenizer mismatch")
        choice = data["choices"][0]
        return {
            "raw_response": choice["text"].strip(),
            "input_tokens": count,
            "output_tokens": data["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"],
            "context_limit": self.plan["context_limit"],
            "rendered_prompt_sha256": digest(rendered),
            "actual_generation_seed": payload["seed"],
            "writer_adapter_enabled": trained,
            "served_model": model,
        }

    def generate(self, messages, random_seed):
        from ttcl.structured_memory.run_benchmark import generation_seed

        return self.complete(
            messages,
            generation_seed(random_seed, "actor_repeat", self.repeat),
            False,
            self.plan["model"]["max_new_tokens"],
            self.plan["model"]["temperature"],
            self.plan["model"]["top_p"],
        )


def worker(root, task, repeat):
    from ttcl.structured_memory.online_bank import run_episode

    plan = read(root / "plan.json")
    os.chdir(BENCH)
    model = RemoteModel(plan, repeat)
    args = SimpleNamespace(
        **plan["model"],
        task=task,
        seed=plan["environment_seed"],
        num_instances=20,
        allow_initial_experience=True,
    )
    memory = {a: "" for a in ARMS}
    writer_cache = {}
    progress = root / "workers" / f"{task}_{repeat}.json"
    for index in plan["indices"]:
        actor_cache = {}
        for arm in ARMS:
            dest = (
                root
                / "evaluation"
                / task
                / str(repeat)
                / arm
                / f"episode_{index + 1:03d}"
            )
            save(
                progress,
                {
                    "phase": "actor",
                    "task": task,
                    "repeat": repeat,
                    "index": index,
                    "arm": arm,
                    "time": time.time(),
                },
            )
            context = memory[arm]
            key = digest(context)
            if key in actor_cache:
                old_row, episode, original = actor_cache[key]
                dest.mkdir(parents=True, exist_ok=False)
                row = dict(old_row, reused_from=str(original))
                save(dest / "trajectory.json", episode)
            else:
                row, episode = run_episode(args, model, index, dest, context)
                # Transport/context failures cannot become ordinary task failures.
                error = row.get("error", "")
                if row["status"] != "complete" and not (
                    "no schema-valid JSON action" in error
                    or "Safety cap exceeded" in error
                ):
                    raise RuntimeError(
                        f"Infrastructure or unexpected environment failure: {error}"
                    )
                actor_cache[key] = (dict(row), episode, dest)
            row.update(task=task, arm=arm, repeat=repeat, actor_adapter_enabled=False)
            save(dest / "memory_before.json", {"text": context, "sha256": key})
            save(dest / "row.json", row)
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in [
                            "task",
                            "arm",
                            "repeat",
                            "episode",
                            "status",
                            "reward",
                            "reused_from",
                        ]
                    }
                ),
                flush=True,
            )
            if arm == "none" or index == plan["indices"][-1]:
                continue
            save(
                progress,
                {
                    "phase": "writer",
                    "task": task,
                    "repeat": repeat,
                    "index": index,
                    "arm": arm,
                    "time": time.time(),
                },
            )
            previous = "" if arm == "delta_reset" else context
            messages = writer_messages(previous, episode)
            trained = arm != "untrained"
            random_seed = seed(20260922, task, repeat, index, "writer")
            cache_key = (
                digest(json.dumps(messages, ensure_ascii=False)),
                trained,
                random_seed,
            )
            if cache_key in writer_cache:
                old, path = writer_cache[cache_key]
                update = dict(old, reused_from=str(path))
            else:
                started = time.monotonic()
                completion = model.complete(
                    messages,
                    random_seed,
                    trained,
                    plan["writer_tokens"],
                    plan["writer_temperature"],
                    plan["writer_top_p"],
                )
                if not completion["raw_response"]:
                    raise ValueError("Empty experience output")
                update = {
                    "messages": messages,
                    **completion,
                    "seconds": time.monotonic() - started,
                }
                writer_cache[cache_key] = (update, dest / "writer.json")
            memory[arm] = update["raw_response"]
            save(dest / "writer.json", update)
    save(
        progress,
        {"phase": "complete", "task": task, "repeat": repeat, "time": time.time()},
    )


def summarize(rows, arms):
    groups = {
        a: {
            (r["repeat"], r["canonical_index"]): r
            for r in rows
            if r["arm"] == a and r["status"] == "complete"
        }
        for a in arms
    }
    keys = sorted(set.intersection(*(set(g) for g in groups.values())))
    for key in keys:
        if len({groups[a][key]["instance_id"] for a in arms}) != 1:
            raise ValueError("Mismatched task identities")
    result = {"paired_count": len(keys), "arms": {}}
    for arm in arms:
        values = [groups[arm][k]["reward"] for k in keys]
        diffs = [groups[arm][k]["reward"] - groups[arms[0]][k]["reward"] for k in keys]
        result["arms"][arm] = {
            "mean_reward": statistics.mean(values) if values else None,
            "delta_vs_" + arms[0]: statistics.mean(diffs) if diffs else None,
            "wins": sum(v > 1e-12 for v in diffs),
            "losses": sum(v < -1e-12 for v in diffs),
            "ties": sum(abs(v) <= 1e-12 for v in diffs),
        }
    return result


def report(root):
    plan = read(root / "plan.json")
    rows = [read(p) for p in sorted((root / "evaluation").rglob("row.json"))]
    result = {}
    lines = [
        "# ALFWorld Delta → CLBench",
        "",
        f"已记录 {len(rows)}/{plan['expected_cells']} 个单元；各任务单独比较官方 reward，越高越好。",
        "所有参数冻结；每条序列从空经验开始。沿用此前每题结束后向 writer 提供总分的变体。",
        "相同输入显式复用；失败保留为缺失，不补零。部分进度不可视为最终排名。",
        "",
    ]
    for task in plan["tasks"]:
        selected = [r for r in rows if r["task"] == task]
        later = [r for r in selected if r["canonical_index"] != plan["indices"][0]]
        result[task] = {
            "all": summarize(selected, ARMS),
            "after_first": summarize(later, ARMS),
            "pairwise": {a: summarize(selected, ["none", a]) for a in ARMS[1:]},
            "training_effect": summarize(selected, ["untrained", "delta"]),
            "failures": [r for r in selected if r["status"] != "complete"],
        }
        lines += [
            f"## {task}",
            "",
            f"四组共同有效配对：{result[task]['all']['paired_count']}/16。",
            "",
            "| 组别 | 已记录 | 失败 | 共同配对均分 | 去掉首题均分 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            own = [r for r in selected if r["arm"] == arm]

            def fmt(v):
                return "—" if v is None else f"{v:.6f}"

            lines.append(
                f"| {arm} | {len(own)}/16 | {sum(r['status'] != 'complete' for r in own)} | "
                f"{fmt(result[task]['all']['arms'][arm]['mean_reward'])} | "
                f"{fmt(result[task]['after_first']['arms'][arm]['mean_reward'])} |"
            )
    writers = [read(p) for p in (root / "evaluation").rglob("writer.json")]
    result["cost"] = {
        "executed_actor_calls": sum(
            r["actor_calls"] for r in rows if not r.get("reused_from")
        ),
        "writer_calls": sum(not w.get("reused_from") for w in writers),
        "writer_length_stops": sum(w["finish_reason"] == "length" for w in writers),
        "actor_reused_cells": sum(bool(r.get("reused_from")) for r in rows),
    }
    lines += [
        "",
        "这是既有开发环境上的小规模迁移测试，非全新环境泛化结论；参数没有在 CLBench 更新。",
        "数据库 reward 同时计入答对和查询效率；队列研究 reward 为相对基线的信息增益，不能横向合并成成功率。",
        "完整协议见 PROTOCOL.md，逐题输出见 evaluation/，配对与缺失详情见 comparison.json。",
    ]
    save(root / "comparison.json", result)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return rows


def audit(root):
    plan = read(root / "plan.json")
    errors = [
        p for p, h in read(root / "input_hashes.json").items() if file_hash(p) != h
    ]
    for task in plan["tasks"]:
        for repeat in plan["repeats"]:
            for arm in ARMS:
                memory = ""
                for index in plan["indices"]:
                    dest = (
                        root
                        / "evaluation"
                        / task
                        / str(repeat)
                        / arm
                        / f"episode_{index + 1:03d}"
                    )
                    row, before = (
                        read(dest / "row.json"),
                        read(dest / "memory_before.json"),
                    )
                    if before["text"] != memory or row["bank_context_sha256"] != digest(
                        memory
                    ):
                        errors.append(str(dest) + ": memory chronology")
                    trace = Path(row.get("reused_from", dest)) / "responses.jsonl"
                    if trace.exists():
                        for event in map(json.loads, trace.read_text().splitlines()):
                            if (
                                event["writer_adapter_enabled"]
                                or event["served_model"] != plan["actor_name"]
                            ):
                                errors.append(str(trace) + ": actor used adapter")
                    if arm == "none" or index == plan["indices"][-1]:
                        continue
                    update, episode = (
                        read(dest / "writer.json"),
                        read(dest / "trajectory.json"),
                    )
                    if update["messages"] != writer_messages(
                        "" if arm == "delta_reset" else memory, episode
                    ):
                        errors.append(str(dest) + ": writer payload")
                    if update["writer_adapter_enabled"] != (arm != "untrained"):
                        errors.append(str(dest) + ": writer adapter")
                    memory = update["raw_response"]
    save(root / "integrity.json", {"passed": not errors, "errors": errors})
    if errors:
        raise ValueError(f"Integrity audit failed: {errors[:5]}")


def supervise(root, gpu):
    import requests

    plan = read(root / "plan.json")
    env = {
        **os.environ,
        "TTCL_WORKSPACE": str(WORKSPACE),
        "TTCL_BENCH": str(BENCH),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": os.pathsep.join(
            map(
                str,
                [
                    WORKSPACE / "ttcl/.runtime/structured_memory_deps",
                    root / "source",
                    BENCH,
                ],
            )
        ),
    }
    server_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    port = plan["api_url"].rsplit(":", 1)[1]
    cmd = [
        PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        plan["model"]["model"],
        "--served-model-name",
        plan["actor_name"],
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(plan["context_limit"]),
        "--gpu-memory-utilization",
        "0.88",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
        "--disable-log-requests",
        "--enable-lora",
        "--max-lora-rank",
        "8",
        "--lora-modules",
        plan["writer_name"] + "=" + plan["adapter"],
    ]
    save(root / "server_command.json", cmd)
    server = None
    workers = []
    try:
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(
                cmd,
                env=server_env,
                cwd=root / "source",
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        save(
            root / "status.json", {"phase": "loading_server", "server_pid": server.pid}
        )
        session = requests.Session()
        session.trust_env = False
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("Model server exited; inspect server.log")
            try:
                response = session.get(plan["api_url"] + "/v1/models", timeout=3)
                names = {v["id"] for v in response.json().get("data", [])}
                if {plan["actor_name"], plan["writer_name"]} <= names:
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Model server startup timeout")
        for task in plan["tasks"]:
            for repeat in plan["repeats"]:
                command = [
                    PYTHON,
                    "-m",
                    "ttcl.experience_evolution.clbench",
                    "worker",
                    "--root",
                    str(root),
                    "--task",
                    task,
                    "--repeat",
                    str(repeat),
                ]
                with (root / f"{task}_{repeat}.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        cwd=root / "source",
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                workers.append((task, repeat, process))
        while True:
            rows = report(root)
            statuses = [
                {"task": t, "repeat": r, "pid": p.pid, "exit_code": p.poll()}
                for t, r, p in workers
            ]
            save(
                root / "status.json",
                {
                    "phase": "evaluating",
                    "recorded_cells": len(rows),
                    "expected_cells": plan["expected_cells"],
                    "workers": statuses,
                },
            )
            if any(p.poll() not in (None, 0) for _, _, p in workers):
                raise RuntimeError("An evaluation worker failed; inspect worker logs")
            if all(p.poll() == 0 for _, _, p in workers):
                break
            if server.poll() is not None:
                raise RuntimeError("Model server exited during evaluation")
            time.sleep(10)
        if len(rows) != plan["expected_cells"]:
            raise RuntimeError("Incomplete evaluation cell count")
        audit(root)
        save(
            root / "status.json",
            {
                "phase": "complete",
                "recorded_cells": len(rows),
                "failed_cells": sum(r["status"] != "complete" for r in rows),
                "expected_cells": plan["expected_cells"],
            },
        )
    except BaseException:
        save(
            root / "status.json",
            {"phase": "failed", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        for _, _, process in workers:
            if process.poll() is None:
                process.terminate()
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["prepare", "supervise", "worker", "report", "audit"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18207)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task")
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "prepare":
        prepare(root, args.port)
    elif args.phase == "supervise":
        supervise(root, args.gpu)
    elif args.phase == "worker":
        worker(root, args.task, args.repeat)
    else:
        globals()[args.phase](root)


if __name__ == "__main__":
    main()
