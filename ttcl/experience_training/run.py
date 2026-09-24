"""Collection, writer-only SFT, and isolated sequential held-out evaluation."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from types import SimpleNamespace

from ttcl.experience_training.core import KEEP, comparisons, restore_bank, select
from ttcl.llm_memory.trajectory_bank import UPDATE_PROMPT, encode
from ttcl.structured_memory.online_bank import read, run_episode, sha
from ttcl.structured_memory import run_benchmark as base

ARMS = ["none", "untrained", "utility_sft", "unfiltered_sft"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(path, value)


class Backend:
    """Separate actor RNG repetitions from the fixed canonical environment seed.

    LoRA is explicitly disabled during EVERY actor call, even when the same
    model object generates the writer updates with an adapter enabled.
    """

    def __init__(self, args, adapter=None):
        self.inner = base.LocalQwen(args)
        self.tokenizer = self.inner.tokenizer
        self.adapter = adapter
        self.repeat = 0
        self.writer = False
        if adapter:
            from peft import PeftModel

            self.inner.model = PeftModel.from_pretrained(
                self.inner.model, adapter, is_trainable=False
            ).eval()
        self.inner.model.requires_grad_(False)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages, seed, **kwargs):
        if not self.writer:
            seed = base.generation_seed(seed, "actor_repeat", self.repeat)
        disabled = self.adapter is not None and not self.writer
        with self.inner.model.disable_adapter() if disabled else nullcontext():
            result = self.inner.generate(messages, seed, **kwargs)
        result.update(
            writer_adapter_enabled=bool(self.adapter and self.writer),
            actual_generation_seed=seed,
        )
        return result


def actor_args(plan, task, count):
    return SimpleNamespace(
        **plan["model"],
        task=task,
        seed=plan["environment_seed"],
        num_instances=count,
        allow_initial_experience=True,
    )


def update_writer(bank, episode, model, output, seed, temperature, tokens):
    output.mkdir(parents=True, exist_ok=True)
    model.writer = True
    try:

        def generate(messages, seed, **kwargs):
            kwargs["temperature"] = temperature
            base.append(
                output / "writer_requests.jsonl",
                {"messages": messages, "seed": seed, "options": kwargs},
            )
            start = time.monotonic()
            result = model.generate(messages, seed, **kwargs)
            base.append(
                output / "writer_generations.jsonl",
                {**result, "seconds": time.monotonic() - start},
            )
            return result

        result = bank.update(episode, generate, seed, model.count, output_tokens=tokens)
        save(output / "bank_update.json", result)
        return result
    finally:
        model.writer = False


def load_source(root, task, index):
    source = root / "data" / task / f"episode_{index + 1:03d}"
    episode = read(source / "trajectory.json")
    state = read(source / "bank_before.json")
    if episode["episode"] != index + 1 or state["last_observed"] != index:
        raise ValueError("Source history chronology mismatch")
    for entry in state["entries"]:
        if any(e["episode"] > index for e in entry["evidence"]):
            raise ValueError("Source bank contains future evidence")
    return episode, state


def collect(root, task):
    plan = read(root / "plan.json")
    output = root / "collection" / task
    output.mkdir(parents=True, exist_ok=True)
    args = actor_args(plan, task, 12)
    os.chdir(base.BENCH)
    model = Backend(args)
    selections = []
    for index in plan["source_indices"]:
        directory = output / f"prefix_{index:03d}"
        if (directory / "result.json").exists():
            selections.append(read(directory / "result.json"))
            continue
        save(
            output / "status.json",
            {"phase": "candidate_generation", "source_index": index},
        )
        episode, before = load_source(root, task, index)
        candidates, contexts = (
            {},
            {"keep": restore_bank(before, plan["bank"]).context()},
        )
        canonical_messages = [
            {"role": "system", "content": UPDATE_PROMPT},
            {
                "role": "user",
                "content": encode(restore_bank(before, plan["bank"]).payload(episode)),
            },
        ]
        save(directory / "writer_input.json", canonical_messages)
        for k, temperature in enumerate(plan["candidate_temperatures"]):
            name = f"candidate_{k}"
            dest = directory / name
            bank = restore_bank(before, plan["bank"])
            if (dest / "bank_update.json").exists():
                result = read(dest / "bank_update.json")
                bank = restore_bank(result["bank_after"], plan["bank"])
            else:
                result = update_writer(
                    bank,
                    episode,
                    model,
                    dest,
                    base.generation_seed(811, task, f"{index}:{k}"),
                    temperature,
                    plan["writer_output_tokens"],
                )
            # All attempts must receive exactly the same public payload; retries
            # can append only validation errors. No future query is supplied.
            for attempt in result["attempts"]:
                if not attempt["messages"][1]["content"].startswith(
                    canonical_messages[1]["content"]
                ):
                    raise ValueError("Unexpected writer input")
            candidates[name] = {
                "accepted": result["accepted"],
                "target": result["parsed_update"],
                "error": result["error"],
                "writer_calls": len(result["attempts"]),
                "writer_input_tokens": sum(
                    a.get("completion", {}).get("input_tokens", 0)
                    for a in result["attempts"]
                ),
                "writer_output_tokens": sum(
                    a.get("completion", {}).get("output_tokens", 0)
                    for a in result["attempts"]
                ),
            }
            if result["accepted"]:
                contexts[name] = bank.context()
        # Each seed uses the same next instance and same action budget.
        rewards, records, cache = {}, [], {}
        for name, context in contexts.items():
            rewards[name] = {}
            for repeat in plan["collection_repeats"]:
                save(
                    output / "status.json",
                    {
                        "phase": "next_task_scoring",
                        "source_index": index,
                        "candidate": name,
                        "repeat": repeat,
                    },
                )
                key = (sha(context), repeat)
                destination = directory / "probes" / name / str(repeat)
                if key in cache:
                    row = dict(cache[key], reused_identical_context=True)
                elif (destination / "metrics.json").exists():
                    row = read(destination / "metrics.json")
                else:
                    model.repeat = repeat
                    row, _ = run_episode(args, model, index + 1, destination, context)
                cache[key] = row
                rewards[name][str(repeat)] = (
                    row["reward"] if row["status"] == "complete" else None
                )
                records.append({"candidate": name, "repeat": repeat, **row})
        if len({r["instance_id"] for r in records}) != 1:
            raise ValueError("Candidate probes did not share the same instance")
        choice = select(candidates, rewards, plan["collection_repeats"])
        result = {
            "task": task,
            "source_index": index,
            "probe_index": index + 1,
            "messages": canonical_messages,
            "candidates": candidates,
            "rewards": rewards,
            "records": records,
            **choice,
        }
        save(directory / "result.json", result)
        selections.append(result)
        save(
            output / "summary.json",
            {
                "prefixes": len(selections),
                "positive_update_labels": sum(
                    r["selected"] not in {None, "keep"} for r in selections
                ),
                "keep_labels": sum(r["selected"] == "keep" for r in selections),
                "no_label": sum(r["selected"] is None for r in selections),
                "invalid_candidates": sum(
                    not c["accepted"]
                    for r in selections
                    for c in r["candidates"].values()
                ),
                "actor_calls": sum(
                    v["actor_calls"]
                    for r in selections
                    for v in r["records"]
                    if not v.get("reused_identical_context")
                ),
                "writer_calls": sum(
                    c["writer_calls"]
                    for r in selections
                    for c in r["candidates"].values()
                ),
                "actor_input_tokens": sum(
                    v["actor_input_tokens"]
                    for r in selections
                    for v in r["records"]
                    if not v.get("reused_identical_context")
                ),
                "actor_output_tokens": sum(
                    v["actor_output_tokens"]
                    for r in selections
                    for v in r["records"]
                    if not v.get("reused_identical_context")
                ),
                "writer_input_tokens": sum(
                    c["writer_input_tokens"]
                    for r in selections
                    for c in r["candidates"].values()
                ),
                "writer_output_tokens": sum(
                    c["writer_output_tokens"]
                    for r in selections
                    for c in r["candidates"].values()
                ),
            },
        )
        print(json.dumps({"task": task, "index": index, **choice}), flush=True)
    save(output / "status.json", {"phase": "complete", "prefixes": len(selections)})


def assemble(root):
    plan = read(root / "plan.json")
    data, control, excluded = [], [], []
    rng = random.Random(plan["training_seed"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["model"], local_files_only=True
    )
    for task in plan["tasks"]:
        for index in plan["source_indices"]:
            row = read(
                root / "collection" / task / f"prefix_{index:03d}" / "result.json"
            )
            if row["selected"] is None:
                continue
            name = row["selected"]
            target = copy.deepcopy(
                KEEP if name == "keep" else row["candidates"][name]["target"]
            )
            valid = [c["target"] for c in row["candidates"].values() if c["accepted"]]
            unfiltered = rng.choice(valid)
            entry = {
                "id": f"{task}:{index}",
                "task": task,
                "source_index": index,
                "probe_index": index + 1,
                "messages": row["messages"],
                "target": target,
                "selected": name,
                "deltas": row["deltas"],
            }
            random_entry = dict(
                entry, target=unfiltered, selected="random_valid_candidate"
            )
            if any(
                encode_training(tokenizer, item, plan["training"]["max_length"]) is None
                for item in (entry, random_entry)
            ):
                excluded.append(entry["id"])
                continue
            data.append(entry)
            control.append(random_entry)
    positives = sum(r["selected"] != "keep" for r in data)
    eligible = (
        len(data) >= plan["minimum_labels"]
        and positives >= plan["minimum_positive_labels"]
    )
    for name, rows in [("utility_sft", data), ("unfiltered_sft", control)]:
        path = root / "training_data" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    save(
        root / "training_data" / "audit.json",
        {
            "eligible": eligible,
            "labels": len(data),
            "positive_labels": positives,
            "keep_labels": len(data) - positives,
            "overlength_excluded_from_both_arms": excluded,
            "no_truncation": True,
            "test_data_used": False,
            "control": "Same selected histories, random valid candidate target independent of utility; not a random-history control.",
            "keep_targets": "Program-authored canonical NOOP JSON, only when all scored candidates are nonbeneficial and at least one harms.",
            "limitation": "Two repeated probes are a noisy training filter, not statistical proof or unseen-task validation.",
        },
    )
    return eligible


def encode_training(tokenizer, row, limit):
    prefix = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": encode(row["target"])}],
        tokenize=True,
    )
    if full[: len(prefix)] != prefix:
        raise ValueError("Chat template token boundary mismatch")
    if len(full) > limit:
        return None
    return {"id": row["id"], "ids": full, "target_length": len(full) - len(prefix)}


def train(root, arm):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ttcl.memory_writer.train import base_fingerprint

    plan = read(root / "plan.json")
    if not read(root / "training_data/audit.json")["eligible"]:
        raise ValueError(
            "Insufficient utility signal: training is not authorized by protocol"
        )
    config = plan["training"]
    output = root / "training" / arm
    output.mkdir(parents=True, exist_ok=False)
    save(output / "status.json", {"phase": "loading"})
    torch.manual_seed(plan["training_seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["model"], local_files_only=True
    )
    data_file = root / "training_data" / f"{arm}.jsonl"
    rows = [json.loads(line) for line in data_file.read_text().splitlines()]
    data = [encode_training(tokenizer, r, config["max_length"]) for r in rows]
    if any(r is None for r in data):
        raise ValueError("Data changed after paired length audit")
    model = AutoModelForCausalLM.from_pretrained(
        plan["model"]["model"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["rank"],
            lora_alpha=2 * config["rank"],
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in n for n, _ in params):
        raise ValueError("Only writer adapter parameters may be trained")
    before = base_fingerprint(model)
    optimizer = torch.optim.AdamW([p for _, p in params], lr=config["learning_rate"])
    rng = random.Random(plan["training_seed"])
    seen, step = 0, 0
    model.train()
    start = time.monotonic()
    for epoch in range(config["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        for offset in range(0, len(order), config["accumulation"]):
            batch = order[offset : offset + config["accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i in batch:
                item = data[i]
                # Compute logits ONLY for assistant targets, while retaining
                # gradient flow through the entire untruncated source history.
                inputs = torch.tensor([item["ids"][:-1]], device="cuda:0")
                targets = torch.tensor(
                    [item["ids"][-item["target_length"] :]], device="cuda:0"
                )
                logits = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    logits_to_keep=item["target_length"],
                    use_cache=False,
                ).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite SFT loss")
                losses.append(float(loss.detach()))
                (loss / len(batch)).backward()
                seen += 1
                del logits, loss, inputs, targets
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0))
            if not math.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            optimizer.step()
            step += 1
            state = {
                "phase": "training",
                "step": step,
                "epoch": epoch + 1,
                "examples_seen": seen,
                "loss": sum(losses) / len(losses),
                "gradient_norm": norm,
                "seconds": time.monotonic() - start,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            base.append(output / "training.jsonl", state)
            save(output / "status.json", state)
            print(json.dumps(state), flush=True)
    after = base_fingerprint(model)
    if before != after:
        raise AssertionError("Frozen base parameter samples changed")
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    save(
        output / "status.json",
        {
            **state,
            "phase": "complete",
            "base_parameter_sample_unchanged": True,
            "base_fingerprint": before,
            "train_examples": len(data),
            "supervised_tokens_per_epoch": sum(d["target_length"] for d in data),
            "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
            "trainable_parameters": sum(p.numel() for _, p in params),
        },
    )


def evaluate(root, task, arm, repeat):
    plan = read(root / "plan.json")
    args = actor_args(plan, task, 20)
    output = root / "evaluation" / task / arm / str(repeat)
    output.mkdir(parents=True, exist_ok=True)
    adapter = root / "training" / arm / "adapter" if arm.endswith("_sft") else None
    os.chdir(base.BENCH)
    model = Backend(args, adapter)
    model.repeat = repeat
    first = plan["test_indices"][0]
    bank = restore_bank(
        {"entries": [], "version": 0, "last_observed": first}, plan["bank"]
    )
    rows = []
    for index in plan["test_indices"]:
        directory = output / f"episode_{index + 1:03d}"
        if (directory / "row.json").exists():
            rows.append(read(directory / "row.json"))
            if arm != "none":
                bank = restore_bank(
                    read(directory / "bank_update.json")["bank_after"], plan["bank"]
                )
            continue
        save(output / "status.json", {"phase": "answer", "index": index})
        context = "" if arm == "none" else bank.context()
        before = bank.state_dict()
        row, episode = run_episode(args, model, index, directory, context)
        if arm != "none":
            save(output / "status.json", {"phase": "update", "index": index})
            result = update_writer(
                bank,
                episode,
                model,
                directory,
                base.generation_seed(repeat, task, f"eval_writer:{index}"),
                0.0,
                plan["writer_output_tokens"],
            )
            if result["bank_before"] != before:
                raise AssertionError("Bank mutated during answer")
            if context != restore_bank(result["bank_before"], plan["bank"]).context():
                raise AssertionError("Actor bank did not match preceding writer state")
            row.update(
                decision=result["decision"],
                entries=len(bank.entries),
                bank_version=bank.version,
                writer_calls=len(result["attempts"]),
                writer_input_tokens=sum(
                    a.get("completion", {}).get("input_tokens", 0)
                    for a in result["attempts"]
                ),
                writer_output_tokens=sum(
                    a.get("completion", {}).get("output_tokens", 0)
                    for a in result["attempts"]
                ),
            )
        if index == first and context:
            raise AssertionError("Held-out evaluation must start with empty memory")
        events = [
            json.loads(s)
            for s in (directory / "responses.jsonl").read_text().splitlines()
        ]
        if any(e.get("writer_adapter_enabled") for e in events):
            raise AssertionError("Actor adapter leaked into inference")
        row.update(task=task, arm=arm, repeat=repeat, actor_adapter_enabled=False)
        save(directory / "row.json", row)
        rows.append(row)
        save(output / "results.json", rows)
        save(output / "bank.json", bank.state_dict())
        print(
            json.dumps(
                {"task": task, "arm": arm, "index": index, "reward": row["reward"]}
            ),
            flush=True,
        )
    save(output / "status.json", {"phase": "complete", "rows": len(rows)})


def report(root):
    plan = read(root / "plan.json")
    state = read(root / "status.json", {})
    lines = [
        "# 后续任务收益训练经验模块",
        "",
        f"状态：{state.get('phase', 'prepared')}",
        "",
        "冻结答题模型；仅训练 writer LoRA。候选评分与梯度使用前12题，第13–20题只用于评估。",
        "",
        "## 候选筛选",
        "",
        "| 任务 | 已完成历史 | 正收益更新标签 | KEEP标签 | 无标签 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in plan["tasks"]:
        s = read(root / "collection" / task / "summary.json", {})
        lines.append(
            f"| {task} | {s.get('prefixes', 0)}/11 | {s.get('positive_update_labels', 0)} | {s.get('keep_labels', 0)} | {s.get('no_label', 0)} |"
        )
    audit = read(root / "training_data/audit.json")
    if audit:
        lines.extend(
            ["", "训练数据审计：`" + json.dumps(audit, ensure_ascii=False) + "`"]
        )
    lines.extend(["", "## 执行进度", "", "| 阶段 | 状态 | 当前进度 |", "|---|---|---|"])
    for task in plan["tasks"]:
        progress = read(root / "collection" / task / "status.json", {})
        lines.append(
            f"| collect_{task} | {progress.get('phase', 'queued')} | {json.dumps(progress, ensure_ascii=False)} |"
        )
    for arm in ["utility_sft", "unfiltered_sft"]:
        progress = read(root / "training" / arm / "status.json", {})
        lines.append(
            f"| train_{arm} | {progress.get('phase', 'queued')} | step={progress.get('step', '—')}, loss={progress.get('loss', '—')} |"
        )
    allrows = []
    for path in (root / "evaluation").glob("*/*/*/results.json"):
        allrows.extend(read(path))
    summaries = {}
    for task in plan["tasks"]:
        rows = [r for r in allrows if r["task"] == task]
        expected = len(plan["test_indices"]) * len(plan["evaluation_repeats"])
        comparison = comparisons(rows, ARMS, expected)
        comparison["utility_contrasts"] = {}
        for control in ["none", "untrained", "unfiltered_sft"]:
            ds = [
                p["rewards"]["utility_sft"] - p["rewards"][control]
                for p in comparison["pairs"]
            ]
            later = [
                p["rewards"]["utility_sft"] - p["rewards"][control]
                for p in comparison["pairs"]
                if p["index"] != plan["test_indices"][0]
            ]
            comparison["utility_contrasts"][control] = {
                "mean_delta": statistics.mean(ds) if ds else None,
                "after_empty_first_mean_delta": statistics.mean(later)
                if later
                else None,
            }
        summaries[task] = comparison
        lines.extend(
            [
                "",
                f"## {task}：共同完成配对 {comparison['paired_count']}/{expected}",
                "",
                "| 组别 | 平均reward | 相对无经验 | 胜/平/负 |",
                "|---|---:|---:|---|",
            ]
        )
        for arm, metrics in comparison["arms"].items():
            score = metrics["mean_reward"]
            delta = metrics["delta_vs_none"]
            lines.append(
                f"| {arm} | {score if score is not None else '—'} | {delta if delta is not None else '—'} | {metrics['wins']}/{metrics['ties']}/{metrics['losses']} |"
            )
        lines.extend(
            [
                "",
                "utility_sft 配对对照：`"
                + json.dumps(comparison["utility_contrasts"])
                + "`",
            ]
        )
    save(root / "comparison.json", summaries)
    cost_fields = [
        "actor_calls",
        "writer_calls",
        "actor_input_tokens",
        "actor_output_tokens",
        "writer_input_tokens",
        "writer_output_tokens",
    ]
    usage = {
        "collection": {
            task: {
                key: read(root / "collection" / task / "summary.json", {}).get(key, 0)
                for key in cost_fields
            }
            for task in plan["tasks"]
        },
        "evaluation": {
            arm: {
                key: sum(r.get(key, 0) for r in allrows if r["arm"] == arm)
                for key in cost_fields
            }
            for arm in ARMS
        },
    }
    save(root / "usage.json", usage)
    lines.extend(
        [
            "",
            "所有均分仅使用四组共同完成的配对；未完成不补零。详细实验定义见 PROTOCOL.md。",
            "候选的两次收益检查仅用于训练标签筛选，不构成显著性检验。单次训练种子、小样本、同环境任务划分。",
        ]
    )
    temp = root / "REPORT.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(root / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "train", "evaluate", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeat", type=int, default=303)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.phase == "collect":
            collect(root, args.task)
        elif args.phase == "train":
            train(root, args.arm)
        elif args.phase == "evaluate":
            evaluate(root, args.task, args.arm, args.repeat)
        else:
            report(root)
    except Exception:
        save(
            root
            / "failures"
            / f"{args.phase}_{args.task}_{args.arm}_{args.repeat}.json",
            {"traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
