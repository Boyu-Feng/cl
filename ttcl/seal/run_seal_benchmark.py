"""SEAL-style continual adaptation using CLBench's BSM prompts and scorer.

Scores are recorded before adaptation; only past public query text enters SFT.
This is a BSM transfer experiment, not the original independent-passage SEAL
evaluation: inner LoRA weights persist between scan windows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.bsm import BENCH, parse_report, parse_scan_observation  # noqa: E402, F401




def token_chunks(ids, length, eos_id):
    """Preserve long source material rather than silently truncating it."""
    for start in range(0, len(ids), length - 1):
        yield ids[start : start + length - 1] + [eos_id]




def parse_json_object(raw):
    decoder = json.JSONDecoder()
    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[position:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object in generated learning material")


def validate_memories(raw, observations):
    """Ground generated memories in exact peaks from the supplied scans."""
    payload = parse_json_object(raw)
    proposed = payload.get("memories")
    if not isinstance(proposed, list):
        raise ValueError("Learning material must contain a memories list")
    by_scan = {item["scan_number"]: item for item in observations}
    memories = []
    rejected = []
    covered = set()
    seen_evidence_sets = set()
    for index, item in enumerate(proposed):
        if not isinstance(item, dict) or not isinstance(item.get("evidence"), list):
            rejected.append({"index": index, "reason": "missing evidence list"})
            continue
        grounded = []
        valid = True
        for evidence in item["evidence"]:
            try:
                scan_number = int(evidence["scan_number"])
                freq = float(evidence["observed_freq_mhz"])
                width = float(evidence["observed_width_mhz"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            source = by_scan.get(scan_number)
            if source is None:
                valid = False
                break
            match = next(
                (
                    peak
                    for peak in source["detected_peaks"]
                    if abs(peak["freq_mhz"] - freq) <= 0.11
                    and abs(peak["width_mhz"] - width) <= 0.11
                ),
                None,
            )
            if match is None:
                valid = False
                break
            evidence_key = (scan_number, match["peak_id"])
            covered.add(evidence_key)
            grounded.append(
                {
                    "scan_number": scan_number,
                    "peak_id": match["peak_id"],
                    "observed_freq_mhz": match["freq_mhz"],
                    "observed_width_mhz": match["width_mhz"],
                    "observed_power_dbm": match["power_dbm"],
                }
            )
        evidence_set = tuple(
            sorted((value["scan_number"], value["peak_id"]) for value in grounded)
        )
        if not valid or not grounded or evidence_set in seen_evidence_sets:
            rejected.append({"index": index, "reason": "ungrounded or duplicate evidence"})
            continue
        seen_evidence_sets.add(evidence_set)
        scan_count = len({value["scan_number"] for value in grounded})
        confidence = "high" if scan_count >= 3 else "medium" if scan_count == 2 else "low"
        memories.append(
            {
                "center_freq_mhz": round(
                    sum(value["observed_freq_mhz"] for value in grounded) / len(grounded),
                    3,
                ),
                "bandwidth_mhz": round(
                    sum(value["observed_width_mhz"] for value in grounded) / len(grounded),
                    3,
                ),
                "observation_count": len(grounded),
                "scan_count": scan_count,
                "confidence": confidence,
                "status": (
                    "persistent_candidate"
                    if scan_count >= 2
                    else "uncertain_single_scan_candidate"
                ),
                "evidence": grounded,
            }
        )
    total_peaks = sum(len(item["detected_peaks"]) for item in observations)
    if not memories:
        raise ValueError("No generated memory was grounded in the observed peaks")
    # The generator sometimes preserves all measurements but fails to merge
    # obvious cross-scan matches. Canonicalize accepted evidence with the same
    # explicit rule in the prompt. Never merge two peaks from the same scan.
    unique_evidence = {}
    for memory in memories:
        for evidence in memory["evidence"]:
            unique_evidence[(evidence["scan_number"], evidence["peak_id"])] = evidence
    clusters = []
    for evidence in sorted(
        unique_evidence.values(), key=lambda item: (item["observed_freq_mhz"], item["scan_number"])
    ):
        destination = None
        for cluster in clusters:
            scans = {item["scan_number"] for item in cluster}
            center = sum(item["observed_freq_mhz"] for item in cluster) / len(cluster)
            width = sum(item["observed_width_mhz"] for item in cluster) / len(cluster)
            if (
                evidence["scan_number"] not in scans
                and abs(evidence["observed_freq_mhz"] - center) <= 3.0
                and abs(evidence["observed_width_mhz"] - width) <= 5.0
            ):
                destination = cluster
                break
        if destination is None:
            clusters.append([evidence])
        else:
            destination.append(evidence)
    memories = []
    for cluster in clusters:
        scan_count = len({item["scan_number"] for item in cluster})
        confidence = "high" if scan_count >= 3 else "medium" if scan_count == 2 else "low"
        memories.append(
            {
                "center_freq_mhz": round(
                    sum(item["observed_freq_mhz"] for item in cluster) / len(cluster), 3
                ),
                "bandwidth_mhz": round(
                    sum(item["observed_width_mhz"] for item in cluster) / len(cluster), 3
                ),
                "observation_count": len(cluster),
                "scan_count": scan_count,
                "confidence": confidence,
                "status": (
                    "persistent_candidate"
                    if scan_count >= 2
                    else "uncertain_single_scan_candidate"
                ),
                "evidence": sorted(cluster, key=lambda item: item["scan_number"]),
            }
        )
    covered = set(unique_evidence)
    return memories, {
        "proposed_memories": len(proposed),
        "accepted_memories": len(memories),
        "rejected_memories": rejected,
        "source_peak_count": total_peaks,
        "covered_source_peaks": len(covered),
        "source_peak_coverage": len(covered) / total_peaks if total_peaks else 1.0,
    }


def build_memory_examples(memories):
    """Turn validated numeric memories into user/assistant supervision."""
    examples = []
    for memory in memories:
        evidence = "; ".join(
            f"scan {item['scan_number']}: {item['observed_freq_mhz']} MHz, "
            f"width {item['observed_width_mhz']} MHz, power {item['observed_power_dbm']} dBm"
            for item in memory["evidence"]
        )
        uncertainty = (
            "It appeared in multiple scans and is a persistent candidate."
            if memory["scan_count"] >= 2
            else "It appeared in only one scan, so it remains uncertain and may be noise."
        )
        examples.append(
            {
                "question": (
                    "What historical evidence supports the transmitter candidate near "
                    f"{memory['center_freq_mhz']} MHz?"
                ),
                "answer": (
                    f"{evidence}. Estimated bandwidth {memory['bandwidth_mhz']} MHz. "
                    f"Confidence: {memory['confidence']}. {uncertainty}"
                ),
            }
        )
    summary = "; ".join(
        f"{item['center_freq_mhz']} MHz / {item['bandwidth_mhz']} MHz / "
        f"{item['confidence']} confidence"
        for item in memories
    )
    examples.append(
        {
            "question": "Which transmitter candidates are supported by the recently observed scans?",
            "answer": f"{summary}. Single-scan candidates remain uncertain.",
        }
    )
    return examples


def score_recall(raw, memories, center_tolerance=1.0, width_tolerance=2.0):
    candidates = []
    try:
        candidates = parse_json_object(raw).get("candidates", [])
    except ValueError:
        # The primary probe deliberately uses the same natural-language format
        # as SFT, so memory quality is not confounded with JSON format transfer.
        candidates = [
            {"center_freq_mhz": match.group(1), "bandwidth_mhz": match.group(2)}
            for match in re.finditer(
                r"(-?\d+(?:\.\d+)?)\s*MHz\s*/\s*(-?\d+(?:\.\d+)?)\s*MHz",
                raw,
            )
        ]
    recalled = []
    for item in candidates if isinstance(candidates, list) else []:
        try:
            recalled.append((float(item["center_freq_mhz"]), float(item["bandwidth_mhz"])))
        except (KeyError, TypeError, ValueError):
            continue
    targets = [(item["center_freq_mhz"], item["bandwidth_mhz"]) for item in memories]
    matched_targets = set()
    matched_recalled = set()
    for target_index, target in enumerate(targets):
        for recall_index, recall in enumerate(recalled):
            if recall_index in matched_recalled:
                continue
            if (
                abs(target[0] - recall[0]) <= center_tolerance
                and abs(target[1] - recall[1]) <= width_tolerance
            ):
                matched_targets.add(target_index)
                matched_recalled.add(recall_index)
                break
    recall = len(matched_targets) / len(targets) if targets else 1.0
    precision = len(matched_recalled) / len(recalled) if recalled else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "target_count": len(targets),
        "recalled_count": len(recalled),
        "matched_count": len(matched_targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class SealMemory:
    def __init__(self, args):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(args.seed)
        self.torch = torch
        self.args = args
        self.updates = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(args.device)
        if args.mode == "ttt":
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.eval()

    def generate(self, text, max_new_tokens):
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if tokens.input_ids.shape[1] > self.args.max_input_tokens:
            raise ValueError("Prompt exceeds --max-input-tokens; increase the budget")
        tokens = tokens.to(self.args.device)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output[0, tokens.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    def respond(self, query):
        prompt = (
            query.prompt
            + "\n\nReturn ONLY one valid JSON object, with no explanation. Schema:\n"
            + json.dumps(query.response_schema.model_json_schema())
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(rendered, self.args.max_new_tokens)

    def adapt(self, history, output_dir):
        """Generate grounded numeric memories, train LoRA, then probe recall."""
        selected = list(history)
        while selected:
            context = json.dumps(selected, ensure_ascii=False, indent=2)
            prompt = (
                "You are writing factual memory for a spectrum-monitoring agent. "
                "The input contains only past scan observations. Group measurements into "
                "the same transmitter candidate when center frequencies differ by at most "
                "3 MHz AND bandwidths differ by at most 5 MHz; otherwise keep them separate. "
                "When the rule matches, put all matching measurements in one evidence list. "
                "Preserve every observed peak as evidence. Do not "
                "invent values, future observations, labels, or certainty. A candidate seen "
                "in one scan is low-confidence and may be noise; repeated support raises "
                "confidence. Return ONLY one JSON object with this schema:\n"
                '{"memories":[{"center_freq_mhz":<number>,"bandwidth_mhz":<number>,'
                '"evidence":[{"scan_number":<integer>,"observed_freq_mhz":<number>,'
                '"observed_width_mhz":<number>}],"confidence":"low|medium|high",'
                '"uncertainty":"<short statement>"}]}\n\n'
                f"PAST SCAN OBSERVATIONS:\n{context}"
            )
            rendered_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if (
                len(self.tokenizer.encode(rendered_prompt, add_special_tokens=False))
                <= self.args.max_input_tokens
            ):
                break
            selected.pop(0)
        if not selected:
            raise ValueError("One observed scan exceeds the self-edit input budget")
        material = self.generate(rendered_prompt, self.args.material_tokens)
        if not material:
            raise ValueError("SEAL generated empty learning material")
        memories, validation = validate_memories(material, selected)
        examples = build_memory_examples(memories)
        update_id = self.updates + 1
        materials_dir = output_dir / "materials"
        materials_dir.mkdir(exist_ok=True)
        material_path = materials_dir / f"update-{update_id}.json"
        material_path.write_text(
            json.dumps(
                {
                    "source_instance_ids": [item["instance_id"] for item in selected],
                    "observation_context": selected,
                    "extraction_prompt": prompt,
                    "raw_completion": material,
                    "validated_memories": memories,
                    "validation": validation,
                    "train_examples": examples,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        torch = self.torch
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.args.learning_rate)
        losses = []
        self.model.train()
        for _ in range(self.args.train_epochs):
            for example in examples:
                user_messages = [{"role": "user", "content": example["question"]}]
                full_messages = user_messages + [
                    {"role": "assistant", "content": example["answer"]}
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    user_messages, tokenize=False, add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)[
                    : self.args.train_seq_length
                ]
                if len(full_ids) <= len(prompt_ids):
                    raise ValueError("No assistant tokens remain in a memory SFT example")
                batch = torch.tensor([full_ids], device=self.args.device)
                labels = torch.tensor(
                    [[-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]],
                    device=self.args.device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(
                    input_ids=batch,
                    attention_mask=torch.ones_like(batch),
                    labels=labels,
                    use_cache=False,
                ).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite inner SFT loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        self.model.eval()
        self.updates = update_id
        # Only PEFT adapters are saved; the input checkpoint is never modified.
        self.model.save_pretrained(output_dir / "latest_adapter")
        self.tokenizer.save_pretrained(output_dir / "latest_adapter")
        scan_numbers = [item["scan_number"] for item in selected]
        # Use the exact held-out prompt used by the aggregate memory example.
        # The facts remain hidden; only the question wording is shared.
        recall_prompt = (
            "Which transmitter candidates are supported by the recently observed scans?"
        )
        rendered_recall = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": recall_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        recall_raw = self.generate(rendered_recall, self.args.recall_tokens)
        recall = score_recall(recall_raw, memories)
        recall.update({"raw_response": recall_raw, "scan_numbers": scan_numbers})
        with (output_dir / "recall_probes.jsonl").open("a") as handle:
            handle.write(json.dumps({"update": update_id, **recall}, ensure_ascii=False) + "\n")
        return {
            "update": update_id,
            "mean_loss": sum(losses) / len(losses),
            "steps": len(losses),
            "material_path": str(material_path),
            "source_peak_coverage": validation["source_peak_coverage"],
            "recall": {key: recall[key] for key in ("precision", "recall", "f1")},
        }


def run(args, memory_factory=SealMemory):
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    data = Path(args.data_path).resolve()
    with data.open() as handle:
        count = sum(bool(line.strip()) for line in handle)
    count = min(args.num_scans or count, count)
    task = BlindSpectrumMonitoringTask(
        dataset_path=str(data),
        num_instances=count,
        seed=args.seed,
        repeat_instructions=True,
    )
    # Capture every argument, including online hyperparameters, for reproducibility.
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    memory = memory_factory(args)
    history = deque(maxlen=args.memory_window)
    query = task.reset()
    records = []
    updates = []
    started = time.monotonic()
    while query is not None and len(records) < count:
        raw = memory.respond(query)
        error = None
        try:
            report = parse_report(raw, query.response_schema)
        except ValueError as exc:
            # CLBench's timeout path assigns zero and advances the scan. Record
            # the actual cause separately rather than silently crediting an empty report.
            error = str(exc)
            report = None
            step = task.step(
                Response(
                    action=query.response_schema(transmitters=[]),
                    metadata={"latency_timeout": True},
                )
            )
        else:
            step = task.step(Response(action=report))
        record = {
            "scan": len(records) + 1,
            "instance_id": query.instance_id,
            "reward": step.instance_outcome.reward,
            "raw_response": raw,
            "report": report.model_dump() if report is not None else None,
            "parse_error": error,
            "updates_before_answer": memory.updates,
        }
        # Explicit allowlist: keep measurements from the public query, while
        # removing task instructions, schemas, scores, and model responses.
        history.append(parse_scan_observation(query.prompt, query.instance_id))
        records.append(record)
        has_next = (
            not step.done and step.next_query is not None and len(records) < count
        )
        with (output / "responses.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.mode == "ttt" and has_next and len(records) % args.update_every == 0:
            update = memory.adapt(history, output)
            updates.append(update)
            with (output / "updates.jsonl").open("a") as handle:
                handle.write(json.dumps(update) + "\n")
        scores = [row["reward"] for row in records]
        progress = {
            "completed": len(records),
            "total": count,
            "num_updates": memory.updates,
            "mean_score": sum(scores) / len(scores),
            "elapsed_seconds": time.monotonic() - started,
            "invalid_reports": sum(row["parse_error"] is not None for row in records),
            "mean_recall_f1": (
                sum(item["recall"]["f1"] for item in updates) / len(updates)
                if updates and all("recall" in item for item in updates)
                else None
            ),
        }
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(
            f"{args.mode} scan={len(records)}/{count} reward={scores[-1]:.4f} "
            f"mean={progress['mean_score']:.4f} updates={memory.updates}",
            flush=True,
        )
        if not has_next:
            break
        query = step.next_query
    evaluation = task.evaluate()
    result = {
        **progress,
        "model": args.model,
        "mode": args.mode,
        "score": evaluation.score,
        "benchmark_metrics": evaluation.metrics,
        "score_curve": scores,
        "summary": evaluation.summary,
        "protocol": (
            "BSM sequential transfer; measurement-only extraction; grounded numeric QA; "
            "score before update; recall probe after update; persistent LoRA; no textual "
            "history at answer time"
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["frozen", "ttt"], default="ttt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-path",
        default=str(
            BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"
        ),
    )
    parser.add_argument("--seal-root", default=str(ROOT / "current_work/SEAL"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--num-scans", type=int)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--memory-window", type=int, default=4)
    # Qwen3-Instruct is sensitive to the original base-model inner-loop rate.
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--train-seq-length", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--material-tokens", type=int, default=1536)
    parser.add_argument("--recall-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "num_scans",
        "update_every",
        "memory_window",
        "train_epochs",
        "lora_r",
        "lora_alpha",
        "max_input_tokens",
        "max_new_tokens",
        "material_tokens",
        "recall_tokens",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_seq_length < 2 or args.learning_rate <= 0:
        parser.error("train-seq-length must be >= 2 and learning-rate must be positive")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model",
                    "mode",
                    "completed",
                    "mean_score",
                    "num_updates",
                    "invalid_reports",
                    "mean_recall_f1",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
