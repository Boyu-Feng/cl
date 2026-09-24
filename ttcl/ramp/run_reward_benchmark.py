"""Prequential BSM evaluation: answer -> scalar reward -> persistent LoRA update."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttcl.ramp.reward_memory import RewardConfig, RewardLearner  # noqa: E402
from ttcl.common.bsm import BENCH, parse_report, parse_scan_observation  # noqa: E402


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--adapter-path", help="Optional existing PEFT adapter")
    parser.add_argument("--method", choices=["ramp", "frozen"], default="ramp")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--do-sample", action="store_true",
                        help="Sample answers; default remains greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--generation-seed", type=int,
                        help="Independent decoding seed (defaults to --seed)")
    parser.add_argument("--candidates-per-scan", type=int, default=1,
                        help="K candidates before feedback; only candidate 0 counts officially")
    parser.add_argument("--experiment-phase", default="evaluation")
    parser.add_argument("--memory-mode", choices=["none", "summary"], default="none",
                        help="Optional bounded summary of previous public scans")
    parser.add_argument("--feedback-memory", choices=["none", "summary"], default="none",
                        help="Read past actions/rewards and same-scan preferences in future prompts")
    parser.add_argument("--feedback-window", type=int, default=16)
    parser.add_argument("--feedback-min-gap", type=float, default=0.01)
    parser.add_argument("--action-mode", choices=["report", "selection"], default="report",
                        help="Generate numeric reports, or select IDs from a public candidate catalog")
    parser.add_argument("--proposal-mode", choices=["sampled", "mutate"], default="sampled",
                        help="Sample every candidate, or mutate public IDs after the official model answer")
    for field in fields(RewardConfig):
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(field.default), default=field.default)
    return parser


def validate_args(parser, args):
    try:
        config = RewardConfig(**{f.name: getattr(args, f.name) for f in fields(RewardConfig)})
        for name in ("update_every", "lora_r", "lora_alpha", "max_input_tokens", "max_new_tokens",
                     "candidates_per_scan"):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if args.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        from ttcl.ramp.feedback_memory import FeedbackMemory

        FeedbackMemory(args.feedback_window, args.feedback_min_gap)
        if args.action_mode == "selection" and args.memory_mode != "summary":
            raise ValueError("--action-mode selection requires --memory-mode summary")
        if args.proposal_mode == "mutate" and (
                args.action_mode != "selection" or args.candidates_per_scan < 2):
            raise ValueError("--proposal-mode mutate requires selection actions and at least two candidates")
    except ValueError as exc:
        parser.error(str(exc))
    return config


def prepare_output(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    return output


def generation_seed(base_seed, query, candidate_index):
    """Stable per-instance seeds keep candidate zero matched across different K."""
    identity = query.instance_id or query.prompt
    payload = json.dumps([base_seed, identity, candidate_index], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


@contextmanager
def isolated_generation_rng(device, seed):
    """Generation neither consumes training RNG nor depends on training RNG use."""
    import torch

    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


class RewardMemory:
    def __init__(self, args):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(args.seed)
        self.args = args
        self.updates = 0
        self.generation_counts = {}
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa", local_files_only=True,
        ).to(args.device)
        if args.adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model, args.adapter_path, is_trainable=args.method != "frozen")
        elif args.method != "frozen":
            self.model = get_peft_model(self.model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM"))
        self.learner = None
        if args.method != "frozen":
            config = RewardConfig(**{f.name: getattr(args, f.name) for f in fields(RewardConfig)})
            self.learner = RewardLearner(self.model, self.tokenizer, config)
        self.model.eval()

    def render(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    def respond(self, query):
        import torch
        # The selection prompt already specifies the compact include-list action.
        # Report mode retains exactly the existing baseline prompt and schema.
        prompt = query.prompt
        if self.args.action_mode == "report":
            prompt += ("\n\nReturn ONLY one valid JSON object, with no explanation. Schema:\n"
                       + json.dumps(query.response_schema.model_json_schema()))
        rendered = self.render(prompt)
        tokens = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        if tokens.input_ids.shape[1] > self.args.max_input_tokens:
            raise ValueError("Prompt exceeds --max-input-tokens")
        tokens = tokens.to(self.args.device)
        self.model.eval()
        identity = query.instance_id or query.prompt
        candidate_index = self.generation_counts.get(identity, 0)
        self.generation_counts[identity] = candidate_index + 1
        base_seed = self.args.seed if self.args.generation_seed is None else self.args.generation_seed
        seed = generation_seed(base_seed, query, candidate_index)
        options = {"do_sample": self.args.do_sample, "use_cache": True,
                   "max_new_tokens": self.args.max_new_tokens,
                   "pad_token_id": self.tokenizer.pad_token_id}
        if self.args.do_sample:
            options.update(temperature=self.args.temperature, top_p=self.args.top_p,
                           top_k=self.args.top_k)
        with isolated_generation_rng(self.args.device, seed), torch.inference_mode():
            output = self.model.generate(**tokens, **options)
        response_ids = output[0, tokens.input_ids.shape[1]:].tolist()
        return {"prompt": rendered,
                "response": self.tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
                "response_ids": response_ids, "prompt_is_rendered": True}

    def observe(self, completion, reward, valid):
        item = self.learner.observe(completion["prompt"], completion["response"], reward,
                                    task_id="bsm", valid=valid,
                                    response_ids=completion["response_ids"])
        return asdict(item)

    def observe_group(self, completions, rewards, valids):
        prompt = completions[0]["prompt"]
        if any(item["prompt"] != prompt for item in completions):
            raise ValueError("Candidate group must share exactly the same rendered prompt")
        candidates = [{"response": item["response"], "reward": reward, "valid": valid,
                       "response_ids": item["response_ids"]}
                      for item, reward, valid in zip(completions, rewards, valids)]
        return [asdict(item) for item in self.learner.observe_group(
            prompt, candidates, task_id="bsm")]

    def adapt(self, output):
        report = self.learner.update()
        self.updates = self.learner.updates
        if report["accepted"]:
            self.save(output)
        return report

    def save(self, output):
        if self.learner is None:
            return
        self.model.save_pretrained(output / "latest_adapter")
        self.tokenizer.save_pretrained(output / "latest_adapter")

    def audit(self, output):
        if self.learner is not None:
            (output / "learner_state.json").write_text(
                json.dumps(self.learner.state_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8")


def response_action(report, schema):
    """Give invalid JSON the same zero-score timeout semantics as the baseline."""
    from src.interface import Response

    if report is None:
        return Response(action=schema(transmitters=[]), metadata={"latency_timeout": True})
    return Response(action=report)


def counterfactual_reward(task, report, schema):
    """Evaluator-only fork: exact task scoring, with no mutation of the live task.

    Only the scalar leaves this function. No latent channels, checks or feedback
    text enter the learner. This is an additional reward call, not a free label.
    """
    fork = copy.deepcopy(task)
    return float(fork.step(response_action(report, schema)).instance_outcome.reward)


def report_geometry(report):
    if report is None:
        return None
    return tuple(sorted((item.center_freq, item.bandwidth) for item in report.transmitters))


def public_id_mutations(raw, catalog, count, seed):
    """Distinct actions derived only from public IDs, before any reward exists.

    Single-ID removal and addition alternate when both are possible. Larger
    groups exhaust those toggles before deterministic multi-ID changes. There
    are only 2**N distinct actions; the caller samples any unfillable slots.
    """
    from ttcl.ramp.spectrum_actions import decode_selection

    decode_selection(raw, catalog)  # Strict ID/type/duplicate/fence validation.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    selected_order = json.loads(raw)["include"]
    selected = set(selected_order)
    available = sorted(row.candidate_id for row in catalog.candidates)
    rng = random.Random(seed)
    removals, additions = sorted(selected), sorted(set(available) - selected)
    rng.shuffle(removals)
    rng.shuffle(additions)
    rng.shuffle(available)

    def toggle_sets():
        for index in range(max(len(removals), len(additions))):
            if index < len(removals):
                yield (removals[index],)
            if index < len(additions):
                yield (additions[index],)
        for size in range(2, len(available) + 1):
            yield from itertools.combinations(available, size)

    actions = []
    for toggled in itertools.islice(toggle_sets(), count):
        # Preserve the official action's order so credit focuses on changed IDs.
        included = [uid for uid in selected_order if uid not in toggled]
        included.extend(sorted(uid for uid in toggled if uid not in selected))
        actions.append(json.dumps({"include": included}, separators=(",", ":")))
    return actions


def generate_candidates(memory, query, catalog, args):
    """Construct the entire candidate group without accessing the evaluator."""
    completions = [memory.respond(query)]
    provenance = [{"proposal_source": "model", "model_generation_index": 0,
                   "mutation_seed": None, "proposal_fallback_reason": None}]
    fallback_reason = None
    if args.proposal_mode == "mutate":
        base_seed = args.seed if args.generation_seed is None else args.generation_seed
        seed = generation_seed(base_seed, query, -1)
        try:
            actions = public_id_mutations(
                completions[0]["response"], catalog, args.candidates_per_scan - 1, seed)
        except ValueError:
            actions = []
            fallback_reason = "invalid_official_selection"
        else:
            if len(actions) < args.candidates_per_scan - 1:
                fallback_reason = "distinct_public_action_space_exhausted"
        for raw in actions:
            ids = memory.tokenizer.encode(raw, add_special_tokens=False)
            if memory.tokenizer.eos_token_id is not None:
                ids.append(memory.tokenizer.eos_token_id)
            completions.append({**completions[0], "response": raw, "response_ids": ids})
            provenance.append({"proposal_source": "public_id_mutation",
                               "model_generation_index": None, "mutation_seed": seed,
                               "proposal_fallback_reason": None})
    model_index = 1
    while len(completions) < args.candidates_per_scan:
        completions.append(memory.respond(query))
        provenance.append({"proposal_source": "model", "model_generation_index": model_index,
                           "mutation_seed": None, "proposal_fallback_reason": fallback_reason})
        model_index += 1
    return completions, provenance


def behavior_diagnostics(report, current, previous):
    """Public-observation diagnostics, never used as targets or reward labels."""
    if report is None or current is None:
        return {"current_peak_copy": False, "history_supported_extra_transmitters": 0}
    current_peaks = current["detected_peaks"]
    peaks = tuple(sorted((peak["freq_mhz"], peak["width_mhz"]) for peak in current_peaks))

    def matches(region, peak):
        return (abs(region.center_freq - peak["freq_mhz"]) <= 2.0
                and abs(region.bandwidth - peak["width_mhz"]) <= 3.0)

    count = sum(not any(matches(region, peak) for peak in current_peaks)
                and any(matches(region, peak) for peak in previous)
                for region in report.transmitters)
    return {"current_peak_copy": report_geometry(report) == peaks,
            "history_supported_extra_transmitters": count}


def run(args, memory_factory=RewardMemory, task_factory=None):
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    with Path(args.data_path).open(encoding="utf-8") as handle:
        count = sum(bool(line.strip()) for line in handle)
    count = min(args.num_scans or count, count)
    if count < 1:
        raise ValueError("At least one scan is required")
    output = prepare_output(args)
    task_factory = task_factory or BlindSpectrumMonitoringTask
    task = task_factory(dataset_path=args.data_path, num_instances=count,
                        seed=args.seed, repeat_instructions=True)
    memory = memory_factory(args)
    feedback_memory = None
    if args.feedback_memory == "summary":
        from ttcl.ramp.feedback_memory import FeedbackMemory

        feedback_memory = FeedbackMemory(args.feedback_window, args.feedback_min_gap)
    public_memory = None
    if args.memory_mode == "summary":
        from ttcl.ramp.spectrum_memory import SpectrumMemory

        public_memory = SpectrumMemory()
    records = []
    previous_peaks = []
    candidate_records = []
    reward_calls = 0
    generation_tokens = 0
    proposed_action_tokens = 0
    model_generated_candidates = 0
    mutation_count = 0
    proposal_fallback_candidates = 0
    learner_feedback_count = 0
    attempts = 0
    started = time.monotonic()
    query = task.reset()
    while query is not None and len(records) < count:
        scan = len(records) + 1
        updates_before_answer = memory.updates
        answer_query = copy.copy(query)
        summary_context = public_memory.context() if public_memory is not None else ""
        catalog = None
        if args.action_mode == "selection":
            from ttcl.ramp.spectrum_actions import build_candidate_catalog, decode_selection, render_selection_prompt

            catalog = build_candidate_catalog(public_memory, query.prompt, query.instance_id)
            answer_query.prompt = render_selection_prompt(catalog)
        elif summary_context:
            answer_query.prompt += "\n\nHistorical public scan evidence:\n" + summary_context
        feedback_context = feedback_memory.render(catalog) if feedback_memory is not None else ""
        if feedback_context:
            answer_query.prompt += "\n\n" + feedback_context
        if catalog is not None:
            append_json(output / "action_catalogs.jsonl", {
                "scan": scan, "instance_id": query.instance_id,
                "catalog": catalog.state_dict(), "prompt": answer_query.prompt,
            })
        # Complete every candidate BEFORE any scoring or feedback. The first
        # candidate is fixed as the official answer; never choose after rewards.
        completions, provenance = generate_candidates(memory, answer_query, catalog, args)
        if any(item["prompt"] != completions[0]["prompt"] for item in completions):
            raise ValueError("Candidates must have identical rendered contexts")
        reports, errors = [], []
        for completion in completions:
            try:
                if catalog is None:
                    report = parse_report(completion["response"], query.response_schema)
                else:
                    report = decode_selection(completion["response"], catalog, query.response_schema)
                # A syntactically valid NaN/Infinity must not poison scoring/logs.
                if any(not math.isfinite(value)
                       for item in report.transmitters
                       for value in (item.center_freq, item.bandwidth, item.estimated_power)):
                    raise ValueError("Non-finite transmitter value")
                reports.append(report)
                errors.append(None)
            except ValueError as exc:
                reports.append(None)
                errors.append(str(exc))
        extra_rewards = [counterfactual_reward(task, report, query.response_schema)
                         for report in reports[1:]]
        step = task.step(response_action(reports[0], query.response_schema))
        rewards = [float(step.instance_outcome.reward), *extra_rewards]
        reward_calls += len(rewards)
        reward = rewards[0]
        report = reports[0]
        error = errors[0]
        try:
            current = parse_scan_observation(query.prompt, query.instance_id)
        except ValueError:
            current = None
        diagnostics = behavior_diagnostics(report, current, previous_peaks)
        record = {"scan": scan, "instance_id": query.instance_id,
                  "reward": reward, "raw_response": completions[0]["response"], "parse_error": error,
                  "report": report.model_dump() if report is not None else None,
                  "updates_before_answer": updates_before_answer,
                  "experiment_phase": args.experiment_phase,
                  "action_mode": args.action_mode,
                  "proposal_mode": args.proposal_mode,
                  "catalog_size": len(catalog.candidates) if catalog is not None else 0,
                  "catalog_omitted_candidates": catalog.omitted_candidates if catalog is not None else 0,
                  "catalog_omitted_current_candidates": catalog.omitted_current_candidates if catalog is not None else 0,
                  "candidate_id": f"scan-{scan:04d}-candidate-000",
                  "oracle_score": max(rewards), "candidate_mean_score": sum(rewards) / len(rewards),
                  "candidate_reward_spread": max(rewards) - min(rewards),
                  "distinct_candidate_geometries": len({report_geometry(item)
                                                        for item in reports if item is not None}),
                  "summary_context": summary_context,
                  "feedback_context": feedback_context, **diagnostics}
        records.append(record)
        append_json(output / "responses.jsonl", record)
        base_seed = args.seed if args.generation_seed is None else args.generation_seed
        for index, (completion, candidate_report, candidate_error, candidate_reward, source) in enumerate(
                zip(completions, reports, errors, rewards, provenance)):
            tokens = len(completion["response_ids"])
            is_model = source["proposal_source"] == "model"
            generation_tokens += tokens if is_model else 0
            proposed_action_tokens += 0 if is_model else tokens
            model_generated_candidates += int(is_model)
            mutation_count += int(not is_model)
            proposal_fallback_candidates += int(source["proposal_fallback_reason"] is not None)
            candidate = {
                "scan": scan, "instance_id": query.instance_id,
                "candidate_id": f"scan-{scan:04d}-candidate-{index:03d}", "candidate_index": index,
                "official": index == 0,
                "phase": "official" if index == 0 else "counterfactual_diagnostic",
                "experiment_phase": args.experiment_phase, "reward": candidate_reward,
                "action_mode": args.action_mode,
                "proposal_mode": args.proposal_mode, **source,
                "raw_response": completion["response"], "parse_error": candidate_error,
                "report": candidate_report.model_dump() if candidate_report is not None else None,
                "updates_before_answer": updates_before_answer,
                "generation_seed": generation_seed(base_seed, answer_query, source["model_generation_index"])
                                   if is_model else None,
                "generation_tokens": tokens if is_model else 0,
                "proposed_action_tokens": 0 if is_model else tokens,
                "action_tokens": tokens,
            }
            candidate_records.append(candidate)
            append_json(output / "candidates.jsonl", candidate)
        # Explicit allowlist: never send observation/checks/latent channels to learner.
        if args.method != "frozen":
            valids = [item is None for item in errors]
            if getattr(args, "advantage_mode", "historical") == "group":
                signals = memory.observe_group(completions, rewards, valids)
            else:
                signals = [memory.observe(completion, candidate_reward, valid)
                           for completion, candidate_reward, valid in zip(completions, rewards, valids)]
            learner_feedback_count += len(signals)
            for index, (completion, candidate_reward, valid, feedback) in enumerate(
                    zip(completions, rewards, valids, signals)):
                append_json(output / "experiences.jsonl", {
                    **completion, "reward": candidate_reward, "valid": valid, "task_id": "bsm",
                    "instance_id": query.instance_id, "training_signal": feedback,
                    "candidate_id": f"scan-{scan:04d}-candidate-{index:03d}",
                    "candidate_index": index, "experiment_phase": args.experiment_phase,
                    "proposal_mode": args.proposal_mode, **provenance[index],
                })
        if feedback_memory is not None:
            episode = feedback_memory.observe(
                scan=scan, current=current, catalog=catalog,
                responses=[item["response"] for item in completions],
                reports=reports, rewards=rewards)
            append_json(output / "feedback_episodes.jsonl", episode)
        # Add this scan only after answering; K candidates never duplicate evidence.
        if public_memory is not None:
            public_memory.observe(query.prompt, query.instance_id)
        if current is not None:
            previous_peaks.extend(current["detected_peaks"])
        has_next = not step.done and step.next_query is not None and len(records) < count
        if args.method != "frozen" and has_next and len(records) % args.update_every == 0:
            update = memory.adapt(output)
            attempts += 1
            append_json(output / "updates.jsonl", update)
            print(f"update={memory.updates} status={update['reason']} "
                  f"pairs={update.get('pair_count', 0)}", flush=True)
        progress = {
            "completed": len(records), "total": count, "num_updates": memory.updates,
            "update_attempts": attempts,
            "mean_score": sum(x["reward"] for x in records) / len(records),
            "invalid_reports": sum(x["parse_error"] is not None for x in records),
            "elapsed_seconds": time.monotonic() - started,
            "candidates_per_scan": args.candidates_per_scan,
            "reward_calls": reward_calls, "official_reward_calls": len(records),
            "diagnostic_reward_calls": reward_calls - len(records),
            "learner_feedback_count": learner_feedback_count,
            "feedback_memory_reward_count": feedback_memory.reward_count if feedback_memory else 0,
            "generation_tokens": generation_tokens,
            "proposed_action_tokens": proposed_action_tokens,
            "model_generated_candidates": model_generated_candidates,
            "mutation_count": mutation_count,
            "proposal_fallback_candidates": proposal_fallback_candidates,
            "mean_candidate_score": sum(x["candidate_mean_score"] for x in records) / len(records),
            "oracle_mean_score": sum(x["oracle_score"] for x in records) / len(records),
            "oracle_gain": sum(x["oracle_score"] - x["reward"] for x in records) / len(records),
            "mean_distinct_candidate_geometries": sum(
                x["distinct_candidate_geometries"] for x in records) / len(records),
            "mean_candidate_reward_spread": sum(x["candidate_reward_spread"] for x in records) / len(records),
            "invalid_candidates": sum(x["parse_error"] is not None for x in candidate_records),
            "current_peak_copy_count": sum(x["current_peak_copy"] for x in records),
            "history_supported_extra_transmitters": sum(
                x["history_supported_extra_transmitters"] for x in records),
            "mean_catalog_size": sum(x["catalog_size"] for x in records) / len(records),
            "max_catalog_size": max(x["catalog_size"] for x in records),
            "catalog_omitted_candidates": sum(x["catalog_omitted_candidates"] for x in records),
            "catalog_omitted_current_candidates": sum(x["catalog_omitted_current_candidates"] for x in records),
        }
        (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"{args.method} scan={len(records)}/{count} reward={reward:.4f} "
              f"mean={progress['mean_score']:.4f} updates={memory.updates}", flush=True)
        if not has_next:
            break
        query = step.next_query
    memory.audit(output)
    if feedback_memory is not None:
        (output / "feedback_memory.json").write_text(
            json.dumps(feedback_memory.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if public_memory is not None:
        (output / "public_memory.json").write_text(
            json.dumps(public_memory.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = task.evaluate()
    decoding = "sampled" if args.do_sample else "greedy"
    budget_note = ("one answer and one scalar reward per scan" if args.candidates_per_scan == 1
                   else f"{args.candidates_per_scan} candidates and scalar rewards per scan; "
                        "candidate 0 alone is official; best-of-K is diagnostic only")
    result = {**progress, "method": args.method, "model": args.model,
              "score": evaluation.score, "score_curve": [x["reward"] for x in records],
              "experiment_phase": args.experiment_phase, "memory_mode": args.memory_mode,
              "feedback_memory": args.feedback_memory,
              "action_mode": args.action_mode,
              "proposal_mode": args.proposal_mode,
              "advantage_mode": getattr(args, "advantage_mode", "historical"),
              "protocol": f"{decoding}; {budget_note}; all candidates generated before feedback; "
                          "score before update; no final update; independent per-instance decoding RNG; "
                          f"memory_mode={args.memory_mode}; action_mode={args.action_mode}; "
                          f"feedback_memory={args.feedback_memory} (past scans only, no extra scoring); "
                          f"proposal_mode={args.proposal_mode}; "
                          "model-generated candidates retain their original action tokens; "
                          "mutation proposals are reencoded off-policy actions using public IDs; "
                          "only scalar rewards supplied to learner; "
                          "no teacher, latent channels or future scans; extra reward budget reported",
              "behavior_diagnostic_tolerance": {"center_mhz": 2.0, "bandwidth_mhz": 3.0}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = build_parser()
    parser.add_argument("--data-path", default=str(
        BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"))
    parser.add_argument("--num-scans", type=int)
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.num_scans is not None and args.num_scans < 1:
        parser.error("--num-scans must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
