"""Frozen, evidence-grounded experience screening and held-out confirmation.

The bank is authored before evaluation. Only past public tool traces ground it.
Reward is used by the offline selector, never as an action-level causal label.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import traceback
from types import SimpleNamespace


ARMS = ["none", "generic", "irrelevant", "candidate_a", "candidate_b", "combined"]
CANDIDATES = ARMS[3:]
PADDING = (
    " A triangle has three sides. A square has four sides. A week has seven days."
    " The letters A, B, C appear in alphabetical order. A metre contains one hundred centimetres."
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def public_evidence(root, task, identity, turn):
    """Read exact past public action/result pairs, never generated reasoning."""
    directory = Path(root) / task / "independent/independent"
    observations = (directory / "public_observations.jsonl").read_text().splitlines()
    responses = (directory / "responses.jsonl").read_text().splitlines()
    for number, line in enumerate(observations, 1):
        observation = json.loads(line)
        if (observation["instance_id"], observation["turn"]) == (identity, turn):
            if observation["instance_complete"]:
                raise ValueError(
                    "Terminal answers and evaluator feedback are not bank evidence"
                )
            for response_number, response_line in enumerate(responses, 1):
                response = json.loads(response_line)
                if (response["instance_id"], response["turn"]) == (
                    identity,
                    turn,
                ) and response.get("action"):
                    action = response["action"].copy()
                    for field in ("thought", "thinking", "reasoning"):
                        action.pop(field, None)
                    return {
                        "instance_id": identity,
                        "turn": turn,
                        "action": action,
                        "observation": observation["content"],
                        "observation_file": str(
                            directory / "public_observations.jsonl"
                        ),
                        "observation_line": number,
                        "observation_sha256": digest(line),
                        "response_file": str(directory / "responses.jsonl"),
                        "response_line": response_number,
                        "response_sha256": digest(response_line),
                    }
    raise ValueError(f"Missing evidence: {task} {identity} {turn}")


def build_bank(source_root):
    def db(identity, turn):
        return public_evidence(source_root, "database_exploration", identity, turn)

    def co(identity, turn):
        return public_evidence(source_root, "cohort_studies", identity, turn)

    catalog = db("1", 3)
    schemas = [db("5", t) for t in (4, 5, 6)] + [db("3", t) for t in (4, 5, 6)]
    assert "items_g1" in catalog["observation"] and "fdbk_g3" in catalog["observation"]
    schema_text = []
    for evidence in schemas:
        sql = evidence["action"]["content"]
        table = sql.split("(")[1].split(")")[0]
        columns = []
        for line in evidence["observation"].splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 3 and parts[0].isdigit():
                columns.append(parts[1] + ":" + parts[2])
        assert columns
        schema_text.append(table + "(" + ", ".join(columns) + ")")
    category = db("5", 7)
    assert "GROUP BY main_cat" in category["action"]["content"]
    assert all(
        x in category["observation"]
        for x in [
            "Office Products",
            "All Electronics",
            "Musical Instruments",
            "Computers",
        ]
    )
    repeat = [co("cohort_studies:herald_suburban", t) for t in (3, 4, 5)]
    assert len({json.dumps(x["action"], sort_keys=True) for x in repeat}) == 1
    assert len({x["observation"] for x in repeat}) == 1
    fit = [
        co("cohort_studies:herald_rural_s2", 4),
        co("cohort_studies:meridian_suburban", 3),
    ]
    assert all(
        "Observable Cohort Fit" in x["observation"]
        and "Unobservable cohorts" in x["observation"]
        for x in fit
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Evidence-grounded knowledge from past interactions, with applicability, use and falsification conditions; utility requires separate evaluation.",
        "authorship": "Assistant-curated from public traces; not autonomous extraction. Verification establishes the stated observations, not positive reward impact.",
        "source_canonical_indices": list(range(6)),
        "database_exploration": {
            "candidate_a": {
                "name": "Observed schema identifiers",
                "scope": "The same unchanged multi_group SQLite database; recheck on a schema change or missing-column error.",
                "content": "Earlier PRAGMA results established these exact schemas: "
                + "; ".join(schema_text)
                + ". Use observed names instead of guessing products_office or products_electronics. Schema types alone do not prove timestamp units, price units, category-to-dataset mappings or join semantics.",
                "evidence": [catalog, *schemas],
                "falsifier": "Current schema differs from the recorded PRAGMA output.",
                "mechanism": "Reduce nonexistent identifiers and redundant schema discovery; reward may still not improve.",
            },
            "candidate_b": {
                "name": "Dataset membership differs from main_cat labels",
                "scope": "The same multi_group database; do not extrapolate one table's category labels to another table.",
                "content": "An actual GROUP BY main_cat on items_g1 returned Office Products, All Electronics, Computers, Musical Instruments and many other categories in this ONE table. Therefore main_cat is not a unique identifier for the three dataset groups. Do not choose the electronics dataset merely by filtering All Electronics in items_g1. First establish the relevant dataset group using its own evidence. When a category filter is required, inspect exact stored values instead of guessing their spelling or case. A zero from an unverified filter is not evidence that the requested population is empty.",
                "evidence": [category, db("4", 4), db("4", 7)],
                "falsifier": "Current table category enumeration or dataset organization differs.",
                "mechanism": "Avoid conflating table-group membership with product category labels.",
            },
        },
        "cohort_studies": {
            "candidate_a": {
                "name": "Repeated identical group queries add no observations",
                "scope": "Within one unchanged study database, for the same tool and exact group_expression; re-run after genuine data or expression changes.",
                "content": "In a past study, estimate_survival_by_group was called three consecutive times with the exact same CASE expression. All three returned identical per-group survival and cohort decomposition. Repeating the query consumed actions without new information. Track executed expressions and reuse their results; to investigate a different grouping, actually change group_expression. Verify the expression differs from the previous one before spending another action. This observation does not establish which alternative grouping gives better predictions.",
                "evidence": repeat,
                "falsifier": "Identical calls on unchanged data return new information.",
                "mechanism": "Reduce duplicate calls and preserve the fixed tool budget.",
            },
            "candidate_b": {
                "name": "Use observable cohort fit as a diagnostic",
                "scope": "A study exposing predict_cohort_survival and a CASE expression using columns and encodings verified in that study.",
                "content": "In two past studies, predict_cohort_survival(group_expression) returned named observable cohorts with sample size n, model survival, empirical KM survival and KL mismatch, plus an explicit unobservable list. A past fit with n=1 had a much larger mismatch than several larger groups. Before final submission, this tool can check a proposed grouping and expose cohort-specific discrepancies. Distinguish per-group estimates from named cohort estimates. Use only current-study results; the listed unobservable cohorts receive no validation from this tool. Small-sample KM and in-sample KL are not population ground truth or guarantees of final reward. Do not reuse another study's numerical estimates or CASE columns without checking its schema.",
                "evidence": fit,
                "falsifier": "Tool semantics differ or required columns are unavailable.",
                "mechanism": "Expose observable prediction errors; final reward benefit remains a hypothesis.",
            },
        },
        "generic": "Work carefully and systematically. Check assumptions and tool arguments against current observations. Keep track of what is known and unknown. Use the available action budget efficiently, examine results before drawing conclusions, and verify the final response follows the task requirements. Prefer evidence to unsupported guesses. Adapt your reasoning to the current question and stop when you have enough evidence to answer.",
        "irrelevant": "Reference facts about elementary geometry and calendars: a triangle has three sides and three vertices; a square has four equal sides; a rectangle has four right angles; a week contains seven days. These background facts do not describe this task's database or any study population.",
    }


def core_context(bank, task, arm):
    if arm == "none":
        return ""
    if arm in ("generic", "irrelevant"):
        return bank[arm]
    keys = ["candidate_a", "candidate_b"] if arm == "combined" else [arm]
    return "\n\n".join(
        f"Scope: {bank[task][key]['scope']}\nContent: {bank[task][key]['content']}\nInvalidation: {bank[task][key]['falsifier']}"
        for key in keys
    )


def matched_context(core, tokenizer, target):
    """Match nonempty arms to within eight tokens using shared irrelevant padding."""
    if not core:
        return ""
    text = "Optional reference. Apply only when relevant to the current task.\n" + core
    text += "\n\nUnrelated control background; this is not task evidence:"

    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)

    if len(encode(text)) > target:
        raise ValueError(
            "Experience exceeds frozen context budget; do not truncate evidence"
        )
    while len(encode(text + PADDING)) <= target:
        text += PADDING
    for word in PADDING.split():
        if len(encode(text + " " + word)) > target:
            break
        text += " " + word
    if not target - 8 <= len(encode(text)) <= target:
        raise ValueError("Length matching failed")
    return text


def paired(rows, arm, control, indices, seeds):
    lookup = {(r["arm"], r["canonical_index"], r["generation_seed"]): r for r in rows}
    differences = []
    for index in indices:
        values = []
        for seed in seeds:
            left = lookup.get((arm, index, seed), {})
            right = lookup.get((control, index, seed), {})
            if left.get("status") != "complete" or right.get("status") != "complete":
                return None  # Never silently select on survivors.
            if left["instance_id"] != right["instance_id"]:
                raise ValueError("Mismatched canonical instances")
            values.append(left["reward"] - right["reward"])
        differences.append({"index": index, "difference": statistics.mean(values)})
    numbers = [x["difference"] for x in differences]
    seed_means = {
        str(seed): statistics.mean(
            lookup[(arm, i, seed)]["reward"] - lookup[(control, i, seed)]["reward"]
            for i in indices
        )
        for seed in seeds
    }
    return {
        "mean_delta": statistics.mean(numbers),
        "n_instances": len(numbers),
        "wins": sum(x > 1e-12 for x in numbers),
        "ties": sum(abs(x) <= 1e-12 for x in numbers),
        "losses": sum(x < -1e-12 for x in numbers),
        "seed_mean_deltas": seed_means,
        "worst_leave_one_instance_out": min(
            (sum(numbers) - x) / (len(numbers) - 1) for x in numbers
        ),
        "pairs": differences,
    }


def select_candidate(rows, indices, seeds):
    evaluations, eligible = {}, []
    for arm in CANDIDATES:
        contrasts = {
            control: paired(rows, arm, control, indices, seeds) for control in ARMS[:3]
        }
        passed = all(
            contrast is not None
            and contrast["mean_delta"] > 0
            and all(delta > 0 for delta in contrast["seed_mean_deltas"].values())
            and contrast["wins"] >= 2
            and contrast["wins"] >= contrast["losses"]
            and contrast["worst_leave_one_instance_out"] > 0
            for contrast in contrasts.values()
        )
        evaluations[arm] = {"eligible": passed, "contrasts": contrasts}
        if passed:
            eligible.append(arm)
    selected = max(
        eligible,
        key=lambda arm: evaluations[arm]["contrasts"]["none"]["mean_delta"],
        default=None,
    )
    return {
        "selected": selected,
        "evaluations": evaluations,
        "rule": "Positive in both screening seeds against all three controls; >=2 instance wins, wins>=losses, positive after dropping any one instance. Max mean reward gain vs none breaks selection; predefined arm order breaks ties.",
    }


def bootstrap_interval(contrast, draws=10000):
    """Resample instances, averaging decoding replicates before resampling."""
    values = [p["difference"] for p in contrast["pairs"]]
    rng = random.Random(74829)
    boot = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return [boot[int(draws * 0.025)], boot[int(draws * 0.975)]]


def render_report(root, plan):
    root = Path(root)
    lines = [
        "# 核验经验：开发筛选与留出确认",
        "",
        "只在完整配对上比较；reward 用于离线经验筛选，未写进执行模型。",
        "",
        "| 任务 | 阶段 | 组别 | 完成/计划 | 平均 reward |",
        "|---|---|---|---:|---:|",
    ]
    for task, spec in plan["tasks"].items():
        path = root / task / "results.json"
        rows = read(path) if path.exists() else []
        selection_path = root / task / "selection.json"
        selection = read(selection_path) if selection_path.exists() else {}
        for phase, indices, seeds in [
            ("screen", spec["screen_indices"], plan["screen_seeds"]),
            ("confirm", spec["confirm_indices"], plan["confirm_seeds"]),
        ]:
            arms = (
                ARMS
                if phase == "screen"
                else ARMS[:3]
                + ([selection["selected"]] if selection.get("selected") else [])
            )
            if phase == "confirm" and not selection.get("selected"):
                continue
            for arm in arms:
                found = [
                    r
                    for r in rows
                    if r["phase"] == phase
                    and r["arm"] == arm
                    and r["status"] == "complete"
                ]
                mean = (
                    f"{statistics.mean(r['reward'] for r in found):.6f}"
                    if found
                    else "—"
                )
                lines.append(
                    f"| {task} | {phase} | {arm} | {len(found)}/{len(indices) * len(seeds)} | {mean} |"
                )
        if selection:
            lines += [
                "",
                f"{task} 开发筛选：{selection.get('selected') or '没有候选通过预定门槛；不启动确认，不宣称经验有效。'}",
                "",
            ]
            for arm, evaluation in selection["evaluations"].items():
                contrast = evaluation["contrasts"]["none"]
                if contrast:
                    lines.append(
                        f"- 开发 {arm} vs none：Δ={contrast['mean_delta']:+.6f}；"
                        f"胜/平/负={contrast['wins']}/{contrast['ties']}/{contrast['losses']}；"
                        f"通过全部筛选条件={evaluation['eligible']}。"
                    )
        confirmation_path = root / task / "confirmation.json"
        if confirmation_path.exists():
            confirmation = read(confirmation_path)
            lines += [f"{task} 确认结论：{confirmation['conclusion']}", ""]
            for control, contrast in confirmation["contrasts"].items():
                if contrast:
                    lines.append(
                        f"- vs {control}: Δ={contrast['mean_delta']:+.6f}; 95% 实例 bootstrap 区间={contrast['interval95']}; 胜/平/负={contrast['wins']}/{contrast['ties']}/{contrast['losses']}。"
                    )
    lines += [
        "",
        "所有带上下文的组使用相同 token 长度（共用无关背景补齐）；none 保持无额外上下文。",
        "screen 是已查看过的开发前缀，不能当泛化证据。confirm 之前冻结候选与选择；确认阶段不更新记忆。",
        "多个 decoding seed 不等于独立数据集。数据库确认仍来自同一个数据库；cohort 留出只有 FORGE/CADENCE 两个研究，置信区间不能代表跨研究总体稳定性。",
        "这是人工核验经验的使用实验，未测试自动提取能力或在线经验增长。测试过的负结果与失败均保留。",
        "协议、证据、固定模型上下文、源码哈希、逐步交互和成本分别见 plan.json / bank.json / contexts.json / source_hashes.json / 任务目录。",
    ]
    temporary = root / "REPORT.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "REPORT.md")
    diagnostic_lines = [
        "# 行为与成本诊断",
        "",
        "这些是机制诊断，不代替官方 reward；仅完整实例计入均值，失败单独计数。",
        "",
        "| 任务 | 阶段 | 组别 | 完成/失败 | 成功率 | 平均调用 | 平均重复分组 | 平均标识符错误 | 平均预测检查 | 输入/输出 tokens（含失败） |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for task in plan["tasks"]:
        path = root / task / "results.json"
        records = read(path) if path.exists() else []
        for phase in ("screen", "confirm"):
            for arm in ARMS:
                group = [r for r in records if r["phase"] == phase and r["arm"] == arm]
                if not group:
                    continue
                complete = [r for r in group if r["status"] == "complete"]
                successes = [
                    r["success"] for r in complete if r.get("success") is not None
                ]
                success = f"{statistics.mean(successes):.3f}" if successes else "—"
                values = [
                    statistics.mean(r["model_calls"] for r in complete)
                    if complete
                    else 0,
                    *[
                        statistics.mean(r["diagnostics"][key] for r in complete)
                        if complete
                        else 0
                        for key in (
                            "duplicate_group_calls",
                            "nonexistent_identifier_errors",
                            "predict_cohort_calls",
                        )
                    ],
                ]
                diagnostic_lines.append(
                    f"| {task} | {phase} | {arm} | {len(complete)}/{len(group) - len(complete)} | {success} | "
                    + " | ".join(f"{value:.2f}" for value in values)
                    + f" | {sum(r['input_tokens'] for r in group)}/{sum(r['output_tokens'] for r in group)} |"
                )
    diagnostic_temp = root / "DIAGNOSTICS.tmp"
    diagnostic_temp.write_text("\n".join(diagnostic_lines) + "\n")
    diagnostic_temp.replace(root / "DIAGNOSTICS.md")


def diagnostics(output, task):
    events = (
        [
            json.loads(line)
            for line in (output / "responses.jsonl").read_text().splitlines()
        ]
        if (output / "responses.jsonl").exists()
        else []
    )
    observations = (
        [
            json.loads(line)
            for line in (output / "public_observations.jsonl").read_text().splitlines()
        ]
        if (output / "public_observations.jsonl").exists()
        else []
    )
    actions = [event["action"] for event in events if event.get("action")]
    tools = [a.get("tool_call", {}) for a in actions]
    expressions = [
        json.dumps(t, sort_keys=True) for t in tools if "group_expression" in t
    ]
    return {
        "duplicate_group_calls": len(expressions) - len(set(expressions)),
        "predict_cohort_calls": sum(
            t.get("tool") == "predict_cohort_survival" for t in tools
        ),
        "sql_errors": sum("ERROR:" in o["content"] for o in observations),
        "nonexistent_identifier_errors": sum(
            "no such table" in o["content"] or "no such column" in o["content"]
            for o in observations
        ),
        "accepted_actions": len(actions),
        "format_retries": sum(bool(e.get("format_retry")) for e in events),
        "logged_input_tokens": sum(e["input_tokens"] for e in events),
        "logged_output_tokens": sum(e["output_tokens"] for e in events),
    }


def run_trial(base, model, args, index, output, context):
    class FixedSystem(base.StructuredSystem):
        def experience_context(self, prompt):
            return context

        def get_run_artifacts(self):
            return {"frozen_context_sha256": digest(context), "online_updates": 0}

    output.mkdir(parents=True, exist_ok=False)
    task = base.make_task(args.task, args.environment_seed, independent=True)
    query = task.reset_baseline_instance(index)
    brief = task.get_agent_brief()
    system = FixedSystem(
        args,
        "independent",
        model,
        output,
        base.format_task_agent_brief(brief) if brief else "",
    )
    system.requested_canonical_index = index
    recorder = base.Recorder(output, system, 1)
    try:
        result = base.run_task(
            task,
            system,
            trace_recorder=recorder,
            show_progress=False,
            reset_system=False,
            initial_query=query,
        )
        if len(result.instance_outcomes) != 1:
            raise ValueError("Expected one official outcome")
        outcome = result.instance_outcomes[0]
        record = {
            "status": "complete",
            "instance_id": outcome.instance_id,
            "reward": outcome.reward,
            "success": outcome.success,
        }
    except Exception as exc:
        record = {
            "status": "failed",
            "instance_id": query.instance_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        connection = getattr(task, "_conn", None)
        if connection is not None:
            connection.close()
    metrics = recorder.metrics()
    write(output / "metrics.json", {**metrics, **record})
    diag = diagnostics(output, args.task)
    if (diag["logged_input_tokens"], diag["logged_output_tokens"]) != (
        system.input_tokens,
        system.output_tokens,
    ):
        raise ValueError("Token accounting mismatch")
    first = json.loads((output / "memory_contexts.jsonl").read_text().splitlines()[0])
    if first["memory_context"] != context:
        raise ValueError("Frozen experience missing or altered in first prompt")
    return {
        **record,
        "canonical_index": index,
        "generation_seed": args.seed,
        "model_calls": system.calls,
        "input_tokens": system.input_tokens,
        "output_tokens": system.output_tokens,
        "elapsed_seconds": metrics["elapsed_seconds"],
        "diagnostics": diag,
        "context_sha256": digest(context),
    }


def worker(root, task_name):
    from ttcl.structured_memory import run_benchmark as base

    root = Path(root)
    plan, contexts = read(root / "plan.json"), read(root / "contexts.json")
    spec = plan["tasks"][task_name]
    directory = root / task_name
    directory.mkdir(exist_ok=False)
    args = SimpleNamespace(
        **plan["model"],
        task=task_name,
        seed=42,
        environment_seed=42,
        num_instances=spec["canonical_total"],
        allow_initial_experience=True,
    )
    os.chdir(base.BENCH)
    model = base.LocalQwen(args)
    rows = []

    def phase(name, arms, indices, seeds):
        for seed in seeds:
            for index in indices:
                order = list(arms)
                random.Random(f"{task_name}:{name}:{index}:{seed}").shuffle(order)
                for arm in order:
                    args.seed = seed
                    output = (
                        directory
                        / name
                        / arm
                        / f"seed_{seed}"
                        / f"instance_{index:03d}"
                    )
                    write(
                        directory / "status.json",
                        {
                            "status": "running",
                            "phase": name,
                            "arm": arm,
                            "canonical_index": index,
                            "seed": seed,
                        },
                    )
                    row = run_trial(
                        base,
                        model,
                        args,
                        index,
                        output,
                        contexts[task_name][arm]["text"],
                    )
                    rows.append({"phase": name, "arm": arm, **row})
                    write(directory / "results.json", rows)

    phase("screen", ARMS, spec["screen_indices"], plan["screen_seeds"])
    selection = select_candidate(rows, spec["screen_indices"], plan["screen_seeds"])
    selection["bank_sha256"] = digest((root / "bank.json").read_text())
    selection["selected_at"] = datetime.now(timezone.utc).isoformat()
    write(directory / "selection.json", selection)
    if selection["selected"]:
        selected = selection["selected"]
        phase(
            "confirm",
            ARMS[:3] + [selected],
            spec["confirm_indices"],
            plan["confirm_seeds"],
        )
        confirm_rows = [row for row in rows if row["phase"] == "confirm"]
        contrasts = {
            control: paired(
                confirm_rows,
                selected,
                control,
                spec["confirm_indices"],
                plan["confirm_seeds"],
            )
            for control in ARMS[:3]
        }
        for contrast in contrasts.values():
            if contrast:
                contrast["interval95"] = bootstrap_interval(contrast)
                if task_name == "cohort_studies":
                    contrast["study_mean_deltas"] = {
                        study: statistics.mean(
                            p["difference"]
                            for p in contrast["pairs"]
                            if p["index"] in scope
                        )
                        for study, scope in {
                            "FORGE": range(12, 16),
                            "CADENCE": range(16, 20),
                        }.items()
                    }
        passed = all(
            c
            and c["interval95"][0] > 0
            and all(d > 0 for d in c["seed_mean_deltas"].values())
            and c["worst_leave_one_instance_out"] > 0
            and all(d > 0 for d in c.get("study_mean_deltas", {}).values())
            for c in contrasts.values()
        )
        write(
            directory / "confirmation.json",
            {
                "selected": selected,
                "contrasts": contrasts,
                "passed": passed,
                "conclusion": "在本次固定任务范围内通过预定确认标准；跨数据库/更多独立研究尚未验证。"
                if passed
                else "未通过预定确认标准；不能称为稳定有效经验。",
            },
        )
    write(
        directory / "status.json",
        {"status": "finished", "selected": selection["selected"], "trials": len(rows)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--task", required=True, choices=["database_exploration", "cohort_studies"]
    )
    args = parser.parse_args()
    worker(args.root.resolve(), args.task)


if __name__ == "__main__":
    main()
