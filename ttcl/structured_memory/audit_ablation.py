"""Read-only trace audit for the experience-content pilot; writes audit.json."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def audit(root):
    manifest = read(root / "manifest.json")
    snapshot_errors = []
    for name, expected in read(root / "source_hashes.json").items():
        path = root / "source" / name
        if not path.exists():
            # Benchmark source hashes are rooted at the workspace.
            path = Path(manifest["environment"]["TTCL_BENCH"]).parents[1] / name
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            snapshot_errors.append(name)
    tasks = {}
    for task in manifest["commands"]:
        task_root = root / task
        if not (task_root / "results.json").exists():
            tasks[task] = {"status": "not_finished"}
            continue
        results = read(task_root / "results.json")
        variants = {}
        for variant, metrics in results.items():
            mode = "independent" if variant == "independent" else "structured"
            folder = task_root / variant / mode
            responses = rows(folder / "responses.jsonl")
            contexts = rows(folder / "memory_contexts.jsonl")
            observations = rows(folder / "public_observations.jsonl")
            cases = rows(folder / "experience_cases.jsonl")
            errors = []
            if len(responses) != metrics["model_calls"]:
                errors.append("model_call_count")
            for key in ("input_tokens", "output_tokens"):
                if sum(r[key] for r in responses) != metrics[key]:
                    errors.append(key)
            accepted = [r for r in responses if r.get("action") is not None]
            if len(accepted) != len(observations):
                errors.append("accepted_action_vs_observation_count")
            if contexts and contexts[0]["memory_context"]:
                errors.append("first_episode_has_history")
            if variant == "independent" and any(r["memory_context"] for r in contexts):
                errors.append("independent_has_history")
            if variant == "focused" and any("reward" in c for c in cases):
                errors.append("scalar_visible_in_focused_cases")
            if variant == "focused_reward":
                by_alias = {c["public_episode"]: c["reward"] for c in cases}
                if (
                    metrics["status"] == "complete"
                    and len(cases) != metrics["completed_instances"]
                ):
                    errors.append("missing_reward_cases")
                for index, outcome in enumerate(metrics["outcomes"], 1):
                    if by_alias.get(f"experience_{index}") != outcome["reward"]:
                        errors.append(f"reward_case_mismatch_{index}")
            if (
                metrics["status"] == "complete"
                and len(contexts) != metrics["completed_instances"]
            ):
                errors.append("episode_count")
            for c in contexts:
                # Canonical IDs can contain hidden variant names. Only public
                # experience aliases may appear in rendered memory.
                if (
                    c["instance_id"]
                    and c["instance_id"].startswith("poker:")
                    and c["instance_id"] in c["memory_context"]
                ):
                    errors.append("canonical_poker_id_in_memory")
            variants[variant] = {
                "status": metrics["status"],
                "errors": errors,
                "model_calls": len(responses),
                "accepted_actions": len(accepted),
                "format_retries": sum(r.get("format_retry", 0) > 0 for r in responses),
                "parse_errors": sum(bool(r.get("parse_error")) for r in responses),
                "packaging_repairs": sum(
                    bool(r.get("packaging_repair")) for r in responses
                ),
                "first_prompt_hash": responses[0].get("rendered_prompt_sha256")
                if responses
                else None,
                "max_memory_chars": max(
                    (len(r["memory_context"]) for r in contexts), default=0
                ),
            }
        complete = [r for r in results.values() if r.get("status") == "complete"]
        paired_sets = (
            all(
                {o["instance_id"] for o in r["outcomes"]}
                == {o["instance_id"] for o in complete[0]["outcomes"]}
                for r in complete
            )
            if complete
            else False
        )
        first_hashes = {
            r["first_prompt_hash"] for r in variants.values() if r["first_prompt_hash"]
        }
        tasks[task] = {
            "variants": variants,
            "complete_instance_sets_match": paired_sets,
            "first_prompts_match": len(first_hashes) == 1,
        }
    return {
        "snapshot_hash_errors": snapshot_errors,
        "tasks": tasks,
        "scope": "Checks trace accounting, first prompts, pairing and scalar routing. Does not prove semantic correctness or causal usefulness of memory.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    (args.root / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
