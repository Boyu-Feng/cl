"""Pure controls and paired summaries; failed runs remain missing."""

import hashlib
import json
import statistics

ARMS = ["keep", "untrained", "utility_sft", "audited", "raw"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def paired_summary(rows):
    groups = {}
    for row in rows:
        key = (row["source_episode"], row["canonical_index"], row["repeat"])
        arm = row["arm"]
        if arm in groups.setdefault(key, {}):
            raise ValueError("Duplicate comparison cell")
        groups[key][arm] = row
    complete = []
    for key, values in sorted(groups.items()):
        if all(a in values and values[a]["status"] == "complete" for a in ARMS):
            if len({v["instance_id"] for v in values.values()}) != 1:
                raise ValueError("Mismatched probe identities")
            complete.append((key, values))
    scores = {}
    for arm in ARMS:
        deltas = [v[arm]["reward"] - v["keep"]["reward"] for _, v in complete]
        scores[arm] = {
            "mean_reward": statistics.mean(v[arm]["reward"] for _, v in complete) if complete else None,
            "delta_vs_keep": statistics.mean(deltas) if deltas else None,
            "wins": sum(d > 1e-9 for d in deltas),
            "ties": sum(abs(d) <= 1e-9 for d in deltas),
            "losses": sum(d < -1e-9 for d in deltas),
            "mean_calls": statistics.mean(v[arm]["actor_calls"] for _, v in complete) if complete else None,
        }
    return {
        "paired_count": len(complete),
        "arms": scores,
        "pairs": [{"source_episode": k[0], "probe_index": k[1], "repeat": k[2],
                   "rewards": {a: v[a]["reward"] for a in ARMS}} for k, v in complete],
        "missing_or_failed": [{"key": k, "arm": a, "status": v.get(a, {}).get("status", "not_run")}
                              for k, v in groups.items() for a in ARMS
                              if v.get(a, {}).get("status") != "complete"],
    }


def extract_raw(episode, before, spec, count, limit=2048):
    """Pack predeclared verbatim excerpts, then old whole entries; record omissions."""
    excerpts = []
    for item in spec:
        step = next(s for s in episode["steps"] if s["step"] == item["step"])
        feedback = step["public_feedback"]
        text = item["text"]
        if text not in feedback:
            raise ValueError("Evidence excerpt is not verbatim public feedback")
        excerpts.append({"episode": episode["episode"], "step": item["step"], "quote": text})
    value = {"public_tool_excerpts": excerpts, "earlier_bank_entries": []}
    prefix = "Earlier public tool evidence, quoted verbatim. Scope may differ in the current task. Earlier bank entries are unverified notes.\n"
    render = lambda: prefix + json.dumps(value, ensure_ascii=False, sort_keys=True)
    if count(render()) > limit:
        raise ValueError("Declared raw excerpts exceed matched context budget")
    omitted = []
    for entry in before["entries"]:
        value["earlier_bank_entries"].append(entry)
        if count(render()) > limit:
            value["earlier_bank_entries"].pop()
            omitted.append(entry["id"])
    return render(), {"excerpts": excerpts, "omitted_old_ids": omitted,
                      "tokens": count(render()), "selection": "Predeclared from past feedback, no probe access; excerpts first, then whole old entries in original order."}
