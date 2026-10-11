"""Replay v53's scored episodes and check causal exposure and matched updates."""

from __future__ import annotations

import json
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import (
    alignment_pair, checked, parser,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import (
    rotated_source_for, source_for,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def main():
    args = parser().parse_args()
    rows, teacher, wins, targets, fixed = checked(args)
    report = json.loads(args.output.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert len(report["rows"]) == len(rows) == 240
    assert report["review_sha256"] == file_hash(args.review)
    assert all(report[k] == value for k, value in fixed.items())
    assert not report["failures"]
    memory = []
    scored = []
    updates = 0
    for i, (row, target, result) in enumerate(zip(
            rows, targets, report["rows"], strict=True)):
        assert result["index"] == i
        assert result["game"] == target["game"]
        assert result["target_input_content_sha256"] == target["input_content_sha256"]
        assert result["memory_before"] == len(memory)
        assert result["teacher_reward"] == teacher["rows"][i]["online"]["reward"]
        assert result["teacher_episode_sha256"] == digest(teacher["rows"][i]["online"])
        same = sum(x["family"] == row["family"] for x in memory)
        expected_source = source_for(row, memory) if same else None
        assert result["source_index"] == expected_source
        assert result["source_episode_sha256"] == (
            memory[expected_source]["episode_sha256"] if expected_source is not None else None)
        active = len(memory) >= 8 and same >= 2
        assert result["adapter_active"] == active
        if i > report["bootstrap"]["last_index"]:
            assert result["hyper_A_before_sha256"] == report["bootstrap"]["hyper_A_sha256"]
            assert result["hyper_A_after_sha256"] == report["bootstrap"]["hyper_A_sha256"]
        if i >= 120:
            scored.append(result)
            arms = ("hyper_online", "hyper_frozen", "online_static_lora")
            if active:
                k = rotated_source_for(row, memory, expected_source)
                assert result["rotated_source_index"] == k
                assert result["rotated_source_episode_sha256"] == memory[k]["episode_sha256"]
                assert memory[k]["family"] == row["family"]
                coherent, shifted = alignment_pair(memory[expected_source]["source_records"], tokenizer)
                assert result["coherent_source_records_sha256"] == digest(coherent)
                assert result["shifted_source_records_sha256"] == digest(shifted)
                arms += ("hyper_rotated", "hyper_shifted_actions")
            else:
                assert result["hyper_rotated"] is None
                assert result["hyper_shifted_actions"] is None
                assert result["hyper_online"] == result["hyper_frozen"]
            for arm in arms:
                ep = result[arm]
                assert ep["status"] == "complete"
                assert ep["initial_observation"] == row["arms"]["base"]["initial_observation"]
                replay(args.data_root / row["game"], ep)
        if i in wins:
            if len(memory) == 7:
                assert result["hyper_update"]["steps"] == 0
                assert result["static_update"]["steps"] == 0
            elif len(memory) >= 8 and expected_source is not None:
                assert result["hyper_update"]["steps"] == 20
                assert result["static_update"]["steps"] == 20
                assert result["hyper_before_update_sha256"] != result["hyper_after_update_sha256"]
                updates += 20
            else:
                assert result["hyper_update"]["steps"] == 0
                assert result["static_update"]["steps"] == 0
            memory.append(wins[i])
        else:
            assert result["hyper_update"]["steps"] == 0
            assert result["static_update"]["steps"] == 0
    assert len(scored) == 120
    assert len(memory) == len(wins)
    assert report["summary"] == {"processed": 240,
        "scored": 120, "own_success_memory": len(memory),
        "hyper_updates": updates, "static_updates": updates,
        **{arm: sum(x[arm]["reward"] for x in scored if x[arm] is not None)
           for arm in ("hyper_online", "hyper_frozen", "hyper_rotated", "hyper_shifted_actions",
                       "online_static_lora")},
        "rotated_n": sum(x["hyper_rotated"] is not None for x in scored),
        "failures": 0}
    assert report["bootstrap"]["own_wins"] == 8
    assert report["bootstrap"]["hyper"]["steps"] == 300
    assert report["bootstrap"]["static"]["steps"] == 300
    assert report["bootstrap"]["hyper_before_sha256"] != report["bootstrap"]["hyper_after_sha256"]
    assert args.hyper_model.exists() and args.static_model.exists()
    print(json.dumps({"passed": True, "summary": report["summary"],
        "report_sha256": file_hash(args.output),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "static_model_sha256": file_hash(args.static_model)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
