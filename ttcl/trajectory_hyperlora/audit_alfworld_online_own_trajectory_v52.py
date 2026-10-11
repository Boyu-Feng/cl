"""Audit chronology, content bindings and environment outcomes for v52."""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import action_rows
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52 import (
    bound, parser,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def main():
    args = parser().parse_args()
    rows, fixed, targets = bound(args)
    report = json.loads(args.output.read_text())
    review = json.loads(args.review.read_text())
    annotations = json.loads(args.annotations.read_text())
    assert report["review_sha256"] == file_hash(args.review)
    assert review["targets"] == targets
    assert all(report[k] == value for k, value in fixed.items())
    assert len(rows) == len(targets) == len(report["rows"]) == 240
    assert not report["failures"]
    own_wins = []
    reviewed = []
    update_steps = 0
    for i, (row, target, result) in enumerate(zip(
            rows, targets, report["rows"], strict=True)):
        game = args.data_root / row["game"]
        assert result["index"] == i and result["game"] == row["game"]
        assert result["target_input_content_sha256"] == target["input_content_sha256"]
        assert result["memory_before"] == len(own_wins)
        assert result["source_index"] is None or 0 <= result["source_index"] < len(own_wins)
        j = result["source_index"]
        assert j == (source_for(target, own_wins) if any(
            x["family"] == target["family"] for x in own_wins) else None)
        if j is not None:
            assert own_wins[j]["family"] == target["family"]
            assert result["source_episode_sha256"] == own_wins[j]["episode_sha256"]
        same_family = sum(x["family"] == target["family"] for x in own_wins)
        assert result["adapter_active"] == (len(own_wins) >= 8 and same_family >= 2)
        for arm in ("online", "frozen_same_sources"):
            episode = result[arm]
            assert episode["status"] == "complete"
            assert episode["initial_observation"] == row["arms"]["base"]["initial_observation"]
            replay(game, episode)
        if not result["adapter_active"]:
            assert result["online"] == result["frozen_same_sources"]
        update_steps += result["update"]["steps"]
        if result["online"]["reward"] == 1:
            episode = result["online"]
            actions = action_rows(game, episode)
            content = {"stream_index": i, "game": target["game"],
                "family": target["family"], "game_sha256": target["game_sha256"],
                "episode_sha256": digest(episode),
                "source_records_sha256": digest(records_from_episode(episode)),
                "actions_sha256": digest(actions),
                "runner_sha256": fixed["runner_sha256"]}
            annotation = annotations["annotations"][len(reviewed)]
            assert all(annotation[k] == value for k, value in content.items())
            assert annotation["input_content_sha256"] == digest(content)
            assert result["annotation_input_content_sha256"] == annotation["input_content_sha256"]
            assert annotation["reviewed_target"] is True
            own_wins.append({"game": target["game"], "family": target["family"],
                             "episode_sha256": digest(episode)})
            reviewed.append(annotation)
        else:
            assert result["annotation_input_content_sha256"] is None
            assert result["update"]["steps"] == 0
        if result["update"]["steps"]:
            assert result["online"]["reward"] == 1
            if result["parameters_before_sha256"] is not None:
                assert result["parameters_before_sha256"] != result["parameters_after_sha256"]
    assert len(reviewed) == len(annotations["annotations"])
    assert annotations["eval_review_sha256"] == file_hash(args.review)
    assert report["summary"] == {
        "n": 240,
        "online": sum(x["online"]["reward"] for x in report["rows"]),
        "frozen_same_sources": sum(x["frozen_same_sources"]["reward"] for x in report["rows"]),
        "own_successes": len(own_wins),
        "online_updates": update_steps,
        "failures": 0,
    }
    assert report["bootstrap"]["own_wins"] == 8
    assert report["bootstrap"]["steps"] == 300
    assert report["bootstrap"]["parameters_before_sha256"] != report["bootstrap"]["parameters_after_sha256"]
    model = Path(args.model_output)
    assert model.exists()
    print(json.dumps({"passed": True, "summary": report["summary"],
                      "report_sha256": file_hash(args.output),
                      "model_sha256": file_hash(model)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
