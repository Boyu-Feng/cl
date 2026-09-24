"""Train RAMP from ordered prompt/response/reward JSONL, without a teacher."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttcl.ramp.run_reward_benchmark import (  # noqa: E402
    RewardMemory,
    append_json,
    build_parser,
    prepare_output,
    validate_args,
)


def run(args, memory_factory=RewardMemory):
    if args.method == "frozen":
        raise ValueError("Offline training requires --method ramp")
    output = prepare_output(args)
    memory = memory_factory(args)
    count = 0
    with Path(args.input).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt, response, reward = row["prompt"], row["response"], row["reward"]
                if not isinstance(prompt, str) or not isinstance(response, str):
                    raise ValueError("prompt/response must be strings")
                rendered = row.get("prompt_is_rendered", False)
                if not isinstance(rendered, bool):
                    raise ValueError("prompt_is_rendered must be boolean")
                if not rendered:
                    prompt = memory.render(prompt)
                item = memory.learner.observe(
                    prompt,
                    response,
                    reward,
                    task_id=row.get("task_id", "default"),
                    valid=row.get("valid", True),
                    response_ids=row.get("response_ids"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid input line {line_number}: {exc}") from exc
            count += 1
            append_json(output / "signals.jsonl", asdict(item))
            if count % args.update_every == 0:
                update = memory.adapt(output)
                append_json(output / "updates.jsonl", update)
                print(json.dumps(update), flush=True)
    if not count:
        raise ValueError("Input contains no examples")
    # Offline training consumes its final window; online evaluation intentionally doesn't.
    if memory.learner.buffer.pending:
        append_json(output / "updates.jsonl", memory.adapt(output))
    memory.save(output)
    memory.audit(output)
    result = {
        "examples": count,
        "num_updates": memory.updates,
        "update_attempts": memory.learner.attempts,
        "adapter": str(output / "latest_adapter"),
        "note": "Training metrics only; evaluate on disjoint held-out tasks.",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = build_parser(__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    validate_args(parser, args)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
