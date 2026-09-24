"""Refresh the run report until both launched benchmark processes settle."""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import time

from report import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--database-pid", type=int, required=True)
    parser.add_argument("--cohort-pid", type=int, required=True)
    args = parser.parse_args()
    processes = {
        "database_exploration": args.database_pid,
        "cohort_studies": args.cohort_pid,
    }
    (args.root / "processes.json").write_text(json.dumps(processes, indent=2))
    while True:
        settled = 0
        for task, pid in processes.items():
            path = args.root / task / "status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            if status.get("status") in {"complete", "failed", "stopped"}:
                settled += 1
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                path.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "reason": "Process exited without a completion summary; inspect the task log.",
                        },
                        indent=2,
                    )
                )
                settled += 1
        with redirect_stdout(io.StringIO()):
            report(args.root)
        if settled == len(processes):
            print("All benchmark processes settled; REPORT.md updated.", flush=True)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
