"""Audit public-goal syntax extension and official ALFWorld holdout traces."""
from __future__ import annotations

from . import probe_state_progress_actor as pilot
from . import probe_state_progress_holdout as holdout
from . import audit_state_progress_actor as audit


def main() -> None:
    pilot._goal = holdout.extended_goal
    audit.main()


if __name__ == '__main__':
    main()
