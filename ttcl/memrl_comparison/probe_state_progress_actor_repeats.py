"""Frozen two-seed repeat of the first state-progress ALFWorld train pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import sha
from . import probe_state_progress_actor as pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    args = parser.parse_args()
    pilot.REPEATS = (94402, 94403)
    design = pilot.design_for(args.plan.resolve(), args.reviews.resolve(), args.url)
    design['base_runner_sha256'] = design.pop('runner_sha256')
    design['runner_sha256'] = sha(Path(__file__))
    design['budget'] = ('Same six reviewed, hash-selected official train games; '
                        'two new actor seeds; native and state-progress arms; '
                        'at most 50 environment actions and actor calls each')
    design['caveat'] = ('Exploratory repeat after seeing the one-seed pilot; '
                        'not a fresh task holdout or valid_unseen evaluation')
    pilot.run(design, args.output.resolve(), False)


if __name__ == '__main__':
    main()
