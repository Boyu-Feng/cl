"""Two fresh actor seeds for the frozen six-game state-progress holdout."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from . import probe_state_progress_actor as pilot
from . import probe_state_progress_holdout as holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve() / 'design.json'
    design = read(source)
    if design['schema'] != 'alf_state_progress_pilot_v1' or len(design['cases']) != 6:
        raise ValueError('Wrong frozen holdout design')
    pilot._goal = holdout.extended_goal
    pilot.REPEATS = (94502, 94503)
    design['source_holdout_design_sha256'] = sha(source)
    design['base_runner_sha256'] = design.pop('runner_sha256')
    design['runner_sha256'] = sha(Path(__file__))
    design['repeats'] = list(pilot.REPEATS)
    design['budget'] = ('Same six content-bound new official train games, '
                        'two fresh actor seeds, two arms, max 50 actions '
                        'and actor calls each')
    design['caveat'] = ('Exploratory repeat after inspecting the first holdout seed; '
                        'same games, not a third independent task set')
    pilot.run(design, args.output.resolve(), False)


if __name__ == '__main__':
    main()
