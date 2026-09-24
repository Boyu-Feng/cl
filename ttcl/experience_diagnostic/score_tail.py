"""Execute predeclared late cells early on another GPU, without new trials.

The main scorer resumes existing row.json cells. Run this only while the main
worker is still on the first history, well before these late cells are reached.
"""

import copy
import json
from pathlib import Path
import sys

from ttcl.experience_diagnostic import run


def main(root, repeat):
    original_read, original_save = run.read, run.save

    def subset_read(path, default=None):
        value = original_read(path, default)
        if path == root / 'plan.json':
            value = copy.deepcopy(value)
            value['source_episodes'] = [17]
            value['probe_offsets'] = [2]
        return value

    def save(path, value):
        if path.parent == root / 'progress':
            path = path.with_name('tail_' + path.name)
        original_save(path, value)

    run.read, run.save = subset_read, save
    # Actual model, context, task, seeds, and action budgets are unchanged.
    save(root / 'scheduling' / f'tail_{repeat}.json', {
        'task': 'cohort_studies', 'source_episode': 17, 'probe_episode': 19,
        'repeat': repeat, 'arms': run.ARMS, 'change': 'execution order only',
    })
    run.score(root, 'cohort_studies', repeat)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), int(sys.argv[2]))
