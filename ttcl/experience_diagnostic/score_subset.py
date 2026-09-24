"""Schedule an untouched subset of the frozen cells on a freed GPU."""

import copy
from pathlib import Path
import sys

from ttcl.experience_diagnostic import run


def main(root, repeat, arms):
    task = 'cohort_studies'
    progress = run.read(root / 'progress' / f'{task}_{repeat}.json')
    if progress.get('source_episode') != 13:
        raise RuntimeError('Main worker is already near this subset; no extra worker started')
    for arm in arms:
        dest = root / 'scores' / task / str(repeat) / '17/18' / arm
        if dest.exists():
            raise RuntimeError(f'Cell already started; refusing concurrent execution: {dest}')
    original_read, original_save = run.read, run.save

    def subset_read(path, default=None):
        value = original_read(path, default)
        if path == root / 'plan.json':
            value = copy.deepcopy(value)
            value['source_episodes'], value['probe_offsets'] = [17], [1]
        return value

    def subset_save(path, value):
        if path.parent == root / 'progress':
            path = path.with_name('subset_' + path.name)
        original_save(path, value)

    run.read, run.save, run.ARMS = subset_read, subset_save, arms
    subset_save(root / 'scheduling' / f'subset_{repeat}.json', {
        'task': task, 'source_episode': 17, 'probe_episode': 18, 'repeat': repeat,
        'arms': arms, 'change': 'execution order only; same frozen plan, no extra trials',
        'main_progress_at_start': progress,
    })
    run.score(root, task, repeat)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3].split(','))
