"""Run the two remaining disjoint late cells with separate scheduling logs."""

from pathlib import Path
import sys

from ttcl.experience_diagnostic import run
from ttcl.experience_diagnostic.score_subset import main


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    save = run.save

    def isolated_save(path, value):
        if path.parent in [root / 'progress', root / 'scheduling']:
            path = path.with_name('prefix_' + path.name)
        save(path, value)

    run.save = isolated_save
    main(root, 606, ['keep', 'untrained'])
