"""Record the protocol, selected tasks, runtime and source before any evaluation."""
from pathlib import Path
from importlib.metadata import version
import shutil
import time
from .core import read, save, digest, workspace_root


if __name__ == '__main__':
    workspace = workspace_root()
    root = workspace / 'ttcl/results/experience_evolution/alfworld_delta_20260922'
    assert not (root / 'freeze.json').exists()
    plan = read(root / 'plan.json')
    assert read(root / 'calibration.json')['reset_check_passed']
    assert not (root / 'evaluation').exists() and not (root / 'training').exists()
    snapshot = root / 'source'
    snapshot.mkdir()
    source_hashes = {}
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in ['.py', '.md']:
            shutil.copy2(path, snapshot / path.name)
            source_hashes[path.name] = digest(path.read_bytes())
    paths_module = workspace / 'ttcl/paths.py'
    shutil.copy2(paths_module, snapshot / 'paths.py')
    source_hashes['paths.py'] = digest(paths_module.read_bytes())
    provenance = read(Path(plan['data_root']) / 'download_provenance.json')
    save(root / 'freeze.json', {'time': time.time(), 'plan_sha256': digest((root/'plan.json').read_bytes()),
         'source_sha256': source_hashes, 'data_hashes_sha256': digest((root/'data_hashes.json').read_bytes()),
         'download': provenance, 'versions': {p: version(p) for p in
         ['torch', 'transformers', 'peft', 'vllm', 'alfworld', 'textworld', 'fast-downward-textworld']}})
    print('Frozen protocol and source before training/evaluation.')
