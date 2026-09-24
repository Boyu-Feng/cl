"""CPU-only checks for a fresh checkout and its frozen experiment sources."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ttcl.experience_evolution.core import python_executable, workspace_root


class PortablePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ttcl-portability-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'new checkout'
        (self.repo / 'ttcl').mkdir(parents=True)
        (self.repo / 'current_work').mkdir()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_root_discovery_from_checkout_and_snapshot(self):
        module = self.repo / 'ttcl/experience_evolution/core.py'
        self.assertEqual(workspace_root(module), self.repo)
        frozen = self.repo / 'results/new/source/ttcl/experience_evolution/core.py'
        self.assertEqual(workspace_root(frozen), self.repo)

    def test_external_snapshot_uses_explicit_workspace(self):
        os.environ['TTCL_WORKSPACE'] = str(self.repo)
        self.assertEqual(workspace_root(self.base / 'separate/source/core.py'), self.repo)

    def test_missing_checkout_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, 'TTCL_WORKSPACE'):
            workspace_root(self.base / 'separate/source/core.py')

    def test_interpreter_fallback_and_runtime_preserve_venv_path(self):
        self.assertEqual(python_executable(self.repo), Path(sys.executable).absolute())
        runtime = self.repo / 'ttcl/.runtime/alf_delta_env/bin/python'
        runtime.parent.mkdir(parents=True)
        runtime.symlink_to(sys.executable)
        self.assertEqual(python_executable(self.repo), runtime)

    def test_explicit_interpreter_and_invalid_override(self):
        selected = self.repo / 'env/bin/python'
        selected.parent.mkdir(parents=True)
        selected.symlink_to(sys.executable)
        os.environ['TTCL_PYTHON'] = 'env/bin/python'
        self.assertEqual(python_executable(self.repo), selected)
        os.environ['TTCL_PYTHON'] = str(selected)
        self.assertEqual(python_executable(self.repo), selected)
        os.environ['TTCL_PYTHON'] = str(self.repo / 'missing/python')
        with self.assertRaisesRegex(FileNotFoundError, 'TTCL_PYTHON'):
            python_executable(self.repo)

    def test_imports_in_relocated_frozen_source(self):
        source = self.repo / 'results/experiment/source'
        package = source / 'ttcl'
        for name in ['experience_evolution', 'experience_v2']:
            (package / name).mkdir(parents=True)
            (package / name / '__init__.py').write_text('')
        (package / '__init__.py').write_text('')
        for name in ['experience_evolution/core.py', 'experience_v2/common.py', 'paths.py']:
            shutil.copy2(REPO / 'ttcl' / name, package / name)
        env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
        output = subprocess.check_output(
            [sys.executable, '-c',
             'import json; from ttcl.experience_evolution import core; '
             'from ttcl.experience_v2 import common; from ttcl import paths; '
             'print(json.dumps([str(core.ROOT), str(common.WORKSPACE), '
             'str(paths.WORKSPACE), str(common.PYTHON), '
             'common.environment(core.ROOT)["TTCL_WORKSPACE"]]))'],
            cwd=self.base, env=env, text=True,
        )
        actual = json.loads(output)
        self.assertEqual(actual[:3], [str(self.repo)] * 3)
        self.assertEqual(actual[3], str(Path(sys.executable).absolute()))
        self.assertEqual(actual[4], str(self.repo))


if __name__ == '__main__':
    unittest.main()
