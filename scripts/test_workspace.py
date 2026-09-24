import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('workspace_layout', Path(__file__).with_name('prepare_workspace.py'))
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class WorkspaceTests(unittest.TestCase):
    def test_fresh_clone_creates_only_local_asset_links(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(layout, 'ROOT', Path(directory)):
            root = Path(directory)
            (root/'config').mkdir()
            (root/'config/asset-layout.json').write_text(json.dumps({'aliases':[
                {'legacy':'ttcl/data','storage':'data/ttcl','kind':'directory'},
                {'legacy':'example/input.json','storage':'data/example/input.json','kind':'file'},
            ]}))
            layout.prepare()
            layout.prepare()
            self.assertEqual((root/'ttcl/data').resolve(),root/'data/ttcl')
            self.assertTrue((root/'example/input.json').is_symlink())
            self.assertFalse((root/'data/example/input.json').exists())

    def test_existing_assets_are_not_moved_without_explicit_option(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(layout, 'ROOT', Path(directory)):
            root = Path(directory)
            (root/'old').mkdir()
            (root/'old/value').write_text('original')
            with self.assertRaises(FileExistsError):
                layout.place('old','data/old','directory')
            self.assertEqual((root/'old/value').read_text(),'original')
            self.assertFalse((root/'old').is_symlink())

    @unittest.skipUnless(os.name == 'posix', 'Atomic asset organization requires Linux renameat2')
    def test_explicit_atomic_move_keeps_existing_open_file_valid(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(layout, 'ROOT', Path(directory)):
            root = Path(directory)
            (root/'old').mkdir()
            (root/'old/value').write_text('original')
            with (root/'old/value').open() as already_open:
                self.assertEqual(layout.place('old','data/old','directory',True),'moved')
                self.assertEqual(already_open.read(),'original')
            self.assertEqual((root/'old/value').read_text(),'original')
            self.assertEqual(layout.place('old','data/old','directory',True),'linked')


if __name__ == '__main__':
    unittest.main()
