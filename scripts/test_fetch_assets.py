"""Check database downloads without network access or heavyweight environments."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('fetch_assets', Path(__file__).with_name('fetch_assets.py'))
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class DatabaseDownloadTests(unittest.TestCase):
    def test_official_two_file_download_is_idempotent(self):
        def download(**kw):
            (Path(kw['local_dir'])/kw['filename']).write_bytes(b'fixture')
        mock = Mock(side_effect=download)
        with tempfile.TemporaryDirectory() as temp, patch.object(assets, 'ROOT', Path(temp)), \
             patch.dict('os.environ', {}, clear=True), \
             patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=mock)}):
            assets.fetch('clbench-db'); assets.fetch('clbench-db')
            self.assertEqual(mock.call_count, 2)
            self.assertEqual({c.kwargs['filename'] for c in mock.call_args_list},
                             {'products.db', 'products_drifted.db'})
            for call in mock.call_args_list:
                self.assertEqual(call.kwargs['repo_id'], 'continual-learning-bench/database-exploration')
                self.assertEqual(call.kwargs['repo_type'], 'dataset')
                self.assertEqual(call.kwargs['local_dir'], str(Path(temp)/'data/clbench/database_exploration'))

    def test_missing_download_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(assets, 'ROOT', Path(temp)), \
             patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=Mock())}):
            with self.assertRaises(FileNotFoundError): assets.fetch('clbench-db')


if __name__ == '__main__': unittest.main()
