"""Integrity checks for pinned public APEx data, without network or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('fetch_assets',Path(__file__).with_name('fetch_assets.py'))
assets=importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class APExDataTests(unittest.TestCase):
    def fixture(self, root):
        config={'dataset':{'repo':'example/data','revision':'fixed-commit',
            'directory':'data/apex/MIA','files':[{'filename':'Train/train.parquet',
            'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}]}}
        (root/'config').mkdir()
        (root/'config/apex-reproduction.json').write_text(json.dumps(config))
        return root/'data/apex/MIA/Train/train.parquet'

    def test_verified_download_is_pinned_and_idempotent(self):
        def fetch(url, target):
            self.assertIn('/fixed-commit/Train/train.parquet',url)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b'fixture')
            return hashlib.sha256(b'fixture').hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=self.fixture(root)
            with patch.object(assets,'ROOT',root), patch.object(assets,'download',side_effect=fetch) as call:
                assets.fetch('apex-data');assets.fetch('apex-data')
                self.assertEqual(call.call_count,1)
                self.assertEqual(target.read_bytes(),b'fixture')
                self.assertTrue((root/'data/apex/MIA/download_provenance.json').is_file())

    def test_corrupt_download_never_becomes_training_input(self):
        def fetch(url,target):
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b'corrupt')
            return hashlib.sha256(b'corrupt').hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=self.fixture(root)
            with patch.object(assets,'ROOT',root),patch.object(assets,'download',side_effect=fetch):
                with self.assertRaises(ValueError):assets.fetch('apex-data')
                self.assertFalse(target.exists())

    def test_existing_changed_file_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=self.fixture(root)
            target.parent.mkdir(parents=True);target.write_bytes(b'changed')
            with patch.object(assets,'ROOT',root),patch.object(assets,'download') as call:
                with self.assertRaises(ValueError):assets.fetch('apex-data')
                call.assert_not_called()
                self.assertEqual(target.read_bytes(),b'changed')


if __name__=='__main__':unittest.main()
