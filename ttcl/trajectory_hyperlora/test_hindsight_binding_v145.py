import copy
import tempfile
import unittest
from pathlib import Path
from .freeze_hindsight_v139 import digest, sha256, validate_manifest


class BindingTest(unittest.TestCase):
    def test_changed_protocol_content_order_and_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'game').write_text('official bytes')
            spec = {'name': 'test', 'credit_steps': 0}
            content = {'index': 0, 'game': 'game', 'game_sha256': sha256(root / 'game'),
                       'protocol': spec, 'source_manifest_sha256': 'source'}
            row = {**content, 'input_content_sha256': digest(content), 'reviewed_target': True}
            manifest = {'protocol_spec': spec, 'source_manifest_sha256': 'source', 'targets': [row]}
            args = {'credit_steps': 0, 'limit': 1}
            validate_manifest(manifest, args, root)
            with self.assertRaisesRegex(ValueError, 'credit_steps'):
                validate_manifest(manifest, {**args, 'credit_steps': 4}, root)
            altered = copy.deepcopy(manifest)
            altered['targets'][0]['index'] = 1
            with self.assertRaises(ValueError):
                validate_manifest(altered, args, root)
            altered = copy.deepcopy(manifest)
            altered['targets'][0]['reviewed_target'] = False
            with self.assertRaises(ValueError):
                validate_manifest(altered, args, root)
            (root / 'game').write_text('changed bytes')
            with self.assertRaisesRegex(ValueError, 'game content'):
                validate_manifest(manifest, args, root)


if __name__ == '__main__':
    unittest.main()
