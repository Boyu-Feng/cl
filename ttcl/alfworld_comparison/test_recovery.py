"""CPU-only integrity checks for an infrastructure restart's frozen bank."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .recovery import restore_bank, verify_reused_bank, _sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


class BankRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='alf-bank-recovery-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.previous, self.current = root / 'previous', root / 'current'
        self.bank = self.previous / 'expel_bank'
        self.dest = self.current / 'expel_bank'
        self.upstream = self.current / 'upstream'
        self.origin = self.previous / 'training_source.json'
        save(self.origin, {'public': 'a completed training trajectory'})
        sources = {str(self.origin): _sha(self.origin)}
        for name in ['expel.py', 'human.py', 'alfworld.py']:
            old, new = self.previous / 'upstream/expel' / name, self.upstream / 'expel' / name
            old.parent.mkdir(parents=True, exist_ok=True); new.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(f'# original {name}\n'); new.write_bytes(old.read_bytes())
            sources[str(old)] = _sha(old)
        save(self.bank / 'source_hashes.json', sources)
        state = {'rules': [['Keep all original rules.', 2]], 'test_feedback_used': False,
                 'frozen_before_evaluation': True,
                 'source_hashes_sha256': _sha(self.bank / 'source_hashes.json')}
        state['bank_id'] = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        save(self.bank / 'state.json', state)
        save(self.bank / 'freeze.json', {'state_sha256': _sha(self.bank / 'state.json'),
             'source_hashes_sha256': _sha(self.bank / 'source_hashes.json'), 'test_feedback_used': False})
        save(self.bank / 'generations/000.json', {'raw_response': 'Keep all original rules.'})
        save(self.previous / 'input_hashes.json', {str(self.origin): _sha(self.origin)})
        save(self.previous / 'bank_hashes.json', {str(p): _sha(p) for p in self.bank.rglob('*') if p.is_file()})

    def test_exact_copy_and_zero_new_calls(self):
        result = restore_bank(self.previous, self.dest, self.upstream)
        self.assertEqual(result['copied_files'], 4)
        self.assertEqual(result['new_critic_calls'], 0)
        self.assertEqual(result['new_environment_episodes'], 0)
        self.assertEqual((self.dest / 'state.json').read_bytes(), (self.bank / 'state.json').read_bytes())
        self.assertEqual((self.dest / 'source_hashes.json').read_bytes(), (self.bank / 'source_hashes.json').read_bytes())
        self.assertEqual(verify_reused_bank(self.dest)['bank_id'], result['bank_id'])
        with self.assertRaises(FileExistsError):
            restore_bank(self.previous, self.dest, self.upstream)

    def test_modified_original_source_rejected(self):
        self.origin.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            restore_bank(self.previous, self.dest, self.upstream)
        self.assertFalse(self.dest.exists())

    def test_changed_current_upstream_rejected(self):
        (self.upstream / 'expel/human.py').write_text('different')
        with self.assertRaisesRegex(ValueError, 'upstream ExpeL source differs'):
            restore_bank(self.previous, self.dest, self.upstream)

    def test_extra_bank_file_rejected(self):
        save(self.bank / 'untracked.json', {'unexpected': True})
        with self.assertRaisesRegex(ValueError, 'exactly the complete bank'):
            restore_bank(self.previous, self.dest, self.upstream)

    def test_reused_state_corruption_rejected(self):
        restore_bank(self.previous, self.dest, self.upstream)
        (self.dest / 'state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'state hash mismatch'):
            verify_reused_bank(self.dest)


if __name__ == '__main__':
    unittest.main()
