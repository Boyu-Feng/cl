"""Paired strict-option first-attempt H versus base comparison, all64 tasks."""
import hashlib
import json
from pathlib import Path

ROOT = Path('results/trajectory_hyperlora')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = ROOT/'hle_hyper_first_v91_comparison_20261010.json'
    assert not output.exists()
    rp = ROOT/'hle_hyper_first_v91_responses_20261010.json'
    sp = ROOT/'hle_hyper_first_v91_scores_20261010.json'
    bp = ROOT/'hle_base_scores_20261010.json'
    audit = json.loads(rp.with_suffix('.audit.json').read_text())
    scores, base = json.loads(sp.read_text()), json.loads(bp.read_text())
    assert audit['passed'] and audit['responses_sha256'] == scores['responses_sha256'] == sha(rp)
    assert scores['scorer_sha256'] == base['scorer_sha256']
    assert scores['grading_keys_sha256'] == base['grading_keys_sha256']
    new = [r for r in scores['rows'] if r['partition'] == 'evaluation']
    old = [r for r in base['rows'] if r['partition'] == 'evaluation']
    assert len(new) == len(old) == 64
    pairs = list(zip(new, old, strict=True))
    assert all(a['id'] == b['id'] and a['input_content_sha256'] == b['input_content_sha256'] for a,b in pairs)
    result = {'n':64, 'hyper_first_wins':sum(a['reward'] for a,b in pairs),
        'base_first_wins':sum(b['reward'] for a,b in pairs),
        'gains':[a['id'] for a,b in pairs if a['reward'] > b['reward']],
        'losses':[a['id'] for a,b in pairs if a['reward'] < b['reward']],
        'bindings':{'hyper_scores':sha(sp), 'base_scores':sha(bp), 'hyper_audit':sha(rp.with_suffix('.audit.json'))},
        'scope':'Frozen ALFWorld H with64 inherited disjoint source attempts; all64 evaluation MCQs, strict parser; no HLE training or full official HLE score'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
