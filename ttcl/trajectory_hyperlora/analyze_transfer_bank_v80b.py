"""Paired success and behavior analysis after the full v80b environment audit."""
import hashlib
import json
from pathlib import Path

ROOT=Path('results/trajectory_hyperlora')
REPORT=ROOT/'alfworld_transfer_bank_v80b_20261010.json'
REVIEW=Path('data/annotations/alfworld_transfer_bank_v80b_20261010.json')
OUTPUT=ROOT/'alfworld_transfer_bank_v80b_comparison_20261010.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def compare(rows,reference):
    seq=lambda ep:[(s['command'],s['observation']) for s in ep['trajectory']]
    return {'n':len(rows),'retrieved_wins':sum(r['arms']['retrieved']['reward'] for r in rows),
        'reference_wins':sum(r['arms'][reference]['reward'] for r in rows),
        'gains':[r['index'] for r in rows if r['arms']['retrieved']['reward']>r['arms'][reference]['reward']],
        'losses':[r['index'] for r in rows if r['arms']['retrieved']['reward']<r['arms'][reference]['reward']],
        'identical_action_feedback_sequences':sum(seq(r['arms']['retrieved'])==seq(r['arms'][reference]) for r in rows)}

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    report=json.loads(REPORT.read_text());review=json.loads(REVIEW.read_text())
    audit_path=REPORT.with_suffix('.audit.json');audit=json.loads(audit_path.read_text())
    assert report['complete'] and not report['failures'] and audit['passed']
    assert audit['report_sha256']==sha(REPORT) and report['review_sha256']==sha(REVIEW)
    assert len(report['rows'])==24 and audit['episodes_replayed']==120
    assert audit['summary']==report['summary']
    groups={}
    for row,target in zip(report['rows'],review['targets'],strict=True):
        assert row['index']==target['index'] and row['game']==target['game']
        assert row['input_content_sha256']==target['input_content_sha256']
        source=review['bank'][target['selected_bank_index']]['source_index']
        groups.setdefault(source,[]).append(row)
    refs=('bootstrap','rotated','random','static')
    result={'bindings':{'report':sha(REPORT),'review':sha(REVIEW),'audit':sha(audit_path)},
        'summary':report['summary'],'comparisons':{a:compare(report['rows'],a) for a in refs},
        'by_selected_source':{str(i):{a:compare(rows,a) for a in refs} for i,rows in groups.items()},
        'limitations':['Exposed development tasks, not a fresh confirmatory test.',
            'Frozen candidate bank, no online learning or hypernetwork generation in this evaluation.',
            'Rotated sources change both direction and magnitude; not a pure semantic intervention.',
            'Bank screening used additional training-query feedback; static LoRA is not an equal-information causal control.']}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparisons']),flush=True)

if __name__=='__main__':main()
