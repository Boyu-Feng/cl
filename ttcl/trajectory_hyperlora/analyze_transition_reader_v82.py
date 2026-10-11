"""Separate parameter sensitivity, behavior change and verified task benefit."""
import hashlib
import json
import math
from pathlib import Path
ROOT=Path('results/trajectory_hyperlora')
REPORT=ROOT/'alfworld_transition_reader_v82_eval_20261010.json'
REVIEW=Path('data/annotations/alfworld_transition_reader_v82_eval_20261010.json')
OUTPUT=ROOT/'alfworld_transition_reader_v82_comparison_20261010.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    report=json.loads(REPORT.read_text());review=json.loads(REVIEW.read_text());audit_path=REPORT.with_suffix('.audit.json');audit=json.loads(audit_path.read_text())
    assert report['complete'] and not report['failures'] and audit['passed']
    assert report['review_sha256']==audit['review_sha256']==sha(REVIEW)
    assert audit['report_sha256']==sha(REPORT) and audit['episodes_replayed']==120
    assert report['summary']==audit['summary']
    rows=report['rows'];assert len(rows)==24
    seq=lambda ep:[(s['command'],s['observation']) for s in ep['trajectory']]
    comparisons={}
    for arm in ('empty','shifted','static','base'):
        comparisons[arm]={'real_wins':sum(r['arms']['real']['reward'] for r in rows),
            'reference_wins':sum(r['arms'][arm]['reward'] for r in rows),
            'gains':[r['index'] for r in rows if r['arms']['real']['reward']>r['arms'][arm]['reward']],
            'losses':[r['index'] for r in rows if r['arms']['real']['reward']<r['arms'][arm]['reward']],
            'different_sequences':[r['index'] for r in rows if seq(r['arms']['real'])!=seq(r['arms'][arm])]}
    first_different=[];entropies=[];fractions=[]
    for row,target in zip(rows,review['targets'],strict=True):
        assert row['input_content_sha256']==target['input_content_sha256']
        real=row['arms']['real']['parameter_reads'];wrong=row['arms']['shifted']['parameter_reads']
        assert real[0]['query_text_sha256']==wrong[0]['query_text_sha256']
        if real[0]['B_sha256']!=wrong[0]['B_sha256']:first_different.append(row['index'])
        for read in real:
            w=read['attention_weights'];entropies.append(-sum(v*math.log(max(v,1e-30)) for v in w)/math.log(len(w)))
            fractions.append(read['effective_relative_norm'])
    result={'bindings':{'report':sha(REPORT),'review':sha(REVIEW),'audit':sha(audit_path)},
        'summary':report['summary'],'comparisons':comparisons,
        'same_initial_query_different_factors':first_different,
        'normalized_attention_entropy':{'mean':sum(entropies)/len(entropies),'min':min(entropies),'max':max(entropies)},
        'effective_relative_norm':{'mean':sum(fractions)/len(fractions),'min':min(fractions),'max':max(fractions)},
        'scope':'Frozen exposed24 after replay training; equal target labels/steps for H and LoRA, different parameters/encoding cost. Parameter sensitivity and attention concentration are not themselves experience utility.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons),flush=True)

if __name__=='__main__':main()
