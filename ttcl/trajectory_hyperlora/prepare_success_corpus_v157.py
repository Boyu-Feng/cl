"""Prepare a candidate CLBench corpus from episodes with explicit positive outcome.

This is an audit-bound candidate corpus: the manifest binds every source file and
input, and excludes an episode if it contains an explicit failure. It does not
invent per-action correctness labels.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=('20261002_v15_poker_303_first30','20261002_v15_database_303_full30','20261002_v15_database_404_full30')

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(args):
    if args.output.exists(): raise RuntimeError('refusing overwrite')
    rows=[]; sources=[]; counts=Counter(); episodes=[]
    for run in RUNS:
        root=ROOT/'results/memrl_grounded_evidence'/run/'typed_grounded'
        for ep in sorted(root.glob('episode_*')):
            path=ep/'public_trajectory.json'
            if not path.exists(): continue
            traj=json.loads(path.read_text()); text=' '.join(str(x.get('public_feedback','')).upper() for x in traj)
            positive=('WON' in text or 'CORRECT' in text)
            failure=('LOST' in text or 'INCORRECT' in text)
            if not positive or failure: continue
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':filehash(path),'steps':len(traj)})
            episodes.append({'path':str(path.relative_to(ROOT)),'outcome':'explicit_positive_without_explicit_failure'})
            prefix=[]
            for step in traj:
                query=str(step['query']); action=json.dumps(step['action'],ensure_ascii=False,sort_keys=True)
                source='Causal trajectory prefix:\n'+'\n'.join(f"step={x['step']}\nobservation={x['query']}\naction={json.dumps(x['action'],ensure_ascii=False,sort_keys=True)}\nfeedback={x.get('public_feedback','')}" for x in prefix) if prefix else 'Causal trajectory prefix is empty.'
                row={'messages':[{'role':'user','content':query}],'query':query,'source':source,'answer':action,
                     'source_file_sha256':filehash(path),'input_content_sha256':digest({'query':query,'source':source})}
                row['generic_row_sha256']=digest(row); rows.append(row);prefix.append(step);counts[run]+=1
    if not rows: raise RuntimeError('no positive episodes')
    rng=random.Random(157); order=list(range(len(rows)));rng.shuffle(order);schedule=[order[i%len(order)] for i in range(max(600,len(rows)))]
    data={'rows':rows,'schedule':schedule}; manifest={'protocol':'external_success_candidate_v157','source_files':sources,'episodes':episodes,'rows':len(rows),'schedule_updates':len(schedule),'schedule_sha256':digest(schedule),'dataset_sha256':digest(data),'target_datasets_forbidden':['xland','alfworld','hle'],'encoder_fields':['source','query'],'forbidden_fields':['domain','benchmark','task_id','answer'],'status':'candidate_requires_review'}
    args.output.mkdir(parents=True);(args.output/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));(args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({'rows':len(rows),'counts':counts,'episodes':len(episodes),'status':manifest['status']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
