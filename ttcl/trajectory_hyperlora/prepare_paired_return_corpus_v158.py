"""Build unreviewed outcome-associated candidates; query equality is not state equality."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def episode(path):
 d=json.loads(path.read_text());t=' '.join(str(x.get('public_feedback','')).upper() for x in d)
 status='success' if ('WON' in t or 'CORRECT' in t) and not ('LOST' in t or 'INCORRECT' in t) else 'failure' if ('LOST' in t or 'INCORRECT' in t) else None
 return d,status
def main(args):
 if args.output.exists():raise RuntimeError('refusing overwrite')
 grouped=defaultdict(list)
 for p in Path(ROOT/'results/memrl_grounded_evidence').glob('20261002_v15_*/*/episode_*/public_trajectory.json'):
  if 'typed_grounded' not in str(p):continue
  d,s=episode(p)
  if s:grouped[hashlib.sha256(str(d[0]['query']).encode()).hexdigest()].append((s,p,d))
 rows=[]; source_files={}; pair_count=0
 for vals in grouped.values():
  for ps,pp,pd in vals:
   if ps!='success':continue
   for fs,fp,fd in vals:
    if fs!='failure':continue
    neg={str(x['query']):x for x in fd}
    prefix=[]
    for step in pd:
     q=str(step['query'])
     if q not in neg:prefix.append(step);continue
     pos=json.dumps(step['action'],ensure_ascii=False,sort_keys=True);n=json.dumps(neg[q]['action'],ensure_ascii=False,sort_keys=True)
     if pos==n:prefix.append(step);continue
     src='Causal trajectory prefix:\n'+'\n'.join(f"step={x['step']}\nobservation={x['query']}\naction={json.dumps(x['action'],ensure_ascii=False,sort_keys=True)}\nfeedback={x.get('public_feedback','')}" for x in prefix) if prefix else 'Causal trajectory prefix is empty.'
     row={'messages':[{'role':'user','content':q}],'query':q,'source':src,'answer':pos,'negative':n,'positive_outcome':'explicit_success','negative_outcome':'explicit_failure','positive_source_sha256':filehash(pp),'negative_source_sha256':filehash(fp)}
     row['input_content_sha256']=digest({'query':q,'source':src});row['generic_row_sha256']=digest(row);rows.append(row);prefix.append(step);pair_count+=1
     source_files[str(pp.relative_to(ROOT))]=filehash(pp);source_files[str(fp.relative_to(ROOT))]=filehash(fp)
 if not rows:raise RuntimeError('no exact-query pairs')
 rng=random.Random(158);order=list(range(len(rows)));rng.shuffle(order);schedule=[order[i%len(order)] for i in range(600)]
 data={'rows':rows,'schedule':schedule};manifest={'protocol':'external_paired_return_v158','source_files':[{'path':p,'sha256':h} for p,h in sorted(source_files.items())],'rows':len(rows),'pairs':pair_count,'schedule_updates':len(schedule),'schedule_sha256':digest(schedule),'dataset_sha256':digest(data),'target_datasets_forbidden':['xland','alfworld','hle'],'supervision':'exact-query action pairs with official episode success/failure markers','status':'candidate_requires_causal_review', 'limitations':'Episode outcomes are observational; histories and executable-action differences require review before training'}
 args.output.mkdir(parents=True);(args.output/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));(args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({'rows':len(rows),'source_files':len(source_files),'pairs':pair_count,'status':manifest['status']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
