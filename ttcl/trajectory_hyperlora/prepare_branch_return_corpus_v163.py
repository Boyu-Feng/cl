"""Convert same-state branch returns into verified local action pairs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def main(args):
 if args.output.exists():raise RuntimeError('refusing overwrite')
 raw=json.loads(args.input.read_text());rows=[]
 for row in raw['rows']:
  branches=row['branches']; best=max(b['reward'] for b in branches); worst=min(b['reward'] for b in branches)
  pos=next(b for b in branches if b['reward']==best);neg=next(b for b in branches if b['reward']==worst)
  if pos['action']==neg['action'] and pos['amount']==neg['amount']:continue
  prefix=row['prefix']; source=(f"initial_observation=\n{prefix['initial_query']}\n"
      f"warm_action={prefix['warm_action']}\nwarm_amount={prefix['warm_amount']}\n"
      f"warm_feedback={prefix['warm_feedback']}")
  q=prefix['branch_query']; ans=json.dumps({'action':pos['action'],'amount':pos['amount']},sort_keys=True)
  negans=json.dumps({'action':neg['action'],'amount':neg['amount']},sort_keys=True)
  r={'messages':[{'role':'user','content':q}],'query':q,'source':source,'answer':ans,'negative':negans,
     'positive_return':best,'negative_return':worst,'state_id':row['state_id'],'prefix_sha256':row['prefix_sha256'],
     'input_content_sha256':digest({'query':q,'source':source})}
  r['generic_row_sha256']=digest(r);rows.append(r)
 schedule=[i%len(rows) for i in range(max(600,len(rows)))]
 data={'rows':rows,'schedule':schedule};manifest={'protocol':'poker_branch_return_corpus_v163','source_artifact_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'rows':len(rows),'schedule_updates':len(schedule),'dataset_sha256':digest(data),'supervision':'same-state branch return; fixed continuation policy','status':'reviewed_branch_return','target_datasets_forbidden':['xland','alfworld','hle']}
 args.output.mkdir(parents=True);(args.output/'dataset.json').write_text(json.dumps(data,indent=2));(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'rows':len(rows),'status':manifest['status']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
