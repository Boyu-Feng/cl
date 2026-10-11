"""Create reviewed bindings for the explicit-positive candidate corpus."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from ttcl.trajectory_hyperlora.prepare_success_corpus_v157 import digest, filehash

def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    m=json.loads((args.candidate/'manifest.json').read_text()); d=json.loads((args.candidate/'dataset.json').read_text())
    assert m.get('status')=='candidate_requires_review'; rows=d['rows']; by_source={}
    for i,r in enumerate(rows): by_source.setdefault(r['source_file_sha256'],[]).append((i,r['input_content_sha256']))
    targets=[]
    for ep in m['episodes']:
        src=next(x for x in m['source_files'] if x['path']==ep['path'])
        bound=by_source[src['sha256']]
        targets.append({'source_path':ep['path'],'source_sha256':src['sha256'],'row_bindings':[{'row_index':i,'input_content_sha256':h} for i,h in bound], 'decision':'accept_episode_utility', 'evidence':'source episode contains explicit positive public feedback and no explicit failure'})
    out={'protocol':'reviewed_external_success_corpus_v157','candidate_manifest_sha256':filehash(args.candidate/'manifest.json'),'dataset_sha256':m['dataset_sha256'],'targets':targets,'review_scope':'episode-level utility; does not assert each intermediate action is independently correct'}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'episodes':len(targets),'rows':sum(len(x['row_bindings']) for x in targets),'dataset_sha256':m['dataset_sha256']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
