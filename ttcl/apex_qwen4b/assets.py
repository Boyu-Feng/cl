"""Fetch pinned public judge/retrievers/wiki25, never APEx or CLBench labels."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import time
import traceback
from .common import read,save,sha,status,workspace


def fetch(root):
    from huggingface_hub import snapshot_download,hf_hub_download
    w=workspace();spec=read(root/'asset_plan.json');status(root,'assets_downloading')
    def model(item):
        target=w/item['directory']
        snapshot_download(item['repo'],revision=item['revision'],local_dir=str(target),
                          allow_patterns=item['allow_patterns'],max_workers=4)
        save(target/'download_provenance.json',item)
        return {'kind':'model','repo':item['repo'],'directory':str(target)}
    def corpus(item):
        dest=w/spec['wiki']['directory']
        path=Path(hf_hub_download(spec['wiki']['repo'],repo_type='dataset',
            revision=spec['wiki']['revision'],filename=item['path'],local_dir=str(dest)))
        if path.stat().st_size!=item['size'] or sha(path)!=item['lfs']['oid']:
            raise ValueError(f'Corpus checksum failed: {path}')
        return {'kind':'wiki','file':item['path'],'bytes':item['size']}
    done=[];errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(model,x) for x in spec['models']]
        jobs += [pool.submit(corpus,x) for x in spec['wiki']['files']]
        for f in as_completed(jobs):
            try:
                result=f.result();done.append(result);print(result,flush=True)
            except Exception as e:
                errors.append({'error':repr(e),'traceback':traceback.format_exc()})
                print(errors[-1],flush=True)
            save(root/'asset_progress.json',{'completed':done,'errors':errors,'expected':len(jobs),'time':time.time()})
    if errors:
        status(root,'asset_download_failed',errors=errors)
        raise RuntimeError('Asset download failures preserved in asset_progress.json')
    save(root/'assets_ready.json',{'completed':done,'asset_plan_sha256':sha(root/'asset_plan.json'),'time':time.time()})
    status(root,'assets_ready_not_yet_training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    fetch(p.parse_args().root.resolve())
