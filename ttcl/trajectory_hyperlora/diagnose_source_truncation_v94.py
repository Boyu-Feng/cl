"""Measure whether declared head/tail truncation retains positive feedback."""
from ttcl.trajectory_hyperlora.joint_domains_v93 import ROOT,MODEL,read,write,filehash,digest
from ttcl.trajectory_hyperlora.continual_update_v94 import trajectory_text
from transformers import AutoTokenizer

def main():
    folder=ROOT/'results/trajectory_hyperlora/continual_xland_v94b_20261010'
    data=read(folder/'collection.json');tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rows=[]
    for split in ('train','test'):
        for group in data[split]:
            for ep in group:
                text=trajectory_text(ep)
                encoding=tok(text,add_special_tokens=False,return_offsets_mapping=True)
                ids=encoding['input_ids'];offsets=encoding['offset_mapping'];kept=set(range(len(ids))) if len(ids)<=2048 else set(range(1024))|set(range(len(ids)-1024,len(ids)))
                spans=[];cursor=len(text.splitlines()[0])+1
                for step,line in enumerate(text.splitlines()[1:]):
                    if ep['trajectory'][step]['reward']>0:
                        marker=line.index('; reward=')
                        start=cursor+marker;end=cursor+len(line)
                        reward_tokens={i for i,(a,b) in enumerate(offsets) if a<end and b>start}
                        full_step={i for i,(a,b) in enumerate(offsets) if a<cursor+len(line) and b>cursor}
                        spans.append(dict(step=step,reward=ep['trajectory'][step]['reward'],reward_feedback_retained=reward_tokens<=kept,whole_transition_retained=full_step<=kept))
                    cursor+=len(line)+1
                rows.append(dict(split=split,episode_sha256=ep['content_sha256'],tokens=len(ids),truncated=len(ids)>2048,positive=ep['positive'],reward_events=spans))
    summary={}
    for split in ('train','test'):
        rr=[r for r in rows if r['split']==split];events=[e for r in rr for e in r['reward_events']]
        summary[split]=dict(episodes=len(rr),truncated=sum(r['truncated'] for r in rr),positive=sum(r['positive'] for r in rr),reward_events=len(events),feedback_retained=sum(e['reward_feedback_retained'] for e in events),whole_transitions_retained=sum(e['whole_transition_retained'] for e in events),max_tokens=max(r['tokens'] for r in rr))
    write(folder/'source_truncation_diagnostic.json',dict(collection_sha256=filehash(folder/'collection.json'),runner_sha256=filehash(__file__),rows=rows,summary=summary,scope='Token-offset accounting only, no model updates or accuracy claim; header return remains visible even when transition is dropped'))
    print(summary)

if __name__=='__main__':main()
