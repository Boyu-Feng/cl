from __future__ import annotations
import json
from pathlib import Path
import pyarrow.parquet as pq
from .common import binding,read,save,sha,workspace


def rows(path):
    columns=['data_id','data_source','prompt','image_caption','reward_model']
    for r in pq.read_table(path,columns=columns).to_pylist():
        question=r['prompt'][-1]['content'].strip();caption=(r['image_caption'] or '').strip()
        target=str(r['reward_model']['ground_truth']).strip()
        if not question or not target:raise ValueError('Empty question or offline reference answer')
        result={'id':r['data_id'],'source':r['data_source'],'question':question,'caption':caption,'target':target}
        result['input_binding']=binding({'question':question,'caption':caption})
        result['supervision_binding']=binding(result)
        yield result


def prepare(root):
    w=workspace();source=w/'data/apex/MIA/Train';spec=read(w/'config/apex-reproduction.json')['dataset']
    for name in ['Train/Executor/fvqa_train.parquet','Train/Executor/fvqa_test.parquet','Train/Planner/fvqa_matpo_train_planner.parquet']:
        item=next(x for x in spec['files'] if x['filename']==name)
        if sha(w/spec['directory']/name)!=item['sha256']:raise ValueError('Official data source changed: '+name)
    train=list(rows(source/'Executor/fvqa_train.parquet'))
    val=list(rows(source/'Executor/fvqa_test.parquet'))
    planner=list(rows(source/'Planner/fvqa_matpo_train_planner.parquet'))
    heldout={r['input_binding'] for r in val}
    excluded=[]
    def clean(seq,label):
        result=[];seen={}
        targets={}
        for r in seq:targets.setdefault(r['input_binding'],set()).add(r['target'])
        conflicts={k for k,values in targets.items() if len(values)>1}
        for r in seq:
            key=r['input_binding']
            if key in conflicts:
                excluded.append({'split':label,'id':r['id'],'input_binding':key,'reason':'conflicting_reference_answers_all_duplicates_quarantined'});continue
            if key in heldout:
                excluded.append({'split':label,'id':r['id'],'reason':'official_validation_input_overlap'});continue
            if key in seen:
                if seen[key]!=r['target']:raise ValueError('Conflicting training labels for identical bound input')
                excluded.append({'split':label,'id':r['id'],'reason':'duplicate_input'});continue
            seen[key]=r['target'];result.append(r)
        return result
    train=clean(train,'executor');planner=clean(planner,'planner')
    directory=w/'data/apex_qwen4b'/root.name;directory.mkdir(parents=True,exist_ok=False)
    for name,seq in [('executor_train',train),('planner_train',planner),('executor_validation',val)]:
        with (directory/(name+'.jsonl')).open('w') as f:
            for r in seq:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save(directory/'exclusions.json',excluded)
    info={'directory':str(directory),'executor_train':len(train),'planner_train':len(planner),
        'executor_validation':len(val),'excluded':excluded,'clbench_used_for_offline_training':False,
        'modality_adaptation':'Official image captions replace pixels; original text-only rows unchanged.',
        'files':{str(p):sha(p) for p in directory.iterdir() if p.is_file()}}
    save(root/'dataset.json',info);return info


def load_rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]
