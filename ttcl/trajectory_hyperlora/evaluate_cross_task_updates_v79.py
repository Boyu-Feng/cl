"""Training-only transfer diagnostic of all v76 updates, before H distillation.

No same-task retries, no test labels, no additional optimization. Future own
success labels are offline training supervision, not information available to
an online actor before that future episode. Loss gains are not task rewards.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash

ROOT = Path('results/trajectory_hyperlora')
TEACHERS = ROOT/'alfworld_support_updates_v76_20261010'
PARENT = ROOT/'alfworld_cold_joint_memory_v63_hybrid_20261010.json'
TARGETS = Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json')
REVIEW = Path('data/annotations/alfworld_cross_task_updates_v79_20261010.json')
OUTPUT = ROOT/'alfworld_cross_task_updates_v79_20261010.json'
BOOT = ROOT/'alfworld_centered_experience_v75b_20261010/bootstrap.pt'


def prepare():
    prior = checked(SimpleNamespace(review=TARGETS))
    parent = json.loads(PARENT.read_text())
    source_review = json.loads(Path('data/annotations/alfworld_support_updates_v76_20261010.json').read_text())
    report = json.loads((TEACHERS/'report.json').read_text())
    audit = json.loads((TEACHERS/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == file_hash(TEACHERS/'report.json')
    assert file_hash(BOOT) == source_review['bootstrap_sha256']
    targets = prior['targets']; pairs, skipped = [], []
    for group, row in zip(source_review['groups'], report['rows'], strict=True):
        i = group['stream_index']; family = parent['rows'][i]['family'] if 'family' in parent['rows'][i] else None
        # Parent rows bind game identities; family is in the scheduled review.
        game_family = lambda game: Path(game).parts[2].split('-')[0]
        family = game_family(group['game'])
        later = [j for j in prior['own_success_indices'] if j > i and
                 game_family(parent['rows'][j]['game']) == family and
                 any(t['stream_index'] == j for t in targets)]
        if not later:
            skipped.append({'source_index': i, 'reason': 'No later same-family own success with reviewed actions in training prefix'})
            continue
        j = later[0]
        assert j < 120 and parent['rows'][j]['game'] != group['game']
        ids = [k for k,t in enumerate(targets) if t['stream_index'] == j]
        # Even spacing caps compute without using losses to choose targets.
        chosen = [ids[k*len(ids)//min(8,len(ids))] for k in range(min(8,len(ids)))]
        artifact = TEACHERS/f'teacher_{i:03d}.pt'
        assert file_hash(artifact) == row['artifact_sha256']
        item = {'source_index': i, 'query_index': j, 'family': family,
                'source_binding': group['input_content_sha256'], 'artifact_sha256': file_hash(artifact),
                'query_targets': chosen, 'query_bindings': [targets[k]['input_content_sha256'] for k in chosen]}
        pairs.append({**item, 'input_content_sha256': digest(item)})
    return {'pairs': pairs, 'skipped': skipped, 'bootstrap_sha256': file_hash(BOOT),
            'parent_sha256': file_hash(PARENT), 'targets_sha256': file_hash(TARGETS),
            'teacher_audit_sha256': file_hash(TEACHERS/'audit.json'),
            'runner_sha256': file_hash(Path(__file__)),
            'scope': 'Offline training-prefix cross-task loss diagnostic; all candidates, no v76 acceptance filtering; zero new optimization steps; no environment success claim'}, targets


def run(device):
    review, targets = prepare()
    assert json.loads(REVIEW.read_text()) == review
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    agent, tok = load_agent(Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'), BOOT.parent/'hyper.pt', device, .6)
    agent.requires_grad_(False); agent.eval(); agent.set_source(None)
    boot = torch.load(BOOT, map_location=device, weights_only=True)
    with torch.no_grad():
        for layer, a in zip(agent.adapters, boot['A'], strict=True): layer.a.copy_(a.to(layer.a))
    a_hash = factors_hash([layer.a for layer in agent.adapters])
    base = boot['B']; cached = {}
    def losses(ids, values):
        for layer,b in zip(agent.adapters,values,strict=True): layer.b = b
        out = []
        with torch.no_grad():
            for k in ids:
                text = tok.apply_chat_template(targets[k]['actor_messages'], tokenize=False, add_generation_prompt=True)
                prefix = tok(text, add_special_tokens=False).input_ids
                out.append(float(target_loss(agent,tok,targets[k],prefix,device)))
        assert all(torch.isfinite(torch.tensor(out)))
        return out
    result = {'complete':False, 'review_sha256':file_hash(REVIEW), 'rows':[], 'failures':[]}
    try:
        for pair in review['pairs']:
            state = torch.load(TEACHERS/f"teacher_{pair['source_index']:03d}.pt", map_location=device, weights_only=True)
            assert state['A_sha256'] == a_hash and state['bootstrap_sha256'] == review['bootstrap_sha256']
            assert state['group_binding'] == pair['source_binding']
            ids = pair['query_targets']; key = tuple(ids)
            if key not in cached: cached[key] = losses(ids,base)
            row = {'binding':pair['input_content_sha256'], 'source_index':pair['source_index'], 'query_index':pair['query_index'], 'base':cached[key]}
            for arm in ('delta','random_delta'):
                row[arm] = losses(ids,[b+d for b,d in zip(base,state[arm],strict=True)])
            mean = lambda x:sum(x)/len(x)
            row['gain'] = mean(row['base'])-mean(row['delta'])
            row['gain_over_random'] = mean(row['random_delta'])-mean(row['delta'])
            row['positive'] = row['gain'] > 1e-6 and row['gain_over_random'] > 1e-6
            result['rows'].append(row)
            result['summary'] = {'pairs':len(result['rows']), 'positive':sum(r['positive'] for r in result['rows'])}
            save(OUTPUT,result); print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
        result['complete'] = True; save(OUTPUT,result)
    except Exception as exc:
        result['failures'].append(repr(exc)); save(OUTPUT,result); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--device',default='cuda:3')
    args = parser.parse_args()
    if args.command == 'prepare':
        if REVIEW.exists(): raise FileExistsError(REVIEW)
        review,_ = prepare(); save(REVIEW,review)
        print(json.dumps({'pairs':len(review['pairs']),'skipped':len(review['skipped'])}))
    else: run(args.device)
