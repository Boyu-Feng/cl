"""Read-only teacher-forced source-content diagnostic on reviewed train targets."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from ttcl.trajectory_hyperlora import alfworld_bootstrap_own_success_v43 as bootstrap
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields, action_fields
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import alignment_pair
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss


def main():
    parser = bootstrap.parser()
    parser.set_defaults(review=Path('data/annotations/alfworld_bootstrap_own_success_v43b_train240_20261010.json'),
        hyper_model=Path('results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43b_hyper_20261010.pt'),
        train_report=Path('results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43b_train_20261010.json'),
        device='cuda:3')
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alfworld_source_action_loss_v56_20261010.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, successes, _, fixed, _, _ = bootstrap.checked(args)
    training = json.loads(args.train_report.read_text())
    assert training['review_sha256'] == file_hash(args.review)
    assert training['hyper_model_sha256'] == file_hash(args.hyper_model)
    assert all(training[k] == v for k, v in fixed.items())
    items = bootstrap.examples(successes)
    # Fixed first/middle/last action of every eligible target, selected without losses.
    selected = []
    for j, k, action in items:
        actions = successes[k]['actions']
        indices = sorted({0, len(actions)//2, len(actions)-1})
        if any(action is actions[i] for i in indices):
            selected.append((j, k, action))
    agent, tokenizer = load_agent(args.model, args.hyper_model, args.device, args.gpu_fraction)
    agent.eval()
    source_cache = {}
    result = {'protocol':'Read-only train-set action likelihood under coherent, shifted-action source and no adapter; not heldout or environment performance',
        'checkpoint_sha256':file_hash(args.hyper_model), 'review_sha256':file_hash(args.review),
        'train_report_sha256':file_hash(args.train_report), 'runner_sha256':file_hash(Path(__file__)),
        'selection':'First, middle and last action per eligible successful target, same source selection as training',
        'selected_actions':len(selected), 'rows':[]}
    for j, k, action in selected:
        if j not in source_cache:
            coherent, shifted = alignment_pair(successes[j]['source_records'], tokenizer)
            fields = [bounded_source_fields(agent, tokenizer, x, args.device) for x in (coherent, shifted)]
            source_cache[j] = (fields, digest(coherent), digest(shifted))
        fields, coherent_hash, shifted_hash = source_cache[j]
        prefix, target = action_fields(agent, tokenizer, action,
            successes[k]['actions'][0]['current_observation'], args.device)
        losses = {}
        with torch.no_grad():
            for name, source in zip(('coherent', 'shifted'), fields, strict=True):
                agent.set_source(source, target_fields=target)
                losses[name] = float(target_loss(agent, tokenizer, action, prefix, args.device))
            agent.set_source(None)
            losses['base'] = float(target_loss(agent, tokenizer, action, prefix, args.device))
        result['rows'].append({'source_index':j,'target_index':k,'family':successes[k]['family'],
            'target_action_sha256':digest(action),'source_records_sha256':coherent_hash,
            'shifted_records_sha256':shifted_hash,'loss':losses})
        if len(result['rows']) % 15 == 0:
            print(json.dumps({'processed':len(result['rows']),'total':len(selected)}),flush=True)
    rows=result['rows']
    result['summary'] = {'n':len(rows),
        'mean_loss':{name:sum(r['loss'][name] for r in rows)/len(rows) for name in ('coherent','shifted','base')},
        'coherent_lower_than_shifted':sum(r['loss']['coherent']<r['loss']['shifted'] for r in rows),
        'coherent_lower_than_base':sum(r['loss']['coherent']<r['loss']['base'] for r in rows),
        'mean_absolute_coherent_shifted_difference':sum(abs(r['loss']['coherent']-r['loss']['shifted']) for r in rows)/len(rows)}
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(result['summary']),flush=True)


if __name__ == '__main__':
    main()
