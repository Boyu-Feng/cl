"""Independently reload v76 teacher factors and recompute later-action losses."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.fit_support_updates_v76 import prepare
from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import effective_norm
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:3')
    args = p.parse_args()
    args.parent_review = Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json')
    args.parent = Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json')
    args.bootstrap = Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_20261010/bootstrap.pt')
    args.checkpoint = args.bootstrap.parent/'hyper.pt'
    args.model = Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
    review_path = Path('data/annotations/alfworld_support_updates_v76_20261010.json')
    root = Path('results/trajectory_hyperlora/alfworld_support_updates_v76_20261010')
    output = root/'audit.json'
    if output.exists(): raise FileExistsError(output)
    review = json.loads(review_path.read_text())
    assert review == prepare(args)
    report = json.loads((root/'report.json').read_text())
    assert report['complete'] and not report['failures']
    assert report['review_sha256'] == file_hash(review_path)
    assert report['summary']['n'] == len(review['groups']) == len(report['rows'])
    targets = checked(SimpleNamespace(review=args.parent_review))['targets']
    bootstrap = torch.load(args.bootstrap, map_location='cpu', weights_only=True)
    agent, tok = load_agent(args.model, args.checkpoint, args.device, .6)
    agent.requires_grad_(False); agent.eval(); agent.set_source(None)
    with torch.no_grad():
        for layer, a in zip(agent.adapters, bootstrap['A'], strict=True): layer.a.copy_(a.to(layer.a))
    assert factors_hash([x.a for x in agent.adapters]) == report['A_sha256']
    shared = [b.to(args.device) for b in bootstrap['B']]
    assert factors_hash(shared) == report['shared_B_sha256']
    reference = effective_norm(agent, shared)
    assert math.isclose(reference, report['reference_effective_norm'], rel_tol=1e-6)
    verified = []
    for group, row in zip(review['groups'], report['rows'], strict=True):
        index = group['stream_index']
        assert row['stream_index'] == index and row['group_binding'] == group['input_content_sha256']
        assert row['training_steps'] == len(group['update_order']) == 20
        artifact = root/f'teacher_{index:03d}.pt'
        assert file_hash(artifact) == row['artifact_sha256']
        state = torch.load(artifact, map_location=args.device, weights_only=True)
        assert state['bootstrap_sha256'] == review['bootstrap_sha256']
        assert state['A_sha256'] == report['A_sha256'] and state['group_binding'] == row['group_binding']
        dn = effective_norm(agent, state['delta'])
        assert math.isclose(dn, row['delta_effective_norm'], rel_tol=1e-5, abs_tol=1e-7)
        assert dn <= review['max_relative']*reference*(1+1e-5)
        assert math.isclose(effective_norm(agent, state['random_delta']), dn, rel_tol=1e-5, abs_tol=1e-7)
        recomputed = {}
        for arm, values in [('query_before', shared),
                            ('query_after', [b+d for b,d in zip(shared,state['delta'],strict=True)]),
                            ('query_random', [b+d for b,d in zip(shared,state['random_delta'],strict=True)])]:
            for layer, b in zip(agent.adapters, values, strict=True): layer.b = b
            losses = []
            with torch.no_grad():
                for k in group['query_indices']:
                    text = tok.apply_chat_template(targets[k]['actor_messages'], tokenize=False, add_generation_prompt=True)
                    prefix = tok(text, add_special_tokens=False).input_ids
                    losses.append(float(target_loss(agent, tok, targets[k], prefix, args.device)))
            assert len(losses) == len(row[arm])
            assert all(math.isfinite(x) and math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-5)
                       for x,y in zip(losses,row[arm],strict=True)), (index,arm,losses,row[arm])
            recomputed[arm] = sum(losses)/len(losses)
        gain = recomputed['query_before']-recomputed['query_after']
        random_gain = recomputed['query_random']-recomputed['query_after']
        assert math.isclose(gain,row['query_gain'],rel_tol=1e-4,abs_tol=1e-5)
        assert math.isclose(random_gain,row['query_gain_over_random'],rel_tol=1e-4,abs_tol=1e-5)
        accepted = gain > review['utility_tolerance'] and random_gain > review['utility_tolerance']
        assert accepted == row['accepted'], (index,'acceptance changed on independent reload')
        verified.append({'index':index,'accepted':accepted,'query_gain':gain,'query_gain_over_random':random_gain})
        print(json.dumps(verified[-1]), flush=True)
    assert report['summary']['accepted'] == sum(r['accepted'] for r in verified)
    assert report['summary']['training_steps'] == sum(r['training_steps'] for r in report['rows']) == review['budget']['teacher_steps']
    save(output, {'passed': True, 'report_sha256': file_hash(root/'report.json'),
                  'review_sha256': file_hash(review_path), 'runner_sha256': file_hash(Path(__file__)),
                  'summary':report['summary'],'recomputed_rows':verified,
                  'scope':'Source/input bindings, frozen A/common B, factor norms, independent neural query-loss recomputation. No optimizer replay or environment-success validation.'})


if __name__ == '__main__': main()
