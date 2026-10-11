"""Replay every episode and verify the causal parameter/residual state chain.

Checks recorded anchor-loss decisions and additive arithmetic; does not claim
to independently recompute the neural anchor losses on CPU.
"""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    parser, paths, checked, anchor_items,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import action_rows
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import effective_norm
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def parameter_hash(values):
    h = hashlib.sha256()
    for name, value in values.items():
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def audit(args):
    _, fixed, targets = checked(args)
    report = json.loads(args.output.read_text())
    annotations = json.loads(args.annotations.read_text())
    assert report.get('complete') and not report['failures']
    assert all(report[k] == v for k, v in fixed.items())
    assert report['review_sha256'] == annotations['eval_review_sha256'] == file_hash(args.review)
    assert len(report['rows']) == len(targets) == args.limit
    pool, state, previous_h, coordinate_hash, norm_agent = [], None, report['initial_parameters_sha256'], None, None
    reference, frozen_a = None, None
    for i, (row, target) in enumerate(zip(report['rows'], targets, strict=True)):
        assert row['index'] == i and row['game'] == target['game']
        assert row['target_input_content_sha256'] == target['input_content_sha256']
        assert row['memory_before'] == len(pool)
        assert row['H_before_sha256'] == previous_h
        assert row['R_before_sha256'] == factors_hash(state)
        same = sum(x['family'] == target['family'] for x in pool)
        j = source_for(target, pool) if same else None
        assert row['source_stream_index'] == (pool[j]['stream_index'] if j is not None else None)
        assert row['source_episode_sha256'] == (pool[j]['episode_sha256'] if j is not None else None)
        assert row['adapter_active'] == (len(pool) >= 8 and same >= 2)
        ep = row['episode']
        assert ep['status'] == 'complete' and ep['steps'] <= 30
        assert digest(ep['initial_observation']) == target['initial_observation_sha256']
        replay(args.data_root/target['game'], ep)
        expected_steps = 0
        if ep['reward'] == 1:
            actions = action_rows(args.data_root/target['game'], ep)
            content = {'stream_index': i, 'game': target['game'], 'family': target['family'],
                'game_sha256': target['game_sha256'], 'episode_sha256': digest(ep),
                'source_records_sha256': digest(records_from_episode(ep)),
                'actions_sha256': digest(actions),
                'runner_sha256': fixed['code_sha256']['alfworld_online_own_trajectory_v52b.py']}
            annotation = annotations['annotations'][len(pool)]
            assert all(annotation[k] == v for k, v in content.items())
            assert annotation['input_content_sha256'] == digest(content) == row['annotation_input_content_sha256']
            if len(pool) == 7:
                expected_steps = 300
                assert report['bootstrap']['index'] == i
            elif len(pool) >= 8 and j is not None and args.mode in ('hyper', 'hybrid'):
                expected_steps = 20
                assert row['update']['R_used_sha256'] == row['R_before_sha256']
            pool.append({'stream_index': i, 'game': target['game'], 'family': target['family'],
                'episode_sha256': digest(ep), 'source_records': records_from_episode(ep),
                'actions': actions, 'annotation_input_content_sha256': digest(content)})
        assert row['update']['steps'] == expected_steps
        write_expected = ep['reward'] == 0 and row['adapter_active'] and args.mode in ('hybrid', 'residual')
        assert (row['write'] is not None) == write_expected
        if write_expected:
            write = row['write']
            anchors = []
            for sj, ti, action in anchor_items(pool):
                anchors.append({'source_stream_index': pool[sj]['stream_index'],
                    'target_stream_index': pool[ti]['stream_index'],
                    'source_episode_sha256': pool[sj]['episode_sha256'],
                    'target_annotation_sha256': pool[ti]['annotation_input_content_sha256'],
                    'action_sha256': digest(action)})
            assert write['anchors'] == anchors
            assert all(max(x['source_stream_index'], x['target_stream_index']) < i for x in anchors)
            assert file_hash(Path(write['delta_file'])) == write['delta_file_sha256']
            delta = torch.load(write['delta_file'], map_location='cpu', weights_only=True)
            norm_agent = SimpleNamespace(adapters=[SimpleNamespace(a=a, rank=4,
                base=SimpleNamespace(out_features=b.shape[1])) for a,b in zip(frozen_a,delta,strict=True)])
            assert factors_hash(delta) == write['delta_sha256']
            content = {'stream_index': i, 'game_sha256': target['game_sha256'],
                'episode_sha256': digest(ep), 'failed_records_sha256': digest(records_from_episode(ep)),
                'own_success_source_episode_sha256': pool[j]['episode_sha256'],
                'H_sha256': previous_h, 'anchors': anchors, 'delta_sha256': factors_hash(delta)}
            assert all(write[k] == v for k, v in content.items())
            assert write['input_content_sha256'] == digest(content)
            trace = write['guard']
            assert trace['reference_norm'] == reference
            assert trace['tolerance'] == .01 and trace['max_relative'] == .25
            old_state = [torch.zeros_like(x) for x in delta] if state is None else state
            accepted = False
            for eta, trial in zip(fixed['write_steps'], trace['trials'], strict=False):
                assert trial['step'] == eta
                proposal = [a+eta*b for a, b in zip(old_state, delta, strict=True)]
                magnitude = effective_norm(norm_agent, proposal)
                scale = min(1., .25*reference/magnitude) if magnitude else 1.
                assert abs(scale-trial['scale']) < 1e-6
                # CPU/GPU reductions can round differently; verify the scale,
                # then reproduce the exact CPU tensor arithmetic used to store R.
                proposal = [x*trial['scale'] for x in proposal]
                change = effective_norm(norm_agent, [a-b for a,b in zip(proposal,old_state,strict=True)])
                if 'skipped' in trial:
                    assert change <= 1.01e-8*reference
                    continue
                assert len(trial['losses']) == len(trace['baseline_losses']) == len(anchors)
                assert all(math.isfinite(x) and x >= 0 for x in trial['losses']+trace['baseline_losses'])
                accepted = all(a <= b+.01 for a,b in zip(trial['losses'],trace['baseline_losses'],strict=True))
                assert accepted == trial['accepted']
                if accepted:
                    assert trial is trace['trials'][-1] and trace['accepted_step'] == eta
                    state = proposal
                    break
            assert accepted == trace['changed']
            if not accepted:
                assert trace['accepted_step'] == 0. and len(trace['trials']) == 4
        assert row['R_after_sha256'] == factors_hash(state)
        if expected_steps == 0:
            assert row['H_after_sha256'] == previous_h
        changed = previous_h != row['H_after_sha256'] or row['R_before_sha256'] != row['R_after_sha256']
        assert ('snapshot' in row) == changed
        if 'snapshot' in row:
            snap = row['snapshot']
            assert file_hash(Path(snap['path'])) == snap['sha256']
            value = torch.load(snap['path'], map_location='cpu', weights_only=True)
            assert parameter_hash(value['trainable_state']) == row['H_after_sha256']
            assert factors_hash(value['R']) == factors_hash(state)
            ahash = factors_hash(value['A'])
            if coordinate_hash is None:
                coordinate_hash = ahash
                assert coordinate_hash == report['bootstrap']['frozen_A_sha256']
                reference = value['reference_norm']
                assert reference == report['bootstrap']['reference_norm'] > 0
                frozen_a = value['A']
            assert ahash == coordinate_hash and value['reference_norm'] == reference
        previous_h = row['H_after_sha256']
    final = report['final_snapshot']
    assert file_hash(Path(final['path'])) == final['sha256']
    value = torch.load(final['path'], map_location='cpu', weights_only=True)
    assert parameter_hash(value['trainable_state']) == previous_h
    assert factors_hash(value['R']) == factors_hash(state)
    assert len(annotations['annotations']) == len(pool)
    if coordinate_hash is not None:
        assert factors_hash(value['A']) == coordinate_hash
        assert value['reference_norm'] == reference
    assert report['summary'] == {'n': len(targets),
        'wins': sum(x['episode']['reward'] for x in report['rows']),
        'own_successes': len(pool), 'training_steps': sum(x['update']['steps'] for x in report['rows']),
        'accepted_residual_writes': sum(bool(x['write'] and x['write']['guard']['changed']) for x in report['rows'])}
    result = {'passed': True, 'episodes_replayed': len(targets), 'summary': report['summary'],
        'report_sha256': file_hash(args.output), 'audit_sha256': file_hash(Path(__file__)),
        'neural_anchor_losses_recomputed': False}
    save(args.output.with_suffix('.audit.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    audit(paths(parser().parse_args()))
