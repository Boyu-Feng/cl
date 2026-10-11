"""Collect reproducible same-state action branches from CLBench poker.

This writes only a source manifest/results artifact. It does not train a model
or convert episode outcomes into per-action labels outside the exact branch.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'current_work/continual-learning-bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from src.interface import Response
from src.tasks.exploitable_poker.task import Poker, PokerAction


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def new_task(seed):
    t = Poker(num_instances=1, seed=seed, opponent_policy='calling_station')
    t.build_canonical_run_state(); t.select_run_instances([0])
    return t, t.build_current_query()


def response(action, amount=None):
    return Response(action=PokerAction(thinking='fixed branch probe', action=action, amount=amount))


def legal_amount(q, action):
    return q.metadata['min_raise'] if action == 'RAISE' else None


def replay_prefix(seed, warm_action, warm_amount):
    t, q = new_task(seed)
    r = t.step(response(warm_action, warm_amount))
    return t, r


def finish(t, result):
    while not result.done:
        q = result.next_query
        legal = q.metadata['legal_actions']
        action = 'CHECK' if 'CHECK' in legal else 'CALL' if 'CALL' in legal else 'FOLD'
        result = t.step(response(action, legal_amount(q, action)))
    return result


def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    rows=[]; seeds=[]
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        probe, q0 = new_task(seed)
        legal0 = q0.metadata['legal_actions']
        warm = 'CHECK' if 'CHECK' in legal0 else 'CALL' if 'CALL' in legal0 else 'RAISE'
        warm_amount = legal_amount(q0, warm)
        warm_result = probe.step(response(warm, warm_amount))
        if warm_result.done or warm_result.next_query is None: continue
        q1 = warm_result.next_query
        legal1 = q1.metadata['legal_actions']
        if len(legal1) < 2: continue
        branch_data=[]
        for action in legal1:
            t, result = replay_prefix(seed, warm, warm_amount)
            q = result.next_query
            amount = legal_amount(q, action)
            result = finish(t, t.step(response(action, amount)))
            outcome = result.instance_outcome
            branch_data.append({'action': action, 'amount': amount,
                                'reward': outcome.reward if outcome else None,
                                'success': outcome.success if outcome else None,
                                'terminal_feedback': result.observation.content})
        prefix={'seed':seed,'initial_query':q0.prompt,'warm_action':warm,'warm_amount':warm_amount,
                'warm_feedback':warm_result.observation.content,'branch_query':q1.prompt,
                'legal_actions':legal1}
        rows.append({'prefix':prefix,'prefix_sha256':digest(prefix),'branches':branch_data,
                     'state_id':f'poker:default:seed{seed}:hand1:after_warm'})
        seeds.append(seed)
    result={'protocol':'poker_same_state_branch_returns_v162','task':'exploitable_poker',
            'opponent_policy':'calling_station','seed_start':args.seed_start,
            'num_seeds_requested':args.num_seeds,'seeds_with_branches':seeds,
            'rows':rows,'limitations':'Continuation after the branched action is a fixed policy; this is a local counterfactual return, not a complete optimal Q value.'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'seeds':len(seeds),'branches':sum(len(x['branches']) for x in rows)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-start',type=int,default=42);p.add_argument('--num-seeds',type=int,default=100);main(p.parse_args())
