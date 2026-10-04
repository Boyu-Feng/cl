"""Audit when ALFWorld exposes goal-object evidence in existing train runs.

This reads first-attempt public observations only. A name appearing in an
observation is not proof that the object was found or carried; the audit tests
when a premise could first be checked, not whether a memory was useful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .annotate_claim_applicability_pilot import REVIEWS
from .probe_goal_slot_credit import _slots


def audit(opportunity_path: Path) -> dict:
    source = read(opportunity_path)
    if source['schema'] != 'alf_credit_q_downstream_opportunity_v1':
        raise ValueError('Wrong audited coalition source')
    review_root = Path(__file__).parents[2] / 'data' / 'annotations'
    rows = []
    for output in source['outputs']:
        coalition = Path(output['output'])
        origin = Path(read(coalition / 'design.json')['origin'])
        kind, review_name = REVIEWS[coalition.name]
        review_path = review_root / review_name
        reviews = read(review_path)['targets']
        plan = read(origin / 'plan.json')
        for case in output['cases']:
            target = origin / 'runs' / case['case']
            official = read(target / 'row.json')
            episode_path = target / 'attempt_1' / 'episode.json'
            episode = read(episode_path)
            if (official['status'] != 'complete' or
                    official['input_sha256'] != case['input_sha256'] or
                    official['game'] != episode['game'] or
                    sha(Path(plan['alf']['data_root']) / official['game']) !=
                    case['input_sha256'] or
                    episode['status'] != 'complete' or
                    episode['reward'] != official['first_attempt'] or
                    episode['steps'] != len(episode['trajectory'])):
                raise ValueError('Official first attempt changed')
            initial = episode['initial_observation']
            if 'Your task is to: ' not in initial:
                raise ValueError('Missing public goal')
            room, rest = initial.split('Your task is to: ', 1)
            task = rest.split('\n', 1)[0].strip()
            matched = [review for review in reviews
                       if review['origin'] == kind and
                       review['case'] == case['case'] and
                       review['input_sha256'] == case['input_sha256'] and
                       review['public_task'] == task and review['reviewed']]
            if len(matched) != 1:
                raise ValueError('Missing reviewed content-bound public target')
            obj, destination, operation = _slots(task)
            object_pattern = re.compile(r'\b' + re.escape(obj) + r'\b', re.I)
            destination_pattern = re.compile(r'\b' + re.escape(destination) +
                                             r'\b', re.I)
            first_mention = next((index for index, step in
                                  enumerate(episode['trajectory'], 1)
                                  if object_pattern.search(step['observation'])),
                                 None)
            rows.append(dict(source_output=str(coalition), case=case['case'],
                             input_sha256=case['input_sha256'],
                             review_sha256=sha(review_path),
                             first_episode_sha256=sha(episode_path),
                             initial_observation_sha256=hashlib.sha256(
                                 initial.encode()).hexdigest(),
                             public_task=task, goal_object=obj,
                             goal_destination=destination,
                             goal_operation=operation,
                             initial_object_mentioned=bool(object_pattern.search(room)),
                             initial_destination_mentioned=bool(
                                 destination_pattern.search(room)),
                             first_feedback_object_mention_step=first_mention,
                             first_attempt_reward=episode['reward'],
                             first_attempt_steps=episode['steps']))
    if len(rows) != 28 or len({r['input_sha256'] for r in rows}) != 28:
        raise ValueError('Expected 28 distinct reviewed training inputs')
    mention_steps = [r['first_feedback_object_mention_step'] for r in rows
                     if r['first_feedback_object_mention_step'] is not None]
    return dict(schema='alf_initial_evidence_timing_audit_v1',
                script_sha256=sha(Path(__file__)),
                opportunity_sha256=sha(opportunity_path), rows=rows,
                totals=dict(inputs=len(rows),
                            initial_object_mentioned=sum(r['initial_object_mentioned']
                                                         for r in rows),
                            initial_destination_mentioned=sum(
                                r['initial_destination_mentioned'] for r in rows),
                            later_feedback_object_mentioned=len(mention_steps),
                            never_mentioned_in_first_attempt=(len(rows) -
                                                               len(mention_steps)),
                            first_feedback_mention_median_step=statistics.median(
                                mention_steps),
                            first_attempt_successes=sum(r['first_attempt_reward']
                                                        for r in rows)),
                caveat='Goal-object string mention is a weak availability marker, not an object-found label. Initial room excludes the separate task goal; task and reviewed content are hash-bound.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opportunities', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.opportunities.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
