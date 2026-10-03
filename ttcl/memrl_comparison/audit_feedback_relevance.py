"""CPU-only audit of task-to-feedback relevance on a frozen online cell.

This is post-hoc diagnosis, not a reward-tuned readout gate. It compares the
current public question with actual source action/feedback, while v30 ranked
groups using prior questions. No actor or writer is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from ttcl.icl_mem0_comparison.protocol import normalize_prompt
from ttcl.icl_mem0_comparison.worker import make_task, base

from .grounded_evidence_v15 import _task_focus
from .memory import Embedder, digest
from .recurrent_feedback import stable_feedback


def audit(run: Path, episode: int, output: Path) -> dict:
    run = run.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    design = read(run / 'design.json')
    if design['benchmark'] != 'clbench' or not 2 <= episode <= design['limit']:
        raise ValueError('Expected an existing CLBench cell after episode one')
    prior = run / 'typed_grounded' / f'episode_{episode-1:03d}'
    current = run / 'typed_grounded' / f'episode_{episode:03d}'
    prior_row = read(prior / 'row.json')
    row = read(current / 'row.json')
    snapshot = prior / 'memory_after.json'
    if (prior_row['status'] != 'complete' or row['status'] != 'complete' or
            sha(snapshot) != prior_row['memory_after_sha256'] or
            sha(Path(design['origin']) / 'plan.json') !=
            design['origin_plan_sha256'] or
            row['initial_query_sha256'] != design['bindings'][episode-1]):
        raise ValueError('Changed source, incomplete cell, or wrong target input')
    retrieval = read(current / 'retrieval.json')
    plan = read(Path(design['origin']) / 'plan.json')
    benchmark = Path(os.environ.get('TTCL_BENCH',
                                    Path(design['origin']) / 'source' / 'bench'))
    os.chdir(benchmark)
    task = make_task(design['task'], plan['task_seed'])
    query = task.reset_baseline_instance(episode-1)
    brief = task.get_agent_brief()
    formatted_brief = base.format_task_agent_brief(brief) if brief else ''
    normalized = normalize_prompt(query.prompt, formatted_brief, episode-1,
                                  plan['tasks'][design['task']])
    if (hashlib.sha256(query.prompt.encode()).hexdigest() != row[
            'initial_query_sha256'] or retrieval['query'] != normalized):
        raise ValueError('Retrieved question differs from target input')
    events = read(snapshot)['grounded_evidence']['events']
    groups = stable_feedback(events)
    selected = retrieval.get('recurrent_feedback', [])
    selected_keys = {(item['action_sha256'], item['feedback_sha256'])
                     for item in selected}
    group_keys = {(digest(group['action']), hashlib.sha256(
        group['feedback'].encode()).hexdigest()) for group in groups}
    if len(selected_keys) != len(selected) or not selected_keys <= group_keys:
        raise ValueError('Selected feedback is not in the frozen prior bank')
    embedder = Embedder(plan['embedding'])
    texts = [_task_focus(retrieval['query'])] + [json.dumps(
        group['action'], ensure_ascii=False, sort_keys=True) + '\n' +
        group['feedback'] for group in groups]
    vectors = embedder.embed(texts)
    scored = []
    for group, vector in zip(groups, vectors[1:]):
        action_sha256 = digest(group['action'])
        feedback_sha256 = hashlib.sha256(group['feedback'].encode()).hexdigest()
        scored.append(dict(action_sha256=action_sha256,
                           feedback_sha256=feedback_sha256,
                           selected=(action_sha256, feedback_sha256) in selected_keys,
                           direct_similarity=sum(a*b for a,b in zip(vectors[0],
                                                                    vector)),
                           source_episode_count=group['episode_count']))
    scored.sort(key=lambda item: -item['direct_similarity'])
    report = dict(schema='feedback_direct_relevance_audit_v1',
                  parent=str(run), parent_design_sha256=sha(run / 'design.json'),
                  episode=episode, input_sha256=row['initial_query_sha256'],
                  prior_snapshot_sha256=sha(snapshot),
                  direct_text='current task focus versus action and observed feedback',
                  selected_count=len(selected), group_count=len(groups),
                  groups=scored,
                  caveat='Post-hoc semantic diagnosis only; no scored-suffix threshold may be fitted from this report')
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.episode, args.output)
    print(f"Audited {result['selected_count']}/{result['group_count']} selected groups")


if __name__ == '__main__':
    main()
