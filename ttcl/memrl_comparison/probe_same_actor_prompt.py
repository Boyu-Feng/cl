"""Replay one audited ALF actor prompt repeatedly without writing memory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from ttcl.alfworld_comparison.environment import (ACTOR_SYSTEM, clean_command,
                                                  make_env)
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .memory import digest


def run(source: Path, source_episode: Path, comparison_episode: Path,
        step: int, repeats: int, output: Path, url: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not audit_source(source)['complete']:
        raise ValueError('Native source is incomplete')
    if (not source_episode.is_relative_to(source) or
            step < 1 or repeats < 2):
        raise ValueError('Prompt selection or repeat count is invalid')
    plan = read(source / 'plan.json')
    plan['url'] = url
    episode, comparison = read(source_episode), read(comparison_episode)
    if (episode['status'] != 'complete' or
            comparison['status'] != 'complete' or
            episode['game'] != comparison['game'] or
            episode['seed'] != comparison['seed'] or
            not 1 <= step <= min(len(episode['generations']),
                                 len(comparison['generations']))):
        raise ValueError('Unmatched complete actor episodes')
    for previous in range(step - 1):
        if (episode['generations'][previous]['text'] !=
                comparison['generations'][previous]['text'] or
                episode['trajectory'][previous]['action'] !=
                comparison['trajectory'][previous]['action']):
            raise ValueError('Earlier actor prefix already diverged')
    selected = episode['generations'][step - 1]
    other = comparison['generations'][step - 1]
    if (selected['seed'] != other['seed'] or
            selected['rendered_prompt_sha256'] !=
            other['rendered_prompt_sha256'] or
            selected['prompt_sha256'] != other['prompt_sha256']):
        raise ValueError('Selected actor prompt or seed differs')
    game = Path(plan['alf']['data_root']) / episode['game']
    env = make_env(game)
    try:
        state = env.reset()
        system = ACTOR_SYSTEM + ('\n\nPast experience:\n' + episode['memory']
                                 if episode['memory'] else '')
        messages = [{'role':'system', 'content':system}]
        for index in range(step):
            messages.append({'role':'user', 'content':str(state['feedback']) +
                             '\nAvailable commands:\n' +
                             '\n'.join(state['admissible_commands'])})
            if index == step - 1:
                available = list(state['admissible_commands'])
                break
            messages.append({'role':'assistant',
                             'content':episode['generations'][index]['text']})
            state,_,_ = env.step(episode['trajectory'][index]['action'])
    finally:
        env.close()
    if digest(messages) != selected['prompt_sha256']:
        raise ValueError('Reconstructed public actor prompt changed')
    output.mkdir(parents=True)
    source_code = Path(__file__)
    design = dict(schema='same_actor_prompt_probe_v1', source=str(source),
                  source_design_sha256=sha(source / 'design.json'),
                  source_episode=str(source_episode),
                  source_episode_sha256=sha(source_episode),
                  comparison_episode=str(comparison_episode),
                  comparison_episode_sha256=sha(comparison_episode),
                  game_sha256=sha(game), step=step, repeats=repeats,
                  prompt_digest=digest(messages),
                  rendered_prompt_sha256=selected['rendered_prompt_sha256'],
                  actor_seed=selected['seed'],
                  actor_temperature=plan['alf']['actor_temperature'],
                  actor_max_tokens=plan['alf']['actor_max_tokens'],
                  source_code_sha256=sha(source_code), url=url)
    save(output / 'design.json', design)
    shutil.copy2(source_code, output / 'probe_source.py')
    client = Client(plan, 0)
    rows = []
    try:
        for index in range(repeats):
            generated = client.complete(
                messages, selected['seed'],
                tokens=plan['alf']['actor_max_tokens'],
                temperature=plan['alf']['actor_temperature'], top_p=1.)
            if generated['rendered_prompt_sha256'] != \
                    selected['rendered_prompt_sha256']:
                raise ValueError('Client and archived prompt rendering differ')
            rows.append(dict(index=index,
                             raw_response=generated['raw_response'],
                             response_sha256=hashlib.sha256(
                                 generated['raw_response'].encode()).hexdigest(),
                             command=clean_command(generated['raw_response'],
                                                   available),
                             output_tokens=generated['output_tokens'],
                             finish_reason=generated['finish_reason']))
    except Exception as exc:
        save(output / 'failure.json', dict(completed=len(rows), expected=repeats,
                                           error=repr(exc), rows=rows))
        raise
    report = dict(design_sha256=sha(output / 'design.json'),
                  source_command=episode['trajectory'][step-1]['action'],
                  comparison_command=comparison['trajectory'][step-1]['action'],
                  source_response_sha256=hashlib.sha256(
                      selected['text'].encode()).hexdigest(),
                  comparison_response_sha256=hashlib.sha256(
                      other['text'].encode()).hexdigest(),
                  available_commands_sha256=digest(available), rows=rows)
    save(output / 'report.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-episode', type=Path, required=True)
    parser.add_argument('--comparison-episode', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    report = run(args.source.resolve(), args.source_episode.resolve(),
                 args.comparison_episode.resolve(), args.step, args.repeats,
                 args.output.resolve(), args.url)
    print(json.dumps(dict(unique_responses=len({row['response_sha256']
                                                for row in report['rows']}),
                          commands=dict(Counter(row['command']
                                                for row in report['rows'])))))


if __name__ == '__main__':
    main()
