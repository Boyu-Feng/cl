"""Audit a frozen repeated ALF actor prompt without rerunning the model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import (ACTOR_SYSTEM, clean_command,
                                                  make_env)
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .memory import digest


def audit(output: Path) -> dict:
    design, report = read(output / 'design.json'), read(output / 'report.json')
    source = Path(design['source'])
    source_episode = Path(design['source_episode'])
    comparison_episode = Path(design['comparison_episode'])
    if (not audit_source(source)['complete'] or
            (output / 'failure.json').exists() or
            not source_episode.is_relative_to(source) or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source_episode) != design['source_episode_sha256'] or
            sha(comparison_episode) != design['comparison_episode_sha256'] or
            sha(output / 'probe_source.py') != design['source_code_sha256'] or
            sha(output / 'design.json') != report['design_sha256']):
        raise ValueError('Frozen same-prompt source or report changed')
    plan = read(source / 'plan.json')
    episode, comparison = read(source_episode), read(comparison_episode)
    step = design['step']
    if (episode['status'] != 'complete' or
            comparison['status'] != 'complete' or
            episode['game'] != comparison['game'] or
            episode['seed'] != comparison['seed'] or
            not 1 <= step <= min(len(episode['generations']),
                                 len(comparison['generations']))):
        raise ValueError('Source actor episodes differ')
    selected, other = (episode['generations'][step-1],
                       comparison['generations'][step-1])
    if (selected['seed'] != other['seed'] or
            selected['seed'] != design['actor_seed'] or
            selected['rendered_prompt_sha256'] !=
            other['rendered_prompt_sha256'] or
            selected['rendered_prompt_sha256'] !=
            design['rendered_prompt_sha256'] or
            selected['prompt_sha256'] != other['prompt_sha256']):
        raise ValueError('Selected prompt or seed changed')
    for index in range(step-1):
        if (episode['generations'][index]['text'] !=
                comparison['generations'][index]['text'] or
                episode['trajectory'][index]['action'] !=
                comparison['trajectory'][index]['action']):
            raise ValueError('Actor prefix diverged before selected prompt')
    game = Path(plan['alf']['data_root']) / episode['game']
    if (sha(game) != design['game_sha256'] or
            plan['alf']['actor_temperature'] !=
            design['actor_temperature'] or
            plan['alf']['actor_max_tokens'] != design['actor_max_tokens']):
        raise ValueError('Official game or actor sampling plan changed')
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
    if (digest(messages) != selected['prompt_sha256'] or
            digest(messages) != design['prompt_digest'] or
            digest(available) != report['available_commands_sha256']):
        raise ValueError('Reconstructed actor prompt changed')
    if (report['source_command'] != episode['trajectory'][step-1]['action'] or
            report['comparison_command'] !=
            comparison['trajectory'][step-1]['action'] or
            report['source_response_sha256'] != hashlib.sha256(
                selected['text'].encode()).hexdigest() or
            report['comparison_response_sha256'] != hashlib.sha256(
                other['text'].encode()).hexdigest()):
        raise ValueError('Archived comparison response changed')
    rows = report['rows']
    if len(rows) != design['repeats']:
        raise ValueError('Missing repeated actor completions')
    for index,row in enumerate(rows):
        raw = row['raw_response']
        if (row['index'] != index or not isinstance(raw, str) or
                hashlib.sha256(raw.encode()).hexdigest() !=
                row['response_sha256'] or
                row['command'] != clean_command(raw, available) or
                not 0 <= row['output_tokens'] <=
                design['actor_max_tokens']):
            raise ValueError('Repeated response or command changed')
    unique = {row['response_sha256'] for row in rows}
    return dict(schema='same_actor_prompt_audit_v1',
                design_sha256=sha(output / 'design.json'),
                complete=True, repeats=len(rows),
                unique_responses=len(unique),
                source_command=report['source_command'],
                comparison_command=report['comparison_command'],
                repeated_commands={command:sum(row['command'] == command
                                                for row in rows)
                                   for command in sorted({row['command']
                                                          for row in rows})},
                caveat='Post-hoc selected prompt and one time-local burst; not a global reproducibility rate')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    if args.report:
        save(args.report.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
