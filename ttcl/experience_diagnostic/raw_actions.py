"""Exploratory fidelity control: retain action/observation associations.

Original quotes and packing order remain fixed. This adds executed tool input,
never actor thoughts, future queries, answers, or reward-selected information.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

from ttcl.experience_diagnostic import run


def prepare(root):
    from transformers import AutoTokenizer
    parent = root.parent
    plan = run.read(parent / 'plan.json')
    plan = dict(plan, arms=['raw_with_actions'], expected_cells=16,
                exploratory=True, base_experiment=str(parent))
    tokenizer = AutoTokenizer.from_pretrained(plan['model']['model'], local_files_only=True)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    run.save(root / 'plan.json', plan)
    paths = [root / 'plan.json']
    for task in plan['tasks']:
        for ep in plan['source_episodes']:
            src = parent / 'inputs' / task / str(ep)
            episode, before = run.read(src / 'trajectory.json'), run.read(src / 'bank_before.json')
            specs = run.read(src / 'raw_spec.json')
            excerpts = []
            for item in specs:
                step = next(s for s in episode['steps'] if s['step'] == item['step'])
                assert item['text'] in step['public_feedback']
                action = step['action'].get('tool_call', step['action'])
                excerpts.append(dict(episode=ep, step=item['step'], executed_action=action, quote=item['text']))
            value = {'public_tool_excerpts': excerpts, 'earlier_bank_entries': []}
            prefix = 'Earlier executed actions and verbatim public tool evidence. Scope may differ in the current task. Earlier bank entries are unverified notes.\n'
            render = lambda: prefix + json.dumps(value, sort_keys=True, ensure_ascii=False)
            if count(render()) > 2048:
                raise ValueError('Action-associated evidence exceeds budget; do not silently truncate')
            omitted = []
            for entry in before['entries']:
                value['earlier_bank_entries'].append(entry)
                if count(render()) > 2048:
                    value['earlier_bank_entries'].pop()
                    omitted.append(entry['id'])
            context = render()
            path = root / 'candidates' / task / str(ep) / 'raw_with_actions_context.json'
            run.save(path, {'context': context, 'tokens': count(context),
                            'sha256': hashlib.sha256(context.encode()).hexdigest(),
                            'omitted_old_ids': omitted, 'source_quotes_unchanged': True})
            paths.append(path)
            print(task, ep, count(context), 'omitted', omitted)
    run.save(root / 'candidate_hashes.json', {str(p): run.hash_file(p) for p in paths})
    (root / 'PROTOCOL.md').write_text(
        '# 原始证据关联补充诊断\n\n'
        '这是主实验运行中发现表示缺少动作参数后追加的探索性对照，不改写五组主实验。'
        '原始raw_spec的反馈片段及其顺序完全不变，添加对应已执行action/tool_call，去掉actor thought。'
        '仍限制2048 tokens，优先新证据，再保留装得下的完整旧条目，记录省略项。'
        '不使用任何新probe结果选择或修改片段、动作或预算。全部新上下文先冻结再评分。'
        '两个任务×两个历史×两道后续题×两个seed，共16单元。'
        '复用原实验同一冻结actor、官方环境、解码和预算；与原keep和raw的同题同seed结果配对。'
        '表示长度和旧条目保留可能变化，因此是证据呈现敏感性检查，不能单独归因于动作信息。\n'
    )


def score(root, task, repeat, episode):
    original_read, original_save = run.read, run.save

    def subset_read(path, default=None):
        value = original_read(path, default)
        if path == root / 'plan.json':
            value = copy.deepcopy(value)
            value['source_episodes'] = [episode]
        return value

    def subset_save(path, value):
        if path.parent == root / 'progress':
            path = path.with_name(f'{episode}_' + path.name)
        original_save(path, value)

    run.read, run.save, run.ARMS = subset_read, subset_save, ['raw_with_actions']
    run.score(root, task, repeat)


if __name__ == '__main__':
    root = Path(sys.argv[2]).resolve()
    if sys.argv[1] == 'prepare':
        prepare(root)
    else:
        score(root, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
