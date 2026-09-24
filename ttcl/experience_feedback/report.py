from __future__ import annotations
import statistics
from ttcl.experience_evolution.core import read, save


def report(root, suite='feedback_transfer'):
    settings = read(root/'plan.json')['suites'][suite]
    result = {'protocol': 'completed_episode_reward_v1', 'tasks': {}}
    lines = ['# CLBench：逐任务 reward 反馈评估', '',
             '每个任务结束后，轨迹及官方 reward 用于更新经验；下一任务才读取更新后的经验。模型参数冻结。',
             '每个场景单独报告；未完成部分只作进度观察。执行错误单独计数，缺失 reward 不补零。',
             '所有组均在本协议下重跑；不继承旧无 reward 输入实验的分数。', '']
    for task, count in settings['tasks'].items():
        rows = [read(p) for p in (root/suite/'clbench'/task).glob('*/*/episode_*/row.json')
                if '.interrupted_' not in str(p)]
        groups = {a: {(r['repeat'], r['canonical_index']): r for r in rows if r['arm']==a
                     and r['status']=='complete' and r['reward'] is not None} for a in settings['arms']}
        common = sorted(set.intersection(*(set(g) for g in groups.values())))
        expected_per_arm = count*len(settings['repeats'])
        complete = len(rows)==expected_per_arm*len(groups)
        t = {'recorded': len(rows), 'expected': expected_per_arm*len(groups),
             'all_attempts_recorded': complete, 'common_scored_pairs': len(common), 'arms': {}}
        lines += [f'## {task}：{len(rows)}/{t["expected"]} 条记录；共同可计分样本 {len(common)}/{expected_per_arm}', '',
                  '全部任务已记录。' if complete else '**评估未完成，以下是共同子集的临时结果。**', '',
                  '| 组 | 可计分数 / 应有数 | 共同子集均值 | 对无经验差值 | 对未训练差值 | 执行错误 |',
                  '|---|---:|---:|---:|---:|---:|']
        for a, g in groups.items():
            values = [g[k]['reward'] for k in common]
            v = {'scored': len(g), 'expected': expected_per_arm,
                 'mean_common': statistics.mean(values) if values else None,
                 'errors': sum(r['arm']==a and r['status']!='complete' for r in rows),
                 'by_repeat': {str(s): {'n': sum(k[0]==s for k in common),
                     'mean_common': statistics.mean(g[k]['reward'] for k in common if k[0]==s)
                     if any(k[0]==s for k in common) else None} for s in settings['repeats']}}
            for baseline in ['none', 'untrained']:
                ds=[g[k]['reward']-groups[baseline][k]['reward'] for k in common]
                later=[g[k]['reward']-groups[baseline][k]['reward'] for k in common if k[1]>0]
                v['vs_'+baseline]={'mean_delta': statistics.mean(ds) if ds else None,
                                  'after_first_n':len(later), 'after_first_delta':statistics.mean(later) if later else None,
                                  'wins':sum(x>1e-12 for x in ds), 'losses':sum(x< -1e-12 for x in ds),
                                  'ties':sum(abs(x)<=1e-12 for x in ds)}
            t['arms'][a]=v
            fmt=lambda x: '—' if x is None else f'{x:+.6f}'
            lines.append(f"| {a} | {len(g)}/{expected_per_arm} | {fmt(v['mean_common'])} | {fmt(v['vs_none']['mean_delta'])} | {fmt(v['vs_untrained']['mean_delta'])} | {v['errors']} |")
        result['tasks'][task]=t
        lines.append('')
    save(root/suite/'comparison.json', result)
    (root/suite/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def audit_chain(root, suite, task, repeat):
    """Read actual saved prompts and memory handoffs, including failure trajectories."""
    from .protocol import public_messages
    settings=read(root/'plan.json')['suites'][suite]
    totals={'rows':0,'writers':0,'negative_rewards':0,'zero_rewards':0,'missing_rewards':0}
    for arm in settings['arms']:
        memory=''
        for index in range(settings['tasks'][task]):
            d=root/suite/'clbench'/task/str(repeat)/arm/f'episode_{index+1:03}'
            row=read(d/'row.json'); episode=read(d/'trajectory.json')
            assert row['reward']==episode['reward']
            assert read(d/'memory_before.json')['text']==memory
            assert row['actor_adapter_enabled'] is False
            totals['rows']+=1
            if arm=='none': continue
            writer=read(d/'writer.json')
            assert writer['messages']==public_messages(memory,episode)
            assert writer['served_model']==settings['arms'][arm]
            memory=writer['raw_response']
            totals['writers']+=1
            r=episode['reward']
            totals['negative_rewards']+=r is not None and r<0
            totals['zero_rewards']+=r==0
            totals['missing_rewards']+=r is None
    save(root/suite/'audits'/f'{task}_{repeat}.json',dict(passed=True,**totals))
    return totals
