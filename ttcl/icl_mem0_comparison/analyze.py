"""Read-only, file-bound analysis of a live ICL/Mem0 run; outputs a new snapshot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid(row):
    value = row.get('reward')
    return row.get('status') == 'complete' and isinstance(value, (int, float)) and math.isfinite(value)


def stats(pairs, bootstrap=5000):
    if not pairs:
        return {'n': 0}
    deltas = [b['reward'] - a['reward'] for a, b in pairs]
    clusters = defaultdict(list)
    for a, b in pairs:
        clusters[a['canonical_index']].append(b['reward'] - a['reward'])
    grouped = [(sum(v), len(v)) for v in clusters.values()]
    rng = random.Random(20260928)
    draws = []
    for _ in range(bootstrap):
        sample = rng.choices(grouped, k=len(grouped))
        draws.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    draws.sort()
    ordered = sorted(deltas, reverse=True)
    def cost(arm):
        return {key: sum(p[arm].get(key, 0) for p in pairs) for key in
                ('actor_calls', 'actor_input_tokens', 'actor_output_tokens', 'writer_calls',
                 'writer_input_tokens', 'writer_output_tokens')}
    return {'n': len(pairs), 'unique_tasks': len(grouped),
            'none_mean': statistics.fmean(a['reward'] for a, _ in pairs),
            'mem0_mean': statistics.fmean(b['reward'] for _, b in pairs),
            'delta': statistics.fmean(deltas), 'median_delta': statistics.median(deltas),
            'wins': sum(d > 0 for d in deltas), 'ties': sum(d == 0 for d in deltas),
            'losses': sum(d < 0 for d in deltas),
            'task_cluster_bootstrap_95': [draws[int(.025*bootstrap)], draws[int(.975*bootstrap)]],
            'sensitivity_only_drop_largest_1': statistics.fmean(ordered[1:]) if len(ordered)>1 else None,
            'sensitivity_only_drop_largest_3': statistics.fmean(ordered[3:]) if len(ordered)>3 else None,
            'largest_positive_deltas': ordered[:5], 'costs': {'none': cost(0), 'mem0': cost(1)}}


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    rows, bindings, paths = {}, {str(Path(__file__).resolve()):sha(Path(__file__).read_bytes())}, {}
    for p in sorted((root/'runs').glob('*/*/*/episode_*/row.json')):
        data = p.read_bytes()
        row = json.loads(data)
        key = (row['task'], row['repeat'], row['canonical_index'], row['arm'])
        rows[key], paths[key] = row, p
        bindings[str(p)] = sha(data)
    plan = json.loads((root/'plan.json').read_text())
    bindings[str(root/'plan.json')] = sha((root/'plan.json').read_bytes())
    report = {'as_of_utc': datetime.now(timezone.utc).isoformat(), 'source': str(root),
              'recorded': len(rows), 'protocol': 'Mem0 versus none, both valid, exclude each chain first task',
              'bootstrap': '5000 resamples of canonical task IDs, all available seeds kept together',
              'domains': {}}
    for task, count in plan['tasks'].items():
        keys = sorted({(t, s, i) for t, s, i, arm in rows if t == task and i > 0})
        pairs = [(rows[*k, 'none'], rows[*k, 'mem0']) for k in keys
                 if all((*k, a) in rows and valid(rows[*k, a]) for a in ('none', 'mem0'))]
        arm_stats = {}
        for arm in plan['arms']:
            rr = [r for (t,s,i,a),r in rows.items() if t == task and a == arm]
            arm_stats[arm] = {'recorded': len(rr), 'scored': sum(valid(r) for r in rr),
                             'failed': sum(not valid(r) for r in rr),
                             'failure_reasons': dict(Counter(r.get('error','unknown') for r in rr if not valid(r)))}
        both_recorded = [(rows[*k,'none'], rows[*k,'mem0']) for k in keys
                         if all((*k,a) in rows for a in ('none','mem0'))]
        availability = Counter(('both_scored' if valid(a) and valid(b) else
                                'only_none_scored' if valid(a) else
                                'only_mem0_scored' if valid(b) else 'both_missing') for a,b in both_recorded)
        report['domains'][task] = {**stats(pairs), 'arms': arm_stats,
            'availability_on_both_recorded': dict(availability),
            'by_seed': {str(s): stats([(a,b) for a,b in pairs if a['repeat']==s],1000) for s in plan['repeats']},
            'first_half': stats([(a,b) for a,b in pairs if a['canonical_index']<count/2],1000),
            'second_half': stats([(a,b) for a,b in pairs if a['canonical_index']>=count/2],1000)}
    cases = []
    for task in plan['tasks']:
        kk = [(t,s,i) for t,s,i,a in rows if t == task and a == 'mem0' and i > 0
              and (t,s,i,'none') in rows and valid(rows[t,s,i,'none']) and valid(rows[t,s,i,'mem0'])]
        kk.sort(key=lambda k: rows[*k,'mem0']['reward']-rows[*k,'none']['reward'])
        selected = list(dict.fromkeys(kk[:1]+kk[-2:]))
        for k in selected:
            evidence = {'task':task, 'repeat':k[1], 'index':k[2],
                        'selection':'descriptive largest positive/negative paired differences; not causal identification'}
            for arm in ('none','mem0'):
                directory = paths[*k,arm].parent
                item = {'row': rows[*k,arm], 'directory': str(directory)}
                for name in ('public_trajectory.json', 'retrieval_001.json'):
                    p = directory/name
                    if p.exists():
                        data = p.read_bytes(); bindings[str(p)] = sha(data)
                        item[name] = json.loads(data)
                evidence[arm] = item
            cases.append(evidence)
    for name, data in [('summary.json',report),('source_bindings.json',bindings),
                       ('rows_snapshot.json',list(rows.values())),('cases.json',cases)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines = ['# Mem0 当前收益的证据分析', '', '快照：'+report['as_of_utc'], '',
             '这是一份运行中实验的只读快照。失败不补零；两组共同有效、排除每条链首题。',
             '任务聚类区间仅描述当前样本的不确定性；连续在线记忆使不同题之间也有依赖，不能将其解释为最终确认检验。', '',
             '|领域|配对|无经验|Mem0|差值|任务聚类 95% 区间|胜/平/负|',
             '|---|---:|---:|---:|---:|---|---|']
    for task,d in report['domains'].items():
        if d['n']:
            lo,hi=d['task_cluster_bootstrap_95']
            lines.append(f"|{task}|{d['n']}|{d['none_mean']:.6f}|{d['mem0_mean']:.6f}|{d['delta']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|{d['wins']}/{d['ties']}/{d['losses']}|")
    lines += ['', '## 缺失评分', '',
        '下表为全部已记录样本（含首题）；失败不补零。各臂运行进度可能不同。', '',
        '|领域|无经验 有分/失败|Mem0 有分/失败|', '|---|---:|---:|']
    for task,d in report['domains'].items():
        a,b=d['arms']['none'],d['arms']['mem0']
        lines.append(f"|{task}|{a['scored']}/{a['failed']}|{b['scored']}/{b['failed']}|")
    lines += ['', '队列研究的共同有效子集变差，与 Mem0 的可计分样本更多可以同时发生；不能把子集均值当成含全部失败的总体效果。', '',
        '## 计算成本（同一批有效配对）', '',
        '|领域|无经验 actor 调用|Mem0 actor 调用|Mem0 额外 writer 调用|总输入＋输出 token 比值|',
        '|---|---:|---:|---:|---:|']
    for task,d in report['domains'].items():
        if not d['n']:continue
        a,b=d['costs']['none'],d['costs']['mem0']
        tokenkeys=['actor_input_tokens','actor_output_tokens','writer_input_tokens','writer_output_tokens']
        ratio=sum(b[k] for k in tokenkeys)/max(1,sum(a[k] for k in tokenkeys))
        lines.append(f"|{task}|{a['actor_calls']}|{b['actor_calls']}|{b['writer_calls']}|{ratio:.2f}×|")
    lines += ['', '减少环境探索不等于减少总模型计算；Mem0 的事实提取也消耗额外推理。', '',
        '## 从实现和实际轨迹可以确认的机制', '',
        '1. 所有权重冻结；Mem0 在每个公开反馈后提取并存储文本，在每个动作前检索最多 10 条。因此现有收益包含同题内辅助和跨题迁移，二者尚未被消融分离。',
        '2. 数据库 404/episode_002 的首轮检索包含先前探索出的 items_g1、fdbk_g1 和字段信息。Mem0 用两条查询获得正确均值；无经验用了六条查询并加了错误的 main_cat 过滤，最终回答错误。该轨迹支持复用 schema 和减少重复探索的解释，但单个案例不能证明因果。',
        '3. 检索可避免把所有历史都塞入提示。当前全历史 ICL 曾发生 262144-token 上下文溢出，Mem0 的检索上下文更紧凑；逐领域共同样本的实际调用/token 成本见 summary.json，Mem0 仍需要额外提取调用，并非无成本。',
        '4. 队列 303/episode_008 的记忆把前题估计写成已确立的生存规律。模型据此声称无需探索，且把 0/20 已用动作误读为预算耗尽，直接提交旧数值。该例说明压缩记忆也会放大自我预测和适用范围错误。',
        '5. 扑克奖励厚尾；均值提升并不等于多数牌局都改善。最大正差值及去掉最大 1/3 项的敏感性统计单列，原始正式分数不删除也不改写。', '',
        '## 可比性限制', '',
        'Mem0 使用 mem0ai 2.0.0、BGE-M3 和原生 dense/BM25/entity 检索。它与早期单条滚动经验 writer 在更新频率、容量、检索机制和额外推理量上不同，不能把差异归因于一个算法开关。',
        '三组共同计分会额外排除 ICL 失败样本，本报告的主要效果使用 Mem0/none 两组交集。可计分性差异另列 availability_on_both_recorded，不把失败强行换成 0。',
        '当前没有随机记忆、移除跨题记忆、固定上下文长度或等提取调用数对照，因此“为什么有效”是由实现与轨迹支持的机制解释，不是已经完成的因果消融。', '',
        '所有使用的 row、案例文件和计划的 SHA-256 见 source_bindings.json；完整选例见 cases.json。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.root,a.output)
    print(json.dumps({'output':str(a.output),'recorded':r['recorded'],
                      'domains':{k:{f:v.get(f) for f in ('n','delta','task_cluster_bootstrap_95')} for k,v in r['domains'].items()}},ensure_ascii=False,indent=2))
