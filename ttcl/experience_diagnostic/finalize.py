"""Combine the preregistered main table and exploratory evidence-format check."""

import hashlib
import json
from pathlib import Path
import statistics
import sys

from ttcl.experience_diagnostic.analyze import analyze


def read(path):
    return json.loads(path.read_text())


def key(row):
    return row['task'], row['source_episode'], row['canonical_index'], row['repeat']


def main(root):
    analyze(root)
    analysis = read(root / 'analysis.json')
    base_rows = [read(p) for p in (root / 'scores').rglob('row.json')]
    assert len(base_rows) == 80 and analysis['checks_passed']
    base = {(key(r), r['arm']): r for r in base_rows}
    assert len(base) == 80
    supplement = root / 'supplement_raw_actions'
    extra = [read(p) for p in supplement.glob('scores/**/row.json')]
    assert len(extra) == 16 and len({key(r) for r in extra}) == 16
    for path, expected in read(supplement / 'candidate_hashes.json').items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for path in (root / 'scheduling').glob('*code.json'):
        manifest = read(path)
        assert hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest() == manifest['sha256']
    summaries = {}
    paired_rows = []
    for row in extra:
        source = supplement / 'scores' / row['task'] / str(row['repeat']) / str(row['source_episode']) / str(row['episode']) / 'raw_with_actions'
        context = read(supplement / 'candidates' / row['task'] / str(row['source_episode']) / 'raw_with_actions_context.json')
        assert context['sha256'] == row['bank_context_sha256']
        events = [json.loads(s) for s in (source / 'responses.jsonl').read_text().splitlines()]
        assert all(e['writer_adapter_enabled'] is False for e in events)
        baseline = base[key(row), 'keep']
        assert row['instance_id'] == baseline['instance_id']
        paired_rows.append(dict(task=row['task'], source_episode=row['source_episode'],
                                probe_episode=row['episode'], repeat=row['repeat'],
                                status=row['status'], reward=row['reward'], keep=baseline['reward'],
                                raw=base[key(row), 'raw']['reward']))
    for task in ['database_exploration', 'cohort_studies']:
        rows = [r for r in extra if r['task'] == task]
        complete = [r for r in rows if r['status'] == 'complete' and base[key(r), 'keep']['status'] == 'complete']
        complete_raw = [r for r in complete if base[key(r), 'raw']['status'] == 'complete']
        summaries[task] = {
            'completed': sum(r['status'] == 'complete' for r in rows), 'expected': 8,
            'paired_vs_keep': len(complete),
            'mean_reward': statistics.mean(r['reward'] for r in complete) if complete else None,
            'delta_vs_keep': statistics.mean(r['reward'] - base[key(r), 'keep']['reward'] for r in complete) if complete else None,
            'delta_vs_original_raw': statistics.mean(r['reward'] - base[key(r), 'raw']['reward'] for r in complete_raw) if complete_raw else None,
            'pairs_vs_original_raw': len(complete_raw),
        }
    summary = {'exploratory': True, 'summaries': summaries, 'pairs': paired_rows,
               'completed_cells': sum(r['status'] == 'complete' for r in extra),
               'actual_actor_calls': sum(r['actor_calls'] for r in extra),
               'actual_actor_input_tokens': sum(r['actor_input_tokens'] for r in extra),
               'actual_actor_output_tokens': sum(r['actor_output_tokens'] for r in extra),
               'checks_passed': True}
    (supplement / 'comparison.json').write_text(json.dumps(summary, indent=2))
    lines = ['# 固定历史诊断：结果解读', '',
             f'主实验80个单元，{sum(r["status"] == "complete" for r in base_rows)}个获得官方分数；另完成动作关联证据补充诊断16个单元，其中{summary["completed_cells"]}个获得官方分数。', '',
             'actor固定为Qwen3-4B；只改变经验上下文。每任务4道目标题、2个采样seed，来源于2个历史点。当前是既有开发环境诊断，不是稳定泛化结论。', '',
             '## 全部有效配对相对保留旧库的差值', '',
             '此表是缺失敏感性检查：每个组使用自身与keep共同完成的全部配对，n不同则不能当作完全同样本排名。预先定义的五组共同完成主表仍保留在REPORT.md。', '',
             '| 组别 | Database Δ（n） | Cohort Δ（n） |', '|---|---:|---:|']
    for arm in ['untrained', 'utility_sft', 'audited', 'raw']:
        values = [analysis['summaries'][t]['secondary_pairwise_available'][arm] for t in ['database_exploration', 'cohort_studies']]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{v["mean_delta_vs_keep"]:+.6f} ({v["paired_count"]})' for v in values) + ' |')
    values = [summaries[t] for t in ['database_exploration', 'cohort_studies']]
    lines.append('| raw_with_actions（补充） | ' + ' | '.join(f'{v["delta_vs_keep"]:+.6f} ({v["paired_vs_keep"]})' for v in values) + ' |')
    lines += ['', '## 已核实的行为证据', '',
              '1. Database第13题后的相同核查经验，在第14题（办公用品）两个seed都引导actor查询fdbk_g2并答415，均错误；经验本身要求验证类别映射，但actor没有执行该检查。第15题（电子产品）两个seed则都答对，reward为0.5333/0.3333，keep两个seed均为0。这是同一记忆存在正、负迁移的具体例子。它支持进一步检查适用性判断，不能把全部差异归为单一条目的因果作用。',
              '2. Database的utility_sft在第13题后选择KEEP，所以这段历史下与keep的输入完全相同，结果显式复用。它避免了untrained的错误类别推断，但这个结果不能证明学会了提取有用的新经验。第17题后的两个Qwen writer都把旧的时间查询条目改成品牌条目，丢失了原有的不同主题内容。',
              '3. Database第18、19题五组均未答对。这些单元不能有效区分writer；历史信息是否足够、actor是否能完成所需SQL推理，都仍是限制。',
              '4. Cohort第14题seed606的utility_sft最终报告格式无效，两次规定修复重试后仍未通过。没有官方reward，不能记成0，也没有额外挑选重跑。原raw组该单元实际reward=-0.583596，keep=-0.011465；若只看五组共同完成均分，这个raw的大幅退步会随SFT缺失一起被排除。因此必须同时查看上面的逐组有效配对表。',
              '5. 在相同历史下，untrained四份更新与旧实验逐字一致；actor的LoRA禁用标记、配对seed和实际经验上下文检查通过。候选8/8通过结构校验，但内容检查仍发现错误外键/类别推断、把局部KL当作生物学验证等问题。格式通过不代表语义通过。', '',
              '## 原始证据呈现补充', '',
              '主实验raw只有工具返回值；补充组在相同反馈片段旁添加原始执行动作参数，不加入actor猜测，不依照未来题目挑选新证据。该补充是在主实验进行中追加，属于探索性检查；两组实际长度和装得下的旧条目可能不同。', '',
              '| 任务 | raw_with_actions相对原raw | 有效配对 |', '|---|---:|---:|']
    for task, value in summaries.items():
        lines.append(f'| {task} | {value["delta_vs_original_raw"]:+.6f} | {value["pairs_vs_original_raw"]} |')
    lines += ['',
              f'动作关联证据相对keep的增量：Database {summaries["database_exploration"]["delta_vs_keep"]:+.6f}，Cohort {summaries["cohort_studies"]["delta_vs_keep"]:+.6f}。',
              ('本轮两个任务的增量均为正，提供了当前actor能够在这些样本上利用历史的正向证据。动作、条件与返回值的对应关系值得作为下一轮writer训练的重点。'
               if all(v['delta_vs_keep'] > 0 for v in summaries.values()) else
               '本轮补充结果没有在两个任务上均带来正收益，需要结合逐题结果判断证据表示的影响。')
              + '补充组同时改变文本长度及部分旧条目保留，不能把全部差值唯一归因于某个字段。']
    lines += ['', '## 推论边界', '',
              '这次同时观察到writer的证据误读，以及actor对经验适用范围执行不可靠的行为。更好的文字事实性并不自动保证每道新题获益。现有小样本不足以判断模型学不会，也不足以确立某种writer的稳定排名。下一步值得用同历史的适用/不适用新题监督经验选择，并把证据支持与迁移效用分开评估。', '',
              'audited由本会话assistant辅助编写和核查，不是独立强writer API盲测；不是严格上界。原样保留的旧条目未全面回查。详细核查范围见AUDIT_NOTES.md。', '',
              '## 产物', '',
              '- REPORT.md：预先定义的五组共同完成主表与逐配对结果。',
              '- comparison.json / analysis.json / behavior.json：主实验分数、缺失敏感性、动作及成本。',
              '- supplement_raw_actions/comparison.json：补充组完整配对及成本。',
              '- inputs、candidates、scores：输入、候选、逐题公开轨迹与模型回复。',
              '- integrity.json、input_hashes.json、candidate_hashes.json、scheduling：完整性和调度记录。', '']
    (root / 'INTERPRETATION.md').write_text('\n'.join(lines))
    report = root / 'REPORT.md'
    marker = '\n## 完整解读与补充诊断\n'
    body = report.read_text().split(marker)[0]
    report.write_text(body + marker + '\n[结果解读](INTERPRETATION.md)包含逐组有效配对、缺失敏感性及动作关联证据补充；[核查范围](AUDIT_NOTES.md)说明辅助核查组的边界。补充诊断为探索性结果，不替换上面的预先定义主表。\n')
    (root / 'finalize_script.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + '\n')
    print(json.dumps({'primary': analysis['summaries'], 'supplement': summaries}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
