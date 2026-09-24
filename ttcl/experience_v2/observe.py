"""Read-only experiment observer; writes a consolidated status/report, never trains."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from .common import ROOT, read, save


def optional(path):
    return read(path) if path.exists() else {}


def summarize(root):
    plan=read(root/'plan.json')
    state={'supervisor':optional(root/'status.json'),
           'training':optional(root/'training_pipeline_status.json'),
           'screening':optional(root/'training/screen_status.json'),
           'updated_at':datetime.now(timezone.utc).isoformat(), 'suites':{}, 'trained_arms':{}}
    lines=['# 实验进度与结果', '', f"更新时间：{state['updated_at']}", '',
           'LoCoMo 仅评估固定 300 条（每段对话 30 条，按题型比例抽样）。用于辅助迁移诊断；',
           '原生成器侧重任务经验、约 200 词，不应把此设置当作长对话事实记忆方法的公平容量排名。',
           '训练仅使用 ALFWorld train；测试集与模型选择分离，所有预先指定候选均报告。', '',
           '## 训练', '', f"当前阶段：{state['training'].get('phase','pending')}",
           f"任务筛选：{state['screening'].get('completed',0)}/288 个环境回合。", '',
           '| 模型 | 状态 | 已更新批次 | 正 / 负 / 零奖励动作 |', '|---|---|---:|---|']
    training_plan=read(root/'training_plan.json')
    for arm in training_plan['training_arms']:
        s=optional(root/'training'/arm/'status.json'); logs=root/'training'/arm/'training.jsonl'
        records=[json.loads(l) for l in logs.read_text().splitlines()] if logs.exists() else []
        counts=[sum(r[k] for r in records) for k in ['positive','negative','zero']]
        state['trained_arms'][arm]={'status':s,'batches':len(records),'signed_counts':counts}
        lines.append(f"| {arm} | {s.get('phase','pending')} | {len(records)}/12 | {' / '.join(map(str,counts))} |")
    for suite,settings in plan['suites'].items():
        status=optional(root/f'{suite}_status.json')
        cells=list((root/suite/'clbench').glob('*/*/*/episode_*/row.json'))
        cells=[p for p in cells if '.interrupted_' not in str(p)]
        qa=list((root/suite/'locomo').glob('*/*/qa/*.json'))
        expected=sum(settings['tasks'].values())*len(settings['repeats'])*len(settings['arms'])
        expected_qa=plan['locomo_questions']*(len(settings['arms'])+(not settings.get('inherit_suite')))
        locomo_cancelled=suite in optional(root/'schedule_override.json').get('skip_locomo_suites',[])
        if locomo_cancelled:expected_qa=0
        locomo_progress='LoCoMo 已按用户要求取消' if locomo_cancelled else f'LoCoMo 条件问答 {len(qa)}/{expected_qa}'
        writers=[read(p) for p in (root/suite/'clbench').glob('*/*/*/episode_*/writer.json')
                 if '.interrupted_' not in str(p)]
        writers += [read(p) for p in (root/suite/'locomo').glob('*/*/session_*.json')]
        state['suites'][suite]={'status':status,'clbench':len(cells),'expected_clbench':expected,
                               'qa':len(qa),'expected_qa':expected_qa,'locomo_cancelled':locomo_cancelled,
                               'writer_calls':len(writers),'writer_length_stops':sum(w['finish_reason']=='length' for w in writers)}
        lines += ['', f'## {suite}', '',f"阶段：{status.get('phase','pending')}；CLBench {len(cells)}/{expected}；{locomo_progress}。",
                  f"已记录 {len(writers)} 次经验更新，其中 {sum(w['finish_reason']=='length' for w in writers)} 次达到输出长度上限。",
                  f'详细结果：[REPORT.md]({suite}/REPORT.md)；失败日志：`{suite}/failures/`。']
    results={}
    for arm in ['none','untrained','delta','absolute',*training_plan['training_arms']]:
        path=root/'alfworld_evaluation'/arm/'scores.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        results[arm]={(r['sequence'],r['repeat'],r['position']):r['reward'] for r in rows if r['position']>0}
    lines += ['', '## 新 ALFWorld 独立测试', '',
              '54 个此前未用于本项目评估的 valid_unseen 实例，18 个三题序列，2 次采样；主指标只计第 2、3 题。',
              '| 模型 | 已记录 / 72 | 后续任务成功率 | 与原 Delta 共同配对均值差 |', '|---|---:|---:|---:|']
    comparison={}
    for arm,values in results.items():
        keys=sorted(set(values)&set(results['delta']))
        differences=[values[k]-results['delta'][k] for k in keys]
        entry={'completed':len(values),'expected':72,
               'success_rate':statistics.mean(values.values()) if values else None,
               'paired_delta':statistics.mean(differences) if differences else None,
               'paired_count':len(keys)}
        if len(keys)==72:
            import numpy as np
            clusters=defaultdict(list)
            for k,d in zip(keys,differences):clusters[k[0]].append(d)
            means=np.array([statistics.mean(v) for v in clusters.values()])
            bootstrap=np.random.default_rng(923).choice(means,size=(10000,len(means)),replace=True).mean(axis=1)
            entry['sequence_bootstrap_95_ci']=np.quantile(bootstrap,[.025,.975]).tolist()
        comparison[arm]=entry
        lines.append(f"| {arm} | {len(values)}/72 | {entry['success_rate']} | {entry['paired_delta']} |")
    state['alfworld']=comparison
    lines += ['', '## 尚未执行的场景', '',
              '销售预测、代码修复需要 Docker。当前用户无法访问 Docker socket，免密码 sudo 也不可用。',
              '它们没有被计为零分，也未用替代环境冒充完成。', '',
              '完整方案：[PROTOCOL.md](PROTOCOL.md)。既有模型和旧实验目录保持原样。']
    save(root/'overview.json',state)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        state=summarize(a.root)
        if not a.watch or state['supervisor'].get('phase') in {'complete','failed'}:break
        time.sleep(30)


if __name__=='__main__':main()
