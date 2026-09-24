from __future__ import annotations
import copy
from pathlib import Path
import random
import shutil
import time
from ttcl.experience_v2.common import WORKSPACE, BENCH, MODEL, read, save, sha_file
from ttcl.experience_evolution.core import FAMILIES

ROOT=WORKSPACE/'ttcl/results/experience_feedback/20260923'
PRIOR=WORKSPACE/'ttcl/results/experience_v2/20260923'


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    names=['delta','absolute','continued_k1','improved_k2_s923','improved_k2_s924']
    origins={name: PRIOR/('adapters/'+name if name in {'delta','absolute'} else 'training/'+name+'/adapter') for name in names}
    for name, path in origins.items(): shutil.copytree(path,root/'adapters'/name)
    settings={'tasks':{'blind_spectrum_monitoring':90,'exploitable_poker':120,
                      'database_exploration':20,'cohort_studies':20},
              'repeats':[303,404], 'url':'http://127.0.0.1:18237',
              'arms':{'none':'frozen-actor','untrained':'frozen-actor',**{n:n for n in names}}}
    smoke=copy.deepcopy(settings); smoke.update(tasks={'exploitable_poker':2},repeats=[303])
    save(root/'plan.json', {'created_at':time.time(),'suites':{'smoke':smoke,'feedback_transfer':settings},
         'score_visibility':'Completed-episode official reward plus trajectory, before next episode only',
         'locomo':'disabled_by_user', 'writer_max_tokens':768,'writer_system':'unchanged',
         'actor_frozen':True,'writer_parameters_frozen_during_evaluation':True,
         'adapter_origins':{n:str(p) for n,p in origins.items()},
         'stages':[{'stage':1,'name':'existing_checkpoints_feedback_clbench','status':'prepared'},
                   {'stage':2,'name':'matched_absolute_delta','status':'prepared_not_started'},
                   {'stage':3,'name':'failure_mixed_curriculum','status':'prepared_not_started'},
                   {'stage':4,'name':'new_vs_previous_memory_reward','status':'optional_not_scheduled'}],
         'excluded_tasks':{'sales_prediction':'Docker unavailable in prior experiment',
                           'codebase_adaptation':'Docker unavailable in prior experiment'}})
    original=read(PRIOR/'training_plan.json')
    curriculum=read(PRIOR/'training/curriculum.json')
    arms={f'{objective}_k2_s{s}':{'seed':s,'paired_samples':2,'objective':objective}
          for s in [923,924] for objective in ['absolute','delta']}
    training=copy.deepcopy(original)
    training.update(training_arms=arms,initialization=str(root/'adapters/delta'),
        actor_url='http://127.0.0.1:18238',curriculum='training/curriculum.json',
        training=curriculum['training'],screening=[],calibration=[],
        comparison='Same old Delta initialization; only objective differs within each seed',
        budget_per_arm={'chains':48,'writer_actions':144,'optimizer_steps':24,
                        'paired_samples_per_memory':2,'actor_episodes':624},
        reward_baseline='same next task with EMPTY experience; recorded in BOTH objectives',
        failure_policy='Keep all source/continuation trajectories regardless of reward; continuation replica 0',
        stage=2)
    save(root/'training_plan.json',training)
    save(root/'training/curriculum.json',curriculum)
    # Stage 3 changes only the curriculum. Sampling uses existing training-split screening,
    # never CLBench or held-out rewards. A zero screening score does not guarantee future failure.
    rng=random.Random(923); sequences=[]; selection=[]
    for family in FAMILIES:
        items=[x for x in original['screening'] if x['family']==family]
        failed=[x for x in items if max(curriculum['scores'][x['game']])==0]
        succeeded=[x for x in items if max(curriculum['scores'][x['game']])>0]
        assert failed and succeeded, family
        for j in range(8):
            pool=failed if j%2==0 else succeeded
            first=rng.choice(pool)
            rest=rng.sample([x for x in items if x['game']!=first['game']],3)
            sequences.append({'id':f'failure_mix:{family}:{j}','family':family,
                              'games':[x['game'] for x in [first,*rest]],
                              'source_screening_group':'zero_in_two_rollouts' if j%2==0 else 'ever_success'})
        selection.append({'family':family,'zero_source_chains':4,'ever_success_source_chains':4})
    rng.shuffle(sequences)
    child=root/'failure_mix'; child.mkdir()
    mixed=copy.deepcopy(training); mixed.update(stage=3,training=sequences,
        comparison='Same initialization and budgets as stage 2; 50% sources from zero-success screening group')
    save(child/'training_plan.json',mixed)
    save(child/'training/curriculum.json',{'training':sequences,'selection':selection,'scores':curriculum['scores']})
    save(root/'next_stage_status.json',{'stage_2':'prepared_not_started','stage_3':'prepared_not_started',
                                      'stage_4':'optional_not_scheduled'})
    paths={g for plan in [training,mixed] for seq in plan['training']+plan['evaluation'] for g in seq['games']}
    assert all('/train/' in g for plan in [training,mixed] for seq in plan['training'] for g in seq['games'])
    save(root/'data_hashes.json',{g:sha_file(Path(training['data_root'])/g) for g in sorted(paths)})
    protocol='''# 下一轮实验：逐任务反馈驱动的经验更新

1. 先重测已有七组 checkpoint。四场景，采样种子 303/404，每组 500 个任务实例。
   所有组从空经验开始，每条任务独立重置环境，只有经验文档跨任务传递。
   交互结束后才读取该次官方 reward，连同完整记录轨迹及旧经验生成新经验。
   失败轨迹保留；中止且无官方 reward 的轨迹保留 null，不能当成 0。
   保持原 WRITER_SYSTEM、200 词要求、768 token 上限和 actor 解码设置。
   最后一题也生成经验以便审计，但不影响本序列已取得的分数。
   actor 和 writer 参数均冻结；无下一题、隐藏答案、评估器内部状态输入。
   这是反馈可见的适配协议，不宣称与无终局 reward 的协议相同。
   无经验组只作对照不更新经验。每个场景/种子各自维护经验，彼此隔离。
   先跑独立的两题扑克 smoke 并核查真实 prompt/经验传递，再运行完整评估。
   smoke 不并入正式成绩；旧评估部分结果不并入本轮。LoCoMo 不运行。

2. 准备 Absolute/Delta 配对训练，待第 1 步结果完成后执行。
   两目标均从同一个旧 Delta adapter 初始化，避免初始模型混淆；不是各自续训旧模型。
   复用 v2 固定 48 条四任务链、相同源任务和随机数规则；seed 923/924，各 K=2。
   每组 144 次经验生成、24 优化步、624 次 actor episode（48+144*2*2）。
   两组都执行 with/empty 分支：Absolute 使用 mean(R_with)，Delta 使用 mean(R_with-R_empty)。
   LR 5e-6，KL 0.01，无 reward 居中，所有经验均进入更新；零优势仍有 KL 项。
   后续在线轨迹会随各自模型变化，不承诺更新后的两组观测完全相同。
   actor 始终冻结；每次 writer 输入包含此前完整记录轨迹及 reward。

3. 准备失败混合课程对照，目录 failure_mix；在第 2 步后单独执行。
   每家族 4 条源任务来自训练筛选两次均失败的池，4 条来自至少成功一次的池。
   这规定任务采样比例，并不保证新轨迹的成功/失败比例。保留所有实际结果。
   同一初始化、目标对照和预算；将课程变化与目标变化分开比较。
   诊断记录源/后续 reward、经验修正及配对收益；失败内容不能按 reward=0 丢弃。

4. new-memory vs previous-memory 的奖励作为后续可选消融，暂不启动，
   不把现有 empty-memory Delta 悄悄替换为新定义。

训练阶段只准备并提供命令，本次自动调度仅启动第 1 步。
Reflexion/ExpeL 独立实验保持运行，其重试/数据划分不同，应单独报告。
'''
    (root/'PROTOCOL.md').write_text(protocol)
    target=root/'source/ttcl';target.mkdir(parents=True)
    (target/'__init__.py').write_text('')
    for name in ['experience_feedback','experience_v2','experience_evolution','common','llm_memory','structured_memory']:
        shutil.copytree(WORKSPACE/'ttcl'/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (child/'source').symlink_to(root/'source',target_is_directory=True)
    files=[p for d in [root/'source',root/'adapters'] for p in d.rglob('*') if p.is_file()]
    files += [root/x for x in ['plan.json','training_plan.json','training/curriculum.json','PROTOCOL.md','data_hashes.json',
                              'failure_mix/training_plan.json','failure_mix/training/curriculum.json']]
    files += list((BENCH/'src').rglob('*.py'))
    save(root/'input_hashes.json',{str(p):sha_file(p) for p in files})
    save(root/'status.json',{'phase':'prepared'})

if __name__=='__main__': prepare(ROOT)
