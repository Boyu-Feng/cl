# XLand 从空历史开始、模型自选试验动作的 LoRA 检验（2026-10-05）

## 问题与协议

此前[三机制实验](X_LAND_MULTI_MECHANISM_MEMRL_COMPARISON_20261005.md)在测试前给超网络三条固定的单步试验历史（动作依次为 `0,3,4`），其 Qwen LoRA 在 90 道留出规则的一步目标上得到 90/90 和 87/90。它验证“已有轨迹→LoRA→后续动作”，没有验证**一开始没有历史、模型自己探索、逐次积累反馈**。

本轮冻结那两个训练好的顺序不变超网络与 Qwen3-4B。在相同 30 组规则内容留出、每组三种隐藏机制、共 90 个实例上，每个实例都从空历史重新开始。第 0 轮不挂 LoRA；模型在**来源布局**自己选 `0/3/4` 中的一个动作，只给它所选动作对应的真实 XLand 单步状态转移、官方 reward 和 done；超网络用这条轨迹重新生成 LoRA。随后重复，最多三次来源试验。每轮在不同的**目标布局**读取当前 LoRA，预测“哪一个动作会在一步后产生目标 tile”。目标探测的答案及转移不写入经验。每次预测的模型输入都由当前实际可见的来源前缀和公开目标状态重新计算 SHA-256；没有把未选动作、机制名或审核答案放进模型输入。

这仍是先前搭建的**受控 XLand 一步任务**，不是官方完整任务的在线通关率。来源试验是在已独立审核的确定性 XLand 转移表中按模型当轮所选动作取得结果；表里另两个未选动作对模型不可见。试验中的 `reward` 是环境原样数值，而本轮主指标沿用先前“一步后目标 tile 是否出现”的动作判定。两者在 `near` 和 `tile_near` 机制下不一致，因此不能把下面的主指标写成官方 reward 改善。三次试验相当于同一隐藏规则下的独立单步回合，不是跨多个不同 ALFWorld/CLBench 任务的长期记忆。

新审核脚本重新从原 XLand 环境转移推导每个目标动作，检查来源动作的共同初态、规则划分和原审核谱系，为新的**空历史及动态前缀输入**单独建立目标清单；没有根据旧 history ID 继承标签。两个超网络 checkpoint 均仅在原训练规则上训练，本轮测试不更新超网络或 Qwen 权重。

## 结果

| 冻结 Qwen 超网络种子 | 空历史 | 自己试 1 次后 | 自己试 2 次后 | 自己试 3 次后 | 三次试验实际用过的不同动作数（平均） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 30/90 | 59/90 | 60/90 | **80/90** | 1.79/3 |
| 43 | 30/90 | 60/90 | **89/90** | **88/90** | 1.99/3 |

两种子的第 3 轮相对空历史分别增加 50 和 58 题，均无原本正确却变错的实例；种子 43 从第 2 到第 3 轮少对一题，说明“更多历史必然更好”也不成立。来源动作由当前冻结策略自己选择，没有固定为训练时的 `0,3,4`；它经常重复动作，仍能利用有限试验反馈。第 0 轮的三个隐藏机制共享同一公开目标，模型几乎都选动作 `4`，所以三选一恰为 30/90。

90 个实例来自 **30 组公开目标相同的三机制配对**，不是 90 条独立规则。第 3 轮种子 42 有 20 组全对、10 组对 2/3；种子 43 有 28 组全对、2 组对 2/3。两个种子也共享这 30 组测试规则。

另存的**官方 reward 数值总和**在第 0／3 轮分别是种子 42 的 `0／19.93`、种子 43 的 `0／29.89`（90 个目标转移各一条）。这不是三机制通用评分：环境的目标是“持有目标物体”，与 `near`、`tile_near` 的一步产物判定不一致，reward 主要反映 `hold` 分支。

按机制拆开，第 3 轮种子 42 的 `hold/near/tile_near` 分别为 **20/30、30/30、30/30**；种子 43 为 **30/30、28/30、30/30**。种子 42 对 `hold` 在第一次反馈后是 30/30，后续重复试验后退到 20/30。这提示目前的超网络会受到冗余历史干扰，不能只报最后总分。

为检验“用到了这段历史的环境反馈”，对每个实例保持模型实际试过的**动作序列**、来源与目标的公开初态、目标 tile、checkpoint 全部不变，只把这几个动作的来源转移换成另一隐藏机制下的转移，再生成 LoRA。第 3 轮结果：

| 种子 | 原反馈下正确 | 错机制反馈下正确 | 两种反馈造成预测变化 |
| --- | ---: | ---: | ---: |
| 42 | 80/90 | **0/90** | 90/90 |
| 43 | 88/90 | **0/90** | 90/90 |

这是强于“任意 LoRA 都能提高”的来源反事实：在本受控设置中，生成的参数确实随模型自己观察到的试验结果而改变，并影响后续动作。错机制反馈在现实中属于另一任务的反事实，诊断是在看过主结果后执行；它不增加独立测试任务数量，也不代表已在完整 XLand 或 ALFWorld 中持续学习。

## 边界与下一步

这版**没有重新训练超网络**，而是把原来用完整三次来源训练好的模型，直接放入从零历史的逐次使用协议。结果说明已有表示能处理部分长度和动作分布变化；但任务机制只有三类、候选动作也仅三种，布局模板受控，三种机制训练期都出现，测试只留出具体规则内容。来源试验实际上是同一隐藏规则的单步回合；跨新规则长期保留经验、真实多步回报和自动决定何时检索经验仍未验证。下一阶段需要用从零开始的训练序列优化超网络，并在更长、包含失败和无关任务的历史中检验负迁移及记忆选择。

本轮 XLand 权重和先前[ALFWorld 空历史试验](ALF_ONLINE_FROM_EMPTY_20261005.md)的权重互相独立。ALFWorld 超网络**已经用 ALFWorld 训练游戏**做过单条成功轨迹的下一题动作模仿，不能把其弱结果归因于“只在 XLand 训练”。ALFWorld 在线时输入却变成多条自身成功／失败轨迹，而且任务更长、动作文本更复杂；XLand 的受控三动作正结果不消除这些差别。

代码入口为 `xland_online_from_empty.py` 和 `diagnose_xland_online_feedback.py`。新审核清单 SHA-256 `7584abe332468ec3b870a93692b205caefedfccb69f27c6a09d298f52e7b6088`；两个 90 实例主结果 SHA-256 分别为 `db3ddd06dc8f41c6e084ab8cb8917c6201f9f6c0abca7cf9087686cb37ed4ef7`、`9c68cfee2691bbb578d8b1118dddcaf47a6fca21f6841c3392678b557105677d`；两份反馈反事实结果 SHA-256 分别为 `c1a572b61c3299195919964f675e9768cc3000970638557bd2f2cfff187c2c65`、`8920c46071436712ce489e9ac404eca5233f1de3e9d93603029a24c8ca64012a`。原始逐题记录与模型权重在 ignored `data/`、`results/`，Git 只保留代码和摘要。运行使用空闲 GPU 3，没有影响其他实验。

按 `EXPERIMENTS.md` 第 4 节准备主线 Python 和公开资产后，从仓库根目录执行；同名结果文件已存在时须换新路径，不覆盖冻结记录：

```bash
ttcl/.runtime/alf_delta_env/bin/python -m ttcl.trajectory_hyperlora.xland_online_from_empty prepare \
  --review data/annotations/xland_multi_online_from_empty_reviewed_20261005.json
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.xland_online_from_empty evaluate \
  --review data/annotations/xland_multi_online_from_empty_reviewed_20261005.json \
  --checkpoint results/trajectory_hyperlora/xland_multi_qwen_invariant_seed42_20261005.pt \
  --device cuda:0 --trials 3 \
  --output results/trajectory_hyperlora/xland_multi_online_seed42_20261005.json
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.diagnose_xland_online_feedback \
  --online-report results/trajectory_hyperlora/xland_multi_online_seed42_20261005.json \
  --review data/annotations/xland_multi_online_from_empty_reviewed_20261005.json \
  --checkpoint results/trajectory_hyperlora/xland_multi_qwen_invariant_seed42_20261005.pt \
  --device cuda:0 \
  --output results/trajectory_hyperlora/xland_multi_online_feedback_swap_seed42_20261005.json
```

种子 43 用对应的旧 checkpoint 与新结果路径执行同一协议。上述两个 checkpoint 是先前训练产物，不随 Git 上传；新服务器须先按前一报告重训，不能仅凭代码仓库复现原机数值。
