# 从动作反馈学习跨任务技能：ALFWorld 小样本实验（2026-10-04 UTC）

后续测试期逐题更新技能表的独立诊断及阴性结果，见 [FEEDBACK_SKILL_ONLINE_20261004.md](FEEDBACK_SKILL_ONLINE_20261004.md)。

**结论：在预先选定的 6 个官方 `valid_unseen` 游戏、每题两个种子上，加载从独立训练轨迹学到的技能表为 12/12 成功，清空技能表为 5/12；配对 7 胜、0 负、5 平。** 官方环境逐步重放、提示哈希、输入内容绑定和奖励审计通过。这是清洗、冷却、加热三个题族的小样本结果，尚不是完整 ALFWorld 或 CLBench 的全面提升。

## 为什么要学这个

此前的状态推进 actor 在看见目标物品时会优先拿取，在看见目标容器时会尝试放置，却把“发出过 `clean/cool/heat`”当作“处理成功”。环境即使返回 `Nothing happens.`，它也可能继续放置尚未处理的物品。更根本的是，拿到物品后仍要让基模自己决定去哪种工具处；它可能直接把物品放回目标容器，在“取出—放回”的循环中耗尽步数。

这里把**动作的局部效果**作为经验单位。对训练轨迹中的动作 $a_t=\text{heat mug 1 with microwave 1}$，只在环境随后明确返回 `You heat the mug 1 using the microwave 1.` 时，才给 `(heat, microwave)` 一个正证据。`Nothing happens.` 是负证据；物品 ID 或工具 ID 不匹配也不能记为成功。一个任务的最终胜负不会直接给其中每步动作赋值。这是用环境反馈缩短信号跨度的信用分配方法。

以不同游戏输入内容哈希去重，令 $n^+_{v,u}$ 为操作 $v$ 在工具类型 $u$ 上得到明确成功反馈的**独立训练游戏数**，$n^-_{v,u}$ 为明确 `Nothing happens.` 的游戏数。当前实现仅在 $n^+_{v,u}\ge2$ 时启用技能，按正证据数优先、负证据数次之选工具。训练轨迹依次加入时，计数和技能选择可以逐任务增量更新；本次对照将预先构建的表冻结后才进入评价，评价轨迹不会回写。

从 32 个含准备动作反馈的不同官方训练游戏里，学到清洗→sinkbasin（16 个正证据游戏）、冷却→fridge（12 个）、加热→microwave（7 个）。`heat ... with stoveburner` 的训练反馈只有失败，因此没有激活这条技能。它是**从经历归纳的非参数技能表**，没有更新语言模型权重，也不是 MemRL 的 intent–experience–Q 记忆。

执行新任务时仍由冻结的原 actor 搜索物品；一旦公开反馈确认已拿到目标物品，技能表会从当前允许的命令中选择去对应工具的位置，必要时打开工具，再执行准备动作。只有环境明确确认准备成功，状态跟踪器才允许“已处理”的放置规则。其他步骤继续由原 actor 处理。空表对照保留相同的状态跟踪、模型、提示方式、种子和 50 步预算，仅删除跨任务技能内容。

例如目标是“把热 mug 放到 shelf”，先前训练游戏给出 `heat→microwave`。新任务里 `take mug 1 from table 1` 得到 `You pick up ...` 后，策略选择 `go to microwave 1`。若 `heat mug 1 with microwave 1` 返回 `Nothing happens.`，不会标记 mug 已加热；到达并打开微波炉后得到 `You heat ...`，才允许把 mug 放到 shelf。这里迁移的是“什么操作在什么工具上被验证有效”，不是照搬旧任务的完整轨迹。

## 冻结对照

训练来源仅为先前官方 `train` 轨迹；新注释目标逐条绑定游戏输入 SHA-256、来源 episode SHA-256、轨迹位置、原始动作与反馈 SHA-256，不能按旧 history ID 复用。评价题在结果可见前按官方路径 SHA-256 顺序选取，每个题族两题；`valid_unseen` 另外排除了此前状态推进实验使用的测试题。每题两个种子、每臂最多 50 个环境动作，同一冻结模型服务，无新训练。

| 数据 | 不同游戏 × 种子 | 空表成功 | 技能表成功 | 配对胜／负／平 | 空表／技能表无效动作 | 空表／技能表 actor 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方 `train` 开发对照 | 6 × 2 | 2/12 | **12/12** | 10／0／2 | 71／2 | 518／162 |
| 官方 `valid_unseen` 独立小样本 | 6 × 2 | 5/12 | **12/12** | 7／0／5 | 42／1 | 378／86 |

开发对照设计哈希为 `4166c38fc8fec429a49c701ee988adff4cb771b6ab161b3618c087b2a2303fdd`，测试小样本设计哈希为 `07f401dec3be43bbbc2b78a54bbb79a96998760ea49c6415ae5cf8f1c388501e`。训练设计与测试设计分别冻结了代码、计划、输入、审核、种子和预算。逐步审计报告在 ignored 的 `results/memrl_credit_training/20261004_feedback_skill_train_pilot_v2_audit.json` 与 `.../20261004_feedback_skill_unseen_small_v1_audit.json`。训练首次运行时输出目录漏了游戏哈希、实际只覆盖每族第一题；该废弃试跑不纳入表格，完整对照从修正后的 `...train_pilot_v2/` 重新运行。训练运行后，测试前只增加了两种公开目标句式／工具打开状态处理；训练运行的原始代码副本在 ignored 的 `...train_pilot_v2/frozen_runner.py`，哈希与设计一致。

这 12 个测试尝试来自 6 个游戏，两个种子并非 12 个独立任务。基模服务同种子也可能有波动；因此只能说当前小样本有显著的正向迹象，不能把 12/12 外推到全部 `valid_unseen`、三次尝试、其他 ALFWorld 题族或 CLBench。它也没有证明学习技能表优于所有手写 ALFWorld 规则；这个对照专门检验**先前任务的反馈经验是否比空表有帮助**。

实现位于 `feedback_skill_actor.py`，训练和测试官方回放审计分别是 `audit_feedback_skill_actor.py` 与 `audit_feedback_skill_unseen.py`，独立测试选择在 `probe_feedback_skill_unseen.py`。运行 `python3 -m unittest ttcl.memrl_comparison.test_feedback_skill_actor -v` 可检查反馈匹配、失败动作与空表行为。`models/`、`data/`、`results/` 不入 Git；新的服务器须按 `EXPERIMENTS.md` 第 4 节和依赖顺序恢复数据与环境，不能据此文档虚构缺失的原始结果。
