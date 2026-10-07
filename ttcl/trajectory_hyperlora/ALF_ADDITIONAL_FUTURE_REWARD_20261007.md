# 扩充自身轨迹到 LoRA 的真实未来回报训练矩阵

先前八条自身成功历史交叉 18 道官方 ALFWorld 训练题时，10 道题的胜负随历史变化；不重叠的 12 题上有六道。但只凭这 18 道训练题拟合的冻结效用选择器，在后 12 题仅 **4/12**，与无 LoRA 及最佳固定来源相同。首个动作的同策略干预也显示，30 个来源配对里只有九个有效，而且集中在三道独立游戏。要训练历史条件化 LoRA，需要更多**不同目标游戏**的未来回报，而不只是同题重复配对。

`alfworld_source_utility_dataset_v3.py` 固定原八条自己完成且成功的来源轨迹、原 Qwen3-4B、原任务条件超网络和 50 步／每步 64 token／两轮历史／两次重复抑制协议。它从官方 `train` 计划的六类任务中各确定性选三道**新游戏**，排除源游戏、旧检查点 600 题候选、热启动题、先前 8×18 训练矩阵和 8×12 开发矩阵的游戏。任务家族仅用于平衡选题，不进入生成器或演员。准备阶段生成 18 个新目标题及 144 个来源—目标题的新内容哈希绑定；报告逐回合保存完整动作、环境反馈、失败记录与官方胜负，可在精确前缀检查后续跑。审核后确认这 18 道新游戏分布于 17 条计划序列，但**18 道均与旧矩阵共享序列 ID**；所以它们仅增加同类序列内的训练游戏，不是独立的序列泛化测试。

完整采集结束后，`audit_alfworld_source_utility_dataset_v3.py` 将逐条在原环境重放 18 个无记忆回合和 144 个历史条件回合，核验游戏内容、动作可执行性、每步反馈、终局奖励和全部输入绑定。`encode_alf_future_utility_features_v3.py` 只复用经来源哈希核对的八条冻结历史编码，再编码新题的初始上下文；不会用目标题答案、终局反馈或人工类别作为目标表示。`train_alf_future_utility_selector_v2_combined.py` 计划把原训练 18 题与新增 18 题合成 36 题，保持 v1 的冻结特征、四种预设变体和正则不变；**模型选择改为整条计划序列留一**。两批游戏合起来只有 19 条不同序列，同序列游戏不能互作独立留出。旧版的单题留一分数因此不能与新版序列留一分数直接作为同一指标比较。已使用过的 8×12 题只能作开发对照；将来若要报告方法增益，仍需冻结新演员和新官方题序，与固定首条、在线持续更新、无记忆和同源文本公平比较。

新增 18 题与 144 个历史条件回合已全部完成，零采集失败；**18 个基线回合和 144 个历史条件回合均由独立脚本在原 ALFWorld 环境逐步重放，命令、反馈、终局奖励与内容绑定全部通过，零重放失败**。这批题上无 LoRA **7/18**；八条自身成功历史的成功数依次为 **6、7、9、7、7、7、9、8/18**，最佳固定来源 **9/18**。只有 **5/18** 题的胜负随来源改变，事后逐题最优为 **10/18**。事后最优读取了答案，不能作为可部署方法成绩。

两批训练游戏合并后，在 19 条计划序列上整序列留一；四种预定冻结特征／目标变体中，最佳 `global_relative` 的跨折来源选择 **19/36**，各折仅从训练序列选出的最佳固定来源同为 **19/36**，无 LoRA **18/36**，事后逐题最优 **25/36**。重新拟合的冻结选择器在已用于旧选择器诊断的 12 道开发题上为 **4/12**，新训练集最佳固定来源 **5/12**，无 LoRA **4/12**，事后最优 **8/12**；旧 18 题选择器曾为 **4/12**。新模型在逐题选择上相对固定来源 **0 题独有成功、1 题独有失败**。新增配对数据没有带来可靠的来源选择收益，不能宣称轨迹条件化选择器已学会把经验配给下一题；12 题是重复使用的开发集，不是新盲测。此实验仅重训选择器，**未更新 LoRA 生成超网络**，更不是在线连续参数更新的独立贡献验证。

对这五道来源敏感的新游戏，另做[首个分歧动作的固定策略干预](ALF_ADDITIONAL_SOURCE_PAIR_CREDIT_20261007.md)：只有 **1/5** 替换后失去成功，其余 **4/5** 仍获胜；全部候选共同前缀和完整干预回合都经原环境独立重放。这限制了从终局胜负构造动作级训练标签的方式。

合并原始报告 SHA-256 `0124999642ba3c6e58706dd5b9b46f0d70efe824672398109f8de345f71bb6c5`，独立回放审核 SHA-256 `7238d7a4077fdf172c7252e695a2be6974a125a0e552914b970ae1e7d391f72a`，冻结选择器权重 SHA-256 `85b3c17469f17765d51471699af5c9671856c16e2f0bd2fa7eb4fed62276c66e`，开发结果 SHA-256 `e7674202f28bab456b66cd3a089409e9662dbd984d74182ec6234722bed05e88`。代码和摘要可上传，原始轨迹、审核 JSON 与权重留在仓库忽略路径。复现前必须先按 `EXPERIMENTS.md` 第 4 节恢复公开数据、旧检查点、自身来源回合和前两份回报矩阵，再按顺序运行下列入口。

```bash
python -m ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 prepare
python -m ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 collect
python -m ttcl.trajectory_hyperlora.audit_alfworld_source_utility_dataset_v3
python -m ttcl.trajectory_hyperlora.encode_alf_future_utility_features_v3
python -m ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v2_combined
```

`collect --resume` 仅用于报告已有完整、无失败、输入哈希一致的前缀；报告记录了失败就应先诊断，而非把失败当成功前缀跳过。新矩阵采集及效用评分器训练结束后，旧 8×12 的开发比较需显式传入已合并且审核的矩阵与审核文件，不能读取未合并的部分结果。

```bash
python -m ttcl.trajectory_hyperlora.evaluate_alf_future_utility_selector_v1 \
  --model results/trajectory_hyperlora/alf_future_utility_selector_combined36_v2_20261007.pt \
  --training-report results/trajectory_hyperlora/alf_future_utility_selector_combined36_v2_20261007.json \
  --matrix results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json \
  --audit results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json \
  --output results/trajectory_hyperlora/alf_future_utility_selector_combined36_holdout12_v2_20261007.json
```

本机采集加速时，在来源配对 **22/144** 条完整保存且零失败后，只中断本轮采集进程，保留该只读前缀。`alfworld_source_utility_shard_v3.py` 按配对索引把剩余区间分为 `22:64`、`64:104`、`104:144`，分别使用当时空闲的 GPU 0、1、3。每份报告绑定相同前缀文件、基础回合、审阅文件、检查点、输入顺序及演员预算；三份都完整且零失败，随后合并为新文件，并对**合并文件**完成独立原环境重放审核。原 22 条前缀、三个分片和合并文件均保留于 ignored 结果目录，不覆盖旧结果。

本机分片完成后的合并、审核和重训必须显式传入合并路径：

```bash
python -m ttcl.trajectory_hyperlora.alfworld_source_utility_shard_v3 merge \
  --shards \
  results/trajectory_hyperlora/alf_own_success_additional8x18_shard22_64_20261007.json \
  results/trajectory_hyperlora/alf_own_success_additional8x18_shard64_104_20261007.json \
  results/trajectory_hyperlora/alf_own_success_additional8x18_shard104_144_20261007.json
python -m ttcl.trajectory_hyperlora.audit_alfworld_source_utility_dataset_v3 \
  --output results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json \
  --audit-output results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json
python -m ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v2_combined \
  --new-matrix results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json \
  --new-audit results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json
```
