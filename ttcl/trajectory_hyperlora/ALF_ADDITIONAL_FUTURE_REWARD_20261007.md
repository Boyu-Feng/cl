# 扩充自身轨迹到 LoRA 的真实未来回报训练矩阵

先前八条自身成功历史交叉 18 道官方 ALFWorld 训练题时，10 道题的胜负随历史变化；不重叠的 12 题上有六道。但只凭这 18 道训练题拟合的冻结效用选择器，在后 12 题仅 **4/12**，与无 LoRA 及最佳固定来源相同。首个动作的同策略干预也显示，30 个来源配对里只有九个有效，而且集中在三道独立游戏。要训练历史条件化 LoRA，需要更多**不同目标游戏**的未来回报，而不只是同题重复配对。

`alfworld_source_utility_dataset_v3.py` 固定原八条自己完成且成功的来源轨迹、原 Qwen3-4B、原任务条件超网络和 50 步／每步 64 token／两轮历史／两次重复抑制协议。它从官方 `train` 计划的六类任务中各确定性选三道**新游戏**，排除源游戏、旧检查点 600 题候选、热启动题、先前 8×18 训练矩阵和 8×12 开发矩阵的游戏。任务家族仅用于平衡选题，不进入生成器或演员。准备阶段生成 18 个新目标题及 144 个来源—目标题的新内容哈希绑定；报告逐回合保存完整动作、环境反馈、失败记录与官方胜负，可在精确前缀检查后续跑。审核后确认这 18 道新游戏分布于 17 条计划序列，但**18 道均与旧矩阵共享序列 ID**；所以它们仅增加同类序列内的训练游戏，不是独立的序列泛化测试。

完整采集结束后，`audit_alfworld_source_utility_dataset_v3.py` 将逐条在原环境重放 18 个无记忆回合和 144 个历史条件回合，核验游戏内容、动作可执行性、每步反馈、终局奖励和全部输入绑定。`encode_alf_future_utility_features_v3.py` 只复用经来源哈希核对的八条冻结历史编码，再编码新题的初始上下文；不会用目标题答案、终局反馈或人工类别作为目标表示。`train_alf_future_utility_selector_v2_combined.py` 计划把原训练 18 题与新增 18 题合成 36 题，**保持 v1 的特征、四种预设变体、正则和按整题留一选择方式不变**，以便把结果变化主要归因于数据量。已使用过的 8×12 题只能作开发对照；将来若要报告方法增益，仍需冻结新演员和新官方题序，与固定首条、在线持续更新、无记忆和同源文本公平比较。

截至本文件创建时，新矩阵**已审核输入并开始采集**，但 144 个历史条件回合尚未全部完成；因此这里不填写来源胜率、选择器收益或投稿级结论。代码和摘要可上传，原始轨迹、审核 JSON 与权重留在仓库忽略路径。复现前必须先按 `EXPERIMENTS.md` 第 4 节恢复公开数据、旧检查点、自身来源回合和前两份回报矩阵，再按顺序运行下列入口。

```bash
python -m ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 prepare
python -m ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 collect
python -m ttcl.trajectory_hyperlora.audit_alfworld_source_utility_dataset_v3
python -m ttcl.trajectory_hyperlora.encode_alf_future_utility_features_v3
python -m ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v2_combined
```

`collect --resume` 仅用于报告已有完整、无失败、输入哈希一致的前缀；报告记录了失败就应先诊断，而非把失败当成功前缀跳过。新矩阵采集及效用评分器训练结束后，旧 8×12 的开发比较需显式传入已合并且审核的矩阵与审核文件，不能读取未合并的部分结果。
