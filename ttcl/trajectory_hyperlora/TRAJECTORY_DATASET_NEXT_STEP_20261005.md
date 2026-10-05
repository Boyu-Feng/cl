# 轨迹→参数经验：公开数据集选择与下一轮验证（2026-10-05）

后续的原始轨迹到 LoRA 受控试验与结果见 [XLand 超网络验证](RAW_TRAJECTORY_HYPERLORA_XLAND_20261005.md)。

## 先修正本地证据

旧无语义槽 HyperLoRA 在四标记奇校验合成测试上正确来源 96/128、错误来源 32/128，但逐来源审计发现每段历史的四个标记都收到同一个动作；96/128 正好是历史多数动作基线。其 LoRA 能保存和重载轨迹相关的**全局动作偏置**，尚未证明按条件使用经验。原始结果及哈希见 [上一轮审计](GENERAL_RELATION_HYPERLORA_20261005.md)。

本轮冻结旧权重，只读测试了每段历史 `LEFT/RIGHT` 数量严格相等的六种四条件规则（每个模型种子共 96 个查询，每条规则有两个来源种子；没有再训练）。训练时用过的四个标记词下，种子 42 的无 LoRA／正确／错来源分别是 **48/62/38**，种子 43 是 **48/48/48**。将来源和问题中的标记一致换成训练未见的四个词后，种子 42 是 **48/50/53**，种子 43 是 **48/48/47**。训练原标记的一次 62/96 只是已见规则组合上的不稳定迹象；在新标记和第二个种子上没有迁移。所有这 96 题都在自建场景，**没有新的 ALFWorld 或 CLBench 分数**。

原始 JSON 保留在 ignored `results/trajectory_hyperlora/pretrained_relation_lora_20261005/`，文件名 `seed{42,43}_balanced_{original,new}_cues.json`。按这个文件顺序的 SHA-256 为：`04e7b6559eb1f3ecc3d3bb7225c89892321ad86970a5e1e3d96d2c09089304c5`、`5949d3d64f4896fe1e94c5a0a24fe3d76dd77061e8aa3bf9e38a29354bc122f9`、`e1152a5f7607fc437db5d179199b1f04836b4e7f2db0a5cb627a5718514be80d`、`6e8da3de2b34ba861b5e488ef71d7190c1209b5270127f8aa1bee89cb3984f24`。旧 `probe_variable_relation_count.py` 默认的三、五条件探针协议没有变；新增 `--cue-counts 4` 与 `--original-cues` 可复跑本项。

## 哪些公开数据集真的匹配

| 候选 | 对“从轨迹学隐含规则”的贴合度 | 数据与工程代价 | 本项目用途 |
| --- | --- | --- | --- |
| [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) + [XLand-100B / Trivial-20B](https://github.com/dunnolab/xland-minigrid-datasets) | **最高**。规则和目标可组合；同一隐藏 ruleset 下可以先交互、再测试后续 episode。公开离线数据按任务保存完整 `states/actions/rewards/dones` 学习历史，另有供特定方法使用的 `expert_actions`。 | 环境可在 CPU 上生成；公开 Trivial-20B 约 60 GB、100B 约 325 GB。先从环境在线生成少量可审核轨迹，不预先下载大包。观察是压缩的 5×5 网格，需给文本 actor 做一致的符号化或先用小策略网络。 | **首选中间台阶**：在真正会变的环境规则上训练和测试轨迹→LoRA，而不是只换单词。 |
| [DeepMind Alchemy](https://github.com/google-deepmind/dm_alchemy) | 很高。潜在化学规律需要试验、反馈、后续决策。 | 官方 Unity 发行版依赖 Docker，仓库已归档；本机过去还有容器权限问题。符号实现的维护和 Python 兼容性需额外确认。 | 第二候选；暂不让容器问题挡住首轮验证。 |
| [Meta-World ML10/ML45](https://github.com/Farama-Foundation/Metaworld) | 高。元学习协议要求从适应回合改进同任务后续表现；ML10/45 留出新任务。 | 连续控制、MuJoCo；若用 Qwen 文本 actor，需要动作离散化或另建小策略底座。 | 参数经验方向的后续连续控制对照。 |
| [ACRE](https://github.com/WellyZhang/ACRE) | 中等。少量证据推断隐藏因果机制，能测试是否只学相关性。 | 主要是视觉场景和查询，缺少原生 agent 动作—反馈轨迹。 | 可作为关系归纳的辅助探针，不能代替轨迹实验。 |
| 本仓库已有 ALFWorld | 对最终文本 agent 最直接；有真实动作、反馈与官方 `won`。 | 多数下一题最佳动作从当前任务就能决定；先前正确／错来源 LoRA 与公共 LoRA 多次同分，源特异信号稀疏。[现有实测](ALF_NEXT_TASK_HYPERLORA_20261005.md) 和 [环境奖励策略](ALF_REWARD_POLICY_20261005.md) 都未证明来源特异增益。 | 在中间台阶通过后做真实文本任务验证，沿用已审核来源与官方分割。 |

以上是**适配程度判断**，不是新数据集的模型成绩。XLand 完整离线历史很适合来源轨迹与未来 episode 配对，但其 `expert_actions` 是最终策略重新标注的辅助字段，不能混进测试来源或当作环境在当时给出的反馈。先用公开环境采样和真实 reward 建立干净协议，节省下载与存储，并避免把离线教师标签当成在线经验。

## 下一步如何逼模型学习经验

1. **反捷径的数据协议。** 同一来源内保持动作频率平衡；构造两段只交换局部规则、但总动作频率相同的轨迹。当前“把四条规则全部取反”的配对会让全局多数动作翻转，所以不足以逼模型学条件关系。新配对要让某个条件的正确动作改变、其余条件保持，检验模型只在被改条件上改变决策。XLand 中对应成对 ruleset、相同布局和初始随机种子，隐藏规则不进入 actor 输入。
2. **无预设槽位的学习器。** 把每一步 `(观察, 动作, 下一观察, reward, done)` 编成可变长的键值事件，使用共享编码器与注意力聚合，而不是按物体名或规则名开固定参数槽。超网络输出冻结 actor 的低秩 LoRA；测试时只挂 LoRA，目标 episode 不再给原轨迹。可先训练一个查询条件轻量读出头作表示诊断，再训练 LoRA；两者都要通过反捷径评估。
3. **训练信号。** 先用训练划分里未来 episode 的正确动作或优势做有证据的监督预热；对同一目标状态比较正确来源、局部错配来源及公共来源。优化 `L_future + λ L_counterfactual + μ L_unchanged`：既要正确来源提高目标动作概率，也要只在被交换的条件上对错来源产生差异，在未变条件上保持输出。若接环境 RL，用相同目标种子和预算的下一回合奖励差更新**来源编码器与 LoRA 生成器**，并加负迁移惩罚；不能只训练一个选择公共／来源 LoRA 的门控后宣称超网络学到了经验。
4. **正式判据。** 预先固定 ruleset 级训练／开发／测试划分与来源哈希；报告实际后续回合回报、按条件准确率、来源特异的正确－错配和正确－公共增益、同一来源内的动作多样性、负迁移数，以及文本经验／完整历史 ICL 对照。特别检查整体动作频率是否能解释收益。开发集调参后测试只运行一次；测试轨迹不得变成训练来源。

工程上先用 XLand 的公开小型**规则配置文件**在线生成 episode，不下载 60 GB 的 Trivial-20B 轨迹包。实查发现 `trivial-21k` 的 21,000 个任务配置只有 **1 种生产规则数组**，因此它仅适合接口预检；真正的规则变化试验改用 `small-1m`，不能凭 21,000 个配置误称规则泛化。保持 XLand 独立 Python 环境，不改现有 ALFWorld/CLBench 环境，也不占用其他 GPU 实验。

## 本轮实际接通的最小环境

在 ignored `ttcl/.runtime/xland_env` 中独立安装了 Python 3.12 CPU 依赖，完整 pin 见 `config/environments/requirements-xland-minigrid.txt`。下载的公开 `trivial-21k` 规则文件约 135 KB，SHA-256 `36b11a3aaec1d00d768283644d978a93897d3bf93f15415dcfa1b1c083878086`；随后下载 **14.2 MB 的 `small-1m` 规则文件**（`small_1m_v3`），SHA-256 `c9c8ec0ec06d6337481e07c13872cd89df201e787093fc718e7ec368910195bf`。均存放于 ignored `data/xland_minigrid/`，没有下载完整轨迹包。

对 `small-1m` 实查有 1,000,000 个配置、**872,409 种不同的生产规则数组**；环境 `XLand-MiniGrid-R1-9x9` 的离散动作数是 6、观察张量为 `(5,5,2)`。`xland_protocol_smoke.py` 对预选训练规则 ID 278035、开发规则 ID 391646 各跑来源／目标两段 5 步随机动作；同 ruleset、同重置种子和同动作序列的来源重放逐步哈希一致。它**只是环境和轨迹格式预检**，没有策略训练、成功率或 LoRA 效果。

`prepare_xland_ruleset_split.py` 先按**生产规则数组**而非完整任务配置内容去重，再用固定种子与 SHA-256 排序选出训练 64、开发 16、测试 16 个互斥规则。这样同一隐藏规则只换目标或初始物体，不会跨集合混入。ignored 冻结清单为 `results/trajectory_hyperlora/xland_small_rule_split_20261005.json`，SHA-256 `2bd152e585e1a3bc3065166fab548fba2381228cc9b5c86418860fe9887098e1`；其中同时保存规则文件 SHA、所选规则 ID 和逐条规则／完整配置摘要。忽略上传的 `xland_small_smoke_20261005.json` SHA-256 为 `e89345eb20ef25da7751f93b4985c8eb34a40461f0f65039dc5b5a536fbbb391`，只是上述两条规则的随机轨迹预检，**不属于模型训练集**。

复跑时先建立独立环境并安装上述 pin，然后执行：

```bash
uv venv --python /path/to/python3.12 ttcl/.runtime/xland_env
uv pip install --python ttcl/.runtime/xland_env/bin/python -r config/environments/requirements-xland-minigrid.txt
ttcl/.runtime/xland_env/bin/python -m ttcl.trajectory_hyperlora.xland_protocol_smoke \
  --benchmark small-1m --rule-ids 278035 391646 \
  --output results/trajectory_hyperlora/xland_small_smoke_fresh.json
ttcl/.runtime/xland_env/bin/python -m ttcl.trajectory_hyperlora.prepare_xland_ruleset_split \
  --output results/trajectory_hyperlora/xland_small_rule_split_fresh.json
```

下一轮真正采集训练轨迹时，必须从这份冻结清单读取 ruleset ID；同一规则的来源／目标随机种子仍需固定并记录。随机探索轨迹不具有“正确经验”的含义，不能直接当作监督标签。现在的规则划分只确保**生产规则数组不重复**；是否存在功能等价但字节不同的规则仍需通过规则变换和回合行为进一步排查。
