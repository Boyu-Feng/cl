# 多场景 Python 结构化经验

## 模型自主维护经验库（2026-09-20）

`launch_online_bank.py` / `online_bank.py` 是两个任务各12题的小测试：
无经验对照与模型经验库。每题结束后同一冻结模型读取本组完整公开操作轨迹、
工具反馈和最终官方scalar，自行决定KEEP/ADD/REVISE/REMOVE；下一题只读更新后的库。
两组按题号交替运行，答题采样与工具预算一致，总结额外调用单独计费记录。

```bash
/home/fengboyu/miniconda3/envs/seal_env/bin/python \
  ttcl/structured_memory/launch_online_bank.py \
  --root ttcl/results/structured_memory/llm_online_bank_NEW --gpus 0,1
```

实际提取prompt位于 `ttcl/llm_memory/trajectory_extraction_prompt.md`，
通用事务式更新逻辑位于 `ttcl/llm_memory/trajectory_bank.py`。
库最多8条/2048 tokens，程序只检查格式、引用和预算，不替模型选经验或核验语义。
每条含scope/lesson/application/limitations/evidence；未验证解释应标hypothesis。
一次格式重试仍失败则保持旧库。不完整执行的reward为null，不补0；可总结其可见部分。

结果目录的 `REPORT.md`、`BANKS.md` 自动刷新。每题的 `trajectory.json`、
`writer_requests.jsonl`、`writer_generations.jsonl`、`bank_update.json` 保留完整审计链。
最后一题后也总结。运行前快照代码和prompt；已启动实验不读取后续工作区改动。
每任务完成后审计银行时序、首题输入一致性、完整writer输入和reward一致性。
这是单种子开发pilot，测试自主总结加显式reward的整体效果，不单独隔离reward贡献。

## 核验经验与留出确认（2026-09-20）

`launch_verified.py` / `verified_experience.py` 将经验内容与收益分开检验。
用前6个实例的公开工具记录核验两条候选；第7–12个实例上比较无经验、
通用提示、无关内容、候选A、候选B和组合，共两个生成种子。
开发reward按运行前固定规则筛选，只有通过者进入未参与提取和选择的确认集，
对照使用三个新生成种子。官方环境seed保持42；生成种子不代表新数据。

```bash
/home/fengboyu/miniconda3/envs/seal_env/bin/python \
  ttcl/structured_memory/launch_verified.py \
  --root ttcl/results/structured_memory/verified_experience_NEW --gpus 0,3
```

可以先加 `--prepare` 冻结协议，再用同一结果目录的 `--start` 启动。
结果目录必须不存在（prepare），或仍处于prepared（start），禁止覆盖旧实验。
后台监督进程自动执行完整筛选和有条件的确认，刷新 `REPORT.md` 和
`DIAGNOSTICS.md`。证据、候选、源码及协议哈希见 `bank.json`、
`CANDIDATES.md`、`plan.json`、`source_hashes.json`，结束后再次审计。

这轮实验是人工核验经验的使用测试，经验在每题首次调用前可见且全程冻结。
不复用测试题工具结果，不训练writer，不宣称自动发现经验。所有非空上下文
以共享无关背景补齐到约768 tokens；无经验组没有该上下文。
筛选要求在每个种子下胜过三个对照，并通过去除任一实例的敏感性检查。
没有候选通过时保留负结果，不启动确认。数据库确认范围为第13–24题，
cohort为第13–20个实例；只有一个数据库和两个留出研究，不能据此声称全域稳定。

## 经验内容开发消融（2026-09-20）

新增 `launch_ablation.py`，主对照为逐题独立、原版经验和任务针对性经验；
数据库与扑克另加相同经验结构下的已完成实例 scalar reward 可见性对照。
本轮不训练参数或 writer，不额外生成候选获取更多评分。

```bash
/home/fengboyu/miniconda3/envs/seal_env/bin/python \
  ttcl/structured_memory/launch_ablation.py \
  --root ttcl/results/structured_memory/experience_ablation_20260920 --gpus 0,1,3
```

每个任务前 12 个 canonical 实例、seed 42，是已查看过的开发前缀。
`focused_memory.py` 实现紧凑数据库 schema/问题关联查询、保留研究范围的分析步骤、
以及跨下注轮次的对手加注响应统计。最后一类不把整局盈亏归因到末步动作，
终局没有展示的对手响应保持未知。所有提取只接收公开交互。

`focused` 和 `focused_reward` 均读取相同结构的历史行动案例；后者额外读取
对应已完成实例的官方标量，不读取评分器标签或 metadata。公开工具反馈本来就可能
包含正确性或盈亏，因此该对照只检验**显式标量的附加作用**，不等于完全无奖励信息
和有奖励信息的对照。不同题目的 reward 不作为动作优劣的配对标签。
队列研究本轮只运行三个主对照，不开放额外隐藏评分。

所有新实验组统一启用仅调整 JSON 包装的确定性修复，以及最多两次格式重试；
原始输出、修复类型、每次重试和 token 成本全部保留。题内上下文上限 65536 tokens，
不静默截断。该协议与旧 v2 有差异，因此无经验组重新运行，旧成绩只作背景。
各组失败互不阻断，未完成结果不填零、不报告完整配对收益。
运行源码会在启动时快照；后续工作区修改不影响已经启动的实验。

下方是此前原版两组实验的入口和协议。

将 BSM 的「公开交互 → Python 整理结构化经验 → 冻结 Qwen3 读取经验」对照扩展到其他五个任务。每个任务使用自己的公开数据格式；这套规则基线不声称模型自己学会总结。

| 任务 | 保存的经验 | 当前环境 |
|---|---|---|
| database_exploration | 表/字段结构、SQL 和结果样例、查询错误、公开纠错；标记 schema 变更 | 已完成官方数据库下载与校验 |
| cohort_studies | 研究范围、字段/量表编码、样本分布、分组生存率、队列组成与不可观测项 | 本地 SQLite 与隔离依赖可用 |
| exploitable_poker | 按公开对手区分动作、下注、回报和已摊牌信息；不把动作次数当完整概率 | 本地环境可用 |
| codebase_adaptation | 按公开仓库名区分路径、命令、测试诊断和提交反馈 | 待 Docker 权限及可选依赖 |
| sales_prediction | 公开表关联、历史销量、描述性趋势、已到期预测误差 | 待 Docker 权限及可选依赖 |

## 运行

```bash
# 默认三个可运行任务，队列在后台运行，GPU 0/1 各一个作业。
bash ttcl/structured_memory/run.sh \
  --root ttcl/results/structured_memory/python_multitask_20260920 \
  --gpus 0,1 --num-instances 12
```

结果目录必须不存在，防止覆盖已有实验。每次复制运行代码，保存命令、源码哈希和队列 PID。顶层 `RESULT.md` 每十秒刷新状态，完成后各任务目录也有独立 `RESULT.md`、`results.json`。失败保留日志，不把缺失样本当零分或忽略后继续宣称完整成绩。

隔离依赖位于 `ttcl/.runtime/structured_memory_deps`，不修改共享训练环境。包括 lifelines 0.30.0、texasholdem 0.11.0、autograd 1.8.0、formulaic 1.2.1、interface-meta 1.3.0、wrapt 1.17.3、Deprecated 1.2.18、autograd-gamma 0.5.0、narwhals 2.0.1。

## 对照

- `independent`：每个 canonical 样本新建官方任务，模型对话与经验均重置。
- `structured`：官方连续任务流程，每个样本重置对话，只继承提取的 Python 经验。
- 两组同一冻结模型、采样参数、按实例/轮数确定的随机种子、官方工具预算与评分器。保留完整题内交互；统一展示题号和任务简介。
- 记忆仅接受公开 query、action、observation.content。评分器 metadata、隐藏策略、未来标签不进入记忆或模型。规范实例 ID 仅用于内部配对/设种子；经验引用采用不含策略信息的别名。
- 工具结果按原任务范围解释。模型提出的答案、预测和猜测不会自动变成真实证据；公开验证结果可归档。
- 不新增候选生成或额外评分。调用数可随模型行为不同，在结果中报告。数据库 SQL 超时仍生效，本地生成耗时只记录，不按墙钟判负。

默认 12 样本、seed 42 是 pilot。数据库 multi_group；队列研究只覆盖前 3/5 阶段；扑克只覆盖同一对手。要评价泛化，需要后续完整序列与多种子。

`memory_contexts.jsonl` 保存每题实际读取的经验；`responses.jsonl` 保存完整模型输入和输出；`public_observations.jsonl` 保存公开工具反馈。经验上下文最多 16000 字符，提取器按完整记录裁剪并标记，题内历史不静默截断。超出总上下文或无法解析响应会标记失败，保留证据。

## 检查

```bash
PYTHONPATH="$PWD/ttcl/.runtime/structured_memory_deps:$PWD/current_work/continual-learning-bench" \
  /home/fengboyu/miniconda3/envs/seal_env/bin/python \
  -m unittest discover -s ttcl/structured_memory -t . -q
```
