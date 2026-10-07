# 新 ALFWorld 游戏上的来源配对动作信用复核

此前 8 条自身成功历史 × 18 道训练游戏的终局回报矩阵中，10 道题随历史改变胜负；按首个分歧动作做固定获胜历史策略的干预后，只有 3/10 道不同游戏失去成功。随后同一批游戏新增 20 个来源配对，六个额外负效应仍集中在原来的三道游戏。详见[原报告](ALF_SOURCE_PAIR_CREDIT_20261007.md)。这不能给其他游戏的首个分歧动作贴负标签。

本轮使用[新增回报矩阵](ALF_ADDITIONAL_FUTURE_REWARD_20261007.md)：同一八条自身成功历史交叉 18 道**不同的官方 `train` 游戏**，全部 18 个无记忆回合与 144 个历史条件回合已在原环境独立重放、零失败。其中仅五道游戏随历史来源改变终局胜负。对这五道游戏，不看干预结果，按首个动作分歧位置、获胜来源 ID、失败来源 ID 排序，每题取排序第一的胜负来源配对。候选内容绑定了新游戏、两条完整回合、共同前缀、正负动作和检查点；独立环境重放前缀并确认两动作均可执行，**5/5** 通过。首个分歧发生在第 0、1 或 2 步。

对每个候选，先用获胜历史完整复现原冻结演员的胜利回合；重置同一游戏、重放共同前缀，仅在该位置强制使用失败历史的动作，此后继续使用**同一个获胜历史 LoRA 策略**。演员仍是 Qwen3-4B、50 步、每步最多 64 个新 token、两轮对话历史与两次重复抑制。五个原胜利回合均精确复现；五个强制动作回合全部完成，其命令、观察、终局回报再经独立原环境重放，零失败。

| 新游戏 ID | 家族 | 首个分歧步 | 替换动作后官方获胜 |
|---:|---|---:|---:|
| 0 | 查看物体 | 1 | 是 |
| 4 | 普通放置 | 1 | 是 |
| 8 | 清洗后放置 | 2 | 是 |
| 13 | 加热后放置 | 0 | **否** |
| 15 | 双物体放置 | 2 | 是 |

所以新增游戏中，**1/5** 的首个分歧动作在固定后续策略下有负向终局作用，**4/5** 替换后仍胜。与旧批次合计，是 **4/15 道不同游戏**发现这种局部作用，而非 10/35 个干预配对对应十道独立题。这里的动作作用只针对本来源、本演员、本预算的确定性继续执行；替换后仍胜也不能证明动作在其他后续策略下无害。

这为训练提供一个明确边界：整回合来源回报可以作为轨迹—目标题效用的监督，但不能把胜负差一律赋给首个不同动作。当前有条件的强动作标签仍仅四道独立游戏；其余游戏需要定位**后续真正改变胜负的决策**，或用完整回合奖励训练生成器并对来源置换、固定首条历史做消融。仅增加五道相似训练游戏没有把动作级强标签扩充到足以支持“已学会通用信用分配”的程度。本轮没有更新超网络权重，也没有验证在线持续 LoRA 更新或优于文本经验。

本机内容 SHA-256：候选标注 `f94e4116d374ee204a96858b6b069406f1ec73cb71c74f28c86ecc4703374e12`，共同前缀审核 `73580705bbcdbde6e8a07958027eaaf594d3271ad8636540383011a0ca7a3cc5`，干预回合 `777a9d763fd905966727dd99a9b841d7dc381a3f151b215ff10814ec17ce43e5`，独立完整回放 `492cf549216e9c0d847ff8d6a7f3aa35ef5dfddc57755e3b54f445f907d4ba47`。原始轨迹和标注留在 Git 忽略目录，仓库只上传审核入口和摘要。

复现前按 `EXPERIMENTS.md` 第 4 节恢复公开资产和旧检查点，并先完成新增 8×18 回报矩阵、合并与审核。在仓库根目录激活主线 Python 环境后，依次运行；前三步复用未改动的 v1 逻辑，但输出全是新文件，最后一步使用新增的五例审核脚本：

```bash
python -m ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v1 \
  --review data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json \
  --report results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json \
  --audit results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json \
  --output data/annotations/alf_source_pair_additional_first_action5_reviewed_20261007.json
python -m ttcl.trajectory_hyperlora.audit_alf_source_pair_action_v1 \
  --review data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json \
  --report results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json \
  --audit results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json \
  --candidates data/annotations/alf_source_pair_additional_first_action5_reviewed_20261007.json \
  --output results/trajectory_hyperlora/alf_source_pair_additional_first_action5_audited_20261007.json
python -m ttcl.trajectory_hyperlora.probe_alf_source_pair_action_v1 \
  --review data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json \
  --report results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json \
  --audit results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json \
  --candidates data/annotations/alf_source_pair_additional_first_action5_reviewed_20261007.json \
  --candidate-audit results/trajectory_hyperlora/alf_source_pair_additional_first_action5_audited_20261007.json \
  --output results/trajectory_hyperlora/alf_source_pair_additional_intervention5_v1_20261007.json \
  --max-cases 5 --device cuda:0 --gpu-fraction .65
python -m ttcl.trajectory_hyperlora.audit_alf_source_pair_intervention_v3
```

新游戏和动作都有新的内容绑定，不能复用旧 10 题的候选标注。末步默认从上述新路径读取并写入新审核结果；审阅文件及原始运行结果都位于仓库忽略目录。
