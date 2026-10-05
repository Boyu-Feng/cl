# 无预设语义槽的轨迹→LoRA：关系预训练、配对训练与纠错反馈

## 保存的上一版与本轮结论

四标记分槽版已固定在 Git 标签 `trajectory-slot-lora-v1-20261005`（提交 `3c847a8`）。四份本地 checkpoint 另存于 ignored 的 `models/trajectory_hyperlora/slot_lora_v1_20261005/`，逐文件 SHA-256 见同目录 `manifest.json`；其 manifest SHA-256 为 `75ec55d3b420860e6d0e16f5ff034a9d611befd863954d4e70ea4c1c6fb658d5`。Git 上传代码与结果摘要，**不上传模型、权重、原始逐题记录或审核数据**。

用户指出固定 `ALPHA/BETA/GAMMA/DELTA` 四个参数槽不可能是通用方法。本轮新模型的编码器、关系表示和 LoRA 生成器**不含按标记命名的参数槽，也没有逐规则专家库**。它从原始轨迹每步的观察、动作和反馈提取固定维度关系矩阵，再直接生成冻结 Qwen3-4B 最后两层的 rank-8 LoRA `B` 因子；`A` 因子在训练规则间共享。标记列表只属于合成环境的数据生成与评分，不是模型结构。

最终探索版本在未见完整四关系组合的 128 道后续题上，两个种子均为：无 LoRA **56/128**、公共 LoRA **48/128**、正确轨迹生成 LoRA **96/128**、相反轨迹生成 LoRA **32/128**。同一道题换成相反历史，两个种子的首动作均在 **128/128** 次比较中改变。一条未见来源生成的标准 PEFT LoRA 独立进程重载后，16 道查询与进程内自定义挂载的首动作 **16/16 相同**，其中正确 **12/16**。这证明参数被实际生成并可脱离原历史提示挂载；不是只把轨迹附在提示里。

## 结构与训练信号

对完成轨迹 `τ={(o_t,a_t,f_t)}`，冻结 Qwen 的 token embedding 后，用小投影得到观察向量 `u_t`、动作向量 `v_t`。反馈门控 `g(f_t)` 控制这一步是否应写入关系：

\[
R(\tau)=\frac{\sum_t g(f_t)u_tv_t^\top}{\sum_t g(f_t)},\qquad
z=P(\operatorname{vec}R),\qquad B_\ell=H_\ell(z),\quad
W'_\ell=W_\ell+\frac{\alpha}{r}B_\ell A_\ell.
\]

第一阶段只训练轻量关系表示和查询读出头：给新任务句子 `q`，用 `q^\top R` 预测后续动作。**没有**给编码器逐步“哪个标记／哪个动作”的人工标签；监督是训练规则中的下一题正确动作。第二阶段冻结已学的关系表示与 Qwen，只优化投影器、共享 `A` 和直接生成 `B` 的头。训练题的同一问题配上动作规律相反但观察和反馈相同的两段历史，对两者分别监督正确动作，并加对数概率差的配对约束。它是后续动作的监督学习和配对训练，**还不是环境终局奖励 RL**。

合成协议沿用固定的八种偶校验训练规则；八种奇校验完整规则不进入训练，开发和测试各四种。测试中的来源轨迹、查询措辞和读数与训练不同。每种测试规则 × 两段新来源 × 四标记 × 四读数共 128 个严格首动作 token 判断。原始结果保存来源、相反来源和查询内容 SHA-256，Qwen 推理提示不含历史文本。

## 为什么先预训练关系

直接端到端训练没有成功。种子 42 的原始步骤简单平均版本，测试正确来源和错来源均为 **80/128**，两来源 128 道题的首动作完全相同；直接加入同题相反历史的配对损失后，正确来源 **58/128**，错来源反而 **88/128**。只读线性探针在其潜变量上对未见组合的四位关系只能读出 **32/64**，说明问题首先在轨迹表示，而不仅是 LoRA 头。

改用观察—动作外积关系矩阵进行下一题预训练后，两个种子的轻量读出器在测试上均为正确来源 **96/128**、错来源 **32/128**，同题换历史 128/128 次动作改变。把这一表示冻结，再直接训练 LoRA 头：普通下一题交叉熵仍塌缩为正确／错来源各 **80/128**；降低头部学习率并使用同题相反历史的配对损失后，两个种子的挂载 LoRA 均恢复到 **96/128 对 32/128**。这些消融说明**关系表示预训练与参数使用的配对目标缺一不可**，但结果是在多次查看同一合成测试集合后的探索发现，不是独立确认。

## 失败轨迹的信用分配

未使用反馈的模型遇到“每次成功动作之前，先做相反动作并收到无效反馈”的历史时，种子 42 的轻量读出从 96/128 降到 80/128，LoRA 正确／错来源均降到 **48/128**。最初只用 `success/rejected` 两个反馈词训练门控，也无法迁移到 `confirmed correct/invalid attempt`：门控给前者约 0.033、后者约 0.029 的权重，几乎丢弃全部步骤。

反馈门控最终采用**冻结已经能读干净轨迹的关系核心，只训练反馈门控**的分阶段方案。训练轨迹随机混合干净与拒绝后纠正过程，正负反馈各用多个表达；门控没有单独的有效／无效步骤标签，仍只从下一题动作损失学习。两个种子的轻量读出，在干净、纠错及另一组反馈表达上都为正确来源 **96/128**、错来源 **32/128**。重新用混合轨迹训练 LoRA 生成头后，两种子在标准纠错测试上也都为 **96/128 对 32/128**；额外反馈表达下分别为 **96/128 对 32/128**、**93/128 对 42/128**。作为失败对照，不冻结关系核心的多措辞训练中，种子 43 仍塌缩到两来源各 48/128。

这里的“信用分配”只涉及同一状态下拒绝动作与随后纠正动作的步骤权重；没有证明真实 ALFWorld 的长程终局奖励归因。反馈表达的扩充是看到首次措辞迁移失败后做的探索修复，额外表达也在开发中查看过，不应当作完全独立的新 benchmark。

## 当前限制与下一步

本任务仍固定四个合成标记、`LEFT/RIGHT` 二选一动作、每条轨迹每类三次示范，以及短程公开反馈。模型结构没有语义槽，但训练分布仍很窄；不具备任意新动作或任务的开放式技能学习证据。开发集的短提示词让 LoRA 各组按严格首动作 token 判分全为 **0/128**，提示鲁棒性未解决。多轮实验使用并查看同一奇校验测试规则，因此表中正结果是**机制探索**，不是冻结的最终确认成绩。ALFWorld、CLBench 和在线环境奖励 RL 均尚未对本新版本运行。

提交前另做只读词汇迁移探针：冻结两个反馈感知模型，把来源轨迹与目标问题里的四个标记一致替换为训练未出现的 `OMEGA/SIGMA/THETA/KAPPA`，不更新参数。干净来源及“无效→纠正”来源在两个种子上都得到无 LoRA **64/128**、正确来源 **96/128**、错来源 **32/128**；同题换历史 128/128 次首动作改变。该探针说明模型没有只能识别原四个标记字符串，但仍是相同的四条件二动作结构、相同的规则组合与评分模板；新标记由研究过程中选定，不是预先冻结的新任务族确认。

下一步应在新的任务族中预先固定提示、轨迹长度、动作词汇和整套隐藏规则划分，保留正确／错／公共／无参数四臂及文本轨迹对照。只有新族多种子仍能显示正确来源的独有收益，才在已审核 ALFWorld `train` 来源—目标配对上训练反馈门控和低维 LoRA 残差；正式测试保持官方环境 `won`、任务分割及冻结预算。针对 ALFWorld 51 道训练配对只有 7 道三臂终局奖励不同这一稀疏性，应先用不泄漏目标答案的过程反馈预热，再以真实下一题增益和负迁移惩罚微调，而不是直接对高维生成头做高方差终局 REINFORCE。

代码为 `general_relation_hyperlora.py`、`generic_relation_pretrain.py`、`pretrained_relation_hyperlora.py`、`feedback_relation_pretrain.py`、`diagnose_generic_relation.py`、`probe_generic_corrections.py`、`probe_unseen_marker_vocab.py` 和 `export_pretrained_relation_lora.py`。本机 ignored 原始记录与权重在 `results/trajectory_hyperlora/{general_relation_20261005,generic_relation_pretrain_20261005,pretrained_relation_lora_20261005,feedback_relation_20261005}/`；实验只使用空闲 GPU 3，单进程显存上限 35%，没有停止或修改其他卡上的实验。

复跑顺序以种子 42 为例，需先安装 `EXPERIMENTS.md` §4 的公开依赖，并为 `RUN_DIR` 选未使用过的 ignored 结果目录；程序拒绝覆盖既有产物：

```bash
RUN_DIR="$PWD/results/trajectory_hyperlora/reproduction_new_run"
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/alf_delta_env/bin/python -m ttcl.trajectory_hyperlora.generic_relation_pretrain \
  --device cuda:0 --gpu-fraction 0.35 --seed 42 --steps 800 \
  --output "$RUN_DIR/clean.json" --checkpoint "$RUN_DIR/clean.pt"
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/alf_delta_env/bin/python -m ttcl.trajectory_hyperlora.feedback_relation_pretrain \
  --device cuda:0 --gpu-fraction 0.35 --seed 42 --steps 1200 \
  --init-clean-checkpoint "$RUN_DIR/clean.pt" --freeze-clean-core --diverse-feedback \
  --output "$RUN_DIR/feedback.json" --checkpoint "$RUN_DIR/feedback.pt"
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/alf_delta_env/bin/python -m ttcl.trajectory_hyperlora.pretrained_relation_hyperlora \
  --device cuda:0 --gpu-fraction 0.35 --seed 42 --steps 400 --lr 0.0001 \
  --paired --correction-prob 0.5 --diverse-feedback \
  --relation-checkpoint "$RUN_DIR/feedback.pt" \
  --output "$RUN_DIR/lora.json" --checkpoint "$RUN_DIR/lora.pt"
```

主要结果 JSON 的 SHA-256：关系预训练种子 42/43 为 `32fff58c5c84ea67ae7c4ad0f8d5e5f5c145186ed333081e0be8b919e1792f20`、`8d1ac1e7dca0ee1f0e107e65c980837922a12676c318d5f6a6e2b6da32e2a18c`；干净历史 LoRA 配对训练 42/43 为 `d1e50636401801a507df795cdf09781525fe511f8d8409b6783c168416718330`、`5358b01026cabee84e3b4a9a7e1124fe2c8b36998ebf1ebcead956fff9cb02c5`；冻结关系核心的反馈读出 42/43 为 `1bb52f7706331289a2ffa47e92079831fe08e81ac40a00f22b7aa219e3ff20e9`、`0359fcf9775361866d94b6b2a93e400c7a21c5a30cdd64e0d9fecdab00bd4b56`；混合历史 LoRA 42/43 为 `108a1a1b2b36a7ab13f4f09415009580c320a3bf479c93f37840d725ed183157`、`ed056f755698c85400d29c5918f3d196a9fa6ebca2af151bd0f0569ebd321f06`；标准纠错探针 42/43 为 `19fe774f73b52693f0d68cc4457cca40e6cb161ecd5857198867edbb774ab7cd`、`f35f537a00ecc6e709d78dbe0ab213771bacc3925f488b571ca892c73ecfa367`。

新标记词探针四份 JSON SHA-256：种子 42 干净／纠错为 `6f7bc1477e98f543f37af2dee77dba4400ee4f7b25db3e9e7885cf19a56483ca`、`fd7fcd09b142bd9cce6e97b8afa3c449e0c72fa1f9dfb094fbdb19e27a6ec0ee`；种子 43 为 `e1be7fda6de671787f774880a0ad1ba87fd26b4ba162426c1beee20af3ae2d74`、`b0189857a6b9c144bc2d391cc2fc0cebcc1908e65a59240ba46b421b998b2f23`。
