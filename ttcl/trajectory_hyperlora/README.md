# 轨迹生成参数经验：可行性与首轮实验（2026-10-04）

## 结论

方向在技术上可行，也有相近论文实证；但本仓库首轮小型原型**尚未证明收益**。冻结 Qwen3-4B 后，轨迹条件化的低秩参数更新确实可以挂载并改变回答；在未见过的轨迹措辞上，生成适配器只答对 2/8，打乱轨迹对照分别是 2/8 和 4/8。直接把轨迹放进上下文为 8/8。这个受控颜色任务仅检验信息能否从完整轨迹转移到权重，不是 ALFWorld 或 CLBench 分数，也未训练 RL。

## 相关工作与真正的差别

- [Text-to-LoRA (2025)](https://arxiv.org/html/2506.06105)：用任务描述生成 LoRA，可从已有 LoRA 重建或通过下游 SFT 端到端训练。论文指出 SFT 版对新任务的泛化优于重建版；相似任务的独立 LoRA 在权重空间未必靠近。因此不宜简单回归旧 LoRA 的 A/B 矩阵。
- [Doc-to-LoRA (2026)](https://arxiv.org/html/2602.15902)：把长上下文映射为可挂载 LoRA，让无上下文的学生模仿有上下文的教师。这直接启发“轨迹→参数经验”的行为蒸馏预训练。
- [LatentSkill (2026)](https://arxiv.org/html/2606.06087) 及其[官方代码](https://github.com/yuaofan0-oss/LatentSkill)：非常接近，先用约 17.1 万技能文档预训练，再以完整 agent 轨迹监督生成 LoRA；论文在自己的 Qwen3-8B／正确技能选择协议下报告 ALFWorld seen 74.3%、unseen 69.4%。它的生成器输入仍是**技能文档**，轨迹主要用于行为监督；本研究要检验的是**完成的原始轨迹直接作为生成器输入**，并用后续任务的收益训练和验证。

这些论文支持架构的可行性，不能推断本项目的 ALFWorld／CLBench 一定提升。LatentSkill 原实验使用 8×H100；本地共享 A100 条件下不应照搬完整规模。

## 目标算法（下一阶段）

令完成且仅含公开可见信息的轨迹为 \(\tau\)，当前任务为 \(x\)。编码器 \(E_\phi\) 读取轨迹的观察、动作、公开反馈和可用奖励，产生潜变量 \(z\)。逐层生成低秩参数：

\[
z=E_\phi(\tau),\qquad (A_{\ell},B_{\ell})=H_\phi(z,e_\ell),\qquad
W'_\ell=W_\ell+g(x,\tau)\frac{\alpha}{r}B_\ell A_\ell.
\]

\(g\) 是对相似任务的使用强度，可由检索相似度和验证效用学习；它不能读取未来标签。每条轨迹保留单独的参数经验及来源，检索后只挂载少量经验证的适配器；不把全部历史 LoRA 永久累加，否则容易干扰和遗忘。生成器与底座分开：底座冻结，更新编码器／超网络。

建议先用同一训练分割中的“已完成轨迹 \(\tau\) → 后续相似任务 \(x\)”配对，做有上下文教师到无上下文 LoRA 学生的行为蒸馏，或仅对成功、经审核的动作做 SFT。随后用真实环境奖励微调生成器。对于同一 \(x\) 和相同随机种子，计算

\[
\Delta R = R(\pi_{\theta+H_\phi(\tau)},x)-R(\pi_\theta,x),\qquad
\nabla_\phi J\approx(\Delta R-b)\nabla_\phi\log q_\phi(z\mid\tau)
-\lambda\nabla_\phi\mathrm{KL}(\pi_{\theta+H_\phi(\tau)}\Vert\pi_\theta).
\]

这是对**超网络生成的参数经验**做策略梯度，不是把整条失败轨迹里的每个动作都当成坏动作。稀疏终局奖励仍有高方差，故先蒸馏预热、再用配对差值和多次采样；每次成功更新还要检查其他任务的退化。评估必须包括无记忆、等预算文本轨迹、直接在线 LoRA、随机／错配轨迹、生成 LoRA、生成 LoRA + 文本的消融。最终用官方 ALFWorld seen/unseen 和 CLBench 全领域评分，严格遵守训练/测试分离、原有动作预算和失败记录；测试轨迹只能在协议允许的在线时点更新记忆，不能回流训练超网络。

## 已完成的最小实验

代码：[pilot.py](pilot.py)、[test_pilot.py](test_pilot.py)。底座为本地 Qwen3-4B-Instruct-2507，冻结全部原参数；轨迹编码器读取 tokenizer 后的**完整原始轨迹**，没有人工动作类别或文本经验总结。原型先生成四个共享 rank-4 LoRA 基底的混合系数，在最后两层 `mlp.down_proj` 形成轨迹专属低秩更新。这是受限的 hypernetwork，**不是** Text-to-LoRA 论文那种逐层直接预测完整 A/B 的实现。训练目标是后续独立查询的回答 token 交叉熵；四种颜色作为受控隐藏约定，训练用四种轨迹措辞，测试用两种未见过的措辞。每个颜色两题，共 8 题。测试时 LoRA 组的 query 不含原轨迹。

| 探索运行 | 训练步 | 无记忆 | 轨迹文本 | 生成 LoRA | 错配轨迹 LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU 编码器 | 80 | 0/8 | 8/8 | 2/8 | 2/8 |
| 注意力池化编码器 | 400 | 0/8 | 8/8 | 2/8 | 4/8 |

第二次运行是在看到第一次结果后改编码器，且重复使用同一小测试集，所以它是探索性诊断，不是独立确认。两次的训练 loss 均下降，但这不能替代未见措辞的准确率；错配轨迹有时更高，说明当前生成器并未可靠地把轨迹中的约定映射到权重。可能的原因包括训练情景太少、只改最后两层和共享基底表达能力不足；本实验不能分离这些因素。没有理由基于这两个结果启动高成本 RL 或宣称真实 benchmark 提升。

运行数据保存在忽略上传的 `results/trajectory_hyperlora/pilot_20261004/{metrics,attention_400}.json`；表中摘要随代码提交。两项 CPU 检查验证关闭适配器时输出逐位等于底座，以及不同潜变量产生不同输出且梯度流回生成系数。两次训练仅使用 GPU 0，进程显存上限设为该卡的 40%，没有停止或改动其他实验。底座配置 SHA-256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`，权重索引 SHA-256 为 `d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c`，三份权重分片的 SHA-256 分别为 `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`、`0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`、`7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`。本实验未保存或提交模型权重。

重跑命令（主线 Python 环境）：

```bash
CUDA_VISIBLE_DEVICES=0 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.pilot \
  --device cuda:0 --gpu-fraction 0.40 --steps 400 \
  --output results/trajectory_hyperlora/pilot_20261004/attention_400.json
```

80 步版本将命令中的 `--steps 400` 改为 `--steps 80 --encoder gru`，输出路径改为 `metrics.json`。

下一步先扩大训练任务的轨迹与查询多样性，并实现逐层直接输出 A/B 的小 rank 版本；独立保留新的测试模板。只有在错配轨迹消融明显变差、生成 LoRA 稳定超过无记忆基线之后，才进入真实 ALFWorld／CLBench 配对奖励的 RL 阶段。
