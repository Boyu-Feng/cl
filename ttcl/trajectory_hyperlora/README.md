# 轨迹生成参数经验：可行性与首轮实验（2026-10-04）

## 结论

方向在技术上可行，也有相近论文实证。本仓库的受控实验发现：直接训练轨迹→LoRA 不稳定；先用轨迹中已发生的回答 token 预热编码器，再训练生成 LoRA，冻结 Qwen3-4B 在**新措辞**的后续同类查询上两种子均为 8/8，无记忆与错配轨迹均为 0/8。这证明小型场景中信息可经 LoRA 挂载并改善回答。但轨迹包含“先答错、再纠正”的多步反馈时，两种子只为 2/8、4/8，信用分配仍未解决。它是受控颜色任务，不是 ALFWorld 或 CLBench 分数，也未训练 RL。

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

代码：[pilot.py](pilot.py)、[test_pilot.py](test_pilot.py)。底座为本地 Qwen3-4B-Instruct-2507，冻结全部原参数；轨迹编码器读取 tokenizer 后的**完整原始轨迹**，没有人工动作类别或文本经验总结。原型先生成四个共享 rank-4 LoRA 基底的混合系数，在最后两层 `mlp.down_proj` 形成轨迹专属低秩更新。这是受限的 hypernetwork，**不是** Text-to-LoRA 论文那种逐层直接预测完整 A/B 的实现。四种颜色是受控隐藏约定。训练用四种轨迹措辞，原测试和新测试各用两种未见措辞，每组每色两题，共 8 题。测试时 LoRA 组的 query 不含原轨迹。预热阶段仅预测源轨迹里已发生的回答 token；之后用独立查询的回答 token 交叉熵训练 LoRA 生成路径。预热仍是监督学习，不是 RL。

| 探索运行 | 训练步 | 无记忆 | 轨迹文本 | 生成 LoRA | 错配轨迹 LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU 编码器 | 80 | 0/8 | 8/8 | 2/8 | 2/8 |
| 注意力池化编码器 | 400 | 0/8 | 8/8 | 2/8 | 4/8 |

第二次运行是在看到第一次结果后改编码器，且重复使用同一小测试集，所以它是探索性诊断。两次的训练 loss 均下降，但这不能替代未见措辞的准确率；错配轨迹有时更高。进一步的单独读出诊断显示编码器可从未见措辞中识别源回答 8/8，因此增加**通用回答 token 重建预热**来增强参数通路，而不是添加手工动作类别。

| 预热 200 步 + LoRA 训练 400 步 | 无记忆 | 轨迹文本 | 生成 LoRA | 错配轨迹 LoRA |
| --- | ---: | ---: | ---: | ---: |
| 种子 42：原测试措辞 | 0/8 | 8/8 | 8/8 | 0/8 |
| 种子 42：新测试措辞 | 0/8 | 8/8 | 8/8 | 0/8 |
| 种子 43：新测试措辞 | 0/8 | 8/8 | 8/8 | 0/8 |
| 种子 42：含错误尝试、纠正和反馈 | 0/8 | 8/8 | 2/8 | 0/8 |
| 种子 43：含错误尝试、纠正和反馈 | 0/8 | 8/8 | 4/8 | 0/8 |

种子 42 的第一次预热运行只测原措辞；随后用两个种子评估完整三组，每组 8 题。新措辞组在看到结果前固定；原措辞组已用于开发，不能当独立确认。预热直接从源轨迹中学习“什么回答出现过”，因此单步测试的信息转移难度较低；多步组暴露了模型尚未稳定区分被拒绝与被接受回答。文本轨迹组 8/8，也说明当前 LoRA 没有超过等信息量的文本基线。小样本和同一组合成任务族不能外推到真实环境。

运行数据保存在忽略上传的 `results/trajectory_hyperlora/pilot_20261004/`；表中摘要随代码提交。两项 CPU 检查验证关闭适配器时输出逐位等于底座，以及不同潜变量产生不同输出且梯度流回生成系数。种子 43 的超网络参数保存为本地 `seed43_hypernetwork.pt`，第 0 条测试轨迹生成的标准 PEFT LoRA 保存为本地 `generated_adapter_case_0/`；独立进程用 `PeftModel.from_pretrained` 挂载后答出 `amber`，与自定义适配器输出一致。**所有权重均在 Git 忽略目录，不上传。**所有训练仅使用 GPU 0，进程显存上限设为该卡的 40%，没有停止或改动其他实验。底座配置 SHA-256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`，权重索引 SHA-256 为 `d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c`，三份权重分片的 SHA-256 分别为 `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`、`0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`、`7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`。

重跑命令（主线 Python 环境）：

```bash
CUDA_VISIBLE_DEVICES=0 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.pilot \
  --device cuda:0 --gpu-fraction 0.40 --seed 43 \
  --warmup-steps 200 --steps 400 --test-set combined \
  --checkpoint-path results/trajectory_hyperlora/pilot_20261004/seed43_hypernetwork.pt \
  --export-case 0 \
  --output results/trajectory_hyperlora/pilot_20261004/seed43_export.json
```

原始 80 步版本将命令中的 `--warmup-steps 200 --steps 400 --test-set combined` 改为 `--warmup-steps 0 --steps 80 --encoder gru --test-set original`，并使用种子 42；原始 400 步版关闭预热并只测原措辞。独立加载导出的 LoRA 可调用 `PeftModel.from_pretrained(base_model, "results/trajectory_hyperlora/pilot_20261004/generated_adapter_case_0")`。`results/` 未纳入 Git，模型和生成器权重也未提交。

下一步应把预热目标改为完整轨迹的多动作重建与公开反馈预测，同时扩大训练任务／轨迹／查询多样性，并实现逐层直接输出 A/B 的小 rank 版本。随后用相同任务、相同采样种子的实际使用／不使用 LoRA 差值训练随机生成器，优先解决“错误尝试→纠正”轨迹的信用分配。进入真实 ALFWorld／CLBench 之前，需用新的任务族和独立测试集验证，不能依据当前颜色任务宣称真实 benchmark 提升。
