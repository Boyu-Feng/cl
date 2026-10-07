# 方向投影的轨迹 LoRA 真实回报训练 v6

前一版 v5 与各向同性 v4 在相同 72 个已审核自身成功轨迹—官方 ALFWorld `train` 目标配对上，各执行 144 个真实环境回合。v5 的动作敏感采样使正负回报不同的配对从 **15/72** 增至 **21/72**，但冻结后在反复使用的 12 题开发集与旧 LoRA、无 LoRA 都是 **4/12**。已审核的 v5 score $\Sigma^{-1}\delta$ 沿其动作敏感方向的平方范数平均仅 **0.638%**：虽然约 53.43% 的采样扰动能量沿该方向，训练更新却几乎都指向正交维度。更多奖励差没有转成稳定有利的参数更新。[完整 v5 协议与证据](ALF_ACTION_SENSITIVE_REWARD_TRAINING_20261007.md)。

v6 只修改训练估计量，不修改冻结演员、来源轨迹编码器、LoRA 生成头、8 维代码基、目标配对顺序、动作敏感方向、两个协方差标准差、标准高斯噪声、官方终局奖励或 144 回合预算。对已固定的单位方向 $u$、$\Sigma=\sigma_\perp^2I+(\sigma_\parallel^2-\sigma_\perp^2)uu^\top$ 和成对扰动 $\delta$，v5 使用

$$
\widehat g_{\mathrm{full}}=\frac{R(\mu+\delta)-R(\mu-\delta)}{2}\Sigma^{-1}\delta.
$$

v6 改用

$$
\widehat g_{\mathrm{proj}}=uu^\top\widehat g_{\mathrm{full}}
=\frac{R(\mu+\delta)-R(\mu-\delta)}{2}
\frac{u^\top\delta}{\sigma_\parallel^2}u.
$$

在条件于当前来源、目标题和预先计算的 $u,\Sigma$ 时，高斯 score 恒等式给出 $\mathbb E[\widehat g_{\mathrm{proj}}]=uu^\top\nabla_\mu\mathbb E_\delta[R(\mu+\delta)]$。所以它是**平滑回报梯度在 $u$ 上的无偏投影**，不是完整八维回报梯度的无偏估计。这个取舍用限制每个样本的更新方向来消除 v5 中占主导的正交 score 噪声；若真正有利的变化长期与 $u$ 正交，v6 也可能更差。$u$ 来自冻结演员对当前两条实际可执行命令的概率间隔，未写死 ALFWorld 类别或动作正确答案，但此实现需要可列举的候选命令，跨任务接口尚待验证。

新训练脚本 `train_alf_projected_reward_v6.py` 和独立原环境审核器 `audit_alf_projected_reward_v6.py` 均为新文件；v5 文件及原始结果保留。72 个新输入绑定在 ignored 的 `data/annotations/alf_projected_reward_train72_reviewed_20261007.json`，加入了 estimator 标识与来源／目标／旧矩阵哈希，不能与 v5 的新回合共用标注。单配对训练冒烟及两条回合原环境重放已通过；其回报两臂都为零，不计为新算法效果。完整 v6 训练与独立留出评测尚未完成，不能从数学推导声称它已经改善官方任务。

评测代码在读取 v6 任何开发或官方在线题结果之前另建：`evaluate_alf_projected_reward_v6.py`／`audit_alf_projected_reward_eval_v6.py` 沿用同一批 12 道旧开发题及同来源旧 LoRA、无记忆、原始文本四臂；`evaluate_alf_fresh_seen_projected_v7.py`／`audit_alf_fresh_seen_projected_v7.py` 使用先前冻结的官方 `valid_seen` 六类各六题和共同自身成功历史流，与 v5 在线对照逐题匹配。该 36 题在 v5 在线运行后不再是对 v6 的完全盲测；v6 的算法、模型选择和评测代码在查看其完整结果前固定，后续仍须真正新题序和多个种子复验。
