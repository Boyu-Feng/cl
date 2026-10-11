# CLBench：单组 LoRA 因子持续更新，按奖励筛选轨迹

## 与已有 ALFWorld 方法的关系

本版沿用 [ALFWorld 逐题累积 LoRA](ALFWORLD_ONLINE_LORA_ACCUMULATION_20261007.md)的运行平均公式，而不是误称为新发明：

`B_next = B_old + (B_new - B_old) / (accepted_count + 1)`。

冻结的超网络只读取本题已完成的自身公开轨迹，生成新 `B` 因子；共享 `A` 因子与 Qwen 演员不在线训练。与 ALFWorld 原实验的区别是，那次按协议纳入成功和失败轨迹；本版先用已有奖励门槛筛选，未通过的轨迹不调用 `write`，持久 `B` 和更新计数保持不变。下一题开始时读出同一组累计因子并挂载，证据候选门控仍在演员生成动作之后运行。

这也不同于旧 CLBench `ParameterMemory`：旧版最多先保存 64 组来源因子，读取时按题目相似度混合；本版始终只有一组持久因子。ALFWorld 另一版持续更新来源向量并按当前题重新生成 LoRA，也与这里的直接因子累积不同。

## 已完成的验证

官方频谱 seed 42 的前 4 题、冻结模型与门控、共同 LoRA 来源流；新审核目标绑定运行器、操作模块、门控、检查点和初始问题哈希。4/4 可评分，0 失败，独立审计通过。题 0 和题 1 各写入一次，题 2 的奖励门槛弃权且参数状态哈希不变；题 3 开始读取的仍是累计 2 次更新的单组 LoRA。题 3 是前缀最后一题，按预定规则不再为不存在的下一题写入。

LoRA 控制臂与 LoRA＋证据臂的 4 题官方均分分别为 **0.22665**、**0.255825**。证据臂选了 2 次 `array_union`，因此该分差不能归因于 LoRA 累积；4 题也不足以声称完整频谱或其他领域提升。完整报告 SHA-256 为 `5cb728469ab292c6cddbe376a1278c4c930ac79828e24f50864273e9f430b911`，审计结果留在 ignored `results/`。先前停止的 2/3 题部分运行也保留，不计作完整成绩。

实现为 `continual_lora_state_v26.py`、`clbench_continual_lora_v26.py`、`audit_clbench_continual_lora_v26.py`，单状态读写测试在 `test_continual_lora_state_v26.py`。这次是对历史 ALFWorld 直接因子平均路线增加奖励筛选后的 CLBench 短前缀验证；尚无完整 90 题或跨领域结果。
