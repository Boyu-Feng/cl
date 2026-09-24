# 模型生成的通用经验记忆

用本地 Qwen3 自己总结经验，替代任务专用的 Python 特征聚类。记忆核心在 `memory.py`，只接收原始任务文本、模型原始回答、公开反馈及可选标量 reward；不导入 CLBench，不读取任务私有字段，不使用频率、带宽、候选匹配规则。`run_benchmark.py` 单独负责当前 BSM 的输入输出适配与官方评分。迁移到别的任务时复用核心，更换环境适配器；当前尚未实测其他任务。

流程：当前题目＋旧记忆 → 回答一次 → 环境评分一次 → 旧记忆＋刚完成的交互 → Qwen3 用固定通用 prompt 生成新记忆 → 下一题。更新时看不到下一题。程序只做数据传递、顺序和长度检查、日志保存；事实和经验的组织由模型完成。

更新 prompt 原文见 `memory.py` 的 `UPDATE_PROMPT`。它要求保存有证据支持的事实、适用条件、过程经验、明确反馈和不确定性，附来源轮次；区分模型猜测与环境证据。单个回答的 reward 不能标记每一部分对错，不用不同题目的分数推断偏好。模型可能不完全遵守，摘要仍可能遗漏、幻觉或误归因。

```bash
# 默认运行三组：独立对照 / 模型摘要 / 模型摘要＋已有 reward
CUDA_VISIBLE_DEVICES=0 NUM_SCANS=12 bash ttcl/llm_memory/run.sh

# 单独运行通用摘要，不提供外部 scalar reward
CUDA_VISIBLE_DEVICES=0 MODE=summary NUM_SCANS=12 bash ttcl/llm_memory/run.sh

# 单独运行摘要＋reward
CUDA_VISIBLE_DEVICES=0 MODE=summary_reward NUM_SCANS=12 bash ttcl/llm_memory/run.sh
```

无需 API key。默认沿用 Qwen3-4B-Instruct-2507、seed 42、答案 temperature=0.7 / top_p=0.9 / top_k=0 / max_new_tokens=1536；各组答题种子逐题相同。记忆更新使用同一冻结模型，贪心生成，默认上限 1024 tokens，prompt 请求约 450 tokens。`--memory-max-new-tokens` 可调。模型生成的记忆是有损摘要，无法保证保留全部历史细节。

每题只评分一次。摘要组每题额外一次总结调用，没有新的评分或候选答案筛选。最后一题后也更新记忆，计入成本。`summary` 不传标量 reward，BSM 中通常只有 `report recorded` 的公开反馈；`summary_reward` 明确开放已获得的整份报告标量分数。两组反馈可见性不同，单独报告。

超出输入上下文限制会停止并写 `failure.json`，不静默截断输入。总结为空或耗尽输出预算时保留旧记忆，记录拒绝及待总结交互；下一轮一起重试，不额外增加当轮调用。最终指标列出未总结交互数和拒绝次数。

结果位于 `ttcl/results/llm_memory/`，包括：`report.md` / `comparison.json`、逐题 `responses.jsonl`、每次完整输入 `requests.jsonl`、模型输出 `generations.jsonl`、允许进入记忆的 `episodes.jsonl`、逐轮 `memory_updates.jsonl` 和最终 `memory.json`。`code_snapshot/` 与 `provenance.json` 保存实现和数据来源。旧规则摘要、全历史 ICL 和 GenericAgent 保留独立入口。

其他环境可以直接构造 `Episode(index, instance_id, task, response, feedback, reward)`，调用 `ExperienceMemory.update(episode, generate, seed)`；`generate` 接收 messages、seed、max_new_tokens、temperature，返回 raw_response 与 finish_reason。`answer_messages` 的 schema 可省略，适用于自然语言任务。模型回答和评分由环境适配器负责。

## 已完成的验证

[2026-09-18，前 12 条对照](../results/llm_memory/qwen3_generic_12_20260918/report.md)：独立、通用摘要、通用摘要＋reward 三组平均分均为 **0.22245**，两种记忆组的 12 条结构化回答都与独立组相同。24 次总结全部正常完成，60 次模型调用的提示哈希和 token 数、记忆传递、reward 可见性均已核验；93 项 CPU 测试通过。

当前实现验证了通用接口和实验流程，没有观察到性能提升。原始记忆中可见遗忘旧事实、过度解释不同题目分数的现象；prompt 的要求不能保证模型遵守。旧规则摘要的 0.27105 不能移用作本方法的成绩。尚未完成 90 条、多种子和跨任务实测。

六类任务的数据与反馈调研、结构化记忆操作及 writer 训练方案见 [通用结构化记忆设计](../docs/GENERAL_MEMORY_DESIGN_20260918.md)。该文档是下一阶段设计，未实现或实测其中的新训练方案。
