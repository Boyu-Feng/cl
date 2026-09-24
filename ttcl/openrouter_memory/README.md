# OpenRouter 大模型自主经验库实验

复用 `llm_online_bank_20260920_v2` 的完整轨迹压缩、reward反馈、事务性经验库更新和官方CLBench执行器。新实验全程不加载模型权重、不训练、不使用GPU。CPU只运行任务环境、校验器和原Qwen tokenizer。

## API key 输入入口

在 `/home/fengboyu/cl` 的终端执行：

```bash
bash ttcl/openrouter_memory/run.sh
```

启动器显示配置后提示输入 OpenRouter API key，输入不回显。key 只存在于本次进程内存及后台子进程环境，不写配置、命令行或日志。也支持预先设置的 `OPENROUTER_API_KEY`；没有安全TTY且未设置环境变量时拒绝回显输入。不要把真实key提交到聊天或写进命令行。

默认：`openai/gpt-6-astra`，Database/Cohort各12题，两组，reasoning=medium，API输出预算8192（包括reasoning），费用上限$50。启动后后台执行，记录 `run.pid`，结果位于新建的 `ttcl/results/openrouter_memory/gpt6_bank_<时间>/REPORT.md`。费用上限按每次调用的保守预留检查，可能提前停止；这是成本保护而非最终账单。额度不足或网络/鉴权失败会明确停止，不能把未完成算作零分。

```bash
# 先小规模检查，仍只使用 API
bash ttcl/openrouter_memory/run.sh --num-instances 3 --max-cost-usd 10

# 查看实时目录中的精确模型ID，不收费
bash ttcl/openrouter_memory/run.sh --list-models gpt-6

# 只准备代码快照和配置，不需要 key
bash ttcl/openrouter_memory/run.sh --prepare-only --root ttcl/results/openrouter_memory/my_test

# 稍后输入 key 并启动已准备目录
bash ttcl/openrouter_memory/run.sh --start ttcl/results/openrouter_memory/my_test

# 固定强actor，额外加另一writer组：将精确ID替换成目录中实际可用模型
bash ttcl/openrouter_memory/run.sh --model openai/gpt-6-astra --comparison-writer-model <精确模型ID>
```

`--writer-model` 可把主经验组的总结模型与actor分开。`--foreground` 留在当前终端。`--provider` 可限制到特定OpenRouter提供商，默认禁止provider fallback，记录实际提供商；从不静默换模型。每次启动目录只能运行一次，不自动重跑可能已经付费的中断任务。

## 实验定义

- independent：强模型每题独立解答，无经验库。
- online_bank：相同强actor读自己的经验库解答；每题结束后writer读取完整公开轨迹和官方scalar，选择KEEP/ADD/REVISE/REMOVE；从空库开始。
- 可选 alternate_writer：actor相同，writer换成另一个API模型，每组只继承自己的经验。

默认两任务各前12题、环境seed42，与原4B试验相同顺序和工具预算。当前任务看不到未来题目；历史终局反馈中已公开的内容保留。writer只看当前结束的轨迹和已有经验，不看下一题或隐藏评分标签。最多8条、9000字符、按原Qwen CPU tokenizer计2048 tokens，以保持与旧实验相同银行容量；这不是GPT实际计费tokens。格式/引用验证失败原子拒绝，最多一次writer格式重试，答题格式最多两次重试。

主比较是同一强模型“经验－无经验”的配对reward增量。与旧4B比较的是两个模型各自的增量，不能直接用GPT-6经验组减4B无经验组。默认同时换actor和writer，不能单独确定小模型的瓶颈在总结还是使用经验；可选第三组用于固定actor比较writer。

原prompt原文冻结为 `EXTRACTION_PROMPT.md`；原始模型回复、全部动作/公开反馈、bank_before/after、token用量、实际API model/provider、reasoning token用量均保存。API自动重试仅针对暂时性网络/429/5xx，最多2次，所有HTTP尝试单独计数；没有产生官方分数的执行失败记缺失，不填零。

GPT-6当前目录不声明temperature/top_p支持，因此不发送；发送可支持的seed，截成31位，同任务两组一致。API seed不能保证确定性，首题空库的两次独立执行也可能不同。reasoning输出不加入经验；只使用最终回复中的显式动作和公开工具反馈。8192 completion预算包含可能的隐藏reasoning，与旧4B的4096可见输出预算不同，报告不能声称等计算比较。

## 成本与模型目录

启动时按精确ID检查实时 `/models`，执行时再保存当前元数据到 `runtime_models.json`。不自动用别的模型替代GPT-6。`api_calls.jsonl` 记录每次API尝试、用途、实际model/provider、usage及成本；`api_usage.json` 汇总实际返回费用及未知费用的保守预留。价格采用目录中的价格上界，包含长上下文阶梯。遇到预算阈值立即停止，不伪造剩余成绩。

公开目录确认（2026-09-21）：`openai/gpt-6-astra`。可用性最终以账户权限及运行时返回为准。本地无key时只完成准备与mock测试，不能宣称真实API推理验证成功。

来源：[OpenAI Docs GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)、[OpenRouter模型目录](https://openrouter.ai/api/v1/models)、[Chat Completions](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)、[provider路由](https://openrouter.ai/docs/guides/routing/provider-selection)、[reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。
