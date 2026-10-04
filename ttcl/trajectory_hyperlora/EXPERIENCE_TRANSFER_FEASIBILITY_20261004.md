# 从轨迹提取参数经验：受控可行性实验（2026-10-04）

## 回答的问题

此前颜色实验只把一个历史答案存进 LoRA，属于记忆。本实验要求迁移**动作规则**：历史中有若干不同读数、模型动作和成功反馈；后续读数及提问措辞都变了，模型须根据奇偶状态选择动作。实验结论是：**在这个小型合成任务中，从成功轨迹训练出的 LoRA 能承载可迁移的规则；尚未证明超网络能直接生成同等质量的 LoRA。**

## 协议

合成环境有两种未告知执行模型的约定：一种是偶数按 `LEFT`、奇数按 `RIGHT`；另一种反过来。当前任务只给读数、奇偶指示和环境原生可选动作。每种约定生成 16 步完成的成功轨迹，含 8 个奇数、8 个偶数及对应成功动作。只用这些轨迹里的成功动作训练一个 rank-4 LoRA；Qwen3-4B-Instruct-2507 底座冻结，LoRA 只挂最后两层 `mlp.down_proj`，每个适配器合计 600 次训练采样。同一底座产生两个独立的轨迹适配器；测试时根据当前历史选择其对应适配器，错配对照故意选择另一个。

训练读数取自 1–89；测试读数 101、102、113、114，均未进入任一轨迹或训练。原测试提问使用新的措辞。比较：不挂载参数、提示词附上完整轨迹、挂正确轨迹 LoRA、挂另一轨迹 LoRA。每种约定四个读数，共 8 个配对测试。动作要求单词输出，判分为首词严格等于 `LEFT` 或 `RIGHT`。每条评测记录含实际 source 和 query 内容的 SHA-256 绑定及期望动作；标签由固定合成环境函数生成，没有复用旧历史 ID 或真实 benchmark 标签。

| 运行 | 无记忆 | 文本轨迹 | 正确轨迹 LoRA | 错配轨迹 LoRA |
| --- | ---: | ---: | ---: | ---: |
| 种子 42，4 个新读数 × 2 规则 | 4/8 | 8/8 | **7/8** | 1/8 |
| 种子 43，独立源轨迹及初始化 | 4/8 | 8/8 | **7/8** | 1/8 |
| 种子 42 的 LoRA 独立进程重载，另一组新读数 121、122、137、138 | 4/8 | 6/8 | **8/8** | 0/8 |

最后一行用标准 PEFT 格式保存的两个 LoRA，另起进程加载 Qwen4B 和适配器；新读数与训练和前一次测试都不重合。首次重载测试使用另一种提问措辞时，基模和文本轨迹组倾向输出解释，5-token 截断后首词判分均为 0/8；该失败保存在 `fresh_reload_seed42.json`。随后改为明确要求只输出 `LEFT/RIGHT` 的新措辞，得到表中 `fresh_reload_v2_seed42.json`。这次措辞调整是在看到首次结果后做的，故第三行是探索性复验，不是盲测。

源轨迹内容 SHA-256：种子 42 的两条分别为 `0e4bfb66ecf865103ce9ad01dd41acec4d22cb193f2bf67f303faddf6100bbc3`、`6f60fe8664d916f7f6793dfd2e0df885c1c460cdde40e2ad7112d265be2e0838`；种子 43 分别为 `21bbc00d67a6379588a31700a528a9fd468f7a3e7e9d2ee99453c11a5209abd9`、`9cb9d7957dd4bde1a22e501b7ac514fe75847651f41aad77320a565e19051fc9`。原测试的四个 query 内容 SHA-256 依次为 `6d72ffba37516a25d625f8f5ad0c8a30c87d33d14b0a8da18e3a9d2123a2369e`、`b1a7bc5c7c76732b719fc6ea8e9a2febe3fa8bbbb5e884b5ab491867c2348c85`、`443b52062e1770761bd9932f15732b84d3d52b0c871a9eb592390bfff856965a`、`1985daa4b11bfbd9d00e0abeb3d57262c072b403b96f88a14b42119977bf5983`；独立重载新测试依次为 `600ee255a31dcd2786498f11adb3e9925172d329fe1b14bd87b25ff8b30695d6`、`952899db0909dbf52fd1d47636ee79bd6b631782ac75a398705a59ca3bcfa4ea`、`57845fd0ff1ce9b7839646ada53a8df947576090d8f5174314f2eb5e8500c32e`、`d8d0aefb9848fd8b99a7457231cd0090bc802dd0b29fa0a74001a190878551d3`。这些绑定来自实际输入字符串，不依赖历史 ID。

正确 LoRA 同时要处理新奇数和新偶数，不可能靠重复历史中的一个动作拿到 7/8；错配 LoRA 的下降也说明参数携带的是轨迹相关的决策偏好。不过任务特意给出奇偶指示，规则家族只有两种，每条轨迹有 16 个成功例子；本结果不能推断从稀疏、失败混杂的 ALFWorld／CLBench 真实轨迹能同样提取经验。样本只有两种子、每种子 8 道评测题，不能给出稳定泛化率。

## 超网络目前的边界

同一受控环境里，直接训练原始轨迹→混合 LoRA 的小超网络，原始 embedding 编码及冻结 Qwen 隐状态编码均为 8/16，与无记忆、错配轨迹持平。先训练两种规则的 LoRA 基底，再训练轨迹路由器也为 8/16；固定顺序的两步轨迹版本为 7/16，错配同为 7/16。用环境原生动作的两次无文字输出探测作为潜变量后，超网络达到 12/16，错配为 6/16，但错误集中在其中一条规则的奇数状态，仍不足以称为可靠的规则 LoRA 生成。训练记录分别在忽略目录的 `seed42.json`、`seed43_frozen_lm.json`、`two_stage_seed42.json`、`ordered_pair_seed42.json`、`probe_seed42.json`。

因此，**参数经验存在性**和**快速从任意轨迹生成这种参数经验**是两道不同的问题。前者在此合成任务得到初步正证据；后者仍需研究更强的轨迹关系编码、轨迹效用监督和可能的 RL 配对奖励。本轮没有训练 RL，也没有 ALFWorld／CLBench 成绩。

## 复现及资产边界

代码为 [support_transfer_pilot.py](support_transfer_pilot.py) 和 [verify_support_transfer.py](verify_support_transfer.py)；早期超网络尝试分别见 [experience_pilot.py](experience_pilot.py)、[two_stage_pilot.py](two_stage_pilot.py)、[probe_router_pilot.py](probe_router_pilot.py)。按仓库第 4 节准备本地模型和主线 Python 环境后运行：

```bash
CUDA_VISIBLE_DEVICES=0 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.support_transfer_pilot \
  --seed 42 --steps 1200 --support-per-parity 8 --export-adapters
CUDA_VISIBLE_DEVICES=0 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.support_transfer_pilot \
  --seed 43 --steps 1200 --support-per-parity 8 \
  --output results/trajectory_hyperlora/experience_pilot_20261004/support_seed43.json
CUDA_VISIBLE_DEVICES=0 ttcl/.runtime/alf_delta_env/bin/python \
  -m ttcl.trajectory_hyperlora.verify_support_transfer \
  --output results/trajectory_hyperlora/experience_pilot_20261004/fresh_reload_v2_seed42.json
```

GPU 0 进程显存上限为总显存的 40%，其他实验未停止或修改。底座配置 SHA-256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`，三份权重分片依次为 `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`、`0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`、`7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`。原始逐题输出和两个可加载的 LoRA 留在 Git 忽略的 `results/trajectory_hyperlora/experience_pilot_20261004/`；Git 只保存代码和本结果摘要，不上传模型或权重。
