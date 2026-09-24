# REEF 在 CLBench 上的测试

`current_work/reef` 是多方法框架。当前入口测 **REEF-GEPA** 的提示词优化，直接调用仓库原生的候选生成、Pareto 候选库和严格验证选择逻辑。适配部分负责把 CLBench 完整公开交互交给 GEPA，并用官方任务重放候选；本地 Qwen 模型替代 HTTP 模型传输。该入口不是 REEF 服务部署测试，也不代表 SAO/SDFT 的模型权重训练。

当前试验：`ttcl/results/reef/gepa_v2_20260921/REPORT.md`，后台监督器每 20 秒更新状态。旧 `gepa_20260921` 因首题未启用经验注入而被检查拦截，保留排错记录，不计入成绩。

```bash
# 在工作区根目录运行，先选当前空闲 GPU。
CUDA_VISIBLE_DEVICES=2 ttcl/.runtime/reef_env/bin/python ttcl/reef/run_gepa.py \
  --task database_exploration --output ttcl/results/reef/my_run/database_exploration
CUDA_VISIBLE_DEVICES=3 ttcl/.runtime/reef_env/bin/python ttcl/reef/run_gepa.py \
  --task cohort_studies --output ttcl/results/reef/my_run/cohort_studies
python ttcl/reef/report.py ttcl/results/reef/my_run
```

默认每任务前 3 题生成候选，接下来的 3 题选提示词，做 2 轮优化；冻结提示词后，在 canonical indices 12–19 上与无经验组配对评测。环境 seed=42。最终测试不再更新提示词；这属于先优化再评测的开发试验，不是持续在线学习。支持 `--rounds`、`--test-count`，以及只验证环境的 `--preflight`。输出目录必须为新目录，避免覆盖结果。

Qwen3-4B 权重、actor 采样和工具预算沿用现有 TTCL 实验；反思使用同一模型、temperature=0。无经验和 GEPA 组每题都重置对话，只保留当轮选定的提示词。模型只能看到公开轨迹和已完成题目的官方 scalar，不读取隐藏评分元数据。环境执行失败会停止并保留错误，不补零。

`ttcl/.runtime/reef_env` 为继承 seal_env 基础依赖的独立虚拟环境，安装当前 REEF 源码与 CPU 核心依赖，未修改共享训练环境。首次运行的具体依赖版本和 REEF commit 保存在结果目录。运行检查：

```bash
ttcl/.runtime/reef_env/bin/python -m unittest ttcl.reef.test_adapter -v
current_work/continual-learning-bench/.venv/bin/ruff check ttcl/reef
```
