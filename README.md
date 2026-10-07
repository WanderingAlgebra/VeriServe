# VeriServe

stage1 保留原实现；stage2/3 的代码按职责集中在 `veriserve_research`，配置、研究说明和原始运行记录继续按阶段保存。

## stage2/3 功能入口

```bash
python -m veriserve_research --help
python -m veriserve_research self-check
python -m veriserve_research probe --config stages/stage2/config.json --prepare-only
python -m veriserve_research intervene --experiment fixed-step --phase prepare
python -m veriserve_research intervene --experiment high-low --phase prepare
python -m veriserve_research analyze --run stages/stage3/runs/20261007-9fe830084094
```

`probe` 保留采集、smoke、拟合与备份参数；`intervene` 保留各协议的 phase、resume 和 stop-after 参数。原 `python -m stages.stage2.run_probe`、`stages.stage3.run_timing`、`stages.stage3.high_low` 命令仍可使用，转发至新实现。新实现使用独立的 `v2-<seed>-<identity>` run，`--resume` 只恢复新实现的执行记录。

公共能力分为 `artifacts`（读写、校验、版本和备份）、`trajectory`（原 token 边界与评分）、`inference`（模型与 KV）、`probe`（采集和拟合）、`intervention`（两类独立干预）和 `analysis`（统计及报告）。stage3 读取 stage2 的材料，不依赖其启动脚本。stage1 的必要共用能力集中通过 `artifacts.legacy` 适配。

stage2/3 的独立环境锁定在 `environments/stage23/pyproject.toml` 和 `uv.lock`，对应已验证的 Python 3.12、PyTorch 2.8.0+cu128 等实际版本。已有可用 GPU 环境可以直接使用上述命令；根环境和 stage1 环境快照保持原样。新机器从仓库根目录使用独立环境：

```bash
uv sync --project environments/stage23 --python 3.12 --frozen
uv run --project environments/stage23 --frozen python -m veriserve_research self-check
uv run --project environments/stage23 --frozen python -m pytest -q
```

所有 CPU 自检使用临时目录。真实运行的 smoke/pilot 会在各自新 run 保存检查记录，并继续要求真实 GPU 检查通过后才能冻结正式实验。新代码保留原实验方法、预算、PRM 规则、bootstrap 和计时范围。本次迁移的 41 项 pytest、探针重拟合和独立 GPU smoke/pilot 均已通过，见[验收记录](docs/refactor_validation.json)；该验证没有冻结正式实验。

独立 GPU 验证使用 `python tests/research_gpu_check.py`，在 `.cache/refactor-validation/` 保存 10 题 smoke、20 题 fixed-step pilot 和 4 题 HIGH/LOW pilot，直接调用执行函数，不触发自动备份或正式协议冻结。

## 历史材料、分析与恢复

旧材料和新材料均可通过 `analyze --run` 读取，默认输出到原 run 的 `analyses/<分析哈希>/`，也可指定 `--output` 到独立目录。原始 token、探针、配对记录及原报告不被覆盖。分析哈希包含分析源码与输入文件校验值，`analysis_provenance.json` 记录完整来源。旧绝对材料路径按当前仓库位置解析，再核对原 SHA-256；新材料使用仓库相对路径。

执行源码与分析源码分别记录 SHA-256。修改报告或配对统计代码会产生新的分析目录；修改采集、拟合、选择、干预或其共用依赖会产生新的执行身份，不能混入既有 run。

继续旧实验使用 [历史 run 与源码提交清单](docs/legacy_runs.json) 中逐文件核对通过的 `source_commit`，在独立 checkout/worktree 使用该提交的原入口与对应环境。旧版 HIGH/LOW 的绝对路径约束仍存在，恢复执行需要原目录位置；新代码的只读重分析已支持移动仓库。清单中的 `original_root` 记录原位置，`source_hashes` 可用于复核。不要将新代码的哈希写回旧 manifest 来绕过恢复校验。

源码提交和最新材料备份提交可能不同。恢复旧执行时，还需把最新已核对的 run 材料及 stage2 依赖放回原相对位置，保留原文件字节。

## 从哪里开始看

- [第一阶段：初步流程](stages/stage1/README.md)：方案、代码和运行方法。
- [第二阶段：逐步 hidden 最终答错风险探针](stages/stage2/README.md)：采集、恢复、CPU 拟合与结果。
- [第三阶段：检查收益](stages/stage3/README.md)：旧固定第二步 NOW/DELAY_2 与新同题 HIGH/LOW 的目的、运行/恢复命令和结果链接。
- [第一阶段正式结果](stages/stage1/results/current/report.md)：实验一至三的总报告。
- [结果目录索引](stages/stage1/results/README.md)：正式结果与历史试跑的区别。

## 目录结构

```text
VeriServe/
├── README.md                  # 项目入口
├── run_experiments.py         # 第一阶段原命令的兼容入口
├── pyproject.toml / uv.lock   # stage1 原环境
├── veriserve_research/        # stage2/3 按职责组织的实现
├── environments/stage23/      # Python 3.12 独立依赖锁
├── tests/                    # 兼容入口、CPU 自检和科学结果回归
└── stages/
    ├── stage1/               # 初步流程、配置、结果与原运行记录
    ├── stage2/               # 配置、兼容命令、轨迹与冻结探针材料
    └── stage3/               # 配置、兼容命令和两类干预结果
```

第一阶段内另保留迁移时的 `pyproject.toml`、`uv.lock`、`.python-version` 环境快照，供归档和重建。日常环境管理仍从项目根目录进行。

## 运行第一阶段

从项目根目录执行：

```bash
uv sync --python 3.11 --frozen
.venv/bin/python run_experiments.py --suite local_core --through exp3 --config configs/4060_prm.yaml --resume
```

原来的启动命令仍可使用；相对配置路径从 `stages/stage1/` 读取，继续写 `configs/4060_prm.yaml` 即可。

入口会切换到第一阶段目录后执行原来的脚本，因此仍使用原来的数据清单、配置哈希和状态文件。迁移本身不会生成新的实验编号。`--resume` 会复用已完成的生成路径；部分测试与缓存耗时基准仍可能重新测量。

仅查看帮助或运行 CPU 测试：

```bash
.venv/bin/python run_experiments.py --help
.venv/bin/python -m pytest -q
```

## 当前进度

实验一、二已完成；实验三已完成 B/C 与缓存测量。原严格一致性记录为 `BLOCKED_A`，其唯一分叉已有 [BF16 数值差异诊断](stages/stage1/results/diagnostics/exp3_test525_20260927/report.md)，不阻断使用 B 的后续研究。实验四已完成 100 题完整策略比较及 384 条局部分支，见[实验四报告](stages/stage1/results/exp4_current/report.md)；实验五至八尚未实施。

继续实验四（自动引用冻结的前三项依赖，单独保存新结果）：

```bash
.venv/bin/python run_experiments.py --suite local_core --through exp4 --config configs/exp4_4060.yaml --resume
```

第二、三阶段的运行和恢复命令见各自 README；第一阶段的依赖快照和原运行记录继续保留。
