# VeriServe

VeriServe 研究数学推理过程中何时检查、检查失败后如何返工，以及准确率的变化是否值得额外处理成本。三个阶段依次研究检查流程、答错风险预测和真实检查时机。

## 从哪里开始看

| 阶段 | 研究问题 | 当前结果与入口 |
| --- | --- | --- |
| [stage1：检查与返工流程](stages/stage1/README.md) | 验证器能否发现错误？保留旧推理、复用缓存和改变检查间隔有什么影响？ | 实验一至四已完成本轮测量；见[结果索引](stages/stage1/results/README.md)和仓库内的[实验四补充分析](stages/stage1/results/diagnostics/exp4_further_20260927/report.md) |
| [stage2：最终答错风险预测](stages/stage2/README.md) | 从中途步骤的模型内部向量，能否预测整条推理最终答错？ | 300 道训练与测试题已采集，另有 10 道流程检查题；逐步探针在整体中途评估中超过进度基线；见[报告](stages/stage2/runs/20261004-36fe9009f1f5/report.md) |
| [stage3：检查时机比较](stages/stage3/README.md) | 立即或推迟检查、在同题不同风险位置检查，最终质量和耗时有什么不同？ | 同题 HIGH/LOW 已完成 191 题配对，准确率差与时间差的区间均跨零；见[报告](stages/stage3/runs/20261007-9fe830084094/report.md)；固定第二步实验另有[记录](stages/stage3/runs/20261004-109173095d08/report.md) |

只看研究结论时，从各阶段 README 的结果概览进入，再阅读对应报告。stage1 的正式结果、补充分析和历史试跑统一列在[结果索引](stages/stage1/results/README.md)。

## 术语与编号

| 名称 | 含义 |
| --- | --- |
| 阶段 / stage | 仓库中的三组研究；stage1 方案内部还单独编号了实验一至八，不能把“实验三”当作 stage3 |
| 推理轨迹 | 模型对一道题生成的全部推理和最终答案；步骤是轨迹中的一个 `Step N` |
| 风险探针 / probe | 读取生成模型内部向量的小型分类器，预测未经检查与返工的轨迹最终答错风险 |
| 过程验证器 / PRM | 给推理步骤打分的模型，用于作出检查判决；与风险探针承担不同任务 |
| PASS / FAIL / UNCERTAIN | 验证器的通过、失败、不确定判决；最终答案是否正确由标准答案另行评分 |
| run | 一次实验执行及其材料目录，保存配置、源码版本、原始记录和结果 |
| smoke / pilot | 少量真实 GPU 题目上的流程检查 / 预运行，用于验证实现；CPU 自检不能替代它们 |
| 冻结 | 在正式评估前固定某个对象；正文会具体说明是题目划分、探针参数、协议还是检查位置 |

**A/B/C 只在各自实验内有效。** stage1 实验三的 A/B/C 是缓存续写、前缀重算、从头重做三种恢复方式；stage2 的 A/B/C 是末尾探针、逐步探针、进度基线。stage3 引用的是 stage2 的探针。

## 当前进度

stage1 实验一、二已完成；实验三已完成“前缀重算与从头重做”的对照和缓存测量。缓存续写方式 A 的原严格一致性记录为 `BLOCKED_A`，唯一分叉已有 [BF16 数值差异诊断](stages/stage1/results/diagnostics/exp3_test525_20260927/report.md)，不阻断使用前缀重算方式 B 的后续研究。实验四已完成 100 题完整策略比较及 384 条局部分支。

stage1 方案中的实验五至八尚未按该方案实施。stage2 的风险探针与 stage3 的时机比较使用各自的模型、数据和协议，进度按各自 README 记录。

## 环境与运行

以下命令均从项目根目录执行。stage1 使用根目录的 Python 3.11 环境；stage2/3 使用 `environments/stage23/` 的独立 Python 3.12 环境。

### stage1

```bash
uv sync --python 3.11 --frozen
.venv/bin/python run_experiments.py --suite local_core --through exp3 --config configs/4060_prm.yaml --resume
```

根入口自动进入 `stages/stage1/`，因此配置参数仍写 `configs/4060_prm.yaml`。继续实验四使用：

```bash
.venv/bin/python run_experiments.py --suite local_core --through exp4 --config configs/exp4_4060.yaml --resume
```

`--resume` 复用已完成的生成路径；部分测试与缓存耗时基准仍可能重新测量。环境准备和辅助脚本见 [stage1 README](stages/stage1/README.md#环境与运行)。

### stage2/3

新机器使用独立依赖锁安装环境并做 CPU 自检：

```bash
uv sync --project environments/stage23 --python 3.12 --frozen
uv run --project environments/stage23 --frozen python -m veriserve_research self-check
uv run --project environments/stage23 --frozen python -m pytest -q
```

在已准备好的 stage2/3 环境中，统一入口如下。若使用上面的 uv 环境，将 `python` 替换为 `uv run --project environments/stage23 --frozen python`。

```bash
python -m veriserve_research --help
python -m veriserve_research probe --config stages/stage2/config.json --prepare-only
python -m veriserve_research intervene --experiment fixed-step --phase prepare
python -m veriserve_research intervene --experiment high-low --phase prepare
python -m veriserve_research analyze --run stages/stage3/runs/20261007-9fe830084094
```

`probe` 负责采集轨迹与拟合探针；`intervene` 负责两类检查时机实验；`analyze` 读取保存材料并输出独立分析。正式执行顺序、GPU 检查和自动 Git/LFS 备份要求见各阶段 README。

## 历史材料、分析与恢复

**实验类型与代码版本是两个独立概念。** 固定第二步和同题 HIGH/LOW 是两种实验类型；两种类型都可以由当前 v2 实现创建新的执行记录。

| 要做的事 | 使用方式 |
| --- | --- |
| 查看已完成结果 | 直接打开各阶段链接的报告 |
| 重分析历史或 v2 材料 | 当前代码的 `analyze --run <材料目录>` |
| 启动或恢复当前实现的执行 | 各阶段的 v2 命令；`--resume` 只恢复相同执行身份的 v2 记录 |
| 继续历史实现的执行 | 使用[历史 run 与源码提交清单](docs/legacy_runs.json)中的 `source_commit` 和相应原入口 |

当前实现创建 `v2-<seed>-<identity>` 目录。原 `python -m stages.stage2.run_probe`、`stages.stage3.run_timing`、`stages.stage3.high_low` 命令是兼容入口，也转发到当前实现；在当前提交执行这些命令不会恢复历史目录。

重分析默认写入原 run 的 `analyses/<分析哈希>/`，可用 `--output` 指定其他目录。原始 token、探针、配对记录及原报告不被覆盖。分析哈希包含分析源码和输入文件校验值，`analysis_provenance.json` 记录来源。旧绝对材料路径按当前仓库位置解析并核对 SHA-256；v2 材料使用仓库相对路径。

执行源码和分析源码分别记录 SHA-256。修改报告或配对统计会产生新的分析目录；修改采集、拟合、选择、干预或共用执行依赖会产生新的执行身份。

恢复历史执行时，在独立 checkout/worktree 使用清单核对过的 `source_commit` 和对应环境。源码提交与最新材料备份提交可能不同，需要把最新已核对的 run 材料及 stage2 依赖放回原相对位置，并保留文件原字节。清单中的 `source_hashes` 用于复核，不能将当前代码哈希写回旧 manifest 绕过校验。

历史 HIGH/LOW 实现还要求清单中 `original_root` 记录的原目录位置；当前代码对历史材料的重分析已支持移动仓库。历史执行的原命令分别保留在 [stage2](stages/stage2/README.md#恢复历史执行) 和 [stage3](stages/stage3/README.md#恢复历史执行) 的恢复说明中。

## 目录结构与实现

```text
VeriServe/
├── README.md                  # 研究概览与导航
├── run_experiments.py         # stage1 兼容入口
├── pyproject.toml / uv.lock   # stage1 环境
├── veriserve_research/        # stage2/3 当前实现
├── environments/stage23/      # stage2/3 独立依赖锁
├── tests/                     # 兼容入口、自检与结果回归
└── stages/
    ├── stage1/                # 初步流程、配置、结果与原实现
    ├── stage2/                # 配置、轨迹与已固定参数的探针
    └── stage3/                # 配置与两类检查时机实验材料
```

stage2/3 按职责分为 `artifacts`（材料读写、校验与备份）、`trajectory`（token 边界与评分）、`inference`（模型与缓存）、`probe`（采集与拟合）、`intervention`（检查与返工）和 `analysis`（统计与报告）。stage3 直接读取 stage2 材料，必要的 stage1 共用能力通过 `artifacts.legacy` 适配。

stage1 内的 `pyproject.toml`、`uv.lock`、`.python-version` 是迁移时保留的环境快照，日常环境管理仍在根目录进行。

## 实现验证记录

重构保留原实验方法、预算、PRM 判决规则、按题重抽样统计和计时范围。迁移时的 41 项 pytest、探针重拟合和独立 GPU smoke/pilot 均通过，见[验收记录](docs/refactor_validation.json)。这些验证没有冻结正式实验；每个 v2 run 仍需通过自身真实 GPU 检查后才能固定协议并正式执行。

CPU 自检使用临时目录。独立 GPU 验证命令为 `python tests/research_gpu_check.py`，在 `.cache/refactor-validation/` 保存 10 题 smoke、20 题固定步骤 pilot 和 4 题 HIGH/LOW pilot；它直接调用执行函数，不触发自动备份或正式协议冻结。
