# 第一阶段：检查与返工流程

本阶段研究数学解题时的检查与返工流程：验证器能否发现错误、失败后是否值得保留旧推理，以及检查间隔怎样影响正确率和成本。[实验方案](第一阶段实验方案.md)中的实验一至四已完成本轮测量。

阅读结果从[结果索引](results/README.md)进入。原始运行目录未纳入 Git，`results/current/` 和 `results/exp4_current/` 需本地运行材料才能打开；仓库内可直接阅读[实验三数值诊断](results/diagnostics/exp3_test525_20260927/report.md)和[实验四补充分析](results/diagnostics/exp4_further_20260927/report.md)。返回[项目首页](../../README.md)。

## 实验与结果概览

| 实验 | 研究内容 | 当前状态 |
| --- | --- | --- |
| 一 | 执行器、可恢复状态与硬件容量 | 已完成本轮检查 |
| 二 | 验证器诊断，以及不检查、终点检查、固定间隔检查的比较 | 已完成；100 道正式题使用预先固定的三种策略 |
| 三 | 失败后保留旧推理的作用，以及复用计算缓存的正确性与耗时 | 前缀重算 / 从头重做对照和缓存测量完成；新增 100 题缓存续写 / 前缀重算完整请求对照已完成 |
| 四 | 完整检查策略与同一历史状态下不同检查时机的比较 | 已完成 100 题完整策略比较和 384 条局部分支 |

实验三的唯一续写分叉已有 [BF16 数值差异诊断](results/diagnostics/exp3_test525_20260927/report.md)。原严格检查记录保留，后续使用前缀重算方式的研究可以继续。实验四的本轮答对数为：不检查 49/100、终点检查 65/100、固定间隔 62/100、随机检查 60/100；成本更低的策略不能因此认定质量相等。

工程状态 `PASS` 表示程序与预注册门槛通过。验证器的 `PASS` 表示它接受当前推理。**这两者都不等于最终答案正确**；最终答案按 GSM8K 的 `####` 标准答案另行评分。

## 本阶段的名称与代号

过程验证器（PRM）给推理步骤打分，适配器再转换为 PASS（通过）、FAIL（失败）、UNCERTAIN（不确定）及通用反馈。检查点是最近一次 PASS 时保留的推理前缀；失败后可以退回这里继续解题。KV 缓存是模型处理已有 token 时保存的中间计算结果。

实验三的 A/B/C 表示三种**失败恢复方式**，与 stage2 的探针 A/B/C 无关：

| 报告代号 | 本文名称 | 失败后如何继续 |
| --- | --- | --- |
| A | 缓存续写 | 保留最近 PASS 前缀，裁剪并复用对应 KV，处理反馈后续写 |
| B | 前缀重算 | 保留相同文字和反馈，重新计算前缀后续写 |
| C | 从头重做 | 保留题目和反馈，重新生成推理，不保留旧推理文字 |

因此，报告中的 B/C 比较保留旧推理的作用；A/B 比较相同 token 内容下的计算方式。前缀重算只重新计算已有文字，不重新生成被保留的前缀。详细例子见[实验三方案](第一阶段实验方案.md#实验三失败后保留旧推理复用缓存是否有价值)。

| 策略名 | 检查时机 |
| --- | --- |
| `unchecked` | 不检查 |
| `endpoint` | 只在最终答案处检查 |
| `k1` / `k2` / `k4` | 按每 1 / 2 / 4 个新增步骤的固定间隔检查，并保留终点检查 |
| 随机检查 | 实验四中每步以固定概率 `0.25` 检查，并保留终点检查 |

实验二的 100 道测试题只使用固定好的 `unchecked`、`endpoint`、`k2`；开发题额外比较 `k1`、`k4`，不根据测试结果选择间隔。实验四单独在原开发集选择固定间隔，选得 `k4`，不改写实验二配置。

实验三的前缀重算 / 从头重做对照只报告真实 FAIL 之后的条件结果；历史缓存一致性检查使用同一模型和同一 token 输入。新增完整请求对照从相同题目与种子出发，允许恢复后续写分叉，比较最终质量和成本。

2026-10-10 完成新增的[缓存续写 / 前缀重算完整请求对照](results/exp3_ab/107ed12e8165/report.md)：A 答对 **65/100**，B 答对 **64/100**；平均完整请求分别为 **3.376 秒**与 **3.396 秒**。A 的实际前向 token 数少 28.8%，但当前样本未显示明确的正确率或耗时优势，也不能据此认定质量等价。本轮接受已诊断的 BF16 分叉，原 9/10 记录保留。旧快照暂不可访问，因此从题目开始重新运行原 100 道正式题 ID 与请求种子，不是原 38 个 FAIL 快照的重放。

## 环境与运行

以下主命令从 **VeriServe 项目根目录**执行，根入口会自动进入本阶段：

```bash
uv sync --python 3.11 --frozen
.venv/bin/python run_experiments.py --suite local_core --through exp3 --config configs/4060_prm.yaml --resume
```

`--through exp1` 运行执行器与硬件测试；`--through exp2` 继续验证器诊断和反馈对照；`--through exp3` 再继续失败恢复和缓存比较。验证器未达到预注册门槛时，依赖它的后续评估停止并写入 `BLOCKED` 报告。

本页中的 `configs/`、`.cache/` 和 `runs/` 均相对于 `stages/stage1/`。本阶段依赖文件是迁移时的快照，日常使用根目录 `.venv`。

### 实验三的完整请求 A/B 对照

在项目根目录、安装好本项目依赖的 CUDA 环境中执行：

```bash
python -m stages.stage1.scripts.run_ab
```

该入口使用固定模型和数据版本，恢复原 100 道题 ID、请求种子及 `4060_prm.yaml` 的采样和预算配置。两组均每 2 个新增步骤检查一次；通过或不确定时保留缓存继续生成，只有失败恢复时分别复用 KV（A）和重算前缀（B）。A/B 交替先运行，按完整题目配对保存，重复执行会跳过已完成配对；代码、配置或环境变化会生成新目录。

模型同时驻留 GPU。完整请求计时包含初始前缀计算、生成、检查和恢复，排除模型加载、预热、写盘和标准答案评分。两组沿用相同逻辑预算，实际前向 token 数另行记录。结构错误仍会停止；已诊断的 BF16 分叉不要求续写文字完全一致，也不预先认定最终质量等价。本轮环境与原 4060 运行不同，具体版本见报告目录的 `manifest.json`。

### 实验四

```bash
.venv/bin/python run_experiments.py --suite local_core --through exp4 --config configs/exp4_4060.yaml --resume
```

`exp4_4060.yaml` 指定已固定的前三项源运行。校验原代码与数据哈希后引用旧基线，只计算新增策略、历史快照和局部分支。

根入口结束后还会运行 `scripts/audit_exp4.py`，核对来源哈希、配对历史、随机种子和预算计数，输出独立评价种子的动作配对表与探索性分层表。在本阶段目录也可运行 `../../.venv/bin/python -m scripts.run_exp4 --config configs/exp4_4060.yaml --resume`。实现集中在 `scripts/run_exp4.py`，沿用 `RunManager` 的验证、记录和预算逻辑，前三项核心代码保留原版本。

只核对已完成的实验四结果时，在 **本阶段目录**执行：

```bash
../../.venv/bin/python -m scripts.audit_exp4 results/exp4_current
```

### 重新准备数据与模型

已有结果的查看不需要执行本节。主命令默认复用已固定的题目清单和模型缓存；首次下载需要网络和足够磁盘空间。

从空缓存重建时，在 **本阶段目录**使用根目录 Python，按顺序执行：

```bash
cd stages/stage1
../../.venv/bin/python -m scripts.fetch_assets --config configs/4060.yaml --data-only
../../.venv/bin/python -m scripts.make_prm_holdout
../../.venv/bin/python -m scripts.fetch_assets --config configs/4060_prm.yaml --models-only
../../.venv/bin/python -m scripts.score_prm_dev
```

开发题选出的阈值记录在 `.cache/prm_dev_threshold.json`，须写入新配置后再正式运行；现有阈值已写入 `configs/4060_prm.yaml`。新配置读取独立的固定题目清单，不覆盖首轮报告。

`.cache/splits_prm.json` 保存题目 ID 和数据版本，`.cache/assets.json` 保存模型版本。正式结果在 `runs/<配置及代码哈希>/`。`--resume` 读取每条路径的原子状态文件，跳过已完成路径；修改配置、代码、提示词或模型版本会得到新 run 目录。

## 结果文件与历史说明

| 文件 / 目录 | 用途 |
| --- | --- |
| `execution_checks.json`、`hardware_profile.json`、`time_estimate.json` | 确定性检查、GPU 容量与时间粗估 |
| `exp1/`、`exp2a/`、`exp2b_dev/`、`exp2b_eval/`、`exp3/` | 各实验的状态、日志与报告 |
| `states/` | 每条路径的完整可恢复状态 |
| `events.jsonl`、`traces.jsonl`、`checkpoints.jsonl`、`verifications.jsonl` | 从状态重建的事件与检查记录，避免续跑重复记账 |
| `phase_times.jsonl` | 离线波次和模型加载的墙钟时间，不能解释为线上请求完成时间 |

首轮通用验证器使用 `configs/4060_1024.yaml`，未通过诊断门槛，结果保存在 `runs/2549b9dd5d948fe5/report.md`。替换后的过程奖励验证器在 2048 token 容量测试中通过，正式结果为 `runs/3660363ff392fea6/report.md`。替换原因、数据隔离和反馈能力限制见[验证器替换记录](验证器替换记录.md)。
