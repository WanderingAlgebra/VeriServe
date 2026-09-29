# 第一阶段：VeriServe 本地实验

本项目实现[第一阶段实验方案](第一阶段实验方案.md)中的实验一至四。实验一至四已完成本轮测量，实验四见[结果报告](results/exp4_current/report.md)。工程状态 `PASS` 表示程序与预注册门槛通过，不表示研究假设成立。正式答案由 GSM8K 的 `####` 标准答案评分，验证器的 PASS 不参与正确率评分。

阅读结果从[当前正式报告](results/current/report.md)进入；全部运行目录的用途见[结果索引](results/README.md)。返回[项目首页](../../README.md)。

## 环境与运行

下面的启动命令从 **VeriServe 项目根目录**执行，根目录入口会自动进入本阶段：

```bash
uv sync --python 3.11 --frozen
.venv/bin/python run_experiments.py --suite local_core --through exp3 --config configs/4060_prm.yaml --resume
```

上述命令复用本工作区已冻结的题目清单和模型缓存。若从空缓存重建，先执行 `scripts.fetch_assets --config configs/4060.yaml --data-only`、`scripts.make_prm_holdout`，再执行 `scripts.fetch_assets --config configs/4060_prm.yaml --models-only` 和 `scripts.score_prm_dev`；开发题选出的阈值须写入新配置后运行正式命令。数据与模型首次下载需要网络及足够磁盘空间。`.cache/splits_prm.json` 固定题目 ID 和数据 revision，`.cache/assets.json` 固定模型 revision。正式结果在 `runs/<配置及代码哈希>/`。同一命令加 `--resume` 会读取每条路径的原子状态文件，跳过已完成路径。修改配置、代码、提示词或模型 revision 会得到新 run 目录。

本文件中的 `configs/`、`.cache/` 和 `runs/` 均相对于 `stages/stage1/`。本阶段的依赖文件是迁移时保留的环境快照，日常使用根目录的 `.venv`。

`--through exp1` 只完成执行器与硬件测试，`--through exp2` 完成验证器诊断和反馈对照，`--through exp3` 继续 B/C 与 A/B。若验证器未达到预注册门槛，后续评估停止并写入 `BLOCKED` 报告。

实验四从项目根目录启动，`exp4_4060.yaml` 指明冻结的源运行；保留原代码与数据哈希校验后引用旧基线，只计算新增策略、历史快照和局部分支：

```bash
.venv/bin/python run_experiments.py --suite local_core --through exp4 --config configs/exp4_4060.yaml --resume
```

在 `stages/stage1/` 内也可使用 `../../.venv/bin/python -m scripts.run_exp4 --config configs/exp4_4060.yaml --resume`。新增代码集中在 `scripts/run_exp4.py`，沿用 `RunManager` 的验证、记录和预算逻辑，前三项核心代码保持原版本。

根目录入口结束后还会运行 `scripts/audit_exp4.py`，核对来源哈希、配对历史、随机种子和预算计数，并输出独立评价种子的动作配对表与探索性分层表。只重新核对已完成结果时，在第一阶段目录运行 `../../.venv/bin/python -m scripts.audit_exp4 results/exp4_current`。

首轮通用验证器使用 `configs/4060_1024.yaml`，因诊断门槛未通过而停止；结果见 `runs/2549b9dd5d948fe5/report.md`。替换后的过程奖励验证器在 2048 token 容量测试中通过，本轮正式结果见 `runs/3660363ff392fea6/report.md`。

替代验证器的独立配置、数据隔离和反馈能力限制见[验证器替换记录](验证器替换记录.md)。

辅助脚本在 **本阶段目录**执行，使用根目录的 Python 环境。以下命令用于准备模型并重做开发评分；现有结果的查看不需要执行它们：

```bash
cd stages/stage1
../../.venv/bin/python -m scripts.fetch_assets --config configs/4060_prm.yaml --models-only
../../.venv/bin/python -m scripts.score_prm_dev
../../.venv/bin/python run_experiments.py --suite local_core --through exp3 --config configs/4060_prm.yaml --resume
```

开发题选出的阈值记录在 `.cache/prm_dev_threshold.json`，当前已写入 `configs/4060_prm.yaml`。新配置读取独立的冻结题目清单，不覆盖首轮报告。

## 结果文件

- `execution_checks.json`、`hardware_profile.json`、`time_estimate.json`：确定性测试、GPU 容量、时间粗估。
- `exp1/`、`exp2a/`、`exp2b_dev/`、`exp2b_eval/`、`exp3/`：各模块状态、日志和报告。
- `states/`：每条路径完整可恢复状态；`events.jsonl`、`traces.jsonl`、`checkpoints.jsonl`、`verifications.jsonl` 由状态重建，防止断点续跑重复记账。
- `phase_times.jsonl`：离线波次与模型加载的墙钟时间，不能解释为线上请求完成时间。

实验二的 100 道测试题仅使用冻结的 `unchecked`、`endpoint`、`k2`。开发题额外比较 `k1`、`k4`，不据测试集选择间隔。实验三的 B/C 只报告发生真实 FAIL 后的条件结果；A/B 在同一模型与 token 序列上比较缓存裁剪后续写和完整前缀重算。
