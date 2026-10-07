# VeriServe

按研究阶段组织实验：stage1 初步流程，stage2 逐步探针，stage3 检查收益。各阶段代码、配置和原始运行记录保留在自己的目录。

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
├── pyproject.toml / uv.lock   # 当前工作区环境
├── tests/                    # 兼容入口的启动检查
└── stages/
    ├── stage1/               # 初步流程、配置、结果与原运行记录
    ├── stage2/               # 步骤边界、hidden 特征、冻结探针与结果
    └── stage3/               # 两类检查收益实验，各自独立的 runs/
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
