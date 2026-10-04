# VeriServe

按研究阶段组织实验。当前已有内容全部归入 **第一阶段 `stages/stage1/`**，以后新增阶段放在同级目录。

## 从哪里开始看

- [第一阶段说明](stages/stage1/README.md)：方案、代码和运行方法。
- [第二阶段：逐步 hidden 最终答错风险探针](stages/stage2/README.md)：采集、恢复、CPU 拟合与结果。
- [当前正式结果](stages/stage1/results/current/report.md)：实验一至三的总报告。
- [结果目录索引](stages/stage1/results/README.md)：正式结果与历史试跑的区别。

## 目录结构

```text
VeriServe/
├── README.md                  # 项目入口
├── run_experiments.py         # 第一阶段原命令的兼容入口
├── pyproject.toml / uv.lock   # 当前工作区环境
├── .venv/                    # 已有 Python 环境
├── tests/                    # 兼容入口的启动检查
└── stages/
    └── stage1/               # 第一阶段全部研究内容
        ├── README.md
        ├── 第一阶段实验方案.md
        ├── 验证器替换记录.md
        ├── run_experiments.py
        ├── veriserve/        # 第一阶段实现
        ├── configs/          # 第一阶段配置
        ├── scripts/          # 数据准备、诊断及辅助脚本
        ├── tests/            # 第一阶段测试
        ├── .cache/           # 冻结的数据 ID、模型 revision 与阈值
        ├── results/          # 供阅读的结果入口
        │   ├── README.md
        │   ├── current/      # 实验一至三的正式结果入口
        │   └── exp4_current/ # 实验四的结果入口
        └── runs/             # 按原哈希保存的全部运行记录
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

开始第二阶段时，在 `stages/stage2/` 放它自己的方案、代码、配置和结果。需要不同依赖时再建独立环境，第一阶段保留现有依赖快照和运行记录。
