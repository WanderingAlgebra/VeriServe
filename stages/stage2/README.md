# 第二阶段：从中途推理预测最终答错风险

本阶段问：**模型还在解题时，能否从它的内部向量预测这条推理最终会答错？**

先让生成模型连续完成解题，过程中不检查、不反馈、不返工。再读取每个正常推理步骤末尾的内部向量（hidden state），训练一个小型分类器，称为风险探针。标签来自整条轨迹的最终答案，分数越高表示预测的最终答错风险越高；它不判断当前步骤是否出错。

返回[项目首页](../../README.md)，或继续阅读 [stage3 的真实检查时机比较](../stage3/README.md)。

## 当前结果

本轮已采集 **300/300 道正式题**，包括 100 道训练题和 200 道测试题，另有 10 道流程检查题（smoke）。结论与原始统计见[完整报告](runs/20261004-36fe9009f1f5/report.md)、[指标](runs/20261004-36fe9009f1f5/metrics.json)和[正式特征远端核验](runs/20261004-36fe9009f1f5/formal_feature_remote_verification.json)。

主要结果是：**逐步风险探针在整体中途评估中，提供了超过随机与简单进度基线的最终答错风险信号。** 下表的 AUROC 衡量分数区分最终答对 / 答错轨迹的能力，随机水平为 0.5；它不是解题正确率。方括号为 95% 区间。

| 比较项 | 如何训练或构造 | 中途位置 AUROC / 差值 |
| --- | --- | --- |
| 末尾探针 A | 用完整轨迹末尾的内部向量训练，再应用到中途位置 | 0.663 [0.611, 0.713] |
| 逐步探针 B | 用中途步骤末尾的内部向量训练 | 0.734 [0.666, 0.795] |
| 进度基线 C | 只用当前步号和已生成 token 数 | 0.583 [0.527, 0.636] |
| 逐步探针减进度基线（B−C） | 同一批题目上的配对差值 | 0.151 [0.095, 0.207] |

两种内部向量探针均选择第 19 层。上述结果使用全部共同中途位置，按题等权；第 1 个有效步骤的 B−C 区间跨零，不能推广为所有早期位置均超过基线。完整轨迹末尾的指标在报告中另列。

| 题目用途 | 计划题数 | 最终答案可评分 | 有有效中途位置 |
| --- | ---: | ---: | ---: |
| 训练 | 100 | 97 | 96 |
| 测试 | 200 | 196 | 191 |

测试的共同中途评估包含 1247 个位置。正式排除 7 道题：截断 3、缺少完整 boxed 答案 2、预测不可解析 1、最终答案格式错误 1。另有 6 道可评分轨迹没有有效中途步骤，材料仍保留。

本阶段只验证这一模型与数据设置下的风险预测能力。它不能直接定位首个错误步骤、决定检查阈值、证明检查收益，或证明在线请求更快；离线重新计算内部向量的时间也不能当作在线低成本。

## 术语与实验设置

本页的 A/B/C 是上表中的探针与基线，**与 stage1 实验三的三种恢复方式无关**。stage3 使用的是这里的逐步探针 B。

| 术语 | 在本阶段的含义 |
| --- | --- |
| 轨迹 / 步骤 | 一道题的完整生成内容 / 其中一个 `Step N` 推理段落 |
| token | 模型读写文本的基本单位，不一定是完整的字或词 |
| hidden state | 模型某一层在指定 token 位置的内部向量，本实验用步骤末尾的位置 |
| 前缀 / 端点 | 截至某位置的原始 token 序列 / 提取内部向量的位置 |
| smoke | 10 道真实 GPU 流程检查题，用来验证特征提取与暂停恢复 |
| 冻结探针 | 固定层、标准化参数和分类器参数，之后才评估测试题 |
| 按题等权 | 一道题不因步骤更多而获得更高总权重 |
| bootstrap | 按题有放回重抽样，用来估计指标与配对差值的不确定性 |

### 模型、数据与拟合

生成器为 Qwen/Qwen2.5-7B-Instruct，使用 BF16 数值精度、SDPA 注意力实现、单卡、每批 1 题和 greedy 解码（每次选择最高分 token）。最多生成 4096 个新 token，总上下文上限为 8192。

数据来自 MATH-500 的原始 `test` split。按 `unique_id` 排序后，用种子 `20261004` 洗牌，依次分出本实验的 100 道训练题、200 道测试题和 10 道 smoke 题；其余 190 道不用。因此，本页的 train/test 是实验内部划分。模型、分词器和数据的精确版本 SHA 见 [config.json](config.json)，首次查询后即固定，不使用浮动 `main`。

末尾探针 A 与逐步探针 B 分别在训练题内部做五折按题交叉验证，选择一个固定层。分类器使用 StandardScaler 标准化和 L2 LogisticRegression，正则化参数 `C=0.1`、`max_iter=2000`；这里的参数 `C` 与进度基线 C 不是同一概念。逐步探针和进度基线的标准化、拟合、验证均按每题总权重 1。

测试题只在保存已固定参数的探针后读取。统计按题重抽样 1000 次，报告配对 AUROC 差；不补抽题、不翻转分数方向、不据测试结果调参。

### 步骤位置与答案评分

步骤端点以原始 `generated_ids` 为准，通过逐前缀解码对齐，不重新分词。跨越文本边界的 token 向前取最后完整内容 token。包含 boxed 或明确最终答案的步骤不作中途位置；最后一个正常推理步骤保留，并标记 `near_end`。轨迹末尾取 EOS（生成结束标记）或控制 token 之前的最后完整内容 token。Qwen2 的 `hidden_states[1..L]` 排除 embedding，最后索引包含 final norm。

评分只读取独立 `Final answer` 行中的最后一个完整 boxed 答案，用 Math-Verify 与标准答案判等。截断、缺失或不完整 boxed、标准 / 预测答案不可解析、解析或评分超时分别排除；不从推理中寻找正确答案。没有有效中途步骤的可评分轨迹仍可用于末尾探针 A 的训练。

## 运行当前实现（v2）

以下命令从仓库根目录、在已准备好的 stage2/3 环境中执行；安装方法见[根目录环境说明](../../README.md#环境与运行)。当前实现位于 `veriserve_research/probe`。

**这些命令创建或恢复 `stages/stage2/runs/v2-<seed>-<identity>/`，不恢复本页展示的历史目录 `20261004-36fe9009f1f5`。** 当前提交中的 `stages.stage2.run_probe` 兼容入口也转发到 v2。

```bash
python -m veriserve_research probe --config stages/stage2/config.json --self-check
python -m veriserve_research probe --config stages/stage2/config.json --prepare-only --resume
python -m veriserve_research probe --config stages/stage2/config.json --backup-only --resume
python -m veriserve_research probe --config stages/stage2/config.json --smoke --resume --stop-after 1
python -m veriserve_research probe --config stages/stage2/config.json --smoke --resume
python -m veriserve_research probe --config stages/stage2/config.json --resume
```

`--stop-after 1` 在第一道 smoke 保存后暂停，下一条命令由独立进程恢复，核对已完成 JSON/NPZ 的哈希并跳过已有生成。完整 smoke 后可再执行一次 `--smoke --resume`，验证 10 题全部跳过。

正式采集前必须通过 10 题逐前缀提取检查、独立进程恢复检查和远端 NPZ 取回核验。首次代码与题目清单推送成功后才启动 smoke 和正式题。自动备份使用现有分支 `experiment/step-hidden-probe`，不新建分支、不强推；服务器需已有 Git 作者信息、Git LFS 和 origin 推送凭据，凭据不写入日志。

只在 CPU 上拟合 / 重评估**已有 v2 特征**时使用：

```bash
python -m veriserve_research probe --config stages/stage2/config.json --fit-only --resume
```

该命令不加载生成模型；标签不足时保存 NA / 不足报告，不借用测试题。若要查看历史特征的分析，使用下一节的 `analyze --run`。

## 重分析与恢复

### 重分析已有材料

以下命令可在当前代码下读取历史材料，不重新生成或拟合，也不覆盖旧报告：

```bash
python -m veriserve_research analyze --run stages/stage2/runs/20261004-36fe9009f1f5
```

新分析默认写入该 run 的 `analyses/<分析哈希>/`。原始 `report.md` 保留为历史结论入口。

### 恢复 v2 执行

在对应源码、配置和材料齐全的仓库中，重复相同命令并加 `--resume`。换机器时先取回 Git LFS 材料：

```bash
git lfs install --local
git lfs pull origin experiment/step-hidden-probe
```

配置、模型 / 数据版本或执行源码变化会产生新的执行身份。正式采集前需完成本 run 的 smoke；不能把旧 run 的检查记录直接当作新 run 已通过。需要改变正式采集实现时保留旧结果，用新执行身份，必要时添加说明性的 `run_note`。

### 恢复历史执行

先按[历史恢复说明](../../README.md#历史材料分析与恢复)，在独立 checkout/worktree 使用[清单](../../docs/legacy_runs.json)记录的 `source_commit`，放回最新已核对的历史材料与对应环境。**只有满足该前提后**，才在历史 checkout 根目录使用原入口：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
# 历史实现中，只使用已保存特征拟合
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```

## 特征检查、备份与文件索引

### 特征提取的数值检查

历史真实 smoke 首次使用“整条轨迹一次前向计算”提取特征，与独立前缀计算相比未通过逐坐标容差：最大绝对误差 1.0、RMS 0.03251。同长度未来内容替换误差为 0；原生 SDPA 后端和 GEMM 精度诊断仍未通过原容差。原失败记录保存在 `single_full_forward_failed_smoke.json` 和 `extraction_diagnostics.json`。

正式方法改为**在每个端点独立计算其原始 token 前缀**，仍使用 BF16、SDPA、`use_cache=False`。不重新分词、不补 token、不改变生成配置。这样避免未来长度改变 BF16 计算形状，代价是重复前向；首次 9 个端点实测约 0.99 秒。

流程检查逐层比较独立 decoder 的末尾向量、未来 token 替换和错一 token 的负对照，记录最大绝对误差（max_abs）与均方根误差（RMS）。固定容差为绝对误差 `1/32` 加相对容差 `0.008`（一个 BF16 相对 ULP），不因结果改变容差。原整段比较保留为诊断，不作为正式方法已通过的检查。

历史本轮的 34 项 CPU 自检、10 道真实逐前缀 smoke、实际暂停恢复、独立科学 / 交叉验证复核和正式 LFS 远端核验已通过。正式采集没有中断或未备份特征。当前 pytest 包含本阶段 CPU 自检与保存材料的结果回归；v2 正式执行仍需本 run 自身的 GPU 检查。

### 保存与备份规则

每题先原子写生成 JSON，再提取特征 NPZ。特征缺失或校验失败时，只重新计算原始 `prompt_ids + generated_ids`，不重新生成答案。损坏 JSON 保留副本，并尝试从本地 Git 已提交版本恢复；没有可恢复 token 或校验不一致时停止并保留全部文件。

smoke 和恢复检查绑定执行源码哈希。正式冻结前修正提取实现时，保留原始 token 与旧特征，再提取和验证；正式冻结后不混入修改后的采集结果。没有中途步骤的 smoke 题记为因果性检查不适用并保留轨迹；全部 smoke 都没有有效位置时不能固定正式运行。

每 25 道正式题提交并常规推送代码、题目清单、JSON 和 NPZ。NPZ 使用原生 Git LFS，按少量层分片避免单文件超过 50 MiB。推送失败停止下一批；修复连接后，`--resume` 先重试备份。至少第一批特征从独立临时仓库通过 `git lfs fetch/checkout` 取回并核对哈希，结果保存到 `backup.json`。

### 文件索引

历史目录名为 `<seed>-<config_hash前12位>`，当前目录名为 `v2-<seed>-<identity>`，都位于 `stages/stage2/runs/`。

| 文件 / 目录 | 用途 |
| --- | --- |
| `report.md`、`metrics.json`、`step_auroc.png` | 结论、指标与逐步曲线 |
| `manifest.json` | 题目原始字段、划分、版本、环境与源码哈希 |
| `records/` | 每题原始 token、文本、评分 / 排除原因、步骤位置、硬件与分阶段耗时 |
| `cv_scores.csv`、`cv_folds.json`、`probe_*.npz/json` | 交叉验证记录与固定好的探针参数 |
| `test_predictions.csv` | 测试题各位置的分数 |
| `smoke_checks.json`、`resume_check.json`、`backup.json` | GPU 流程检查、恢复检查与远端备份状态 |

尚未执行的阶段在报告中标为未执行 / NA。CPU 合成样本仅用于实现自检。

参考：[原论文 §3.1/§4.1/B/C](https://arxiv.org/html/2605.09502v1)、[Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[Math-Verify 安装/解析](https://github.com/huggingface/Math-Verify)。逐步训练、题目权重、提示词和生成上限是本实验适配。
