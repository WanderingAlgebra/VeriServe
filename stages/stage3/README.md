# 第三阶段：检查时机与返工收益

本阶段实际执行检查和返工，比较不同检查时机下的最终正确率与处理耗时。[stage2](../stage2/README.md) 已验证中途向量中的最终答错风险信号；本阶段进一步研究这个信号与检查时机的关系。

这里引用的逐步风险探针 B 来自 stage2，预测的是**未经干预的整条轨迹最终答错风险**。它与 stage1 的前缀重算方式 B 无关，也不直接预测当前步骤错误或检查收益。返回[项目首页](../../README.md)。

## 两类实验与当前结果

本阶段有两类独立实验，分别称为“固定第二步”和“同题 HIGH/LOW”。下文用实验名称区分它们；代码重构后的版本统一称为 v2。

| 比较项 | 固定第二步 NOW/DELAY_2 | 同题 HIGH/LOW |
| --- | --- | --- |
| 研究问题 | 在相同第二步状态立即检查，还是再写两步后检查？ | 同一道题，在参考轨迹最高或最低风险的位置检查，有什么差别？ |
| 两条路径 | NOW：第二步结束后立即检查；DELAY_2：再完成两个普通步骤后检查，提前到终点则在终点检查 | HIGH：逐步探针 B 分数最高的位置；LOW：分数最低的位置 |
| 数据来源 | stage2 未使用的 190 题，分为 20 pilot、40 dev、130 test | stage2 原 smoke 前 4 题作 pilot，原 test 200 题作正式比较 |
| 执行起点 | 同一个第二步公共快照 | 两条路径各自从题目 prompt 开始 |
| 允许提交答案的终点判决 | PASS 或 UNCERTAIN | 仅 PASS |
| 已接受前缀再次被打低分 | 记录冲突，按 UNCERTAIN 处理 | 记录冲突，仍为 FAIL，回滚最近 PASS |
| 主计时指标 | `T_postfork_wall`：公共快照之后的处理时间 | `T_request`：从处理题目输入到提交 / 终止的总请求时间 |
| 实验类型标识 | `fixed_step_now_vs_delay` | `within_question_high_low` |
| 配置 | [config.json](config.json) | [high_low_config.json](high_low_config.json) |
| 结果入口 | [配对报告](runs/20261004-109173095d08/report.md) / [CSV](runs/20261004-109173095d08/paired_results.csv) | [配对报告](runs/20261007-9fe830084094/report.md) / [CSV](runs/20261007-9fe830084094/paired_results.csv) / [来源清单](runs/20261007-9fe830084094/manifest.json) |

同题 HIGH/LOW 在 2026-10-07 通过 CPU 自检和 4 题真实 GPU 预运行后固定协议。原 test 200 题中 191 题合格，全部完成配对，共 382 次真实请求；其余 9 题因参考轨迹截断或格式问题排除，原因逐题保留。

| 同题 HIGH/LOW 路径 | 答对 / 共同可评估题数 | 平均总请求时间 |
| --- | ---: | ---: |
| 最高风险位置 HIGH | 142/191 | 12.626 秒 |
| 最低风险位置 LOW | 145/191 | 12.930 秒 |

准确率差和时间差的 95% 配对重抽样区间均跨零，当前证据不能证明哪条路径更优，也不能证明等效。完整失败计数、位置分布及固定第二步实验的结果见各自报告。

## 分数、判决与运行术语

| 名称 | 含义与用途 |
| --- | --- |
| 逐步风险探针 B | stage2 用中途步骤内部向量训练的分类器，分数高表示预测原轨迹最终答错风险高 |
| 进度基线 C | stage2 只使用步号和已生成 token 数的基线，用于固定第二步实验的风险分组比较 |
| `z` / `q` | 探针的线性输出 / 经 `sigmoid(z)` 转换的答错风险分数；同题位置按 `z` 排序，避免 `q` 接近 0 或 1 时数值饱和 |
| PRM 步骤分数 | 过程验证器给完整步骤的分数，低于 `0.35` 触发低分判决；与探针分数方向和用途不同 |
| PASS / FAIL / UNCERTAIN | 检查通过 / 失败 / 不确定；两类实验的终点提交规则见上表 |
| Math-Verify / gold | 最终答案与标准答案的判等工具 / 标准答案；不参与选择检查位置或返工 |
| 参考轨迹 | 未经 PRM 检查或反馈干预的完整解题轨迹，用来预先选择 HIGH/LOW 位置 |
| prompt / prefill / decode | 题目输入 / 计算已有输入 / 逐 token 续写 |
| KV / 检查点 / 快照 | 模型计算缓存 / 最近 PASS 的恢复位置 / 保存的原始 token 与 CPU 特征 |
| pending / lookahead token | 为识别步骤结束而已生成、但跨越逻辑步骤边界的 token，仍保留并计费 |
| `near_end` | 接近最终答案的普通推理步骤标记，不表示已经进入最终答案行 |
| pilot / dev / test | GPU 预运行 / 开发划分 / 正式评估划分；预运行用于检查实现 |

固定第二步报告中的 B/C low、mid、high 是**不同题目快照的风险分组**；同题 HIGH/LOW 是**一道题内的两个步骤位置**，两者不是同一种比较。

## 同题 HIGH/LOW：最高与最低风险位置

### 数据与位置选择

复用 [stage2 来源清单](../stage2/runs/20261004-36fe9009f1f5/manifest.json)、原记录、`feature_files` 指向的 NPZ 特征文件、逐步探针 B 参数和 `hidden_states[19]`。按 NPZ 的 `layers` / `positions` 核对步骤端点，优先复用已有分数与第 19 层特征；只有特征缺失或参考 token 改变时，才补提取对应独立原始前缀。

正式题保持 stage2 原 test 的 200 题顺序，不按原答案对错过滤。这是在已用留出题上的探索性诊断。位置选择规则如下：

- 参考轨迹必须完整、格式合法且未经检查与反馈干预。
- 候选只包含已闭合的普通 `Step N`，排除 boxed、明确最终答案与 `Final answer`；至少有两个候选步骤才合格，保留 `near_end` 普通步骤。
- 按逐步探针 B 的 `z` 选择最高 / 最低位置；同分取最早步骤。全同分题仍保留，即使 HIGH=LOW 也各自实跑。
- 干预前固定位置，在 `selections/<key>.json` 保存步骤、原始 token 前缀、`q/z`、`near_end` 和来源哈希；截断或格式排除保留原因。

**选择最高 / 最低位置需要看完整参考轨迹，因此本实验是离线位置诊断。** 分数与步骤位置可能共同影响结果，本轮不能独立证明探针超越位置、定位局部错误，或回答中途检查是否优于不检查 / 随机检查。区间为 `[0,0]` 也不能证明总体等效；失败早停带来的更短时间不能单独解释为收益，不因结果方向挑样本或提前停止。

### 检查与返工流程

两条路径独立处理题目输入，使用独立 KV 和预算；按题交替 HIGH→LOW / LOW→HIGH 的执行顺序。首检前逐 token 核对参考前缀。只有发现不兼容时，才用同一 stage3 流式实现重新采集该题无干预参考，保存旧材料和原因，再固定位置。

每条路径只有一次主动中途检查，之后仅执行终点检查与必要返工：

1. PASS 时保存当前通过检查的原始 token 前缀与 KV，然后继续。
2. FAIL 时回滚最近 PASS 检查点并加入确定性反馈；尚无 PASS 时只保留题目 prompt，从 Step 1 重来。
3. 任何完整步骤低于 `0.35` 均为 FAIL。即使低分出现在已接受前缀，也记录 `checkpoint_conflict` 并回滚最近 PASS；无可评分步骤或检查结果不可用时不能 PASS。
4. `Final answer` 纳入 PRM 检查。仅终点 PASS 允许提交，再由 Math-Verify 评分；否则继续预算内返工，或因预算 / 格式限制终止。

识别步骤边界时的 pending token 保留并计费，PRM 输入包括步骤尾部，检查时暂停继续生成。PASS 检查点保留覆盖已评分尾部的最短原 token 前缀及 KV。若跨界 token 携带下一标题片段，返工时仅从推理文本中裁去该未评分片段，原 token 和计费保留。[协议固定前的 pilot 诊断](runs/20261007-acde3028d49c/protocol_correction.json)记录了修正；该次历史 pilot 不计入正式 run 结果。

### 计时与统计

`T_request` 从每条路径的 prompt prefill 开始，至提交或终止结束，测量同步后的真实墙钟时间。它包括生成、PRM、真实 KV 裁剪与回滚、反馈 prefill、控制和相同的路径内原子持久化；模型加载与预热、离线参考与特征、Git 备份、标准答案评分不计入，离线开销另存。

质量指标 `Y=1` 要求终点允许提交且 Math-Verify 判对。预算、格式或未通过检查的算法失败计为 0；标准答案 / 评分错误单列。质量只用双方共同可评估的题目，时间只比较同 GPU、同实现的有效配对。

配对差值为 `ΔY=Y_HIGH−Y_LOW`、`ΔT=T_LOW−T_HIGH`，正值分别表示 HIGH 更准、更快。固定种子，按题配对 bootstrap（有放回重抽样）1000 次估计区间。`T_request` 与下文的 `T_postfork_wall` 计时起点不同，不能混为一个指标。

## 固定第二步：立即检查与推迟两步

固定 stage2 的 7B greedy 生成器和探针参数，在原 `unused_ids` 顺序划分的 20 道 pilot、40 道 dev、130 道 test 上执行。每个合格的第二步公共快照均运行 NOW 与 DELAY_2 两条路径，不用风险分数筛掉某个动作。

开发题只根据合格快照的逐步探针 B / 进度基线 C 分数确定低、中、高三组的三分位边界，不使用动作结果；正式 test 前固定分组边界并成功备份。这些分组用于比较不同风险快照的检查收益，区别于同题 HIGH/LOW 的步骤极值选择。

本实验沿用 stage1 的 PRM 量化方式、阈值 `0.35` 与确定性通用反馈。适配层把跨 token 边界、暂存于 pending 的步骤尾部归回已接受步骤。已接受前缀低分仍按 UNCERTAIN 处理，`Final answer` 纳入验证；终点 PASS 或 UNCERTAIN 均可提交。没有历史检查点时，反馈将原 `Steps 1–0` 说明替换为“从 Step 1 重新开始”。

`T_postfork_wall` 从公共快照之后开始计时。公共生成、离线特征 / 探针与实验分叉恢复单独记录；FAIL 后的真实恢复耗时保留。路径内解析、控制及检查时原子状态 / 事件持久化计入；模型下载 / 加载、外部 Git 备份与事后 Math-Verify 评分不计入。

## 共有模型与预算

生成器为 Qwen2.5-7B-Instruct，采用 greedy 解码、BF16 精度、SDPA 注意力、每批 1 题。过程验证器为 Qwen2.5-Math-PRM-7B，采用 NF4 4-bit 量化、BF16 计算与 double quant；精确版本见两份配置。

| 预算项 | 上限 |
| --- | ---: |
| 每次尝试新生成 token | 4096 |
| 一条请求累计生成 token | 8192 |
| 上下文 token / 单次 PRM 输入 token | 各 8192 |
| PRM 调用次数 | 4 |
| 回滚次数 | 2 |
| 累计非 decode 前向输入 token | 65536 |

被撤销的 token 仍计费；回滚只清零本次尝试的生成计数，不清零累计预算。各动作执行前检查预算。

## 运行当前实现（v2）

以下命令从仓库根目录、在已准备好的 stage2/3 环境中执行，见[环境说明](../../README.md#环境与运行)。实现位于 `veriserve_research/intervention` 和 `analysis`。

**两种实验在当前代码下都创建或恢复 `stages/stage3/runs/v2-<seed>-<identity>/`。** 本页的两个已保存结果目录属于历史实现，不能用下面命令直接恢复。当前 `stages.stage3.run_timing` 和 `stages.stage3.high_low` 兼容入口也转发到 v2。

### 同题 HIGH/LOW

默认配置为 `stages/stage3/high_low_config.json`，也可显式传入 `--config`。按顺序执行：

```bash
python -m veriserve_research intervene --experiment high-low --self-check
python -m veriserve_research intervene --experiment high-low --phase prepare
python -m veriserve_research intervene --experiment high-low --phase pilot --resume
python -m veriserve_research intervene --experiment high-low --phase freeze --resume
python -m veriserve_research intervene --experiment high-low --phase test --resume
python -m veriserve_research intervene --experiment high-low --phase analyze --resume
```

CPU 自检覆盖位置选择、端点、一次主动中途检查、PASS 检查点、FAIL 截断与反馈、累计预算、评分和差值方向。4 题真实 GPU pilot 另核对首检 token 一致、PASS 暂停继续，以及强制 FAIL 后真实 KV 裁剪、保留和增量 prefill；强制 FAIL 自检不计入研究结果。仅当四题 GPU pilot 通过才允许 `freeze` 固定协议和正式执行。没有 BF16 CUDA 时保存未执行状态，NA 不能当作结果。

### 固定第二步

默认配置为 `stages/stage3/config.json`。第一次 pilot 暂停后，使用下一条命令在独立进程恢复：

```bash
python -m veriserve_research intervene --experiment fixed-step --phase prepare --resume
python -m veriserve_research intervene --experiment fixed-step --phase self-check
python -m veriserve_research intervene --experiment fixed-step --phase backup --resume
python -m veriserve_research intervene --experiment fixed-step --phase pilot --resume --stop-after 1
python -m veriserve_research intervene --experiment fixed-step --phase pilot --resume
python -m veriserve_research intervene --experiment fixed-step --phase dev --resume
python -m veriserve_research intervene --experiment fixed-step --phase test --resume
python -m veriserve_research intervene --experiment fixed-step --phase analyze --resume
python -m veriserve_research intervene --experiment fixed-step --phase backup --resume
```

`prepare` 在模型下载前固定 PRM 版本 SHA，再确定 v2 目录。test 前会核对 GPU pilot、恢复检查和协议 / 分组备份。只有本地固定文件或旧成功备份不能放行 test，需先保存待备份状态并完成新备份。

## 重分析、恢复与备份

### 重分析已有材料

当前代码可直接读取历史或 v2 材料，不覆盖原报告。默认输出到对应 run 的 `analyses/<分析哈希>/`：

```bash
python -m veriserve_research analyze --run stages/stage3/runs/20261007-9fe830084094
python -m veriserve_research analyze --run stages/stage3/runs/20261004-109173095d08
```

分析仅用 CPU，pilot/dev/test 分开报告，未实际执行的阶段标为未执行 / NA。

### 恢复 v2 执行与自动备份

使用相同源码、配置和材料，重复对应阶段命令并加 `--resume`。完整有效配对保留；基础设施中断后的原配对文件移入 `interrupted_pairs/`，两个路径一起重新计时。固定第二步使用原 token 快照；同题 HIGH/LOW 重新运行两个完整请求。

两类实验每 10 个完整题报告进度，并在计时外做 Git/LFS 备份，使用现有分支 `experiment/step-hidden-probe`。固定第二步备份包含 stage3、当前实现及必要环境 / LFS 元数据；推送失败停止下一批，恢复先重试备份。第一批完整配对从独立临时 checkout 取回快照、两条路径和 NPZ，核对 SHA-256。损坏 JSON 保留原字节并优先从已提交备份恢复，不能恢复则停止。

同题 HIGH/LOW 在无远端权限时保存本地状态。它的备份失败处理与固定第二步不同，不能把两者当作同一个恢复协议。长期快照只保存 token 和 CPU 特征，不保存 GPU KV，也不改写 stage1/stage2 的原材料。

### 恢复历史执行

先按[历史恢复说明](../../README.md#历史材料分析与恢复)，在独立 checkout/worktree 使用[清单](../../docs/legacy_runs.json)中与目标 run 对应的 `source_commit`，放回最新已核对的 stage3 材料、stage2 依赖和对应环境。历史 HIGH/LOW 还要求清单中的原目录位置。

**只有在对应历史 checkout 中**，才使用以下原入口恢复。`--phase` 选择目标 run 尚需执行的阶段，示例为 test：

```bash
# 固定第二步的历史实现
python -m stages.stage3.run_timing --config stages/stage3/config.json --phase test --resume
# 同题 HIGH/LOW 的历史实现，需另用它对应的源码提交
python -m stages.stage3.high_low --config stages/stage3/high_low_config.json --phase test --resume
```

### 文件索引

历史目录为 `<seed>-<config_hash>`，当前目录为 `v2-<seed>-<identity>`，均位于 `stages/stage3/runs/`。

| 文件 / 目录 | 用途 |
| --- | --- |
| `report.md`、`metrics.json`、`paired_results.csv` | 结论、指标与逐题配对结果 |
| `manifest.json` | 原题字段、原顺序 ID、配置、探针、来源哈希和环境 |
| `protocol_frozen.json` | 正式执行前固定的协议；固定步骤的风险分组材料另随 run 保存 |
| `selections/<key>.json` | 同题 HIGH/LOW 的参考来源、候选分数与固定位置 |
| `snapshots/<key>.json/.npz` | 固定第二步的原 token 前缀、pending 和选定层特征 |
| `arms/<key>.HIGH.json` / `.LOW.json` | 同题两条路径的 token、PRM 判决、预算、终止原因与时间 |
| `arms/<key>.NOW.json` / `.DELAY_2.json` | 固定第二步两条路径的对应记录 |
| `interrupted_pairs/` | 中断后保留的原配对文件 |
