# 第三阶段：检查收益

实现已迁入 `veriserve_research/intervention` 和 `analysis`，原模块命令保留转发。新实现创建独立 v2 run，原结果不覆盖。继续旧实验使用核对过的原提交；新实现可直接重分析旧 run，见[历史恢复与分析说明](../../README.md#历史材料分析与恢复)。

```bash
python -m veriserve_research self-check
python -m veriserve_research intervene --experiment fixed-step --phase prepare
python -m veriserve_research intervene --experiment high-low --phase prepare
python -m veriserve_research analyze --run stages/stage3/runs/20261007-9fe830084094
```

本阶段比较真实检查与返工后的质量和成本，保留两类独立实验。新实验的目的仅是回答：**同一道题，在原始完整轨迹的 B 最高分步骤检查，与最低分步骤检查，最终准确率和请求总处理成本有什么不同？**

| 实验 | 目的与计时 | 配置 / 入口 | 结果 |
| --- | --- | --- | --- |
| `fixed_step_now_vs_delay`（旧） | 固定第二步公共快照，比较 NOW 与 DELAY_2；使用 `T_postfork_wall` | [config.json](config.json)；`stages.stage3.run_timing` | [既有配对报告](runs/20261004-109173095d08/report.md) / [CSV](runs/20261004-109173095d08/paired_results.csv) |
| `within_question_high_low`（新） | 同题 B 全轨迹最高/最低分位置，各自从 prompt 开始实跑；使用 `T_request` | [high_low_config.json](high_low_config.json)；`stages.stage3.high_low` | [新报告](runs/20261007-9fe830084094/report.md) / [CSV](runs/20261007-9fe830084094/paired_results.csv) / [来源 manifest](runs/20261007-9fe830084094/manifest.json) |

## 同题 HIGH/LOW：一次中途检查与总请求计时

新实验复用 [stage2 来源 manifest](../stage2/runs/20261004-36fe9009f1f5/manifest.json)、原记录与 `feature_files` 指向的 NPZ、原 B 参数及 `hidden_states[19]`。按 NPZ 的 `layers` / `positions` 核对步骤端点，优先复用分数与已有第 19 层特征；只有缺失或参考 token 改变时补提取对应独立原始前缀。pilot 使用原 smoke 前 4 题，仅作自检；正式使用原 test 的 200 题原顺序，不按原答案对错过滤。这是已有留出题上的探索性诊断。

参考须完整、格式合法且未经 PRM/反馈干预；候选只含已闭合的普通 Step N，排除 boxed、明确最终答案与 Final answer，至少两步才合格。保留 near_end 普通步骤；截断和格式排除保留原因。按 B 的线性 logit `z` 选择 HIGH/LOW，避免 sigmoid 饱和；相同 `z` 选最早步骤，全同分仍保留，HIGH=LOW 也各自实跑。干预前在 `selections/<key>.json` 保存步骤、原 token 前缀、`q/z`、near_end 与来源哈希。

两条路径独立 prompt prefill、KV 和预算，按题交替 HIGH→LOW / LOW→HIGH。首检前逐 token 核对原参考；仅在发现不兼容时，用同一 stage3 流式实现重新采集该题无干预参考，保存旧材料与原因，再冻结位置。跨边界 lookahead/pending token 保留并计费，PRM 包含步骤尾部，检查时暂停继续生成。

PASS 检查点保留覆盖已评分尾部的最短原 token 前缀及 KV；跨 token 携带的下一标题片段在返工推理文本中裁去，原 token 与计费仍保留。[冻结前 pilot 诊断](runs/20261007-acde3028d49c/protocol_correction.json)记录了此边界修正，旧 pilot 不进入新 run 的结果。

每路径只有一次主动中途检查，随后只做终点检查与返工。PASS 保存最近 PASS 前缀和 KV 后继续；FAIL 回滚该检查点，无 PASS 时保留题目 prompt，从 Step 1 重来，追加确定性反馈增量 prefill。任何完整步骤低于 0.35 均为 FAIL；旧接受前缀低分标 `checkpoint_conflict`，仍回滚最近 PASS，直到终点 PASS 或预算/格式终止。无可评分步骤或结果不可用不能 PASS。Final answer 纳入 PRM，只有终点 PASS 可提交并由 Math-Verify 评分。**旧实验的 UNCERTAIN 提交默认规则保持原样；新 HIGH/LOW 两路径都使用上述严格规则。**

生成器、PRM 与精确 revision 见配置：Qwen2.5-7B-Instruct，greedy/BF16/SDPA/batch=1；Qwen2.5-Math-PRM-7B，NF4/BF16 compute/double quant。预算保持每次尝试 4096、累计生成 8192、上下文和单次 PRM 输入 8192、PRM 最多 4 次、回滚最多 2 次、累计非 decode 前向输入 65536。撤销 token 仍计费，回滚只清零尝试计数，各动作前检查预算。

`T_request` 为各路径从 prompt prefill 到提交/终止的真实同步 wall time，包含全部生成、PRM、真实 KV crop/回滚、反馈 prefill、控制与相同的臂内原子持久化。加载/warmup、离线参考/特征、Git 和 gold 评分不计入，离线开销另存。它与旧快照后 `T_postfork_wall` 是不同指标。Y=终点允许提交且 Math-Verify 判对；预算、格式或未通过检查的算法失败为 0，gold/评分错误单列。质量只用共同可评估配对；同 GPU/实现配对统计 `ΔY=Y_HIGH−Y_LOW`、`ΔT=T_LOW−T_HIGH`，正值分别表示 HIGH 更准/更快；沿用固定种子 1000 次按题配对 bootstrap。

在仓库根目录按顺序运行；默认配置为 `stages/stage3/high_low_config.json`，也可显式传 `--config`：

```bash
python -m stages.stage3.high_low --self-check
python -m stages.stage3.high_low --phase pilot --resume
python -m stages.stage3.high_low --phase freeze --resume
python -m stages.stage3.high_low --phase test --resume
python -m stages.stage3.high_low --phase analyze --resume
```

`--self-check` 复用旧 CPU 检查并补位置/端点、单次中途检查、PASS checkpoint、FAIL 截断/反馈、累计预算与评分/差值方向断言。真实 GPU pilot 另核对首检 token 一致、PASS 暂停继续与强制 FAIL 的真实 KV crop/保留/增量 prefill；强制 FAIL 自检不混入科学结果。只有四题 GPU pilot 通过才允许 `freeze` / 正式执行。没有 BF16 CUDA 时完成代码与 CPU 自检，保存未执行状态，不能把 NA 当结果。

2026-10-07 的新 run 已在同一 RTX 4090 上通过 CPU 自检和 4 题真实 GPU pilot，再冻结协议。原 test 200 题中 191 题合格，全部完成 HIGH/LOW 配对（382 次真实请求）；其余 9 题因参考轨迹截断或格式问题排除，原因逐题保留。HIGH 答对 142/191，LOW 答对 145/191，平均 `T_request` 分别为 12.626 / 12.930 秒；准确率差与时间差的 95% 配对 bootstrap 区间均跨 0，不能据此证明等效或哪条路径更优。完整统计、失败计数及位置分布见上表的新报告。

恢复使用同一命令加 `--resume`：完整有效配对保留；基础设施中断的旧文件移入 `interrupted_pairs/`，该题两个请求时钟一起重新实跑。每 10 个完整题报告进度并在计时外沿用 Git/LFS 备份；无远端权限则保存本地状态。原 token、PRM 判决、预算和终止原因在 `arms/<key>.HIGH.json` / `.LOW.json`，汇总为 `paired_results.csv`、`metrics.json`、`report.md`，新 run 不覆盖旧 run。

最高/最低位置需要看全参考轨迹，属于离线位置诊断。B 预测未经干预轨迹最终答错风险，不是局部错误或检查收益。分数与位置可能共同影响结果，本轮不能独立证明 B 超越位置或定位局部错误，也不能回答中途检查是否优于不检查/随机位置。[0,0] 正确性区间不能证明总体等效，失败早停快不能单独算收益；不因结果方向挑样本或提前停。

## 固定第二步 NOW/DELAY_2：保留原运行与恢复语义

固定第二阶段 7B greedy 生成器与 B/C 探针，在原 `unused_ids` 顺序划分的
20 道 pilot、40 道 dev、130 道 test 上执行 Step 2 公共快照的 NOW / DELAY_2 配对。
B 表示未经干预完整轨迹最终答错风险；每个合格快照均执行两个动作。
PRM 沿用第一阶段 NF4 / BF16 compute / double quant 与确定性通用反馈，阈值固定 0.35。
第三阶段薄包装保留跨原始 token 边界落在 pending 中的步骤尾部，
将其归回已接受步骤；旧前缀低分仍为 UNCERTAIN，Final answer 纳入验证。
checkpoint=0 的反馈只将原 Steps 1–0 说明替换为重新从 Step 1 开始。

在仓库根目录使用已有 Python：

```bash
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase prepare --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase self-check
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase backup --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase pilot --resume --stop-after 1
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase pilot --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase dev --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase test --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase analyze --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase backup --resume
```

`prepare` 在模型下载前固定 PRM SHA，再确定 `runs/<seed>-<config_hash>/`。
`manifest.json` 保留原题完整字段、原顺序 ID、配置、探针和导入源文件哈希与环境。
`snapshots/<key>.json/.npz` 保存原 token 前缀、pending 和选定层特征；
`arms/<key>.NOW.json` / `.DELAY_2.json` 保存独立路径、真实检查、反馈、预算与时间。
不保存 GPU KV，不改动 stage1/stage2。

每 10 个完整题目进行普通提交和推送，只纳入 stage3 与必要 LFS 元数据；
推送失败会停止后续批次，恢复先重试备份。
第一批有配对结果后，从独立临时 checkout 取回快照、两臂和 NPZ 并验证 SHA256。
损坏 JSON 保留原字节，并优先从已成功提交备份恢复；无法恢复则明确停止。

dev 的 B/C 三分位边界只使用合格快照的风险分数，test 开始前冻结并成功备份。
恢复时会核对成功备份所包含的冻结文件、分组和 source 哈希；
只有本地冻结文件或旧的成功备份不能放行 test，必须先保存待备份状态并补推。
主时间指标 `T_postfork_wall` 是“从公共快照到结束的后续处理时间”；
公共生成、离线特征/探针和实验分叉恢复单独记录，FAIL 后真实恢复不扣除。
臂内解析、控制和检查时的原子状态/事件持久化开销计入后续时间，
模型下载/加载、外部 Git 备份和事后 Math-Verify 评分不计入。
基础设施中断后，原配对臂文件移入 `interrupted_pairs/` 留存，
使用保存的同一原始 token 快照重新完成两个臂，并以新的完整同卡配对作为正式计时。
长期快照只保存 token 与 CPU 特征，不保存 GPU KV。
`analyze` 仅 CPU 读取已保存结果，pilot/dev/test 分开报告；未真实执行的阶段标为未执行 / NA。
