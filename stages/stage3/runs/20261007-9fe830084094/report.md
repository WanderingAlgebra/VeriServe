# Stage 3：同题 HIGH/LOW 离线检查位置诊断

运行 `20261007-9fe830084094`；experiment_type=`within_question_high_low`；配置哈希 `9fe8300840942d483d8064ec35b201362a661ca92c6d400a5ce24374711ad748`。

目的：同一道题，在未经干预完整参考轨迹的 B 最高分与最低分普通步骤检查，比较最终准确率与总请求处理成本。每条路径只有一次主动中途检查，之后仅终点检查与返工；终点 PASS 才允许提交。旧 fixed_step_now_vs_delay 的 UNCERTAIN 提交默认规则保留，本轮不沿用。

ΔY = Y_HIGH − Y_LOW；ΔT = T_LOW − T_HIGH（秒），正值分别表示 HIGH 更准、更快。Y=终点允许提交且 Math-Verify 判对；格式、预算或未通过检查的算法失败为 0。gold 数据错误、评分错误单列，质量只用双方可评估配对。

T_request 从各自 prompt prefill 至提交/终止实测，含生成、PRM、真实回滚、反馈增量 prefill、控制及相同的臂内原子持久化；CUDA 边界同步。加载/warmup、离线参考与特征、Git、gold 评分排除。该指标不是旧 snapshot 后 T_postfork_wall 的改名。时间仅比较同 GPU/实现有效配对，多后端不混合。按题配对 bootstrap 1000 次，固定种子 20261007，95% percentile CI。

来源 manifest：`stages/stage2/runs/20261004-36fe9009f1f5/manifest.json`。冻结位置与 token 前缀见 selections/；两路径原始结果见 arms/；保留源哈希。

准备时实际 HEAD：`ca7200ecbd0e0edd54e05539bc43fba790a5a784`；用户核对过的 HEAD：`eeb47f1f0b28ee70992d2201ad1db8f5481baaaa`。执行源码逐文件 SHA256 见 manifest.json；完整模型/tokenizer/dataset revision 见配置。

| 环境字段 | 记录值 |
| --- | --- |
| Python | 3.12.3 |
| numpy | 2.3.2 |
| torch | 2.8.0+cu128 |
| transformers | 4.57.3 |
| bitsandbytes | 0.50.2 |
| math-verify | 0.9.0 |

| 执行 GPU | UUID | 实现 / attention / dtype | torch / transformers / bitsandbytes |
| --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 | dae856e4-2c2a-b830-134f-97927b826cec | dynamic_cache_crop / sdpa / bfloat16 | 2.8.0+cu128 / 4.57.3 / 0.50.2 |

## pilot

执行状态：**COMPLETE**。pilot 仅自检；正式 test 为原留出 200 题顺序上的探索性诊断。

| 计划 | 已保存位置 | 合格 | 完整配对 | 质量配对 | 时间配对 | 待完成配对 | 基础设施中断 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 4 | 4 | 4 | 4 | 0 | 0 |

| 路径 | 答对 / 共同可评估 | 正确率 | 平均总请求秒 | PRM / 回滚均值 | 生成 / 撤销 token 均值 | 生成 / 撤销 token 总数 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| HIGH | 1 / 4 | 0.2500 | 49.0573 | 2.0000 / 1.0000 | 2666.2500 / 425.0000 | 10665.0000 / 1700.0000 |
| LOW | 1 / 4 | 0.2500 | 26.0404 | 3.2500 / 1.5000 | 1417.7500 / 763.7500 | 5671.0000 / 3055.0000 |

| ΔY [95% CI] | ΔT 秒 [95% CI] | HIGH 对 / LOW 错 | HIGH 错 / LOW 对 |
| --- | --- | ---: | ---: |
| 0.0000 [0.0000, 0.0000] | -23.0169 [-47.9205, 2.8590] | 0 | 0 |

| 路径 | 检查步号均值 | 前缀 token 均值 | near_end 数 / 比例 | 首检判决 | 算法失败终止次数 | 错答提交 | checkpoint_conflict |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| HIGH | 4.5000 | 336.5000 | 0 / 0.0000 | `{"FAIL": 2, "PASS": 2}` | `{"BUDGET_ATTEMPT": 2, "BUDGET_ROLLBACK": 1}` | 0 | 0 |
| LOW | 2.0000 | 177.5000 | 0 / 0.0000 | `{"FAIL": 1, "PASS": 3}` | `{"BUDGET_ROLLBACK": 3}` | 0 | 0 |

位置关系：`{"HIGH_later": 4}`；全同分题 0（保留并各自实跑）。

| 记录类别 | 原因与计数 |
| --- | --- |
| selection_exclusions | `{}` |
| pending_selections | `{}` |
| quality_pair_exclusions | `{}` |
| HIGH grading exclusions | `{"NONE": 4}` |
| LOW grading exclusions | `{"NONE": 4}` |

后端/请求时间无效完整配对：0。ΔY/ΔT 有效 bootstrap 次数 1000/1000（各应为 1000，NA 为 0）。

## test

执行状态：**INCOMPLETE**。pilot 仅自检；正式 test 为原留出 200 题顺序上的探索性诊断。

| 计划 | 已保存位置 | 合格 | 完整配对 | 质量配对 | 时间配对 | 待完成配对 | 基础设施中断 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 200 | 191 | 60 | 60 | 60 | 131 | 0 |

| 路径 | 答对 / 共同可评估 | 正确率 | 平均总请求秒 | PRM / 回滚均值 | 生成 / 撤销 token 均值 | 生成 / 撤销 token 总数 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| HIGH | 45 / 60 | 0.7500 | 13.4148 | 2.2833 / 0.5000 | 729.5333 / 252.2167 | 43772.0000 / 15133.0000 |
| LOW | 47 / 60 | 0.7833 | 14.0663 | 2.3333 / 0.4500 | 765.1500 / 243.8667 | 45909.0000 / 14632.0000 |

| ΔY [95% CI] | ΔT 秒 [95% CI] | HIGH 对 / LOW 错 | HIGH 错 / LOW 对 |
| --- | --- | ---: | ---: |
| -0.0333 [-0.1167, 0.0333] | 0.6515 [-1.3341, 3.1624] | 2 | 4 |

| 路径 | 检查步号均值 | 前缀 token 均值 | near_end 数 / 比例 | 首检判决 | 算法失败终止次数 | 错答提交 | checkpoint_conflict |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| HIGH | 3.8377 | 250.4607 | 35 / 0.1832 | `{"FAIL": 13, "PASS": 47}` | `{"BUDGET_ROLLBACK": 12}` | 3 | 0 |
| LOW | 3.4241 | 197.1937 | 28 / 0.1466 | `{"FAIL": 6, "PASS": 54}` | `{"BUDGET_ATTEMPT": 1, "BUDGET_ROLLBACK": 8}` | 4 | 0 |

位置关系：`{"HIGH_earlier": 101, "HIGH_later": 90}`；全同分题 0（保留并各自实跑）。

| 记录类别 | 原因与计数 |
| --- | --- |
| selection_exclusions | `{"FORMAT_ERROR: missing, repeated or non-sequential Step N markers": 5, "REFERENCE_FINAL_FORMAT_ERROR": 2, "REFERENCE_LENGTH_TRUNCATED": 2}` |
| pending_selections | `{}` |
| quality_pair_exclusions | `{}` |
| HIGH grading exclusions | `{"NONE": 60}` |
| LOW grading exclusions | `{"NONE": 60}` |

后端/请求时间无效完整配对：0。ΔY/ΔT 有效 bootstrap 次数 1000/1000（各应为 1000，NA 为 0）。

## 离线开销

只汇总已保存的真实值；特征复用不补造耗时。

| 阶段 | 离线计时字段 | 数量 | 均值秒 | 合计秒 |
| --- | --- | ---: | ---: | ---: |
| pilot | offline_feature_seconds | 4 | 0.0133 | 0.0533 |
| pilot | selection_wall_seconds | 4 | 0.0182 | 0.0727 |
| test | offline_feature_seconds | 200 | 0.0108 | 2.1514 |
| test | selection_wall_seconds | 200 | 0.0153 | 3.0544 |

## 解释边界

无真实 GPU pilot / 正式配对时，实验未执行，所有效果为 NA。无正确性差异导致 [0,0] 不能证明总体等效；失败早停较快不能单独视为收益。完整配对与未完成记录均保留，不能按正负结果扩缩样本。

最高/最低位置选择需看完整参考轨迹，属于离线诊断。B 预测未经干预轨迹最终答错风险，不是局部错误或检查收益。分数和位置可能同时影响结果；本轮不能独立证明 B 超越位置、定位局部错误，或中途检查优于不检查/随机位置。pilot 与正式结果分开，强制 FAIL 自检不进入科学结果。
