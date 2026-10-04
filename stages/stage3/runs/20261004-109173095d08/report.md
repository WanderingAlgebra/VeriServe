# 第三阶段：探针风险与即时检查收益

运行：`20261004-109173095d08`。配置哈希：`109173095d08eef1316a08995ffdfe9825c0b69c52d929bc4c7b34e851363971`。

主时间指标为“从公共快照到结束的后续处理时间” T_postfork_wall，不是完整请求端到端时间。ΔY = Y_NOW − Y_DELAY_2；ΔT = T_DELAY_2 − T_NOW（秒），正值分别表示立即检查更准确、更快。
合格锚点上的算法失败计 Y=0；gold 数据错误、评分超时/错误及基础设施中断分别计数。质量采用两臂均可评估的配对分母；时间按相同 GPU 与运行实现分开。bootstrap 按题配对重抽 1000 次，95% percentile CI。

风险分组状态：NOT_FROZEN。分界仅来自 dev 合格公共快照的 q，不用动作结果，也不在 test 重划。

## pilot

计划 20 题；已保存锚点状态 20 题；合格 20 题（覆盖率 1.0000）；完整配对 20，有效质量配对 20，有效时间配对 20。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 20 | 20 | 0.7500 | 0.8000 | -0.0500 [-0.1500, 0.0000] | 0.0766 [-1.4928, 1.5668] | 0 | 1 | 105.4500 |
| B low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 7.5167/5.2796 | 7.5933/5.3099 | NOW:2.3500/0.4500; DELAY_2:2.2500/0.4500 | NOW:518.9000/169.4500; DELAY_2:519.0500/153.3000 | NOW:0.0500/0.0000; DELAY_2:0.0500/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 15, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 16, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

C 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

NOW 后续时间均值/中位数/p95（描述性）：7.5167/5.2796/22.4919 秒；PRM 调用/回滚均值 2.3500/0.4500；生成/撤销 token 合计 10378.0000/3389.0000。
首次检查判决次数：`{"FAIL": 1, "PASS": 19}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 20 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1075.6000/27.3000。
终止原因：`{"BUDGET_ROLLBACKS": 3, "FINAL_PASS": 16, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 3, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 16}`；保存状态：`{"COMPLETE": 20}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：7.5933/5.3099/32.6290 秒；PRM 调用/回滚均值 2.2500/0.4500；生成/撤销 token 合计 10381.0000/3066.0000。
首次检查判决次数：`{"FAIL": 1, "PASS": 19}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 20 臂）；提前终点 2 臂；累计非 decode 前向/反馈 token 均值 1220.5500/27.3000。
终止原因：`{"BUDGET_ROLLBACKS": 2, "FINAL_PASS": 17, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 2, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 17}`；保存状态：`{"COMPLETE": 20}`。

公共快照 near_end 2 题；DELAY_2 delay_collapsed 2 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## dev

计划 40 题；已保存锚点状态 20 题；合格 20 题（覆盖率 0.5000）；完整配对 20，有效质量配对 20，有效时间配对 20。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{"MISSING_SNAPSHOT": 20}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 20 | 20 | 0.6500 | 0.7000 | -0.0500 [-0.1500, 0.0000] | -0.2894 [-2.0549, 1.2552] | 0 | 1 | 131.4500 |
| B low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 14.0630/6.3958 | 13.7736/6.0969 | NOW:2.4500/0.6000; DELAY_2:2.3500/0.6000 | NOW:902.8000/258.5000; DELAY_2:885.2500/236.9000 | NOW:0.0500/0.0000; DELAY_2:0.1000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 6, "CORRECT": 13, "FORMAT_OR_PREDICTION_PARSE": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 5, "CORRECT": 14, "FORMAT_OR_PREDICTION_PARSE": 1}` | 1000/1000;1000/1000 |
| B low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

C 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

NOW 后续时间均值/中位数/p95（描述性）：14.0630/6.3958/50.7938 秒；PRM 调用/回滚均值 2.4500/0.6000；生成/撤销 token 合计 18056.0000/5170.0000。
首次检查判决次数：`{"FAIL": 1, "PASS": 19}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 20 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1249.1500/36.3000。
终止原因：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 5, "FINAL_PASS": 13, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 5, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 13}`；保存状态：`{"COMPLETE": 20}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：13.7736/6.0969/41.6610 秒；PRM 调用/回滚均值 2.3500/0.6000；生成/撤销 token 合计 17705.0000/4738.0000。
首次检查判决次数：`{"FAIL": 2, "PASS": 18}`；FAIL/UNCERTAIN 比例：0.1000/0.0000（有首次检查记录 20 臂）；提前终点 1 臂；累计非 decode 前向/反馈 token 均值 1334.8500/36.0000。
终止原因：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 4, "FINAL_PASS": 14, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 4, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 14}`；保存状态：`{"COMPLETE": 20}`。

公共快照 near_end 0 题；DELAY_2 delay_collapsed 1 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## test

计划 130 题；已保存锚点状态 0 题；合格 0 题（覆盖率 0.0000）；完整配对 0，有效质量配对 0，有效时间配对 0。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{"MISSING_SNAPSHOT": 130}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| B high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C low | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C mid | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |
| C high | 0 | 0 | NA | NA | NA（NO_VALID_PAIRED_QUALITY） | NA（NO_VALID_PAIRED_TIME） | 0 | 0 | NA |

总体质量：NA：尚无可评估证据。 总体时间：NA：尚无可评估证据。
总体 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| B high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C low | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C mid | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |
| C high | NA/NA | NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:NA/NA; DELAY_2:NA/NA | NOW:`{}`; DELAY_2:`{}` | 0/1000;0/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

C 高风险减低风险的收益差：ΔY NA（EMPTY_LOW_OR_HIGH_GROUP）；ΔT 秒 NA（EMPTY_LOW_OR_HIGH_GROUP）。
高低差 ΔY/ΔT 有效 bootstrap 次数：0/1000/0/1000。
质量异质性：NA：尚无可评估证据。 时间异质性：NA：尚无可评估证据。

NOW 后续时间均值/中位数/p95（描述性）：NA/NA/NA 秒；PRM 调用/回滚均值 NA/NA；生成/撤销 token 合计 NA/NA。
首次检查判决次数：`{}`；FAIL/UNCERTAIN 比例：NA/NA（有首次检查记录 0 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 NA/NA。
终止原因：`{}`；评分排除：`{}`；保存状态：`{}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：NA/NA/NA 秒；PRM 调用/回滚均值 NA/NA；生成/撤销 token 合计 NA/NA。
首次检查判决次数：`{}`；FAIL/UNCERTAIN 比例：NA/NA（有首次检查记录 0 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 NA/NA。
终止原因：`{}`；评分排除：`{}`；保存状态：`{}`。

公共快照 near_end 0 题；DELAY_2 delay_collapsed 0 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## 解释与后续

B 预测未经干预的整条轨迹最终答错风险，不是当前步错误概率或检查收益概率。B 高低组配对收益差才检验“高风险是否更值得立即检查”；首次检查失败率本身不能回答该问题。C 是仅有步骤/长度的对照，固定第二步仍留下前缀长度差异；若 B 与 C 的收益模式相似，长度可能解释部分效果。B 在 stage2 的 AUROC 较高不构成本轮检查时机收益证据。
正式 B 异质性质量/时间：NA（EMPTY_LOW_OR_HIGH_GROUP） / NA（EMPTY_LOW_OR_HIGH_GROUP）；C：NA（EMPTY_LOW_OR_HIGH_GROUP） / NA（EMPTY_LOW_OR_HIGH_GROUP）。
实际长度与对照：B 低/高组前缀均值 NA/NA token; C 低/高组前缀均值 NA/NA token；质量高低收益差点估计 NA；时间高低收益差点估计 NA。同向时长度信号与 B 收益模式相容，但这不证明长度完全解释 B；不同向也不能证明 B 有独立作用，本轮没有做长度调整。
正式配对中立即检查修复 0 题、破坏 0 题；NOW 检查/回滚合计 NA/NA、撤销 token NA; DELAY_2 检查/回滚合计 NA/NA、撤销 token NA。
修复/破坏次数、首次检查 PASS/FAIL/UNCERTAIN、返工次数和撤销 token 可描述收益与成本的机制，但没有额外随机化，不能把它们直接解释为因果中介。格式失败和预算耗尽保持在算法失败分母，所有排除均保留记录。
反馈是 PRM 分数驱动的确定性通用 diagnosis/hint 模板；本轮只覆盖无历史 PASS 的普通 Step 2，不能直接推广到多位置或已有 PASS 状态。q 的独立前缀特征提取与探针时间属于离线诊断开销。小组 p95 仅为描述，分组样本过少、CI 跨零和 BF16 数值/运行后端差异都限制结论。
当前没有足够正式证据支持推进风险阈值策略；结果为 NA 或 CI 跨零时如实保留，不调整 test 寻找正结果。

pilot/dev/test 分开报告；未执行指标均为 NA。离线诊断和公共生成开销见下表（只对实际保存值汇总）：

| 阶段 | 公共生成秒均值 | 离线特征秒均值 | 探针秒均值 | 实验快照恢复秒均值 |
| --- | ---: | ---: | ---: | ---: |
| pilot | 1.9988 | 0.0374 | 0.0001 | 0.0387 |
| dev | 2.4662 | 0.0365 | 0.0001 | 0.0383 |
| test | NA | NA | NA | NA |
