# 第三阶段：探针风险与即时检查收益

运行：`20261004-109173095d08`。配置哈希：`109173095d08eef1316a08995ffdfe9825c0b69c52d929bc4c7b34e851363971`。

主时间指标为“从公共快照到结束的后续处理时间” T_postfork_wall，不是完整请求端到端时间。ΔY = Y_NOW − Y_DELAY_2；ΔT = T_DELAY_2 − T_NOW（秒），正值分别表示立即检查更准确、更快。
合格锚点上的算法失败计 Y=0；gold 数据错误、评分超时/错误及基础设施中断分别计数。质量采用两臂均可评估的配对分母；时间按相同 GPU 与运行实现分开。bootstrap 按题配对重抽 1000 次，95% percentile CI。

风险分组状态：FROZEN。分界仅来自 dev 合格公共快照的 q，不用动作结果，也不在 test 重划。

## pilot

计划 20 题；已保存锚点状态 20 题；合格 20 题（覆盖率 1.0000）；完整配对 20，有效质量配对 20，有效时间配对 20。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 20 | 20 | 0.7500 | 0.8000 | -0.0500 [-0.1500, 0.0000] | 0.0766 [-1.4928, 1.5668] | 0 | 1 | 105.4500 |
| B low | 6 | 6 | 0.8333 | 0.8333 | 0.0000 [0.0000, 0.0000] | 0.1235 [-0.0120, 0.3576] | 0 | 0 | 77.1667 |
| B mid | 5 | 5 | 0.8000 | 0.8000 | 0.0000 [0.0000, 0.0000] | 2.8324 [0.0297, 6.4846] | 0 | 0 | 164.4000 |
| B high | 9 | 9 | 0.6667 | 0.7778 | -0.1111 [-0.3333, 0.0000] | -1.4857 [-3.6699, -0.1386] | 0 | 1 | 91.5556 |
| C low | 9 | 9 | 0.7778 | 0.8889 | -0.1111 [-0.3333, 0.0000] | -1.0123 [-3.3355, 0.2526] | 0 | 1 | 61.3333 |
| C mid | 7 | 7 | 0.7143 | 0.7143 | 0.0000 [0.0000, 0.0000] | -0.4779 [-1.1374, 0.0177] | 0 | 0 | 106.8571 |
| C high | 4 | 4 | 0.7500 | 0.7500 | 0.0000 [0.0000, 0.0000] | 3.4971 [-0.0173, 8.0624] | 0 | 0 | 202.2500 |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 7.5167/5.2796 | 7.5933/5.3099 | NOW:2.3500/0.4500; DELAY_2:2.2500/0.4500 | NOW:518.9000/169.4500; DELAY_2:519.0500/153.3000 | NOW:0.0500/0.0000; DELAY_2:0.0500/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 15, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 16, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B low | 3.6023/3.8132 | 3.7258/3.8276 | NOW:2.3333/0.3333; DELAY_2:2.1667/0.3333 | NOW:276.0000/38.3333; DELAY_2:282.3333/37.3333 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 5}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 5}` | 1000/1000;1000/1000 |
| B mid | 9.3431/6.5579 | 12.1755/9.8352 | NOW:2.2000/0.4000; DELAY_2:2.2000/0.6000 | NOW:677.8000/172.8000; DELAY_2:828.4000/272.6000 | NOW:0.2000/0.0000; DELAY_2:0.2000/0.0000 | NOW:`{"CORRECT": 4, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"CORRECT": 4, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B high | 9.1118/4.2053 | 7.6261/4.2101 | NOW:2.4444/0.5556; DELAY_2:2.3333/0.4444 | NOW:592.5556/255.0000; DELAY_2:505.0000/164.3333 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 6, "FORMAT_OR_PREDICTION_PARSE": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 7, "FORMAT_OR_PREDICTION_PARSE": 1}` | 1000/1000;1000/1000 |
| C low | 6.2510/4.9369 | 5.2386/4.9584 | NOW:2.4444/0.4444; DELAY_2:2.3333/0.3333 | NOW:406.3333/105.3333; DELAY_2:344.2222/37.6667 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 7}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 8}` | 1000/1000;1000/1000 |
| C mid | 9.1714/6.3783 | 8.6934/6.0527 | NOW:2.2857/0.4286; DELAY_2:2.1429/0.4286 | NOW:611.1429/225.2857; DELAY_2:583.8571/194.8571 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 5, "FORMAT_OR_PREDICTION_PARSE": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 5, "FORMAT_OR_PREDICTION_PARSE": 1}` | 1000/1000;1000/1000 |
| C high | 7.4692/3.9407 | 10.9663/5.5859 | NOW:2.2500/0.5000; DELAY_2:2.2500/0.7500 | NOW:610.7500/216.0000; DELAY_2:799.0000/340.7500 | NOW:0.2500/0.0000; DELAY_2:0.2500/0.0000 | NOW:`{"CORRECT": 3, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"CORRECT": 3, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY -0.1111 [-0.3862, 0.0000]；ΔT 秒 -1.6092 [-4.1656, -0.1291]。
高低差 ΔY/ΔT 有效 bootstrap 次数：997/1000/997/1000。
质量异质性：小样本区间可能不稳定；95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：小样本区间可能不稳定；95% CI 完全低于零，为推迟检查在该指标上更有收益提供初步证据。

C 高风险减低风险的收益差：ΔY 0.1111 [0.0000, 0.3750]；ΔT 秒 4.5094 [-0.0633, 10.7158]。
高低差 ΔY/ΔT 有效 bootstrap 次数：992/1000/992/1000。
质量异质性：小样本区间可能不稳定；95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：小样本区间可能不稳定；95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

NOW 后续时间均值/中位数/p95（描述性）：7.5167/5.2796/22.4919 秒；PRM 调用/回滚均值 2.3500/0.4500；生成/撤销 token 合计 10378.0000/3389.0000。
首次检查判决次数：`{"FAIL": 1, "PASS": 19}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 20 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1075.6000/27.3000。
终止原因：`{"BUDGET_ROLLBACKS": 3, "FINAL_PASS": 16, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 3, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 16}`；保存状态：`{"COMPLETE": 20}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：7.5933/5.3099/32.6290 秒；PRM 调用/回滚均值 2.2500/0.4500；生成/撤销 token 合计 10381.0000/3066.0000。
首次检查判决次数：`{"FAIL": 1, "PASS": 19}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 20 臂）；提前终点 2 臂；累计非 decode 前向/反馈 token 均值 1220.5500/27.3000。
终止原因：`{"BUDGET_ROLLBACKS": 2, "FINAL_PASS": 17, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 2, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 17}`；保存状态：`{"COMPLETE": 20}`。

公共快照 near_end 2 题；DELAY_2 delay_collapsed 2 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## dev

计划 40 题；已保存锚点状态 40 题；合格 40 题（覆盖率 1.0000）；完整配对 40，有效质量配对 40，有效时间配对 40。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 40 | 40 | 0.7000 | 0.7000 | 0.0000 [-0.1000, 0.1000] | -0.6435 [-2.1821, 0.8828] | 2 | 2 | 121.8250 |
| B low | 14 | 14 | 0.7857 | 0.9286 | -0.1429 [-0.3571, 0.0000] | -1.0579 [-2.7036, 0.1480] | 0 | 2 | 81.3571 |
| B mid | 13 | 13 | 0.6923 | 0.6923 | 0.0000 [0.0000, 0.0000] | -1.5540 [-4.9155, 0.7710] | 0 | 0 | 181.5385 |
| B high | 13 | 13 | 0.6154 | 0.4615 | 0.1538 [0.0000, 0.3846] | 0.7132 [-1.9043, 3.8041] | 2 | 0 | 105.6923 |
| C low | 14 | 14 | 0.7143 | 0.7857 | -0.0714 [-0.2857, 0.1429] | -1.3184 [-3.2583, 0.1642] | 1 | 2 | 62.0714 |
| C mid | 13 | 13 | 0.9231 | 0.8462 | 0.0769 [0.0000, 0.2308] | 1.3592 [0.0215, 3.6485] | 1 | 0 | 94.9231 |
| C high | 13 | 13 | 0.4615 | 0.4615 | 0.0000 [0.0000, 0.0000] | -1.9195 [-5.7471, 1.0536] | 0 | 0 | 213.0769 |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 11.0651/5.6010 | 10.4216/6.0969 | NOW:2.4000/0.5000; DELAY_2:2.2000/0.4750 | NOW:729.4750/217.4000; DELAY_2:692.8000/174.6250 | NOW:0.0500/0.0000; DELAY_2:0.1000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 8, "CORRECT": 28, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 3}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 7, "CORRECT": 28, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 4}` | 1000/1000;1000/1000 |
| B low | 4.7073/2.5554 | 3.6494/2.5806 | NOW:2.3571/0.4286; DELAY_2:2.0000/0.2857 | NOW:341.1429/108.2143; DELAY_2:282.5000/57.8571 | NOW:0.0714/0.0000; DELAY_2:0.0714/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 11, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"CORRECT": 13, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B mid | 12.6269/7.7048 | 11.0728/7.7913 | NOW:2.3077/0.3846; DELAY_2:2.1538/0.3846 | NOW:874.7692/265.0769; DELAY_2:786.6923/159.0769 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 9, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 9, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 2}` | 1000/1000;1000/1000 |
| B high | 16.3504/6.1987 | 17.0635/7.6014 | NOW:2.5385/0.6923; DELAY_2:2.4615/0.7692 | NOW:1002.3846/287.3077; DELAY_2:1040.7692/315.9231 | NOW:0.0769/0.0000; DELAY_2:0.2308/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 4, "CORRECT": 8, "WRONG_SUBMITTED": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 6, "CORRECT": 6, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| C low | 6.6203/4.6028 | 5.3020/4.3954 | NOW:2.5714/0.6429; DELAY_2:2.2143/0.5000 | NOW:427.0714/178.6429; DELAY_2:353.8571/101.2857 | NOW:0.0000/0.0000; DELAY_2:0.0000/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 10, "FORMAT_OR_PREDICTION_PARSE": 1}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 11, "FORMAT_OR_PREDICTION_PARSE": 1}` | 1000/1000;1000/1000 |
| C mid | 10.3671/4.5871 | 11.7262/4.6522 | NOW:2.0769/0.1538; DELAY_2:2.0769/0.3077 | NOW:665.0000/57.8462; DELAY_2:737.6923/100.0769 | NOW:0.0000/0.0000; DELAY_2:0.0769/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 1, "CORRECT": 12}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 11}` | 1000/1000;1000/1000 |
| C high | 16.5500/7.9574 | 14.6305/8.0201 | NOW:2.5385/0.6923; DELAY_2:2.3077/0.6154 | NOW:1119.6154/418.6923; DELAY_2:1012.9231/328.1538 | NOW:0.1538/0.0000; DELAY_2:0.2308/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 4, "CORRECT": 6, "WRONG_SUBMITTED": 3}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 6, "WRONG_SUBMITTED": 4}` | 1000/1000;1000/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY 0.2967 [0.0667, 0.6001]；ΔT 秒 1.7710 [-1.4537, 5.3896]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 完全高于零，为立即检查在该指标上更有收益提供初步证据。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

C 高风险减低风险的收益差：ΔY 0.0714 [-0.1818, 0.3339]；ΔT 秒 -0.6011 [-4.6807, 3.1259]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

NOW 后续时间均值/中位数/p95（描述性）：11.0651/5.6010/44.4668 秒；PRM 调用/回滚均值 2.4000/0.5000；生成/撤销 token 合计 29179.0000/8696.0000。
首次检查判决次数：`{"FAIL": 2, "PASS": 38}`；FAIL/UNCERTAIN 比例：0.0500/0.0000（有首次检查记录 40 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1180.7750/30.2750。
终止原因：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 7, "FINAL_PASS": 31, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 7, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 31}`；保存状态：`{"COMPLETE": 40}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：10.4216/6.0969/33.5867 秒；PRM 调用/回滚均值 2.2000/0.4750；生成/撤销 token 合计 27712.0000/6985.0000。
首次检查判决次数：`{"FAIL": 4, "PASS": 36}`；FAIL/UNCERTAIN 比例：0.1000/0.0000（有首次检查记录 40 臂）；提前终点 5 臂；累计非 decode 前向/反馈 token 均值 1223.0250/28.4500。
终止原因：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 6, "FINAL_PASS": 32, "MISSING_OR_INCOMPLETE_BOXED": 1}`；评分排除：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 6, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 32}`；保存状态：`{"COMPLETE": 40}`。

公共快照 near_end 2 题；DELAY_2 delay_collapsed 5 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## test

计划 130 题；已保存锚点状态 80 题；合格 80 题（覆盖率 0.6154）；完整配对 80，有效质量配对 80，有效时间配对 80。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{"MISSING_SNAPSHOT": 50}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 80 | 80 | 0.6875 | 0.6750 | 0.0125 [-0.0250, 0.0625] | 0.1147 [-1.3324, 1.4873] | 2 | 1 | 107.3125 |
| B low | 27 | 27 | 0.9259 | 0.8889 | 0.0370 [0.0000, 0.1111] | 0.5703 [-0.1451, 1.8996] | 1 | 0 | 85.1481 |
| B mid | 18 | 18 | 0.6111 | 0.6111 | 0.0000 [0.0000, 0.0000] | 0.1708 [-0.5914, 0.8581] | 0 | 0 | 132.8889 |
| B high | 35 | 35 | 0.5429 | 0.5429 | 0.0000 [-0.0857, 0.0857] | -0.2656 [-3.1146, 2.4397] | 1 | 1 | 111.2571 |
| C low | 31 | 31 | 0.8065 | 0.8065 | 0.0000 [0.0000, 0.0000] | -0.0293 [-0.4275, 0.2657] | 0 | 0 | 59.1935 |
| C mid | 21 | 21 | 0.6667 | 0.7143 | -0.0476 [-0.1429, 0.0000] | 1.9033 [-0.3187, 4.8203] | 0 | 1 | 99.0952 |
| C high | 28 | 28 | 0.5714 | 0.5000 | 0.0714 [0.0000, 0.1786] | -1.0674 [-4.1767, 1.6085] | 2 | 0 | 166.7500 |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 10.4495/6.1976 | 10.5642/6.2692 | NOW:2.3875/0.4625; DELAY_2:2.2375/0.4500 | NOW:681.6500/244.1625; DELAY_2:686.0500/222.2625 | NOW:0.0375/0.0000; DELAY_2:0.1266/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 13, "CORRECT": 55, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 2, "WRONG_SUBMITTED": 9}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 16, "CORRECT": 54, "FORMAT_OR_PREDICTION_PARSE": 2, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 7}` | 1000/1000;1000/1000 |
| B low | 5.2718/2.7994 | 5.8420/2.8550 | NOW:2.1111/0.1111; DELAY_2:2.0370/0.1481 | NOW:376.8889/61.2593; DELAY_2:406.8148/80.9630 | NOW:0.0000/0.0000; DELAY_2:0.0370/0.0000 | NOW:`{"CORRECT": 25, "WRONG_SUBMITTED": 2}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 24, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B mid | 8.2840/8.7602 | 8.4547/8.7277 | NOW:2.1111/0.3889; DELAY_2:2.0000/0.3889 | NOW:588.3333/102.1111; DELAY_2:596.7222/113.2778 | NOW:0.1667/0.0000; DELAY_2:0.2353/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 11, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 2}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 11, "FORMAT_OR_PREDICTION_PARSE": 2, "WRONG_SUBMITTED": 2}` | 1000/1000;1000/1000 |
| B high | 15.5575/8.8172 | 15.2919/8.8277 | NOW:2.7429/0.7714; DELAY_2:2.5143/0.7143 | NOW:964.7429/458.3143; DELAY_2:947.4000/387.3143 | NOW:0.0000/0.0000; DELAY_2:0.1429/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 10, "CORRECT": 19, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 5}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 11, "CORRECT": 19, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 4}` | 1000/1000;1000/1000 |
| C low | 6.0321/4.3874 | 6.0028/4.2967 | NOW:2.1290/0.1935; DELAY_2:2.0000/0.1935 | NOW:393.0000/76.5161; DELAY_2:389.9677/68.2258 | NOW:0.0000/0.0000; DELAY_2:0.0333/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 25, "OTHER_ALGORITHM_FAILURE": 2, "WRONG_SUBMITTED": 2}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 2, "CORRECT": 25, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 2}` | 1000/1000;1000/1000 |
| C mid | 9.2631/5.6563 | 11.1664/5.6789 | NOW:2.4286/0.4286; DELAY_2:2.2381/0.4286 | NOW:608.7619/197.1905; DELAY_2:711.5238/227.6190 | NOW:0.0000/0.0000; DELAY_2:0.1429/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 14, "WRONG_SUBMITTED": 4}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 4, "CORRECT": 15, "WRONG_SUBMITTED": 2}` | 1000/1000;1000/1000 |
| C high | 16.2300/9.6888 | 15.1626/10.0799 | NOW:2.6429/0.7857; DELAY_2:2.5000/0.7500 | NOW:1055.8929/465.0000; DELAY_2:994.7500/388.7857 | NOW:0.1071/0.0000; DELAY_2:0.2143/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 8, "CORRECT": 16, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 3}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 10, "CORRECT": 14, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 3}` | 1000/1000;1000/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY -0.0370 [-0.1572, 0.0625]；ΔT 秒 -0.8359 [-4.3005, 2.4453]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

C 高风险减低风险的收益差：ΔY 0.0714 [0.0000, 0.1739]；ΔT 秒 -1.0381 [-4.4635, 1.9824]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

NOW 后续时间均值/中位数/p95（描述性）：10.4495/6.1976/33.5622 秒；PRM 调用/回滚均值 2.3875/0.4625；生成/撤销 token 合计 54532.0000/19533.0000。
首次检查判决次数：`{"FAIL": 3, "PASS": 77}`；FAIL/UNCERTAIN 比例：0.0375/0.0000（有首次检查记录 80 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1236.1750/28.0375。
终止原因：`{"BUDGET_ROLLBACKS": 13, "FINAL_PASS": 64, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONSEQUENTIAL_STEPS": 2}`；评分排除：`{"BUDGET_ROLLBACKS": 13, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 64, "NONSEQUENTIAL_STEPS": 2}`；保存状态：`{"COMPLETE": 80}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：10.5642/6.2692/38.3761 秒；PRM 调用/回滚均值 2.2375/0.4500；生成/撤销 token 合计 54884.0000/17781.0000。
首次检查判决次数：`{"FAIL": 10, "PASS": 69}`；FAIL/UNCERTAIN 比例：0.1266/0.0000（有首次检查记录 79 臂）；提前终点 3 臂；累计非 decode 前向/反馈 token 均值 1330.4375/26.9000。
终止原因：`{"BUDGET_ROLLBACKS": 16, "FINAL_PASS": 60, "FINAL_UNCERTAIN": 1, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONSEQUENTIAL_STEPS": 1, "STEP_FORMAT_ERROR: FORMAT_ERROR: missing, repeated or non-sequential Step N markers": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 16, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 61, "NONSEQUENTIAL_STEPS": 1, "STEP_FORMAT_ERROR: FORMAT_ERROR: missing, repeated or non-sequential Step N markers": 1}`；保存状态：`{"COMPLETE": 80}`。

公共快照 near_end 0 题；DELAY_2 delay_collapsed 3 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## 解释与后续

B 预测未经干预的整条轨迹最终答错风险，不是当前步错误概率或检查收益概率。B 高低组配对收益差才检验“高风险是否更值得立即检查”；首次检查失败率本身不能回答该问题。C 是仅有步骤/长度的对照，固定第二步仍留下前缀长度差异；若 B 与 C 的收益模式相似，长度可能解释部分效果。B 在 stage2 的 AUROC 较高不构成本轮检查时机收益证据。
正式 B 异质性质量/时间：-0.0370 [-0.1572, 0.0625] / -0.8359 [-4.3005, 2.4453]；C：0.0714 [0.0000, 0.1739] / -1.0381 [-4.4635, 1.9824]。
实际长度与对照：B 低/高组前缀均值 85.1481/111.2571 token; C 低/高组前缀均值 59.1935/166.7500 token；质量高低收益差点估计 不同向；时间高低收益差点估计 同向。同向时长度信号与 B 收益模式相容，但这不证明长度完全解释 B；不同向也不能证明 B 有独立作用，本轮没有做长度调整。
正式配对中立即检查修复 2 题、破坏 1 题；NOW 检查/回滚合计 191.0000/37.0000、撤销 token 19533.0000; DELAY_2 检查/回滚合计 179.0000/36.0000、撤销 token 17781.0000。
修复/破坏次数、首次检查 PASS/FAIL/UNCERTAIN、返工次数和撤销 token 可描述收益与成本的机制，但没有额外随机化，不能把它们直接解释为因果中介。格式失败和预算耗尽保持在算法失败分母，所有排除均保留记录。
反馈是 PRM 分数驱动的确定性通用 diagnosis/hint 模板；本轮只覆盖无历史 PASS 的普通 Step 2，不能直接推广到多位置或已有 PASS 状态。q 的独立前缀特征提取与探针时间属于离线诊断开销。小组 p95 仅为描述，分组样本过少、CI 跨零和 BF16 数值/运行后端差异都限制结论。
当前没有足够正式证据支持推进风险阈值策略；结果为 NA 或 CI 跨零时如实保留，不调整 test 寻找正结果。

pilot/dev/test 分开报告；未执行指标均为 NA。离线诊断和公共生成开销见下表（只对实际保存值汇总）：

| 阶段 | 公共生成秒均值 | 离线特征秒均值 | 探针秒均值 | 实验快照恢复秒均值 |
| --- | ---: | ---: | ---: | ---: |
| pilot | 1.9988 | 0.0374 | 0.0001 | 0.0387 |
| dev | 2.2924 | 0.0362 | 0.0001 | 0.0380 |
| test | 2.0329 | 0.0362 | 0.0001 | 0.0377 |
