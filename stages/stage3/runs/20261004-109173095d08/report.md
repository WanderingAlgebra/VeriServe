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

计划 130 题；已保存锚点状态 130 题；合格 130 题（覆盖率 1.0000）；完整配对 130，有效质量配对 130，有效时间配对 130。
锚点不合格原因：`{}`；未执行/待恢复锚点：`{}`；待完成配对 0；基础设施中断配对 0；质量评估排除：`{}`。

| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 总体 | 130 | 130 | 0.6769 | 0.6692 | 0.0077 [-0.0308, 0.0462] | 0.4964 [-0.8978, 1.7759] | 4 | 3 | 108.3923 |
| B low | 45 | 45 | 0.9556 | 0.9111 | 0.0444 [0.0000, 0.1111] | 1.0527 [-0.0727, 2.8900] | 2 | 0 | 96.6889 |
| B mid | 29 | 29 | 0.6207 | 0.6207 | 0.0000 [0.0000, 0.0000] | 0.4753 [-0.2403, 1.3097] | 0 | 0 | 125.6207 |
| B high | 56 | 56 | 0.4821 | 0.5000 | -0.0179 [-0.1071, 0.0536] | 0.0602 [-3.2350, 2.7992] | 2 | 3 | 108.8750 |
| C low | 49 | 49 | 0.7959 | 0.7755 | 0.0204 [0.0000, 0.0612] | 0.9223 [0.0487, 2.1668] | 1 | 0 | 59.4490 |
| C mid | 36 | 36 | 0.6389 | 0.6667 | -0.0278 [-0.0833, 0.0000] | 1.1268 [-0.1827, 2.8577] | 0 | 1 | 97.9444 |
| C high | 45 | 45 | 0.5778 | 0.5556 | 0.0222 [-0.0672, 0.1333] | -0.4719 [-4.6246, 2.9869] | 3 | 2 | 170.0444 |

总体质量：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 总体时间：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。
总体 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。

| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总体 | 9.5039/5.0668 | 10.0003/5.1899 | NOW:2.3692/0.4462; DELAY_2:2.2231/0.4769 | NOW:630.5077/198.3692; DELAY_2:656.6615/215.3846 | NOW:0.0462/0.0000; DELAY_2:0.1550/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 21, "CORRECT": 88, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 2, "WRONG_SUBMITTED": 18}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 24, "CORRECT": 87, "FORMAT_OR_PREDICTION_PARSE": 2, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 16}` | 1000/1000;1000/1000 |
| B low | 4.6829/2.8672 | 5.7356/2.9015 | NOW:2.0889/0.0889; DELAY_2:1.9556/0.1333 | NOW:355.7111/51.5556; DELAY_2:412.6000/92.7556 | NOW:0.0000/0.0000; DELAY_2:0.0444/0.0000 | NOW:`{"CORRECT": 43, "WRONG_SUBMITTED": 2}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 3, "CORRECT": 41, "WRONG_SUBMITTED": 1}` | 1000/1000;1000/1000 |
| B mid | 7.7211/6.9529 | 8.1963/8.4207 | NOW:2.2414/0.4138; DELAY_2:2.1034/0.4483 | NOW:550.2414/114.7931; DELAY_2:575.1724/125.9310 | NOW:0.1034/0.0000; DELAY_2:0.2143/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 4, "CORRECT": 18, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 5}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 5, "CORRECT": 18, "FORMAT_OR_PREDICTION_PARSE": 2, "WRONG_SUBMITTED": 4}` | 1000/1000;1000/1000 |
| B high | 14.3012/8.0161 | 14.3615/8.8456 | NOW:2.6607/0.7500; DELAY_2:2.5000/0.7679 | NOW:892.8929/359.6250; DELAY_2:894.9821/360.2500 | NOW:0.0536/0.0000; DELAY_2:0.2143/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 17, "CORRECT": 27, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 11}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 16, "CORRECT": 28, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 11}` | 1000/1000;1000/1000 |
| C low | 5.4986/4.2927 | 6.4209/4.1738 | NOW:2.2041/0.2857; DELAY_2:2.1020/0.3265 | NOW:363.3265/75.8367; DELAY_2:412.7959/104.7755 | NOW:0.0408/0.0000; DELAY_2:0.0833/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 5, "CORRECT": 39, "OTHER_ALGORITHM_FAILURE": 2, "WRONG_SUBMITTED": 3}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 5, "CORRECT": 38, "FORMAT_OR_PREDICTION_PARSE": 1, "OTHER_ALGORITHM_FAILURE": 1, "WRONG_SUBMITTED": 4}` | 1000/1000;1000/1000 |
| C mid | 8.5491/4.3881 | 9.6759/4.4173 | NOW:2.4444/0.4722; DELAY_2:2.2500/0.4722 | NOW:568.3056/190.3333; DELAY_2:628.7222/195.0278 | NOW:0.0278/0.0000; DELAY_2:0.1389/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 6, "CORRECT": 23, "WRONG_SUBMITTED": 7}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 7, "CORRECT": 24, "WRONG_SUBMITTED": 5}` | 1000/1000;1000/1000 |
| C high | 14.6291/8.7969 | 14.1573/9.6913 | NOW:2.4889/0.6000; DELAY_2:2.3333/0.6444 | NOW:971.2000/338.2222; DELAY_2:944.5556/352.1111 | NOW:0.0667/0.0000; DELAY_2:0.2444/0.0000 | NOW:`{"BUDGET_OR_INPUT_LIMIT": 10, "CORRECT": 26, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 8}`; DELAY_2:`{"BUDGET_OR_INPUT_LIMIT": 12, "CORRECT": 25, "FORMAT_OR_PREDICTION_PARSE": 1, "WRONG_SUBMITTED": 7}` | 1000/1000;1000/1000 |

结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。

B 高风险减低风险的收益差：ΔY -0.0623 [-0.1641, 0.0333]；ΔT 秒 -0.9925 [-4.6250, 2.0391]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

C 高风险减低风险的收益差：ΔY 0.0018 [-0.1076, 0.1009]；ΔT 秒 -1.3942 [-5.4789, 1.9612]。
高低差 ΔY/ΔT 有效 bootstrap 次数：1000/1000/1000/1000。
质量异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。 时间异质性：95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。

NOW 后续时间均值/中位数/p95（描述性）：9.5039/5.0668/31.4265 秒；PRM 调用/回滚均值 2.3692/0.4462；生成/撤销 token 合计 81966.0000/25788.0000。
首次检查判决次数：`{"FAIL": 6, "PASS": 124}`；FAIL/UNCERTAIN 比例：0.0462/0.0000（有首次检查记录 130 臂）；提前终点 0 臂；累计非 decode 前向/反馈 token 均值 1165.1077/26.9923。
终止原因：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 20, "FINAL_PASS": 106, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONSEQUENTIAL_STEPS": 2}`；评分排除：`{"BUDGET_GENERATION_ATTEMPT": 1, "BUDGET_ROLLBACKS": 20, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 106, "NONSEQUENTIAL_STEPS": 2}`；保存状态：`{"COMPLETE": 130}`。

DELAY_2 后续时间均值/中位数/p95（描述性）：10.0003/5.1899/37.3908 秒；PRM 调用/回滚均值 2.2231/0.4769；生成/撤销 token 合计 85366.0000/28000.0000。
首次检查判决次数：`{"FAIL": 20, "PASS": 109}`；FAIL/UNCERTAIN 比例：0.1550/0.0000（有首次检查记录 129 臂）；提前终点 9 臂；累计非 decode 前向/反馈 token 均值 1286.2154/28.4462。
终止原因：`{"BUDGET_ROLLBACKS": 24, "FINAL_PASS": 102, "FINAL_UNCERTAIN": 1, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONSEQUENTIAL_STEPS": 1, "STEP_FORMAT_ERROR: FORMAT_ERROR: missing, repeated or non-sequential Step N markers": 1}`；评分排除：`{"BUDGET_ROLLBACKS": 24, "MISSING_OR_INCOMPLETE_BOXED": 1, "NONE": 103, "NONSEQUENTIAL_STEPS": 1, "STEP_FORMAT_ERROR: FORMAT_ERROR: missing, repeated or non-sequential Step N markers": 1}`；保存状态：`{"COMPLETE": 130}`。

公共快照 near_end 1 题；DELAY_2 delay_collapsed 9 题；两臂后端不一致 0 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。

## 解释与后续

B 预测未经干预的整条轨迹最终答错风险，不是当前步错误概率或检查收益概率。B 高低组配对收益差才检验“高风险是否更值得立即检查”；首次检查失败率本身不能回答该问题。C 是仅有步骤/长度的对照，固定第二步仍留下前缀长度差异；若 B 与 C 的收益模式相似，长度可能解释部分效果。B 在 stage2 的 AUROC 较高不构成本轮检查时机收益证据。
正式 B 异质性质量/时间：-0.0623 [-0.1641, 0.0333] / -0.9925 [-4.6250, 2.0391]；C：0.0018 [-0.1076, 0.1009] / -1.3942 [-5.4789, 1.9612]。
实际长度与对照：B 低/高组前缀均值 96.6889/108.8750 token; C 低/高组前缀均值 59.4490/170.0444 token；质量高低收益差点估计 不同向；时间高低收益差点估计 同向。同向时长度信号与 B 收益模式相容，但这不证明长度完全解释 B；不同向也不能证明 B 有独立作用，本轮没有做长度调整。
正式配对中立即检查修复 4 题、破坏 3 题；NOW 检查/回滚合计 308.0000/58.0000、撤销 token 25788.0000; DELAY_2 检查/回滚合计 289.0000/62.0000、撤销 token 28000.0000。
修复/破坏次数、首次检查 PASS/FAIL/UNCERTAIN、返工次数和撤销 token 可描述收益与成本的机制，但没有额外随机化，不能把它们直接解释为因果中介。格式失败和预算耗尽保持在算法失败分母，所有排除均保留记录。
反馈是 PRM 分数驱动的确定性通用 diagnosis/hint 模板；本轮只覆盖无历史 PASS 的普通 Step 2，不能直接推广到多位置或已有 PASS 状态。q 的独立前缀特征提取与探针时间属于离线诊断开销。小组 p95 仅为描述，分组样本过少、CI 跨零和 BF16 数值/运行后端差异都限制结论。
当前没有足够正式证据支持推进风险阈值策略；结果为 NA 或 CI 跨零时如实保留，不调整 test 寻找正结果。

pilot/dev/test 分开报告；未执行指标均为 NA。离线诊断和公共生成开销见下表（只对实际保存值汇总）：

| 阶段 | 公共生成秒均值 | 离线特征秒均值 | 探针秒均值 | 实验快照恢复秒均值 |
| --- | ---: | ---: | ---: | ---: |
| pilot | 1.9988 | 0.0374 | 0.0001 | 0.0387 |
| dev | 2.2924 | 0.0362 | 0.0001 | 0.0380 |
| test | 2.0520 | 0.0365 | 0.0001 | 0.0380 |
