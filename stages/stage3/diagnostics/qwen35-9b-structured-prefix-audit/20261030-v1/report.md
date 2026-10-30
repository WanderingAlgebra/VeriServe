# Qwen3.5-9B：57前缀结构化输出重跑

首轮已运行0/57条，可用PASS/FAIL共0条；失败/未运行分类：{'not_run': 57}。
首轮判定分布：{'未完成': 57}。重试共0次，状态为{}；合并后可用0/57条。

原首检为39 PASS、18 UNCERTAIN、0 FAIL；18条均伴随格式/反馈长度或截断问题。新结果的格式成功率与数学判断一致性分开报告。

## 首轮与Codex暂时对照

| Codex暂时对照 | 9B PASS | 9B FAIL | 未完成 |
|---|---:|---:|---:|
| PASS | 0 | 0 | 36 |
| FAIL | 0 | 0 | 21 |

## 含重试的最终对照

| Codex暂时对照 | 9B PASS | 9B FAIL | 未完成 |
|---|---:|---:|---:|
| PASS | 0 | 0 | 36 |
| FAIL | 0 | 0 | 21 |

仅在有效新判断上，与Codex标签的一致率为None；按题等权为None（0题）。这不是相对绝对真值的准确率。57个前缀来自20题，存在前缀包含关系。

## 原先10条分歧

| 编号 | 主要推导/结论 | 原9B | 新首轮 | 重试后 | Codex |
|---|---|---|---|---|---|
| 10 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 20 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 21 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 23 | 是 | PASS | 未完成 | 未完成 | FAIL |
| 25 | 是 | PASS | 未完成 | 未完成 | FAIL |
| 42 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 48 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 51 | 局部问题 | PASS | 未完成 | 未完成 | FAIL |
| 52 | 是 | PASS | 未完成 | 未完成 | FAIL |
| 54 | 是 | PASS | 未完成 | 未完成 | FAIL |

## 成本和异常

首轮输出token：{'count': 0, 'total': 0, 'mean': None, 'median': None, 'max': None}。
首轮每请求秒数：{'count': 0, 'total': 0, 'mean': None, 'median': None, 'max': None}。
首轮结束原因：{}。原首检失败分类：{'Invalid feedback type or length': 9, 'VERIFIER_LENGTH_TRUNCATED': 5, 'Invalid \\escape: line 4 column 238 (char 286)': 1, 'Invalid \\escape: line 4 column 886 (char 929)': 1, 'Invalid \\escape: line 4 column 368 (char 423)': 1, 'Invalid \\escape: line 4 column 195 (char 239)': 1}。
重试输出token：{'count': 0, 'total': 0, 'mean': None, 'median': None, 'max': None}；重试耗时：{'count': 0, 'total': 0, 'mean': None, 'median': None, 'max': None}。
原首检输出token：{'count': 57, 'total': 11407, 'mean': 200.12280701754386, 'median': 31, 'max': 1024}；原首检生成耗时：{'count': 57, 'total': 504.5577324721962, 'mean': 8.851890043371863, 'median': 5.300360724329948, 'max': 30.94153644144535}。

新运行按阶段保持模型驻留，原运行每条启动并退出模型进程。原seconds包含进程启动、模型加载等，新request_seconds不包含阶段级模型初始化，不能直接据二者之差宣称结构化输出加速。模型初始化、显存峰值和版本保存在sessions/；所有原始输出、token ID、结束原因和每次尝试保存在各阶段目录。
新完成输出中超过旧300字符上限的条目：[]。这部分可用性改善也受放宽字符上限影响，不能全部归因于语法约束。

## 配置与归因限制

模型Qwen/Qwen3.5-9B，revision=c202236235762e1c871ad0ccb60c8ee5ba337b9a。复用原始57条请求中的完整用户JSON、原题、检查点编号和逐条随机种子，未输入参考答案、原9B结论、Codex标签或后续轨迹。prepare逐条检查了原始token往返、JSON载荷及Codex编号/ID/review_key对应。
实际vLLM版本0.20.0。接口经本机签名检查，并查阅[vLLM对应版本文档](https://docs.vllm.ai/en/v0.20.0/features/structured_outputs/)，采用SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))和xgrammar后端。

以下变化均已在manifest记录：
- JSON Schema via SamplingParams.structured_outputs; backend=xgrammar.
- UNCERTAIN removed; no concrete unresolved error -> PASS. This changes the decision space.
- PASS diagnosis now contains a concrete explanation; hint remains empty.
- Removed300-character diagnosis/hint cap; fields remain mandatory.
- Resident sequential engine per invocation, rather than one model initialization per sample. max_num_seqs=1 and prefix caching disabled; requests have independent original seeds.
- Original eos IDs, sampling, model revision, precision, context and first-round1024-token cap retained.

原核验标准中的当前采用推导、自我纠正和中途/终点区别保留。旧协议终点额外要求答案完整且受推理支持；Codex标准主要检查当前未纠正数学错误，两者存在口径差别。
二选一移除了原数学不确定出口；PASS增加诊断理由；删除300字符限制；加入语法约束改变可选token及随机采样路径。因此数学标签变化不能单独归因于“修好了JSON”。重试提高预算的结果也不能替换首轮统计。
此次使用既有57条进行探索性诊断，Codex为暂时对照且自身存在复核范围和题意歧义限制。此次不创建或更新原实验质量放行文件。

## 新9B与Codex不一致的条目

共0条：[]。双方完整理由见[分歧明细](disagreements.md)及comparison.csv，供复核，不裁定哪方一定正确。

## 下一步

先复核23、25、52、54的双方具体推导，再处理局部笔误和其他分歧。若需要量化纯语法约束的因果作用，应另行保持提示词、判定空间、解析规则和引擎生命周期完全相同作对照；本次不把混合配置变化当作该因果实验。
