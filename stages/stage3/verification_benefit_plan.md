# Qwen3.5 验证收益实验方案

## 目的与当前状态

本实验比较：在同一道题、同一个推理前缀处，直接继续解题，与立即验证后继续或返工，最终答案质量、token 消耗和处理时间有什么不同。

已实现第一轮生成、选点、两条分支、预算统计和按题分析的执行代码，并完成 CPU 模拟检查。**独立题目上的验证器核验和 32GB GPU 预跑尚未完成**，CPU 检查不能证明真实推理可用。本次准备不启动实验，也不重新执行 Stage2。

此前的 `verification_benefit_selection.json` 和选点脚本针对旧 Stage2 轨迹生成，只保留作历史准备记录；其中的检查点、覆盖数量、配对种子和探针分数均不能作为本轮新轨迹的选点或结果。本轮的题目可沿用相同 manifest 中的题目，但每题的第一轮回答必须重新生成。

## 题目和第一轮回答

从 `stages/stage2/runs/v2-20261013-e136a41fea81/manifest.json` 的 100 道训练题中，按原顺序取前 5 道作为预跑题，接下来的 20 道作为探索题。两批不重叠，不使用 200 道测试题，也不按正确与否、长度或探针分数筛选。只复用题目及其标准答案，不复用旧生成轨迹或旧检查点。

使用 Qwen3.5-4B 逐题重新生成第一轮回答，保存完整 prompt token、生成 token、原始文本、最终评分和终止原因。第一轮最多输出 **65,536 token**，包括 Thinking 和最终回答；到达上限仍未结束就截断并记录。生成用 vLLM，精度为 BF16，不加载 Transformers 模型提取 hidden states。

## 从第一轮回答确定检查点

第一轮结束后，分别从已输出 8,192、16,384 和 32,768 token 的位置向后查找，选第一个完整、合法的 Thinking 段落结尾作为检查点，排除明确的最终答案段。只有从原始 token 可以完整还原的段落结尾才使用。

这里的段落是按原生 Thinking 中的双换行识别的检查点，不保证是完整数学步骤。段落结束需要参考后面的分隔符才能在离线轨迹中确认；这不表示在线运行时可以提前知道段落已结束。三个长度档只表示生成长度，不代表不同题目处于相同的语义推理阶段。

提前结束、截断或没有合格检查点时，逐档记录原因，不强行续写，也不换题。多个长度档选中同一检查点时只执行一次，并记录该检查点对应的所有档位。新轨迹生成后才产生本轮逐题选点清单和实际覆盖数量。

## 在每个检查点比较两条路径

每个合格检查点运行两条分支，每条分支各运行一次。两条分支使用同一道题、相同的原始 prompt 和检查点前缀，采用相同的 4B 采样参数和事先固定的配对随机种子。配对种子不要求干预后生成相同 token。

| 情况 | 动作 | 4B 后续输出预算 |
| --- | --- | --- |
| A：不验证 | 从原始检查点前缀继续解题 | `65,536 − n` |
| B：验证通过或不确定 | 调用一次验证器，再从相同前缀继续解题 | `65,536 − n` |
| B：验证失败 | 回到原始题目，加入验证反馈后重新生成 | **额外最多 65,536** |

表中 `n` 是第一轮到该检查点为止已输出的 token 数。失败时这段前缀被撤销，但其生成消耗仍要记录；返工额度不减去 `n`。每条路径最多验证一次，后续不再做中途或终点验证。验证器使用 Qwen3.5-9B 的现有直接判断模式；格式错误、截断等按既有严格解析规则记录并处理为 UNCERTAIN。两条分支的最终答案都用现有标准答案评分方式判分，验证器的 PASS 不是答对标签。

## token 用量与超限判断

对每条第二轮路径，主要比较指标为：

**第一轮完整回答的输出 token 数 + 第二轮完整回答的输出 token 数**，并记录该和是否大于 65,536。

第一轮截断时使用实际输出数，并保留未完成标记。第二轮直接续写时，完整回答长度为保留前缀 `n` 加后续新生成 token；验证失败重新生成时，完整回答长度为返工新生成 token 数。各自到 65,536 上限未完成时均记为截断。对未完成的回答，超限判断仅描述已观察到的 token 数，不能推断完整回答本来会有多长。

上述比较会在直接续写时把保留的前缀计入第二轮回答长度，因此它是**回答长度之和**，不是实际生成成本。实际成本另行报告：第一轮生成 token、第二轮新增生成 token、撤销前缀 token、9B 验证输入和输出 token、反馈输入 token，以及所有重新 prefill 的输入 token。验证和反馈不占 4B 的输出预算，也不混入上述两个回答长度。

例如，第一轮完整输出 30,000 token，在第 16,384 token 处验证失败，返工输出 50,000 token。回答长度之和是 `30,000 + 50,000 = 80,000`，超过 65,536；第一轮与返工的实际 4B 新生成量也是 80,000。若选择直接续写且新增 20,000 token，第二轮回答长度是 `16,384 + 20,000 = 36,384`，回答长度之和是 66,384；此时实际新生成量是 `30,000 + 20,000 = 50,000`。

## 执行方式和计时

单张 32GB vGPU 上优先使用 4B 和 9B 分时加载、卸载或切换，不假定两者能同时驻留。恢复时使用保存的原始 token 前缀，不将它重新包装成用户消息；允许用 vLLM 重新 prefill。上下文容量还需容纳题目和反馈，不能直接等同于 65,536 的输出预算。

记录公共前缀准备与恢复的耗时；验证引起的模型切换、验证调用、额外重算、反馈和返工都计入 B 分支成本，并单列模型加载、生成、评分和持久化时间。每条路径保存原始输出、终止原因、评分、验证判决与反馈、token 计数、实际耗时和显存记录。重新 prefill 的结果不能称为在线 KV Cache 复用收益。

采样需保持原有规则：presence penalty 对已生成或反馈中出现过的 token 施加一次惩罚，不作用于原始题目输入。现有 vLLM 接口将保存的前缀视为输入，不能自动恢复这段生成历史；执行入口必须处理并验证这一点。

## 分析与验收

报告两条分支的可评分数量和正确率变化、A 错 B 对及 A 对 B 错的数量、回答长度之和及其超限比例，以及实际生成、重算和时间差。缺失、截断和不可评分情况单独报告。分长度档比较，并在新轨迹有可用探针分数时作探索性分组；旧轨迹的分数不能直接套用到新轨迹，缺失记为 NA。本轮不重新采集整批 hidden states，也不训练新的收益预测器。

同题的多个检查点不视为独立题：总体汇总按题等权，区间估计按题重抽样。三个长度档不作为不同题目的共同语义阶段，训练题上的探针分数也不能视作独立泛化评估。

正式探索批次之前，需先完成代码和 CPU 检查，再进行 5 题真实 GPU 预跑，检查原始 token 恢复、生成预算、回滚反馈、评分、计时、分时切换与 32GB 显存容量。旧 48GB 测试不能当作本卡的验收记录。本轮按用户要求允许模型提供核验参考标签，保留核验阈值和协议冻结要求；不能把格式检查或容量通过当作验证器质量通过。

采集前和采集中继续遵守 Stage3 的 Git/LFS 远端备份要求；备份失败时停止后续采集。预跑完成后不得自动启动探索批次。

CPU 准备与检查命令（在仓库根目录执行，不启动 GPU）：

```bash
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase prepare
environments/qwen35/.venv/bin/python -m unittest tests.test_verification_benefit_cpu -v
/root/veriserve-vllm-env/bin/python -m unittest tests.test_verification_benefit_logits -v
```

未来的 GPU 预跑入口是 `environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase pilot`；它在启动任何模型前要求本轮对应的 vLLM 验证器在独立题目上的核验通过，且远端备份成功。**目前尚无通过的核验记录，因此该入口当前会停止，不可当作已具备预跑条件。** 探索批次还要求预跑验收记录，并须另行显式启动。

核验优先使用 manifest 中与实验题互不重叠的 20 道 `verifier_audit` 题。采集时按固定长度位置选取前缀和终点，生成不含 9B 验证器判决的标注输入。按用户要求，参考标签可以由模型或人提供，不要求人工确认。默认使用 Qwen3.5-4B 标注；它可读取题目、前缀和标准解答，但看不到 9B 的判决。标准答案和解答不输入在线验证器。

程序记录标签来源、标注模型和原始输出，并按参考标签检查：至少 20 个正确前缀和 20 个错误前缀，错误 PASS、正确 FAIL、所有输入的 UNCERTAIN/格式错误/截断比例均不超过 20%，Balanced Accuracy 至少 80%。模型格式错误、截断或无法判断时记为 `UNDETERMINABLE`，不计入正确或错误类别。若类别不足或未通过，不启动预跑。类别不足时可用 `--audit-extra N` 按原 manifest 的未用题固定顺序追加核验题；预留记录阻止不同核验配置重复使用题目，不按模型判决挑题。

**模型标签可能有误。** 使用模型标签时，上述指标表示 9B 与参考模型判断的一致性，不能称为相对于人工真值的准确率。默认标注模型和生成器同为 Qwen3.5-4B，相关性可能影响结果；报告保留这一限制。也可提供带来源的人工或模型标签文件，但人工确认不再是放行条件。此变更只适用于本轮验证收益实验，旧 HIGH/LOW 材料保持原样。

以下命令仅供后续明确启动核验时使用；前两条会运行 GPU 模型并要求远端备份。模型标注不读取 9B 的判决：

```bash
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase audit-collect
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase audit-label
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase audit-report
```

也可在最后一条命令加 `--labels <标签文件.json>`；每个标签需记录 `label_source`（`human` 或 `model`）和 `labeler`。程序根据实际来源报告结果，不把模型标签写成经过人工确认的标签。

核验通过后，后续明确授权运行时可依次使用下面的命令。`capacity` 用合成输入检查各模型的完整配置上下文能否在 32GB 内运行；它不能代替真实预跑。`forced-check` 以短输出验证从原前缀续写和从题目加反馈返工，诊断结果不计入实验。`certify-pilot` 检查核验、容量、诊断、5 题材料及计时后冻结协议；缺少证据时会停止。

```bash
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase capacity
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase pilot
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase forced-check
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase certify-pilot
# 仅在预跑验收通过并明确决定执行探索批次后运行：
environments/qwen35/.venv/bin/python -m stages.stage3.verification_benefit_run --phase exploratory
```
