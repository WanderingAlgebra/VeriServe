# 逐步 hidden 最终答错风险探针

在无检查、反馈、返工的连续生成中，用中途正常推理步骤的末尾 hidden 预测**完整轨迹最终答错**。分数不表示该步骤局部错误。只在完成且可评分的轨迹上评估，不补抽题、不翻转方向、不调参。

Qwen/Qwen2.5-7B-Instruct，BF16、单卡、batch 1、SDPA、greedy；4096 新 token / 8192 总 token。MATH-500 test 按 unique_id 排序，以 20261004 洗牌，依次取 100 train、200 test、10 smoke；另 190 不使用。模型、tokenizer、数据精确 SHA 见 [config.json](config.json)，首次 Hub 查询结果已固定，不使用浮动 main。

A 用轨迹末尾训练，B 用中途步骤训练，C 仅用当前步号和已生成 token 数。A/B 分别在训练内五折按题交叉验证选择一个固定层；StandardScaler + L2 LogisticRegression（C=0.1、max_iter=2000）。B/C 的标准化、拟合与验证按每题总权重 1。测试题在保存冻结探针后才读取；bootstrap 按题抽取 1000 次，报告 paired AUROC 差。

普通函数集中在 [run_probe.py](run_probe.py)，复用 stage1 原子 JSON 工具；stage1 实现和运行记录不改动。

## 实际环境

本服务器已有 `/root/miniconda3/bin/python`：Python 3.12.3、PyTorch 2.8.0+cu128；保持现有 GPU 环境，未安装新 CUDA/PyTorch。RTX 4090 实测 49,140 MiB，驱动 595.58.03（nvidia-smi CUDA 13.2，PyTorch CUDA build 12.8）；BF16 SDPA 前向通过。

补齐 Transformers 4.57.3、Datasets 4.8.5、scikit-learn 1.9.1、Math-Verify 0.9.0 和必要传递依赖；Math-Verify 按官方 extra `antlr4_13_2` 安装（ANTLR 4.13.2）。Git LFS 3.0.2 通过服务器 apt 安装。完整实际版本保存在运行 manifest.json 和 self_check.json。根 pyproject/uv.lock 是 stage1 的 Python 3.11 环境，不对它做整套同步或修改。

换机器时先选用已能跑 BF16 GPU 的 Python，再仅补缺少的依赖。例如（`python` 换成你的环境路径）：

```bash
uv pip install --python /path/to/python 'transformers==4.57.3' 'datasets==4.8.5' 'scikit-learn==1.9.1' 'math-verify[antlr4_13_2]==0.9.0'
```

## 执行与恢复

从仓库根目录运行。首次代码/清单推送成功后才启动 smoke 和正式题；推送使用现有分支 `experiment/step-hidden-probe`，不新建分支、不强推。需先在服务器配置 Git 作者和 origin 推送凭据，不在日志或聊天保存凭据。

```bash
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume --stop-after 1
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume
/root/miniconda3/bin/python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
```

`--stop-after 1` 在第一道 smoke 持久化后暂停；独立进程恢复，校验已完成 JSON/NPZ 哈希不变并跳过生成。也可完整 smoke 后再次执行 `--smoke --resume` 验证 10 题全部跳过。正式启动要求 10 题 causal 检查和恢复检查通过、已从远端取回 NPZ 核对哈希。causal 检查逐层记录后续 token 替换和独立前缀重前向的实际 max_abs/RMS；固定 BF16 容差 1/32 绝对误差 + 一个 BF16 相对 ULP（0.008），不因结果修改容差。

无 GPU 的 CPU 拟合/重评估不会加载生成模型：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```

缺少足够训练标签会保存 NA/不足报告，不借用测试题。每题先原子写生成 JSON，再提取 NPZ；特征缺失或校验失败时只重前向原始 `prompt_ids + generated_ids`，不重新生成。无法读取的损坏 JSON 留副本并尝试从本地 Git 已提交版本恢复；无可恢复 token 或校验不一致时停止，保留所有文件，不能重新生成来覆盖有效特征。smoke 与恢复检查绑定脚本/common 哈希；采集实现变化时复用原始 token，重新提取和验证，旧版本特征保留。无中途步骤的 smoke 题记为 causal 检查不适用，其轨迹仍保留；全部 smoke 均无有效位置时不允许冻结正式运行。

每 25 道正式题提交并常规推送代码、清单、JSON、NPZ；NPZ 用原生 LFS，小量层分片避免单文件大于 50 MiB。推送失败立即停止下一批，保留本地数据和 events.jsonl 错误；修复连接后原命令 `--resume` 会先重试备份。至少第一批特征通过独立临时仓库 `git lfs fetch/checkout` 取回并校验，结果在 backup.json。

换机器恢复已有仓库（已存在的 clone 不要再次 clone）：

```bash
git switch experiment/step-hidden-probe
git lfs install --local
git lfs pull origin experiment/step-hidden-probe
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
# 或只使用已保存特征
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```

无本地记录的正式采集只能在 smoke 完成后运行。配置/revision 变化自动产生新 run_id；正式冻结后采集脚本变化拒绝混入旧记录。如必须修改正式采集实现，应在配置增加说明性 `run_note` 生成新目录，不修改旧结果。

## 结果入口与解释范围

运行目录为 `stages/stage2/runs/<seed>-<config_hash前12位>/`。其中 report.md 是结论入口；manifest.json 保存全部题目原始字段、划分、revision、环境和脚本哈希；records/ 每题有原始 token、文本、评分/排除原因、步骤 token 对齐、硬件和分阶段耗时。cv_scores.csv、cv_folds.json、probe_*.npz/json、test_predictions.csv、metrics.json 和 step_auroc.png 保存分析。

正式结果未产生时 report.md 明确列出未执行阶段与 NA，不生成示意实验数字。CPU 自检仅用临时合成样本，不能当作 smoke 或正式结果。实际 smoke、恢复和远端核验分别保存 smoke_checks.json、resume_check.json、backup.json。

端点以原始 generated_ids 为权威；只用 prefix decode 对齐，不重分词。跨界 token 向前取最后完整内容 token；包含 boxed 或明确最终答案的步骤排除，最后正常步骤保留 near_end。轨迹末尾取 EOS/控制 token 前最后完整内容 token。Qwen2 hidden_states[1..L] 排除 embedding，最后索引含 final norm。

评分只从独立 Final answer 行取最后完整 boxed，以 Math-Verify 判等。截断、缺失/不完整 boxed、标准/预测答案不可解析、解析/评分超时分别排除；不从推理中寻找正确答案。无有效中途步骤的轨迹保留并计数，可用于 A 的末尾训练。

只回答这个设置下中途 hidden 是否提供超过随机和简单进度基线的最终答错风险信号；不能推出首个错误步骤、检查阈值、检查收益、降低请求完成时间或其他模型上的通用性。离线重前向时间不当作在线低成本。

参考：[原论文 §3.1/§4.1/B/C](https://arxiv.org/html/2605.09502v1)、[Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[Math-Verify 安装/解析](https://github.com/huggingface/Math-Verify)。逐步训练、题目权重、提示词和生成上限是本实验适配。
