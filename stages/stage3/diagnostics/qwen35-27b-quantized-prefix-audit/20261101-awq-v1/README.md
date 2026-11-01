# Qwen3.5-27B 4-bit 中途验证器对照

本目录只比较验证器，不执行生成器返工，不写入任何正式质量放行记录。原57条输入、9B原始输出及第二轮Codex标签保持不变。Codex标签为非盲模型暂定对照，不是独立专家真值。

## 固定条件

- 模型：`cyankiwi/Qwen3.5-27B-AWQ-4bit`，revision `9d8d63632afafe90bdfcb70164d3e2a0955619f7`。
- 格式：AWQ校准、`compressed-tensors`的`pack-quantized`，权重INT4、group_size=32、symmetric；未量化模块及具体配置见manifest。激活BF16，KV默认BF16，GDN状态FP32。不是全参数均为4-bit。
- 使用与既有9B结构化首轮完全相同的system prompt、user JSON、checkpoint编号、JSON Schema、每样本种子、采样参数及非Thinking的assistant前缀。tokenizer重新编码后57条token序列均与9B相同。只向模型传递题目和当前完整前缀等既有允许字段，不传参考答案、标准解答、9B判断或Codex标签。
- PASS/FAIL仍按已有当前有效推理口径；已纠正、明确放弃或由后续独立论证替代的错误不单独判FAIL；中途尚未完成不单独判错。
- 总上下文81920，`max_tokens=None`，不设固定输出token上限；最长完整输入61051，仍有20869输出空间。上下文耗尽或非完整EOS不生成数学标签。
- 单GPU、仅验证器常驻。1024-token chunked prefill、eager执行、显存比例0.90、禁用prefix caching；与9B的2048块、默认编译及148480总上下文不同，成本比较须保留这些差异。

## 执行

在仓库根目录，使用既有`/root/veriserve-vllm-env/bin/python`：

```bash
/root/veriserve-vllm-env/bin/python -m stages.stage3.quantized_prefix_audit --phase check
/root/veriserve-vllm-env/bin/python -m stages.stage3.quantized_prefix_audit --phase capacity
PYTHONPATH=. VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false /root/veriserve-vllm-env/bin/python stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v1/run_comparison.py
python stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v1/analyze.py
```

先单独完成第4、21、34条真实短、中、最长前缀请求，三个输入长度分别728、14632、61051。实际三条均完整EOS、无OOM/截断；21生成了字面backspace控制token196，严格JSON解析失败，另外两条可解析。不能把格式失败等同于显存容量失败。新增显式comparison入口读取这些证据后运行全部57条，不修改初始冻结runner及容量记录，不修补21的文本，不改提示词/采样/语法约束；格式失败仍作为不可用结果保留。详见capacity_assessment.json。这不是正式格式或质量放行。

采集前、每10条及阶段结束限定路径Git/LFS备份；失败停止后续采集。已有首轮记录不覆盖，进程中断的intent/response保留。需要重试时另建目录，不能用挑选重试结果替换首轮统计。

## 文件

- `manifest.json`、`input_inventory.json`、`payloads.json`、`requests/`：模型身份、配置、输入长度、完整无标签输入及token请求。
- `candidate_models.json`：候选模型的固定版本、文件大小和量化配置。官方GPTQ-Int4文件总量约30.26GB，不适合作为本卡长前缀测试的首选；所选社区版本约20.08GB。
- `download_metadata.json`、`download_complete.json`、下载日志：下载来源与权重SHA256验证。仅下载新候选，不删除已有4B/9B模型。一个约4GB分片放在系统盘，其余放数据盘，避免填满单个文件系统。
- `capacity/`、`comparison/`、`sessions/`：容量和首轮原始输出、完整token、解析状态、请求耗时、模型加载及20ms采样的全设备显存峰值。vLLM预分配KV池包含在显存中，不能将其解释为单条请求的净增量。
- `comparison.csv/json`、`summary.json`：全部57行和20题的描述性比较，缺失/异常单列。不得将同题前缀视为独立题。
- `thinkprm_readonly.json`：ThinkPRM-14B配置与来源，只读检查，未加载或执行该模型。

最终质量结论以完成后的报告及逐条理由审查为准。标签一致不代表定位、理由和提示正确，提示文本可用也不代表实际返工成功。
