# 无固定输出上限的57前缀诊断

用户在原1024-token首轮进行中要求取消输出上限。因此原20261030-v1已停止并保存11条完整/截断结果及1条中断尝试，本目录重新对57条执行，不混入旧部分结果。

所有新smoke、首轮和重试都显式使用SamplingParams(max_tokens=None)。已检查vLLM0.20实际安装代码：None会被解析为max_model_len减输入token数，即只受148480总上下文物理边界限制。每条记录max_tokens=null、context_output_capacity、实际output_tokens、耗时及结束原因；若耗尽上下文仍按未完成处理。

执行工作树为/root/project/VeriServe-step-hidden-probe。原题、前缀、模型、采样和原始随机种子保持不变，Codex暂时对照从主工作树读取。

先运行3个无固定输出上限的真实smoke（4、23、34），以stop-after1停在完整请求边界后重启，确认恢复。截断解析验证引用前一目录已完成的1-token诊断，不在用户新指令后再发有限输出预算请求。

运行或恢复首轮：

```bash
/root/veriserve-vllm-env/bin/python -u -m stages.stage3.structured_prefix_audit --run stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-unbounded-v2 --phase first
```

同一命令跳过所有已记录首轮结果。必要时用phase retry在单独目录重试，仍为max_tokens=None，每个失败样本至多自动执行一次这一阶段。

重建对照表和报告：

```bash
python stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-unbounded-v2/analyze.py
```

每次请求先写intent、返回后保存原始response、再写解析attempt。格式、截断、上下文耗尽或服务异常不生成数学标签。执行源码prepare后固定，不改写旧manifest。采集前、每10条及阶段结束限定路径Git/LFS备份，失败停止后续采集。
