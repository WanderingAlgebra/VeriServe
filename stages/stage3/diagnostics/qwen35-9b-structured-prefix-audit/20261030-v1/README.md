# 运行和恢复

这是对既有57条前缀的独立诊断，不修改原实验、原9B结果或Codex标签，不产生原实验的质量放行文件。

执行工作树：`/root/project/VeriServe-step-hidden-probe`，分支`experiment/step-hidden-probe`。Codex对照文件从主工作树`/root/project/VeriServe`读取；其路径、内容校验和、逐条标签及理由保存在`manifest.json`。原始9B请求、随机种子与结果也逐条关联到manifest。

在执行工作树根目录，使用已有环境：

```bash
/root/veriserve-vllm-env/bin/python -m unittest tests.test_structured_prefix_audit -v
```

本目录已prepare，重复prepare会拒绝覆盖。另开独立目录时才使用`--phase prepare`。

本轮先执行了3个真实GPU smoke：4、23、34。第一次以`--stop-after 1`只完成4；随后重启进程继续相同phase，自动跳过4并完成23、34。`resume_check.json`确认原结果未改写且没有新增第二次尝试。另有`smoke_truncation`用前缀4和1-token上限，确认截断被保存为`judgment=null`。这些调用均不混入首轮57条统计。

```bash
/root/veriserve-vllm-env/bin/python -u -m stages.stage3.structured_prefix_audit \
  --run stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-v1 --phase first
```

同一命令会跳过已有首轮结果，包括已经记录的失败。原始首轮不会被重试覆盖。仅对首轮失败样本提高预算时使用独立phase：

```bash
/root/veriserve-vllm-env/bin/python -u -m stages.stage3.structured_prefix_audit \
  --run stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-v1 --phase retry --budget 2048
```

若仍有未完成，可再次使用`--phase retry --budget 4096`；已经成功的样本自动跳过，同预算失败也不会重复收费。所有尝试按`attempt_01.json`、`attempt_02.json`保存。每次请求前先落盘intent，推理返回后立即保存response，完成解析后保存attempt。异常和截断不生成数学标签。

重建报告不启动模型：

```bash
python stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-v1/analyze.py
```

`comparison.csv`保存57条逐项对照；`disagreements.md`列出新9B与Codex的双方理由；`summary.json`为机器可读统计。原始输出、token、结束原因和每次耗时位于`first/`、`retry/`及`smoke*/`。`sessions/`另记模型初始化、显存峰值和版本。

按照本实验备份要求，采集前、每10条及阶段结束执行限定路径的Git/LFS提交和推送；不会暂存已有用户改动。备份失败会停止后续采集，文件保留本地。诊断代码一旦prepare即固定，修改执行代码需要另建run，不能改写原manifest绕过身份检查。
