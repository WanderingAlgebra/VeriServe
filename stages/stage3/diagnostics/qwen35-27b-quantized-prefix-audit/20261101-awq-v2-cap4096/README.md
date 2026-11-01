# 27B AWQ：4096输出token上限的完整57条对照

本目录为独立后续运行。此前`20261101-awq-v1`在第27条约526.6秒仍未返回后中断，保留26条响应、第27条中断及全部日志。中断请求无法获取部分输出/token数，记未知，不生成数学标签。旧部分结果不混入本目录。

本轮全部57条重新使用相同的4096输出token技术上限。该上限不是判定口径；达到上限即记截断，不以残缺文本判PASS/FAIL。原9B无固定输出上限，已保存的最长输出3013 token；预算差异需在成本及质量比较中明确披露。

模型、revision、量化、81,920总上下文、非Thinking模板、system prompt、结构化schema、完整原题/前缀、逐样本种子、采样、1024 prefill和eager执行均与v1保持相同。不截断任何输入，不提供标准答案、后续推理、9B判断或Codex标签。仍以既有第二轮Codex模型复核作为暂定对照，不修改标签，不执行生成器返工，不写正式质量放行。

复用v1中三条完整原题/前缀的真实容量证据：728、14632、61051输入token均生成至EOS，无OOM/截断；实际输出均低于4096。第21条JSON控制字符错误原样保留。相同引擎/上下文降低输出预算不增加显存需求，故不重复这三条容量请求。复用来源和哈希见`capacity_reuse.json`。

运行（仓库根目录）：

```bash
PYTHONPATH=. VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false /root/veriserve-vllm-env/bin/python stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v2-cap4096/run_comparison.py
python stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v2-cap4096/analyze.py
```

每10条及阶段结束限定路径Git/LFS备份。所有57行均保留，格式失败、输出截断及异常单列。权重下载、候选比较、ThinkPRM只读配置及原容量证据见相邻v1目录；运行原始输出在本目录`comparison/`。
