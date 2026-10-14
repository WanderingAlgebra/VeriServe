# 第二阶段：逐步 hidden 最终答错风险探针

原生 Thinking 内的真实双换行定义候选检查点，不保证每段是完整数学推理单元。
step_1/2/3 等历史字段在本 run 中表示第 1/2/3 个有效检查点；HIGH/LOW 是离线参考位置诊断。
格式示例固定在配置中（0 个简单问答），不来自实验题集；stage2/stage3 输入一致。
最终响应允许解释，仅评分末尾唯一且完整的 Final answer boxed 行；Thinking 内引用不参与评分。

运行：`v2-20261013-854c63b5c942`；报告更新于 2026-10-14T21:34:52.039812+08:00。

标签是完整轨迹最终答错（1）或答对（0）。主指标仅用于完成且可评分轨迹的正常中途检查点；不是局部错误标签。

模型 `Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；tokenizer `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
数据 `KbsdJames/omni-math-rule` / rule，revision `bda09972374225ab2186582615307d909965b864`。

这是采用逐步训练、题目权重、固定双换行检查点和独立测试的适配实验，不宣称完全复现论文。
原生 Thinking；采样与逐题种子见记录；新 token 81920，总上下文 131072。
按题去重后排序、固定种子划分：{'smoke': 10, 'test': 200, 'train': 100, 'verifier_audit': 20, 'verifier_dev': 20}；数据完整性见 dataset_integrity.json。
hidden_states[1..L] 排除 embedding；实际层数、维度和末层 final norm 约定见模型检查。
presence penalty 对生成与反馈后缀的每个已见 token 施加一次，不作用于题目 prompt。
探索实验；旧结果只作历史参照，不能将差异归因于单一变化。


特征逐个端点重前向其原始 token 前缀，保持 BF16 / SDPA / use_cache=False。
逐前缀提取另与独立 decoder 的最后位置比较，并检验同长度未来替换及错一 token 的负对照；容差未扩大。

## 实际环境

```json
{
  "python": "3.12.3",
  "executable": "/root/project/VeriServe-step-hidden-probe/environments/qwen35/.venv/bin/python",
  "platform": "Linux-5.15.0-78-generic-x86_64-with-glibc2.35",
  "versions": {
    "torch": "2.8.0+cu128",
    "transformers": "5.19.0",
    "datasets": "4.8.5",
    "numpy": "2.3.2",
    "scikit-learn": "1.9.1",
    "math-verify": "0.9.0",
    "antlr4-python3-runtime": "4.13.2",
    "latex2sympy2-extended": "1.11.0",
    "huggingface-hub": "1.33.0",
    "matplotlib": "3.10.5"
  },
  "cuda_build": "12.8",
  "cuda_available": true,
  "nvidia_smi": "NVIDIA GeForce RTX 4090, 49140 MiB, 48507 MiB, 595.58.03",
  "bf16_supported": true,
  "bf16_sdpa_forward": true,
  "free_total_bytes": [
    50353602560,
    50866487296
  ]
}
```

## 采集与排除

```json
{
  "formal_planned": 300,
  "formal_generated": 0
}
```

未完成原因：Prepared; collection and remote backup have not yet been verified

## 自检与备份

```json
{
  "cpu_self_check": null,
  "additional_implementation_checks": {
    "checked_at": "2026-10-04T12:33:07.087305+00:00",
    "layer_index_and_final_output": true,
    "note": "Implementation checks on a tiny randomly initialized Qwen2 decoder, not the requested 7B model; no experimental trajectories.",
    "pinned_tokenizer_alignment": [
      {
        "boundary_alignment": true,
        "effective_steps": 2,
        "excluded_steps": 0,
        "tokens": 29
      },
      {
        "boundary_alignment": true,
        "effective_steps": 1,
        "excluded_steps": 1,
        "tokens": 42
      }
    ],
    "script_sha256": "866da90db05d1d5467e16b2834b7fb6ff03fdfc63872a65caa0aad0f8081f745",
    "tiny_qwen_bf16_sdpa_causal": {
      "atol": 0.03125,
      "changed_future": {
        "max_abs": 0.0,
        "max_abs_per_layer": [
          0.0,
          0.0
        ],
        "passed": true,
        "rms": 0.0
      },
      "independent_prefix": {
        "max_abs": 0.0,
        "max_abs_per_layer": [
          0.0,
          0.0
        ],
        "passed": true,
        "rms": 0.0
      },
      "passed": true,
      "position": 5,
      "rtol": 0.008,
      "tolerance_note": "BF16 rounding: atol=1/32, rtol=one BF16 relative ULP; report raw errors"
    }
  },
  "smoke": null,
  "resume": null,
  "backup": null
}
```

AUROC / CI / 所选层：NA；正式数据不足或未采集，不能回答是否存在信号。

这项实验不能识别首个错误检查点、判断当前检查是否值得、证明降低完成时间或推广到其他模型。
离线重前向耗时不代表在线提取成本；生成、重前向与探针评分耗时分别记录。

检查点端点对齐原始 token，跨界 token 向前对齐；包含 boxed 或明确最终答案的检查点排除，最后正常检查点保留 near_end 标记。

评分仅解析单独 Final answer 行的最后完整 boxed；Math-Verify 标准答案包装为 $...$，无字符串 fallback，解析/比较超时单独排除。

## 文件与恢复

配置与环境见 config.json / manifest.json；运行日志见 events.jsonl；探针参数见 probe_{A,B,C}.npz/json。
分支：[experiment/step-hidden-probe](https://github.com/WanderingAlgebra/VeriServe/tree/experiment/step-hidden-probe)。
本地 HEAD：`413763d22b8e3a9fc0149503fc4db7ac38c0d1dd`；远端备份状态与提交以 backup.json 为准。

从仓库根目录执行（替换成可用 Python 环境）：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```
