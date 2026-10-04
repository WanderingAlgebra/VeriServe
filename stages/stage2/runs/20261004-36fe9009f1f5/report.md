# 第二阶段：逐步 hidden 最终答错风险探针

运行：`20261004-36fe9009f1f5`；报告更新于 2026-10-04T20:41:39.406944+08:00。

标签是完整轨迹最终答错（1）或答对（0）。主指标仅用于完成且可评分轨迹的正常中途步骤；不是局部错误标签。

模型 `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`；tokenizer `a09a35458c702b33eeacc393d103063234e8bc28`。
数据 `HuggingFaceH4/MATH-500` / test，revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。
按 unique_id 排序后用种子 20261004 洗牌：100 train / 200 test / 10 smoke，剩余 190 不使用；原始字段见 manifest.json。
BF16 / 单卡 / batch 1 / SDPA / greedy；4096 新 token、8192 总 token；C=0.1、max_iter=2000；所有 Transformer 层分别做训练内五折选层。

这是采用逐步训练、题目权重、明确 Step 标记和独立测试的适配实验，不宣称完全复现论文。

## 实际环境

```json
{
  "bf16_sdpa_forward": true,
  "bf16_supported": true,
  "cuda_available": true,
  "cuda_build": "12.8",
  "executable": "/root/miniconda3/bin/python",
  "free_total_bytes": [
    50353602560,
    50866487296
  ],
  "nvidia_smi": "NVIDIA GeForce RTX 4090, 49140 MiB, 48507 MiB, 595.58.03",
  "platform": "Linux-5.15.0-78-generic-x86_64-with-glibc2.35",
  "python": "3.12.3",
  "versions": {
    "antlr4-python3-runtime": "4.13.2",
    "datasets": "4.8.5",
    "huggingface-hub": "0.36.2",
    "latex2sympy2-extended": "1.11.0",
    "math-verify": "0.9.0",
    "matplotlib": "3.10.5",
    "numpy": "2.3.2",
    "scikit-learn": "1.9.1",
    "torch": "2.8.0+cu128",
    "transformers": "4.57.3"
  }
}
```

## 采集与排除

```json
{
  "train": {
    "generated": 0,
    "completed": 0,
    "correct": 0,
    "wrong": 0,
    "truncated": 0,
    "unscorable": 0,
    "no_steps": 0,
    "format_error_trajectories": 0,
    "features_missing_or_corrupt": 0,
    "planned": 100,
    "not_generated": 100,
    "analyzable_end": 0,
    "analyzable_intermediate": 0
  },
  "test": {
    "generated": 0,
    "completed": 0,
    "correct": 0,
    "wrong": 0,
    "truncated": 0,
    "unscorable": 0,
    "no_steps": 0,
    "format_error_trajectories": 0,
    "features_missing_or_corrupt": 0,
    "planned": 200,
    "not_generated": 200,
    "analyzable_end": 0,
    "analyzable_intermediate": 0
  },
  "smoke": {
    "generated": 0,
    "completed": 0,
    "correct": 0,
    "wrong": 0,
    "truncated": 0,
    "unscorable": 0,
    "no_steps": 0,
    "format_error_trajectories": 0,
    "features_missing_or_corrupt": 0,
    "planned": 10,
    "not_generated": 10,
    "analyzable_end": 0,
    "analyzable_intermediate": 0
  }
}
```

## 自检与备份

```json
{
  "cpu_self_check": {
    "checked_at": "2026-10-04T12:39:54.262871+00:00",
    "checks": [
      "grade \\frac{1}{2} vs 0.5",
      "grade \\sqrt{8} vs 2\\sqrt{2}",
      "grade x+x vs 2x",
      "grade 2 vs 3",
      "missing boxed",
      "incomplete boxed",
      "last complete nested box",
      "only final line graded",
      "truncation excluded",
      "post-final text excluded",
      "cross-boundary token aligned backward",
      "second-step original token index",
      "answer step excluded",
      "last content token before EOS",
      "near-end retained",
      "anomalous numbering not guessed",
      "incomplete Unicode endpoint excluded",
      "each question total weight one",
      "weighted scaler",
      "A/B/C fitting and grouped CV",
      "layer tie chooses smaller index",
      "weighted evaluation and paired bootstrap",
      "single-class bootstrap reports NA",
      "resume skips complete JSON and NPZ without generation",
      "damaged NPZ detected",
      "unrecoverable JSON never triggers regeneration",
      "no-step causal check is not a failure"
    ],
    "environment": {
      "executable": "/root/miniconda3/bin/python",
      "platform": "Linux-5.15.0-78-generic-x86_64-with-glibc2.35",
      "python": "3.12.3",
      "versions": {
        "antlr4-python3-runtime": "4.13.2",
        "datasets": "4.8.5",
        "huggingface-hub": "0.36.2",
        "latex2sympy2-extended": "1.11.0",
        "math-verify": "0.9.0",
        "matplotlib": "3.10.5",
        "numpy": "2.3.2",
        "scikit-learn": "1.9.1",
        "torch": "2.8.0+cu128",
        "transformers": "4.57.3"
      }
    },
    "note": "Synthetic CPU checks only; actual model causal/GPU and remote LFS checks belong to smoke",
    "passed": true,
    "script_sha256": "4719975530b8db4f420f0b62ab6b2f12f4088d102db73e8de012e806da6c0031"
  },
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
  "backup": {
    "commit": "e721101e58afd9487ba047fd034c49e644400ac8",
    "feature_remote_verification": null,
    "pushed_at": "2026-10-04T12:40:32.250531+00:00",
    "status": "PUSHED"
  }
}
```

AUROC / CI / 所选层：NA；正式数据不足或未采集，不能回答是否存在信号。

这项实验不能识别首个错误步骤、判断当前检查是否值得、证明降低完成时间或推广到其他模型。
离线重前向耗时不代表在线提取成本；生成、重前向与探针评分耗时分别记录。

步骤端点对齐原始 token，跨界 token 向前对齐；包含 boxed 或明确最终答案的步骤排除，最后正常步骤保留 near_end 标记。
hidden_states[1..L] 排除 embedding，Qwen2 的索引 L 含最终 norm；轨迹末尾取 EOS/控制 token 前最后完整内容 token。

评分仅解析单独 Final answer 行的最后完整 boxed；Math-Verify 标准答案包装为 $...$，无字符串 fallback，解析/比较超时单独排除。

## 文件与恢复

配置与环境见 config.json / manifest.json；运行日志见 events.jsonl；探针参数见 probe_{A,B,C}.npz/json。
分支：[experiment/step-hidden-probe](https://github.com/WanderingAlgebra/VeriServe/tree/experiment/step-hidden-probe)。
本地 HEAD：`e721101e58afd9487ba047fd034c49e644400ac8`；远端备份状态与提交以 backup.json 为准。

从仓库根目录执行（替换成可用 Python 环境）：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```
