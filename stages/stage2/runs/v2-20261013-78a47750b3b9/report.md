# 第二阶段：逐步 hidden 最终答错风险探针

原生 Thinking 内的真实双换行定义候选检查点，不保证每段是完整数学推理单元。
step_1/2/3 等历史字段在本 run 中表示第 1/2/3 个有效检查点；HIGH/LOW 是离线参考位置诊断。
格式示例固定在配置中（0 个简单问答），不来自实验题集；stage2/stage3 输入一致。
最终响应允许解释，仅评分末尾唯一且完整的 Final answer boxed 行；Thinking 内引用不参与评分。

运行：`v2-20261013-78a47750b3b9`；报告更新于 2026-10-14T09:42:00.144198+08:00。

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
  "bf16_sdpa_forward": true,
  "bf16_supported": true,
  "cuda_available": true,
  "cuda_build": "12.8",
  "executable": "/root/project/VeriServe-step-hidden-probe/environments/qwen35/.venv/bin/python",
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
    "huggingface-hub": "1.33.0",
    "latex2sympy2-extended": "1.11.0",
    "math-verify": "0.9.0",
    "matplotlib": "3.10.5",
    "numpy": "2.3.2",
    "scikit-learn": "1.9.1",
    "torch": "2.8.0+cu128",
    "transformers": "5.19.0"
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
    "analyzable_intermediate": 0,
    "normal_completion_rate": 0.0,
    "scorable_rate": 0.0,
    "correct_submission_rate": 0.0,
    "mathematical_accuracy_scorable": null
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
    "analyzable_intermediate": 0,
    "normal_completion_rate": 0.0,
    "scorable_rate": 0.0,
    "correct_submission_rate": 0.0,
    "mathematical_accuracy_scorable": null
  },
  "smoke": {
    "generated": 1,
    "completed": 1,
    "correct": 0,
    "wrong": 1,
    "truncated": 0,
    "unscorable": 0,
    "no_steps": 0,
    "format_error_trajectories": 0,
    "features_missing_or_corrupt": 0,
    "planned": 10,
    "not_generated": 9,
    "analyzable_end": 1,
    "analyzable_intermediate": 1,
    "normal_completion_rate": 0.1,
    "scorable_rate": 0.1,
    "correct_submission_rate": 0.0,
    "mathematical_accuracy_scorable": 0.0
  }
}
```

未完成原因：Backup failed; local files retained, collection stopped: Post "[redacted URL] proxyconnect tcp: dial tcp 127.0.0.1:17897: connect: connection refused


## 自检与备份

```json
{
  "cpu_self_check": {
    "checked_at": "2026-10-14T01:10:26.856308+00:00",
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
      "mixed final content and newline token retained",
      "completed Unicode endpoint ∛",
      "completed Unicode endpoint 𝑥",
      "completed Unicode endpoint ∞",
      "completed Unicode endpoint 汉",
      "completed Unicode endpoint 🧮",
      "each question total weight one",
      "weighted scaler",
      "A/B/C fitting and grouped CV",
      "layer tie chooses smaller index",
      "weighted evaluation and paired bootstrap",
      "single-class bootstrap reports NA",
      "resume skips complete JSON and NPZ without generation",
      "damaged NPZ detected",
      "unrecoverable JSON never triggers regeneration",
      "valid JSON with invalid structure is preserved and rejected",
      "no-step causal check is not a failure"
    ],
    "environment": {
      "executable": "/root/project/VeriServe-step-hidden-probe/environments/qwen35/.venv/bin/python",
      "platform": "Linux-5.15.0-78-generic-x86_64-with-glibc2.35",
      "python": "3.12.3",
      "versions": {
        "antlr4-python3-runtime": "4.13.2",
        "datasets": "4.8.5",
        "huggingface-hub": "1.33.0",
        "latex2sympy2-extended": "1.11.0",
        "math-verify": "0.9.0",
        "matplotlib": "3.10.5",
        "numpy": "2.3.2",
        "scikit-learn": "1.9.1",
        "torch": "2.8.0+cu128",
        "transformers": "5.19.0"
      }
    },
    "note": "Synthetic CPU checks only; actual model causal/GPU and remote LFS checks belong to smoke",
    "passed": true,
    "script_sha256": "f87a600858e08da758706d570a2ae4c7d1fc027d36649bd203225b0aa2c93e08"
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
  "smoke": {
    "completed_parseable_questions": 1,
    "not_applicable": 0,
    "passed": false,
    "questions": {
      "8cbe0c48c29daa190277e30a3c53fe756b4e80c8aa78420c2a5ded9983d59d69": {
        "applicable": true,
        "atol": 0.03125,
        "changed_future": {
          "max_abs": 0.0,
          "max_abs_per_layer": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "passed": true,
          "rms": 0.0
        },
        "completed_parseable": true,
        "completion": "EOS",
        "exclusion": null,
        "extraction": "independent_original_prefixes",
        "independent_prefix": {
          "max_abs": 0.0,
          "max_abs_per_layer": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "passed": true,
          "rms": 0.0
        },
        "passed": true,
        "position": 217,
        "previous_token_negative_control": {
          "detected": true,
          "max_abs": 25.4375,
          "max_abs_per_layer": [
            0.65625,
            0.6484375,
            0.36328125,
            0.5712890625,
            0.83203125,
            0.935546875,
            1.091796875,
            1.59375,
            1.357421875,
            1.466796875,
            1.609375,
            1.6318359375,
            1.884521484375,
            2.049560546875,
            2.269775390625,
            2.58984375,
            2.9140625,
            3.2578125,
            3.35546875,
            4.25,
            4.61328125,
            4.85546875,
            5.32421875,
            5.7265625,
            5.875,
            6.05078125,
            17.375,
            6.7421875,
            7.6875,
            9.4375,
            8.0078125,
            25.4375
          ],
          "passed": false,
          "rms": 0.7003049850463867
        },
        "rtol": 0.008,
        "scorable": true,
        "single_full_changed_future": {
          "max_abs": 0.0,
          "max_abs_per_layer": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "passed": true,
          "rms": 0.0
        },
        "single_full_vs_prefix_diagnostic": {
          "gating": false,
          "max_abs": 0.21875,
          "max_abs_per_layer": [
            0.00048828125,
            0.00048828125,
            0.0078125,
            0.00146484375,
            0.001953125,
            0.001953125,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00439453125,
            0.0078125,
            0.005859375,
            0.005859375,
            0.0078125,
            0.0078125,
            0.0078125,
            0.03125,
            0.03125,
            0.03125,
            0.0625,
            0.0625,
            0.125,
            0.03125,
            0.0625,
            0.0625,
            0.125,
            0.0625,
            0.125,
            0.1875,
            0.125,
            0.21875
          ],
          "note": "Retained full-once BF16 shape discrepancy; features use independent original prefixes",
          "passed": false,
          "rms": 0.010573504492640495,
          "used_for_features": false
        },
        "tolerance_note": "BF16 rounding: atol=1/32, rtol=one BF16 relative ULP; report raw errors"
      }
    },
    "scorable_questions": 1,
    "source_code": {
      "common_sha256": "289a98ce0707d9383b573cf16a48da234a447ef51058f9e92ae7f021dc1c7c98",
      "execution_sources": {
        "stages/stage1/veriserve/__init__.py": "ff898c775de7edd0158af3d21a664a49bf21ef75b0bf0b9f7066b37ba4fd2665",
        "stages/stage1/veriserve/common.py": "289a98ce0707d9383b573cf16a48da234a447ef51058f9e92ae7f021dc1c7c98",
        "stages/stage1/veriserve/data.py": "599745356c71d74f1e96f8de5d9f1f5c30832534b740d61ab44624cdab3a0e4c",
        "stages/stage1/veriserve/model.py": "7dfd702e3be8f6143af9aad8503bfe322e22a70e52f0fa8075afbf8eeabaa5de",
        "stages/stage1/veriserve/prm.py": "df53e83e4d212076e38ec3ea4e8f6aa2fba50098a3dd2d433b4eb2e8c580e1f2",
        "stages/stage1/veriserve/state.py": "50b2ff717e956f4f4878e53b8e6e1fb9d155792fedb4d60738405c0ac7da3365",
        "veriserve_research/__init__.py": "b91a1030c85f6e57d2da048a5683b6456a2580e0d8ce724dec6d30ee4d42a650",
        "veriserve_research/__main__.py": "a4ba058e34feb88f1bb7be96ec23e0eac1dc879fa2a5251c27d974e2faf7a336",
        "veriserve_research/artifacts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "veriserve_research/artifacts/backup.py": "7da2d0989965ca5032ccb9d1a3133a6f0d561ec17540027ec7b86eb35b299981",
        "veriserve_research/artifacts/io.py": "fb70a9f59be25ab9a44c72d380f21897b6895e5c93204bc31516de135dc24acb",
        "veriserve_research/artifacts/legacy.py": "bc98eea48016ea6b70812df6a4f05c0b50981d0415486d654e4ceb66af0f549a",
        "veriserve_research/artifacts/materials.py": "94991c3f2e649b010ced6b6eeda492a35e55e304cf1c891f1289df821f214d0e",
        "veriserve_research/artifacts/provenance.py": "33399d2d5e37dea42b875768f4eab4690e6b927d5a60f940ff71b41dab5b12ea",
        "veriserve_research/inference.py": "b6e7d4efe65bd004ee3797379f3b11290e5048229c896c34fbd74278891ab8e9",
        "veriserve_research/probe/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "veriserve_research/probe/collect.py": "f87a600858e08da758706d570a2ae4c7d1fc027d36649bd203225b0aa2c93e08",
        "veriserve_research/probe/fit.py": "0bfb79008b2178169ba8bdb2ab9cab9a29626bf0faa23a075ff0cdfee9e073ab",
        "veriserve_research/probe/materials.py": "3ba86c6aa63b8f8f5d5848a506027e0dc1e167bcc75c1b2162fb44b35a3991bb",
        "veriserve_research/trajectory.py": "d79b247284fb98488cbdc29fef19928d7443b492c1fd5e8d20d3b6c07c7f0358"
      },
      "script_sha256": "91655842d384255d989ca7401bff22e55117f732e9447e4c87eb6a5232fa30c8"
    },
    "updated_at": "2026-10-14T01:41:59.227795+00:00"
  },
  "resume": null,
  "backup": {
    "error": "Post \"[redacted URL] proxyconnect tcp: dial tcp 127.0.0.1:17897: connect: connection refused\n",
    "status": "FAILED",
    "time": "2026-10-14T01:41:59.831273+00:00"
  }
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
本地 HEAD：`20d2e27b9188964266ba8adf6ce98637553f7e30`；远端备份状态与提交以 backup.json 为准。

从仓库根目录执行（替换成可用 Python 环境）：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```
