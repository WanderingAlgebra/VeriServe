# 第二阶段：逐步 hidden 最终答错风险探针

运行：`v2-20261013-2f15553ff1e5`；报告更新于 2026-10-13T21:29:33.361160+08:00。

标签是完整轨迹最终答错（1）或答对（0）。主指标仅用于完成且可评分轨迹的正常中途步骤；不是局部错误标签。

模型 `Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；tokenizer `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
数据 `KbsdJames/omni-math-rule` / rule，revision `bda09972374225ab2186582615307d909965b864`。

这是采用逐步训练、题目权重、明确 Step 标记和独立测试的适配实验，不宣称完全复现论文。
原生 Thinking；采样与逐题种子见记录；新 token 32768，总上下文 65536。
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
  "smoke": {
    "generated": 10,
    "completed": 5,
    "correct": 0,
    "wrong": 0,
    "truncated": 5,
    "unscorable": 10,
    "no_steps": 10,
    "format_error_trajectories": 10,
    "features_missing_or_corrupt": 5,
    "planned": 10,
    "STEP_FORMAT_ERROR": 10,
    "analyzable_end": 0,
    "analyzable_intermediate": 0
  },
  "formal_generated": 0,
  "formal_planned": 300
}
```

未完成原因：Smoke checks incomplete

## 自检与备份

```json
{
  "cpu_self_check": {
    "checked_at": "2026-10-13T13:29:15.687614+00:00",
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
    "script_sha256": "c84a8bd997df9c957ef7a662dd5df7066b8a5f3d3abd98a50b73403b63cc50c9"
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
    "not_applicable": 10,
    "passed": false,
    "questions": {
      "0f23a157f9d1cbb480ba0278fd79965960a66696d265382d736cc9894730f2d2": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "30908b6528753e36eb83667e2184bc75fd93134d44dca8f3d2a2079342ae822c": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "6113b9f050d676a6858e5fd2c7c96c21aadc51890244efebd6415024d2358355": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "66e9762e872ce0fd44499350967561202d4c7450fb2762d8d4fbee382e857432": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "768eae894c4b077e6491275907b34520b257d7725b86b6364c24392f7dfe50c0": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "8cbe0c48c29daa190277e30a3c53fe756b4e80c8aa78420c2a5ded9983d59d69": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "8d921bb20bae08d9cc071dea699a7fa358f8ceb4d5c04595b7ff7cfa4ea5e51d": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "b06dd6f0e8140ab15e5f660bc6712886b667213459eb3739c23eded89f8c459a": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "da479a3d912813cafd000d6da2242458f8d38e1ce9cb95bf0de0f3b01794aef3": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      },
      "df9af8a321516208b329d6aea7e7e313cda29ccc9e67811a8b20ac10ca2b1b46": {
        "applicable": false,
        "passed": null,
        "reason": "no intermediate position; trajectory retained"
      }
    },
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
        "veriserve_research/inference.py": "f9d73ca6ff97ad4128426e9d47168087386f52abfed4553c0e31d516b60e1e3c",
        "veriserve_research/probe/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "veriserve_research/probe/collect.py": "c84a8bd997df9c957ef7a662dd5df7066b8a5f3d3abd98a50b73403b63cc50c9",
        "veriserve_research/probe/fit.py": "0bfb79008b2178169ba8bdb2ab9cab9a29626bf0faa23a075ff0cdfee9e073ab",
        "veriserve_research/probe/materials.py": "e0be242e549ac57219cc50acdff615f20cb8f00d175cc8b4897e5550db945b73",
        "veriserve_research/trajectory.py": "21b74f7b2d70b3ec1282aa2ac352868a3b485357e7fbde2ee157e1c93a65beed"
      },
      "script_sha256": "de122e3170ec57b4ba1380ededd25f261f242cf1820ce81b7265d6b63439c9da"
    },
    "updated_at": "2026-10-13T13:28:34.299592+00:00"
  },
  "resume": {
    "checked_at": "2026-10-13T13:29:33.319040+00:00",
    "hashes_after_for_existing": {
      "records/smoke/29fb09aef2d2b8192e12.features-de122e3170ec.npz": "2907ba34439550e71eb702dfb44d624b4d292c48c98b3338a84dd7c5bcad9a4c",
      "records/smoke/29fb09aef2d2b8192e12.json": "ea33806896f3e559f5722b50471672c8bdabf4ec5a64966d27259596fedc9d52",
      "records/smoke/8fe8ba3e885ab91a95bb.features-de122e3170ec.npz": "028996a15cd9fc7ca86f7546766bc19024d10a144ce9551d46f7386409bd41fc",
      "records/smoke/8fe8ba3e885ab91a95bb.json": "afd4981b3e24e282047366bcbb10796b5697096d8e024fe88b9e8af286f18bab",
      "records/smoke/9ccccfe8203576afba40.features-de122e3170ec.npz": "3efe72b0ab517153046827f1816717e665ae9a8d3011d551fe33d7103d1788f7",
      "records/smoke/9ccccfe8203576afba40.json": "57171a5361ebe250bb51dde19088ed116cca8227b253c0464111de517afeed0d",
      "records/smoke/eca310ad5d6fad19fe38.features-de122e3170ec.npz": "f11a37265772777054ac98c49ab7541f4dd3aa27cff13f6d5a621f6b950c4e4d",
      "records/smoke/eca310ad5d6fad19fe38.json": "7a6d24a6736cf25918fe24a1a0942f6943e90c4a5eb94ab3af92ea90aaf452b9",
      "records/smoke/f45649608d5fd4570df0.features-de122e3170ec.npz": "ed99e929224e0ea733acd8aab3e0fffdc2600f70935cf3a3cbbe397ff5a027a2",
      "records/smoke/f45649608d5fd4570df0.json": "5300cb7640503a58ea64330e895a5f4fca86e0cccca81d7c0696d8fffa8468f2"
    },
    "hashes_before": {
      "records/smoke/29fb09aef2d2b8192e12.features-de122e3170ec.npz": "2907ba34439550e71eb702dfb44d624b4d292c48c98b3338a84dd7c5bcad9a4c",
      "records/smoke/29fb09aef2d2b8192e12.json": "ea33806896f3e559f5722b50471672c8bdabf4ec5a64966d27259596fedc9d52",
      "records/smoke/8fe8ba3e885ab91a95bb.features-de122e3170ec.npz": "028996a15cd9fc7ca86f7546766bc19024d10a144ce9551d46f7386409bd41fc",
      "records/smoke/8fe8ba3e885ab91a95bb.json": "afd4981b3e24e282047366bcbb10796b5697096d8e024fe88b9e8af286f18bab",
      "records/smoke/9ccccfe8203576afba40.features-de122e3170ec.npz": "3efe72b0ab517153046827f1816717e665ae9a8d3011d551fe33d7103d1788f7",
      "records/smoke/9ccccfe8203576afba40.json": "57171a5361ebe250bb51dde19088ed116cca8227b253c0464111de517afeed0d",
      "records/smoke/eca310ad5d6fad19fe38.features-de122e3170ec.npz": "f11a37265772777054ac98c49ab7541f4dd3aa27cff13f6d5a621f6b950c4e4d",
      "records/smoke/eca310ad5d6fad19fe38.json": "7a6d24a6736cf25918fe24a1a0942f6943e90c4a5eb94ab3af92ea90aaf452b9",
      "records/smoke/f45649608d5fd4570df0.features-de122e3170ec.npz": "ed99e929224e0ea733acd8aab3e0fffdc2600f70935cf3a3cbbe397ff5a027a2",
      "records/smoke/f45649608d5fd4570df0.json": "5300cb7640503a58ea64330e895a5f4fca86e0cccca81d7c0696d8fffa8468f2"
    },
    "passed": true,
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
        "veriserve_research/inference.py": "f9d73ca6ff97ad4128426e9d47168087386f52abfed4553c0e31d516b60e1e3c",
        "veriserve_research/probe/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "veriserve_research/probe/collect.py": "c84a8bd997df9c957ef7a662dd5df7066b8a5f3d3abd98a50b73403b63cc50c9",
        "veriserve_research/probe/fit.py": "0bfb79008b2178169ba8bdb2ab9cab9a29626bf0faa23a075ff0cdfee9e073ab",
        "veriserve_research/probe/materials.py": "e0be242e549ac57219cc50acdff615f20cb8f00d175cc8b4897e5550db945b73",
        "veriserve_research/trajectory.py": "21b74f7b2d70b3ec1282aa2ac352868a3b485357e7fbde2ee157e1c93a65beed"
      },
      "script_sha256": "de122e3170ec57b4ba1380ededd25f261f242cf1820ce81b7265d6b63439c9da"
    },
    "unchanged_files": 10
  },
  "backup": {
    "commit": "d9791b7e9fc2bddf494064ad4b971a3ab721374d",
    "feature_remote_verification": {
      "path": "stages/stage2/runs/v2-20261013-2f15553ff1e5/records/smoke/29fb09aef2d2b8192e12.features-de122e3170ec.npz",
      "remote_commit": "9239e3a8e1faca29435f4ac3d4ce70a1eac9eccd",
      "sha256": "2907ba34439550e71eb702dfb44d624b4d292c48c98b3338a84dd7c5bcad9a4c",
      "verified_at": "2026-10-13T13:29:10.847709+00:00"
    },
    "pushed_at": "2026-10-13T13:29:26.408373+00:00",
    "status": "PUSHED"
  }
}
```

AUROC / CI / 所选层：NA；正式数据不足或未采集，不能回答是否存在信号。

这项实验不能识别首个错误步骤、判断当前检查是否值得、证明降低完成时间或推广到其他模型。
离线重前向耗时不代表在线提取成本；生成、重前向与探针评分耗时分别记录。

步骤端点对齐原始 token，跨界 token 向前对齐；包含 boxed 或明确最终答案的步骤排除，最后正常步骤保留 near_end 标记。

评分仅解析单独 Final answer 行的最后完整 boxed；Math-Verify 标准答案包装为 $...$，无字符串 fallback，解析/比较超时单独排除。

## 文件与恢复

配置与环境见 config.json / manifest.json；运行日志见 events.jsonl；探针参数见 probe_{A,B,C}.npz/json。
分支：[experiment/step-hidden-probe](https://github.com/WanderingAlgebra/VeriServe/tree/experiment/step-hidden-probe)。
本地 HEAD：`d9791b7e9fc2bddf494064ad4b971a3ab721374d`；远端备份状态与提交以 backup.json 为准。

从仓库根目录执行（替换成可用 Python 环境）：

```bash
python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check
python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --resume
python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume
```
