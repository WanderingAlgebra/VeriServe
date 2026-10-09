"""Whole-request stage-one A/B comparison; preserve the historical 9/10 result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from stages.stage1.veriserve.common import atomic_json, load_config, read_json, stable_hash
from stages.stage1.veriserve.data import gold_answer, is_correct
from stages.stage1.veriserve.model import LanguageModel, cuda_now, feedback_text, parse_verdict
from stages.stage1.veriserve.prm import ProcessRewardModel
from stages.stage1.veriserve.runner import request_seed
from stages.stage1.veriserve.state import Budget, PathState
from veriserve_research.inference import Session

ROOT = Path(__file__).resolve().parents[3]
GENERATOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VERIFIER_REVISION = "0610740060112df12585d00a1c5f4624d2f59051"
DATA_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SOURCE = ROOT / "stages/stage1/results/diagnostics/exp4_further_20260927/a_paths.csv"


class ResidentGenerator(LanguageModel):
    """Reuse stage-one sampling/parsing; only FAIL recovery differs between arms."""

    def __init__(self, model, arm):
        if arm not in ("A", "B"):
            raise ValueError(arm)
        self.model, self.tokenizer, self.device = model.model, model.tokenizer, model.device
        self.arm, self.session, self.restore = arm, None, None
        self.actual_forward_tokens = 0
        self.recoveries = []

    def _prefill(self, ids):
        start = cuda_now()
        if self.restore is not None:
            checkpoint, feedback = self.restore
            if ids != checkpoint + feedback:
                raise RuntimeError("Recovery token prefix changed")
            before = self.session.cache.get_seq_length()
            if self.arm == "A":
                charged = self.session.rollback(checkpoint, feedback)
            else:
                self.session = Session(self.model, ids)
                charged = len(ids)
            self.recoveries.append({"cache_before": before, "checkpoint_tokens": len(checkpoint),
                                    "feedback_tokens": len(feedback), "forward_tokens": charged})
            self.restore = None
        elif self.session is None:
            self.session = Session(self.model, ids)
            charged = len(ids)
        else:
            if self.session.ids != ids:
                raise RuntimeError("PASS/pending token context changed")
            charged = len(ids) - self.session.cache.get_seq_length()
            self.session.ensure()
        self.actual_forward_tokens += charged
        return self.session.cache, self.session.logits[None], cuda_now() - start

    def _next_logits(self, token, cache):
        if cache is not self.session.cache:
            raise RuntimeError("Generation used a different KV cache")
        self.session.ids.append(token)
        self.session.forward([token])
        self.actual_forward_tokens += 1
        return self.session.logits[None]

    def generate_path(self, state, config):
        result = super().generate_path(state, config)
        sampled = state.context_ids + result.kept_ids + result.pending_ids
        if sampled[:len(self.session.ids)] != self.session.ids:
            raise RuntimeError("Sampled tokens differ from the cached prefix")
        # The final sampled token need not yet be in KV; Session.ensure/rollback handles it.
        self.session.ids = sampled
        return result


def budget(config):
    return Budget(**config["budget"], context_tokens=config["context_limit"])


def gpu_check(model, run):
    prefix = model.generator_prompt("Compute 1+1.") + model.text_ids("Step 1: 1+1=2.\n\n")
    rejected = model.text_ids("Step 2: Incorrect continuation.\n")
    feedback = model.text_ids("\nCheck the arithmetic and continue.\n")
    a, b = ResidentGenerator(model, "A"), ResidentGenerator(model, "B")
    for generator in (a, b):
        generator._prefill(prefix + rejected)
        generator.session.ids += model.text_ids("\n")  # pending token absent from KV
    kept = [(layer.keys[..., :len(prefix), :].clone(), layer.values[..., :len(prefix), :].clone())
            for layer in a.session.cache.layers]
    for generator in (a, b):
        generator.restore = (prefix, feedback)
        generator._prefill(prefix + feedback)
        if generator.session.cache.get_seq_length() != len(prefix + feedback):
            raise RuntimeError("GPU recovery cache length mismatch")
        if not torch.isfinite(generator.session.logits).all():
            raise RuntimeError("GPU recovery produced non-finite logits")
    for layer, (keys, values) in zip(a.session.cache.layers, kept):
        if not torch.equal(layer.keys[..., :len(prefix), :], keys) or not torch.equal(layer.values[..., :len(prefix), :], values):
            raise RuntimeError("GPU crop changed retained cache tensors")
    result = {"status": "PASS", "kept_KV_exact": True, "prefix_tokens": len(prefix),
              "feedback_tokens": len(feedback), "A": a.recoveries, "B": b.recoveries,
              "logit_max_abs_difference": float((a.session.logits.float() - b.session.logits.float()).abs().max()),
              "note": "Real forced recovery outside scientific requests; BF16 logit equality is diagnostic only"}
    atomic_json(run / "gpu_check.json", result)
    del a, b, kept
    return result


def run_request(row, arm, model, verifier, config, identity):
    generator = ResidentGenerator(model, arm)
    state = PathState.create(row["id"], row["question"], row["gold"], "k2", row["seed"],
                             model.generator_prompt(row["question"]), budget(config), identity)
    start = cuda_now()
    while not state.finished:
        if state.status == "WAIT_GENERATION":
            if not state.can_generate():
                break
            logical = not state.checks or state.checks[-1]["effective_verdict"] == "FAIL"
            input_tokens = len(state.context_ids) + len(state.pending_ids)
            if logical and state.prefill_tokens + input_tokens > state.budget.prefill_tokens:
                state.finish("BUDGET_PREFILL", state.candidate)
                break
            result = generator.generate_path(state, config)
            state.rng_state = result.rng_state
            state.accept_generated(result.kept_ids, result.pending_ids, result.reason, result.answer,
                                   input_tokens, result.prefill_seconds, result.decode_seconds,
                                   current_step=result.current_step)
        if state.finished:
            break
        accepted = model.decode(state.checkpoint_ids[len(state.prompt_ids):])
        new = model.decode(state.context_ids[state.segment_start:])
        ids, _, _ = verifier.input_ids(state.question, accepted, new)
        if len(ids) > state.budget.context_tokens or state.prefill_tokens + len(ids) > state.budget.prefill_tokens:
            state.finish("BUDGET_PREFILL_OR_CONTEXT", state.candidate)
            break
        raw, inputs, outputs, seconds = verifier.verify(state.question, accepted, new)
        state.charge_verification(inputs, outputs, seconds, raw)
        parsed = parse_verdict(raw)
        if parsed is None:
            raise RuntimeError("Deterministic PRM adapter produced invalid JSON")
        feedback = model.text_ids(feedback_text(parsed["diagnosis"], parsed["hint"], state.checkpoint_step)) \
            if parsed["verdict"] == "FAIL" else []
        effective = state.apply_verdict(parsed["verdict"], parsed["first_error_step"],
                                        parsed["diagnosis"], parsed["hint"], feedback, state.current_step)
        if effective == "FAIL" and not state.finished:
            generator.restore = (state.checkpoint_ids.copy(), feedback)
    elapsed = cuda_now() - start
    result = {"arm": arm, "question_id": row["id"], "seed": row["seed"],
              "T_request": elapsed, "correct": is_correct(state.final_answer, state.gold),
              "actual_forward_tokens": generator.actual_forward_tokens,
              "recoveries": generator.recoveries, "state": state.to_dict()}
    del generator
    return result


def prepare(config_path, output):
    import pyarrow.parquet as pq

    config = load_config(config_path)
    csv_rows = list(csv.DictReader(SOURCE.open()))
    selected = [row for row in csv_rows if row["policy"] == "unchecked"]
    if len(selected) != 100 or len({row["question_id"] for row in selected}) != 100:
        raise RuntimeError("Expected exactly the original 100 evaluation IDs")
    data_path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                                repo_type="dataset", revision=DATA_REVISION)
    data = pq.read_table(data_path).to_pylist()
    rows = []
    for item in selected:
        qid = item["question_id"]
        seed = request_seed(config["seed"], qid)
        if seed != int(item["seed"]):
            raise RuntimeError("Archived request seed differs")
        row = data[int(qid.removeprefix("test-"))]
        rows.append({"id": qid, "seed": seed, "question": row["question"], "gold": gold_answer(row["answer"])})
    sources = [Path(__file__), ROOT / "veriserve_research/inference.py"] + list(
        (ROOT / "stages/stage1/veriserve").glob("*.py"))
    design = {"experiment": "stage1_whole_request_cache_A_vs_recompute_B", "config": config,
              "generator_revision": GENERATOR_REVISION, "verifier_revision": VERIFIER_REVISION,
              "dataset_revision": DATA_REVISION, "rows": rows,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources},
              "question_id_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "bitsandbytes", "accelerate")},
              "python": sys.version, "cuda": torch.version.cuda,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "gpu": str(torch.cuda.get_device_properties(0)),
              "sampling": "Original stage1 temperature/top_p and identical archived per-question seeds",
              "scope": "Fresh full requests on archived 100 IDs; historical 38 FAIL snapshots unavailable",
              "data_provenance_limit": "Original dataset revision/row bytes unavailable; archived IDs are read from the explicitly pinned dataset revision above",
              "recovery": "Both keep KV across PASS/UNCERTAIN; A crops KV on FAIL, B recomputes prefix on FAIL",
              "budget_accounting": "Same original logical prefill budget in both arms; actual forward tokens also recorded",
              "timing": "Synchronized prompt prefill through termination; generation/PRM/control included; model loading, warmup, disk writes and gold scoring excluded",
              "numerical_policy": "User authorizes proceeding with historical 9/10 and diagnosed BF16 divergence; original BLOCKED_A unchanged; token/cache structural errors still stop",
              "bootstrap": 1000}
    identity = stable_hash(design)
    run = output / identity[:12]
    run.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / "manifest.json")
    if manifest is not None and manifest != design:
        raise RuntimeError("Run identity mismatch")
    atomic_json(run / "manifest.json", design)
    return config, rows, run, identity


def summarize(run, rows, config):
    pairs = []
    for row in rows:
        path = run / "pairs" / f"{row['id']}.json"
        if path.exists():
            pairs.append(read_json(path))
    summary = {"planned": len(rows), "complete": len(pairs), "arms": {}, "effects": {}}
    for arm in ("A", "B"):
        values = [pair[arm] for pair in pairs]
        summary["arms"][arm] = {"correct": sum(x["correct"] for x in values),
                                 "mean_T_request": float(np.mean([x["T_request"] for x in values])) if values else None,
                                 "mean_actual_forward_tokens": float(np.mean([x["actual_forward_tokens"] for x in values])) if values else None,
                                 "rollbacks": sum(len(x["recoveries"]) for x in values),
                                 "terminations": dict(Counter(x["state"]["termination"] for x in values))}
    for variable in ("accuracy_A_minus_B", "seconds_B_minus_A"):
        values = np.array([int(pair["A"]["correct"]) - int(pair["B"]["correct"]) if variable.startswith("accuracy")
                           else pair["B"]["T_request"] - pair["A"]["T_request"] for pair in pairs])
        if len(values):
            rng = np.random.default_rng(config["seed"])
            samples = values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
            summary["effects"][variable] = {"mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}
    atomic_json(run / "metrics.json", summary)
    lines = ["# Stage1 A/B 完整请求对照", "", f"已完成 {len(pairs)}/{len(rows)} 对；{'COMPLETE' if len(pairs) == len(rows) else 'INCOMPLETE'}。",
             "", "A 在 FAIL 时裁剪并复用 KV；B 在 FAIL 时重算相同前缀。两者均从题目开始，按 k2 检查，PASS/UNCERTAIN 后保留缓存继续生成。",
             "", "使用原 100 道正式题 ID 和请求种子，重新生成独立材料；不复用缺失的旧 FAIL 快照。模型、数据版本和实际环境见 manifest.json。",
             "历史 9/10 与 BLOCKED_A 保留；经用户授权，已诊断的 BF16 数值分叉不作为本轮停止条件。", "",
             "T_request 为同步后的完整请求墙钟时间，含生成、PRM、恢复和控制；不含模型加载、预热、写盘和标准答案评分。",
             "逻辑预算沿用 stage1；实际前向 token 数单独记录。原 4060 耗时与本机耗时不直接比较。", "",
             "| 路径 | 答对 | 平均请求秒 | 平均实际前向 token |", "| --- | ---: | ---: | ---: |"]
    for arm, values in summary["arms"].items():
        if pairs:
            lines.append(f"| {arm} | {values['correct']}/{len(pairs)} | {values['mean_T_request']:.4f} | {values['mean_actual_forward_tokens']:.1f} |")
    lines += ["", "配对差值的正值分别表示 A 更准 / 更快；95% 区间按题重抽样 1000 次。"]
    for name, effect in summary["effects"].items():
        lines.append(f"- {name}: {effect['mean']:.6f}, 95% CI {effect['ci95']}")
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "stages/stage1/configs/4060_prm.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "stages/stage1/results/exp3_ab")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    config, rows, run, identity = prepare(args.config, args.output)
    print(f"RUN {run}", flush=True)
    summarize(run, rows, config)
    if args.prepare_only:
        return
    generator_path = snapshot_download(config["generator"]["id"], revision=GENERATOR_REVISION,
                                       allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    verifier_path = snapshot_download(config["verifier"]["id"], revision=VERIFIER_REVISION, local_files_only=True)
    model = LanguageModel(generator_path)
    verifier = ProcessRewardModel(verifier_path, config["verifier"]["threshold"])
    try:
        # Warm both identical execution arms and the real verifier outside request timing.
        for arm in ("A", "B"):
            warm = ResidentGenerator(model, arm)
            cache, logits, _ = warm._prefill(model.generator_prompt("Compute 1+1."))
            rng = torch.Generator(device=model.device).manual_seed(config["seed"])
            token = warm._sample(logits, rng, config["generation"]["temperature"], config["generation"]["top_p"])
            warm._next_logits(token, cache)
            del warm
        verifier.verify("Compute 1+1.", "", "Step 1: 1+1=2.\n\nFinal answer: 2.\n")
        gpu_check(model, run)
        print("GPU cache crop/recompute check PASS; BF16 equality is not a gate", flush=True)
        for index, row in enumerate(rows):
            path = run / "pairs" / f"{row['id']}.json"
            if path.exists():
                saved = read_json(path)
                if saved.get("identity") != identity or any(saved[a]["seed"] != row["seed"] for a in ("A", "B")):
                    raise RuntimeError("Saved pair provenance mismatch")
                if args.stop_after and index + 1 >= args.stop_after:
                    break
                continue
            pair = {"identity": identity, "order": ["A", "B"] if index % 2 == 0 else ["B", "A"]}
            for arm in pair["order"]:
                pair[arm] = run_request(row, arm, model, verifier, config, identity)
            failures = [pair[arm]["state"]["fail_snapshots"] for arm in ("A", "B")]
            if bool(failures[0]) != bool(failures[1]):
                raise RuntimeError("Identical pre-recovery requests disagreed on the first FAIL")
            if failures[0] and any(failures[0][0]["state"][key] != failures[1][0]["state"][key]
                                   for key in ("context_ids", "checkpoint_ids", "pending_ids", "rng_state")):
                raise RuntimeError("Paired histories differ before the first FAIL")
            atomic_json(path, pair)
            summary = summarize(run, rows, config)
            print(f"PAIRS {summary['complete']}/{len(rows)} A={summary['arms']['A']['correct']} B={summary['arms']['B']['correct']} "
                  f"seconds A={pair['A']['T_request']:.3f} B={pair['B']['T_request']:.3f}", flush=True)
            if args.stop_after and summary["complete"] >= args.stop_after:
                break
    finally:
        verifier.close()
        model.close()


if __name__ == "__main__":
    main()
