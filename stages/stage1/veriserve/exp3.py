from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

from .common import atomic_json, read_json, stable_hash
from .data import is_correct
from .model import LanguageModel, cuda_now, feedback_text
from .state import PathState

if TYPE_CHECKING:
    from .runner import RunManager


def select_snapshots(run: "RunManager") -> list[dict[str, Any]]:
    candidates = []
    for row in run.splits["evaluation"]:
        raw = read_json(run.state_path("exp2b_eval", "k2", row["id"]))
        if raw is None:
            raise RuntimeError("Experiment 2 evaluation paths are incomplete")
        state = PathState.from_dict(raw)
        if state.fail_snapshots:
            candidates.append({"question_id": row["id"], "snapshot": state.fail_snapshots[0]})
    candidates.sort(key=lambda item: item["question_id"])
    cap = int(run.config["experiment3"]["max_fail_snapshots"])
    initial = [item for item in candidates if len(item["snapshot"]["state"]["checkpoint_ids"]) ==
               len(item["snapshot"]["state"]["prompt_ids"])]
    nonempty = [item for item in candidates if len(item["snapshot"]["state"]["checkpoint_ids"]) >
                len(item["snapshot"]["state"]["prompt_ids"])]
    selected = initial[:cap // 2] + nonempty[:cap - cap // 2]
    selected_ids = {item["question_id"] for item in selected}
    selected += [item for item in candidates if item["question_id"] not in selected_ids][
        :cap - len(selected)]
    selected.sort(key=lambda item: item["question_id"])
    atomic_json(run.root / "exp3" / "selected_snapshots.json", [
        {"question_id": item["question_id"],
         "checkpoint_tokens": len(item["snapshot"]["state"]["checkpoint_ids"]),
         "nonempty_checkpoint": len(item["snapshot"]["state"]["checkpoint_ids"]) >
                                len(item["snapshot"]["state"]["prompt_ids"])}
        for item in selected])
    return selected


def _branch(run: "RunManager", item: dict[str, Any], arm: str) -> PathState:
    original = item["snapshot"]["state"]
    state = PathState.from_dict(copy.deepcopy(original))
    state.request_id = item["question_id"] + "-" + arm
    state.history_id = stable_hash((state.history_id, arm))[:16]
    diagnosis = item["snapshot"]["diagnosis"]
    hint = item["snapshot"]["hint"]
    if arm == "B":
        feedback_text_value = feedback_text(diagnosis, hint, state.checkpoint_step)
    else:
        feedback_text_value = ("\n[Verification feedback]\n" + diagnosis.strip() + "\n" + hint.strip()
                               + "\n[Continue]\nDiscard the previous attempt and restart from Step 1.\n")
    feedback = run.tokenizer.encode(feedback_text_value, add_special_tokens=False)
    step = state.current_step
    state.apply_verdict("FAIL", item["snapshot"]["first_error_step"],
                        diagnosis, hint, feedback, step)
    if arm == "C":
        state.checkpoint_ids = state.prompt_ids.copy()
        state.checkpoint_id = "restart_from_prompt"
        state.checkpoint_step = 0
        state.context_ids = state.prompt_ids.copy() + feedback
        state.segment_start = len(state.context_ids)
        state.pending_ids = []
        state.record("restart_from_prompt", retained_steps=0)
    return state


def _paired_paths(run: "RunManager", selected: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[Path, PathState] = {}
    base_costs = {}
    for item in selected:
        qid = item["question_id"]
        original = item["snapshot"]["state"]
        base_costs[qid] = (original["generator_seconds"], original["verifier_seconds"])
        for arm in ("B", "C"):
            path = run.state_path("exp3", arm, qid)
            raw = read_json(path) if run.resume else None
            states[path] = PathState.from_dict(raw) if raw else _branch(run, item, arm)
            if raw is None:
                run._save_state(path, states[path])
    if states:
        run.run_paths(states)
        run._collect_logs("exp3", states)
    pairs = []
    for item in selected:
        qid = item["question_id"]
        b = states[run.state_path("exp3", "B", qid)]
        c = states[run.state_path("exp3", "C", qid)]
        pre_gen, pre_ver = base_costs[qid]
        pairs.append({
            "question_id": qid,
            "nonempty_checkpoint": len(item["snapshot"]["state"]["checkpoint_ids"]) >
                                   len(item["snapshot"]["state"]["prompt_ids"]),
            "B_correct": is_correct(b.final_answer, b.gold),
            "C_correct": is_correct(c.final_answer, c.gold),
            "B_termination": b.termination, "C_termination": c.termination,
            "B_suffix_generator_seconds": b.generator_seconds - pre_gen,
            "C_suffix_generator_seconds": c.generator_seconds - pre_gen,
            "B_suffix_verifier_seconds": b.verifier_seconds - pre_ver,
            "C_suffix_verifier_seconds": c.verifier_seconds - pre_ver,
            "B_generated_tokens": b.generated_tokens - item["snapshot"]["state"]["generated_tokens"],
            "C_generated_tokens": c.generated_tokens - item["snapshot"]["state"]["generated_tokens"],
        })
    result = {
        "pairs": pairs,
        "count": len(pairs),
        "nonempty_checkpoint_count": sum(p["nonempty_checkpoint"] for p in pairs),
        "B_correct": sum(p["B_correct"] for p in pairs),
        "C_correct": sum(p["C_correct"] for p in pairs),
        "B_only_correct": sum(p["B_correct"] and not p["C_correct"] for p in pairs),
        "C_only_correct": sum(p["C_correct"] and not p["B_correct"] for p in pairs),
    }
    atomic_json(run.root / "exp3" / "bc_pairs.json", result)
    return result


def _append_feedback(model: LanguageModel, cache: Any, feedback_ids: list[int]) -> torch.Tensor:
    output = model.model(input_ids=torch.tensor([feedback_ids], device=model.device),
                         past_key_values=cache, use_cache=True)
    return output.logits[:, -1, :]


def _greedy(model: LanguageModel, cache: Any, logits: torch.Tensor, count: int) -> list[int]:
    tokens = []
    for _ in range(count):
        token = int(torch.argmax(logits, dim=-1).item())
        tokens.append(token)
        if token == model.tokenizer.eos_token_id:
            break
        logits = model._next_logits(token, cache)
    return tokens


@torch.inference_mode()
def _consistency_case(model: LanguageModel, snapshot: dict[str, Any],
                      count: int, tolerance: float) -> dict[str, Any]:
    raw = snapshot["state"]
    prefix = raw["checkpoint_ids"]
    rejected = raw["context_ids"][len(prefix):]
    if not rejected:
        return {"passed": False, "error": "No rejected suffix to crop"}
    feedback = model.text_ids(feedback_text(snapshot["diagnosis"], snapshot["hint"],
                                             raw["checkpoint_step"]))
    cache_a, _, rebuild_seconds = model._prefill(prefix + rejected)
    before_reserved = int(torch.cuda.memory_reserved())
    start = cuda_now()
    cache_a.crop(-len(rejected))
    cropped_length = cache_a.get_seq_length()
    logits_a = _append_feedback(model, cache_a, feedback)
    restore_seconds = cuda_now() - start
    after_reserved = int(torch.cuda.memory_reserved())
    cache_b, logits_b, recompute_seconds = model._prefill(prefix + feedback)
    difference = float((logits_a.float() - logits_b.float()).abs().max().item())
    top_a = int(torch.argmax(logits_a, dim=-1).item())
    top_b = int(torch.argmax(logits_b, dim=-1).item())
    greedy_a = _greedy(model, cache_a, logits_a, count)
    greedy_b = _greedy(model, cache_b, logits_b, count)
    return {
        "passed": difference <= tolerance and top_a == top_b and greedy_a == greedy_b
                  and cropped_length == len(prefix),
        "prefix_tokens": len(prefix), "rejected_tokens": len(rejected),
        "feedback_tokens": len(feedback), "cropped_length": cropped_length,
        "logit_max_abs_difference": difference, "next_token_a": top_a,
        "next_token_b": top_b, "greedy_equal": greedy_a == greedy_b,
        "offline_rebuild_seconds": rebuild_seconds,
        "a_restore_feedback_seconds": restore_seconds,
        "b_recompute_seconds": recompute_seconds,
        "reserved_before_crop": before_reserved,
        "reserved_after_crop": after_reserved,
    }


@torch.inference_mode()
def _benchmark_case(model: LanguageModel, prefix: list[int], feedback: list[int],
                    greedy_tokens: int) -> dict[str, float]:
    cache_a, logits_a, offline_rebuild = model._prefill(prefix)
    start = cuda_now()
    logits_a = _append_feedback(model, cache_a, feedback)
    a_feedback = cuda_now() - start
    start = cuda_now()
    _greedy(model, cache_a, logits_a, greedy_tokens)
    a_decode = cuda_now() - start
    cache_b, logits_b, b_recompute = model._prefill(prefix + feedback)
    start = cuda_now()
    _greedy(model, cache_b, logits_b, greedy_tokens)
    b_decode = cuda_now() - start
    return {"offline_rebuild_seconds": offline_rebuild,
            "a_feedback_seconds": a_feedback, "a_decode_seconds": a_decode,
            "b_recompute_seconds": b_recompute, "b_decode_seconds": b_decode}


def _lengthen(prefix: list[int], filler: list[int], length: int) -> list[int]:
    if len(prefix) >= length:
        return prefix[:length]
    return (prefix + (filler * ((length - len(prefix)) // len(filler) + 1)))[:length]


def _cache_study(run: "RunManager", selected: list[dict[str, Any]]) -> dict[str, Any]:
    directory = run.root / "exp3"
    if not selected:
        result = {"status": "INCONCLUSIVE", "reason": "No natural FAIL snapshots"}
        atomic_json(directory / "cache_consistency.json", result)
        return result
    model = LanguageModel(run.assets["generator"]["path"])
    try:
        natural = sorted(selected, key=lambda x: (
            len(x["snapshot"]["state"]["checkpoint_ids"]) <=
            len(x["snapshot"]["state"]["prompt_ids"]), x["question_id"]))
        natural = natural[:int(run.config["experiment3"]["cache_snapshots"])]
        cases = []
        for item in natural:
            path = directory / "cache_cases" / f"{item['question_id']}.json"
            existing = read_json(path) if run.resume else None
            if existing is None:
                try:
                    existing = _consistency_case(
                        model, item["snapshot"],
                        int(run.config["experiment3"]["greedy_tokens"]),
                        float(run.config["experiment3"]["logit_max_abs_tolerance"]))
                except torch.cuda.OutOfMemoryError as exc:
                    existing = {"passed": False, "error": "OOM", "detail": str(exc)}
                atomic_json(path, existing)
            cases.append({"question_id": item["question_id"], **existing})
        consistency_status = ("INCONCLUSIVE" if len(cases) < 10 else
                              ("PASS" if all(c["passed"] for c in cases) else "BLOCKED"))
        consistency = {"status": consistency_status, "cases": cases,
                       "passed": sum(c["passed"] for c in cases), "total": len(cases)}
        atomic_json(directory / "cache_consistency.json", consistency)

        filler = model.text_ids(" Step 1: calculate a small arithmetic expression.\n")
        feedback_filler = model.text_ids(" Recalculate the previous arithmetic carefully. ")
        base_prefix = natural[0]["snapshot"]["state"]["checkpoint_ids"]
        profile = read_json(run.root / "hardware_profile.json", {})
        expanded = profile.get("roles", {}).get("generator", {}).get("lengths", {}).get("3072", {})
        benchmark_limit = (int(run.config["expanded_context_limit"])
                           if expanded.get("status") == "PASS" else int(run.config["context_limit"]))
        benchmarks = []
        repeats = int(run.config["experiment3"]["benchmark_repetitions"])
        for prefix_len in (256, 512, 1024, 2048):
            for feedback_len in (64, 128, 256):
                if prefix_len + feedback_len + 32 > benchmark_limit:
                    benchmarks.append({"prefix_tokens": prefix_len,
                                       "feedback_tokens": feedback_len,
                                       "status": "SKIPPED_CONTEXT"})
                    continue
                prefix = _lengthen(base_prefix, filler, prefix_len)
                feedback = _lengthen([], feedback_filler, feedback_len)
                values = []
                for _ in range(repeats + 1):
                    value = _benchmark_case(model, prefix, feedback,
                                            int(run.config["experiment3"]["greedy_tokens"]))
                    if values or _ == 1:
                        values.append(value)
                benchmarks.append({"prefix_tokens": len(prefix),
                                   "feedback_tokens": len(feedback),
                                   "status": "MEASURED", "repetitions": values,
                                   "medians": {key: statistics.median(v[key] for v in values)
                                               for key in values[0]}})
        atomic_json(directory / "cache_benchmark.json", benchmarks)
        plotted = [item for item in benchmarks if item["status"] == "MEASURED"
                   and item["feedback_tokens"] == 128]
        if plotted:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([x["prefix_tokens"] for x in plotted],
                    [x["medians"]["a_feedback_seconds"] for x in plotted], "o-", label="A cached feedback")
            ax.plot([x["prefix_tokens"] for x in plotted],
                    [x["medians"]["b_recompute_seconds"] for x in plotted], "o-", label="B full prefill")
            ax.set(xlabel="Prefix tokens", ylabel="Seconds", title="Feedback arrival cost (128 tokens)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory / "prefix_latency.png", dpi=160)
            plt.close(fig)
        return {"consistency": consistency, "benchmarks": benchmarks}
    finally:
        model.close()


def run_experiment3(run: "RunManager") -> dict[str, Any]:
    directory = run.root / "exp3"
    directory.mkdir(exist_ok=True)
    selected = select_snapshots(run)
    bc = _paired_paths(run, selected)
    cache = _cache_study(run, selected)
    status = "PASS"
    if not selected or bc["nonempty_checkpoint_count"] < 5:
        status = "INCONCLUSIVE"
    if cache.get("consistency", {}).get("status") == "BLOCKED":
        status = "BLOCKED_A"
    result = {"status": status, "bc": bc, "cache_status": cache.get("consistency", {}).get("status")}
    atomic_json(directory / "summary.json", result)
    report = ["# 实验三：历史检查点与 KV 复用", "", f"工程状态：{status}", "",
              f"自然 FAIL 快照：{bc['count']}；其中非空历史检查点：{bc['nonempty_checkpoint_count']}。",
              f"B 答对 {bc['B_correct']}；C 答对 {bc['C_correct']}。",
              f"A/B 数值一致性：{cache.get('consistency', {}).get('passed', 0)}/"
              f"{cache.get('consistency', {}).get('total', 0)}。", "",
              "B/C 是发生 FAIL 后的条件结果；总体收益以实验二的全题结果为准。"]
    (directory / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
