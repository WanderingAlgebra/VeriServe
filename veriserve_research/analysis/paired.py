"""Question-paired bootstrap statistics and formatting shared by reports."""
from __future__ import annotations
import numpy as np
from ..artifacts.materials import finite as finite, group as group, exclusion as exclusion, backend as backend










def interval(point, samples, n, reason=None):
    return {"estimate": point, "ci95": np.quantile(samples, [.025, .975]).tolist() if samples else None,
            "bootstrap_valid": len(samples), "bootstrap_skipped": n - len(samples),
            "reason": reason if not samples else None}


def paired_effect(rows, variable, cfg):
    valid = [r for r in rows if r["quality_valid" if variable == "Y" else "time_valid"]]
    n = cfg.get("bootstrap", 1000)
    if not valid:
        return interval(None, [], n, "NO_VALID_PAIRED_QUALITY" if variable == "Y" else "NO_VALID_PAIRED_TIME")
    if variable == "T" and len({r["backend"] for r in valid}) > 1:
        return interval(None, [], n, "MIXED_BACKENDS_SEE_BY_BACKEND")
    values = np.asarray([r["arms"]["NOW"]["Y"] - r["arms"]["DELAY_2"]["Y"] if variable == "Y"
                         else r["arms"]["DELAY_2"]["timing"]["T_postfork_wall"]
                         - r["arms"]["NOW"]["timing"]["T_postfork_wall"] for r in valid], dtype=float)
    rng = np.random.default_rng(cfg["seed"])
    samples = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(n)]
    result = interval(float(values.mean()), samples, n)
    result["paired_questions"] = len(valid)
    return result


def distribution(values):
    values = [float(v) for v in values if finite(v)]
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95_descriptive": float(np.quantile(values, .95)) if values else None,
            "sum": float(sum(values)) if values else None}


def checks(arm):
    return arm.get("checks") or [e for e in arm.get("events", []) if e.get("kind") == "CHECK"]


def failure_category(arm):
    termination = arm.get("termination") or ""
    reason = (arm.get("grading") or {}).get("exclusion") or ""
    if "BUDGET" in termination or "LIMIT" in termination:
        return "BUDGET_OR_INPUT_LIMIT"
    if "FORMAT" in termination or reason in {"NO_FINAL_ANSWER", "FINAL_FORMAT_ERROR", "MISSING_OR_INCOMPLETE_BOXED", "PREDICTION_UNPARSEABLE", "PREDICTION_ERROR"}:
        return "FORMAT_OR_PREDICTION_PARSE"
    if exclusion(arm) is not None:
        return "EVALUATION_UNAVAILABLE"
    if arm.get("Y") == 1:
        return "CORRECT"
    return "WRONG_SUBMITTED" if arm.get("submitted_reasoning_ids") is not None or termination in ("FINAL_PASS", "FINAL_UNCERTAIN", "EOS") else "OTHER_ALGORITHM_FAILURE"


def number(value):
    return f"{value:.4f}" if finite(value) else "NA"


def effect(value):
    ci = value.get("ci95")
    return f"{number(value.get('estimate'))} [{number(ci[0])}, {number(ci[1])}]" if ci else f"NA（{value.get('reason') or '未执行'}）"


def bootstrap_count(value, total):
    return f"{value.get('bootstrap_valid', 0)}/{total}"




def interpretation(value):
    ci = value.get("ci95")
    if ci is None:
        return "NA：尚无可评估证据。"
    size = min(value["low_questions"], value["high_questions"]) if "low_questions" in value else value.get("paired_questions")
    if size == 1:
        return "仅一个有效配对或高/低组仅一题，bootstrap 区间可能退化，不足以支持推广结论。"
    caveat = "小样本区间可能不稳定；" if size is not None and size < 10 else ""
    if ci[0] > 0:
        return caveat + "95% CI 完全高于零，为立即检查在该指标上更有收益提供初步证据。"
    if ci[1] < 0:
        return caveat + "95% CI 完全低于零，为推迟检查在该指标上更有收益提供初步证据。"
    return caveat + "95% CI 跨零，当前证据不足；不能据此宣称两种动作等效。"
