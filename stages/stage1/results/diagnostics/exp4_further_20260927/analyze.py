"""Read-only experiment-four reanalysis. Run with the repository's .venv Python.

No model imports, downloads, inference, or writes under runs/. All CSVs/figures
and report.md are regenerated here. Assertions audit accounting and selection.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics as stats
import sys
import tarfile
from datetime import datetime, timezone
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
STAGE = OUT.parents[2]
ROOT = STAGE.parents[1]
RUN = STAGE / "runs/7bf883613b3ff02a"
PARENT = STAGE / "runs/3660363ff392fea6"
ACTIONS = ["now", "delay1", "delay2", "endpoint_only"]
POLICIES = ["unchecked", "endpoint", "k4", "random"]
GROUPS = ["initial", "historical_pass"]
SEEDS = [17, 29, 43]
WEIGHTS = [0, .05, .1, .2, .5, 1]
BOOT_SEED = 20260922
BOOT_N = 2000
INPUTS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path.read_text()


def js(path):
    return json.loads(read(path))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-10), (a, b)


def mean(rows, key):
    return stats.mean(r[key] for r in rows)


def spread(values):
    return {"mean": stats.mean(values), "sd": stats.stdev(values) if len(values) > 1 else 0,
            "min": min(values), "max": max(values)}


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def pure_functions(source, names, namespace):
    """Reuse audited pure functions without importing inference-capable modules."""
    tree = ast.parse(read(source))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def bootstrap(values):
    # Exactly the original algorithm: Python random.choices, 2000, indices 50/1949.
    rng = random.Random(BOOT_SEED)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(BOOT_N))
    return [samples[50], samples[1949]]


def cluster_bootstrap(rows, key, equal_question=False):
    """Resample question IDs; keep all their states and paired-seed averages."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["question_id"]].append(row[key])
    ids = sorted(clusters)
    rng = random.Random(BOOT_SEED)
    samples = []
    for _ in range(BOOT_N):
        chosen = rng.choices(ids, k=len(ids))
        if equal_question:
            samples.append(stats.mean(stats.mean(clusters[q]) for q in chosen))
        else:
            samples.append(sum(sum(clusters[q]) for q in chosen) / sum(len(clusters[q]) for q in chosen))
    samples.sort()
    return [samples[50], samples[1949]]


def audit_tokens(state):
    """Replay token provenance, not text: feedback/prompt are None, generated
    tokens have unique (event index, offset). This avoids counting old feedback
    as wasted generation and includes look-ahead pending tokens on rollback.
    """
    context = [None] * len(state["prompt_ids"])
    checkpoint = context.copy()
    pending = []
    discarded = set()
    rollback_rows = []
    branch_index = -1
    precheck = set()
    first_verify = None
    first_verdict = None
    generated = 0
    fail_index = 0
    for i, e in enumerate(state["events"]):
        assert e["action_id"] == i
        kind = e["event"]
        if kind == "oracle_branch":
            branch_index = i
        elif kind == "pending_merged":
            assert len(pending) == e["tokens"]
            context.extend(pending)
            pending = []
        elif kind == "generate":
            assert not pending
            assert len(context) == e["prefill_tokens"]
            n = e["kept_tokens"] + e["pending_tokens"]
            labels = [(i, k) for k in range(n)]
            context.extend(labels[:e["kept_tokens"]])
            pending = labels[e["kept_tokens"]:]
            generated += n
            if branch_index >= 0 and first_verify is None:
                precheck.update(labels)
        elif kind == "verify_call" and branch_index >= 0 and first_verify is None:
            first_verify = i
        elif kind == "verdict" and branch_index >= 0 and first_verdict is None:
            first_verdict = e["effective"]
        elif kind == "checkpoint":
            assert len(context) == e["tokens"]
            checkpoint = context.copy()
        elif kind == "rollback":
            assert len(checkpoint) == e["checkpoint_tokens"]
            fail = state["fail_snapshots"][fail_index]["state"]
            fail_index += 1
            assert len(fail["context_ids"]) == len(context)
            assert len(fail["pending_ids"]) == len(pending)
            assert len(fail["checkpoint_ids"]) == len(checkpoint)
            gone = {t for t in context[len(checkpoint):] + pending if t is not None}
            assert not discarded.intersection(gone)
            discarded.update(gone)
            rollback_rows.append({"event_index": i, "discarded_generated_tokens": len(gone),
                                  "precheck_new_discarded_tokens": len(gone & precheck),
                                  "feedback_tokens_inserted": e["feedback_tokens"]})
            context = checkpoint.copy() + [None] * e["feedback_tokens"]
            pending = []
    assert generated == state["generated_tokens"]
    assert fail_index == len(state["fail_snapshots"])
    assert len(context) == len(state["context_ids"])
    assert len(pending) == len(state["pending_ids"])
    retained = {x for x in context + pending if x is not None}
    assert generated == len(discarded) + len(retained)
    return {"discarded_generated_tokens": len(discarded), "retained_generated_tokens": len(retained),
            "suffix_generated_discarded_tokens": sum(i > branch_index for i, _ in discarded),
            "precheck_new_generated_tokens": len(precheck),
            "precheck_new_eventually_discarded_tokens": len(precheck & discarded),
            "precheck_new_first_fail_discarded_tokens": sum(r["precheck_new_discarded_tokens"] for r in
                rollback_rows if branch_index < r["event_index"] and first_verdict == "FAIL"
                and r["event_index"] < next((j for j, e in enumerate(state["events"])
                    if j > first_verify and e["event"] == "verdict"), len(state["events"]))),
            "first_check_exists": first_verify is not None,
            "first_effective_verdict": first_verdict or "NO_CHECK", "rollbacks": rollback_rows}


def metrics(state, start=0):
    events = state["events"][start:]
    gen = [e for e in events if e["event"] == "generate"]
    ver = [e for e in events if e["event"] == "verify_call"]
    before = True
    initial_decode = rework_decode = 0.
    before_tokens = rework_tokens = 0
    # Before/after FIRST actual rollback, not a claim that every later token is wasted.
    for e in events:
        if e["event"] == "rollback":
            before = False
        elif e["event"] == "generate":
            n = e["kept_tokens"] + e["pending_tokens"]
            if before:
                initial_decode += e["decode_seconds"]
                before_tokens += n
            else:
                rework_decode += e["decode_seconds"]
                rework_tokens += n
    pre = sum(e["prefill_seconds"] for e in gen if e["prefill_kind"] == "INITIAL")
    recovery = sum(e["prefill_seconds"] for e in gen if e["prefill_kind"] == "FAIL_RESTORE")
    offline = sum(e["prefill_seconds"] for e in gen if e["prefill_kind"] == "OFFLINE_RESTORE")
    assert all(e["prefill_kind"] in {"INITIAL", "FAIL_RESTORE", "OFFLINE_RESTORE"} for e in gen)
    decision = sum(e["seconds"] for e in events if e["event"] == "policy_decision")
    verification = sum(e["seconds"] for e in ver)
    generation = initial_decode + rework_decode + pre
    result = {"service_seconds": generation + recovery + verification + decision,
              "generation_seconds": generation, "initial_prefill_seconds": pre,
              "before_first_rollback_decode_seconds": initial_decode,
              "after_first_rollback_decode_seconds": rework_decode,
              "verification_seconds": verification, "recovery_seconds": recovery,
              "decision_seconds": decision, "offline_restore_seconds": offline,
              "generated_tokens": before_tokens + rework_tokens,
              "before_first_rollback_generated_tokens": before_tokens,
              "after_first_rollback_generated_tokens": rework_tokens,
              "checks": len(ver), "verifier_input_tokens": sum(e["input_tokens"] for e in ver),
              "verifier_output_tokens": sum(e["output_tokens"] for e in ver),
              "rollbacks": sum(e["event"] == "rollback" for e in events),
              "feedback_tokens_inserted": sum(e["feedback_tokens"] for e in events if e["event"] == "rollback"),
              "fail_restore_input_tokens": sum(e["prefill_tokens"] for e in gen if e["prefill_kind"] == "FAIL_RESTORE"),
              "generation_calls": len(gen)}
    if start == 0:
        close(sum(e["prefill_seconds"] + e["decode_seconds"] for e in gen), state["generator_seconds"])
        close(verification, state["verifier_seconds"])
        close(offline, state["offline_restore_seconds"])
        close(recovery, state["feedback_prefill_seconds"])
        assert result["checks"] == state["verifier_calls"]
        assert result["generated_tokens"] == state["generated_tokens"]
        assert result["verifier_output_tokens"] == state["verifier_output_tokens"]
        assert sum(e["input_tokens"] for e in ver) + sum(e["prefill_tokens"] for e in gen
            if e["prefill_kind"] != "OFFLINE_RESTORE") == state["prefill_tokens"]
    return result


def phase_rows(state, path_id, start=0):
    rows = []
    for e in state["events"][start:]:
        base = {"path_id": path_id, "question_id": state["request_id"], "event_index": e["action_id"]}
        if e["event"] == "generate":
            kind = e["prefill_kind"]
            rows.append({**base, "phase": kind, "seconds": e["prefill_seconds"],
                         "in_service_cost": kind != "OFFLINE_RESTORE", "tokens": e["prefill_tokens"]})
            rows.append({**base, "phase": "decode", "seconds": e["decode_seconds"],
                         "in_service_cost": True, "tokens": e["kept_tokens"] + e["pending_tokens"]})
        elif e["event"] in {"verify_call", "policy_decision"}:
            rows.append({**base, "phase": e["event"], "seconds": e["seconds"],
                         "in_service_cost": True, "tokens": e.get("input_tokens", 0)})
    return rows


def select(index, identifiers, held, weight, cost_unit, with_interval=True):
    training = [s for s in SEEDS if s != held]
    utility = lambda row: int(row["correct"]) - weight * row["suffix_service_seconds"] / cost_unit
    estimates = {(s, a): sum(utility(index[s, a, seed]) for seed in training) / 2
                 for s in identifiers for a in ACTIONS}
    scores = {a: sum(estimates[s, a] for s in identifiers) for a in ACTIONS}
    reference = max(ACTIONS, key=scores.get)  # first maximum: original deterministic tie rule
    chosen = {s: max(ACTIONS, key=lambda a: estimates[s, a]) for s in identifiers}
    details = []
    for s in identifiers:
        a, b = index[s, chosen[s], held], index[s, reference, held]
        details.append({"group": a["group"], "snapshot_id": s, "question_id": a["question_id"],
                        "evaluation_seed": held, "selection_seeds": training, "lambda": weight,
                        "chosen_action": chosen[s], "uniform_action": reference,
                        "exact_tie_actions": [x for x in ACTIONS if estimates[s, x] == estimates[s, chosen[s]]],
                        "exact_tie_count": sum(estimates[s, x] == estimates[s, chosen[s]] for x in ACTIONS),
                        "selection_estimates": {x: estimates[s, x] for x in ACTIONS},
                        "selected_correct": int(a["correct"]), "uniform_correct": int(b["correct"]),
                        "selected_seconds": a["suffix_service_seconds"], "uniform_seconds": b["suffix_service_seconds"],
                        "quality_gap": int(a["correct"]) - int(b["correct"]),
                        "cost_gap_selected_minus_uniform": a["suffix_service_seconds"] - b["suffix_service_seconds"],
                        "utility_gap": utility(a) - utility(b)})
    gaps = [r["utility_gap"] for r in details]
    return {"evaluation_seed": held, "selection_seeds": training, "lambda": weight,
            "count": len(identifiers), "selected_correct": sum(r["selected_correct"] for r in details),
            "uniform_correct": sum(r["uniform_correct"] for r in details), "uniform_action": reference,
            "selected_mean_seconds": mean(details, "selected_seconds"),
            "uniform_mean_seconds": mean(details, "uniform_seconds"),
            "utility_gap": stats.mean(gaps), "conditional_bootstrap_ci": bootstrap(gaps) if with_interval else [],
            "tied_snapshots": sum(r["exact_tie_count"] > 1 for r in details),
            "uniform_tie_actions": [a for a in ACTIONS if scores[a] == scores[reference]],
            "action_counts": dict(Counter(chosen.values()))}, details


def main():
    # Preserve all original bytes, including reports, not just the files used below.
    original_hashes = {str(p.relative_to(ROOT)): digest(p) for folder in (RUN, PARENT)
                       for p in sorted(folder.rglob("*")) if p.is_file()}
    config = js(RUN / "config.json")
    parent_config = js(PARENT / "config.json")
    for document in [ROOT / "README.md", STAGE / "README.md", STAGE / "results/README.md",
                     STAGE / "第一阶段实验方案.md", STAGE / "scripts/audit_exp4.py"]:
        read(document)
    cost_unit = config["cost_unit_seconds"]  # frozen 7.748207573824038; displayed as 7.748
    assert config["settings"]["actions"] == ACTIONS
    assert config["settings"]["cost_weights"] == WEIGHTS
    assert config["settings"]["selection_seeds"] == [17, 29]
    assert config["settings"]["evaluation_seed"] == 43
    assert config["config"] == parent_config["config"]
    assert config["assets"] == parent_config["assets"]
    for relative, expected in config["identity"]["inputs"].items():
        assert digest(PARENT / relative) == expected, relative
    for relative, expected in config["identity"]["code"].items():
        assert digest(STAGE / relative) == expected, relative
    with tarfile.open(RUN / "source_snapshot.tar.gz") as archive:
        for relative, expected in config["identity"]["code"].items():
            assert hashlib.sha256(archive.extractfile(relative).read()).hexdigest() == expected
    reused = js(RUN / "reused_trajectories.json")
    for r in reused:
        relative = r["path"].split("/states/", 1)[1]
        assert digest(PARENT / "states" / relative) == r["sha256"]
    score_ns = pure_functions(STAGE / "veriserve/data.py", ["numeric_value", "is_correct"],
                             dict(re=re, Decimal=Decimal, InvalidOperation=InvalidOperation, Fraction=Fraction))
    is_correct = score_ns["is_correct"]
    original_summary = js(RUN / "exp4/summary.json")
    coverage = js(RUN / "snapshot_coverage.json")
    old_audit = js(RUN / "exp4/artifact_audit.json")
    held_pairs = js(RUN / "exp4/held_seed_action_pairs.json")
    old_comparisons = list(csv.DictReader(read(RUN / "exp4/oracle_comparison.csv").splitlines()))
    read(RUN / "report.md")
    read(PARENT / "exp2_report.md")
    a_rows, stages, rollback_events = [], [], []
    a_common = {}
    full_csv = list(csv.DictReader(read(RUN / "exp4/full_paths.csv").splitlines()))
    for raw in full_csv:
        p, q = raw["policy"], raw["question_id"]
        folder = PARENT / "states/exp2b_eval" if p in POLICIES[:2] else RUN / "states/exp4_eval"
        state = js(folder / p / f"{q}.json")
        common = {k: state[k] for k in ["question", "gold", "budget", "prompt_ids", "seed"]}
        assert a_common.setdefault(q, common) == common
        m = metrics(state)
        tokens = audit_tokens(state)
        close(m["service_seconds"], float(raw["service_seconds"]))
        assert is_correct(state["final_answer"], state["gold"]) == (raw["correct"] == "True")
        assert (state["final_answer"] is None) == (raw["unfinished"] == "True")
        row = {"question_id": q, "policy": p, "source_run": folder.parent.parent.name, "seed": str(state["seed"]),
               "correct": int(raw["correct"] == "True"), "unfinished": int(raw["unfinished"] == "True"),
               "termination": state["termination"], **m,
               "discarded_generated_tokens": tokens["discarded_generated_tokens"],
               "retained_generated_tokens": tokens["retained_generated_tokens"],
               "endpoint_checks": sum(c["candidate"] is not None for c in state["checks"]),
               "intermediate_checks": sum(c["candidate"] is None for c in state["checks"])}
        a_rows.append(row)
        stages.extend(phase_rows(state, f"A/{p}/{q}"))
        rollback_events.extend({"path_id": f"A/{p}/{q}", **r} for r in tokens["rollbacks"])
    a_summary = []
    terminations = []
    a_index = {(r["question_id"], r["policy"]): r for r in a_rows}
    assert len(a_rows) == len(a_index) == 400
    for p, correct, expected_seconds in zip(POLICIES, [49, 65, 62, 60], [9.23, 17.41, 12.68, 12.13]):
        rows = [r for r in a_rows if r["policy"] == p]
        assert {r["question_id"] for r in rows} == set(config["evaluation_ids"])
        result = {"policy": p, "n": len(rows), "correct": sum(r["correct"] for r in rows),
                  "unfinished": sum(r["unfinished"] for r in rows)}
        result.update({"mean_" + k: mean(rows, k) for k in rows[0]
                       if isinstance(rows[0][k], (int, float)) and k not in {"correct", "unfinished"}})
        result.update({"total_" + k: sum(r[k] for r in rows) for k in
                       ["checks", "endpoint_checks", "intermediate_checks", "generated_tokens", "verifier_input_tokens",
                        "verifier_output_tokens", "rollbacks", "discarded_generated_tokens"]})
        result["seconds_per_verification"] = sum(r["verification_seconds"] for r in rows) / result["total_checks"] if result["total_checks"] else None
        result["input_tokens_per_verification"] = result["total_verifier_input_tokens"] / result["total_checks"] if result["total_checks"] else None
        result["decode_seconds_per_token"] = sum(r["generation_seconds"] - r["initial_prefill_seconds"] for r in rows) / result["total_generated_tokens"]
        assert result["correct"] == correct
        assert abs(result["mean_service_seconds"] - expected_seconds) < .01
        close(result["mean_service_seconds"], original_summary["baselines"][p]["mean_service_seconds"])
        a_summary.append(result)
        terminations.extend({"policy": p, "termination": t, "count": n}
                            for t, n in sorted(Counter(r["termination"] for r in rows).items()))
    a_pairs, a_pair_summary = [], []
    pair_metrics = ["correct", "service_seconds", "generation_seconds", "verification_seconds", "recovery_seconds",
                    "decision_seconds", "checks", "generated_tokens", "after_first_rollback_generated_tokens",
                    "after_first_rollback_decode_seconds", "discarded_generated_tokens", "rollbacks"]
    for p, ref in [("k4", "endpoint"), ("k4", "unchecked"), ("random", "endpoint"), ("random", "k4")]:
        pairs = []
        for q in config["evaluation_ids"]:
            a, b = a_index[q, p], a_index[q, ref]
            pairs.append({"question_id": q, "policy": p, "reference": ref,
                          "policy_correct": a["correct"], "reference_correct": b["correct"],
                          "policy_termination": a["termination"], "reference_termination": b["termination"],
                          **{"delta_" + k: a[k] - b[k] for k in pair_metrics}})
        a_pairs.extend(pairs)
        for k in pair_metrics:
            vals = [r["delta_" + k] for r in pairs]
            a_pair_summary.append({"policy": p, "reference": ref, "metric": k, "n": len(vals),
                                   **spread(vals), "ci": bootstrap(vals),
                                   "negative_pairs": sum(v < 0 for v in vals), "positive_pairs": sum(v > 0 for v in vals),
                                   "zero_pairs": vals.count(0)})
    for old in original_summary["paired"]:
        row = next(x for x in a_pair_summary if x["policy"] == old["policy_a"] and
                   x["reference"] == old["policy_b"] and x["metric"] == "correct")
        close(row["mean"], old["accuracy_difference"])
        assert row["ci"] == old["bootstrap_95_interval"]

    snapshots = {}
    for path in sorted((RUN / "snapshots").rglob("*.json")):
        s = js(path)
        assert stable_hash(s["state"]) == s["state_hash"]
        snapshots[s["id"]] = s
    assert Counter(s["group"] for s in snapshots.values()) == {"initial": 20, "historical_pass": 12}
    assert len({s["question_id"] for s in snapshots.values()}) == 20
    branches = [json.loads(x) for x in read(RUN / "exp4_oracle/branches.jsonl").splitlines()]
    b_csv = list(csv.DictReader(read(RUN / "exp4/oracle_branches.csv").splitlines()))
    assert [r["branch_id"] for r in branches] == [r["branch_id"] for r in b_csv]
    seed_map, b_metrics, precheck_rows = {}, [], []
    for b, raw in zip(branches, b_csv):
        state = js(RUN / "states/exp4_oracle" / (b["branch_id"] + ".json"))
        history = snapshots[b["snapshot_id"]]["state"]
        start = len(history["events"])
        assert state["events"][:start] == history["events"]
        assert state["checks"][:len(history["checks"])] == history["checks"]
        assert state["budget"] == history["budget"]
        assert state["question"] == history["question"] and state["gold"] == history["gold"]
        assert state["events"][start]["event"] == "oracle_branch"
        expected_seed = int(stable_hash((BOOT_SEED, f"oracle/{b['snapshot_id']}/{b['replicate']}"))[:12], 16)
        assert state["seed"] == b["seed"] == expected_seed
        pair = b["snapshot_id"], b["replicate"]
        assert seed_map.setdefault(pair, b["seed"]) == b["seed"]
        full, hist, suffix = metrics(state), metrics(history), metrics(state, start)
        close(full["service_seconds"] - hist["service_seconds"], b["suffix_service_seconds"])
        close(suffix["service_seconds"], b["suffix_service_seconds"])
        close(float(raw["suffix_service_seconds"]), b["suffix_service_seconds"])
        assert b["correct"] == is_correct(state["final_answer"], state["gold"])
        for key, new_key in [("generated_tokens", "new_generated_tokens"), ("verifier_calls", "new_checks"), ("prefill_tokens", "new_prefill_tokens")]:
            assert state[key] - history[key] == b[new_key]
            assert state[key] <= state["budget"][key]
        t = audit_tokens(state)
        base = {k: b[k] for k in ["snapshot_id", "question_id", "group", "action", "replicate", "seed"]}
        b_metrics.append({**base, "correct": int(b["correct"]), "unfinished": int(b["unfinished"]),
                          "termination": b["termination"], **suffix,
                          "suffix_generated_discarded_tokens": t["suffix_generated_discarded_tokens"]})
        precheck_rows.append({**base, **{k: v for k, v in t.items() if k.startswith("precheck_") or k.startswith("first_")}})
        stages.extend(phase_rows(state, "B/" + b["branch_id"], start))
        rollback_events.extend({"path_id": "B/" + b["branch_id"], **r} for r in t["rollbacks"] if r["event_index"] > start)
    index = {(b["snapshot_id"], b["action"], b["replicate"]): b for b in branches}
    assert len(index) == len(branches) == 384
    assert len(seed_map) == 96
    b_pairs, b_snapshots, b_summary = [], [], []
    for s in sorted(snapshots):
        for action in ACTIONS[1:]:
            rows = []
            for seed in SEEDS:
                now, delay = index[s, "now", seed], index[s, action, seed]
                rows.append({"snapshot_id": s, "question_id": now["question_id"], "group": now["group"],
                             "comparison": action, "replicate": seed, "actual_seed": now["seed"],
                             "now_correct": int(now["correct"]), "delay_correct": int(delay["correct"]),
                             "now_seconds": now["suffix_service_seconds"], "delay_seconds": delay["suffix_service_seconds"],
                             "delta_T": delay["suffix_service_seconds"] - now["suffix_service_seconds"],
                             "delta_q": int(now["correct"]) - int(delay["correct"])})
            b_pairs.extend(rows)
            base = {k: rows[0][k] for k in ["snapshot_id", "question_id", "group", "comparison"]}
            b_snapshots.append({**base, **{f"{k}_{stat}": value for k in ["delta_T", "delta_q"]
                                         for stat, value in spread([r[k] for r in rows]).items()},
                                "positive_T_seeds": sum(r["delta_T"] > 0 for r in rows),
                                "negative_T_seeds": sum(r["delta_T"] < 0 for r in rows),
                                "nonzero_q_seeds": sum(r["delta_q"] != 0 for r in rows)})
    for group in GROUPS + ["combined"]:
        for action in ACTIONS[1:]:
            rows = [r for r in b_snapshots if r["comparison"] == action and (group == "combined" or r["group"] == group)]
            for weighting in (["snapshot", "question"] if group == "combined" else ["snapshot"]):
                equal = weighting == "question"
                result = {"group": group, "comparison": action, "weighting": weighting,
                          "snapshots": len(rows), "questions": len({r["question_id"] for r in rows})}
                for k in ["delta_T", "delta_q"]:
                    by_q = defaultdict(list)
                    for row in rows:
                        by_q[row["question_id"]].append(row[k + "_mean"])
                    result[k] = stats.mean(stats.mean(v) for v in by_q.values()) if equal else mean(rows, k + "_mean")
                    result[k + "_question_cluster_ci"] = cluster_bootstrap(rows, k + "_mean", equal)
                    result[k + "_mean_within_snapshot_seed_sd"] = mean(rows, k + "_sd")
                b_summary.append(result)
    b_seed_summary = []
    for group, action, seed in itertools.product(GROUPS, ACTIONS[1:], SEEDS):
        rows = [r for r in b_pairs if r["group"] == group and r["comparison"] == action and r["replicate"] == seed]
        b_seed_summary.append({"group": group, "comparison": action, "replicate": seed, "snapshots": len(rows),
                               "delta_T": mean(rows, "delta_T"), "delta_q": mean(rows, "delta_q")})
    for old in held_pairs:
        if old["action_a"] == "now":
            row = next(r for r in b_seed_summary if r["group"] == old["group"] and
                       r["comparison"] == old["action_b"] and r["replicate"] == 43)
            close(row["delta_T"], -old["mean_suffix_seconds_a_minus_b"])
            close(row["delta_q"], old["accuracy_difference"])

    # Original pure analysis is executed for exact reproducibility, never the runner/auditor.
    old_fn = pure_functions(STAGE / "scripts/run_exp4.py", ["analyze_oracle"], {"random": random})["analyze_oracle"]
    reproduced = old_fn(branches, config["settings"], cost_unit)
    assert reproduced == original_summary["oracle"]
    for old, raw in zip(reproduced["comparisons"], old_comparisons):
        close(old["utility_gap"], float(raw["utility_gap"]))
    # Leakage check: changing held outcomes/costs arbitrarily must not change selection.
    for group, held in itertools.product(GROUPS, SEEDS):
        ids = sorted(s for s, v in snapshots.items() if v["group"] == group)
        original, ds = select(index, ids, held, .1, cost_unit, False)
        altered = {key: ({**value, "correct": not value["correct"], "suffix_service_seconds": 100000 + i}
                         if key[2] == held else value) for i, (key, value) in enumerate(index.items())}
        changed, new_ds = select(altered, ids, held, .1, cost_unit, False)
        assert original["uniform_action"] == changed["uniform_action"]
        assert [r["chosen_action"] for r in ds] == [r["chosen_action"] for r in new_ds]
    c_rounds, c_details, loo, c_combined = [], [], [], []
    rounded_same = True
    for group, held, weight in itertools.product(GROUPS, SEEDS, WEIGHTS):
        ids = sorted(s for s, v in snapshots.items() if v["group"] == group)
        result, details = select(index, ids, held, weight, cost_unit)
        c_rounds.append({"group": group, **result})
        c_details.extend(details)
        rounded_result, rounded_details = select(index, ids, held, weight, 7.748, False)
        rounded_same &= (rounded_result["uniform_action"] == result["uniform_action"] and
                         [r["chosen_action"] for r in details] == [r["chosen_action"] for r in rounded_details])
        if held == 43:
            old = next(r for r in reproduced["comparisons"] if r["group"] == group and r["lambda"] == weight)
            close(result["utility_gap"], old["utility_gap"])
            assert [round(v, 4) for v in result["conditional_bootstrap_ci"]] == old["bootstrap_95_interval"]
        for omitted in sorted({r["question_id"] for r in details}):
            kept = [r for r in details if r["question_id"] != omitted]
            new, _ = select(index, [r["snapshot_id"] for r in kept], held, weight, cost_unit, False)
            full_sign = np.sign(result["utility_gap"])
            fixed_gap = mean(kept, "utility_gap")
            loo.append({"group": group, "evaluation_seed": held, "lambda": weight, "omitted_question": omitted,
                        "full_utility_gap": result["utility_gap"], "fixed_choices_gap": fixed_gap,
                        "reselected_reference_gap": new["utility_gap"], "new_uniform_action": new["uniform_action"],
                        "fixed_direction_changed": bool(np.sign(fixed_gap) != full_sign),
                        "reselected_direction_changed": bool(np.sign(new["utility_gap"]) != full_sign),
                        "fixed_sign_reversed": bool(np.sign(fixed_gap) * full_sign < 0),
                        "reselected_sign_reversed": bool(np.sign(new["utility_gap"]) * full_sign < 0)})
    for held, weight in itertools.product(SEEDS, WEIGHTS):
        rows = [r for r in c_details if r["evaluation_seed"] == held and r["lambda"] == weight]
        assert len(rows) == 32
        c_combined.append({"evaluation_seed": held, "lambda": weight, "snapshots": 32, "questions": 20,
                           "reference": "group-specific training-selected uniform actions",
                           "utility_gap_snapshot_weighted": mean(rows, "utility_gap"),
                           "question_cluster_ci": cluster_bootstrap(rows, "utility_gap"),
                           "utility_gap_question_weighted": stats.mean(stats.mean(r["utility_gap"] for r in rows if r["question_id"] == q)
                                                                         for q in sorted({r["question_id"] for r in rows})),
                           "equal_question_cluster_ci": cluster_bootstrap(rows, "utility_gap", True)})
        for omitted in sorted({r["question_id"] for r in rows}):
            kept = [r for r in rows if r["question_id"] != omitted]
            new_details = []
            for group in GROUPS:
                ids = [r["snapshot_id"] for r in kept if r["group"] == group]
                _, ds = select(index, sorted(ids), held, weight, cost_unit, False)
                new_details.extend(ds)
            full_gap, fixed_gap, new_gap = mean(rows, "utility_gap"), mean(kept, "utility_gap"), mean(new_details, "utility_gap")
            loo.append({"group": "combined", "evaluation_seed": held, "lambda": weight, "omitted_question": omitted,
                        "full_utility_gap": full_gap, "fixed_choices_gap": fixed_gap,
                        "reselected_reference_gap": new_gap, "new_uniform_action": {g: next(r["uniform_action"] for r in new_details if r["group"] == g) for g in GROUPS},
                        "fixed_direction_changed": bool(np.sign(fixed_gap) != np.sign(full_gap)),
                        "reselected_direction_changed": bool(np.sign(new_gap) != np.sign(full_gap)),
                        "fixed_sign_reversed": bool(np.sign(fixed_gap) * np.sign(full_gap) < 0),
                        "reselected_sign_reversed": bool(np.sign(new_gap) * np.sign(full_gap) < 0)})
    consistency = []
    for s, weight in itertools.product(sorted(snapshots), WEIGHTS):
        rows = [r for r in c_details if r["snapshot_id"] == s and r["lambda"] == weight]
        choices = {r["evaluation_seed"]: r["chosen_action"] for r in rows}
        consistency.append({"snapshot_id": s, "question_id": snapshots[s]["question_id"], "group": snapshots[s]["group"],
                            "lambda": weight, **{f"held_{seed}": choices[seed] for seed in SEEDS},
                            "unique_actions": len(set(choices.values())), "all_equal": len(set(choices.values())) == 1,
                            "pairwise_agreement": sum(choices[a] == choices[b] for a, b in itertools.combinations(SEEDS, 2)) / 3})
    noise = []
    for group, action in itertools.product(GROUPS, ACTIONS):
        ids = sorted(s for s, v in snapshots.items() if v["group"] == group)
        qualities = [[int(index[s, action, seed]["correct"]) for seed in SEEDS] for s in ids]
        noise.append({"group": group, "action": action, "snapshots": len(ids),
                      "snapshots_with_quality_seed_disagreement": sum(len(set(v)) > 1 for v in qualities),
                      "mean_within_snapshot_cost_seed_sd": stats.mean(stats.stdev(index[s, action, seed]["suffix_service_seconds"] for seed in SEEDS) for s in ids)})
    phase_audit = []
    for run in [RUN, PARENT]:
        times = [json.loads(x) for x in read(run / "phase_times.jsonl").splitlines()]
        phase_audit.append({"run_id": run.name, "generator_wave_seconds": sum(r["seconds"] for r in times if r["role"] == "generator"),
                            "verifier_wave_seconds": sum(r["seconds"] for r in times if r["role"] == "verifier"),
                            "total_wave_seconds": sum(r["seconds"] for r in times), "waves": len(times)})
    # New-run wave residual: subtract only newly executed events, not inherited histories.
    new_actual = Counter()
    for path in sorted((RUN / "states").rglob("*.json")):
        state = js(path)
        start = next((i for i, e in enumerate(state["events"]) if e["event"] == "oracle_branch"), 0)
        for e in state["events"][start:]:
            if e["event"] == "generate":
                new_actual["generator"] += e["prefill_seconds"] + e["decode_seconds"]
            elif e["event"] == "verify_call":
                new_actual["verifier"] += e["seconds"]
    for role in ["generator", "verifier"]:
        phase_audit[0][role + "_recorded_compute_seconds"] = new_actual[role]
        phase_audit[0][role + "_unattributed_wave_residual_seconds"] = phase_audit[0][role + "_wave_seconds"] - new_actual[role]
    precheck_summary = []
    for group, action in itertools.product(GROUPS, ACTIONS):
        rows = [r for r in precheck_rows if r["group"] == group and r["action"] == action]
        precheck_summary.append({"group": group, "action": action, "branches": len(rows),
                                "first_check_exists": sum(r["first_check_exists"] for r in rows),
                                "first_FAIL": sum(r["first_effective_verdict"] == "FAIL" for r in rows),
                                **{"mean_" + k: mean(rows, k) for k in ["precheck_new_generated_tokens", "precheck_new_eventually_discarded_tokens", "precheck_new_first_fail_discarded_tokens"]}})
    loo_summary = []
    for group, held, weight in itertools.product(GROUPS + ["combined"], SEEDS, WEIGHTS):
        rows = [r for r in loo if r["group"] == group and r["evaluation_seed"] == held and r["lambda"] == weight]
        loo_summary.append({"group": group, "evaluation_seed": held, "lambda": weight,
                            "full_utility_gap": rows[0]["full_utility_gap"],
                            **{k: [r["omitted_question"] for r in rows if r[k]] for k in
                               ["fixed_direction_changed", "reselected_direction_changed", "fixed_sign_reversed", "reselected_sign_reversed"]},
                            "fixed_gap_min": min(r["fixed_choices_gap"] for r in rows),
                            "fixed_gap_max": max(r["fixed_choices_gap"] for r in rows),
                            "reselected_gap_min": min(r["reselected_reference_gap"] for r in rows),
                            "reselected_gap_max": max(r["reselected_reference_gap"] for r in rows)})
    for name, rows in [("a_paths.csv", a_rows), ("a_cost_summary.csv", a_summary), ("a_paired_questions.csv", a_pairs),
                       ("a_paired_summary.csv", a_pair_summary), ("a_termination_counts.csv", terminations),
                       ("b_branches_audited.csv", b_metrics), ("b_paired_seeds.csv", b_pairs), ("b_paired_snapshots.csv", b_snapshots),
                       ("b_group_summary.csv", b_summary), ("b_group_by_seed.csv", b_seed_summary),
                       ("b_first_check_waste.csv", precheck_rows), ("c_rotations.csv", c_rounds),
                       ("b_first_check_waste_summary.csv", precheck_summary), ("c_loo_summary.csv", loo_summary),
                       ("c_snapshot_selections.csv", c_details), ("c_leave_one_question_out.csv", loo),
                       ("c_combined_cluster.csv", c_combined), ("c_action_consistency.csv", consistency),
                       ("seed_variability.csv", noise), ("ordered_phases.csv", stages), ("rollback_events.csv", rollback_events),
                       ("seed_mapping.csv", [{"snapshot_id": s, "replicate": rep, "actual_seed": value} for (s, rep), value in sorted(seed_map.items())])]:
        write_csv(name, rows)
    assert len(c_details) == 576 and len(loo) == 936 and len(consistency) == 192
    assert rounded_same
    write_json("reproduced_original.json", {"baselines": original_summary["baselines"], "paired": original_summary["paired"], "oracle": reproduced})
    all_data = {"a_summary": a_summary, "a_pairs": a_pair_summary, "terminations": terminations,
                "b_summary": b_summary, "b_seed_summary": b_seed_summary, "c_rounds": c_rounds,
                "phase_audit": phase_audit, "noise": noise, "precheck_summary": precheck_summary,
                "loo_summary": loo_summary, "b_snapshots": b_snapshots, "b_pairs": b_pairs,
                "c_combined": c_combined}
    write_json("summary.json", all_data)
    figures(a_summary, b_snapshots, c_rounds)
    report(all_data, config, precheck_rows, c_details, consistency, loo, coverage)
    assert all(digest(ROOT / p) == h for p, h in original_hashes.items()), "An original artifact changed"
    manifest = {"command": ".venv/bin/python stages/stage1/results/diagnostics/exp4_further_20260927/analyze.py",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "script_sha256": digest(Path(__file__)), "run": RUN.name, "parent_run": PARENT.name,
                "cost_unit_seconds": cost_unit, "rounded_7_748_choices_identical": bool(rounded_same),
                "bootstrap": {"replications": BOOT_N, "seed": BOOT_SEED, "quantile_sorted_indices": [50, 1949]},
                "audit": {"status": "PASS", "a_paths": len(a_rows), "snapshots": len(snapshots),
                          "b_branches": len(branches), "c_group_lambda_rounds": len(c_rounds),
                          "token_provenance_conserved": True, "original_analysis_exactly_reproduced": True,
                          "held_label_cost_perturbation_cannot_change_selection": True,
                          "original_files_unchanged": len(original_hashes), "old_artifact_audit": old_audit},
                "input_sha256": INPUTS, "all_original_sha256": original_hashes,
                "output_sha256": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    write_json("manifest.json", manifest)
    print(json.dumps(manifest["audit"], ensure_ascii=False, indent=2))


def figures(a_summary, b_snapshots, c_rounds):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(4)
    for key, label in [("generation_seconds", "Generation (incl. initial prefill)"),
                       ("verification_seconds", "Verification"), ("recovery_seconds", "FAIL recovery")]:
        vals = [r["mean_" + key] for r in a_summary]
        ax.bar(POLICIES, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set(ylabel="Mean recorded service cost (s/request)", title="A: all 100 paired questions; policy decision time < 1 ms/request")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a_cost_decomposition.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (group, action) in zip(axes.flat, itertools.product(GROUPS, ["delay1", "delay2"])):
        rows = [r for r in b_snapshots if r["group"] == group and r["comparison"] == action]
        ax.errorbar([r["delta_T_mean"] for r in rows], [r["delta_q_mean"] for r in rows],
                    xerr=[r["delta_T_sd"] for r in rows], yerr=[r["delta_q_sd"] for r in rows],
                    fmt="o", capsize=2, alpha=.65, markersize=4)
        ax.axhline(0, color="grey", linewidth=.8)
        ax.axvline(0, color="grey", linewidth=.8)
        ax.set(title=f"{group}: {action} vs now", xlabel="Delta T = delay - now (s); right: now cheaper",
               ylabel="Delta q = q(now) - q(delay); up: now better")
    fig.suptitle("B: one point per snapshot, mean +/- sample SD over 3 paired seeds (not CI)")
    fig.tight_layout()
    fig.savefig(OUT / "b_paired_effects.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, group in zip(axes, GROUPS):
        for seed in SEEDS:
            rows = [r for r in c_rounds if r["group"] == group and r["evaluation_seed"] == seed]
            ax.plot([r["lambda"] for r in rows], [r["utility_gap"] for r in rows], "o-", label=f"held {seed}")
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set(title=group, xlabel="Cost weight lambda", ylabel="Selected - uniform held-seed utility")
        ax.legend()
    fig.suptitle("C: overlapping rotations, not independent replications")
    fig.tight_layout()
    fig.savefig(OUT / "c_rotation_gaps.png", dpi=180)
    plt.close(fig)


def report(data, config, precheck_rows, details, consistency, loo, coverage):
    def f(x, n=3):
        return f"{x:.{n}f}" if x is not None else "—"

    def ci(values, n=3):
        return f"[{f(values[0], n)}, {f(values[1], n)}]"

    def table(headers, rows):
        return "\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "".join(
            "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |\n" for row in rows) + "\n"

    names = {"initial": "初始", "historical_pass": "历史 PASS", "combined": "合并（按状态等权）"}
    a = {r["policy"]: r for r in data["a_summary"]}
    pair = {(r["policy"], r["reference"], r["metric"]): r for r in data["a_pairs"]}
    ep, k4 = a["endpoint"], a["k4"]
    b = {(r["group"], r["comparison"]): r for r in data["b_summary"] if r["weighting"] == "snapshot"}
    c = {(r["group"], r["evaluation_seed"], r["lambda"]): r for r in data["c_rounds"]}
    saving = ep["mean_service_seconds"] - k4["mean_service_seconds"]
    lines = ["# 实验四进一步分析（已有数据，2026-09-27）\n",
             "本报告由同目录 `analyze.py` 从原始状态、事件和冻结配置生成；图、表、CSV 均来自实际运行。原数据与原报告保留。\n",
             "## 先读这一页\n",
             f"**① 当前最可靠的发现。** 四 A 的 400 条路径完全复现：unchecked、endpoint、k4、random 分别答对 "
             f"{a['unchecked']['correct']}、{ep['correct']}、{k4['correct']}、{a['random']['correct']}/100，平均服务成本分别为 "
             f"{a['unchecked']['mean_service_seconds']:.2f}、{ep['mean_service_seconds']:.2f}、{k4['mean_service_seconds']:.2f}、{a['random']['mean_service_seconds']:.2f} 秒。"
             f"k4 比 endpoint 少 {saving:.3f} 秒，几乎全部来自生成计时；生成量少 {ep['mean_generated_tokens']-k4['mean_generated_tokens']:.2f} token/题，"
             f"但回滚次数为 {k4['total_rollbacks']} 对 {ep['total_rollbacks']}，不是更少回滚。四 B 的三种子配对均值显示状态组有不同方向："
             f"初始组 delay1 比 now 便宜 {-b['initial','delay1']['delta_T']:.3f} 秒，历史组 now 比 delay1 便宜 {b['historical_pass','delay1']['delta_T']:.3f} 秒。"
             "这是固定历史下的后续净效果，质量差仍须同时看。\n",
             "**② 原结论需要收窄。** 不能把服务秒数下降解释为质量匹配后的收益或含排队完成时间的下降。"
             "旧基线与新策略的每 token 实测时间也不同，设备状态和执行负载尚未排除。历史组原评价种子 43 的正向效用未跨种子稳定保留："
             "评价 17 时所有正 λ 为负；评价 29 时只有 λ=1 为正。原轮次多答对的一题是 test-210；删除它仍有极小的计时收益，不能概括成所有收益都由它构成。"
             "状态差异足以支持小规模补采样，但不足以确认可学习、可部署的自适应收益，更不是可靠上界。\n",
             "**③ 下一步最值得做的一项实验。** 固定现有 32 个快照、4 个动作，预登记 3 个新的配对种子，共 **384 条新增后续路径**。"
             "冻结原 17/29 选出的逐快照动作和组内统一参考，评价全部六个 λ，同时保留 now/delay1/delay2 的 ΔT、Δq。"
             "到 384 条即停止，不为追求正结果加种子；把“所有预定正 λ 在三个新种子上方向均为正、汇总质量差不负、逐题删除后方向仍正”作为"
             "进入独立题集研究的严格筛选条件（不是总体质量非劣证明）。若不满足，保留不稳定结论，暂不训练探针。本轮没有执行这些路径。\n",
             "## 来源、配置和复现核对\n",
             "数据位置为 `stages/stage1/runs/7bf883613b3ff02a`，父运行是 `stages/stage1/runs/3660363ff392fea6`。"
             "读取了两级 README、实验方案、原报告、配置、源代码归档，以及用户列出的全部九类文件。"
             "配置中的代码哈希、源归档内容、引用输入哈希、复用轨迹哈希均核对通过；未调用会改写原报告的原 audit 入口。"
             "原 `analyze_oracle` 和答案评分函数通过 AST 只加载纯函数进行复现，没有导入模型执行器。\n"]
    lines.append(table(["项目", "核对结果"], [
        ["四 A", "100 道相同题 × 4 策略；unchecked/endpoint 复用父运行，k4/random 为新运行"],
        ["四 B", "20 initial + 12 historical_pass；历史组题目是前 20 题的子集，共 20 道题"],
        ["分支", "32 状态 × 4 动作 × 3 种子 = 384；历史、预算、生成计数、配对种子和后续成本核对通过"],
        ["新增状态", "原运行 40 开发 + 200 完整评估 + 40 历史 + 384 分支 = 664"],
        ["生成器", config["assets"]["generator"]["id"] + "，revision " + config["assets"]["generator"]["revision"]],
        ["验证器", config["assets"]["verifier"]["id"] + "，revision " + config["assets"]["verifier"]["revision"]],
        ["采样/预算", "temperature=0.7，top_p=0.95；生成 2048 token、6 次验证、prefill 16384 token、上下文 2048 token"],
        ["验证配置", "PRM NF4/BF16，阈值 0.35；输出 JSON/反馈是确定性适配器，不是验证模型生成文本"],
        ["归一化常数", f"冻结配置精确值 {config['cost_unit_seconds']} 秒；原报告显示为 7.748 秒，本报告沿用精确值"],
        ["舍入核对", "另以 7.748 重算，36 个组×轮次×λ 的动作和参考选择均相同；未调参"],
        ["原 C 汇总", "纯函数输出与原 summary.json 完全相同；CSV、held_seed_action_pairs.json 亦核对通过"],
    ]))
    lines.append("三个编号 17/29/43 是 replicate 标签，实际种子为 "
                 "`int(SHA256(canonical_json((20260922, 'oracle/{snapshot_id}/{replicate}')))[:12],16)`。"
                 "同一快照/replicate 的四动作使用相同未来种子；不同动作消费随机数的进度可能不同，不代表逐 token 反事实完全一致。"
                 "96 组实际映射见 [seed_mapping.csv](seed_mapping.csv)。\n")
    lines.append(table(["缺失历史快照的题", "终止原因"], [[r["question_id"], r["reason"]] for r in coverage if not r["available"]]))
    lines.append("历史组按能形成自然 PASS 后状态而条件选择，不能外推到所有题，也不能把两组差异直接解释为 PASS 历史的因果作用。\n")
    lines.append("## 四 A：固定检查为什么观测上更便宜\n")
    lines.append("主表始终保留全部 100 道题；未完成计为不正确。服务成本定义严格复用代码："
                 "`generator_seconds + verifier_seconds - offline_restore_seconds + policy_decision_seconds`。"
                 "以下类别互不重复，另保留很小的策略决策成本以使账目严格闭合。\n")
    lines.append(table(["策略", "对/错/未完成", "生成秒/题", "验证秒/题", "恢复秒/题", "决策秒/题", "服务秒/题", "扣除离线恢复秒/题"], [
        [p, f"{r['correct']}/{100-r['correct']-r['unfinished']}/{r['unfinished']}", f(r["mean_generation_seconds"]),
         f(r["mean_verification_seconds"]), f(r["mean_recovery_seconds"]), f(r["mean_decision_seconds"], 6),
         f(r["mean_service_seconds"]), f(r["mean_offline_restore_seconds"])] for p, r in a.items()]))
    lines.append("生成＝全部 `generate.decode_seconds`（首次及后续返工）＋`INITIAL.prefill_seconds`。"
                 "恢复＝`FAIL_RESTORE.prefill_seconds`：当前 B 实现每次 FAIL 后对“已接受前缀＋反馈”完整 prefill，"
                 "这是原算法恢复方式实际执行并计费的成本。它不是理论最小恢复成本，不能把整段前缀重算说成任何 KV 裁剪实现都必须支付。"
                 "记录没有把保留前缀和反馈输入的耗时单独计时，因此恢复内部不能再按 token 比例分成实测秒数。"
                 "`OFFLINE_RESTORE` 是本机卸载/波次暂停后对已付费前缀的恢复，包含快照续接、PASS/UNCERTAIN 后续接，单列后扣除。\n")
    lines.append("`decode_seconds` 的同步计时区间含采样、token 解码、停止条件检查等循环开销，并非纯 GPU kernel 时间；"
                 "PRM `verify_call.seconds` 主要覆盖张量构造、前向和分数计算，未覆盖全部输入分词、适配器解析、反馈分词、落盘。"
                 "CPU 回滚列表复制也无独立计时。这里的“服务计算成本”是经过核对的原服务计时口径，不宣称记录了全部端到端执行成本。\n")
    lines.append("![四 A 成本分解](a_cost_decomposition.png)\n")
    lines.append(table(["策略", "检查总数（中途/终点）", "验证输入/输出 token", "单次验证秒", "输入 token/次", "生成 token", "回滚次数", "丢弃生成 token"], [
        [p, f"{r['total_checks']}（{r['total_intermediate_checks']}/{r['total_endpoint_checks']}）",
         f"{r['total_verifier_input_tokens']}/{r['total_verifier_output_tokens']}", f(r["seconds_per_verification"]),
         f(r["input_tokens_per_verification"], 1), r["total_generated_tokens"], r["total_rollbacks"], r["total_discarded_generated_tokens"]] for p, r in a.items()]))
    lines.append("验证输出 token 全部为 0 是 PRM 的实际实现：只有前向评分，JSON 是适配器产物，不能把 JSON 长度当验证生成 token。"
                 "丢弃生成 token 通过事件逐位置追踪，与每次 FAIL 的保存状态及最终保留长度交叉验证；包括回滚时丢弃的 pending look-ahead token，"
                 "排除 prompt 和反馈 token。它已经属于生成量，绝不另加一份成本。没有逐 token 耗时，不能准确给丢弃部分分配秒数。\n")
    lines.append(table(["策略", "首次回滚前 decode 秒/题", "首次回滚后 decode 秒/题", "首次回滚前/后 token/题", "decode 毫秒/token（总和之比）"], [
        [p, f(r["mean_before_first_rollback_decode_seconds"]), f(r["mean_after_first_rollback_decode_seconds"]),
         f"{f(r['mean_before_first_rollback_generated_tokens'],2)}/{f(r['mean_after_first_rollback_generated_tokens'],2)}",
         f(1000*r["decode_seconds_per_token"])] for p, r in a.items()]))
    lines.append("“首次回滚后”是可重复的返工阶段指标，包括返工后的正常继续生成，不等于所有 token 都被浪费。未回滚路径全部归入首次回滚前。\n")
    lines.append(table(["k4 − endpoint 配对指标", "均值差", "95% 题目配对区间", "负/零/正的题数"], [
        [label, f(r["mean"]), ci(r["ci"]), f"{r['negative_pairs']}/{r['zero_pairs']}/{r['positive_pairs']}"]
        for key, label in [("correct", "正确率"), ("service_seconds", "服务秒"), ("generated_tokens", "生成 token"),
                           ("after_first_rollback_generated_tokens", "首次回滚后 token"), ("discarded_generated_tokens", "丢弃 token"), ("rollbacks", "回滚次数")]
        for r in [pair["k4", "endpoint", key]]]))
    lines.append(f"在计时账目上，生成项减少 {ep['mean_generation_seconds']-k4['mean_generation_seconds']:.3f} 秒/题，"
                 f"验证项反而增加 {k4['mean_verification_seconds']-ep['mean_verification_seconds']:.6f} 秒/题，"
                 f"恢复减少 {ep['mean_recovery_seconds']-k4['mean_recovery_seconds']:.6f} 秒/题。"
                 f"k4 单次验证约便宜 {100*(1-k4['seconds_per_verification']/ep['seconds_per_verification']):.1f}%，"
                 f"但检查多 {k4['total_checks']-ep['total_checks']} 次，故验证总成本几乎没降。"
                 f"生成量少 {100*(1-k4['total_generated_tokens']/ep['total_generated_tokens']):.1f}%，"
                 f"首次回滚后少 {ep['mean_after_first_rollback_generated_tokens']-k4['mean_after_first_rollback_generated_tokens']:.2f} token/题，"
                 f"丢弃少 {ep['mean_discarded_generated_tokens']-k4['mean_discarded_generated_tokens']:.2f} token/题；"
                 "支持“少生成、返工生成较短”的描述，不支持“少回滚”。返工量和丢弃量差的题目区间含零，证据强度不应混同于总生成量差。\n")
    lines.append(f"**测时混杂仍明显。** 生成 decode 总和/token 总和在 endpoint 为 {ep['decode_seconds_per_token']*1000:.3f} ms，"
                 f"k4 为 {k4['decode_seconds_per_token']*1000:.3f} ms。这个比率同时受设备状态、前缀长度、步骤和 CPU 工作影响，"
                 "不是经控制的吞吐基准。旧策略和新策略跨运行执行，没有交错测时/重复测量；不能把约 4.728 秒全部归因于检查算法，"
                 "也不能按 token 比例反推“硬件贡献”的实测秒数。\n")
    reasons = sorted({r["termination"] for r in data["terminations"]})
    term = {(r["policy"], r["termination"]): r["count"] for r in data["terminations"]}
    lines.append(table(["结束原因"] + POLICIES, [[t] + [term.get((p,t), 0) for p in POLICIES] for t in reasons]))
    lines.append("k4 的检查预算耗尽为 14 题，endpoint 为 9 题；最终有答案的路径分别为 76、75。"
                 "因此较早预算停止可能贡献部分少生成，不能把所有缩短都当纠错收益；现有记录能识别终止类型，"
                 "不能在已发生的策略分叉之后隔离“纯早停”反事实。FINAL_PASS 仍可能答错。"
                 "全部逐题质量转换、结束原因和分项差见 [a_paired_questions.csv](a_paired_questions.csv)，没有只保留答对题来声称质量匹配。\n")
    lines.append(table(["参照比较", "正确率差", "95% 配对区间", "平均服务秒差"], [
        [p+" − "+ref, f(pair[p,ref,"correct"]["mean"]), ci(pair[p,ref,"correct"]["ci"]), f(pair[p,ref,"service_seconds"]["mean"])]
        for p, ref in [("k4","unchecked"),("k4","endpoint"),("random","endpoint"),("random","k4")]]))
    phase = data["phase_audit"][0]
    lines.append(table(["实验四新运行波次", "墙钟秒", "新执行模型计时秒（含离线恢复）", "未归属残差秒"], [
        [role, f(phase[role+"_wave_seconds"]), f(phase[role+"_recorded_compute_seconds"]), f(phase[role+"_unattributed_wave_residual_seconds"])]
        for role in ["generator", "verifier"]]))
    lines.append(f"波次合计 {phase['total_wave_seconds']/3600:.3f} 小时。残差按新执行事件求和（分支先去掉继承历史）后相减，"
                 "包含装卸模型、分词、CPU 调度、保存状态等，不能精确拆成模型加载。波次日志没有逐请求时间戳/到达时间，"
                 "不能把它分摊成用户延迟。父运行波次还包含其他实验，见 summary.json，未伪造四 A 旧路径的单独加载成本。\n")
    lines.append("## 四 B：现在与延后检查的局部净效果\n")
    lines.append("定义 `ΔT=T(delay)−T(now)`、`Δq=q(now)−q(delay)`，两个正方向都表示 now 更好。"
                 "每条后续成本先从完整成本减同一个快照的已支付历史；逐快照先做同 seed 配对，再取三个 replicate 的平均。"
                 "now/delay1/delay2 首次检查后都回 k4；endpoint_only 始终 endpoint，单独列为不同后续策略比较。"
                 "以下区间只重采样题目，固定已有三个种子；SD 是三个配对差的样本标准差（ddof=1），不是置信区间。\n")
    lines.append(table(["组", "延后动作", "状态/题", "ΔT 秒", "95% 题目区间", "Δq", "95% 题目区间", "平均组内 seed SD（T/q）"], [
        [names[r["group"]], r["comparison"], f"{r['snapshots']}/{r['questions']}", f(r["delta_T"]), ci(r["delta_T_question_cluster_ci"]),
         f(r["delta_q"]), ci(r["delta_q_question_cluster_ci"]), f"{f(r['delta_T_mean_within_snapshot_seed_sd'])}/{f(r['delta_q_mean_within_snapshot_seed_sd'])}"]
        for r in data["b_summary"] if r["group"] in GROUPS and r["comparison"] != "endpoint_only"]))
    lines.append(table(["组", "延后动作", "seed17：ΔT/Δq", "seed29：ΔT/Δq", "seed43：ΔT/Δq"], [
        [names[g], action] + [f"{f(r['delta_T'])}/{f(r['delta_q'])}" for seed in SEEDS for r in data["b_seed_summary"]
                             if r["group"] == g and r["comparison"] == action and r["replicate"] == seed]
        for g, action in itertools.product(GROUPS,["delay1","delay2"]) ]))
    lines.append("初始组总体更支持延后，历史组在成本上更支持现在检查；但历史组质量差的区间均跨零。"
                 "这是值得继续检验的组别差异，不能证明单个快照可预测出较优动作。三个种子内的成本 SD 常与均值差同量级；"
                 "它混合生成路径随机性与实际测时变化，没有独立重复计时来区分二者。\n")
    lines.append("![四 B 逐快照配对效果](b_paired_effects.png)\n"
                 "横轴向右表示马上检查更省成本，纵轴向上表示马上检查质量更好。误差线为三个配对种子的 ±1 SD；"
                 "所有快照都绘出，包括差为零者，重叠点不代表缺失。完整 64 个主要比较的三个种子、均值、SD 和范围见 "
                 "[逐快照中文附表](b_snapshot_report.md)；全部逐种子数值见 [b_paired_seeds.csv](b_paired_seeds.csv)。\n")
    lines.append(table(["不同后续策略，仅描述", "ΔT 秒", "Δq", "成本区间", "质量区间"], [
        [names[g]+"：endpoint_only vs now", f(r["delta_T"]), f(r["delta_q"]), ci(r["delta_T_question_cluster_ci"]), ci(r["delta_q_question_cluster_ci"])]
        for g in GROUPS for r in [b[g,"endpoint_only"]]]))
    lines.append("endpoint_only 不能把全部差异归因于当前检查时机。历史组此比较的三种子均值质量区间为 [0,0]，"
                 "其逐种子质量差仍有波动；这是固定种子平均后题目差为零的退化现象，不是种子随机性为零。\n")
    lines.append(table(["合并两组的权重", "延后动作", "ΔT 秒", "按题聚类区间", "Δq", "按题聚类区间"], [
        ["状态等权" if r["weighting"] == "snapshot" else "题目等权", r["comparison"], f(r["delta_T"]),
         ci(r["delta_T_question_cluster_ci"]), f(r["delta_q"]), ci(r["delta_q_question_cluster_ci"])]
        for r in data["b_summary"] if r["group"] == "combined" and r["comparison"] != "endpoint_only"]))
    lines.append("合并只是补充：状态等权会让拥有两类快照的题权重更大；题目等权先平均同题两种状态。"
                 "两者 bootstrap 都抽 20 道题，抽中某题时带上该题所有相关快照及已配对的三种子，不把 32 状态当独立题。\n")
    lines.append("### 首次检查前新增生成的回滚浪费\n")
    lines.append(table(["组", "动作", "分支数/实际首次检查/首次FAIL", "检查前新增 token/分支", "其中首次FAIL丢弃", "其中最终曾被回滚丢弃"], [
        [names[r["group"]], r["action"], f"{r['branches']}/{r['first_check_exists']}/{r['first_FAIL']}",
         f(r["mean_precheck_new_generated_tokens"], 2), f(r["mean_precheck_new_first_fail_discarded_tokens"], 2),
         f(r["mean_precheck_new_eventually_discarded_tokens"], 2)] for r in data["precheck_summary"]]))
    lines.append("分母保留所有分支；endpoint_only 有分支在任何检查前终止，其“检查前”窗口截止终止，CSV 用 first_check_exists 标识。"
                 "新增量不含快照中已有生成（包括已有 pending token）；追踪以后任何真实 rollback 是否丢弃这些新 token。"
                 "now 的窗口没有新生成，故该指标为 0，并不说明 now 没有后续返工。"
                 "延后时确实观察到新增生成被丢弃，但被验证器拒绝的内容可能正确；"
                 "该量只是错误传播/回滚浪费的观测代理，不能认作真实错误传播的完整因果收益，也不是已识别的 G。\n")
    lines.append("## 四 C：跨种子选择与评价\n")
    lines.append("保留原轮次 17/29→43，另做 29/43→17、17/43→29。每轮逐快照动作和组内统一参考都只由两个选择种子产生；"
                 "评价种子没有参与选择。效用为 `q − λT/7.748…`，全用冻结配置精确常数。"
                 "`max` 遇到精确并列按原顺序 now、delay1、delay2、endpoint_only 取第一个，没有容差并列或新 tie-break。"
                 "这些是“两次选择样本选出的动作”，不是真实最优动作。它们看过离线答案标签，仍不是部署策略。\n")
    for group in GROUPS:
        lines.append(f"### {names[group]}组：全部三轮、六个 λ\n")
        lines.append(table(["选→评", "λ", "逐状态正确/参考正确", "统一参考", "逐状态秒/参考秒", "效用差", "条件95%区间", "动作数 now/d1/d2/end"], [
            ["/".join(map(str,r["selection_seeds"]))+"→"+str(r["evaluation_seed"]), r["lambda"],
             f"{r['selected_correct']}/{r['uniform_correct']}（分母{r['count']}）", r["uniform_action"],
             f"{f(r['selected_mean_seconds'])}/{f(r['uniform_mean_seconds'])}", f(r["utility_gap"], 5), ci(r["conditional_bootstrap_ci"], 5),
             "/".join(str(r["action_counts"].get(a,0)) for a in ACTIONS)] for r in data["c_rounds"] if r["group"] == group]))
    lines.append("![三轮效用差](c_rotation_gaps.png)\n")
    lines.append("原轮次历史组五个正 λ 选出相同动作，均多答对 1 题；新轮换表明收益方向不稳定。"
                 "初始组评价 29 在 λ=0.05、0.1、0.2 有很小的正效用，其质量数与统一参考相同、条件区间跨零；"
                 "其余初始轮次/权重无正向结果。因此原报告“初始无收益”应限定为原 43 评价轮次，"
                 "不能扩展为所有种子的数学结论。历史组评价 17 的所有正 λ 为负，评价 29 仅 λ=1 为正，且其区间仍跨零。"
                 "没有依据挑选这个 λ/种子宣布成功；三轮共享全部分支，不能称三次独立复现。\n")
    lines.append(table(["组", "λ", "三轮动作完全一致/状态数", "平均两两一致率", "选择并列状态数（评17/29/43）"], [
        [names[g], w, f"{sum(r['all_equal'] for r in rs)}/{len(rs)}", f(mean(rs,"pairwise_agreement")),
         "/".join(str(c[g,seed,w]["tied_snapshots"]) for seed in SEEDS)]
        for g,w in itertools.product(GROUPS,WEIGHTS)
        for rs in [[r for r in consistency if r["group"] == g and r["lambda"] == w]]]))
    lines.append("两个选择种子只有 0、0.5、1 三档正确率估计，因此 λ=0 的最优估计经常并列，"
                 "精确并列经原动作顺序处理会偏向 now。正 λ 下没有精确并列，只说明实测时间的小数打破并列，"
                 "不代表动作已显著可区分；可能恰恰把测时噪声变成选择依据。"
                 "全部动作估计与并列集合保存在 c_snapshot_selections.csv。\n")
    lines.append(table(["组", "动作", "跨3 seed 正确性变化的状态/总数", "同状态成本 seed SD 均值（秒）"], [
        [names[r["group"]],r["action"],f"{r['snapshots_with_quality_seed_disagreement']}/{r['snapshots']}",f(r["mean_within_snapshot_cost_seed_sd"])]
        for r in data["noise"]]))
    lines.append("组别方向差与一些稳定正确/错误状态说明并非完全没有状态信号；但是具体动作的三轮一致率和收益方向较弱。"
                 "当前状态相关信号尚不足以稳定超越生成随机性及测时扰动，不能据此估算可靠的自适应上界。\n")
    lines.append("### test-210 与逐题删除：敏感性分析，不替换主结果\n")
    lines.append(table(["评价seed", "λ", "test-210动作/参考", "其正确差", "其效用差（未平均）", "全组效用差", "删210固定选择", "删210重选参考"], [
        [seed,w,d["chosen_action"]+"/"+d["uniform_action"],d["quality_gap"],f(d["utility_gap"],5),
         f(r["full_utility_gap"],5),f(r["fixed_choices_gap"],5),f(r["reselected_reference_gap"],5)]
        for seed,w in itertools.product(SEEDS,WEIGHTS)
        for d in [next(x for x in details if x["group"] == "historical_pass" and x["evaluation_seed"] == seed and x["lambda"] == w and x["question_id"] == "test-210")]
        for r in [next(x for x in loo if x["group"] == "historical_pass" and x["evaluation_seed"] == seed and x["lambda"] == w and x["omitted_question"] == "test-210")]]))
    lines.append("原 43 评价轮次正 λ 的**质量净增全部来自 test-210**。但删去它后其余 11 状态有微小成本节省，效用仍正："
                 "不能声称整个效用严格只由一题产生。轮换到评价 29、λ=1，多答对的一题变为 **test-916**，test-210 的直接差为零；"
                 "不过删除 test-210 会影响选择种子中的组内统一参考，从 delay1 变为 now，重选参考后效用转负。"
                 "这说明应分别讨论“直接贡献”与“影响参考动作选择”，不能简单问所有轮次是否由同一题支撑。\n")
    lines.append("逐题删除做两种口径：①保持完整样本已选动作与参考不变，仅删除评价贡献；②删除整题后仅用原两个选择种子重新选组内参考，"
                 "逐状态动作依然只依赖自己的两个选择样本。两种都没有偷看评价种子；合并组时同题所有状态一起删。"
                 "“方向改变”包括零↔非零，“反号”只计正↔负。完整 936 个删除结果见 CSV。\n")
    # Show all group/round/lambda cells, including cells with no influential deletion.
    lines.append(table(["组", "评seed", "λ", "全组差", "固定选择方向改变/反号数", "重选参考方向改变/反号数", "重选参考删除后范围"], [
        [names[r["group"]],r["evaluation_seed"],r["lambda"],f(r["full_utility_gap"],5),
         f"{len(r['fixed_direction_changed'])}/{len(r['fixed_sign_reversed'])}",
         f"{len(r['reselected_direction_changed'])}/{len(r['reselected_sign_reversed'])}",
         ci([r["reselected_gap_min"],r["reselected_gap_max"]],5)] for r in data["loo_summary"] if r["group"] in GROUPS]))
    lines.append("原 43 历史组删除任意一道题均未改变正 λ 的方向，但极小计时余量不能排除测时误差。"
                 "评价 17 删除 test-979 会让正 λ 从负转正；评价 29 多个删除会改变参考及方向。"
                 "因此‘对单题删除稳定’和‘对生成种子稳定’是不同命题。合并结果与题目等权/状态等权聚类区间见 "
                 "c_combined_cluster.csv，删除的题名见 c_loo_summary.csv；不把合并表作为新的优选主结论。\n")
    lines.append("## 统计区间到底覆盖什么\n")
    lines.append("原四 A 在固定 100 题顺序上形成同题差，以 `random.Random(20260922).choices` 抽取 100 个差值，"
                 "重复 2000 次；排序后取下标 50 和 1949。配对没有破坏。原四 C 在各组内先固定从 17/29 选出的动作与参考，"
                 "再对 43 评价的逐快照效用差做同样抽样；**bootstrap 内没有重新选动作、没有重采样种子、没有重新测时**。"
                 "每组内各题只有一个快照，因此组内快照抽样等同题目抽样；原实现没有合并两组的独立快照 bootstrap。"
                 "本次合并分析改为题目聚类抽样，保留关联状态，计算方式已在上文写明。\n")
    lines.append(table(["不确定性", "现有区间是否覆盖", "边界"], [
        ["题目构成", "覆盖已有题目经验分布下的抽样变化", "固定题集的非参数近似；历史组仅限已形成快照的条件总体，无法处理缺失选择偏差"],
        ["生成随机性", "不完整覆盖", "三种子均值及 SD/轮换仅作敏感性描述；区间固定这些实际种子"],
        ["动作选择误差", "原区间不覆盖", "选择动作/统一参考在每个区间内固定；轮换和重选参考 LOO 只显示部分不稳定性"],
        ["测时误差、设备漂移", "不覆盖", "没有对同一轨迹在受控设备状态下重复计时；成本区间只是已观测题目成本的抽样"],
        ["多重比较/调参", "未提供同时保证", "六 λ、多个动作和组共享样本；全部公开，不按区间正负筛选结论"],
    ]))
    historical0 = [r for r in details if r["group"] == "historical_pass" and r["evaluation_seed"] == 43 and r["lambda"] == 0]
    positive = [r for r in details if r["group"] == "historical_pass" and r["evaluation_seed"] == 43 and r["lambda"] == .1]
    assert all(r["utility_gap"] == 0 for r in historical0)
    zero_n = sum(r["utility_gap"] == 0 for r in positive)
    positive_n = sum(r["utility_gap"] > 0 for r in positive)
    negative_n = sum(r["utility_gap"] < 0 for r in positive)
    lines.append(f"**历史组 [0,0] 的具体来源。** 原轮次 λ=0 时 {len(historical0)} 个状态都选 now，统一参考也为 now，"
                 "所以输入 bootstrap 的每个差恰好等于零，任何重采样都为零。它不证明算法等价、真实总体无收益或中间步骤无错。\n")
    lines.append(f"**略高于零的区间。** 原轮次 λ=0.1 的 12 个差中，{positive_n} 正、{zero_n} 零、{negative_n} 负。"
                 f"其中 test-210 有质量净增，其余正差只是 5 个状态的小量计时节省。抽样完全不含 test-210 的理论概率为 "
                 f"(11/12)^12={100*(11/12)**12:.1f}%，仍可能抽到这些小正值；全部抽到零差的概率为 (6/12)^12={100*(6/12)**12:.4f}%。"
                 "因此这种固定观察值的重采样自然会产生略高于零的第 2.5 百分位数。它无法检验这些微小时间差能否在重复测时下保留，"
                 "也不覆盖新种子的动作选择误差。不能将其解读为普遍的自适应效果显著为正。\n")
    lines.append("## 与研究建模对齐\n")
    lines.append(table(["目标量", "可直接复用的字段", "现在能说什么 / 缺什么"], [
        ["C_ver", "verify_call.seconds/input_tokens/output_tokens/retry/raw；checks；verifier_seconds", "可估本机该 PRM 对实际输入的验证成本，含历史前缀长度效应；输出为0，输入分词和装卸未完整纳入。不可直接迁移到批处理/其他设备"],
        ["回滚浪费 / G 代理", "generate.kept_tokens/pending_tokens、pending_merged、checkpoint、rollback、fail_snapshots.context_ids/checkpoint_ids/pending_ids", "可准确数被丢弃的生成 token 和首次检查前新增且丢弃的 token；无逐 token 秒数、可信错标签，不能估真实错误的全部因果损失"],
        ["后续净效应 ΔT", "完整服务成本减 inherited_service_seconds（或同快照历史重算）", "同时含验证次数/长度、不同续写、返工、恢复、预算终止和质量差；是整个干预路径的净效果，不能直接命名 G"],
        ["中间错误标签", "PRM step_scores/first_error_step、PASS/FAIL；最终 gold correctness", "有验证器预测和答案标签，尚无这些快照的可信步骤真值。父实验 ProcessBench 诊断标签属于另一个样本集，不能转贴给本实验快照"],
        ["C_sys", "仅有单路径顺序和离线波次总时", "无到达/排队/资源占用/并行重叠/同机干扰数据，当前不能测量一个请求检查给其他请求造成的完成时间外部性"],
    ]))
    lines.append("例如延后检查既可能多写并丢弃 token，也可能少做几次验证、偶然走到更短续写、或更早耗尽预算。"
                 "所有这些共同进入 ΔT；若研究式中的 G 指“及时查出真实错误后避免的未来浪费”，必须明确错误真值、比较起点、"
                 "共同后续策略与反事实，再扣分项。现有 ΔT、Δq 和浪费 token 都应保留各自名称。"
                 "**服务计算成本下降与包含排队的用户完成时间下降不是同一个结果**：增加一个验证任务可能阻塞其他请求，"
                 "也可能释放后续生成资源，单请求成本表无法判定其 C_sys。\n")
    lines.append("### 最小多请求回放方案（本轮只准备数据，不运行调度仿真）\n")
    lines.append("已有 `action_id` 给出完整的路径内事件顺序，generate 有整段 prefill/decode 秒数，verify_call 有秒数；"
                 "已导出 [ordered_phases.csv](ordered_phases.csv)，A 是完整路径，B 只保留快照后的已执行后缀。"
                 "阶段顺序足以做粗粒度固定轨迹回放，但没有每个 token/每个完整步骤的耗时，也没有完整请求时间线；未计时的 CPU 阶段不能填成实测 0。\n")
    lines.append("最小回放先只用四 A 100 题各策略的完整、已观测轨迹：每题为一个请求，四种策略分别回放同一到达序列，"
                 "保持该路径生成→验证→恢复的依赖顺序，阶段不可抢占。先声明单个串行资源、模型可无代价切换的理想假设，"
                 "服务量按原口径扣 OFFLINE_RESTORE；这只是队列机制演示，实际 RTX 4060 的装卸开销和内存限制未模拟。"
                 "给定两条明示为合成的到达序列：100 请求在时刻0到达的 burst，以及间隔取所有策略平均服务时间的最小值 1.25 倍的低负载序列；"
                 "同一序列用于全部策略。比较阶段 FCFS 和验证优先，两者均不改变已经记录的检查决策、答案与后续生成；"
                 "报告等待时间、完成时间和 makespan 为假设条件下的模拟值，不称为测得 C_sys。输入路径数为400，新增模型路径数为0。\n")
    lines.append("B 若用于局部回放，只能把一个已记录快照当作新的释放点，每次选择现存四动作中的一条完整后缀，"
                 "并以题为单位避免把同题多个相关状态误当独立新请求。绝不在某条分支中途改动作并拼接另一个状态的后续轨迹。"
                 "模拟无法重现 batching、KV/显存争用、模型切换、内核并发、GPU频率/温度变化及调度导致的生成随机数或数值变化。"
                 "进入系统实验前还需记录每阶段 ready/start/end、请求 arrival/finish、模型驻留/切换和资源指标。\n")
    missing = [
        {"item": "恢复内部拆分", "available": "FAIL_RESTORE 的整次 prefill 秒与长度、反馈插入长度", "missing": "反馈单独prefill、保留前缀重算的分开计时；CPU rollback 时间", "minimum": "补阶段打点；本轮不按长度分摊秒数"},
        {"item": "浪费生成秒数", "available": "token 精确来源/去向、整段 decode 秒", "missing": "逐 token 或步骤耗时", "minimum": "仅报告浪费 token；未来记录步骤边界时间，不比例造数"},
        {"item": "跨运行测时混杂", "available": "旧/新配置同源及阶段计时", "missing": "随机交错、同轨迹重复测时、设备状态", "minimum": "可选400条完整路径的受控配对复测，详见下文"},
        {"item": "动作泛化", "available": "32快照×4动作×3种子", "missing": "更多独立评价种子、未见题目/状态", "minimum": "优先384条冻结快照续写；仍不能证明未见题泛化"},
        {"item": "错误真值", "available": "最终答案分数、PRM预测", "missing": "这些快照的步骤错误人工核对标签", "minimum": "32快照双人盲标，64份标注；0新增模型路径"},
        {"item": "C_sys/真实延迟", "available": "路径内顺序、部分阶段服务时长", "missing": "arrival/ready/start/end/finish、资源干扰、并行轨迹", "minimum": "先0新增路径的固定轨迹回放方案；后续受控多请求打点实验"},
    ]
    write_csv("data_gaps.csv", missing)
    lines.append("## 数据缺失与最小补充实验（仅建议）\n")
    lines.append(table(["缺口", "现有部分", "缺失部分", "最小处理"], [[r["item"],r["available"],r["missing"],r["minimum"]] for r in missing]))
    lines.append("1. **优先补种子：384 条。** 32×4×3，三组新的 replicate 可预登记为 59、71、83，"
                 "仍通过同一个 hash 映射到每快照配对种子。每个快照/seed 四动作随机化执行顺序，保存 GPU 状态和完整阶段打点。"
                 "冻结原 17/29 的六套选择结果及其组内参考，三个新 seed 均只评价；不把新种子再用于挑 λ。"
                 "now/delay1/delay2 的配对质量差、成本差为共同主输出；endpoint_only 单列。主要筛选条件见首屏，"
                 "同时公开所有不满足条件的格子和 test-210 删除结果。只在预登记的384条全部完成后判断；若数据缺损先补齐缺损格子，"
                 "不改变样本设计；完成后停止，无论结果正负。通过后才考虑另一个独立题集；失败就维持不稳定结论。"
                 "此规模是判别现有线索能否重复的最小实用补充，不保证能建立质量非劣；用户的可接受质量损失尚未定量，不能自行宣布满足要求。\n")
    lines.append("2. **若优先回答四 A 的测时原因：可选400条，独立于上项。** 100题×endpoint/k4×2次受控重复，"
                 "使用预登记的同题种子、相同设备/配置和预算，按匹配波次随机化两策略顺序，记录每段长度及温度/频率/显存。"
                 "同时公开质量差和成本差，固定400条后停止；两次重复都保持成本下降，且不能由吞吐漂移解释，才加强算法成本下降的解释。"
                 "两次仍不足以给可靠的质量等价结论；若轨迹因数值/随机性不同，不能把重复间差全部归入计时误差。\n")
    lines.append("3. **步骤标签：0条模型路径、64份人工标注。** 32个快照各由两位核对者不看 PRM 和最终分支答案，"
                 "标注快照已有步骤是否有错、首个错误位置与证据；分歧仲裁后停止。PASS 前缀也需检查。"
                 "这只解决32个当前状态的真值，不把之后新生成的步骤自动标真。系统方面先按上一节做0新增路径的回放设计，"
                 "真实并发采样需另定资源配置；本轮没有运行复杂调度仿真。\n")
    lines.append("## 可复跑交付与核验\n\n在项目根目录执行：\n\n```bash\n"
                 ".venv/bin/python stages/stage1/results/diagnostics/exp4_further_20260927/analyze.py\n```\n\n"
                 "仅依赖标准库及项目已有 numpy/matplotlib；无需模型、GPU或网络。脚本会覆盖本进一步分析目录的派生产物，"
                 "不会覆盖 runs 中的数据或原报告。运行中的断言验证原文件内容哈希不变、计时加总闭合、token 守恒、种子/预算一致，"
                 "并精确复现原纯分析函数。manifest.json 记录全部输入与原目录哈希、输出哈希、脚本版本、环境和运行时间。\n")
    files = [
        ("analyze.py", "唯一执行入口；数据审计、分析、绘图、报告生成"),
        ("a_paths.csv", "400条逐路径成本/生成/验证/回滚指标"),
        ("a_paired_questions.csv", "400个同题策略比较，全部题保留"),
        ("a_cost_summary.csv", "四策略互斥成本分解与token汇总"),
        ("a_paired_summary.csv", "同题配对差和区间；a_termination_counts.csv 保存结束原因"),
        ("b_paired_seeds.csv", "288个状态×延后动作×seed配对，含endpoint_only单独标识"),
        ("b_paired_snapshots.csv", "96个状态比较的三seed均值、SD、范围"),
        ("b_snapshot_report.md", "逐快照三个seed的中文阅读附表"),
        ("b_group_summary.csv", "两组及合并后的题目聚类统计"),
        ("b_branches_audited.csv", "384条分支的后续成本分项"),
        ("b_first_check_waste.csv", "逐分支首次检查窗口新增及丢弃量"),
        ("c_rotations.csv", "36个组×seed轮换×λ主结果"),
        ("c_snapshot_selections.csv", "576个逐状态选择、并列估计及评价差"),
        ("c_action_consistency.csv", "192个状态×λ的跨轮动作一致性"),
        ("c_leave_one_question_out.csv", "936个组/合并组的整题删除敏感性"),
        ("c_loo_summary.csv", "54个组×轮次×λ的敏感性汇总及题名"),
        ("c_combined_cluster.csv", "18个合并结果，20题聚类区间"),
        ("seed_mapping.csv", "96组实际配对种子映射"),
        ("ordered_phases.csv", "可供未来固定轨迹回放的有序、已计时阶段"),
        ("rollback_events.csv", "回滚事件及生成token精确丢弃计数"),
        ("reproduced_original.json", "原A/C关键汇总的精确复现"),
        ("data_gaps.csv", "数据缺口与最小处理清单"),
        ("summary.json", "报告数值的机器可读汇总"),
        ("manifest.json", "来源、配置口径、版本、哈希和检查结果"),
    ]
    lines.append(table(["产物", "内容"], [[f"[{name}]({name})",desc] for name,desc in files]))
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Full per-snapshot table, including each seed, its mean, SD and range.
    appendix = ["# 四 B 逐快照配对结果\n", "ΔT=延后−现在（秒），Δq=现在正确−延后正确；正值均有利于现在检查。"
                "每个 seed 单元为 ΔT / Δq；均值与SD按三个配对种子计算。SD不是置信区间。\n"]
    for actions, title in [(["delay1","delay2"], "相同后续k4策略：主要比较"), (["endpoint_only"], "后续策略不同：单独比较")]:
        appendix.append("## " + title + "\n")
        for group in GROUPS:
            appendix.append("### " + names[group] + "\n")
            appendix.append(table(["题", "延后动作"] + [f"seed {s}：T/q" for s in SEEDS] + ["T均值±SD", "T范围", "q均值±SD", "q范围"], [
                [r["question_id"],r["comparison"]] + [f"{f(v['delta_T'])}/{v['delta_q']}" for seed in SEEDS for v in data["b_pairs"]
                  if v["snapshot_id"] == r["snapshot_id"] and v["comparison"] == r["comparison"] and v["replicate"] == seed] +
                [f"{f(r['delta_T_mean'])} ± {f(r['delta_T_sd'])}",ci([r["delta_T_min"],r["delta_T_max"]]),
                 f"{f(r['delta_q_mean'])} ± {f(r['delta_q_sd'])}",ci([r["delta_q_min"],r["delta_q_max"]])]
                for r in data["b_snapshots"] if r["group"] == group and r["comparison"] in actions]))
    (OUT / "b_snapshot_report.md").write_text("\n".join(appendix), encoding="utf-8")


if __name__ == "__main__":
    main()
