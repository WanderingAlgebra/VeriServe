"""Question-weighted fitting, grouped CV and frozen probe evaluation."""
from __future__ import annotations
import json
import time
import warnings
from collections import Counter
import numpy as np
from ..artifacts.io import atomic_json, read_json, save_npz, write_csv, now, stable_hash
from ..artifacts.provenance import environment
from .materials import load_examples
from .report import plot_metrics, report


def method_rows(examples, method, layer):
    xs, ys, weights, ids = [], [], [], []
    for ex in examples:
        if method == "A":
            if not ex["has_end"]:
                continue
            values = [ex["hidden"][layer - 1, -1]]
        elif method == "B":
            values = ex["hidden"][layer - 1, :len(ex["steps"])].tolist()
        else:
            values = [[s["step_number"], s["prefix_token_count"]] for s in ex["steps"]]
        for value in values:
            xs.append(value)
            ys.append(ex["y"])
            weights.append(1 / len(values))
            ids.append(ex["id"])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys), np.asarray(weights), ids


def fit_lr(x, y, weight, cfg):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) != 2:
        raise ValueError("Training fold has a single label class")
    scaler = StandardScaler().fit(x, sample_weight=weight)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        lr = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], solver="lbfgs")
        lr.fit(scaler.transform(x), y, sample_weight=weight)
    return {"mean": scaler.mean_, "scale": scaler.scale_, "w": lr.coef_[0],
            "b": float(lr.intercept_[0]), "iterations": int(lr.n_iter_[0])}


def predict(probe, x):
    from scipy.special import expit
    return expit(((x - probe["mean"]) / probe["scale"]) @ probe["w"] + probe["b"])


def auc(y, scores, weights):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(np.asarray(y)[np.asarray(weights) > 0])) < 2:
        return None
    return float(roc_auc_score(y, scores, sample_weight=weights))


def select_probes(train, cfg, run):
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    train = sorted(train, key=lambda ex: ex["id"])
    if len(train) == 0 or min(Counter(ex["y"] for ex in train).get(c, 0) for c in (0, 1)) < cfg["folds"]:
        raise ValueError("Fewer than five training questions in either class; no test data borrowed")
    folds = list(StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
                 .split(np.zeros(len(train)), [ex["y"] for ex in train]))
    fold_records = []
    for a, b in folds:
        train_ids, val_ids = {train[i]["id"] for i in a}, {train[i]["id"] for i in b}
        assert not train_ids & val_ids
        assert {train[i]["y"] for i in a} == {0, 1} == {train[i]["y"] for i in b}
        fold_records.append({"train_ids": sorted(train_ids), "validation_ids": sorted(val_ids)})
    atomic_json(run / "cv_folds.json", fold_records)
    results, probes, coverage, failures = [], {}, {}, {}
    layers = range(1, train[0]["hidden"].shape[0] + 1)
    with threadpool_limits(limits=1):
        for method in ("A", "B", "C"):
            eligible = [ex for ex in train if ex["has_end"]] if method == "A" else [ex for ex in train if ex["steps"]]
            coverage[method] = {"questions": len(eligible), "correct": sum(ex["y"] == 0 for ex in eligible),
                                "wrong": sum(ex["y"] == 1 for ex in eligible)}
            if min(coverage[method]["correct"], coverage[method]["wrong"]) < cfg["folds"]:
                failures[method] = "Fewer than five applicable questions in either class"
                continue
            best = None
            for layer in (layers if method in {"A", "B"} else [None]):
                scores, error = [], None
                for fold, (a, b) in enumerate(folds):
                    try:
                        xa, ya, wa, ida = method_rows([train[i] for i in a], method, layer)
                        xb, yb, wb, idb = method_rows([train[i] for i in b], method, layer)
                        assert not set(ida) & set(idb)
                        if len(np.unique(yb)) != 2:
                            raise ValueError("Validation fold has one applicable class")
                        probe = fit_lr(xa, ya, wa, cfg)
                        score = auc(yb, predict(probe, xb), wb)
                        scores.append(score)
                        results.append({"method": method, "layer": layer, "fold": fold + 1,
                                        "auroc": score, "error": ""})
                    except (ValueError, Warning) as exc:
                        error = str(exc)
                        results.append({"method": method, "layer": layer, "fold": fold + 1,
                                        "auroc": None, "error": error})
                        break
                if error:
                    # Do not hide non-convergence by selecting another layer.
                    failures[method] = f"Layer {layer}: {error}"
                    best = None
                    break
                mean = float(np.mean(scores))
                results.append({"method": method, "layer": layer, "fold": "mean", "auroc": mean, "error": ""})
                if best is None or mean > best[0]:  # Ascending layers: exact ties retain smaller index.
                    best = (mean, layer)
            if best is not None:
                x, y, w, _ = method_rows(train, method, best[1])
                try:
                    probe = fit_lr(x, y, w, cfg)
                    probe.update(layer=best[1], cv_auroc=best[0], method=method)
                    save_npz(run / f"probe_{method}.npz", mean=probe["mean"], scale=probe["scale"],
                             w=probe["w"], b=np.asarray(probe["b"]))
                    atomic_json(run / f"probe_{method}.json", {k: v for k, v in probe.items() if not isinstance(v, np.ndarray)})
                    probes[method] = probe
                except (ValueError, Warning) as exc:
                    failures[method] = str(exc)
    write_csv(run / "cv_scores.csv", results, ["method", "layer", "fold", "auroc", "error"])
    # This file is saved before reading any test feature or calculating a test prediction.
    atomic_json(run / "fit_status.json", {"selected": {m: p["layer"] for m, p in probes.items()},
                                         "coverage": coverage, "failures": failures,
                                         "frozen_at": now(), "config_hash": stable_hash(cfg)})
    return probes, coverage, failures


def bootstrap_metrics(rows, cfg, paired=True):
    """Resample question IDs, applying multiplicity to all rows of that question."""
    if not rows:
        return {"questions": 0, "correct": 0, "wrong": 0, "positions": 0, "methods": {}, "paired": {}}
    ids = sorted({r["id"] for r in rows})
    index = {uid: i for i, uid in enumerate(ids)}
    groups = np.asarray([index[r["id"]] for r in rows])
    counts = Counter(r["id"] for r in rows)
    weights = np.asarray([1 / counts[r["id"]] for r in rows])
    y = np.asarray([r["label"] for r in rows])
    methods = [m for m in ("A", "B", "C") if all(r.get(m) is not None for r in rows)]
    predictions = {m: np.asarray([r[m] for r in rows]) for m in methods}
    point = {m: auc(y, predictions[m], weights) for m in methods}
    samples = {m: [] for m in methods}
    differences = {f"{m}-{n}": [] for m, n in (("A", "C"), ("B", "C"), ("B", "A"))
                   if m in methods and n in methods and paired}
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(cfg["bootstrap"]):
        multiplicities = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        w = weights * multiplicities[groups]
        if len(np.unique(y[w > 0])) != 2:
            continue
        replicate = {m: auc(y, predictions[m], w) for m in methods}
        for m, value in replicate.items():
            samples[m].append(value)
        for key in differences:
            m, n = key.split("-")
            differences[key].append(replicate[m] - replicate[n])
    def interval(values):
        return np.quantile(values, [0.025, 0.975]).tolist() if values else None
    labels = {r["id"]: r["label"] for r in rows}
    return {"questions": len(ids), "positions": len(rows), "correct": sum(v == 0 for v in labels.values()),
            "wrong": sum(v == 1 for v in labels.values()),
            "methods": {m: {"auroc": point[m], "ci95": interval(samples[m]),
                            "bootstrap_valid": len(samples[m]), "bootstrap_skipped": cfg["bootstrap"] - len(samples[m])}
                        for m in methods},
            "paired": {key: {"difference": point[key[0]] - point[key[2]] if point[key[0]] is not None else None,
                              "ci95": interval(values), "bootstrap_valid": len(values),
                              "bootstrap_skipped": cfg["bootstrap"] - len(values)}
                       for key, values in differences.items()}}


def evaluate(test, probes, cfg, run):
    rows, ends = [], []
    seconds = {m: 0.0 for m in probes}
    common = []
    for ex in test:
        if not ex["steps"]:
            continue
        common.append(ex)
        predictions = {}
        for method, probe in probes.items():
            x = (np.asarray([[s["step_number"], s["prefix_token_count"]] for s in ex["steps"]])
                 if method == "C" else ex["hidden"][probe["layer"] - 1, :len(ex["steps"])])
            start = time.perf_counter()
            predictions[method] = predict(probe, x)
            seconds[method] += time.perf_counter() - start
        for i, step in enumerate(ex["steps"]):
            rows.append({"id": ex["id"], "label": ex["y"], "effective_step": i + 1,
                         "step_number": step["step_number"], "prefix_token_count": step["prefix_token_count"],
                         "near_end": step["near_end"],
                         **{m: float(p[i]) for m, p in predictions.items()}})
    for ex in test:
        if ex["has_end"]:
            ends.append({"id": ex["id"], "label": ex["y"],
                         **{m: float(predict(p, ex["hidden"][p["layer"] - 1, -1:])[0])
                            for m, p in probes.items() if m in {"A", "B"}}})
    write_csv(run / "test_predictions.csv", rows,
              ["id", "label", "effective_step", "step_number", "prefix_token_count", "near_end", *probes])
    write_csv(run / "test_end_predictions.csv", ends, ["id", "label", *[m for m in probes if m != "C"]])
    metrics = {"intermediate_all": bootstrap_metrics(rows, cfg),
               **{f"step_{i}": bootstrap_metrics([r for r in rows if r["effective_step"] == i], cfg)
                  for i in (1, 2, 3)},
               "full_trajectory_end": bootstrap_metrics(ends, cfg),
               "scoring_seconds": seconds,
               "test_common_questions": len(common),
               "test_excluded_from_common": cfg["split_sizes"]["test"] - len(common)}
    plot_metrics(metrics, run)
    return metrics, rows


def fit_only(cfg, run, manifest):
    from threadpoolctl import threadpool_limits
    train, train_counts = load_examples(run, manifest, "train")
    counts = {"train": train_counts}
    try:
        probes, coverage, failures = select_probes(train, cfg, run)
    except (ValueError, Warning) as exc:
        # Save honest NA results without ever loading a generator or borrowing test data.
        _, counts["test"] = load_examples(run, manifest, "test")
        _, counts["smoke"] = load_examples(run, manifest, "smoke")
        metrics = {"status": "INSUFFICIENT_TRAINING", "reason": str(exc), "counts": counts,
                   **{k: bootstrap_metrics([], cfg) for k in
                      ("step_1", "step_2", "step_3", "intermediate_all", "full_trajectory_end")}}
        write_csv(run / "cv_scores.csv", [], ["method", "layer", "fold", "auroc", "error"])
        write_csv(run / "test_predictions.csv", [], ["id", "label", "effective_step", "step_number", "prefix_token_count", "near_end", "A", "B", "C"])
        plot_metrics(metrics, run)
        atomic_json(run / "metrics.json", metrics)
        reason = str(exc)
        if read_json(run / "backup.json", {}).get("status") == "FAILED":
            reason += "; origin backup failed: " + read_json(run / "backup.json")["error"]
        report(run, manifest, counts, reason=reason)
        print(str(exc), flush=True)
        return
    test, counts["test"] = load_examples(run, manifest, "test")
    with threadpool_limits(limits=1):
        metrics, rows = evaluate(test, probes, cfg, run)
    complete = all(counts[split].get("generated", 0) == cfg["split_sizes"][split] for split in ("train", "test"))
    metrics.update(status=("COMPLETE" if complete else "PARTIAL_COLLECTION") if len(probes) == 3 else "METHOD_FAILURE",
                   counts=counts, selected_layers={m: p["layer"] for m, p in probes.items()},
                   training_coverage=coverage, failures=failures, analysis_environment=environment())
    atomic_json(run / "metrics.json", metrics)
    report(run, manifest, counts, metrics, rows, coverage, failures)
    print(json.dumps({"selected": metrics["selected_layers"], "intermediate": metrics["intermediate_all"]}), flush=True)
