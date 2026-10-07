"""Re-evaluate original features with frozen probes; never retrain on held-out data."""
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ..artifacts.io import atomic_json, read_json
from ..probe.fit import evaluate
from ..probe.materials import load_examples
from ..probe.report import plot_metrics, report
from . import analysis_output


def analyze(cfg, run, manifest=None, *, output_dir=None):
    run = Path(run)
    manifest = manifest or read_json(run / "manifest.json")
    output = analysis_output(run, output_dir)
    probes = {}
    for method in ("A", "B", "C"):
        metadata = read_json(run / f"probe_{method}.json")
        if metadata is None:
            continue
        with np.load(run / f"probe_{method}.npz", allow_pickle=False) as weights:
            probes[method] = {**metadata, **{key: weights[key].copy() for key in ("mean", "scale", "w", "b")}}
        p = probes[method]
        if (p.get("method") != method or p["mean"].shape != p["scale"].shape
                or p["mean"].shape != p["w"].shape or not (p["scale"] > 0).all()
                or not all(np.isfinite(p[key]).all() for key in ("mean", "scale", "w", "b"))
                or float(p["b"]) != metadata["b"]):
            raise ValueError(f"Invalid frozen probe {method}")
    if not probes:
        raise ValueError("No frozen probes; fit a new v2 run before evaluation")
    counts = {}
    test = None
    for split in ("train", "test", "smoke"):
        examples, counts[split] = load_examples(run, manifest, split)
        if split == "test":
            test = examples
    with threadpool_limits(limits=1):
        metrics, rows = evaluate(test, probes, cfg, output)
    fit = read_json(run / "fit_status.json", {})
    complete = all(counts[split].get("generated", 0) == cfg["split_sizes"][split] for split in ("train", "test"))
    metrics.update(status=("COMPLETE" if complete else "PARTIAL_COLLECTION") if len(probes) == 3 else "METHOD_FAILURE",
                   counts=counts, selected_layers={m: p["layer"] for m, p in probes.items()},
                   training_coverage=fit.get("coverage", {}), failures=fit.get("failures", {}),
                   analysis_output=str(output))
    atomic_json(output / "metrics.json", metrics)
    plot_metrics(metrics, output)
    report(output, manifest, counts, metrics, rows, metrics["training_coverage"], metrics["failures"], source_run=run)
    return metrics
