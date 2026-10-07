"""Read-only analysis of legacy/v2 inputs; all outputs have independent provenance."""
from __future__ import annotations

from pathlib import Path

from .. import ROOT
from ..artifacts.io import atomic_json, digest, read_json, stable_hash
from ..artifacts.provenance import TYPES, analysis_hashes
from ..artifacts.materials import validate_materials


def experiment_type(manifest):
    value = manifest.get("experiment_type")
    if value in TYPES.values():
        return next(name for name, kind in TYPES.items() if kind == value)
    splits = set(manifest.get("splits", {}))
    if splits == {"train", "test", "smoke"}:
        return "probe"
    if splits == {"pilot", "dev", "test"}:
        return "fixed-step"
    raise ValueError("Unrecognized experiment manifest; no source files were modified")


def analysis_output(run, output=None):
    run = Path(run).resolve()
    sources = analysis_hashes()
    files = [run / name for name in ("manifest.json", "dev_groups.json", "fit_status.json")]
    files += list(run.glob("probe_*.json")) + list(run.glob("probe_*.npz"))
    for folder in ("records", "snapshots", "selections", "arms"):
        files += [p for p in (run / folder).rglob("*") if p.suffix in (".json", ".npz")]
    inputs = {p.relative_to(run).as_posix(): digest(p) for p in sorted(files) if p.is_file()}
    identity = stable_hash({"analysis_sources": sources, "inputs": inputs})
    output = Path(output).resolve() if output is not None else run / "analyses" / identity[:12]
    # Never let --output overwrite an authoritative legacy or v2 execution directory.
    if output == run or (output / "manifest.json").exists():
        raise ValueError("Analysis output must be separate from execution inputs")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "analysis_provenance.json", {
        "schema_version": 2, "analysis_hash": identity, "analysis_sources": sources,
        "source_run": run.name, "input_hashes": inputs,
    })
    return output


def find_run(cfg, experiment):
    stage = "stage2" if experiment == "probe" else "stage3"
    matches = []
    for path in (ROOT / "stages" / stage / "runs").glob("*/manifest.json"):
        manifest = read_json(path)
        if manifest.get("config_hash") == stable_hash(cfg) and experiment_type(manifest) == experiment:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} matching runs; use analyze --run with an explicit path")
    return matches[0]


def analyze_run(run, output=None):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest is missing or invalid")
    validate_materials(manifest)
    kind = experiment_type(manifest)
    if kind == "probe":
        from .probe import analyze
    elif kind == "fixed-step":
        from .fixed_step import analyze
    else:
        from .high_low import analyze
    result = analyze(manifest["config"], run, manifest, output_dir=output)
    print(f"ANALYSIS {run.name}: {result['analysis_output']}", flush=True)
    return result
