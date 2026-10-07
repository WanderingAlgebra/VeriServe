"""Explicit execution/report identities; ordinary SHA-256, never semantic hashing."""
from __future__ import annotations
import importlib.metadata
import os
import platform
import re
import subprocess
from pathlib import Path
from veriserve_research import ROOT
from .io import digest, stable_hash
SHA = re.compile(r"^[0-9a-f]{40}$")
BRANCH = "experiment/step-hidden-probe"

COMMON = (
    "__init__.py", "__main__.py", "trajectory.py", "inference.py",
    "artifacts/__init__.py", "artifacts/io.py", "artifacts/legacy.py",
    "artifacts/provenance.py", "artifacts/materials.py", "artifacts/backup.py",
    "probe/__init__.py", "probe/materials.py", "probe/fit.py",
)
EXECUTION = {
    "probe": ("probe/collect.py",),
    "fixed-step": (
        "artifacts/storage.py", "intervention/__init__.py",
        "intervention/protocol.py", "intervention/groups.py", "intervention/fixed_step.py",
    ),
    "high-low": (
        "artifacts/storage.py", "intervention/__init__.py",
        "intervention/protocol.py", "intervention/groups.py", "intervention/fixed_step.py",
        "intervention/high_low/__init__.py", "intervention/high_low/run.py",
        "intervention/high_low/selection.py", "intervention/high_low/runtime.py",
    ),
}
TYPES = {"probe": "step_hidden_probe", "fixed-step": "fixed_step_now_vs_delay",
         "high-low": "within_question_high_low"}


def source_hashes(experiment="fixed-step", *, root=None):
    root = Path(root) if root is not None else ROOT
    paths = [root / "veriserve_research" / p for p in COMMON + EXECUTION[experiment]]
    paths += [root / "stages/stage1/veriserve" / f"{name}.py"
              for name in ("common", "prm", "model", "data", "state", "__init__")]
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def analysis_hashes(*, root=None):
    root = Path(root) if root is not None else ROOT
    paths = {root / "veriserve_research" / p for p in COMMON}
    paths.update((root / "veriserve_research/analysis").glob("*.py"))
    paths.update(root / "veriserve_research" / p for p in
                 ("probe/report.py", "intervention/groups.py", "intervention/protocol.py"))
    paths.update(root / "stages/stage1/veriserve" / f"{name}.py"
                 for name in ("common", "prm", "model", "data", "state", "__init__"))
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def run_identity(cfg, experiment, materials=None):
    identity = stable_hash({"schema_version": 2, "experiment_type": TYPES[experiment],
                            "config": cfg, "inputs": materials or {},
                            "execution_sources": source_hashes(experiment)})
    return f"v2-{cfg['seed']}-{identity[:12]}"


def manifest_version(experiment):
    return {"schema_version": 2, "experiment_type": TYPES[experiment],
            "source_hashes": source_hashes(experiment)}


def require_current(manifest, experiment):
    if manifest.get("schema_version") != 2:
        raise ValueError("Legacy execution cannot resume here; use its recorded source commit")
    if manifest.get("experiment_type") != TYPES[experiment]:
        raise ValueError("Experiment type differs from the execution entry")
    if manifest.get("source_hashes") != source_hashes(experiment):
        raise ValueError("Execution source changed; preserve this run and start a new v2 run")


def environment(gpu=False):
    versions = {}
    for name in ("torch", "transformers", "datasets", "numpy", "scikit-learn", "math-verify",
                 "antlr4-python3-runtime", "latex2sympy2-extended", "huggingface-hub", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "executable": os.sys.executable,
              "platform": platform.platform(), "versions": versions}
    if gpu:
        import torch
        result.update(cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        try:
            result["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["nvidia_smi"] = "unavailable"
        if torch.cuda.is_available():
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
            with torch.inference_mode():
                a = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16)
                out = torch.nn.functional.scaled_dot_product_attention(a, a, a, is_causal=True)
            result["bf16_sdpa_forward"] = bool(torch.isfinite(out).all())
            result["free_total_bytes"] = list(torch.cuda.mem_get_info())
    return result


def resolve_revision(repo, kind):
    from huggingface_hub import HfApi
    try:
        info = HfApi().repo_info(repo, repo_type=kind, timeout=30)
        if not SHA.fullmatch(info.sha):
            raise RuntimeError("Hub did not return an exact revision SHA")
        return info.sha
    except Exception:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(HF_HUB_CACHE) / (("models--" if kind == "model" else "datasets--")
                                     + repo.replace("/", "--"))
        ref = cache / "refs/main"
        if ref.exists():
            sha = ref.read_text().strip()
            if SHA.fullmatch(sha) and (cache / "snapshots" / sha).is_dir():
                return sha
        snapshots = sorted(p.name for p in (cache / "snapshots").glob("*") if SHA.fullmatch(p.name))
        if len(snapshots) == 1:
            return snapshots[0]
        raise RuntimeError(f"Cannot resolve {repo}: no unambiguous, revisioned local cache")
