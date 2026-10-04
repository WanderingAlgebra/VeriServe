"""Stage 3 uses atomic files and ordinary Git/LFS backups, with no scheduler."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stages.stage1.veriserve.common import atomic_json, read_json, stable_hash
from stages.stage2.run_probe import (ROOT, BRANCH, SHA, atomic_bytes, digest,
                                     environment as stage2_environment, git, now,
                                     safe_error)

HERE = Path(__file__).resolve().parent
STAGE2 = ROOT / "stages/stage2/runs/20261004-36fe9009f1f5"
# The initial resolution preceded downloads; preserve its provenance without changing config_hash.
INITIAL_VERIFIER_RESOLUTION = {
    "repo": "Qwen/Qwen2.5-Math-PRM-7B",
    "sha": "0610740060112df12585d00a1c5f4624d2f59051",
    "method": "hub_model_info",
    "stage1_recorded_revision_found": False,
    "revisioned_local_cache_found": False,
    "persisted_before_download": True,
}


def source_hashes():
    paths = [ROOT / "stages/stage2/run_probe.py", ROOT / "stages/stage2/config.json"]
    paths += [ROOT / "stages/stage1/veriserve" / f"{name}.py"
              for name in ("common", "prm", "model", "state")]
    paths += sorted(HERE.glob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def environment():
    result = stage2_environment(gpu=False)
    for name in ("bitsandbytes", "accelerate", "scipy", "pyyaml"):
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][name] = None
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
             "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = "unavailable"
    return result


def workingtree():
    # Artifact writes are excluded so recording the manifest does not dirty its own provenance.
    paths = [*source_hashes(), "stages/stage3/config.json", "stages/stage3/README.md",
             ".gitignore", ".gitattributes"]
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths)
    return {"head_commit": git("rev-parse", "HEAD"), "source_clean": not bool(status),
            "source_status": status.splitlines(),
            "scope": "source files, config, documentation and LFS metadata; excludes run artifacts"}


def verifier_provenance(cfg, config_path=None):
    if cfg.get("verifier_revision_source"):
        return cfg["verifier_revision_source"]
    if (cfg["verifier"]["id"] == INITIAL_VERIFIER_RESOLUTION["repo"]
            and cfg["revisions"]["verifier"] == INITIAL_VERIFIER_RESOLUTION["sha"]):
        result = dict(INITIAL_VERIFIER_RESOLUTION)
        if config_path is not None:
            result["config_file_mtime_utc"] = datetime.fromtimestamp(
                Path(config_path).stat().st_mtime, timezone.utc).isoformat()
            result["timestamp_basis"] = "observed persisted config mtime; exact Hub response timestamp was not recorded"
        return result
    return {"method": "configured_exact_revision", "repo": cfg["verifier"]["id"],
            "sha": cfg["revisions"]["verifier"], "original_resolution_provenance": "unavailable"}


def event(run, kind, **values):
    from stages.stage1.veriserve.common import append_jsonl
    append_jsonl(Path(run) / "events.jsonl", {"time": now(), "event": kind, **values})


def save_json(path, value):
    atomic_json(Path(path), value)


def archive_diagnostic(run, name):
    """Keep prior hardware/resource diagnostics before a later session replaces them."""
    if name not in ("runtime_environment.json", "capacity_check.json"):
        raise ValueError("Only session diagnostic files can be archived here")
    run = Path(run)
    path = run / name
    if not path.exists():
        return None
    archive = run / "session_checks" / f"{time.time_ns()}-{name}"
    atomic_bytes(archive, path.read_bytes())
    event(run, "SESSION_DIAGNOSTIC_ARCHIVED", original=name,
          archive=str(archive.relative_to(run)), sha256=digest(archive))
    return archive


def safe_read_json(path, default=None):
    """Preserve corrupt bytes; restore a committed copy, or stop without regeneration."""
    path = Path(path)
    try:
        return read_json(path, default)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        preserved = path.with_name(f"{path.name}.corrupt-{now().replace(':', '')}")
        shutil.copyfile(path, preserved)
        run = next((p for p in path.parents if (p / "manifest.json").exists()
                    and p.parent == HERE / "runs"), None)
        commit = None
        if run is not None and path.name != "backup.json":
            try:
                status = read_json(run / "backup.json", {})
                commit = status.get("last_successful_commit") or (
                    status.get("commit") if status.get("status") == "PUSHED" else None)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if commit and re.fullmatch(r"[0-9a-f]{40}", commit):
            relative = str(path.relative_to(ROOT))
            result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                                    capture_output=True, timeout=30)
            if result.returncode == 0:
                try:
                    value = json.loads(result.stdout)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                else:
                    atomic_bytes(path, result.stdout)
                    event(run, "CORRUPT_FILE_RESTORED", path=relative, commit=commit,
                          preserved=str(preserved.relative_to(ROOT)))
                    return value
        raise RuntimeError(f"Corrupt JSON retained at {preserved}; no valid committed copy. "
                           "Stop and recover the original record before continuing.") from exc


def resolve_verifier(repo):
    """Use recorded stage 1 assets before a local revisioned cache or Hub resolution."""
    candidates = [ROOT / "stages/stage1/.cache/assets.json", ROOT / ".cache/assets.json",
                  Path("/root/autodl-tmp/VeriServe/stages/stage1/.cache/assets.json"),
                  Path("/root/autodl-tmp/VeriServe/.cache/assets.json")]
    candidates += sorted((ROOT / "stages/stage1").glob("**/manifest.json"))
    candidates += sorted((ROOT / "stages/stage1").glob("**/assets.json"))

    def revisions(value):
        found = set()
        if isinstance(value, dict):
            if value.get("id") == repo and SHA.fullmatch(str(value.get("revision", ""))):
                found.add(value["revision"])
            if isinstance(value.get("model_revisions"), dict):
                rev = value["model_revisions"].get("verifier")
                if SHA.fullmatch(str(rev)) and repo in json.dumps(value):
                    found.add(rev)
            for child in value.values():
                found.update(revisions(child))
        elif isinstance(value, list):
            for child in value:
                found.update(revisions(child))
        return found

    found = set()
    sources = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            values = revisions(read_json(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if values:
            found.update(values)
            sources.append(str(path))
    if len(found) > 1:
        raise RuntimeError(f"Conflicting recorded stage 1 PRM revisions in {sources}")
    if found:
        return found.pop(), {"method": "stage1_recorded_assets", "paths": sources}
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    snapshots = sorted(p.name for p in (cache / "snapshots").glob("*")
                       if SHA.fullmatch(p.name))
    if len(snapshots) == 1:
        return snapshots[0], {"method": "local_revisioned_cache", "path": str(cache)}
    from huggingface_hub import HfApi
    sha = HfApi().model_info(repo, timeout=30).sha
    if not SHA.fullmatch(str(sha)):
        raise RuntimeError("Hub did not return an exact verifier SHA")
    return sha, {"method": "hub_model_info", "resolved_at": now()}


def probe_hashes(stage2):
    result = {}
    for name, dimensions in (("B", 3584), ("C", 2)):
        metadata = safe_read_json(stage2 / f"probe_{name}.json")
        if metadata.get("method") != name or metadata.get("layer") != (19 if name == "B" else None):
            raise ValueError(f"Unexpected fixed probe {name} metadata")
        with np.load(stage2 / f"probe_{name}.npz", allow_pickle=False) as weights:
            if set(weights.files) != {"mean", "scale", "w", "b"}:
                raise ValueError(f"Unexpected fixed probe {name} arrays")
            for key in ("mean", "scale", "w"):
                if weights[key].shape != (dimensions,) or not np.isfinite(weights[key]).all():
                    raise ValueError(f"Invalid probe {name} {key}")
            if (weights["scale"] <= 0).any() or float(weights["b"]) != metadata["b"]:
                raise ValueError(f"Probe {name} metadata/weights mismatch")
        for ext in ("json", "npz"):
            path = stage2 / f"probe_{name}.{ext}"
            result[str(path.relative_to(ROOT))] = digest(path)
    return result


def prepare(config_path, resume=True):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = safe_read_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Stage 3 config must be a JSON object")
    stage2 = ROOT / cfg["stage2_run"]
    original = safe_read_json(stage2 / "manifest.json")
    baseline = safe_read_json(ROOT / "stages/stage2/config.json")
    for name in ("model", "prompt", "dataset", "dataset_split", "attention", "dtype",
                 "batch_size", "do_sample"):
        if cfg[name] != baseline[name]:
            raise ValueError(f"Stage 3 must keep original stage 2 {name}")
    revision = cfg.setdefault("revisions", {})
    for name in ("model", "tokenizer", "dataset"):
        if revision.get(name) != baseline["revisions"][name]:
            raise ValueError(f"Stage 3 must keep fixed stage 2 {name} SHA")
    if not revision.get("verifier"):
        if any((HERE / "runs").glob("*/protocol_frozen.json")):
            raise ValueError("Verifier revision cannot be resolved into a frozen protocol")
        revision["verifier"], provenance = resolve_verifier(cfg["verifier"]["id"])
        revision["verifier_tokenizer"] = revision["verifier"]
        cfg["verifier_revision_source"] = provenance
        atomic_json(config_path, cfg)  # Exact SHA is durable before downloads or run_id.
    if not revision.get("verifier_tokenizer"):
        if any((HERE / "runs").glob("*/protocol_frozen.json")):
            raise ValueError("Tokenizer revision cannot be added to a frozen protocol")
        revision["verifier_tokenizer"] = revision["verifier"]
        atomic_json(config_path, cfg)
    for name, sha in revision.items():
        if not SHA.fullmatch(str(sha)):
            raise ValueError(f"{name} revision must be an exact 40-character SHA")
    if revision["verifier_tokenizer"] != revision["verifier"]:
        raise ValueError("PRM model and tokenizer must use the same fixed SHA")
    unused = original["unused_ids"]
    if len(unused) != 190 or len(set(unused)) != 190:
        raise ValueError("Original stage 2 unused_ids must contain exactly 190 unique IDs")
    used = {row["unique_id"] for rows in original["splits"].values() for row in rows}
    if used & set(unused):
        raise ValueError("Stage 3 IDs intersect original stage 2 splits")
    expected_sizes = {"pilot": 20, "dev": 40, "test": 130}
    if cfg["split_sizes"] != expected_sizes:
        raise ValueError("Stage 3 splits must remain 20/40/130")
    config_hash = stable_hash(cfg)
    run = HERE / "runs" / f"{cfg['seed']}-{config_hash[:12]}"
    run.mkdir(parents=True, exist_ok=True)
    manifest = safe_read_json(run / "manifest.json")
    probes = probe_hashes(stage2)
    if manifest:
        if manifest["config_hash"] != config_hash or manifest["probe_hashes"] != probes:
            raise ValueError("Existing manifest/config/probe mismatch; preserve and use a new run")
        if [r["unique_id"] for phase in expected_sizes for r in manifest["splits"][phase]] != unused:
            raise ValueError("Existing ordered split IDs changed")
        if (manifest["stage2_manifest_sha256"] != digest(stage2 / "manifest.json")
                or any(manifest["split_hashes"][phase] != stable_hash(manifest["splits"][phase])
                       for phase in expected_sizes)):
            raise ValueError("Original manifest or restored question fields changed")
        frozen = safe_read_json(run / "protocol_frozen.json", {})
        if frozen and frozen.get("source_hashes") != source_hashes():
            raise ValueError("Frozen source files changed; preserve old records and use a new run")
        if not frozen and not any((run / "snapshots").glob("**/*.json")):
            updated = {**manifest, "prepare_base_commit": manifest.get("prepare_base_commit", manifest["code_commit"]),
                       "source_hashes": source_hashes(), "environment": environment(),
                       "code_commit": git("rev-parse", "HEAD"), "workingtree": workingtree(),
                       "verifier_revision_source": verifier_provenance(cfg, config_path)}
            if updated != manifest:
                atomic_json(run / "manifest.json", updated)
                event(run, "PREPARE_PROVENANCE_REFRESHED", code_commit=updated["code_commit"],
                      prepare_base_commit=updated["prepare_base_commit"], source_hashes=updated["source_hashes"])
                manifest = updated
        return cfg, run, manifest
    from datasets import load_dataset
    rows = list(load_dataset(cfg["dataset"], split=cfg["dataset_split"],
                             revision=revision["dataset"]))
    by_id = {row["unique_id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(unused) <= by_id.keys():
        raise ValueError("Fixed dataset has duplicate or missing unique_id values")
    splits, offset = {}, 0
    for phase, size in expected_sizes.items():
        splits[phase] = [by_id[uid] for uid in unused[offset:offset + size]]
        offset += size
    manifest = {
        "run_id": run.name, "created": now(), "config": cfg, "config_hash": config_hash,
        "revisions": revision, "splits": splits, "ordered_ids": unused,
        "split_hashes": {phase: stable_hash(values) for phase, values in splits.items()},
        "stage2_manifest_sha256": digest(stage2 / "manifest.json"),
        "probe_hashes": probes, "source_hashes": source_hashes(),
        "code_commit": git("rev-parse", "HEAD"), "prepare_base_commit": git("rev-parse", "HEAD"),
        "workingtree": workingtree(), "environment": environment(),
        "verifier_revision_source": verifier_provenance(cfg, config_path),
        "feature_extraction": cfg["feature_extraction"],
        "probe_interpretation": "risk of incorrect unintervened complete trajectory",
        "feedback_adaptation": "checkpoint_step=0: No reasoning steps have been accepted. Restart from Step 1.",
        "grading": original["grading"],
        "timing": "T_postfork_wall is follow-up processing time from the common snapshot; offline feature/probe and fork restoration are separate",
    }
    atomic_json(run / "manifest.json", manifest)
    event(run, "PREPARED", sizes=expected_sizes, source_hashes=manifest["source_hashes"])
    return cfg, run, manifest


def completed_problems(run):
    run = Path(run)
    counts = {phase: 0 for phase in ("pilot", "dev", "test")}
    for path in (run / "snapshots").glob("**/*.json"):
        snapshot = safe_read_json(path)
        phase = snapshot.get("phase")
        if phase not in counts:
            continue
        if snapshot.get("status") == "NO_ELIGIBLE_ANCHOR":
            counts[phase] += 1
            continue
        key = stable_hash(snapshot["unique_id"])[:20]
        arms = [safe_read_json(run / "arms" / f"{key}.{arm}.json", {})
                for arm in ("NOW", "DELAY_2")]
        if all(a.get("completed") or a.get("status") == "COMPLETED" for a in arms):
            counts[phase] += 1
    return counts


def verify_remote(run, commit):
    """Retrieve the first complete paired record and selected feature from a fresh checkout."""
    run = Path(run)
    selected = None
    for snap in sorted((run / "snapshots").glob("**/*.json")):
        data = safe_read_json(snap)
        key = stable_hash(data["unique_id"])[:20]
        arms = [run / "arms" / f"{key}.{name}.json" for name in ("NOW", "DELAY_2")]
        feature = snap.with_suffix(".npz")
        records = [safe_read_json(p, {}) for p in arms]
        if (feature.exists() and
                all(a.get("completed") or a.get("status") == "COMPLETED" for a in records)):
            selected = [snap, *arms, feature]
            break
    if selected is None:
        return None
    relative = [str(p.relative_to(ROOT)) for p in selected]
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    ssh = subprocess.run(["git", "config", "--get", "core.sshCommand"], cwd=ROOT,
                         capture_output=True, text=True)
    if ssh.returncode == 0:
        env["GIT_SSH_COMMAND"] = ssh.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="stage3-remote-verify-") as tmp:
        folder = Path(tmp)
        git("init", cwd=folder)
        git("remote", "add", "origin", git("remote", "get-url", "origin"), cwd=folder)
        git("fetch", "--depth=1", "origin", BRANCH, cwd=folder, env=env)
        actual_commit = git("rev-parse", "FETCH_HEAD", cwd=folder)
        if actual_commit != commit:
            raise RuntimeError("Remote branch moved before independent backup verification")
        git("checkout", "--detach", "FETCH_HEAD", cwd=folder, env=env)
        npz = relative[-1]
        pointer = git("show", f"FETCH_HEAD:{npz}", cwd=folder)
        if not pointer.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Remote stage 3 feature is not a native Git LFS pointer")
        git("lfs", "fetch", f"--include={npz}", "--exclude=", "origin", "FETCH_HEAD",
            cwd=folder, env=env)
        git("lfs", "checkout", npz, cwd=folder, env=env)
        hashes = {p: digest(folder / p) for p in relative}
        if any(hashes[p] != digest(ROOT / p) for p in relative):
            raise RuntimeError("Independently retrieved snapshot/arms/NPZ hashes differ")
        return {"remote_commit": actual_commit, "verified_at": now(), "sha256": hashes}


def backup(run, reason="manual"):
    """Failure is durable and raises, so the caller cannot begin the next batch."""
    run = Path(run)
    previous = safe_read_json(run / "backup.json", {})
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    paths = ["stages/stage3", ".gitignore", ".gitattributes"]
    last = previous.get("last_successful_commit") or (
        previous.get("commit") if previous.get("status") == "PUSHED" else None)
    atomic_json(run / "backup.json", {**previous, "status": "PENDING", "time": now(),
                "reason": reason, "last_successful_commit": last})
    event(run, "BACKUP_PENDING", reason=reason, last_successful_commit=last)
    try:
        if git("branch", "--show-current") != BRANCH:
            raise RuntimeError(f"Expected authorized existing branch {BRANCH}")
        git("lfs", "version")
        for feature in run.glob("**/*.npz"):
            relative = str(feature.relative_to(ROOT))
            if not git("check-attr", "filter", "--", relative).endswith(": lfs"):
                raise RuntimeError(f"Stage 3 NPZ is not tracked by native LFS: {relative}")
            if subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0:
                raise RuntimeError(f"Stage 3 NPZ is ignored: {relative}")
        git("add", "--", *paths)
        if git("diff", "--cached", "--name-only", "--", *paths):
            git("commit", "--only", "-m", f"Stage3 {reason}", "--", *paths)
        commit = git("rev-parse", "HEAD")
        if list(run.glob("**/*.npz")):
            git("lfs", "push", "origin", "HEAD", env=env)
        git("push", "--porcelain", "origin", f"HEAD:refs/heads/{BRANCH}", env=env)
        remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}", env=env).split()
        if not remote or remote[0] != commit:
            raise RuntimeError("Remote did not confirm the ordinary pushed commit")
        verification = previous.get("remote_verification") or verify_remote(run, commit)
        frozen_path, groups_path = run / "protocol_frozen.json", run / "dev_groups.json"
        status = {"status": "PUSHED", "commit": commit, "last_successful_commit": commit,
                  "stage3_source_commit": commit, "source_hashes": source_hashes(),
                  "protocol_frozen_sha256": digest(frozen_path) if frozen_path.exists() else None,
                  "dev_groups_sha256": digest(groups_path) if groups_path.exists() else None,
                  "pushed_at": now(), "completed_problems": completed_problems(run),
                  "remote_verification": verification}
        atomic_json(run / "backup.json", status)
        event(run, "BACKUP_SUCCEEDED", commit=commit, reason=reason,
              remote_verified=bool(verification))
        return status
    except Exception as exc:
        status = {"status": "FAILED", "error": safe_error(exc), "time": now(),
                  "last_successful_commit": last,
                  "remote_verification": previous.get("remote_verification")}
        atomic_json(run / "backup.json", status)
        event(run, "BACKUP_FAILED", error=safe_error(exc), last_successful_commit=last)
        raise RuntimeError(f"Backup failed; retained local data, stop before next batch: {safe_error(exc)}") from exc


def ensure_frozen_backup(run, frozen):
    """A frozen file alone never proves it was pushed; persist pending state before retry."""
    run = Path(run)
    frozen_path, groups_path = run / "protocol_frozen.json", run / "dev_groups.json"
    if digest(groups_path) != frozen["dev_groups_sha256"]:
        raise RuntimeError("Frozen dev groups hash changed; preserve files and recover the original groups")
    status = safe_read_json(run / "backup.json", {})
    if (status.get("status") == "PUSHED"
            and status.get("protocol_frozen_sha256") == digest(frozen_path)
            and status.get("dev_groups_sha256") == frozen["dev_groups_sha256"]
            and status.get("source_hashes") == frozen["source_hashes"]
            and status.get("stage3_source_commit") == status.get("commit")):
        return status
    last = status.get("last_successful_commit") or (
        status.get("commit") if status.get("status") == "PUSHED" else None)
    atomic_json(run / "backup.json", {**status, "status": "PENDING", "time": now(),
                "reason": "frozen protocol must be pushed before test",
                "last_successful_commit": last})
    event(run, "FROZEN_BACKUP_PENDING", last_successful_commit=last,
          protocol_frozen_sha256=digest(frozen_path))
    return backup(run, "freeze dev groups and protocol before independent test")


def self_check():
    """One CPU-only check for storage trust boundaries, callable from the main self-check."""
    with tempfile.TemporaryDirectory(prefix="stage3-storage-check-") as tmp:
        path = Path(tmp) / "record.json"
        save_json(path, {"token_ids": [1, 2, 3]})
        assert safe_read_json(path)["token_ids"] == [1, 2, 3]
        path.write_text("{broken", encoding="utf-8")
        try:
            safe_read_json(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Corrupt records must not become new generation requests")
        assert path.read_text() == "{broken"
        assert len(list(path.parent.glob("record.json.corrupt-*"))) == 1
        run = path.parent / "run"
        diagnostic = run / "runtime_environment.json"
        original_bytes = b'{ "gpu": "earlier GPU", "session": 1 }\n'
        atomic_bytes(diagnostic, original_bytes)
        archive = archive_diagnostic(run, diagnostic.name)
        assert diagnostic.read_bytes() == archive.read_bytes() == original_bytes
        save_json(diagnostic, {"gpu": "later GPU", "session": 2})
        assert archive.read_bytes() == original_bytes
        save_json(run / "dev_groups.json", {"B": [1, 2], "C": [1, 2]})
        frozen = {"dev_groups_sha256": digest(run / "dev_groups.json"), "source_hashes": {"code.py": "hash"}}
        save_json(run / "protocol_frozen.json", frozen)
        pushed = {"status": "PUSHED", "commit": "a" * 40, "stage3_source_commit": "a" * 40,
                  "source_hashes": frozen["source_hashes"], "dev_groups_sha256": frozen["dev_groups_sha256"],
                  "protocol_frozen_sha256": digest(run / "protocol_frozen.json")}
        save_json(run / "backup.json", pushed)
        assert ensure_frozen_backup(run, frozen) == pushed
        save_json(run / "backup.json", {"status": "PUSHED", "commit": "a" * 40})
        original_backup = globals()["backup"]
        def fake_backup(folder, reason):
            pending = safe_read_json(folder / "backup.json")
            assert pending["status"] == "PENDING" and pending["last_successful_commit"] == "a" * 40
            return {"status": "SELF_CHECK_RETRY_OBSERVED"}
        globals()["backup"] = fake_backup
        try:
            assert ensure_frozen_backup(run, frozen)["status"] == "SELF_CHECK_RETRY_OBSERVED"
        finally:
            globals()["backup"] = original_backup
    assert probe_hashes(STAGE2)
    original = read_json(STAGE2 / "manifest.json")
    ids = original["unused_ids"]
    assert len(ids) == len(set(ids)) == 190
    assert not set(ids) & {r["unique_id"] for rows in original["splits"].values() for r in rows}
    return {"status": "PASSED", "checks": ["atomic_roundtrip", "corruption_preserved",
            "fixed_probe_metadata_weights", "190_original_unused_ids_disjoint", "frozen_backup_interrupt_retry",
            "session_diagnostics_archived_byte_exact"]}


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False))
