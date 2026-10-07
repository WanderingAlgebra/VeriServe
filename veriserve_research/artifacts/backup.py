"""Native Git/LFS transport; each experiment retains its own failure policy."""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
from veriserve_research import ROOT
from .io import atomic_json, read_json, stable_hash, digest, git, now, event, safe_error
from .provenance import BRANCH, source_hashes
from .materials import safe_read_json


def remote_feature_check(run):
    files = sorted(run.glob("records/**/*.npz"))
    if not files:
        return None
    feature = files[0]
    relative = str(feature.relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix="stage2-lfs-verify-") as tmp:
        folder = Path(tmp)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        ssh = subprocess.run(["git", "config", "--get", "core.sshCommand"], cwd=ROOT, capture_output=True, text=True)
        if ssh.returncode == 0:
            env["GIT_SSH_COMMAND"] = ssh.stdout.strip()
        git("init", cwd=folder)
        git("remote", "add", "origin", git("remote", "get-url", "origin"), cwd=folder)
        git("fetch", "--depth=1", "origin", BRANCH, cwd=folder, env=env)
        pointer = git("show", f"FETCH_HEAD:{relative}", cwd=folder, env=env)
        if not pointer.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Remote feature is not a native Git LFS pointer")
        git("update-ref", "HEAD", "FETCH_HEAD", cwd=folder, env=env)
        git("checkout", "FETCH_HEAD", "--", relative, cwd=folder, env=env)
        git("lfs", "fetch", f"--include={relative}", "--exclude=", "origin", "FETCH_HEAD",
            cwd=folder, env=env)
        git("lfs", "checkout", relative, cwd=folder, env=env)
        actual = digest(folder / relative)
        if actual != digest(feature):
            raise RuntimeError("Downloaded remote NPZ hash differs from local feature")
        return {"path": relative, "sha256": actual, "remote_commit": git("rev-parse", "FETCH_HEAD", cwd=folder),
                "verified_at": now()}


def backup_probe(run, message):
    """Use native Git/LFS; any failure prevents the next formal batch from starting."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        if git("branch", "--show-current") != BRANCH:
            raise RuntimeError(f"Expected existing branch {BRANCH}; switch safely before running")
        git("lfs", "version")
        paths = ["stages/stage2", "README.md", ".gitignore", ".gitattributes", "veriserve_research", "environments/stage23"]
        for p in run.glob("records/**/*.npz"):
            relative = str(p.relative_to(ROOT))
            attrs = git("check-attr", "filter", "--", relative)
            if not attrs.endswith(": lfs"):
                raise RuntimeError(f"Feature is not tracked by Git LFS: {relative}")
            if subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0:
                raise RuntimeError(f"Feature ignored by Git: {relative}")
        git("add", "--", *paths)
        dirty = git("diff", "--cached", "--name-only", "--", *paths)
        if dirty:
            # --only avoids committing unrelated changes already staged by the user.
            git("commit", "--only", "-m", message, "--", *paths)
        commit = git("rev-parse", "HEAD")
        if list(run.glob("records/**/*.npz")):
            git("lfs", "push", "origin", "HEAD", env=env)
        git("push", "--porcelain", "origin", f"HEAD:refs/heads/{BRANCH}", env=env)
        remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}", env=env).split()
        if not remote or remote[0] != commit:
            raise RuntimeError("Remote branch did not confirm the pushed commit")
        previous = read_json(run / "backup.json", {})
        verification = previous.get("feature_remote_verification")
        if not verification:
            verification = remote_feature_check(run)
        status = {"status": "PUSHED", "commit": commit, "pushed_at": now(),
                  "feature_remote_verification": verification}
        atomic_json(run / "backup.json", status)
        print(f"PUSHED {commit} remote_feature_verified={bool(verification)}", flush=True)
        return status
    except Exception as exc:
        event(run, "BACKUP_FAILED", error=safe_error(exc),
              unconfirmed_features=[str(p.relative_to(ROOT)) for p in run.glob("records/**/*.npz")])
        atomic_json(run / "backup.json", {"status": "FAILED", "error": safe_error(exc), "time": now()})
        raise RuntimeError(f"Backup failed; local files retained, collection stopped: {safe_error(exc)}") from exc


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


def backup_fixed_step(run, reason="manual"):
    """Failure is durable and raises, so the caller cannot begin the next batch."""
    from . import storage
    run = Path(run)
    previous = safe_read_json(run / "backup.json", {})
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    paths = ["stages/stage3", ".gitignore", ".gitattributes", "veriserve_research", "environments/stage23"]
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
                  "pushed_at": now(), "completed_problems": storage.completed_problems(run),
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


def backup_high_low(run, completed_pairs, reason):
    """Same ordinary Git/LFS backup, scoped to this experiment and its two READMEs."""
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    scoped = ['stages/stage3/high_low', 'stages/stage3/high_low_config.json',
              str(run.relative_to(ROOT)), 'stages/stage3/README.md', 'README.md', 'veriserve_research',
              'environments/stage23', '.gitignore', '.gitattributes']
    status = {'status': 'PENDING', 'reason': reason, 'created': now(), 'completed_pairs': completed_pairs}
    atomic_json(run / 'backup.json', status)
    try:
        if git('branch', '--show-current') != BRANCH:
            raise RuntimeError('Backup must use the authorized existing branch')
        git('add', '--', *scoped)
        if git('diff', '--cached', '--name-only', '--', *scoped):
            git('commit', '--only', '-m', f'Stage3 HIGH/LOW {reason}', '--', *scoped)
        commit = git('rev-parse', 'HEAD')
        if list(run.rglob('*.npz')):
            git('lfs', 'push', 'origin', 'HEAD', env=env)
        git('push', '--porcelain', 'origin', f'HEAD:refs/heads/{BRANCH}', env=env)
        if git('ls-remote', 'origin', f'refs/heads/{BRANCH}', env=env).split()[0] != commit:
            raise RuntimeError('Remote did not confirm the pushed commit')
        status.update(status='PUSHED', commit=commit, pushed_at=now())
    except Exception as exc:
        # The protocol explicitly permits local saving when remote access is unavailable.
        status.update(status='LOCAL_ONLY', error=safe_error(exc))
        print(f'BACKUP LOCAL_ONLY: {status["error"]}', flush=True)
    atomic_json(run / 'backup.json', status)
    return status

