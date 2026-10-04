"""Frozen step-hidden risk experiment. Run from the repository root; --self-check is CPU-only."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import subprocess
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from stages.stage1.veriserve.common import atomic_json, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BRANCH = "experiment/step-hidden-probe"
STEP = re.compile(r"(?m)^Step ([1-9]\d*):")
FINAL = re.compile(r"(?m)^Final answer:[^\n]*")
ANSWER_STEP = re.compile(r"(?i)\b(?:final answer|(?:the|our) answer\s+(?:is|equals)|answer is)\b")
SHA = re.compile(r"^[0-9a-f]{40}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_code():
    return {"script_sha256": digest(__file__),
            "common_sha256": digest(ROOT / "stages/stage1/veriserve/common.py")}


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_npz(path, **arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    atomic_bytes(path, b.getvalue())


def safe_error(error):
    # Remote failures may echo a credential-bearing URL; never persist that URL.
    return re.sub(r"(?:https?|ssh)://\S+", "[redacted URL]", str(error))


def git(*args, cwd=ROOT, env=None):
    result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True,
                            timeout=600)
    if result.returncode:
        raise RuntimeError(safe_error(result.stderr))
    return result.stdout.strip()


def event(run, kind, **values):
    from stages.stage1.veriserve.common import append_jsonl
    append_jsonl(run / "events.jsonl", {"time": now(), "event": kind, **values})


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


def prepare(config_path):
    cfg = read_json(config_path)
    revisions = cfg.setdefault("revisions", {})
    for name, kind, repo in (("model", "model", cfg["model"]),
                             ("dataset", "dataset", cfg["dataset"])):
        if not revisions.get(name):
            revisions[name] = resolve_revision(repo, kind)
            atomic_json(config_path, cfg)  # Persist first resolution even if the next download fails.
        if not SHA.fullmatch(revisions[name]):
            raise ValueError(f"{name} revision must be a 40-character SHA")
    revisions.setdefault("tokenizer", revisions["model"])
    if not SHA.fullmatch(revisions["tokenizer"]):
        raise ValueError("tokenizer revision must be a SHA")
    atomic_json(config_path, cfg)
    cfg_hash = stable_hash(cfg)
    run = HERE / "runs" / f"{cfg['seed']}-{cfg_hash[:12]}"
    run.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / "manifest.json")
    if manifest:
        if manifest["config_hash"] != cfg_hash:
            raise ValueError("Manifest/config mismatch; use a new run_id")
        return cfg, run, manifest
    from datasets import load_dataset
    rows = sorted(list(load_dataset(cfg["dataset"], split=cfg["dataset_split"],
                                    revision=revisions["dataset"])), key=lambda x: x["unique_id"])
    if len({r["unique_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate unique_id in dataset")
    random.Random(cfg["seed"]).shuffle(rows)
    offset, splits = 0, {}
    for name in ("train", "test", "smoke"):
        size = cfg["split_sizes"][name]
        splits[name] = rows[offset:offset + size]
        if len(splits[name]) != size:
            raise ValueError("Dataset too short")
        offset += size
    manifest = {"run_id": run.name, "created": now(), "config": cfg, "config_hash": cfg_hash,
                "revisions": revisions, "splits": splits,
                "unused_ids": [r["unique_id"] for r in rows[offset:]],
                "environment": environment(gpu=True), "code_commit": git("rev-parse", "HEAD"),
                "script_sha256": digest(__file__), "common_sha256": digest(
                    ROOT / "stages/stage1/veriserve/common.py"),
                "hidden_index_convention": "hidden_states[1..L]; index L includes final norm in Qwen2",
                "grading": {"library": "Math-Verify", "extraction": "last complete boxed on separate Final answer line",
                            "parse": "LatexExtractionConfig(boxed_match_priority=0), no_fallback, first_match, raise_on_error",
                            "gold_wrapper": "$...$", "timeout_seconds": cfg["score_timeout_seconds"]}}
    atomic_json(run / "manifest.json", manifest)
    event(run, "PREPARED", sizes={k: len(v) for k, v in splits.items()})
    return cfg, run, manifest


def boxed(text):
    """Last complete boxed expression; nested and escaped braces are handled, never math itself."""
    found = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            escapes = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                escapes += 1
                j -= 1
            if escapes % 2:
                continue
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                found.append(text[start:i])
                break
    return found[-1] if found else None


def grade(text, gold, completion, timeout):
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.utils import TimeoutException
    result = {"answer": None, "label": None, "exclusion": None}
    lines = list(FINAL.finditer(text))
    if lines:
        result["answer"] = boxed(lines[-1].group())
    if completion != "EOS":
        result["exclusion"] = completion
        return result
    if not lines:
        result["exclusion"] = "NO_FINAL_ANSWER"
        return result
    if len(lines) != 1 or text[lines[-1].end():].strip():
        result["exclusion"] = "FINAL_FORMAT_ERROR"
        return result
    if result["answer"] is None:
        result["exclusion"] = "MISSING_OR_INCOMPLETE_BOXED"
        return result
    options = {"extraction_config": [LatexExtractionConfig(boxed_match_priority=0)],
               "fallback_mode": "no_fallback", "extraction_mode": "first_match",
               "parsing_timeout": timeout, "raise_on_error": True}
    phase = "GOLD"
    try:
        expected = parse(f"${gold}$", **options)
        if not expected:
            result["exclusion"] = "GOLD_UNPARSEABLE"
            return result
        phase = "PREDICTION"
        predicted = parse(f"${result['answer']}$", **options)
        if not predicted:
            result["exclusion"] = "PREDICTION_UNPARSEABLE"
            return result
        phase = "VERIFY"
        correct = verify(expected, predicted, timeout_seconds=timeout, raise_on_error=True)
        result["label"] = int(not correct)
    except TimeoutException:
        result["exclusion"] = f"{phase}_TIMEOUT"
    except Exception as exc:
        result["exclusion"] = f"{phase}_ERROR"
        result["grading_error"] = safe_error(exc)
    return result


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def aligned_end(tokenizer, ids, text, char_end, char_start=0):
    """Align a character boundary to original token IDs, without re-tokenizing any text."""
    cache = {0: ""}
    def prefix(n):
        if n not in cache:
            cache[n] = decode(tokenizer, ids[:n])
        return cache[n]
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(prefix(mid)) <= char_end:
            lo = mid
        else:
            hi = mid - 1
    n = lo
    previous = 0
    while n:
        if not text.startswith(prefix(n)) or len(prefix(n)) > char_end:
            n -= 1
            continue
        previous = n - 1
        while previous and not text.startswith(prefix(previous)):
            previous -= 1  # A preceding token may contain only a partial UTF-8 character.
        if text[len(prefix(previous)):len(prefix(n))].strip():
            break
        n -= 1
    if not n or len(prefix(previous)) < char_start:
        raise ValueError("FORMAT_ERROR: no complete content token within step")
    if "\ufffd" in prefix(n)[-1:]:
        raise ValueError("FORMAT_ERROR: incomplete UTF-8 token")
    return {"end_token_index": n - 1, "prefix_token_count": n,
            "char_end": char_end, "aligned_char_end": len(prefix(n)),
            "boundary_aligned_back": len(prefix(n)) != char_end}


def boundaries(tokenizer, generated_ids):
    special = set(tokenizer.all_special_ids)
    content_ids = list(generated_ids)
    while content_ids and content_ids[-1] in special:
        content_ids.pop()
    text = decode(tokenizer, content_ids)
    result = {"steps": [], "excluded_steps": [], "format_errors": [], "end": None}
    if not text.rstrip():
        result["format_errors"].append("FORMAT_ERROR: empty generation")
        return text, result
    if any(t in special for t in content_ids):
        result["format_errors"].append("FORMAT_ERROR: interior control token")
        return text, result
    try:
        result["end"] = aligned_end(tokenizer, content_ids, text, len(text))
    except ValueError as exc:
        result["format_errors"].append(str(exc))
    matches = list(STEP.finditer(text))
    final = list(FINAL.finditer(text))
    final_start = final[0].start() if final else len(text)
    numbers = [int(m.group(1)) for m in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        result["format_errors"].append("FORMAT_ERROR: missing, repeated or non-sequential Step N markers")
        return text, result
    if not matches:
        result["format_errors"].append("FORMAT_ERROR: no Step N marker")
    for i, match in enumerate(matches):
        stop = min(matches[i + 1].start() if i + 1 < len(matches) else len(text), final_start)
        if stop <= match.start():
            result["format_errors"].append("FORMAT_ERROR: step after Final answer")
            continue
        segment = text[match.start():stop].rstrip()
        step = {"step_number": int(match.group(1)), "text": segment, "char_start": match.start()}
        if re.search(r"\\boxed\s*\{", segment) or ANSWER_STEP.search(segment):
            step["reason"] = "EXPLICIT_FINAL_OR_BOXED"
            result["excluded_steps"].append(step)
            continue
        if not text[match.end():stop].strip():
            result["format_errors"].append(f"FORMAT_ERROR: empty Step {step['step_number']}")
            continue
        try:
            step.update(aligned_end(tokenizer, content_ids, text,
                                    match.start() + len(segment), match.end()))
            step["effective_step"] = len(result["steps"]) + 1
            step["near_end"] = False
            result["steps"].append(step)
        except ValueError as exc:
            result["format_errors"].append(str(exc))
    # Keep reliable positions even when a separate step failed, but never guess an anomalous marker.
    if result["steps"]:
        result["steps"][-1]["near_end"] = True
    return text, result


def record_path(run, split, uid):
    return run / "records" / split / (hashlib.sha256(uid.encode()).hexdigest()[:20] + ".json")


def valid_features(path, record):
    files = record.get("feature_files", [])
    if not files:
        return False
    try:
        for entry in files:
            p = path.parent / entry["name"]
            if digest(p) != entry["sha256"]:
                return False
            with np.load(p, allow_pickle=False) as arrays:
                if not np.isfinite(arrays["hidden"]).all():
                    return False
        return True
    except (OSError, ValueError, EOFError):
        return False


def read_record(path, run):
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if (not isinstance(value, dict) or not {"prompt_ids", "generated_ids", "token_sha256", "stage"} <= value.keys()
                or any(not isinstance(value[key], list) or not value[key]
                       or any(type(token) is not int or token < 0 for token in value[key])
                       for key in ("prompt_ids", "generated_ids"))):
            raise ValueError("Invalid original token record structure")
    except (ValueError, OSError) as exc:
        # Never replace original tokens or good features because a JSON became unreadable.
        event(run, "CORRUPT_JSON", path=str(path.relative_to(run)), error=safe_error(exc))
        atomic_bytes(path.with_suffix(f".corrupt-{time.time_ns()}"), path.read_bytes())
        try:
            restored = json.loads(git("show", f"HEAD:{path.relative_to(ROOT)}"))
            if stable_hash({"prompt_ids": restored["prompt_ids"], "generated_ids": restored["generated_ids"]}) != restored["token_sha256"]:
                raise ValueError("Committed token checksum mismatch")
        except (RuntimeError, ValueError, KeyError):
            raise ValueError(f"Unreadable original token JSON {path}; preserved files, recover from a valid backup") from exc
        atomic_json(path, restored)
        event(run, "JSON_RECOVERED_FROM_GIT", path=str(path.relative_to(run)))
        value = restored
    if value:
        if stable_hash({"prompt_ids": value["prompt_ids"], "generated_ids": value["generated_ids"]}) != value["token_sha256"]:
            raise ValueError(f"Token checksum mismatch in {path}; preserve and inspect manually")
    return value


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for collection; use --fit-only on CPU")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revisions"]["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revisions"]["model"], dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda:0").eval()
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Model parameters are not uniformly BF16")
    with torch.inference_mode():
        test = model.model(input_ids=torch.tensor([[tokenizer.eos_token_id]], device="cuda"),
                           use_cache=False, output_hidden_states=True)
    if len(test.hidden_states) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Unexpected hidden_states index convention")
    return model, tokenizer


def extract(model, ids, positions):
    import torch
    with torch.inference_mode():
        inputs = torch.tensor([ids], device="cuda:0")
        out = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                          use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = np.stack([h[0, positions].float().cpu().numpy() for h in out.hidden_states[1:]])
    return hidden


def causal_check(model, record, cfg):
    import torch
    steps = record["boundaries"]["steps"]
    if not steps:
        return {"passed": None, "applicable": False, "reason": "no intermediate position; trajectory retained"}
    ids = record["prompt_ids"] + record["generated_ids"]
    position = len(record["prompt_ids"]) + steps[0]["end_token_index"]
    original = extract(model, ids, [position])
    changed = ids[:position + 1] + [model.config.vocab_size // 2] * (len(ids) - position - 1)
    future = extract(model, changed, [position])
    prefix = extract(model, ids[:position + 1], [position])
    checks = {}
    for name, other in (("changed_future", future), ("independent_prefix", prefix)):
        diff = np.abs(original - other)
        checks[name] = {"max_abs": float(diff.max()), "rms": float(np.sqrt(np.mean(diff ** 2))),
                        "max_abs_per_layer": diff.max(axis=(1, 2)).tolist(),
                        "passed": bool(np.allclose(original, other, atol=cfg["causal_atol"],
                                                   rtol=cfg["causal_rtol"]))}
    checks.update(passed=all(v["passed"] for v in checks.values()), applicable=True, position=position,
                  atol=cfg["causal_atol"], rtol=cfg["causal_rtol"],
                  tolerance_note="BF16 rounding: atol=1/32, rtol=one BF16 relative ULP; report raw errors")
    torch.cuda.empty_cache()
    return checks


def collect_one(cfg, run, split, row, model, tokenizer, hardware):
    import torch
    path = record_path(run, split, row["unique_id"])
    record = read_record(path, run)
    code = source_code()
    if (record and record.get("stage") == "FEATURES" and valid_features(path, record)
            and record.get("feature_source_code") == code):
        print(f"SKIP {split} {row['unique_id']}", flush=True)
        return record, False
    if not record:
        prompt = cfg["prompt"].replace("{problem}", row["problem"])
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=True, add_generation_prompt=True)
        budget = min(cfg["max_new_tokens"], cfg["max_total_tokens"] - len(prompt_ids))
        if budget <= 0:
            raise ValueError("Prompt exceeds the frozen context limit")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = torch.tensor([prompt_ids], device="cuda:0")
            eos = model.generation_config.eos_token_id
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    do_sample=False, max_new_tokens=budget, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
        torch.cuda.synchronize()
        generated_ids = output[0, len(prompt_ids):].tolist()
        eos_ids = eos if isinstance(eos, list) else [eos]
        completion = ("EOS" if generated_ids[-1] in eos_ids else
                      "CONTEXT_LIMIT" if len(prompt_ids) + len(generated_ids) >= cfg["max_total_tokens"]
                      else "LENGTH_TRUNCATED")
        record = {"unique_id": row["unique_id"], "split": split, "source": row,
                  "prompt_ids": prompt_ids, "generated_ids": generated_ids,
                  "raw_text": decode(tokenizer, generated_ids), "completion": completion,
                  "stage": "GENERATED", "generated_at": now(), "hardware": hardware,
                  "code_commit": git("rev-parse", "HEAD"), "script_sha256": digest(__file__),
                  "config_hash": stable_hash(cfg),
                  "token_sha256": stable_hash({"prompt_ids": prompt_ids, "generated_ids": generated_ids}),
                  "timings": {"generation_seconds": time.perf_counter() - start},
                  "generation_peak_bytes": torch.cuda.max_memory_allocated()}
        atomic_json(path, record)  # Save tokens before scoring or hidden extraction.
        del output, inputs
    start = time.perf_counter()
    text, bounds = boundaries(tokenizer, record["generated_ids"])
    record["boundaries"] = bounds
    record.update(grade(text, row["answer"], record["completion"], cfg["score_timeout_seconds"]))
    record["timings"]["boundary_and_grading_seconds"] = time.perf_counter() - start
    record["stage"] = "SCORED"
    atomic_json(path, record)
    selected = bounds["steps"] + ([bounds["end"]] if bounds["end"] is not None else [])
    if not selected:
        record["stage"] = "NO_FEATURE_POSITIONS"
        atomic_json(path, record)
        return record, True
    positions = [len(record["prompt_ids"]) + s["end_token_index"] for s in selected]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    hidden = extract(model, record["prompt_ids"] + record["generated_ids"], positions)
    torch.cuda.synchronize()
    record["timings"]["reforward_seconds"] = time.perf_counter() - start
    record["reforward_peak_bytes"] = torch.cuda.max_memory_allocated()
    previous_features = {f["name"]: f["sha256"] for f in record.get("feature_files", [])}
    record["feature_files"] = []
    # ponytail: shard only beyond 40 MiB raw; layer shards keep each compressed file below 50 MiB.
    layers_per_file = max(1, (40 * 1024 * 1024) // hidden[0].nbytes)
    for lo in range(0, len(hidden), layers_per_file):
        hi = min(lo + layers_per_file, len(hidden))
        suffix = f".features-{code['script_sha256'][:12]}"
        feature_path = path.with_suffix(suffix + ".npz") if len(hidden) <= layers_per_file else path.with_suffix(suffix + f".layers{lo + 1}-{hi}.npz")
        preserved = feature_path.exists() and digest(feature_path) == previous_features.get(feature_path.name)
        if not preserved:
            save_npz(feature_path, hidden=hidden[lo:hi], layers=np.arange(lo + 1, hi + 1),
                     positions=np.asarray(positions, dtype=np.int64))
        if feature_path.stat().st_size > 50 * 1024 * 1024:
            raise RuntimeError("Single feature shard exceeds 50 MiB; reduce layer shard size")
        record["feature_files"].append({"name": feature_path.name, "sha256": digest(feature_path)})
    record["stage"] = "FEATURES"
    record["feature_hardware"] = hardware
    record["feature_source_code"] = code
    atomic_json(path, record)
    print(f"DONE {split} {row['unique_id']} label={record['label']} exclusion={record['exclusion']} "
          f"steps={len(bounds['steps'])} tokens={len(record['generated_ids'])}", flush=True)
    return record, True


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
        git("fetch", "--depth=1", "--filter=blob:none", "origin", BRANCH, cwd=folder, env=env)
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


def backup(run, message):
    """Use native Git/LFS; any failure prevents the next formal batch from starting."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        if git("branch", "--show-current") != BRANCH:
            raise RuntimeError(f"Expected existing branch {BRANCH}; switch safely before running")
        git("lfs", "version")
        paths = ["stages/stage2", "README.md", ".gitignore", ".gitattributes"]
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


def snapshot_hashes(run, split):
    hashes = {}
    for path in sorted((run / "records" / split).glob("*.json")):
        record = read_record(path, run)
        if (record and record.get("stage") == "FEATURES" and valid_features(path, record)
                and record.get("feature_source_code") == source_code()):
            for p in [path, *[path.parent / f["name"] for f in record["feature_files"]]]:
                hashes[str(p.relative_to(run))] = digest(p)
    return hashes


def load_examples(run, manifest, split):
    examples = []
    counts = Counter({key: 0 for key in ("generated", "completed", "correct", "wrong", "truncated",
                                       "unscorable", "no_steps", "format_error_trajectories", "features_missing_or_corrupt")})
    counts["planned"] = len(manifest["splits"][split])
    for row in manifest["splits"][split]:
        path = record_path(run, split, row["unique_id"])
        record = read_record(path, run)
        if not record:
            counts["not_generated"] += 1
            continue
        counts["generated"] += 1
        counts["completed"] += record["completion"] == "EOS"
        counts["truncated"] += record["completion"] in {"LENGTH_TRUNCATED", "CONTEXT_LIMIT"}
        if record.get("label") is None:
            counts["unscorable"] += 1
            counts[record.get("exclusion") or "not_scored"] += 1
        else:
            counts["correct" if record["label"] == 0 else "wrong"] += 1
        counts["no_steps"] += not record.get("boundaries", {}).get("steps", [])
        counts["format_error_trajectories"] += bool(record.get("boundaries", {}).get("format_errors"))
        if not valid_features(path, record):
            counts["features_missing_or_corrupt"] += 1
            continue
        if record.get("label") is None:
            continue
        parts, layers = [], []
        for entry in record["feature_files"]:
            with np.load(path.parent / entry["name"], allow_pickle=False) as data:
                parts.append(data["hidden"])
                layers.extend(data["layers"].tolist())
        hidden = np.concatenate(parts, axis=0)
        if layers != list(range(1, len(layers) + 1)):
            raise ValueError("Feature layer indices are not contiguous hidden_states[1..L]")
        steps = record["boundaries"]["steps"]
        expected_positions = [len(record["prompt_ids"]) + s["end_token_index"] for s in steps]
        if record["boundaries"]["end"]:
            expected_positions.append(len(record["prompt_ids"]) + record["boundaries"]["end"]["end_token_index"])
        with np.load(path.parent / record["feature_files"][0]["name"], allow_pickle=False) as data:
            if data["positions"].tolist() != expected_positions:
                raise ValueError("Saved feature positions differ from token metadata")
        if hidden.shape[1] != len(expected_positions):
            raise ValueError("Feature/position count differs")
        examples.append({"id": row["unique_id"], "y": record["label"], "steps": steps,
                         "hidden": hidden, "has_end": record["boundaries"]["end"] is not None,
                         "record": record})
    counts["analyzable_end"] = sum(x["has_end"] for x in examples)
    counts["analyzable_intermediate"] = sum(bool(x["steps"]) for x in examples)
    return examples, dict(counts)


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


def write_csv(path, rows, fields):
    b = io.StringIO(newline="")
    writer = csv.DictWriter(b, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, b.getvalue().encode())


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


def plot_metrics(metrics, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for offset, method, color in ((-0.16, "A", "#4c78a8"), (0, "B", "#e45756"), (0.16, "C", "#72b7b2")):
        for i in (1, 2, 3):
            value = metrics[f"step_{i}"].get("methods", {}).get(method, {})
            center, ci = value.get("auroc"), value.get("ci95")
            if center is None or ci is None:
                ax.text(i + offset, 0.08 + (offset + .16) / 4, "NA", ha="center", fontsize=8, color=color)
            else:
                ax.vlines(i + offset, ci[0], ci[1], color=color)
                ax.plot(i + offset, center, "o", color=color)
        ax.plot([], [], "o", color=color, label=method)
    ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
    ax.set(xticks=[1, 2, 3], xlabel="Effective intermediate step", ylabel="Test AUROC (95% question bootstrap CI)", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=160)
    atomic_bytes(run / "step_auroc.png", b.getvalue())
    plt.close(fig)


def metric_text(value):
    if not value or value.get("auroc") is None:
        return "NA"
    ci = value.get("ci95")
    return f"{value['auroc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{value['auroc']:.4f} [NA]"


def report(run, manifest, counts, metrics=None, rows=None, coverage=None, failures=None, reason=None):
    cfg = manifest["config"]
    fit = read_json(run / "fit_status.json", {})
    lines = ["# 第二阶段：逐步 hidden 最终答错风险探针", "",
             f"运行：`{run.name}`；报告更新于 {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。", "",
             "标签是完整轨迹最终答错（1）或答对（0）。主指标仅用于完成且可评分轨迹的正常中途步骤；不是局部错误标签。", "",
             f"模型 `{cfg['model']}`，revision `{cfg['revisions']['model']}`；tokenizer `{cfg['revisions']['tokenizer']}`。",
             f"数据 `{cfg['dataset']}` / test，revision `{cfg['revisions']['dataset']}`。",
             f"按 unique_id 排序后用种子 {cfg['seed']} 洗牌：100 train / 200 test / 10 smoke，剩余 190 不使用；原始字段见 manifest.json。",
             "BF16 / 单卡 / batch 1 / SDPA / greedy；4096 新 token、8192 总 token；C=0.1、max_iter=2000；所有 Transformer 层分别做训练内五折选层。", "",
             "这是采用逐步训练、题目权重、明确 Step 标记和独立测试的适配实验，不宣称完全复现论文。", "",
             "## 实际环境", "", "```json", json.dumps(manifest["environment"], ensure_ascii=False, indent=2), "```", "",
             "## 采集与排除", "", "```json", json.dumps(counts, ensure_ascii=False, indent=2), "```", ""]
    if reason:
        lines += [f"未完成原因：{reason}", ""]
    lines += ["## 自检与备份", "", "```json", json.dumps({
        "cpu_self_check": read_json(HERE / "self_check.json"),
        "additional_implementation_checks": read_json(HERE / "implementation_checks.json"),
        "smoke": read_json(run / "smoke_checks.json"), "resume": read_json(run / "resume_check.json"),
        "backup": read_json(run / "backup.json")}, ensure_ascii=False, indent=2), "```", ""]
    if metrics:
        lines += ["## 独立测试", "", f"选层（A/B 独立选择）：`{fit.get('selected', {})}`。",
                  f"训练覆盖：`{coverage}`；失败：`{failures}`。", "",
                  "| 位置 | 题数（正确/错误） | A AUROC [95% CI] | B AUROC [95% CI] | C AUROC [95% CI] |",
                  "|---|---|---|---|---|"]
        for key in ("step_1", "step_2", "step_3", "intermediate_all", "full_trajectory_end"):
            item = metrics[key]
            lines.append(f"| {key} | {item['questions']} ({item['correct']}/{item['wrong']}) | "
                         + " | ".join(metric_text(item["methods"].get(m)) for m in ("A", "B", "C")) + " |")
        lines += ["", "C 不用于完整轨迹末尾对照；A/B 末尾单独报告。所有中途比较使用共同有效位置。", "",
                  "每题总权重为 1；bootstrap 按题重抽 1000 次，整题全部步骤随同抽取，单类重复样本跳过。有效次数与 paired AUROC 差及其 CI 见 metrics.json。", "",
                  "```json", json.dumps({k: metrics[k]["paired"] for k in ("step_1", "step_2", "step_3", "intermediate_all")}, indent=2), "```", "",
                  "![前 3 个有效中途步骤](step_auroc.png)", "", "## 代表性风险轨迹", ""]
        by_id = {}
        for row in rows or []:
            by_id.setdefault(row["id"], []).append(row)
        if "B" in fit.get("selected", {}):
            for label in (0, 1):
                candidates = sorted([r for r in rows or [] if r["effective_step"] == 1 and r["label"] == label], key=lambda r: r["B"])
                for first in ([candidates[0], candidates[-1]] if len(candidates) > 1 else candidates):
                    trajectory = ", ".join(f"{r['effective_step']}:{r['B']:.3f}" for r in by_id[first["id"]])
                    lines += [f"- `{first['id']}`，最终标签 {label}，B 的逐步风险：{trajectory}。"]
        main = metrics["intermediate_all"]
        b = main["methods"].get("B", {})
        delta = main["paired"].get("B-C", {})
        supported = bool(b.get("ci95") and b["ci95"][0] > .5 and delta.get("ci95") and delta["ci95"][0] > 0)
        lines += ["", "## 结论", "",
                  ("此设置下，B 在全部中途位置提供了超过随机与简单进度基线的最终答错风险信号（两项 bootstrap CI 均支持）。"
                   if supported else "此设置下，未同时获得超过随机与简单进度基线的明确证据；低分、反向或跨零区间按原方向报告。"), ""]
    else:
        lines += ["AUROC / CI / 所选层：NA；正式数据不足或未采集，不能回答是否存在信号。", ""]
    lines += ["这项实验不能识别首个错误步骤、判断当前检查是否值得、证明降低完成时间或推广到其他模型。",
              "离线重前向耗时不代表在线提取成本；生成、重前向与探针评分耗时分别记录。", "",
              "步骤端点对齐原始 token，跨界 token 向前对齐；包含 boxed 或明确最终答案的步骤排除，最后正常步骤保留 near_end 标记。",
              "hidden_states[1..L] 排除 embedding，Qwen2 的索引 L 含最终 norm；轨迹末尾取 EOS/控制 token 前最后完整内容 token。", "",
              "评分仅解析单独 Final answer 行的最后完整 boxed；Math-Verify 标准答案包装为 $...$，无字符串 fallback，解析/比较超时单独排除。", "",
              "## 文件与恢复", "", "配置与环境见 config.json / manifest.json；运行日志见 events.jsonl；探针参数见 probe_{A,B,C}.npz/json。",
              "分支：[experiment/step-hidden-probe](https://github.com/WanderingAlgebra/VeriServe/tree/experiment/step-hidden-probe)。",
              f"本地 HEAD：`{git('rev-parse', 'HEAD')}`；远端备份状态与提交以 backup.json 为准。", "",
              "从仓库根目录执行（替换成可用 Python 环境）：", "", "```bash",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume", "```", ""]
    atomic_bytes(run / "report.md", "\n".join(lines).encode())


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


def self_check(cfg):
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    for gold, prediction, label in ((r"\frac{1}{2}", "0.5", 0), (r"\sqrt{8}", r"2\sqrt{2}", 0),
                                     ("x+x", "2x", 0), ("2", "3", 1)):
        result = grade("Step 1: compute\n\nFinal answer: \\boxed{" + prediction + "}", gold, "EOS", 5)
        check(f"grade {gold} vs {prediction}", result["label"] == label)
    check("missing boxed", grade("Final answer: 2", "2", "EOS", 5)["label"] is None)
    check("incomplete boxed", grade(r"Final answer: \boxed{\frac{1}{2}", "0.5", "EOS", 5)["label"] is None)
    check("last complete nested box", boxed(r"\boxed{3} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}")
    check("only final line graded", grade("Step 1: \\boxed{2}\nFinal answer: \\boxed{3}", "2", "EOS", 5)["label"] == 1)
    check("truncation excluded", grade(r"Final answer: \boxed{2}", "2", "LENGTH_TRUNCATED", 5)["label"] is None)
    check("post-final text excluded", grade("Final answer: \\boxed{2}\nmore reasoning", "2", "EOS", 5)["label"] is None)

    class Tokenizer:
        all_special_ids = [999]
        def __init__(self, pieces):
            self.pieces = pieces
        def decode(self, ids, **kwargs):
            return "".join(self.pieces[i] if i != 999 else "<eos>" for i in ids)
    pieces = ["Step 1:", " Work", ".\n\nSt", "ep 2:", " More", ".", "\n\nStep 3:",
              " The final answer is ", r"\boxed{2}", "\n\nFinal answer:", r" \boxed{2}"]
    tok = Tokenizer(pieces)
    text, bounds = boundaries(tok, list(range(len(pieces))) + [999])
    check("cross-boundary token aligned backward", bounds["steps"][0]["end_token_index"] == 1
          and bounds["steps"][0]["boundary_aligned_back"])
    check("second-step original token index", bounds["steps"][1]["end_token_index"] == 5)
    check("answer step excluded", len(bounds["steps"]) == 2 and len(bounds["excluded_steps"]) == 1)
    check("last content token before EOS", bounds["end"]["end_token_index"] == 10)
    check("near-end retained", bounds["steps"][-1]["near_end"])
    _, bad = boundaries(Tokenizer(["Step 1: x\nStep 3: y"]), [0])
    check("anomalous numbering not guessed", not bad["steps"] and bool(bad["format_errors"]))
    unicode_tok = Tokenizer(["Step 1: ", "\ufffd", "\nFinal answer: \\boxed{2}"])
    _, bad = boundaries(unicode_tok, [0, 1, 2])
    check("incomplete Unicode endpoint excluded", not bad["steps"])
    _, mixed = boundaries(Tokenizer(["Step 1: x", "\nFinal answer:", " \\boxed{2}\n", "\n\n"]), [0, 1, 2, 3, 999])
    check("mixed final content and newline token retained", mixed["end"]["end_token_index"] == 2)
    class ByteTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")
    for character in ("∛", "𝑥", "∞", "汉", "🧮"):
        pieces = [b"Step 1: "] + [bytes([byte]) for byte in character.encode()] + [b"\nFinal answer: \\boxed{2}"]
        _, completed = boundaries(ByteTokenizer(pieces), list(range(len(pieces))))
        check(f"completed Unicode endpoint {character}", len(completed["steps"]) == 1
              and completed["steps"][0]["end_token_index"] == len(pieces) - 2)

    examples = []
    for i in range(24):
        y = i % 2
        steps = [{"step_number": j + 1, "prefix_token_count": 10 * (j + 1), "near_end": j == i % 3}
                 for j in range(1 + i % 3)]
        hidden = np.array([[[4 * y + j / 10, i / 100] for j in range(len(steps) + 1)],
                           [[4 * y + j / 10, i / 100] for j in range(len(steps) + 1)]], dtype=np.float32)
        examples.append({"id": str(i), "y": y, "steps": steps, "hidden": hidden, "has_end": True})
    x, y, w, ids = method_rows(examples, "B", 1)
    check("each question total weight one", all(abs(sum(w[j] for j in range(len(ids)) if ids[j] == uid) - 1) < 1e-12 for uid in set(ids)))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x, sample_weight=w)
    check("weighted scaler", np.allclose(scaler.mean_, np.average(x, axis=0, weights=w)))
    with tempfile.TemporaryDirectory(prefix="stage2-cpu-self-check-") as tmp:
        folder = Path(tmp)
        test_cfg = {**cfg, "bootstrap": 20}
        probes, _, failures = select_probes(examples, test_cfg, folder)
        check("A/B/C fitting and grouped CV", len(probes) == 3 and not failures)
        check("layer tie chooses smaller index", probes["A"]["layer"] == probes["B"]["layer"] == 1)
        metrics, _ = evaluate(examples, probes, test_cfg, folder)
        check("weighted evaluation and paired bootstrap", metrics["intermediate_all"]["methods"]["B"]["auroc"] == 1
              and metrics["intermediate_all"]["paired"]["B-C"]["bootstrap_valid"] == 20)
        one_class = bootstrap_metrics([{"id": "x", "label": 0, "A": .2, "B": .3, "C": .4}], test_cfg)
        check("single-class bootstrap reports NA", one_class["methods"]["B"]["auroc"] is None
              and one_class["methods"]["B"]["bootstrap_skipped"] == 20)
        row = {"unique_id": "self-check"}
        path = record_path(folder, "smoke", row["unique_id"])
        path.parent.mkdir(parents=True)
        feature = path.with_suffix(".npz")
        save_npz(feature, hidden=np.ones((2, 1, 2), dtype=np.float32), layers=np.array([1, 2]), positions=np.array([1]))
        record = {"prompt_ids": [1], "generated_ids": [2], "stage": "FEATURES",
                  "feature_source_code": source_code(),
                  "token_sha256": stable_hash({"prompt_ids": [1], "generated_ids": [2]}),
                  "feature_files": [{"name": feature.name, "sha256": digest(feature)}]}
        atomic_json(path, record)
        before = snapshot_hashes(folder, "smoke")
        _, changed = collect_one(cfg, folder, "smoke", row, None, None, {})
        check("resume skips complete JSON and NPZ without generation", not changed and before == snapshot_hashes(folder, "smoke"))
        atomic_bytes(feature, b"damaged")
        check("damaged NPZ detected", not valid_features(path, record))
        atomic_bytes(path, b"damaged JSON")
        try:
            read_record(path, folder)
        except ValueError:
            check("unrecoverable JSON never triggers regeneration", True)
        else:
            check("unrecoverable JSON never triggers regeneration", False)
        for invalid in ({}, [], None, False, 0):
            atomic_json(path, invalid)
            try:
                read_record(path, folder)
            except ValueError:
                pass
            else:
                raise AssertionError("Existing invalid JSON must never be treated as a missing record")
        check("valid JSON with invalid structure is preserved and rejected", True)
        check("no-step causal check is not a failure", causal_check(None, {"boundaries": {"steps": []}}, cfg)["passed"] is None)
    result = {"passed": True, "checked_at": now(), "checks": checks, "environment": environment(),
              "script_sha256": digest(__file__),
              "note": "Synthetic CPU checks only; actual model causal/GPU and remote LFS checks belong to smoke"}
    atomic_json(HERE / "self_check.json", result)
    print(f"SELF_CHECK passed ({len(checks)} checks)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--prepare-only", action="store_true", help="Resolve SHA/splits and save manifest before first push")
    mode.add_argument("--backup-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Pause smoke after N questions to demonstrate restart")
    args = parser.parse_args()
    if args.self_check:
        self_check(read_json(args.config))
        return
    if args.stop_after is not None and (not args.smoke or args.stop_after < 1):
        parser.error("--stop-after is only a positive smoke question count")
    cfg, run, manifest = prepare(args.config)
    print(f"RUN {run.name}", flush=True)
    if args.prepare_only:
        report(run, manifest, {"formal_planned": 300, "formal_generated": len(list(run.glob('records/train/*.json'))) + len(list(run.glob('records/test/*.json')))},
               reason="Prepared; collection and remote backup have not yet been verified")
        return
    if args.fit_only:
        fit_only(cfg, run, manifest)
        return
    if args.backup_only:
        backup(run, "stage2: checkpoint frozen probe experiment")
        return
    if not args.resume and list(run.glob("records/**/*.json")):
        parser.error("Existing records: specify --resume to reuse original tokens")
    model, tokenizer = None, None
    try:
        # Initial code/manifest push also retries any failed previous batch before collecting again.
        backup(run, "stage2: prepare code and frozen split manifest")
        if not args.smoke:
            smoke = read_json(run / "smoke_checks.json", {})
            resume = read_json(run / "resume_check.json", {})
            if not smoke.get("passed") or not resume.get("passed"):
                raise RuntimeError("Run all ten smoke questions and a separate --smoke --resume restart first")
            if smoke.get("source_code") != source_code() or resume.get("source_code") != source_code():
                raise RuntimeError("Smoke/resume checks refer to different code; rerun --smoke --resume")
            if not read_json(run / "backup.json", {}).get("feature_remote_verification"):
                raise RuntimeError("Smoke NPZ has not been retrieved from the remote and verified")
            current = source_code()
            if manifest.get("formal_frozen_code") and manifest["formal_frozen_code"] != current:
                raise RuntimeError("Collection code changed after freezing; inspect changes and use a new run_id")
            manifest["formal_frozen_code"] = current
            manifest.update(current)
            manifest["code_commit"] = git("rev-parse", "HEAD")
            manifest.setdefault("formal_frozen_at", now())
            atomic_json(run / "manifest.json", manifest)
            backup(run, "stage2: freeze formal collection after independent smoke")
        splits = ["smoke"] if args.smoke else ["train", "test"]
        before = snapshot_hashes(run, "smoke") if args.smoke else {}
        hardware = environment(gpu=True)
        checks = read_json(run / "smoke_checks.json", {"questions": {}})
        if checks.get("source_code") != source_code():
            checks = {"questions": {}, "source_code": source_code()}
        completed = 0
        for split in splits:
            for row in manifest["splits"][split]:
                path = record_path(run, split, row["unique_id"])
                old = read_record(path, run)
                needs_features = not (old and old.get("stage") == "FEATURES" and valid_features(path, old)
                                      and old.get("feature_source_code") == source_code())
                needs_causal = args.smoke and row["unique_id"] not in checks["questions"]
                if (needs_features or needs_causal) and model is None:
                    model, tokenizer = load_model(cfg)
                    event(run, "MODEL_LOADED", hardware=hardware,
                          layers=model.config.num_hidden_layers, hidden_size=model.config.hidden_size,
                          attention=model.config._attn_implementation, bf16_forward=True)
                record, changed = collect_one(cfg, run, split, row, model, tokenizer, hardware)
                completed += 1
                if args.smoke and needs_causal:
                    checks["questions"][row["unique_id"]] = causal_check(model, record, cfg)
                    applicable = [v for v in checks["questions"].values() if v.get("applicable", True)]
                    checks["passed"] = (len(checks["questions"]) == cfg["split_sizes"]["smoke"]
                                        and bool(applicable) and all(v["passed"] for v in applicable))
                    checks["not_applicable"] = len(checks["questions"]) - len(applicable)
                    checks["updated_at"] = now()
                    atomic_json(run / "smoke_checks.json", checks)
                    if checks["questions"][row["unique_id"]]["passed"] is False:
                        raise RuntimeError("Causal smoke check failed; preserve errors and inspect before formal collection")
                if not args.smoke and completed % cfg["backup_every"] == 0:
                    backup(run, f"stage2: persist formal questions through {completed}/300")
                if args.stop_after and completed >= args.stop_after:
                    event(run, "SMOKE_PAUSED", completed=completed)
                    backup(run, "stage2: smoke pause before restart verification")
                    print("SMOKE_PAUSED; restart with --smoke --resume", flush=True)
                    return
        if args.smoke:
            after = snapshot_hashes(run, "smoke")
            audit = {"passed": bool(before) and all(after.get(k) == v for k, v in before.items()),
                     "source_code": source_code(),
                     "unchanged_files": len(before), "hashes_before": before,
                     "hashes_after_for_existing": {k: after.get(k) for k in before}, "checked_at": now()}
            atomic_json(run / "resume_check.json", audit)
            if before and not audit["passed"]:
                raise RuntimeError("Previously complete smoke JSON/feature changed during resume")
            _, smoke_counts = load_examples(run, manifest, "smoke")
            report(run, manifest, {"smoke": smoke_counts, "formal_generated": 0, "formal_planned": 300},
                   reason="Smoke complete; formal collection pending" if checks["passed"] else "Smoke checks incomplete")
            backup(run, "stage2: save independent smoke and restart audit")
            return
        del model, tokenizer
        model, tokenizer = None, None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        event(run, "MODEL_RELEASED_BEFORE_CPU_FIT")
        fit_only(cfg, run, manifest)
        backup(run, "stage2: save frozen probes and independent test report")
    except (Exception, KeyboardInterrupt) as exc:
        event(run, "STOPPED", error=safe_error(exc), exception_type=type(exc).__name__)
        counts = {}
        for split in ("train", "test", "smoke"):
            _, counts[split] = load_examples(run, manifest, split)
        report(run, manifest, counts, reason=safe_error(exc))
        raise
    finally:
        if model is not None:
            del model, tokenizer
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
