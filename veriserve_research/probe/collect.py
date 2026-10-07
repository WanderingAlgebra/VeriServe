"""Probe experiment orchestration; library consumers use materials/fit/inference."""
from __future__ import annotations
import argparse
import gc
import random
import tempfile
import time
from pathlib import Path
import numpy as np
from veriserve_research import ROOT
from ..artifacts.io import atomic_json, read_json, stable_hash, digest, atomic_bytes, save_npz, now, git, event, safe_error
from ..artifacts.provenance import environment, resolve_revision, SHA, run_identity, source_hashes, manifest_version, require_current
from ..artifacts.backup import backup_probe as backup
from ..trajectory import grade, boxed, boundaries, decode
from ..inference import load_model, extract, causal_check
from .materials import record_path, valid_features, read_record, load_examples
from .fit import select_probes, method_rows, bootstrap_metrics, evaluate, fit_only
from .report import report
HERE = ROOT / "stages/stage2"


def source_code():
    hashes = source_hashes("probe")
    return {"script_sha256": stable_hash(hashes), "execution_sources": hashes,
            "common_sha256": digest(ROOT / "stages/stage1/veriserve/common.py")}


def prepare(config_path):
    cfg = read_json(config_path)
    original_config_hash = stable_hash(cfg)
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
    if stable_hash(cfg) != original_config_hash:
        atomic_json(config_path, cfg)
    cfg_hash = stable_hash(cfg)
    run = HERE / "runs" / run_identity(cfg, "probe", revisions)
    run.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / "manifest.json")
    if manifest:
        require_current(manifest, "probe")
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
    manifest = {**manifest_version("probe"), "run_id": run.name, "created": now(), "config": cfg, "config_hash": cfg_hash,
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
    record["feature_extraction"] = "independent_original_prefixes; BF16 decoder use_cache=False"
    atomic_json(path, record)
    print(f"DONE {split} {row['unique_id']} label={record['label']} exclusion={record['exclusion']} "
          f"steps={len(bounds['steps'])} tokens={len(record['generated_ids'])}", flush=True)
    return record, True


def snapshot_hashes(run, split):
    hashes = {}
    for path in sorted((run / "records" / split).glob("*.json")):
        record = read_record(path, run)
        if (record and record.get("stage") == "FEATURES" and valid_features(path, record)
                and record.get("feature_source_code") == source_code()):
            for p in [path, *[path.parent / f["name"] for f in record["feature_files"]]]:
                hashes[str(p.relative_to(run))] = digest(p)
    return hashes


def self_check(cfg, output_dir=None):
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
    if output_dir is not None:
        atomic_json(Path(output_dir) / "self_check.json", result)
    print(f"SELF_CHECK passed ({len(checks)} checks)", flush=True)
    return result


def main(argv=None):
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
    args = parser.parse_args(argv)
    if args.self_check:
        with tempfile.TemporaryDirectory(prefix="veriserve-probe-check-") as folder:
            self_check(read_json(args.config), folder)
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
        if args.smoke:
            self_check(cfg, run)
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
