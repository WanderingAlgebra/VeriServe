from __future__ import annotations
import csv
import math
import re
import time
from pathlib import Path
import numpy as np
from veriserve_research.artifacts.io import read_json, stable_hash, atomic_bytes, save_npz, digest
from veriserve_research.trajectory import STEP, FINAL, ANSWER_STEP, boundaries, boxed, decode
from veriserve_research.probe.materials import record_path
from ...artifacts.materials import material_key


def load_source(stage2_run):
    """Read manifest order directly, including records without a stage2 label."""
    folder = Path(stage2_run)
    manifest = read_json(folder / "manifest.json")
    cfg = manifest["config"]
    if (cfg["model"] != "Qwen/Qwen2.5-7B-Instruct" or cfg["do_sample"] is not False
            or cfg["dtype"] != "bfloat16" or cfg["attention"] != "sdpa" or cfg["batch_size"] != 1):
        raise ValueError("Source must be the frozen Qwen BF16 SDPA batch=1 greedy run")
    if len(manifest["splits"]["test"]) != 200 or len(manifest["splits"]["smoke"]) < 4:
        raise ValueError("Expected the original 200 test questions and at least four smoke questions")
    splits = {"pilot": manifest["splits"]["smoke"][:4], "test": manifest["splits"]["test"]}
    ids = [row["unique_id"] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)) or set(ids) & {r["unique_id"] for r in manifest["splits"]["train"]}:
        raise ValueError("Duplicate questions or train/test overlap in source manifest")
    return manifest, splits


def _probe(folder):
    meta = read_json(folder / "probe_B.json")
    with np.load(folder / "probe_B.npz", allow_pickle=False) as data:
        probe = {k: data[k].astype(np.float64) for k in ("mean", "scale", "w", "b")}
    if (meta.get("layer") != 19 or meta.get("method") != "B"
            or not np.isclose(float(probe["b"]), meta["b"], atol=0, rtol=0)
            or probe["mean"].ndim != 1 or probe["mean"].shape != probe["scale"].shape
            or probe["mean"].shape != probe["w"].shape
            or not all(np.isfinite(v).all() for v in probe.values())
            or not np.all(probe["scale"] > 0)):
        raise ValueError("Frozen B JSON/NPZ mismatch or invalid hidden_states[19] parameters")
    return probe


def _reference_bounds(record, tokenizer):
    """Validate completion/format without consulting the gold or original outcome."""
    if record.get("completion") != "EOS":
        raise ValueError("REFERENCE_" + str(record.get("completion", "INCOMPLETE")))
    prompt, generated = record.get("prompt_ids"), record.get("generated_ids")
    if any(not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids)
           for ids in (prompt, generated)):
        raise ValueError("INVALID_REFERENCE_TOKENS")
    if record.get("token_sha256") != stable_hash({"prompt_ids": prompt, "generated_ids": generated}):
        raise ValueError("REFERENCE_TOKEN_HASH_MISMATCH")
    if tokenizer is not None:
        text, bounds = boundaries(tokenizer, generated)
        if decode(tokenizer, generated) != record.get("raw_text"):
            raise ValueError("REFERENCE_TEXT_TOKEN_MISMATCH")
        saved = record.get("boundaries")
        if saved is not None and saved != bounds:
            raise ValueError("REFERENCE_BOUNDARY_MISMATCH")
    else:
        bounds = record.get("boundaries", {})
        end = bounds.get("end")
        if not end:
            raise ValueError("MISSING_REFERENCE_BOUNDARIES")
        # The frozen model is Qwen; its terminal control suffix is not reasoning.
        raw = record.get("raw_text", "")
        text = raw[:end["char_end"]]
        suffix = raw[end["char_end"]:]
        if not suffix or re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)", "", suffix):
            raise ValueError("INVALID_REFERENCE_EOS_SUFFIX")
        if re.search(r"<\|(?:im_start|im_end|endoftext)\|>", text):
            raise ValueError("REFERENCE_INTERIOR_CONTROL_TOKEN")
    if "[Verification feedback]" in text:
        raise ValueError("INTERVENED_REFERENCE")
    if bounds.get("format_errors"):
        raise ValueError("; ".join(bounds["format_errors"]))
    finals, markers = list(FINAL.finditer(text)), list(STEP.finditer(text))
    if (len(finals) != 1 or text[finals[0].end():].strip()
            or boxed(finals[0].group()) is None):
        raise ValueError("REFERENCE_FINAL_FORMAT_ERROR")
    if [int(m.group(1)) for m in markers] != list(range(1, len(markers) + 1)):
        raise ValueError("REFERENCE_NONSEQUENTIAL_STEPS")
    ordinary = []
    for i, marker in enumerate(markers):
        stop = min(markers[i + 1].start() if i + 1 < len(markers) else len(text), finals[0].start())
        if stop <= marker.start() or not text[marker.end():stop].strip():
            raise ValueError("REFERENCE_UNCLOSED_OR_EMPTY_STEP")
        segment = text[marker.start():stop].rstrip()
        if not re.search(r"\\boxed\s*\{", segment) and not ANSWER_STEP.search(segment):
            ordinary.append((int(marker.group(1)), segment, marker.start()))
    steps = bounds.get("steps", [])
    if [(s["step_number"], s["text"], s["char_start"]) for s in steps] != ordinary:
        raise ValueError("REFERENCE_CANDIDATE_METADATA_MISMATCH")
    for step in steps:
        count = step["prefix_token_count"]
        if (type(count) is not int or not 0 < count <= len(generated)
                or step["end_token_index"] != count - 1
                or step["char_end"] != step["char_start"] + len(step["text"])):
            raise ValueError("REFERENCE_ENDPOINT_MISMATCH")
    if len(steps) < 2:
        raise ValueError("FEWER_THAN_TWO_COMPLETE_NORMAL_STEPS")
    return bounds


def _saved_vectors(path, record, width, material):
    """NPZ layers/positions are authoritative; never infer layer 19 by array offset."""
    if not record.get("feature_files"):
        raise ValueError("FEATURES_MISSING")
    if not record.get("feature_extraction", "").startswith("independent_original_prefixes"):
        raise ValueError("FEATURE_PREFIX_EXTRACTION_UNVERIFIED")
    bounds = record["boundaries"]
    positions = [len(record["prompt_ids"]) + s["end_token_index"] for s in bounds["steps"]]
    if bounds.get("end") is not None:
        positions.append(len(record["prompt_ids"]) + bounds["end"]["end_token_index"])
    selected = None
    seen = set()
    for entry in record["feature_files"]:
        feature_path = path.parent / entry["name"]
        actual = digest(feature_path)
        material[material_key(feature_path)] = actual
        if actual != entry["sha256"]:
            raise ValueError("FEATURE_HASH_MISMATCH")
        with np.load(feature_path, allow_pickle=False) as data:
            layers, stored_positions = data["layers"], data["positions"]
            if (layers.ndim != 1 or not np.issubdtype(layers.dtype, np.integer)
                    or stored_positions.ndim != 1 or not np.issubdtype(stored_positions.dtype, np.integer)
                    or stored_positions.tolist() != positions
                    or any(int(layer) < 1 or int(layer) in seen for layer in layers)
                    or len(set(layers.tolist())) != len(layers)):
                raise ValueError("FEATURE_LAYERS_OR_POSITIONS_MISMATCH")
            seen.update(int(layer) for layer in layers)
            indices = np.flatnonzero(layers == 19)
            if len(indices):
                hidden = data["hidden"]
                if hidden.shape != (len(layers), len(positions), width):
                    raise ValueError("FEATURE_SHAPE_MISMATCH")
                selected = hidden[int(indices[0]), :len(bounds["steps"])].astype(np.float64)
    if selected is None or not np.isfinite(selected).all():
        raise ValueError("FEATURE_LAYER19_MISSING_OR_NONFINITE")
    return selected


def _saved_scores(folder, uid):
    path = folder / "test_predictions.csv"
    if not path.exists():
        return {}, None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {(int(r["step_number"]), int(r["prefix_token_count"])): float(r["B"])
                for r in csv.DictReader(handle) if r["id"] == uid}
    return rows, path


def _cached_vectors(path, record, bounds, width, revision):
    with np.load(path, allow_pickle=False) as data:
        expected = [len(record["prompt_ids"]) + s["end_token_index"] for s in bounds["steps"]]
        if (data["layers"].tolist() != [19] or not np.issubdtype(data["layers"].dtype, np.integer)
                or not np.issubdtype(data["positions"].dtype, np.integer)
                or data["positions"].tolist() != expected
                or str(data["token_sha256"].item()) != record["token_sha256"]
                or str(data["model_revision"].item()) != revision
                or str(data["extraction"].item()) != "independent_original_prefixes"):
            raise ValueError("CACHED_FEATURE_REFERENCE_MISMATCH")
        hidden = data["hidden"]
        if hidden.shape != (1, len(expected), width) or not np.isfinite(hidden).all():
            raise ValueError("CACHED_FEATURE_SHAPE_OR_VALUES_INVALID")
        return hidden[0].astype(np.float64)


def _finish(result, started, feature_seconds=0.):
    result["reason"] = "; ".join(result["exclusion_reasons"]) or None
    result["timings"] = {"selection_wall_seconds": time.perf_counter() - started,
                         "offline_feature_seconds": feature_seconds}
    result["selection_sha256"] = stable_hash(result)
    return result


def freeze_reference(stage2_run, split, row, tokenizer=None, model=None, reference=None,
                     feature_cache_path=None):
    """Return a serializable frozen selection before executing either intervention.

    `reference` is an untouched stream recollection from run_timing.Session. Missing
    saved features use the existing selected-layer independent-prefix forward;
    without a GPU model, NEEDS_FEATURES preserves the pending work explicitly.
    """
    started = time.perf_counter()
    folder = Path(stage2_run)
    phase = "pilot" if split in {"smoke", "pilot"} else split
    source_split = "smoke" if phase == "pilot" else split
    if source_split not in {"smoke", "test"}:
        raise ValueError("Only the original smoke pilot and test are allowed")
    manifest, rows = load_source(folder)
    if row["unique_id"] not in {r["unique_id"] for r in rows[phase]}:
        raise ValueError("Question is outside the frozen manifest cohort")
    material = {material_key(folder / name): digest(folder / name)
                for name in ("manifest.json", "probe_B.json", "probe_B.npz")}
    path = record_path(folder, source_split, row["unique_id"])
    original = read_json(path)
    if original is not None:
        material[material_key(path)] = digest(path)
    result = {"unique_id": row["unique_id"], "phase": phase, "source_split": source_split,
              "status": "EXCLUDED", "exclusion_reasons": [], "positions": {},
              "source_material_hashes": material, "hidden_state_index": 19,
              "selection_semantics": "offline whole-trajectory B error-risk extrema; not online or local error"}
    record = reference if reference is not None else original
    if record is None:
        result.update(status="NEEDS_REFERENCE", exclusion_reasons=["SOURCE_RECORD_MISSING"])
        return _finish(result, started)
    if reference is None and record.get("config_hash") != manifest["config_hash"]:
        result["exclusion_reasons"] = ["SOURCE_CONFIG_HASH_MISMATCH"]
        return _finish(result, started)
    if reference is not None and record.get("reference_origin") != "stage3_stream_unintervened_greedy":
        raise ValueError("Replacement reference must record untouched stage3 streaming origin")
    try:
        bounds = _reference_bounds(record, tokenizer)
    except (ValueError, KeyError, TypeError) as exc:
        result["exclusion_reasons"] = [str(exc)]
        return _finish(result, started)
    probe = _probe(folder)
    result.update(prompt_ids=record["prompt_ids"], generated_ids=record["generated_ids"],
                  reference_token_sha256=record["token_sha256"],
                  reference_origin=record.get("reference_origin", "stage2_original_unintervened_greedy"),
                  reference_sha256=stable_hash(record), candidates=[], feature_repair_reasons=[])
    feature_started = time.perf_counter()
    vectors = np.full((len(bounds["steps"]), len(probe["w"])), np.nan)
    reused = np.zeros(len(bounds["steps"]), dtype=bool)
    feature_sources = [None] * len(bounds["steps"])
    try:
        if (original is None or original.get("config_hash") != manifest["config_hash"]
                or original.get("token_sha256") != stable_hash({k: original[k] for k in ("prompt_ids", "generated_ids")})):
            raise ValueError("ORIGINAL_FEATURE_TOKEN_PROVENANCE_UNAVAILABLE")
        saved_vectors = _saved_vectors(path, original, len(probe["w"]), material)
        original_steps = {(s["step_number"], s["prefix_token_count"]): i
                          for i, s in enumerate(original["boundaries"]["steps"])}
        for i, step in enumerate(bounds["steps"]):
            count = step["prefix_token_count"]
            old_index = original_steps.get((step["step_number"], count))
            if (old_index is not None and original["prompt_ids"] == record["prompt_ids"]
                    and original["generated_ids"][:count] == record["generated_ids"][:count]):
                vectors[i], reused[i] = saved_vectors[old_index], True
                feature_sources[i] = "stage2_saved_hidden_states[19]"
        if not reused.all():
            result["feature_repair_reasons"].append("REFERENCE_PREFIX_TOKENS_CHANGED")
    except (OSError, ValueError, KeyError, EOFError) as exc:
        result["feature_repair_reasons"].append(str(exc))
    missing = np.flatnonzero(~np.isfinite(vectors).all(axis=1)).tolist()
    result["reused_original_feature_prefixes"] = int(reused.sum())
    if missing:
        cache = Path(feature_cache_path) if feature_cache_path is not None else None
        if cache is not None and cache.exists():
            try:
                cached = _cached_vectors(cache, record, bounds, len(probe["w"]), manifest["revisions"]["model"])
                for i in missing:
                    vectors[i] = cached[i]
                    feature_sources[i] = "stage3_cached_independent_original_prefixes_hidden_states[19]"
                missing = []
            except (OSError, ValueError, KeyError, EOFError) as cache_error:
                result["feature_repair_reasons"].append(str(cache_error))
        if missing:
            if model is None:
                result.update(status="NEEDS_FEATURES", exclusion_reasons=["SELECTED_LAYER_FEATURES_REQUIRED"])
                result["missing_feature_step_numbers"] = [bounds["steps"][i]["step_number"] for i in missing]
                return _finish(result, started, time.perf_counter() - feature_started)
            import torch
            from veriserve_research.inference import feature
            torch.cuda.synchronize()
            for i in missing:
                step = bounds["steps"][i]
                vectors[i] = feature(model, record["prompt_ids"] + record["generated_ids"][:step["prefix_token_count"]])
                feature_sources[i] = "stage3_reextracted_independent_original_prefixes_hidden_states[19]"
            torch.cuda.synchronize()
            if cache is not None:
                # Preserve an invalid cache before replacing it with verified prefix features.
                if cache.exists():
                    atomic_bytes(cache.with_suffix(f".preserved-{time.time_ns()}.npz"), cache.read_bytes())
                save_npz(cache, layers=np.asarray([19]),
                             positions=np.asarray([len(record["prompt_ids"]) + s["end_token_index"] for s in bounds["steps"]]),
                             hidden=vectors[np.newaxis].astype(np.float32),
                             token_sha256=np.asarray(record["token_sha256"]),
                             model_revision=np.asarray(manifest["revisions"]["model"]),
                             extraction=np.asarray("independent_original_prefixes"))
        if cache is not None:
            material[material_key(cache)] = digest(cache)
    feature_seconds = time.perf_counter() - feature_started
    if vectors.shape != (len(bounds["steps"]), len(probe["w"])) or not np.isfinite(vectors).all():
        raise ValueError("Selected-layer prefix features are invalid")
    logits = ((vectors - probe["mean"]) / probe["scale"]) @ probe["w"] + probe["b"]
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite B linear logits")
    saved, predictions_path = _saved_scores(folder, row["unique_id"]) if reused.any() else ({}, None)
    if predictions_path is not None:
        material[material_key(predictions_path)] = digest(predictions_path)
    for i, (step, logit) in enumerate(zip(bounds["steps"], logits)):
        z = float(logit)
        q = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        saved_q = saved.get((step["step_number"], step["prefix_token_count"]))
        use_saved = reused[i] and saved_q is not None and math.isfinite(saved_q) and abs(saved_q - q) <= 1e-12
        result["candidates"].append({**step, "q": saved_q if use_saved else q, "z": z,
                                     "q_source": "stage2_test_predictions.csv" if use_saved else feature_sources[i],
                                     "feature_source": feature_sources[i]})
    # Python's first max/min resolves exact logit ties to the earliest original step.
    for arm, choose in (("HIGH", max), ("LOW", min)):
        step = choose(result["candidates"], key=lambda s: s["z"])
        prefix = record["generated_ids"][:step["prefix_token_count"]]
        result["positions"][arm] = {**step, "prefix_ids": prefix,
                                    "prefix_sha256": stable_hash(record["prompt_ids"] + prefix)}
    result.update(status="ELIGIBLE", all_same_z=bool(np.all(logits == logits[0])),
                  all_equal_scores=bool(np.all(logits == logits[0])),
                  high_low_same_position=result["positions"]["HIGH"]["step_number"] == result["positions"]["LOW"]["step_number"])
    return _finish(result, started, feature_seconds)


def self_check():
    """One CPU fixture catches endpoint, saturation, tie and outcome-filter regressions."""
    import tempfile
    from veriserve_research.artifacts.io import atomic_json
    class CharacterTokenizer:
        all_special_ids = [0]
        def decode(self, ids, **kwargs):
            return "".join(chr(t) if t else "<|im_end|>" for t in ids)
    tokenizer = CharacterTokenizer()
    with tempfile.TemporaryDirectory(prefix="high-low-selection-") as tmp:
        folder = Path(tmp)
        row = {"unique_id": "test/0", "problem": "unused"}
        manifest = {"config_hash": "fixture", "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct", "do_sample": False,
            "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1},
            "revisions": {"model": "fixture"}, "splits": {
            "smoke": [{"unique_id": f"smoke/{i}"} for i in range(4)],
            "test": [row] + [{"unique_id": f"test/{i}"} for i in range(1, 200)],
            "train": [{"unique_id": "train/0"}]}}
        atomic_json(folder / "manifest.json", manifest)
        atomic_json(folder / "probe_B.json", {"method": "B", "layer": 19, "b": 0.})
        save_npz(folder / "probe_B.npz", mean=np.zeros(2), scale=np.ones(2), w=np.array([1., 0.]), b=np.array(0.))
        text = "Step 1: one\n\nStep 2: two\n\nStep 3: three\n\nFinal answer: \\boxed{3}"
        ids = list(map(ord, text)) + [0]
        _, bounds = boundaries(tokenizer, ids)
        record = {"prompt_ids": [1], "generated_ids": ids, "raw_text": tokenizer.decode(ids),
                  "completion": "EOS", "config_hash": "fixture", "label": None,
                  "boundaries": bounds, "feature_extraction": "independent_original_prefixes"}
        record["token_sha256"] = stable_hash({"prompt_ids": [1], "generated_ids": ids})
        path = record_path(folder, "test", row["unique_id"])
        feature_path = path.with_suffix(".npz")
        positions = [1 + s["end_token_index"] for s in bounds["steps"] + [bounds["end"]]]
        hidden = np.zeros((3, len(positions), 2))
        hidden[1, :3, 0] = [1000., 1001., -1000.]
        save_npz(feature_path, layers=np.array([18, 19, 20]), positions=np.array(positions), hidden=hidden)
        record["feature_files"] = [{"name": feature_path.name, "sha256": digest(feature_path)}]
        atomic_json(path, record)
        selected = freeze_reference(folder, "test", row, tokenizer)
        assert selected["status"] == "ELIGIBLE"  # label=None must never exclude a question.
        assert selected["positions"]["HIGH"]["step_number"] == 2
        assert selected["positions"]["LOW"]["step_number"] == 3
        assert selected["positions"]["LOW"]["near_end"]
        assert selected["candidates"][0]["q"] == selected["candidates"][1]["q"] == 1.
        for value in selected["positions"].values():
            assert value["prefix_ids"] == ids[:value["end_token_index"] + 1]
        replacement = dict(record, reference_origin="stage3_stream_unintervened_greedy")
        replacement["generated_ids"] = list(map(ord, text.replace("boxed{3}", "boxed{4}"))) + [0]
        replacement["raw_text"] = tokenizer.decode(replacement["generated_ids"])
        replacement["token_sha256"] = stable_hash({k: replacement[k] for k in ("prompt_ids", "generated_ids")})
        replacement["boundaries"] = boundaries(tokenizer, replacement["generated_ids"])[1]
        unchanged_prefixes = freeze_reference(folder, "test", row, tokenizer, reference=replacement)
        assert unchanged_prefixes["status"] == "ELIGIBLE" and unchanged_prefixes["reused_original_feature_prefixes"] == 3
        replacement["generated_ids"] = list(map(ord, text.replace("three", "other"))) + [0]
        replacement["raw_text"] = tokenizer.decode(replacement["generated_ids"])
        replacement["token_sha256"] = stable_hash({k: replacement[k] for k in ("prompt_ids", "generated_ids")})
        replacement["boundaries"] = boundaries(tokenizer, replacement["generated_ids"])[1]
        changed_prefix = freeze_reference(folder, "test", row, tokenizer, reference=replacement)
        assert changed_prefix["status"] == "NEEDS_FEATURES" and changed_prefix["reused_original_feature_prefixes"] == 2
        assert changed_prefix["missing_feature_step_numbers"] == [3]
        hidden[1, :, :] = 0.
        save_npz(feature_path, layers=np.array([18, 19, 20]), positions=np.array(positions), hidden=hidden)
        record["feature_files"][0]["sha256"] = digest(feature_path)
        atomic_json(path, record)
        tied = freeze_reference(folder, "test", row, tokenizer)
        assert tied["all_same_z"] and tied["high_low_same_position"]
        assert tied["positions"]["HIGH"]["step_number"] == 1
        save_npz(feature_path, layers=np.array([18, 19, 20]), positions=np.array(positions) + 1, hidden=hidden)
        record["feature_files"][0]["sha256"] = digest(feature_path)
        atomic_json(path, record)
        assert freeze_reference(folder, "test", row, tokenizer)["status"] == "NEEDS_FEATURES"
        cache = folder / "selected.npz"
        save_npz(cache, layers=np.asarray([19]), positions=np.asarray(positions[:3]), hidden=hidden[1:2, :3],
                     token_sha256=np.asarray(record["token_sha256"]), model_revision=np.asarray("fixture"),
                     extraction=np.asarray("independent_original_prefixes"))
        assert freeze_reference(folder, "test", row, tokenizer, feature_cache_path=cache)["status"] == "ELIGIBLE"
        save_npz(cache, layers=np.asarray([19]), positions=np.asarray(positions[:3]), hidden=hidden[1:2, :3],
                     token_sha256=np.asarray("different_reference"), model_revision=np.asarray("fixture"),
                     extraction=np.asarray("independent_original_prefixes"))
        assert freeze_reference(folder, "test", row, tokenizer, feature_cache_path=cache)["status"] == "NEEDS_FEATURES"
        record["completion"] = "LENGTH_TRUNCATED"
        atomic_json(path, record)
        assert freeze_reference(folder, "test", row, tokenizer)["status"] == "EXCLUDED"
        assert len(load_source(folder)[1]["test"]) == 200
    return {"status": "PASS", "layer_position_alignment": True, "saturated_q_ranked_by_z": True,
            "earliest_exact_tie": True, "near_end_retained": True, "missing_label_retained": True,
            "bad_feature_positions_require_extraction": True, "truncated_reference_excluded": True,
            "feature_cache_requires_matching_tokens_positions_revision": True,
            "only_changed_reference_prefixes_require_extraction": True}
