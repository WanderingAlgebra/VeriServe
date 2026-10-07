"""Manual GPU validation: python tests/research_gpu_check.py (no Git commits/pushes)."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, digest, now, read_json, safe_error, stable_hash
from veriserve_research.artifacts.provenance import environment, source_hashes
from veriserve_research.inference import causal_check, load_resident
from veriserve_research.intervention import fixed_step
from veriserve_research.intervention.high_low import run as high_low, runtime, selection
from veriserve_research.probe import collect
from veriserve_research.probe.materials import record_path


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required; no precision/model substitution")
    sources = {name: source_hashes(name) for name in ("probe", "fixed-step", "high-low")}
    identity = stable_hash({"sources": sources, "validator": digest(__file__)})
    output = args.output or ROOT / ".cache/refactor-validation" / f"v2-{identity[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    probe_cfg = read_json(ROOT / "stages/stage2/config.json")
    fixed_cfg = read_json(ROOT / "stages/stage3/config.json")
    high_cfg = read_json(ROOT / "stages/stage3/high_low_config.json")
    material = ROOT / high_cfg["stage2_run"]
    original, splits = selection.load_source(material)
    fixed_material = read_json(ROOT / "stages/stage3/runs/20261004-109173095d08/manifest.json")
    hardware = environment(gpu=True)
    backend = {"gpu": torch.cuda.get_device_name(0), "uuid": str(torch.cuda.get_device_properties(0).uuid),
               "implementation": fixed_cfg["cache_backend"], "torch": torch.__version__}
    result = {"status": "RUNNING", "validation_only": True, "execution_sources": sources,
              "validator_sha256": digest(__file__), "created": now(), "pid": os.getpid(),
              "environment": hardware, "probe_smoke": {}, "fixed_step_pilot": {}, "high_low_pilot": {}}
    summary = output / "validation.json"

    def save():
        atomic_json(summary, result)

    save()
    print(f"GPU VALIDATION {output}", flush=True)
    model = tokenizer = verifier = None
    try:
        model, tokenizer, verifier = load_resident(high_cfg, output / "models")
        probe_run = output / "probe_smoke"
        for row in original["splits"]["smoke"]:
            record, _ = collect.collect_one(probe_cfg, probe_run, "smoke", row, model, tokenizer, hardware)
            previous = read_json(record_path(material, "smoke", row["unique_id"]))
            if (record["prompt_ids"] != previous["prompt_ids"] or record["generated_ids"] != previous["generated_ids"]):
                raise RuntimeError(f"Native greedy smoke tokens changed: {row['unique_id']}")
            check = causal_check(model, record, probe_cfg)
            if check["passed"] is False:
                raise RuntimeError(f"Causal extraction smoke failed: {row['unique_id']}")
            result["probe_smoke"][row["unique_id"]] = check
            save()
            print(f"GPU SMOKE {len(result['probe_smoke'])}/10", flush=True)
        before = collect.snapshot_hashes(probe_run, "smoke")
        for row in original["splits"]["smoke"]:
            _, changed = collect.collect_one(probe_cfg, probe_run, "smoke", row, None, None, hardware)
            assert not changed
        assert before == collect.snapshot_hashes(probe_run, "smoke")
        result["resume_hashes"] = {str(probe_run / path): sha for path, sha in before.items()}
        fixed_run = output / "fixed_step_pilot"
        probes = fixed_step.probe_files(fixed_cfg)
        for index, row in enumerate(fixed_material["splits"]["pilot"]):
            snapshot = fixed_step.initial_snapshot(fixed_cfg, fixed_run, "pilot", row, model, tokenizer, probes, backend)
            if snapshot["status"] == "ELIGIBLE":
                fixed_step.pilot_checks(fixed_cfg, fixed_run, snapshot, model, tokenizer, verifier)
                arms = ("NOW", "DELAY_2") if index % 2 == 0 else ("DELAY_2", "NOW")
                pair = [fixed_step.execute_arm(fixed_cfg, fixed_run, row, snapshot, arm, model, tokenizer, verifier, backend)
                        for arm in arms]
                assert all(arm["completed"] for arm in pair)
                result["fixed_step_pilot"][row["unique_id"]] = {"status": "PASS", "arms": len(pair)}
            else:
                result["fixed_step_pilot"][row["unique_id"]] = {"status": snapshot["status"], "reason": snapshot.get("reason")}
            save()
            print(f"GPU FIXED PILOT {index + 1}/20", flush=True)
        high_run = output / "high_low_pilot"
        for index, row in enumerate(splits["pilot"]):
            selected = high_low.freeze_selection(high_cfg, high_run, row, "pilot", tokenizer, model)
            while True:
                if selected["status"] != "ELIGIBLE":
                    raise RuntimeError(f"HIGH/LOW pilot is ineligible: {row['unique_id']}")
                try:
                    check = runtime.gpu_checks(high_cfg, high_run, row, selected, model, tokenizer, verifier)
                    arms = ("HIGH", "LOW") if index % 2 == 0 else ("LOW", "HIGH")
                    for arm in arms:
                        runtime.execute_path(high_cfg, high_run, row, selected, arm, model, tokenizer, verifier, backend)
                    assert high_low.complete_pair(high_run, row)
                    result["high_low_pilot"][row["unique_id"]] = {
                        "status": check["status"], "reference_prefix_match": check["reference_prefix_match"],
                        "recollected_streaming": bool(selected.get("recollected_streaming")), "arms": 2,
                    }
                    break
                except runtime.ReferenceMismatch as exc:
                    selected = high_low.recollect(high_cfg, high_run, row, "pilot", selected, model, tokenizer, exc)
            save()
            print(f"GPU HIGH/LOW PILOT {index + 1}/4", flush=True)
        assert sources == {name: source_hashes(name) for name in sources}
        result.update(status="PASS", finished=now())
        save()
        print("GPU VALIDATION PASS", flush=True)
    except BaseException as exc:
        result.update(status="FAILED", error=safe_error(exc), finished=now())
        save()
        raise
    finally:
        del model, tokenizer, verifier
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
