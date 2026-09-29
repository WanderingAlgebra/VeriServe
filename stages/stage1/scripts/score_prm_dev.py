"""Score only the frozen development diagnostic cases for PRM threshold selection."""
from __future__ import annotations

import json
from pathlib import Path

from veriserve.common import atomic_json, read_json, stable_hash
from veriserve.prm import ProcessRewardModel


def jobs_for_dev(rows: list[dict]) -> list[dict]:
    jobs = []
    for row in rows:
        if row["partition"] != "dev":
            continue
        label, steps = row["label"], row["steps"]
        if label == -1:
            valid = steps[:min(2, len(steps))]
            jobs.append({"id": row["id"] + "-valid", "kind": "valid",
                         "question": row["problem"], "accepted": "",
                         "new": "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(valid)),
                         "expected_step": None})
        else:
            accepted = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps[:label]))
            if label > 0:
                jobs.append({"id": row["id"] + "-valid", "kind": "valid",
                             "question": row["problem"], "accepted": "", "new": accepted,
                             "expected_step": None})
            jobs.append({"id": row["id"] + "-error", "kind": "error",
                         "question": row["problem"], "accepted": accepted,
                         "new": f"Step {label+1}: {steps[label]}",
                         "expected_step": label + 1})
    return jobs


def metrics(results: list[dict], threshold: float) -> dict:
    valid = [r for r in results if r["job"]["kind"] == "valid"]
    error = [r for r in results if r["job"]["kind"] == "error"]
    def first_low(r: dict) -> int | None:
        return next((item["step"] for item in r["scores"] if item["score"] < threshold), None)
    return {
        "threshold": threshold,
        "valid_count": len(valid), "error_count": len(error),
        "false_positive_rate": sum(first_low(r) is not None for r in valid) / len(valid),
        "error_detection_rate": sum(first_low(r) == r["job"]["expected_step"]
                                    for r in error) / len(error),
    }


def main() -> None:
    splits = read_json(Path(".cache/splits_prm.json"))
    assets = read_json(Path(".cache/assets.json"))
    if not splits or not assets or assets["verifier"]["id"] != "Qwen/Qwen2.5-Math-PRM-7B":
        raise RuntimeError("Freeze splits and download the PRM model first")
    output = Path(".cache/prm_dev_scores.json")
    jobs = jobs_for_dev(splits["diagnostic"])
    prior = read_json(output, {})
    if prior and prior["model_revision"] != assets["verifier"]["revision"]:
        raise RuntimeError("Existing development scores use a different model revision")
    results = {r["job"]["id"]: r for r in prior.get("results", [])}
    missing = [job for job in jobs if job["id"] not in results]
    if missing:
        model = ProcessRewardModel(assets["verifier"]["path"])
        try:
            for i, job in enumerate(missing, 1):
                ids = model.input_ids(job["question"], job["accepted"], job["new"])[0]
                if len(ids) > 1024:
                    raise RuntimeError(f"Development case {job['id']} exceeds 1024 tokens")
                raw, input_tokens, _, seconds = model.verify(job["question"], job["accepted"], job["new"])
                result = {"job": job, "scores": json.loads(raw)["step_scores"],
                          "input_tokens": input_tokens, "seconds": seconds}
                results[job["id"]] = result
                atomic_json(output, {"model_revision": assets["verifier"]["revision"],
                                     "results": [results[j["id"]] for j in jobs if j["id"] in results]})
                if i % 5 == 0:
                    print(f"PRM development: {i}/{len(missing)}", flush=True)
        finally:
            model.close()
    ordered = [results[job["id"]] for job in jobs]
    candidates = [metrics(ordered, round(i / 100, 2)) for i in range(5, 96, 5)]
    allowed = [row for row in candidates if row["false_positive_rate"] <= 0.20]
    selected = max(allowed, key=lambda row: (row["error_detection_rate"],
                                              -abs(row["threshold"] - 0.5))) if allowed else None
    report = {"model_revision": assets["verifier"]["revision"],
              "splits_hash": stable_hash(splits),
              "selected": selected, "candidates": candidates,
              "selection_rule": "maximize exact error detection with false positive rate <= 0.20; tie closest to 0.5"}
    atomic_json(Path(".cache/prm_dev_threshold.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
