"""Freeze a new ProcessBench holdout before scoring a replacement verifier."""
from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset

from veriserve.common import atomic_json, read_json
from veriserve.data import normalize_question


def main() -> None:
    source = read_json(Path(".cache/splits.json"))
    if source is None:
        raise RuntimeError("Create the original splits first")
    destination = Path(".cache/splits_prm.json")
    if destination.exists():
        raise RuntimeError("Fresh holdout already frozen; refusing to replace it")
    process = list(load_dataset("Qwen/ProcessBench", split="gsm8k",
                                revision=source["revisions"]["processbench"]))
    original_questions = {normalize_question(r["problem"]) for r in source["diagnostic"]}
    main_questions = {normalize_question(r["question"])
                      for split in ("debug", "development", "evaluation") for r in source[split]}
    excluded = original_questions | main_questions
    rng = random.Random(20260923)
    selected = []
    for label, count in (("error", 25), ("clean", 5)):
        pool = [r for r in process if (r["label"] >= 0 if label == "error" else r["label"] == -1)]
        rng.shuffle(pool)
        chosen = 0
        for row in pool:
            key = normalize_question(row["problem"])
            if key in excluded:
                continue
            excluded.add(key)
            selected.append({"id": row["id"], "problem": row["problem"],
                             "steps": row["steps"], "label": row["label"],
                             "partition": "holdout"})
            chosen += 1
            if chosen == count:
                break
        if chosen < count:
            raise RuntimeError(f"Not enough fresh {label} ProcessBench cases")
    updated = dict(source)
    updated["diagnostic"] = [dict(row) for row in source["diagnostic"] if row["partition"] == "dev"] + selected
    updated["diagnostic_holdout_seed"] = 20260923
    updated["diagnostic_holdout_source"] = "disjoint from all original diagnostics and main evaluation questions"
    atomic_json(destination, updated)
    print(f"Frozen {len(updated['diagnostic'])} diagnostic cases at {destination}")


if __name__ == "__main__":
    main()
