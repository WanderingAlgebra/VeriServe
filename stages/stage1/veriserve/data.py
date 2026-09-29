from __future__ import annotations

import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from .common import atomic_json, read_json


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def numeric_value(text: str) -> Decimal | None:
    value = text.strip().replace(",", "").replace("$", "")
    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", value):
        try:
            part = Fraction(value.replace(" ", ""))
            return Decimal(part.numerator) / Decimal(part.denominator)
        except (ZeroDivisionError, InvalidOperation):
            return None
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def gold_answer(text: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", text)
    if not matches:
        raise ValueError("GSM8K answer lacks #### delimiter")
    return matches[-1].strip()


def predicted_answer(text: str) -> str | None:
    matches = re.findall(r"Final answer:\s*([^\n]+)", text, re.IGNORECASE)
    if not matches:
        return None
    candidate = matches[-1].strip().rstrip(".; ")
    number = re.search(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:\s*/\s*\d+)?", candidate)
    return number.group(0) if number else None


def is_correct(output: str | None, gold: str) -> bool:
    if output is None:
        return False
    left = numeric_value(output)
    right = numeric_value(gold)
    return left is not None and right is not None and left == right


def _select_unique(rows: list[dict[str, Any]], count: int, rng: random.Random,
                   excluded: set[str]) -> list[dict[str, Any]]:
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    selected = []
    for index in indexes:
        row = rows[index]
        key = normalize_question(row.get("problem", row.get("question", "")))
        if not key or key in excluded:
            continue
        selected.append(row)
        excluded.add(key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} unique records available; need {count}")
    return selected


def prepare_splits(path: Path, seed: int, force: bool = False) -> dict[str, Any]:
    existing = read_json(path)
    if existing is not None and not force:
        return existing
    api = HfApi()
    revisions = {
        "gsm8k": api.dataset_info("openai/gsm8k").sha,
        "processbench": api.dataset_info("Qwen/ProcessBench").sha,
    }
    process = list(load_dataset("Qwen/ProcessBench", split="gsm8k", revision=revisions["processbench"]))
    rng = random.Random(seed)
    seen_process: set[str] = set()
    errors = _select_unique([r for r in process if r["label"] >= 0], 50, rng, seen_process)
    clean = _select_unique([r for r in process if r["label"] == -1], 10, rng, seen_process)
    diagnostic = errors[:25] + clean[:5] + errors[25:] + clean[5:]
    diagnostic_rows = []
    for index, row in enumerate(diagnostic):
        diagnostic_rows.append({
            "id": row["id"], "problem": row["problem"], "steps": row["steps"],
            "label": row["label"], "partition": "dev" if index < 30 else "holdout",
        })

    train = list(load_dataset("openai/gsm8k", "main", split="train", revision=revisions["gsm8k"]))
    test = list(load_dataset("openai/gsm8k", "main", split="test", revision=revisions["gsm8k"]))
    train_rows = [{"id": f"train-{i}", **row} for i, row in enumerate(train)]
    test_rows = [{"id": f"test-{i}", **row} for i, row in enumerate(test)]
    excluded = set(seen_process)
    debug = _select_unique(train_rows, 10, rng, excluded)
    development = _select_unique(train_rows, 40, rng, excluded)
    evaluation = _select_unique(test_rows, 100, rng, excluded)

    def pack(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"id": r["id"], "question": r["question"],
                 "gold": gold_answer(r["answer"])} for r in rows]

    result = {
        "seed": seed, "revisions": revisions,
        "debug": pack(debug), "development": pack(development),
        "evaluation": pack(evaluation), "diagnostic": diagnostic_rows,
    }
    atomic_json(path, result)
    return result
