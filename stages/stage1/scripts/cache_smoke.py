"""One-case cache smoke test; never counted as experiment three."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from veriserve.common import atomic_json, read_json
from veriserve.exp3 import _branch, _consistency_case
from veriserve.model import LanguageModel, feedback_text
from veriserve.runner import RunManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/4060_1024.yaml"))
    args = parser.parse_args()
    manager = RunManager(args.config, resume=True)
    if manager.root.resolve() != args.run.resolve():
        raise RuntimeError("Run hash does not match this source and config")
    candidate = None
    for path in sorted((manager.root / "states" / "exp1" / "k2").glob("*.json")):
        state = read_json(path)
        if state["fail_snapshots"]:
            candidate = (path.stem, state["fail_snapshots"][0])
            break
    if candidate is None:
        raise RuntimeError("No natural FAIL snapshot in experiment one")
    request_id, snapshot = candidate
    item = {"question_id": request_id, "snapshot": snapshot}
    b = _branch(manager, item, "B")
    c = _branch(manager, item, "C")
    model = LanguageModel(manager.assets["generator"]["path"])
    try:
        consistency = _consistency_case(model, snapshot, 32, 0.5)
        raw = snapshot["state"]
        prefix = raw["checkpoint_ids"]
        rejected = raw["context_ids"][len(prefix):]
        feedback = model.text_ids(feedback_text(snapshot["diagnosis"], snapshot["hint"],
                                                raw["checkpoint_step"]))
        cache_a, _, _ = model._prefill(prefix + rejected)
        cache_a.crop(-len(rejected))
        logits_a = model.model(input_ids=torch.tensor([feedback], device=model.device),
                               past_key_values=cache_a, use_cache=True).logits[:, -1, :]
        cache_b, logits_b, _ = model._prefill(prefix + feedback)
        divergence = None
        for index in range(32):
            token_a = int(torch.argmax(logits_a, dim=-1).item())
            token_b = int(torch.argmax(logits_b, dim=-1).item())
            if token_a != token_b:
                top_a = torch.topk(logits_a[0].float(), 2).values.tolist()
                top_b = torch.topk(logits_b[0].float(), 2).values.tolist()
                divergence = {"token_index": index, "a_token": token_a,
                              "b_token": token_b,
                              "a_top2": top_a, "b_top2": top_b,
                              "max_abs_logit_difference": float(
                                  (logits_a.float() - logits_b.float()).abs().max().item())}
                break
            logits_a = model._next_logits(token_a, cache_a)
            logits_b = model._next_logits(token_b, cache_b)
    finally:
        model.close()
    result = {
        "label": "ENGINEERING_SMOKE_ONLY",
        "source": "experiment_one_first_natural_fail",
        "request_id": request_id,
        "B_context_tokens": len(b.context_ids),
        "C_context_tokens": len(c.context_ids),
        "same_remaining_budget": b.budget == c.budget and
                                 b.generated_tokens == c.generated_tokens and
                                 b.verifier_calls == c.verifier_calls,
        "consistency": consistency,
        "first_greedy_divergence": divergence,
    }
    atomic_json(manager.root / "engineering_smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
