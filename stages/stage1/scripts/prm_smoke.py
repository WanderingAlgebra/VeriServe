"""One-case PRM loading, scoring, and GPU headroom check."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from veriserve.common import atomic_json, read_json
from veriserve.prm import ProcessRewardModel


def main() -> None:
    assets = read_json(Path(".cache/assets.json"))
    if not assets or assets["verifier"]["id"] != "Qwen/Qwen2.5-Math-PRM-7B":
        raise RuntimeError("Download Qwen/Qwen2.5-Math-PRM-7B first")
    model = ProcessRewardModel(assets["verifier"]["path"])
    try:
        question = "What is 3 plus 4?"
        raw, input_tokens, output_tokens, seconds = model.verify(
            question, "", "Step 1: 3 + 4 = 7.")
        free, total = torch.cuda.mem_get_info()
        result = {"model_revision": assets["verifier"]["revision"],
                  "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "seconds": seconds, "free_bytes": free, "total_bytes": total,
                  "verdict": json.loads(raw)}
        atomic_json(Path(".cache/prm_smoke.json"), result)
        print(json.dumps(result, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
