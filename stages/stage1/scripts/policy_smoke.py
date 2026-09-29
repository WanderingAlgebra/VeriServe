"""One-question control-policy integration check, excluded from scientific results."""

from __future__ import annotations

import argparse
from pathlib import Path

from veriserve.common import atomic_json
from veriserve.data import is_correct
from veriserve.runner import RunManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/4060_1024.yaml"))
    args = parser.parse_args()
    run = RunManager(args.config, resume=True)
    official_root = run.root
    run.root = official_root / "engineering_policy_smoke_run"
    run.root.mkdir(exist_ok=True)
    row = run.splits["debug"][0]
    states = run.load_states("integration", ["unchecked", "endpoint"], [row])
    run.run_paths(states)
    result = {"label": "ENGINEERING_SMOKE_ONLY", "question_id": row["id"],
              "policies": {state.policy: {
                  "termination": state.termination,
                  "answer": state.final_answer,
                  "correct": is_correct(state.final_answer, state.gold),
                  "verifier_calls": state.verifier_calls,
                  "generated_tokens": state.generated_tokens,
              } for state in states.values()}}
    atomic_json(official_root / "engineering_policy_smoke.json", result)
    print(result)


if __name__ == "__main__":
    main()
