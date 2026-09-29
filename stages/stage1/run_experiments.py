from __future__ import annotations

import argparse
from pathlib import Path

from veriserve.runner import RunManager
from veriserve.report import finalize
from veriserve.common import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["local_core"], required=True)
    parser.add_argument("--through", choices=["exp1", "exp2", "exp3"], default="exp3")
    parser.add_argument("--config", type=Path, default=Path("configs/4060.yaml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = RunManager(args.config, resume=args.resume)
    print(f"Run: {run.root}", flush=True)
    try:
        run.run_experiment1()
        if args.through != "exp1":
            result = run.run_experiment2()
            if result["status"] != "BLOCKED" and args.through == "exp3":
                run.run_experiment3()
        (run.root / "run_error.json").unlink(missing_ok=True)
    except Exception as exc:
        atomic_json(run.root / "run_error.json", {"type": type(exc).__name__,
                                                  "message": str(exc)})
        raise
    finally:
        finalize(run)


if __name__ == "__main__":
    main()
