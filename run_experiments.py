"""Keep the original root command working after moving stage one."""
import os
import argparse
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    # ponytail: stage1 compatibility only; add stage selection when stage2 has a runner.
    stage_dir = Path(__file__).resolve().parent / "stages" / "stage1"
    os.chdir(stage_dir)
    sys.path.insert(0, str(stage_dir))
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["local_core"], required=True)
    parser.add_argument("--through", choices=["exp1", "exp2", "exp3", "exp4"], default="exp3")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.through == "exp4":
        from scripts.run_exp4 import run
        from scripts.audit_exp4 import audit
        result = run(args.config or Path("configs/exp4_4060.yaml"), args.resume)
        audit(result)
    else:
        runpy.run_path("run_experiments.py", run_name="__main__")
