"""Functional entrypoints for the stage-two/three research workflows."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from . import ROOT


def self_check(experiment="all"):
    from .artifacts.io import read_json
    from .probe.collect import self_check as probe_check
    from .intervention.fixed_step import cpu_checks as fixed_check
    from .intervention.high_low.run import cpu_checks as high_check

    result = {}
    with tempfile.TemporaryDirectory(prefix="veriserve-cpu-checks-") as folder:
        folder = Path(folder)
        if experiment in ("all", "probe"):
            result["probe"] = probe_check(read_json(ROOT / "stages/stage2/config.json"), folder / "probe")
        if experiment in ("all", "fixed-step"):
            result["fixed-step"] = fixed_check(read_json(ROOT / "stages/stage3/config.json"), folder / "fixed")
        if experiment in ("all", "high-low"):
            result["high-low"] = high_check(read_json(ROOT / "stages/stage3/high_low_config.json"), folder / "high")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", add_help=False, help="Collect trajectories and fit risk probes")
    commands.add_parser("intervene", add_help=False, help="Run fixed-step or HIGH/LOW interventions")
    analysis = commands.add_parser("analyze", help="Read a saved run and write an independent analysis")
    analysis.add_argument("--run", required=True, type=Path)
    analysis.add_argument("--output", type=Path)
    checks = commands.add_parser("self-check", help="CPU-only checks in temporary directories")
    checks.add_argument("--experiment", choices=("all", "probe", "fixed-step", "high-low"), default="all")
    args, remaining = parser.parse_known_args(argv)
    if args.command == "probe":
        from .probe.collect import main as run
        return run(remaining)
    if args.command == "intervene":
        selector = argparse.ArgumentParser(description="Select the intervention protocol; remaining flags go to its runner")
        selector.add_argument("--experiment", choices=("fixed-step", "high-low"), required=True)
        selected, flags = selector.parse_known_args(remaining)
        if selected.experiment == "fixed-step":
            from .intervention.fixed_step import main as run
        else:
            from .intervention.high_low.run import main as run
        return run(flags)
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "analyze":
        from .analysis import analyze_run
        analyze_run(args.run, args.output)
    else:
        result = self_check(args.experiment)
        print(json.dumps({k: v.get("status", "PASS" if v.get("passed") else "FAILED")
                          for k, v in result.items()}))


if __name__ == "__main__":
    main()
