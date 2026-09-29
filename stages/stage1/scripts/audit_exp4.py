"""Audit completed experiment-four artifacts and write descriptive snapshot strata.

Run from stages/stage1: python -m scripts.audit_exp4 runs/<run-id>
This reads saved trajectories only; it never changes or reruns a model decision.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from veriserve.common import atomic_json, read_json, stable_hash


def audit(root: Path) -> dict:
    if read_json(root / "progress.json")["stage"] != "complete":
        raise ValueError("Wait for the experiment to finish before auditing")
    design = read_json(root / "config.json")
    settings = design["settings"]
    source = Path(design["parent_run"])
    for relative, expected in design["identity"]["inputs"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
    for relative, expected in design["identity"]["code"].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected, relative
    snapshots = {}
    for path in (root / "snapshots").rglob("*.json"):
        snapshot = read_json(path)
        assert stable_hash(snapshot["state"]) == snapshot["state_hash"], path
        snapshots[snapshot["id"]] = snapshot
    manifest = read_json(root / "branch_manifest.json")
    assert len(manifest) == len(snapshots) * len(settings["actions"]) * 3
    seen = set()
    paired_seeds = {}
    for item in manifest:
        key = (item["snapshot_id"], item["replicate"], item["action"])
        assert key not in seen
        seen.add(key)
        state = read_json(Path(item["path"]))
        history = snapshots[item["snapshot_id"]]["state"]
        assert state["budget"] == history["budget"]
        assert state["question"] == history["question"] and state["gold"] == history["gold"]
        assert state["events"][:len(history["events"])] == history["events"]
        assert state["checks"][:len(history["checks"])] == history["checks"]
        inherited = state["events"][len(history["events"])]
        assert inherited["event"] == "oracle_branch"
        for counter in ("generated_tokens", "prefill_tokens", "verifier_calls"):
            assert inherited[f"inherited_{counter}"] == history[counter]
        pair = (item["snapshot_id"], item["replicate"])
        assert paired_seeds.setdefault(pair, state["seed"]) == state["seed"]
    state_count = 0
    for path in (root / "states").rglob("*.json"):
        state = read_json(path)
        state_count += 1
        assert state["status"] in {"FINISHED", "SNAPSHOT_READY"}, path
        events = state["events"]
        assert state["generated_tokens"] == sum(
            e["kept_tokens"] + e["pending_tokens"] for e in events if e["event"] == "generate"), path
        assert state["verifier_calls"] == sum(e["event"] == "verify_call" for e in events), path
        logical_prefill = sum(e["input_tokens"] for e in events if e["event"] == "verify_call")
        logical_prefill += sum(e["prefill_tokens"] for e in events
                               if e["event"] == "generate" and e["prefill_kind"] != "OFFLINE_RESTORE")
        assert logical_prefill == state["prefill_tokens"], path
        for counter in ("generated_tokens", "prefill_tokens", "verifier_calls"):
            assert state[counter] <= state["budget"][counter], (path, counter)
        if len(state["context_ids"]) + len(state["pending_ids"]) > state["budget"]["context_tokens"]:
            assert state["termination"] == "CONTEXT_LIMIT", path
    rows = [json.loads(line) for line in (root / "exp4_oracle/branches.jsonl").read_text().splitlines()]
    held = [r for r in rows if r["replicate"] == settings["evaluation_seed"]]
    features = {}
    for key, snapshot in snapshots.items():
        state = snapshot["state"]
        features[key] = {
            "remaining_generation_tokens": state["budget"]["generated_tokens"] - state["generated_tokens"],
            "remaining_checks": state["budget"]["verifier_calls"] - state["verifier_calls"],
            "prefix_tokens": len(state["context_ids"]),
            "tokens_since_checkpoint": len(state["context_ids"]) - len(state["checkpoint_ids"]),
        }
    boundaries = {"remaining_generation_tokens": 1024, "remaining_checks": 3,
                  "prefix_tokens": 512, "tokens_since_checkpoint": 128}
    strata = []
    for group, feature, action in itertools.product(settings["snapshot_groups"], boundaries, settings["actions"]):
        threshold = boundaries[feature]
        for low in (True, False):
            selected = [r for r in held if r["group"] == group and r["action"] == action
                        and (features[r["snapshot_id"]][feature] <= threshold) == low]
            if selected:
                strata.append({"group": group, "feature": feature,
                               "stratum": f"{'<=' if low else '>'}{threshold}", "action": action,
                               "count": len(selected), "correct": sum(r["correct"] for r in selected),
                               "unfinished": sum(r["unfinished"] for r in selected),
                               "mean_suffix_seconds": sum(r["suffix_service_seconds"] for r in selected) / len(selected)})
    paired = []
    indexed = {(r["snapshot_id"], r["action"]): r for r in held}
    for group in settings["snapshot_groups"]:
        identifiers = sorted(k for k, s in snapshots.items() if s["group"] == group)
        if not identifiers:
            continue
        for a, b in itertools.combinations(settings["actions"], 2):
            differences = [int(indexed[s, a]["correct"]) - int(indexed[s, b]["correct"]) for s in identifiers]
            rng = random.Random(settings["seed"])
            boot = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
            paired.append({"group": group, "action_a": a, "action_b": b, "count": len(identifiers),
                           "a_only_correct": differences.count(1), "b_only_correct": differences.count(-1),
                           "accuracy_difference": sum(differences) / len(differences),
                           "bootstrap_95_interval": [boot[50], boot[1949]],
                           "mean_suffix_seconds_a_minus_b": sum(indexed[s, a]["suffix_service_seconds"]
                           - indexed[s, b]["suffix_service_seconds"] for s in identifiers) / len(identifiers)})
    output = root / "exp4"
    atomic_json(output / "snapshot_features.json", features)
    atomic_json(output / "held_seed_action_pairs.json", paired)
    frequency = []
    for policy in ("unchecked", "endpoint", design["fixed_policy"], "random"):
        folder = source / "states/exp2b_eval" if policy in {"unchecked", "endpoint"} else root / "states/exp4_eval"
        states = [read_json(p) for p in (folder / policy).glob("*.json")]
        calls = sum(s["verifier_calls"] for s in states)
        generated = sum(s["generated_tokens"] for s in states)
        frequency.append({"policy": policy, "count": len(states), "checks": calls,
                          "intermediate_checks": sum(c["candidate"] is None for s in states for c in s["checks"]),
                          "endpoint_checks": sum(c["candidate"] is not None for s in states for c in s["checks"]),
                          "generated_tokens": generated,
                          "checks_per_1000_generated_tokens": 1000 * calls / generated})
    atomic_json(output / "check_frequency.json", frequency)
    with (output / "exploratory_strata.csv").open("w", newline="") as handle:
        if strata:
            writer = csv.DictWriter(handle, fieldnames=list(strata[0]))
            writer.writeheader()
            writer.writerows(strata)
    result = {"status": "PASS", "states_checked": state_count, "snapshots_checked": len(snapshots),
              "paired_branches_checked": len(manifest), "parent_and_execution_code_hashes_match": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "strata_note": "Exploratory fixed bins; small groups are descriptive, not confirmatory evidence.",
              "audit_scope": "Saved history, paired seeds, immutable dependencies, counters and budget caps."}
    atomic_json(output / "artifact_audit.json", result)
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    for report_path, local, parent in ((root / "report.md", "exp4/", ""), (output / "report.md", "", "../")):
        report = report_path.read_text().split("## 产物与复现")[0]
        report += "## 产物与复现\n\n"
        report += "动作名称：`now`＝马上检查，`delay1`＝再写 1 步，`delay2`＝再写 2 步，`endpoint_only`＝等到最终答案。\n\n"
        report += "| 内容 | 文件 |\n| --- | --- |\n"
        for label, path in (
            ("冻结配置", parent + "config.json"), ("快照及缺失原因", parent + "snapshot_coverage.json"),
            ("完整策略数据", local + "full_paths.csv"), ("局部分支数据", local + "oracle_branches.csv"),
            ("独立种子的动作配对差异", local + "held_seed_action_pairs.json"),
            ("探索性预算与长度分层", local + "exploratory_strata.csv"),
            ("中途／终点检查次数及每千生成 token 检查率", local + "check_frequency.json"),
            ("质量与成本图", local + "quality_cost.png"), ("时机选择差距图", local + "oracle_gap.png"),
            ("数据与预算核对", local + "artifact_audit.json"),
            ("引用的旧轨迹", parent + "reused_trajectories.json"), ("执行波次墙钟时间", parent + "phase_times.jsonl"),
        ):
            report += f"| {label} | [{Path(path).name}]({path}) |\n"
        report += "\n分层表只作探索性描述，小样本分组不单独当作确定性结论。波次墙钟时间包含模型加载和卸载，不代表线上排队延迟。\n"
        report_path.write_text(report)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), ensure_ascii=False, indent=2))
