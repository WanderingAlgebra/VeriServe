"""Question-paired HIGH/LOW reporting; no model load or outcome-based selection."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from stages.stage1.veriserve.common import atomic_json
from stages.stage2.run_probe import atomic_bytes, write_csv
from stages.stage3.analyze import (backend, checks, distribution, effect, exclusion,
                                   finite, number, paired_effect)
from stages.stage3.storage import safe_read_json

ARMS = ("HIGH", "LOW")
PHASES = ("pilot", "test")
EXCLUDED = {"NO_ELIGIBLE_ANCHOR", "EXCLUDED", "INELIGIBLE"}
BUDGET_KEYS = ("generated_tokens", "revoked_tokens", "prm_calls", "rollbacks",
               "forward_tokens", "feedback_tokens")


def outcome(arm):
    """Only a terminal PASS may submit; unsubmitted algorithm failures are zero."""
    reason = exclusion(arm)
    if reason is not None:
        return None, reason
    submitted = arm.get("termination") == "FINAL_PASS" and arm.get("submitted_reasoning_ids") is not None
    return int(submitted and arm["Y"] == 1), None


def load_rows(run, manifest):
    selections, arms = {}, {}
    for folder, target in (("selections", selections), ("arms", arms)):
        for path in sorted((run / folder).rglob("*.json")):
            record = safe_read_json(path)
            if not record or not record.get("unique_id"):
                continue
            if folder == "arms" and record.get("arm") not in ARMS:
                continue
            key = (record.get("phase"), record["unique_id"])
            if folder == "arms":
                key += (record["arm"],)
            if key in target:
                raise ValueError(f"Duplicate authoritative {folder} record: {key}")
            target[key] = record
    rows = []
    for phase in PHASES:
        for item in manifest["splits"].get(phase, []):
            uid = item["unique_id"]
            selection = selections.get((phase, uid), {})
            pair = {name: arms.get((phase, uid, name)) for name in ARMS}
            values = {name: outcome(pair[name]) for name in ARMS}
            complete = selection.get("status") == "ELIGIBLE" and all(
                pair[name] and pair[name].get("status") == "COMPLETE" for name in ARMS)
            same_backend = complete and backend(pair["HIGH"]) != "UNKNOWN_BACKEND" and (
                backend(pair["HIGH"]) == backend(pair["LOW"]))
            times = {name: ((pair[name] or {}).get("timing") or {}).get("T_request") for name in ARMS}
            rows.append({"unique_id": uid, "phase": phase, "selection": selection,
                         "arms": pair, "Y": {name: values[name][0] for name in ARMS},
                         "exclusions": {name: values[name][1] for name in ARMS},
                         "eligible": selection.get("status") == "ELIGIBLE", "complete": bool(complete),
                         "quality_valid": bool(complete and all(values[name][1] is None for name in ARMS)),
                         "time_valid": bool(same_backend and all(finite(times[name]) and times[name] >= 0 for name in ARMS)),
                         "backend": backend(pair["HIGH"]) if same_backend else None})
    return rows


def paired(rows, variable, cfg):
    # Reuse the old question-bootstrap math only. These temporary helper keys are
    # never persisted: the new arms measure full requests, not post-snapshot time.
    adapted = []
    for row in rows:
        pair = {}
        for new, old in (("HIGH", "NOW"), ("LOW", "DELAY_2")):
            arm = row["arms"][new] or {}
            pair[old] = {"Y": row["Y"][new], "timing": {
                "T_postfork_wall": (arm.get("timing") or {}).get("T_request")}}
        adapted.append({**row, "arms": pair})
    return paired_effect(adapted, variable, {"seed": cfg["seed"], "bootstrap": 1000})


def arm_summary(rows, name):
    records = [r["arms"][name] for r in rows if r["eligible"] and r["arms"][name]]
    terminal = [arm for arm in records if arm.get("status") == "COMPLETE"]
    values = [r["Y"][name] for r in rows if r["quality_valid"]]
    times = [r["arms"][name]["timing"]["T_request"] for r in rows if r["time_valid"]]
    if len({r["backend"] for r in rows if r["time_valid"]}) > 1:
        times = []
    positions = [(r["selection"].get("positions") or {}).get(name, {}) for r in rows if r["eligible"]]
    first = [checks(arm)[0] for arm in terminal if checks(arm)]
    return {"recorded": len(records), "complete": len(terminal), "quality_n": len(values),
            "correct": sum(values), "accuracy": sum(values) / len(values) if values else None,
            "time_seconds": distribution(times),
            "budget": {key: distribution([(arm.get("budget") or {}).get(key) for arm in terminal]) for key in BUDGET_KEYS},
            "termination_counts": dict(Counter(arm.get("termination", "UNKNOWN") for arm in terminal)),
            "failure_counts": dict(Counter(arm.get("termination", "UNKNOWN") for arm in terminal
                                          if arm.get("termination") != "FINAL_PASS")),
            "wrong_submitted": sum(arm.get("termination") == "FINAL_PASS" and outcome(arm)[0] == 0 for arm in terminal),
            "grading_exclusion_counts": dict(Counter(outcome(arm)[1] or "NONE" for arm in terminal)),
            "status_counts": dict(Counter(arm.get("status", "UNKNOWN") for arm in records)),
            "first_check_verdicts": dict(Counter(c.get("effective_verdict", c.get("raw_verdict", "UNKNOWN")) for c in first)),
            "checkpoint_conflicts": sum(bool(c.get("checkpoint_conflict")) for arm in terminal for c in checks(arm)),
            "step_number": distribution([p.get("step_number") for p in positions]),
            "prefix_tokens": distribution([p.get("prefix_token_count") for p in positions]),
            "near_end": sum(bool(p.get("near_end")) for p in positions),
            "near_end_fraction": sum(bool(p.get("near_end")) for p in positions) / len(positions) if positions else None}


def summarize(rows, cfg, *, by_backend=True):
    quality = [r for r in rows if r["quality_valid"]]
    eligible = [r for r in rows if r["eligible"]]
    relation = Counter()
    for row in eligible:
        p = row["selection"]["positions"]
        high, low = p["HIGH"]["step_number"], p["LOW"]["step_number"]
        relation["HIGH_earlier" if high < low else "HIGH_later" if high > low else "same_position"] += 1
    result = {"planned": len(rows), "selection_recorded": sum(bool(r["selection"]) for r in rows),
              "eligible": len(eligible), "complete_pairs": sum(r["complete"] for r in rows),
              "quality_pairs": len(quality), "time_pairs": sum(r["time_valid"] for r in rows),
              "selection_exclusions": dict(Counter(r["selection"].get("reason", r["selection"].get("status"))
                                                     for r in rows if r["selection"].get("status") in EXCLUDED)),
              "pending_selections": dict(Counter(r["selection"].get("status", "MISSING") for r in rows
                                                   if not r["eligible"] and r["selection"].get("status") not in EXCLUDED)),
              "pending_pairs": sum(not r["complete"] for r in eligible),
              "infrastructure_pairs": sum(any(str((r["arms"][a] or {}).get("status", "")).startswith("INFRA") for a in ARMS) for r in eligible),
              "quality_pair_exclusions": dict(Counter(reason for r in rows if r["complete"]
                                                       for reason in set(r["exclusions"].values()) - {None})),
              "backend_or_time_invalid_pairs": sum(r["complete"] and not r["time_valid"] for r in rows),
              "delta_Y": paired(rows, "Y", cfg), "delta_T_seconds": paired(rows, "T", cfg),
              "HIGH_correct_LOW_wrong": sum(r["Y"]["HIGH"] == 1 and r["Y"]["LOW"] == 0 for r in quality),
              "HIGH_wrong_LOW_correct": sum(r["Y"]["HIGH"] == 0 and r["Y"]["LOW"] == 1 for r in quality),
              "position_relation": dict(relation),
              "all_equal_scores": sum(bool(r["selection"].get("all_equal_scores")) for r in eligible),
              "arms": {name: arm_summary(rows, name) for name in ARMS}}
    result["execution_status"] = "NOT_EXECUTED" if not any(result["arms"][a]["recorded"] for a in ARMS) else (
        "COMPLETE" if not result["pending_selections"] and not result["pending_pairs"] else "INCOMPLETE")
    if by_backend:
        result["by_backend"] = {b: summarize([r for r in rows if r["backend"] == b], cfg, by_backend=False)
                                for b in sorted({r["backend"] for r in rows if r["time_valid"]})}
    return result


def report(run, manifest, metrics):
    lines = ["# Stage 3：同题 HIGH/LOW 离线检查位置诊断", "",
             f"运行 `{run.name}`；experiment_type=`within_question_high_low`；配置哈希 `{manifest.get('config_hash', 'NA')}`。", "",
             "目的：同一道题，在未经干预完整参考轨迹的 B 最高分与最低分普通步骤检查，比较最终准确率与总请求处理成本。每条路径只有一次主动中途检查，之后仅终点检查与返工；终点 PASS 才允许提交。旧 fixed_step_now_vs_delay 的 UNCERTAIN 提交默认规则保留，本轮不沿用。", "",
             "ΔY = Y_HIGH − Y_LOW；ΔT = T_LOW − T_HIGH（秒），正值分别表示 HIGH 更准、更快。Y=终点允许提交且 Math-Verify 判对；格式、预算或未通过检查的算法失败为 0。gold 数据错误、评分错误单列，质量只用双方可评估配对。", "",
             "T_request 从各自 prompt prefill 至提交/终止实测，含生成、PRM、真实回滚、反馈增量 prefill、控制及相同的臂内原子持久化；CUDA 边界同步。加载/warmup、离线参考与特征、Git、gold 评分排除。该指标不是旧 snapshot 后 T_postfork_wall 的改名。时间仅比较同 GPU/实现有效配对，多后端不混合。按题配对 bootstrap 1000 次，固定种子 " + str(metrics["seed"]) + "，95% percentile CI。", "",
             "来源 manifest：`" + str(manifest.get("stage2_manifest", manifest.get("source_manifest", manifest.get("config", {}).get("stage2_run", "见 manifest.json")))) + "`。冻结位置与 token 前缀见 selections/；两路径原始结果见 arms/；保留源哈希。", "",
             f"准备时实际 HEAD：`{manifest.get('code_commit', 'NA')}`；用户核对过的 HEAD：`{manifest.get('checked_head', 'NA')}`。执行源码逐文件 SHA256 见 manifest.json；完整模型/tokenizer/dataset revision 见配置。", "",
             "| 环境字段 | 记录值 |", "| --- | --- |"]
    environment = manifest.get("environment") or {}
    lines.append(f"| Python | {environment.get('python', 'NA')} |")
    for name in ("numpy", "torch", "transformers", "bitsandbytes", "math-verify"):
        lines.append(f"| {name} | {(environment.get('versions') or {}).get(name) or 'NA'} |")
    lines += ["", "| 执行 GPU | UUID | 实现 / attention / dtype | torch / transformers / bitsandbytes |",
              "| --- | --- | --- | --- |"]
    for info in metrics["backends"]:
        implementation = " / ".join(str(info.get(k, "NA")) for k in ("implementation", "attention", "dtype"))
        versions = " / ".join(str(info.get(k, "NA")) for k in ("torch", "transformers", "bitsandbytes"))
        lines.append(f"| {info.get('gpu', 'NA')} | {info.get('uuid', 'NA')} | {implementation} | {versions} |")
    if not metrics["backends"]:
        lines.append("| 未执行真实路径 | NA | NA | NA |")
    lines.append("")
    for phase in PHASES:
        stat = metrics["phases"][phase]
        lines += [f"## {phase}", "", f"执行状态：**{stat['execution_status']}**。pilot 仅自检；正式 test 为原留出 200 题顺序上的探索性诊断。", "",
                  "| 计划 | 已保存位置 | 合格 | 完整配对 | 质量配对 | 时间配对 | 待完成配对 | 基础设施中断 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                  f"| {stat['planned']} | {stat['selection_recorded']} | {stat['eligible']} | {stat['complete_pairs']} | {stat['quality_pairs']} | {stat['time_pairs']} | {stat['pending_pairs']} | {stat['infrastructure_pairs']} |", "",
                  "| 路径 | 答对 / 共同可评估 | 正确率 | 平均总请求秒 | PRM / 回滚均值 | 生成 / 撤销 token 均值 | 生成 / 撤销 token 总数 |",
                  "| --- | ---: | ---: | ---: | --- | --- | --- |"]
        for name in ARMS:
            arm = stat["arms"][name]
            budget = arm["budget"]
            amounts = lambda keys, field: " / ".join(number(budget[k][field]) for k in keys)
            lines.append(f"| {name} | {arm['correct']} / {arm['quality_n']} | {number(arm['accuracy'])} | {number(arm['time_seconds']['mean'])} | {amounts(('prm_calls', 'rollbacks'), 'mean')} | {amounts(('generated_tokens', 'revoked_tokens'), 'mean')} | {amounts(('generated_tokens', 'revoked_tokens'), 'sum')} |")
        lines += ["", "| ΔY [95% CI] | ΔT 秒 [95% CI] | HIGH 对 / LOW 错 | HIGH 错 / LOW 对 |",
                  "| --- | --- | ---: | ---: |",
                  f"| {effect(stat['delta_Y'])} | {effect(stat['delta_T_seconds'])} | {stat['HIGH_correct_LOW_wrong']} | {stat['HIGH_wrong_LOW_correct']} |", "",
                  "| 路径 | 检查步号均值 | 前缀 token 均值 | near_end 数 / 比例 | 首检判决 | 算法失败终止次数 | 错答提交 | checkpoint_conflict |",
                  "| --- | ---: | ---: | --- | --- | --- | ---: | ---: |"]
        for name in ARMS:
            arm = stat["arms"][name]
            lines.append(f"| {name} | {number(arm['step_number']['mean'])} | {number(arm['prefix_tokens']['mean'])} | {arm['near_end']} / {number(arm['near_end_fraction'])} | `{json.dumps(arm['first_check_verdicts'], ensure_ascii=False, sort_keys=True)}` | `{json.dumps(arm['failure_counts'], ensure_ascii=False, sort_keys=True)}` | {arm['wrong_submitted']} | {arm['checkpoint_conflicts']} |")
        lines += ["", f"位置关系：`{json.dumps(stat['position_relation'], ensure_ascii=False, sort_keys=True)}`；全同分题 {stat['all_equal_scores']}（保留并各自实跑）。", "",
                  "| 记录类别 | 原因与计数 |", "| --- | --- |"]
        for key in ("selection_exclusions", "pending_selections", "quality_pair_exclusions"):
            lines.append(f"| {key} | `{json.dumps(stat[key], ensure_ascii=False, sort_keys=True)}` |")
        for name in ARMS:
            arm = stat["arms"][name]
            lines.append(f"| {name} grading exclusions | `{json.dumps(arm['grading_exclusion_counts'], sort_keys=True)}` |")
        lines += ["", f"后端/请求时间无效完整配对：{stat['backend_or_time_invalid_pairs']}。ΔY/ΔT 有效 bootstrap 次数 {stat['delta_Y']['bootstrap_valid']}/{stat['delta_T_seconds']['bootstrap_valid']}（各应为 1000，NA 为 0）。", ""]
        if len(stat["by_backend"]) > 1:
            lines += ["| GPU/实现 | 时间配对 | ΔT 秒 [95% CI] |", "| --- | ---: | --- |"]
            for name, group in stat["by_backend"].items():
                lines.append(f"| `{name}` | {group['time_pairs']} | {effect(group['delta_T_seconds'])} |")
            lines.append("")
    lines += ["## 离线开销", "", "只汇总已保存的真实值；特征复用不补造耗时。", "",
              "| 阶段 | 离线计时字段 | 数量 | 均值秒 | 合计秒 |", "| --- | --- | ---: | ---: | ---: |"]
    for phase, timings in metrics["offline_timing"].items():
        for key, value in timings.items():
            lines.append(f"| {phase} | {key} | {value['n']} | {number(value['mean'])} | {number(value['sum'])} |")
    lines += ["", "## 解释边界", "",
              "无真实 GPU pilot / 正式配对时，实验未执行，所有效果为 NA。无正确性差异导致 [0,0] 不能证明总体等效；失败早停较快不能单独视为收益。完整配对与未完成记录均保留，不能按正负结果扩缩样本。", "",
              "最高/最低位置选择需看完整参考轨迹，属于离线诊断。B 预测未经干预轨迹最终答错风险，不是局部错误或检查收益。分数和位置可能同时影响结果；本轮不能独立证明 B 超越位置、定位局部错误，或中途检查优于不检查/随机位置。pilot 与正式结果分开，强制 FAIL 自检不进入科学结果。"]
    atomic_bytes(run / "report.md", ("\n".join(lines) + "\n").encode())


def analyze(cfg, run, manifest=None):
    run = Path(run)
    manifest = manifest or safe_read_json(run / "manifest.json")
    if manifest.get("experiment_type") != "within_question_high_low":
        raise ValueError("HIGH/LOW analysis requires within_question_high_low manifest")
    rows = load_rows(run, manifest)
    metrics = {"run_id": run.name, "experiment_type": "within_question_high_low", "seed": cfg["seed"],
               "bootstrap": 1000, "phases": {phase: summarize([r for r in rows if r["phase"] == phase], cfg) for phase in PHASES},
               "offline_timing": {}, "backends": list({backend(r["arms"][name]): r["arms"][name]["backend"]
                                                        for r in rows for name in ARMS
                                                        if r["arms"][name] and r["arms"][name].get("backend")}.values())}
    flat = []
    for row in rows:
        selection = row["selection"]
        item = {"unique_id": row["unique_id"], "phase": row["phase"], "selection_status": selection.get("status"),
                "selection_reason": selection.get("reason"), "valid_quality_pair": row["quality_valid"],
                "valid_time_pair": row["time_valid"], "backend": row["backend"],
                "all_equal_scores": selection.get("all_equal_scores")}
        for name in ARMS:
            arm, position = row["arms"][name] or {}, (selection.get("positions") or {}).get(name, {})
            item.update({f"{name}_{key}": arm.get(key) for key in ("status", "termination")})
            item[f"{name}_Y"] = row["Y"][name]
            item[f"{name}_exclusion"] = row["exclusions"][name]
            item[f"{name}_T_request"] = (arm.get("timing") or {}).get("T_request")
            item.update({f"{name}_{key}": position.get(key) for key in ("step_number", "prefix_token_count", "q", "z", "near_end")})
            item.update({f"{name}_{key}": (arm.get("budget") or {}).get(key) for key in BUDGET_KEYS})
        item["delta_Y"] = row["Y"]["HIGH"] - row["Y"]["LOW"] if row["quality_valid"] else None
        item["delta_T_seconds"] = item["LOW_T_request"] - item["HIGH_T_request"] if row["time_valid"] else None
        flat.append(item)
    for phase in PHASES:
        timings = [{**(r["selection"].get("timings") or {}), **(r["selection"].get("timing") or {})}
                   for r in rows if r["phase"] == phase]
        keys = sorted({key for timing in timings for key, value in timing.items() if finite(value)})
        metrics["offline_timing"][phase] = {key: distribution([timing.get(key) for timing in timings]) for key in keys}
    atomic_json(run / "metrics.json", metrics)
    write_csv(run / "paired_results.csv", flat, list(flat[0]) if flat else ["unique_id", "phase", "delta_Y", "delta_T_seconds"])
    report(run, manifest, metrics)
    return metrics


def self_check():
    from tempfile import TemporaryDirectory
    cfg = {"seed": 20261004}
    with TemporaryDirectory() as directory:
        run = Path(directory)
        manifest = {"experiment_type": "within_question_high_low", "splits": {"test": [{"unique_id": "a"}, {"unique_id": "b"}, {"unique_id": "c"}, {"unique_id": "pending"}]}}
        for uid in ("a", "b", "c"):
            atomic_json(run / "selections" / f"{uid}.json", {"unique_id": uid, "phase": "test", "status": "ELIGIBLE",
                        "positions": {name: {"step_number": step, "prefix_token_count": 10 * step, "near_end": name == "LOW"}
                                      for name, step in zip(ARMS, (1, 2))}})
            for name in ARMS:
                fail = uid == "b" and name == "HIGH"
                arm = {"unique_id": uid, "phase": "test", "arm": name, "status": "COMPLETE", "Y": int(not fail),
                       "termination": "BUDGET_ROLLBACKS" if fail else "FINAL_PASS", "backend": {"gpu": "synthetic"},
                       "submitted_reasoning_ids": None if fail else [1], "grading": {},
                       "timing": {"T_request": 1 if name == "HIGH" else 3}, "budget": {"generated_tokens": 10}}
                if uid == "c":
                    arm.update(Y=None, grading={"exclusion": "GOLD_UNPARSEABLE"})
                atomic_json(run / "arms" / f"{uid}.{name}.json", arm)
        stat = analyze(cfg, run, manifest)["phases"]["test"]
        assert stat["quality_pairs"] == 2 and stat["arms"]["HIGH"]["accuracy"] == .5
        assert stat["delta_Y"]["estimate"] == -.5 and stat["delta_T_seconds"]["estimate"] == 2
        assert stat["delta_Y"]["bootstrap_valid"] == 1000
        assert stat["HIGH_wrong_LOW_correct"] == 1 and stat["quality_pair_exclusions"] == {"GOLD_DATA": 1}
        assert stat["execution_status"] == "INCOMPLETE" and stat["pending_selections"] == {"MISSING": 1}
        assert stat["position_relation"] == {"HIGH_earlier": 3}
        assert outcome({"status": "COMPLETE", "Y": 1, "termination": "FINAL_UNCERTAIN"}) == (0, None)
        empty = summarize([], cfg)
        assert empty["delta_Y"]["estimate"] is None and empty["execution_status"] == "NOT_EXECUTED"
        assert all((run / name).exists() for name in ("report.md", "metrics.json", "paired_results.csv"))
    return {"high_low_paired_analysis": "PASS", "request_time_and_delta_directions": "PASS",
            "algorithm_failures_in_denominator": "PASS", "gold_errors_separate": "PASS", "unexecuted_NA": "PASS"}


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False))
