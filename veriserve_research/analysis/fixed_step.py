"""Fixed-step reports read saved data and write derived outputs."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ..artifacts.io import atomic_json, read_json, atomic_bytes, write_csv
from .paired import finite, group, exclusion, interval, paired_effect, distribution, checks, failure_category, number, effect, bootstrap_count, interpretation
from ..intervention.groups import load_rows, freeze_groups
from . import analysis_output
ARMS = ("NOW", "DELAY_2")
GROUPS = ("low", "mid", "high")
PHASES = ("pilot", "dev", "test")


def arm_summary(rows, name):
    records = [r["arms"][name] for r in rows if r["eligible"] and r["arms"][name]]
    terminal = [a for a in records if a.get("status") == "COMPLETE"]
    quality = [r["arms"][name]["Y"] for r in rows if r["quality_valid"]]
    time_rows = [r for r in rows if r["time_valid"]]
    # Different hardware is summarized below, never pooled into a request-time estimate.
    times = [r["arms"][name]["timing"]["T_postfork_wall"] for r in time_rows]
    if len({r["backend"] for r in time_rows}) > 1:
        times = []
    first = [checks(a)[0] for a in terminal if checks(a)]
    verdicts = Counter(c.get("effective_verdict", c.get("raw_verdict", "UNKNOWN")) for c in first)
    budget = {key: distribution([(a.get("budget") or {}).get(key) for a in terminal])
              for key in ("generated_tokens", "revoked_tokens", "prm_calls", "rollbacks", "forward_tokens", "feedback_tokens")}
    return {"recorded": len(records), "complete": len(terminal), "quality_n": len(quality),
            "correct": sum(quality), "accuracy": float(np.mean(quality)) if quality else None,
            "time_seconds": distribution(times), "budget": budget,
            "outcome_categories": dict(Counter(failure_category(a) for a in terminal)),
            "termination_counts": dict(Counter(a.get("termination", "UNKNOWN") for a in terminal)),
            "grading_exclusion_counts": dict(Counter((a.get("grading") or {}).get("exclusion") or "NONE" for a in terminal)),
            "status_counts": dict(Counter(a.get("status", "UNKNOWN") for a in records)),
            "first_check": {"n": len(first), "effective_verdict_counts": dict(verdicts),
                            "FAIL_fraction": verdicts["FAIL"] / len(first) if first else None,
                            "UNCERTAIN_fraction": verdicts["UNCERTAIN"] / len(first) if first else None},
            "first_check_extra_steps": distribution([c.get("extra_steps") for c in first]),
            "first_check_input_tokens": distribution([c.get("input_tokens") for c in first]),
            "early_endpoint": sum(bool(c.get("early_endpoint")) for c in first)}


def summary(rows, cfg, *, by_backend=True):
    quality = [r for r in rows if r["quality_valid"]]
    result = {"questions": len(rows), "eligible": sum(r["eligible"] for r in rows),
              "complete_pairs": sum(r["complete"] for r in rows), "quality_pairs": len(quality),
              "time_pairs": sum(r["time_valid"] for r in rows),
              "delta_Y": paired_effect(rows, "Y", cfg), "delta_T_seconds": paired_effect(rows, "T", cfg),
              "wrong_to_correct": sum(r["arms"]["NOW"]["Y"] == 1 and r["arms"]["DELAY_2"]["Y"] == 0 for r in quality),
              "correct_to_wrong": sum(r["arms"]["NOW"]["Y"] == 0 and r["arms"]["DELAY_2"]["Y"] == 1 for r in quality),
              "prefix_tokens": distribution([r["snapshot"].get("prefix_length") for r in rows if r["eligible"]]),
              "near_end": sum(bool(r["snapshot"].get("near_end")) for r in rows if r["eligible"]),
              "delay_collapsed": sum(bool((r["arms"]["DELAY_2"] or {}).get("delay_collapsed") or r["snapshot"].get("delay_collapsed")) for r in rows if r["eligible"]),
              "arms": {a: arm_summary(rows, a) for a in ARMS}}
    if by_backend:
        result["by_backend"] = {b: summary([r for r in rows if r["backend"] == b], cfg, by_backend=False)
                                for b in sorted({r["backend"] for r in rows if r["time_valid"]})}
    return result


def heterogeneity(rows, method, variable, cfg):
    valid = [r for r in rows if r["quality_valid" if variable == "Y" else "time_valid"]]
    n = cfg.get("bootstrap", 1000)
    low = [r for r in valid if r.get(f"group_{method}") == "low"]
    high = [r for r in valid if r.get(f"group_{method}") == "high"]
    if not low or not high:
        return interval(None, [], n, "EMPTY_LOW_OR_HIGH_GROUP")
    if variable == "T" and len({r["backend"] for r in valid}) > 1:
        return interval(None, [], n, "MIXED_BACKENDS_SEE_BY_BACKEND")
    def delta(row):
        return row["arms"]["NOW"]["Y"] - row["arms"]["DELAY_2"]["Y"] if variable == "Y" else (
            row["arms"]["DELAY_2"]["timing"]["T_postfork_wall"] - row["arms"]["NOW"]["timing"]["T_postfork_wall"])
    point = float(np.mean([delta(r) for r in high]) - np.mean([delta(r) for r in low]))
    rng = np.random.default_rng(cfg["seed"])
    samples = []
    for _ in range(n):
        draw = [valid[i] for i in rng.integers(0, len(valid), len(valid))]
        lo = [delta(r) for r in draw if r.get(f"group_{method}") == "low"]
        hi = [delta(r) for r in draw if r.get(f"group_{method}") == "high"]
        if lo and hi:
            samples.append(float(np.mean(hi) - np.mean(lo)))
    result = interval(point, samples, n, "NO_VALID_BOOTSTRAP_REPLICATES")
    result.update(low_questions=len(low), high_questions=len(high))
    return result


def phase_metrics(rows, cfg, boundaries):
    for row in rows:
        for method in ("B", "C"):
            row[f"group_{method}"] = group(row["snapshot"].get(f"q_{method}"), boundaries.get(method)) if row["eligible"] else None
    counts, pending = Counter(), Counter()
    for row in rows:
        if not row["snapshot"]:
            pending["MISSING_SNAPSHOT"] += 1
        elif row["snapshot"].get("status") == "NO_ELIGIBLE_ANCHOR":
            counts[row["snapshot"].get("reason") or row["snapshot"].get("status", "UNKNOWN_SNAPSHOT")] += 1
        elif not row["eligible"]:
            pending[row["snapshot"].get("status", "UNKNOWN_SNAPSHOT")] += 1
    pair_exclusions, pending_pairs, infrastructure_pairs = Counter(), 0, 0
    for row in rows:
        if row["eligible"] and not row["quality_valid"]:
            reasons = set(row["exclusions"].values()) - {None}
            pending_pairs += "PENDING" in reasons
            infrastructure_pairs += "INFRASTRUCTURE" in reasons
            for reason in reasons - {"PENDING", "INFRASTRUCTURE"}:
                pair_exclusions[reason] += 1
    result = {"planned": len(rows), "snapshot_recorded": sum(bool(r["snapshot"]) for r in rows),
              "eligible": sum(r["eligible"] for r in rows),
              "eligible_coverage": sum(r["eligible"] for r in rows) / len(rows) if rows else None,
              "anchor_exclusions": dict(counts), "pending_anchors": dict(pending),
              "pending_pairs": pending_pairs, "infrastructure_interrupted_pairs": infrastructure_pairs,
              "quality_pair_exclusions": dict(pair_exclusions),
              "different_backend_pairs": sum(r["complete"] and r["backend"] is None for r in rows),
              "overall": summary(rows, cfg), "risk": {}}
    for method in ("B", "C"):
        result["risk"][method] = {
            "boundaries": boundaries.get(method),
            "groups": {g: summary([r for r in rows if r[f"group_{method}"] == g], cfg) for g in GROUPS},
            "high_minus_low_delta_Y": heterogeneity(rows, method, "Y", cfg),
            "high_minus_low_delta_T_seconds": heterogeneity(rows, method, "T", cfg),
            "heterogeneity_by_backend": {b: {
                "high_minus_low_delta_Y": heterogeneity([r for r in rows if r["backend"] == b], method, "Y", cfg),
                "high_minus_low_delta_T_seconds": heterogeneity([r for r in rows if r["backend"] == b], method, "T", cfg)}
                for b in sorted({r["backend"] for r in rows if r["time_valid"]})}}
    return result


def report(run, manifest, metrics):
    lines = ["# 第三阶段：探针风险与即时检查收益", "", f"运行：`{metrics['run_id']}`。配置哈希：`{manifest.get('config_hash', 'NA')}`。", "",
             "主时间指标为“从公共快照到结束的后续处理时间” T_postfork_wall，不是完整请求端到端时间。ΔY = Y_NOW − Y_DELAY_2；ΔT = T_DELAY_2 − T_NOW（秒），正值分别表示立即检查更准确、更快。",
             "合格锚点上的算法失败计 Y=0；gold 数据错误、评分超时/错误及基础设施中断分别计数。质量采用两臂均可评估的配对分母；时间按相同 GPU 与运行实现分开。bootstrap 按题配对重抽 1000 次，95% percentile CI。", "",
             f"风险分组状态：{metrics['dev_groups_status']}。分界仅来自 dev 合格公共快照的 q，不用动作结果，也不在 test 重划。", ""]
    for phase in PHASES:
        data = metrics["phases"][phase]
        total = data["overall"]
        lines += [f"## {phase}", "", f"计划 {data['planned']} 题；已保存锚点状态 {data['snapshot_recorded']} 题；合格 {data['eligible']} 题（覆盖率 {number(data['eligible_coverage'])}）；完整配对 {total['complete_pairs']}，有效质量配对 {total['quality_pairs']}，有效时间配对 {total['time_pairs']}。",
                  f"锚点不合格原因：`{json.dumps(data['anchor_exclusions'], ensure_ascii=False, sort_keys=True)}`；未执行/待恢复锚点：`{json.dumps(data['pending_anchors'], ensure_ascii=False, sort_keys=True)}`；待完成配对 {data['pending_pairs']}；基础设施中断配对 {data['infrastructure_interrupted_pairs']}；质量评估排除：`{json.dumps(data['quality_pair_exclusions'], ensure_ascii=False, sort_keys=True)}`。", "",
                  "| 范围 | 合格题数 | 质量配对 | NOW 正确率 | DELAY_2 正确率 | ΔY [95% CI] | ΔT 秒 [95% CI] | 错→对 | 对→错 | 前缀 token 均值 |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |"]
        table = [("总体", total)] + [(f"{method} {g}", data["risk"][method]["groups"][g]) for method in ("B", "C") for g in GROUPS]
        for label, stat in table:
            lines.append(f"| {label} | {stat['eligible']} | {stat['quality_pairs']} | {number(stat['arms']['NOW']['accuracy'])} | {number(stat['arms']['DELAY_2']['accuracy'])} | {effect(stat['delta_Y'])} | {effect(stat['delta_T_seconds'])} | {stat['wrong_to_correct']} | {stat['correct_to_wrong']} | {number(stat['prefix_tokens']['mean'])} |")
        lines += ["", f"总体质量：{interpretation(total['delta_Y'])} 总体时间：{interpretation(total['delta_T_seconds'])}",
                  f"总体 ΔY/ΔT 有效 bootstrap 次数：{bootstrap_count(total['delta_Y'], metrics['bootstrap'])}/{bootstrap_count(total['delta_T_seconds'], metrics['bootstrap'])}。", "",
                  "| 范围 | NOW 时间均值/中位秒 | DELAY_2 时间均值/中位秒 | 检查/回滚均值 | 生成/撤销 token 均值 | 首检 FAIL/UNCERTAIN 比例 | 结果分类次数 | ΔY;ΔT 有效 bootstrap |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for label, stat in table:
            arms = stat["arms"]
            times = ["/".join(number(arms[a]["time_seconds"][v]) for v in ("mean", "median")) for a in ARMS]
            fractions = "; ".join(f"{a}:{number(arms[a]['first_check']['FAIL_fraction'])}/{number(arms[a]['first_check']['UNCERTAIN_fraction'])}" for a in ARMS)
            failures = "; ".join(f"{a}:`{json.dumps(arms[a]['outcome_categories'], ensure_ascii=False, sort_keys=True)}`" for a in ARMS)
            lines.append(f"| {label} | {times[0]} | {times[1]} | {mechanism_text(arms, 'budget', ('prm_calls', 'rollbacks'))} | {mechanism_text(arms, 'budget', ('generated_tokens', 'revoked_tokens'))} | {fractions} | {failures} | {bootstrap_count(stat['delta_Y'], metrics['bootstrap'])};{bootstrap_count(stat['delta_T_seconds'], metrics['bootstrap'])} |")
        lines += ["", "结果分类：CORRECT=判对；WRONG_SUBMITTED=已提交答错；FORMAT_OR_PREDICTION_PARSE=格式/预测解析失败；BUDGET_OR_INPUT_LIMIT=预算/输入上限；OTHER_ALGORITHM_FAILURE=其他算法终止；EVALUATION_UNAVAILABLE=评估错误。精确终止原因、评分错误和基础设施状态另存 metrics.json。", ""]
        for method in ("B", "C"):
            risk = data["risk"][method]
            lines += [f"{method} 高风险减低风险的收益差：ΔY {effect(risk['high_minus_low_delta_Y'])}；ΔT 秒 {effect(risk['high_minus_low_delta_T_seconds'])}。",
                      f"高低差 ΔY/ΔT 有效 bootstrap 次数：{bootstrap_count(risk['high_minus_low_delta_Y'], metrics['bootstrap'])}/{bootstrap_count(risk['high_minus_low_delta_T_seconds'], metrics['bootstrap'])}。",
                      f"质量异质性：{interpretation(risk['high_minus_low_delta_Y'])} 时间异质性：{interpretation(risk['high_minus_low_delta_T_seconds'])}", ""]
        for name in ARMS:
            arm = total["arms"][name]
            t = arm["time_seconds"]
            budget = arm["budget"]
            lines += [f"{name} 后续时间均值/中位数/p95（描述性）：{number(t['mean'])}/{number(t['median'])}/{number(t['p95_descriptive'])} 秒；PRM 调用/回滚均值 {number(budget['prm_calls']['mean'])}/{number(budget['rollbacks']['mean'])}；生成/撤销 token 合计 {number(budget['generated_tokens']['sum'])}/{number(budget['revoked_tokens']['sum'])}。",
                      f"首次检查判决次数：`{json.dumps(arm['first_check']['effective_verdict_counts'], ensure_ascii=False, sort_keys=True)}`；FAIL/UNCERTAIN 比例：{number(arm['first_check']['FAIL_fraction'])}/{number(arm['first_check']['UNCERTAIN_fraction'])}（有首次检查记录 {arm['first_check']['n']} 臂）；提前终点 {arm['early_endpoint']} 臂；累计非 decode 前向/反馈 token 均值 {number(budget['forward_tokens']['mean'])}/{number(budget['feedback_tokens']['mean'])}。",
                      f"终止原因：`{json.dumps(arm['termination_counts'], ensure_ascii=False, sort_keys=True)}`；评分排除：`{json.dumps(arm['grading_exclusion_counts'], ensure_ascii=False, sort_keys=True)}`；保存状态：`{json.dumps(arm['status_counts'], ensure_ascii=False, sort_keys=True)}`。", ""]
        lines += [f"公共快照 near_end {total['near_end']} 题；DELAY_2 delay_collapsed {total['delay_collapsed']} 题；两臂后端不一致 {data['different_backend_pairs']} 题。各组检查/回滚、失败分类、时间分布、有效 bootstrap 次数及分后端结果详见 metrics.json。", ""]
        if len(total["by_backend"]) > 1:
            lines += ["存在多种 GPU/运行后端，总体时间标为 NA；分别如下：", ""]
            for hardware, stat in total["by_backend"].items():
                lines.append(f"- `{hardware}`：{stat['time_pairs']} 对；ΔT 秒 {effect(stat['delta_T_seconds'])}。")
            lines.append("")
    test = metrics["phases"]["test"]
    b = test["risk"]["B"]
    c = test["risk"]["C"]
    test_complete = not test["pending_anchors"] and not test["pending_pairs"] and not test["infrastructure_interrupted_pairs"]
    support = test_complete and any(v.get("ci95") and v["ci95"][0] > 0 and min(v.get("low_questions", 0), v.get("high_questions", 0)) > 1
                                   for v in (b["high_minus_low_delta_Y"], b["high_minus_low_delta_T_seconds"]))
    direction = []
    for key, label in (("high_minus_low_delta_Y", "质量"), ("high_minus_low_delta_T_seconds", "时间")):
        bv, cv = b[key].get("estimate"), c[key].get("estimate")
        relation = "NA" if not finite(bv) or not finite(cv) else "同向" if np.sign(bv) == np.sign(cv) else "不同向"
        direction.append(f"{label}高低收益差点估计 {relation}")
    lengths = "; ".join(f"{m} 低/高组前缀均值 {number(test['risk'][m]['groups']['low']['prefix_tokens']['mean'])}/{number(test['risk'][m]['groups']['high']['prefix_tokens']['mean'])} token" for m in ("B", "C"))
    total = test["overall"]
    arms = total["arms"]
    mechanism = f"正式配对中立即检查修复 {total['wrong_to_correct']} 题、破坏 {total['correct_to_wrong']} 题；"
    mechanism += "; ".join(f"{a} 检查/回滚合计 {number(arms[a]['budget']['prm_calls']['sum'])}/{number(arms[a]['budget']['rollbacks']['sum'])}、撤销 token {number(arms[a]['budget']['revoked_tokens']['sum'])}" for a in ARMS) + "。"
    lines += ["## 解释与后续", "",
              "B 预测未经干预的整条轨迹最终答错风险，不是当前步错误概率或检查收益概率。B 高低组配对收益差才检验“高风险是否更值得立即检查”；首次检查失败率本身不能回答该问题。C 是仅有步骤/长度的对照，固定第二步仍留下前缀长度差异；若 B 与 C 的收益模式相似，长度可能解释部分效果。B 在 stage2 的 AUROC 较高不构成本轮检查时机收益证据。",
              f"正式 B 异质性质量/时间：{effect(b['high_minus_low_delta_Y'])} / {effect(b['high_minus_low_delta_T_seconds'])}；C：{effect(c['high_minus_low_delta_Y'])} / {effect(c['high_minus_low_delta_T_seconds'])}。",
              f"实际长度与对照：{lengths}；{'；'.join(direction)}。同向时长度信号与 B 收益模式相容，但这不证明长度完全解释 B；不同向也不能证明 B 有独立作用，本轮没有做长度调整。",
              mechanism,
              "修复/破坏次数、首次检查 PASS/FAIL/UNCERTAIN、返工次数和撤销 token 可描述收益与成本的机制，但没有额外随机化，不能把它们直接解释为因果中介。格式失败和预算耗尽保持在算法失败分母，所有排除均保留记录。",
              "反馈是 PRM 分数驱动的确定性通用 diagnosis/hint 模板；本轮只覆盖无历史 PASS 的普通 Step 2，不能直接推广到多位置或已有 PASS 状态。q 的独立前缀特征提取与探针时间属于离线诊断开销。小组 p95 仅为描述，分组样本过少、CI 跨零和 BF16 数值/运行后端差异都限制结论。",
              ("至少一项正式 B 高低组收益差的 CI 支持正向异质性，可为下一阶段阈值策略提供初步依据；仍须联合评估质量、时间与 C/长度解释，当前不设 λ、不从 test 搜索阈值。" if support else "当前没有足够正式证据支持推进风险阈值策略；结果为 NA 或 CI 跨零时如实保留，不调整 test 寻找正结果。"), "",
              "pilot/dev/test 分开报告；未执行指标均为 NA。离线诊断和公共生成开销见下表（只对实际保存值汇总）：", "",
              "| 阶段 | 公共生成秒均值 | 离线特征秒均值 | 探针秒均值 | 实验快照恢复秒均值 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for phase, timings in metrics["offline_timing"].items():
        lines.append(f"| {phase} | {number(timings['T_common_generation']['mean'])} | {number(timings['T_offline_feature']['mean'])} | {number(timings['T_probe']['mean'])} | {number(timings['T_snapshot_restore']['mean'])} |")
    atomic_bytes(run / "report.md", ("\n".join(lines) + "\n").encode())


def plot(metrics, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = metrics["phases"]["test"]["risk"]
    for ax, key, ylabel in zip(axes, ("delta_Y", "delta_T_seconds"), ("NOW minus DELAY accuracy", "DELAY minus NOW time (s)")):
        points = 0
        for method, offset, color in (("B", -.08, "tab:blue"), ("C", .08, "tab:orange")):
            labeled = False
            for i, g in enumerate(GROUPS):
                metric = data[method]["groups"][g][key]
                if metric["estimate"] is not None and metric["ci95"]:
                    lo, hi = metric["ci95"]
                    ax.vlines(i + offset, lo, hi, color=color)
                    ax.plot(i + offset, metric["estimate"], "o", color=color, label=method if not labeled else None)
                    labeled = True
                    points += 1
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set(xticks=range(3), xticklabels=GROUPS, xlabel="Frozen dev risk group", ylabel=ylabel)
        if points:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
        else:
            ax.text(.5, .5, "NA: no valid test pairs", ha="center", transform=ax.transAxes)
    fig.suptitle("Step 2 paired effects: 95% question bootstrap CI")
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    atomic_bytes(run / "risk_effects.png", buffer.getvalue())


def analyze(cfg, run, manifest=None, *, freeze_dev=False, output_dir=None):
    run = Path(run)
    manifest = manifest or read_json(run / "manifest.json")
    output = analysis_output(run, output_dir)
    if freeze_dev:
        freeze_groups(cfg, run, manifest)
    frozen = read_json(run / "dev_groups.json") or {}
    boundaries = frozen.get("boundaries", {}) if frozen.get("status") == "FROZEN" else {}
    rows = load_rows(run, manifest)
    metrics = {"run_id": run.name, "bootstrap": cfg.get("bootstrap", 1000), "seed": cfg["seed"],
               "dev_groups_status": frozen.get("status", "NOT_FROZEN"),
               "dev_groups_reason": frozen.get("reason"),
               "phases": {p: phase_metrics([r for r in rows if r["phase"] == p], cfg, boundaries) for p in PHASES},
               "offline_timing": {}}
    flat = []
    for r in rows:
        s = r["snapshot"]
        item = {"unique_id": r["unique_id"], "phase": r["phase"], "snapshot_status": s.get("status"),
                "anchor_reason": s.get("reason"), "q_B": s.get("q_B"), "q_C": s.get("q_C"),
                "prefix_length": s.get("prefix_length"), "near_end": s.get("near_end"),
                "group_B": r["group_B"], "group_C": r["group_C"], "valid_quality_pair": r["quality_valid"],
                "valid_time_pair": r["time_valid"], "backend": r["backend"]}
        for name in ARMS:
            a = r["arms"][name] or {}
            item.update({f"{name}_{key}": a.get(key) for key in ("status", "Y", "termination")})
            item[f"{name}_exclusion"] = r["exclusions"][name]
            item[f"{name}_T_postfork_wall"] = (a.get("timing") or {}).get("T_postfork_wall")
            for key in ("generated_tokens", "revoked_tokens", "prm_calls", "rollbacks", "forward_tokens", "feedback_tokens"):
                item[f"{name}_{key}"] = (a.get("budget") or {}).get(key)
        item["delta_Y"] = item["NOW_Y"] - item["DELAY_2_Y"] if r["quality_valid"] else None
        item["delta_T_seconds"] = item["DELAY_2_T_postfork_wall"] - item["NOW_T_postfork_wall"] if r["time_valid"] else None
        flat.append(item)
    for phase in PHASES:
        phase_rows = [r for r in rows if r["phase"] == phase]
        metrics["offline_timing"][phase] = {key: distribution([(r["snapshot"].get("timing") or {}).get(key) for r in phase_rows])
                                            for key in ("T_common_generation", "T_offline_feature", "T_probe")}
        metrics["offline_timing"][phase]["T_snapshot_restore"] = distribution([
            ((r["arms"][a] or {}).get("timing") or {}).get("T_snapshot_restore") for r in phase_rows for a in ARMS])
    metrics["analysis_output"] = str(output)
    atomic_json(output / "metrics.json", metrics)
    if flat:
        write_csv(output / "paired_results.csv", flat, list(flat[0]))
    else:
        write_csv(output / "paired_results.csv", [], ["unique_id", "phase", "delta_Y", "delta_T_seconds"])
    report(output, manifest, metrics)
    plot(metrics, output)
    return metrics


def self_check():
    cfg = {"seed": 20261004, "bootstrap": 1000}
    def row(uid, now, delay, g, failure=False):
        return {"unique_id": uid, "eligible": True, "complete": True, "quality_valid": True,
                "time_valid": True, "backend": "gpu", "group_B": g, "group_C": g,
                "snapshot": {"prefix_length": 10}, "arms": {
                    "NOW": {"status": "COMPLETE", "Y": now, "termination": "BUDGET" if failure else "EOS", "timing": {"T_postfork_wall": 1}},
                    "DELAY_2": {"status": "COMPLETE", "Y": delay, "termination": "EOS", "timing": {"T_postfork_wall": 2}}}}
    rows = [row("a", 0, 1, "low", True), row("b", 1, 0, "high")]
    rows[0]["arms"]["DELAY_2"].update(checks=[{"effective_verdict": "UNCERTAIN", "extra_steps": 1,
                                                      "early_endpoint": True, "input_tokens": 12}],
                                          budget={"forward_tokens": 100, "feedback_tokens": 5})
    stat = summary(rows, cfg)
    assert stat["quality_pairs"] == 2 and stat["delta_Y"]["estimate"] == 0
    assert stat["correct_to_wrong"] == 1 and stat["wrong_to_correct"] == 1
    assert stat["delta_T_seconds"]["estimate"] == 1 and stat["delta_T_seconds"]["bootstrap_valid"] == 1000
    assert stat["arms"]["DELAY_2"]["first_check_extra_steps"]["mean"] == 1
    assert stat["arms"]["DELAY_2"]["early_endpoint"] == 1
    assert stat["arms"]["DELAY_2"]["budget"]["forward_tokens"]["sum"] == 100
    het = heterogeneity(rows, "B", "Y", cfg)
    assert het["estimate"] == 2 and 0 < het["bootstrap_valid"] < 1000
    assert heterogeneity(rows[:1], "B", "Y", cfg)["estimate"] is None
    assert paired_effect([], "T", cfg)["estimate"] is None
    assert exclusion({"status": "COMPLETE", "Y": None, "grading": {"exclusion": "GOLD_UNPARSEABLE"}}) == "GOLD_DATA"
    assert exclusion({"status": "COMPLETE", "Y": None, "grading": {"exclusion": "VERIFY_TIMEOUT"}}) == "SCORING_ERROR"
    assert exclusion({"status": "COMPLETE", "Y": 0, "grading": {"exclusion": "PREDICTION_UNPARSEABLE"}}) is None
    assert exclusion({"status": "COMPLETE", "Y": 0, "grading": {"exclusion": "PREDICTION_ERROR"}}) is None
    assert exclusion({"status": "COMPLETE", "Y": 0, "grading": {"exclusion": "FINAL_FORMAT_ERROR"}}) is None
    assert failure_category({"status": "COMPLETE", "Y": 0, "termination": "BUDGET_ROLLBACKS"}) == "BUDGET_OR_INPUT_LIMIT"
    assert failure_category({"status": "COMPLETE", "Y": 0, "termination": "EOS_FORMAT_FAILURE"}) == "FORMAT_OR_PREDICTION_PARSE"
    assert failure_category({"status": "COMPLETE", "Y": 0, "termination": "FINAL_PASS", "grading": {"exclusion": None}}) == "WRONG_SUBMITTED"
    assert group(.2, [.2, .5]) == "low" and group(.5, [.2, .5]) == "mid"
    rows[1]["backend"] = "other_gpu"
    assert paired_effect(rows, "T", cfg)["reason"] == "MIXED_BACKENDS_SEE_BY_BACKEND"
    pending = {"snapshot": {"status": "GENERATING"}, "eligible": False, "quality_valid": False,
               "complete": False, "time_valid": False, "backend": None, "arms": dict.fromkeys(ARMS),
               "exclusions": dict.fromkeys(ARMS, "PENDING")}
    pending_metrics = phase_metrics([pending], cfg, {})
    assert pending_metrics["anchor_exclusions"] == {} and pending_metrics["pending_anchors"] == {"GENERATING": 1}
    return {"paired_analysis": "PASS", "algorithm_failure_in_denominator": "PASS", "NA_and_empty_groups": "PASS",
            "gold_and_scoring_errors_separate": "PASS", "frozen_boundary_inclusivity": "PASS", "backend_time_separation": "PASS",
            "pending_is_not_scientific_exclusion": "PASS", "saved_first_checks_and_forward_budget": "PASS"}


def mechanism_text(arms, key, fields):
    return "; ".join(f"{name}:" + "/".join(number(arms[name][key][field]["mean"]) for field in fields) for name in ARMS)
