"""Summarize a run stopped by the preregistered verifier gate."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    diagnostic = read(root / "exp2a" / "summary.json")
    if diagnostic["gate_passed"]:
        raise SystemExit("Verifier gate passed; blocked report is not applicable")
    jobs = [value for path in sorted((root / "exp2a").glob("*.json"))
            if (value := read(path)).get("job") is not None]
    holdout = [item for item in jobs if item["job"]["partition"] == "holdout"]
    errors = [item for item in holdout if item["job"]["kind"] == "error"]
    valid = [item for item in holdout if item["job"]["kind"] == "valid"]
    missed = [item for item in errors if not (item["parsed"] is not None and
              item["parsed"]["verdict"] == "FAIL" and
              item["parsed"]["first_error_step"] == item["job"]["expected_step"])]
    false_positives = [item for item in valid if item["parsed"] is not None and
                       item["parsed"]["verdict"] == "FAIL"]
    counts = {
        "missed_as_pass": sum(item["parsed"] is not None and
                              item["parsed"]["verdict"] == "PASS" for item in missed),
        "missed_as_uncertain": sum(item["parsed"] is not None and
                                   item["parsed"]["verdict"] == "UNCERTAIN" for item in missed),
        "missed_as_wrong_step": sum(item["parsed"] is not None and
                                    item["parsed"]["verdict"] == "FAIL" for item in missed),
        "false_positive_count": len(false_positives),
    }
    report = ["# 实验二 A：留出集失败分析", "",
              "工程状态：BLOCKED。以下均来自预先固定的留出诊断集；未据此修改提示词或门槛。", "",
              f"- JSON 有效：{sum(x['json_valid'] for x in holdout)}/{len(holdout)}。",
              f"- 有效前缀误报：{len(false_positives)}/{len(valid)}。",
              f"- 含错前缀精确报错：{len(errors)-len(missed)}/{len(errors)}，"
              "门槛为至少 50%。",
              f"- 漏报分解：PASS {counts['missed_as_pass']}，UNCERTAIN "
              f"{counts['missed_as_uncertain']}，错误定位 {counts['missed_as_wrong_step']}。",
              "", "## 漏报示例", ""]
    for item in missed[:5]:
        job = item["job"]
        parsed = item["parsed"] or {}
        report += [f"### {job['case_id']}（应报 Step {job['expected_step']}）", "",
                   f"题目：{job['question']}", "",
                   f"待检查段：{job['new']}", "",
                   f"实际结果：{parsed.get('verdict', 'INVALID')}，位置 "
                   f"{parsed.get('first_error_step')}。", ""]
    (root / "exp2a" / "failure_analysis.md").write_text("\n".join(report), encoding="utf-8")
    (root / "exp3").mkdir(exist_ok=True)
    (root / "exp3" / "report.md").write_text(
        "# 实验三：历史检查点与 KV 复用\n\n"
        "工程状态：BLOCKED。实验二 A 的留出集报错率未达到预注册门槛；"
        "按方案未收集正式评估路径的自然 FAIL 快照，因此未运行 B/C 配对及 A/B 正式微基准。"
        "独立的小规模工程检查不得视为本实验结果。\n", encoding="utf-8")
    exp1 = read(root / "exp1" / "summary.json")["k2"]
    overall = ["# 前三项实验阶段报告", "",
               "| 模块 | 工程状态 | 结果 |", "|---|---|---|",
               f"| 实验一 | PASS | 10/10 路径结束，{exp1['correct']}/10 标准答案正确；"
               f"{exp1['checks']} 次检查 |",
               f"| 实验二 A | BLOCKED | 留出集含错前缀报错 {len(errors)-len(missed)}/{len(errors)}，"
               "未达 50% 门槛 |",
               "| 实验二 B | 未运行 | 依赖实验二 A 通过 |",
               "| 实验三 | 未运行 | 缺少通过门槛后的自然 FAIL 快照 |", "",
               "1024 token 为本轮统一上下文上限。7B NF4 验证器在 2048 token 测试时"
               "仅余约 0.30 GiB，低于 0.5 GiB 预留，因此整轮使用 1024 token。", "",
               "工程阻塞不等于研究假设为假；当前证据显示首要问题是验证器对含错段的漏报。", ""]
    smoke_path = root / "engineering_smoke.json"
    if smoke_path.exists():
        smoke = read(smoke_path)
        divergence = smoke["first_greedy_divergence"]
        overall += ["独立工程预检：B/C 分支保留了相同剩余预算；单个真实 FAIL 快照的 A/B "
                    f"下一 token 最大 logits 差为 {smoke['consistency']['logit_max_abs_difference']}，"
                    f"32 token greedy 在第 {divergence['token_index'] + 1} 个 token 分歧。"
                    "该预检不计入实验三；正式 A/B 一致性仍待验证。", ""]
    policy_smoke_path = root / "engineering_policy_smoke.json"
    if policy_smoke_path.exists():
        policy_smoke = read(policy_smoke_path)
        base = policy_smoke["policies"]["unchecked"]
        endpoint = policy_smoke["policies"]["endpoint"]
        overall += ["控制策略单题工程预检：不检查路径调用验证器 "
                    f"{base['verifier_calls']} 次，终点检查路径调用 "
                    f"{endpoint['verifier_calls']} 次；两者答案均错误，而终点检查给出 PASS。"
                    "该预检不计入正式质量比较。", ""]
    (root / "report.md").write_text("\n".join(overall), encoding="utf-8")
    (root / "exp2a" / "failure_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    sources = [Path("README.md"), Path("pyproject.toml"), Path("uv.lock"), Path(".python-version"),
               Path("run_experiments.py"), Path("configs/4060_1024.yaml")]
    sources += sorted(Path("veriserve").glob("*.py"))
    sources += sorted(Path("scripts").glob("*.py"))
    sources += sorted(Path("tests").glob("*.py"))
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for source in sources:
            archive.add(source)
    print(root / "report.md")


if __name__ == "__main__":
    main()
