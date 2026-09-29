"""Top-level report and source snapshot for an experiment run."""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from .common import read_json

if TYPE_CHECKING:
    from .runner import RunManager


def finalize(run: "RunManager") -> None:
    root = run.root
    exp1 = read_json(root / "exp1" / "summary.json")
    diagnostic = read_json(root / "exp2a" / "summary.json")
    exp2 = read_json(root / "exp2_summary.json")
    exp3 = read_json(root / "exp3" / "summary.json")
    error = read_json(root / "run_error.json")
    rows = ["# 前三项实验阶段报告", "",
            f"配置哈希：`{run.run_hash}`。", "",
            "| 模块 | 状态 | 关键结果 |", "|---|---|---|"]
    if exp1:
        values = exp1["k2"]
        rows.append(f"| 实验一 | PASS | {values['count']} 题完成，"
                    f"{values['correct']} 题正确，{values['checks']} 次检查 |")
    else:
        rows.append("| 实验一 | 未完成 | — |")
    if diagnostic:
        h = diagnostic["metrics"]["holdout"]
        status = "PASS" if diagnostic["gate_passed"] else "BLOCKED"
        rows.append(f"| 实验二 A | {status} | 首错检出 {h['error_detection_rate']:.1%}；"
                    f"误报 {h['false_positive_rate']:.1%}；JSON 有效 {h['json_valid_rate']:.1%} |")
        if not diagnostic["gate_passed"]:
            jobs = [value for path in sorted((root / "exp2a").glob("*.json"))
                    if (value := read_json(path)).get("job") is not None]
            held = [item for item in jobs if item["job"]["partition"] == "holdout"]
            missed = [item for item in held if item["job"]["kind"] == "error" and
                      (item["parsed"] is None or item["parsed"]["verdict"] != "FAIL" or
                       item["parsed"]["first_error_step"] != item["job"]["expected_step"])]
            false_positives = [item for item in held if item["job"]["kind"] == "valid" and
                               item["parsed"] is not None and item["parsed"]["verdict"] == "FAIL"]
            details = ["# 留出集失败分析", "",
                       f"漏报或定位错误 {len(missed)} 例；误报 {len(false_positives)} 例。", ""]
            for item in missed[:10]:
                job = item["job"]
                parsed = item["parsed"] or {}
                details += [f"## {job['id']}", "", f"应报 Step {job['expected_step']}；"
                            f"实际 {parsed.get('verdict', 'INVALID')}，"
                            f"位置 {parsed.get('first_error_step')}。", ""]
            (root / "exp2a" / "failure_analysis.md").write_text(
                "\n".join(details) + "\n", encoding="utf-8")
            (root / "exp3").mkdir(exist_ok=True)
            (root / "exp3" / "report.md").write_text(
                "# 实验三\n\n状态：BLOCKED。验证器未通过预设门槛，"
                "未生成正式评估的自然 FAIL 快照。\n", encoding="utf-8")
    else:
        rows.append("| 实验二 A | 未完成 | — |")
    if exp2:
        values = exp2["evaluation"]
        rows.append("| 实验二 B | 完成 | " + "; ".join(
            f"{policy} {item['correct']}/{item['count']}"
            for policy, item in values.items()) + " |")
    else:
        rows.append("| 实验二 B | 未运行 | 需通过验证器门槛 |")
    if exp3:
        bc = exp3["bc"]
        rows.append(f"| 实验三 | {exp3['status']} | B/C {bc['count']} 对；"
                    f"非空检查点 {bc['nonempty_checkpoint_count']} 对；"
                    f"A/B {exp3['cache_status']} |")
    else:
        rows.append("| 实验三 | 未运行 | 需正式 FAIL 快照 |")
    rows += ["", "`PASS` 仅表示相应工程步骤或门槛通过，不代表答案质量有收益。"]
    if run.config["verifier"].get("type") == "process_reward":
        rows += ["", "本轮验证器为过程奖励模型。其 JSON、诊断和提示由确定性适配器生成；"
                 "它只定位低分步骤，未生成具体错误解释。"]
    if error:
        rows += ["", f"运行异常：`{error['type']}`，详见 `run_error.json`。"]
    (root / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sources = [Path("README.md"), Path("pyproject.toml"), Path("uv.lock"),
               Path(".python-version"), Path("run_experiments.py"), run.config_path,
               Path("验证器替换记录.md")]
    for folder in ("veriserve", "scripts", "tests"):
        sources.extend(sorted(Path(folder).glob("*.py")))
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for source in sources:
            if source.exists():
                archive.add(source)
