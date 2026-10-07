"""Probe figures and reports, separate from scientific execution identity."""
from __future__ import annotations
import json
import io
from datetime import datetime
from zoneinfo import ZoneInfo
from veriserve_research import ROOT
from ..artifacts.io import atomic_bytes, read_json, git
HERE = ROOT / "stages/stage2"


def plot_metrics(metrics, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for offset, method, color in ((-0.16, "A", "#4c78a8"), (0, "B", "#e45756"), (0.16, "C", "#72b7b2")):
        for i in (1, 2, 3):
            value = metrics[f"step_{i}"].get("methods", {}).get(method, {})
            center, ci = value.get("auroc"), value.get("ci95")
            if center is None or ci is None:
                ax.text(i + offset, 0.08 + (offset + .16) / 4, "NA", ha="center", fontsize=8, color=color)
            else:
                ax.vlines(i + offset, ci[0], ci[1], color=color)
                ax.plot(i + offset, center, "o", color=color)
        ax.plot([], [], "o", color=color, label=method)
    ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
    ax.set(xticks=[1, 2, 3], xlabel="Effective intermediate step", ylabel="Test AUROC (95% question bootstrap CI)", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=160)
    atomic_bytes(run / "step_auroc.png", b.getvalue())
    plt.close(fig)


def metric_text(value):
    if not value or value.get("auroc") is None:
        return "NA"
    ci = value.get("ci95")
    return f"{value['auroc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{value['auroc']:.4f} [NA]"


def report(run, manifest, counts, metrics=None, rows=None, coverage=None, failures=None, reason=None, *, source_run=None):
    source_run = source_run or run
    cfg = manifest["config"]
    fit = read_json(source_run / "fit_status.json", {})
    lines = ["# 第二阶段：逐步 hidden 最终答错风险探针", "",
             f"运行：`{source_run.name}`；报告更新于 {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。", "",
             "标签是完整轨迹最终答错（1）或答对（0）。主指标仅用于完成且可评分轨迹的正常中途步骤；不是局部错误标签。", "",
             f"模型 `{cfg['model']}`，revision `{cfg['revisions']['model']}`；tokenizer `{cfg['revisions']['tokenizer']}`。",
             f"数据 `{cfg['dataset']}` / test，revision `{cfg['revisions']['dataset']}`。",
             f"按 unique_id 排序后用种子 {cfg['seed']} 洗牌：100 train / 200 test / 10 smoke，剩余 190 不使用；原始字段见 manifest.json。",
             "BF16 / 单卡 / batch 1 / SDPA / greedy；4096 新 token、8192 总 token；C=0.1、max_iter=2000；所有 Transformer 层分别做训练内五折选层。", "",
             "这是采用逐步训练、题目权重、明确 Step 标记和独立测试的适配实验，不宣称完全复现论文。", "",
             "特征逐个端点重前向其原始 token 前缀，保持 BF16 / SDPA / use_cache=False。",
             "原单次整段重前向与独立前缀的逐坐标比较在 smoke 中失败；诊断原样保留，不称通过，正式特征不采用该方式。",
             "逐前缀提取另与独立 decoder 的最后位置比较，并检验同长度未来替换及错一 token 的负对照；容差未扩大。", "",
             "## 实际环境", "", "```json", json.dumps(manifest["environment"], ensure_ascii=False, indent=2), "```", "",
             "## 采集与排除", "", "```json", json.dumps(counts, ensure_ascii=False, indent=2), "```", ""]
    if reason:
        lines += [f"未完成原因：{reason}", ""]
    lines += ["## 自检与备份", "", "```json", json.dumps({
        "cpu_self_check": read_json(source_run / "self_check.json"),
        "additional_implementation_checks": read_json(HERE / "implementation_checks.json"),
        "smoke": read_json(source_run / "smoke_checks.json"), "resume": read_json(source_run / "resume_check.json"),
        "backup": read_json(source_run / "backup.json")}, ensure_ascii=False, indent=2), "```", ""]
    if metrics:
        lines += ["## 独立测试", "", f"选层（A/B 独立选择）：`{fit.get('selected', {})}`。",
                  f"训练覆盖：`{coverage}`；失败：`{failures}`。", "",
                  "| 位置 | 题数（正确/错误） | A AUROC [95% CI] | B AUROC [95% CI] | C AUROC [95% CI] |",
                  "|---|---|---|---|---|"]
        for key in ("step_1", "step_2", "step_3", "intermediate_all", "full_trajectory_end"):
            item = metrics[key]
            lines.append(f"| {key} | {item['questions']} ({item['correct']}/{item['wrong']}) | "
                         + " | ".join(metric_text(item["methods"].get(m)) for m in ("A", "B", "C")) + " |")
        lines += ["", "C 不用于完整轨迹末尾对照；A/B 末尾单独报告。所有中途比较使用共同有效位置。", "",
                  "每题总权重为 1；bootstrap 按题重抽 1000 次，整题全部步骤随同抽取，单类重复样本跳过。有效次数与 paired AUROC 差及其 CI 见 metrics.json。", "",
                  "```json", json.dumps({k: metrics[k]["paired"] for k in ("step_1", "step_2", "step_3", "intermediate_all")}, indent=2), "```", "",
                  "![前 3 个有效中途步骤](step_auroc.png)", "", "## 代表性风险轨迹", ""]
        by_id = {}
        for row in rows or []:
            by_id.setdefault(row["id"], []).append(row)
        if "B" in fit.get("selected", {}):
            for label in (0, 1):
                candidates = sorted([r for r in rows or [] if r["effective_step"] == 1 and r["label"] == label], key=lambda r: r["B"])
                for first in ([candidates[0], candidates[-1]] if len(candidates) > 1 else candidates):
                    trajectory = ", ".join(f"{r['effective_step']}:{r['B']:.3f}" for r in by_id[first["id"]])
                    lines += [f"- `{first['id']}`，最终标签 {label}，B 的逐步风险：{trajectory}。"]
        main = metrics["intermediate_all"]
        b = main["methods"].get("B", {})
        delta = main["paired"].get("B-C", {})
        supported = bool(b.get("ci95") and b["ci95"][0] > .5 and delta.get("ci95") and delta["ci95"][0] > 0)
        lines += ["", "## 结论", "",
                  ("此设置下，B 在全部中途位置提供了超过随机与简单进度基线的最终答错风险信号（两项 bootstrap CI 均支持）。"
                   if supported else "此设置下，未同时获得超过随机与简单进度基线的明确证据；低分、反向或跨零区间按原方向报告。"), ""]
    else:
        lines += ["AUROC / CI / 所选层：NA；正式数据不足或未采集，不能回答是否存在信号。", ""]
    lines += ["这项实验不能识别首个错误步骤、判断当前检查是否值得、证明降低完成时间或推广到其他模型。",
              "离线重前向耗时不代表在线提取成本；生成、重前向与探针评分耗时分别记录。", "",
              "步骤端点对齐原始 token，跨界 token 向前对齐；包含 boxed 或明确最终答案的步骤排除，最后正常步骤保留 near_end 标记。",
              "hidden_states[1..L] 排除 embedding，Qwen2 的索引 L 含最终 norm；轨迹末尾取 EOS/控制 token 前最后完整内容 token。", "",
              "评分仅解析单独 Final answer 行的最后完整 boxed；Math-Verify 标准答案包装为 $...$，无字符串 fallback，解析/比较超时单独排除。", "",
              "## 文件与恢复", "", "配置与环境见 config.json / manifest.json；运行日志见 events.jsonl；探针参数见 probe_{A,B,C}.npz/json。",
              "分支：[experiment/step-hidden-probe](https://github.com/WanderingAlgebra/VeriServe/tree/experiment/step-hidden-probe)。",
              f"本地 HEAD：`{git('rev-parse', 'HEAD')}`；远端备份状态与提交以 backup.json 为准。", "",
              "从仓库根目录执行（替换成可用 Python 环境）：", "", "```bash",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --self-check",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --prepare-only --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --backup-only --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --smoke --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --resume",
              "python -m stages.stage2.run_probe --config stages/stage2/config.json --fit-only --resume", "```", ""]
    atomic_bytes(run / "report.md", "\n".join(lines).encode())
