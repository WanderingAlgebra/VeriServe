"""Experiment four; reuse frozen stage-one dependencies without rerunning them."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import random
import tarfile
import time
from pathlib import Path
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from veriserve.common import atomic_json, load_config, read_json, stable_hash
from veriserve.data import is_correct
from veriserve.model import LanguageModel
from veriserve.runner import RunManager, request_seed
from veriserve.state import PathState
from transformers import AutoTokenizer


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_seconds(state: PathState) -> float:
    decisions = sum(e.get("seconds", 0) for e in state.events if e["event"] == "policy_decision")
    return state.generator_seconds + state.verifier_seconds - state.offline_restore_seconds + decisions


def choose_fixed(development: dict) -> str:
    return min(("k1", "k2", "k4"), key=lambda p: (
        -development[p]["accuracy"],
        (development[p]["generator_seconds"] + development[p]["verifier_seconds"]
         - development[p]["offline_restore_seconds"]) / development[p]["count"],
        int(p[1:]),
    ))


def geometric_interval(probability: float, seed: int) -> int:
    if not 0 < probability <= 1:
        raise ValueError("Random check probability must be in (0, 1]")
    if probability == 1:
        return 1
    return 1 + int(math.log1p(-random.Random(seed).random()) / math.log1p(-probability))


def logical_prefill_needed(state: PathState) -> bool:
    # A restored oracle snapshot is offline work; charge only the initial input or a real FAIL.
    for event in reversed(state.events):
        if event["event"] == "generate":
            return False
        if event["event"] == "rollback":
            return True
    return True


def accept_generation(state: PathState, result, input_tokens: int, logical: bool) -> None:
    before = state.prefill_tokens
    state.rng_state = result.rng_state
    state.accept_generated(result.kept_ids, result.pending_ids, result.reason, result.answer,
                           input_tokens, result.prefill_seconds, result.decode_seconds,
                           current_step=result.current_step)
    # PathState assumes each generation pause leads to a check. Experiment four also pauses
    # for snapshots; a resumed, already-paid prefix must not be charged as another FAIL.
    if not logical and state.prefill_tokens > before:
        state.prefill_tokens = before
        event = next(e for e in reversed(state.events) if e["event"] == "generate")
        if event["prefill_kind"] == "FAIL_RESTORE":
            state.feedback_prefill_seconds -= result.prefill_seconds
        state.offline_restore_seconds += result.prefill_seconds
        event["prefill_kind"] = "OFFLINE_RESTORE"


def branch_state(snapshot: dict, action: str, replicate: int,
                 run_hash: str, seed: int) -> PathState:
    state = PathState.from_dict(copy.deepcopy(snapshot["state"]))
    state.config_hash = run_hash
    state.policy = action
    state.seed = request_seed(seed, f"oracle/{snapshot['id']}/{replicate}")
    state.rng_state = None  # Same independent future seed for all four actions; history is fixed.
    state.history_id = stable_hash((state.history_id, snapshot["id"], action, replicate))[:16]
    state.status = "WAIT_CHECK" if action == "now" else "WAIT_GENERATION"
    state.record("oracle_branch", snapshot_id=snapshot["id"], action=action,
                 replicate=replicate, inherited_checks=len(state.checks),
                 inherited_generated_tokens=state.generated_tokens,
                 inherited_prefill_tokens=state.prefill_tokens,
                 inherited_verifier_calls=state.verifier_calls,
                 inherited_service_seconds=service_seconds(state))
    return state


class Experiment4(RunManager):
    def __init__(self, config_path: Path, resume: bool):
        # Reuse the proven execution methods; this constructor freezes an independent run.
        self.config_path = config_path
        self.settings = load_config(config_path)
        self.resume = resume
        self.source = Path(self.settings["source_run"]).resolve()
        parent = read_json(self.source / "config.json")
        if not parent or not read_json(self.source / "exp2_summary.json", {}).get("diagnostic", {}).get("gate_passed"):
            raise ValueError("Experiment four requires a completed parent with a passing verifier gate")
        self.config = parent["config"]
        self.assets = parent["assets"]
        self.splits = read_json(self.source / "splits.json")
        if len(self.splits["development"]) != 40 or len(self.splits["evaluation"]) != 100:
            raise ValueError("Expected the frozen 40/100 question splits")
        if len(set(self.settings["selection_seeds"] + [self.settings["evaluation_seed"]])) != 3:
            raise ValueError("Two selection seeds and a separate evaluation seed are required")
        development = read_json(self.source / "exp2b_dev/summary.json")
        if self.settings["fixed_selection"] != "highest_dev_accuracy_then_lowest_service_cost":
            raise ValueError("Unsupported fixed-interval selection rule")
        if self.settings["random_probability"] != "inverse_selected_fixed_interval":
            raise ValueError("Unsupported random policy selection rule")
        if self.settings["actions"] != ["now", "delay1", "delay2", "endpoint_only"]:
            raise ValueError("The frozen local action set must contain the four planned actions")
        self.fixed = choose_fixed(development)
        self.probability = 1 / int(self.fixed[1:])
        self.cost_unit = (development["unchecked"]["generator_seconds"]
                          - development["unchecked"]["offline_restore_seconds"]) / development["unchecked"]["count"]
        self.history_rows = random.Random(self.settings["seed"]).sample(
            self.splits["evaluation"], self.settings["history_questions"])
        self.source_inputs = {}
        for name in ("config.json", "splits.json", "exp2_summary.json", "exp2b_dev/summary.json"):
            self.source_inputs[name] = file_hash(self.source / name)
        for stage, policies, rows in (
            ("exp2b_dev", ["unchecked", "endpoint", "k1", "k2", "k4"], self.splits["development"]),
            ("exp2b_eval", ["unchecked", "endpoint"], self.splits["evaluation"]),
        ):
            for policy in policies:
                for row in rows:
                    relative = f"states/{stage}/{policy}/{row['id']}.json"
                    self.source_inputs[relative] = file_hash(self.source / relative)
        sources = sorted(Path("veriserve").glob("*.py")) + [Path(__file__), config_path]
        self.code_hashes = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p): file_hash(p)
                            for p in sources}
        # Reusing old baseline trajectories requires unchanged core generation and scoring code.
        with tarfile.open(self.source / "source_snapshot.tar.gz") as archive:
            for p in sorted(Path("veriserve").glob("*.py")):
                archived = archive.extractfile(str(p))
                if archived is None or archived.read() != p.read_bytes():
                    raise ValueError(f"Core code changed since the parent run: {p}; baseline reuse is unsafe")
        environment = {name: version(name) for name in ("torch", "transformers", "bitsandbytes", "accelerate")}
        environment["cuda"] = torch.version.cuda
        identity = {"settings": self.settings, "parent_hash": parent["hash"], "environment": environment,
                    "inputs": self.source_inputs, "code": self.code_hashes}
        self.run_hash = stable_hash(identity)
        self.root = Path("runs") / self.run_hash[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.assets["generator"]["path"], use_fast=True)
        self.progress_stage = "initializing"
        self.reused = []
        design = {
            "run_hash": self.run_hash, "parent_run": str(self.source),
            "config": self.config, "settings": self.settings, "assets": self.assets,
            "fixed_policy": self.fixed, "random_check_probability": self.probability,
            "cost_unit_seconds": self.cost_unit,
            "environment": environment,
            "selection_data": "parent experiment-two development only",
            "history_question_ids": [row["id"] for row in self.history_rows],
            "development_ids": [row["id"] for row in self.splits["development"]],
            "evaluation_ids": [row["id"] for row in self.splits["evaluation"]],
            "identity": identity,
            "notes": ["Generator and verifier are loaded in offline waves; no online latency claim.",
                      "Geometric intervals are equivalent to independent Bernoulli checks at step boundaries.",
                      "Continuation seed is shared across actions, independent across replicates.",
                      "Initial-checkpoint diagnostics are additional to the historical-PASS snapshot group."],
        }
        frozen = read_json(self.root / "config.json")
        if frozen is not None and frozen != design:
            raise ValueError("Frozen experiment-four manifest mismatch")
        atomic_json(self.root / "config.json", design)
        atomic_json(self.root / "splits.json", self.splits)
        maximum_branches = len(self.history_rows) * len(self.settings["snapshot_groups"]) * 4 * 3
        atomic_json(self.root / "run_manifest.json", {
            "experiment": 4, "source_run": str(self.source), "fixed_policy": self.fixed,
            "random_check_probability": self.probability,
            "development_new_paths": 40, "development_reused_paths": 200,
            "evaluation_new_paths": 200, "evaluation_reused_paths": 200,
            "history_paths": len(self.history_rows) * len(self.settings["snapshot_groups"]),
            "maximum_oracle_branches": maximum_branches,
            "selection_seeds": self.settings["selection_seeds"],
            "evaluation_seed": self.settings["evaluation_seed"],
            "estimated_machine_hours": [2, 8],
            "estimate_basis": "Prior complete-path service times plus offline loading; update from progress logs.",
        })
        with tarfile.open(self.root / "source_snapshot.tar.gz", "w:gz") as archive:
            for source in sources:
                archive.add(source, arcname=str(source.relative_to(Path.cwd()) if source.is_absolute() else source))
        if not (self.root / "report.md").exists():
            (self.root / "report.md").write_text(
                "# 实验四\n\n状态：运行中。进度见 `progress.json`，冻结配置见 `config.json`。\n")

    def effective_policy(self, state: PathState) -> str:
        if state.policy == "random":
            decision_seed = request_seed(state.seed, f"random/{state.history_id}/{state.verifier_calls}")
            interval = geometric_interval(self.probability, decision_seed)
            state.record("random_check_interval", probability=self.probability,
                         interval=interval, decision_seed=decision_seed)
            return f"k{interval}"
        if state.policy == "history_initial":
            return "k1"
        if state.policy == "history_historical_pass":
            return "k1" if state.checkpoints else self.settings["history_policy"]
        if state.policy in self.settings["actions"]:
            if state.policy == "endpoint_only":
                return "endpoint"
            branch = next(e for e in reversed(state.events) if e["event"] == "oracle_branch")
            if len(state.checks) == branch["inherited_checks"]:
                return {"now": self.fixed, "delay1": "k1", "delay2": "k2"}[state.policy]
            return self.fixed
        return state.policy

    def capture_snapshot(self, state: PathState) -> None:
        if not state.policy.startswith("history_") or state.status != "WAIT_CHECK":
            return
        group = state.policy.removeprefix("history_")
        if state.candidate is not None:
            if group == "initial" or state.checkpoints:
                state.finish("HISTORY_FINAL_BEFORE_SNAPSHOT", state.candidate)
            return
        if group == "historical_pass" and not state.checkpoints:
            return
        identifier = f"{group}/{state.request_id}"
        raw = state.to_dict()
        snapshot = {"id": identifier, "group": group, "question_id": state.request_id,
                    "state": raw, "state_hash": stable_hash(raw)}
        atomic_json(self.root / "snapshots" / f"{identifier}.json", snapshot)
        state.status = "SNAPSHOT_READY"
        state.record("snapshot_saved", snapshot_id=identifier, state_hash=snapshot["state_hash"])

    def _generate_wave(self, states: dict[Path, PathState]) -> None:
        pending = [(p, s) for p, s in states.items() if s.status == "WAIT_GENERATION"]
        if not pending:
            return
        started = time.perf_counter()
        model = LanguageModel(self.assets["generator"]["path"])
        try:
            for index, (path, state) in enumerate(pending, 1):
                if not state.can_generate():
                    self._save_state(path, state)
                    continue
                logical = logical_prefill_needed(state)
                input_tokens = len(state.context_ids) + len(state.pending_ids)
                if logical and state.prefill_tokens + input_tokens > state.budget.prefill_tokens:
                    state.finish("BUDGET_PREFILL", state.candidate)
                    self._save_state(path, state)
                    continue
                original_policy = state.policy
                decision_start = time.perf_counter()
                effective = self.effective_policy(state)
                state.record("policy_decision", effective_policy=effective,
                             seconds=time.perf_counter() - decision_start)
                state.merge_pending()
                try:
                    state.policy = effective
                    result = model.generate_path(state, self.config)
                except torch.cuda.OutOfMemoryError:
                    atomic_json(self.root / "oom.json", {"path": str(path), "phase": self.progress_stage})
                    raise
                finally:
                    state.policy = original_policy
                accept_generation(state, result, input_tokens, logical)
                self.capture_snapshot(state)
                self._save_state(path, state)
                if index % 5 == 0 or index == len(pending):
                    self.progress(states, generator_wave_done=index, generator_wave_total=len(pending))
        finally:
            model.close()
            self._append_phase("generator", len(pending), time.perf_counter() - started)

    def progress(self, states: dict[Path, PathState], **extra) -> None:
        value = {"stage": self.progress_stage, "total": len(states),
                 "finished": sum(s.finished for s in states.values()),
                 "snapshots": sum(s.status == "SNAPSHOT_READY" for s in states.values()),
                 "wait_generation": sum(s.status == "WAIT_GENERATION" for s in states.values()),
                 "wait_check": sum(s.status == "WAIT_CHECK" for s in states.values()),
                 "updated_at": time.time(), **extra}
        atomic_json(self.root / "progress.json", value)
        print(json.dumps(value), flush=True)

    def run_paths(self, states: dict[Path, PathState]) -> dict[Path, PathState]:
        for _ in range(50):
            if all(s.finished or s.status == "SNAPSHOT_READY" for s in states.values()):
                self.progress(states)
                return states
            self._generate_wave(states)
            self._verify_wave(states)
            self.progress(states)
        raise RuntimeError("More than 50 waves; check state or budget accounting")

    def reused_states(self, stage: str, policies: list[str], rows: list[dict]) -> dict[Path, PathState]:
        values = {}
        for policy in policies:
            for row in rows:
                path = self.source / "states" / stage / policy / f"{row['id']}.json"
                state = PathState.from_dict(read_json(path))
                if not state.finished or state.question != row["question"] or state.gold != row["gold"]:
                    raise ValueError(f"Invalid source trajectory: {path}")
                values[path] = state
                self.reused.append({"path": str(path), "sha256": file_hash(path),
                                    "question_id": row["id"], "policy": policy,
                                    "source_config_hash": state.config_hash})
        return values

    def execute(self) -> None:
        self.progress_stage = "development"
        dev = self.reused_states("exp2b_dev", ["unchecked", "endpoint", "k1", "k2", "k4"],
                                 self.splits["development"])
        new = self.load_states("exp4_dev", ["random"], self.splits["development"])
        self.run_paths(new)
        dev.update(new)
        self._summary("exp4_dev", dev)
        self.progress_stage = "evaluation"
        evaluation = self.reused_states("exp2b_eval", ["unchecked", "endpoint"], self.splits["evaluation"])
        new = self.load_states("exp4_eval", [self.fixed, "random"], self.splits["evaluation"])
        self.run_paths(new)
        evaluation.update(new)
        self._summary("exp4_eval", evaluation)
        atomic_json(self.root / "reused_trajectories.json", self.reused)
        self.progress_stage = "common_histories"
        histories = self.load_states("exp4_history", [f"history_{g}" for g in self.settings["snapshot_groups"]],
                                     self.history_rows)
        self.run_paths(histories)
        self._collect_logs("exp4_history", histories)
        snapshots, coverage = [], []
        for state in histories.values():
            group = state.policy.removeprefix("history_")
            path = self.root / "snapshots" / group / f"{state.request_id}.json"
            snapshot = read_json(path)
            if snapshot:
                if snapshot["state_hash"] != stable_hash(snapshot["state"]):
                    raise ValueError(f"Snapshot checksum mismatch: {path}")
                snapshots.append(snapshot)
            coverage.append({"question_id": state.request_id, "group": group,
                             "available": snapshot is not None,
                             "reason": "SNAPSHOT_READY" if snapshot else state.termination})
        atomic_json(self.root / "snapshot_coverage.json", coverage)
        self.progress_stage = "oracle_branches"
        branches, metadata = {}, []
        replicates = self.settings["selection_seeds"] + [self.settings["evaluation_seed"]]
        for snapshot in snapshots:
            for replicate in replicates:
                for action in self.settings["actions"]:
                    branch_id = f"{snapshot['group']}/{snapshot['question_id']}/{replicate}/{action}"
                    path = self.root / "states/exp4_oracle" / f"{branch_id}.json"
                    saved = read_json(path) if self.resume else None
                    branches[path] = PathState.from_dict(saved) if saved else branch_state(
                        snapshot, action, replicate, self.run_hash, self.settings["seed"])
                    self._save_state(path, branches[path])
                    metadata.append({"branch_id": branch_id, "snapshot_id": snapshot["id"],
                                     "group": snapshot["group"], "question_id": snapshot["question_id"],
                                     "replicate": replicate, "action": action, "path": str(path),
                                     "history_service_seconds": service_seconds(PathState.from_dict(snapshot["state"]))})
        atomic_json(self.root / "branch_manifest.json", metadata)
        self.run_paths(branches)
        self._collect_logs("exp4_oracle", branches)
        rows = []
        for item in metadata:
            state = branches[Path(item["path"])]
            inherited = next(e for e in state.events if e["event"] == "oracle_branch")
            rows.append({**item, "correct": is_correct(state.final_answer, state.gold),
                         "answer": state.final_answer, "unfinished": state.final_answer is None,
                         "termination": state.termination, "seed": state.seed,
                         "suffix_service_seconds": service_seconds(state) - item["history_service_seconds"],
                         "full_service_seconds": service_seconds(state),
                         "new_checks": state.verifier_calls - inherited["inherited_verifier_calls"],
                         "new_generated_tokens": state.generated_tokens - inherited["inherited_generated_tokens"],
                         "new_prefill_tokens": state.prefill_tokens - inherited["inherited_prefill_tokens"]})
        (self.root / "exp4_oracle/branches.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        self.make_report(evaluation, rows, coverage)
        self.progress_stage = "complete"
        self.progress(branches)

    def make_report(self, evaluation: dict[Path, PathState], rows: list[dict], coverage: list[dict]) -> None:
        directory = self.root / "exp4"
        directory.mkdir(exist_ok=True)
        full = []
        for state in evaluation.values():
            full.append({"question_id": state.request_id, "policy": state.policy,
                         "correct": is_correct(state.final_answer, state.gold),
                         "unfinished": state.final_answer is None, "checks": state.verifier_calls,
                         "service_seconds": service_seconds(state), "termination": state.termination,
                         "generator_seconds": state.generator_seconds,
                         "verifier_seconds": state.verifier_seconds,
                         "offline_restore_seconds": state.offline_restore_seconds})
        policies = ["unchecked", "endpoint", self.fixed, "random"]
        summary = {p: {"count": len(group := [r for r in full if r["policy"] == p]),
                       "correct": sum(r["correct"] for r in group),
                       "unfinished": sum(r["unfinished"] for r in group),
                       "checks": sum(r["checks"] for r in group),
                       "mean_service_seconds": sum(r["service_seconds"] for r in group) / len(group)}
                   for p in policies}
        paired = []
        indexed = {(r["question_id"], r["policy"]): r for r in full}
        for a, b in ((self.fixed, "unchecked"), (self.fixed, "endpoint"), ("random", self.fixed)):
            differences = [int(indexed[q["id"], a]["correct"]) - int(indexed[q["id"], b]["correct"])
                           for q in self.splits["evaluation"]]
            rng = random.Random(self.settings["seed"])
            sampled = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                             for _ in range(2000))
            paired.append({"policy_a": a, "policy_b": b, "a_only_correct": differences.count(1),
                           "b_only_correct": differences.count(-1),
                           "accuracy_difference": sum(differences) / len(differences),
                           "bootstrap_95_interval": [sampled[50], sampled[1949]]})
        local = analyze_oracle(rows, self.settings, self.cost_unit)
        atomic_json(directory / "summary.json", {"status": "COMPLETE" if rows else "INCONCLUSIVE",
                    "baselines": summary, "paired": paired, "oracle": local, "coverage": coverage})
        write_csv(directory / "full_paths.csv", full)
        write_csv(directory / "oracle_branches.csv", rows)
        write_csv(directory / "oracle_comparison.csv", local["comparisons"])
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, value in summary.items():
            ax.scatter(value["mean_service_seconds"], value["correct"] / value["count"], label=policy)
            ax.annotate(policy, (value["mean_service_seconds"], value["correct"] / value["count"]))
        ax.set(xlabel="Mean logical service time (seconds)", ylabel="Accuracy (100 questions)",
               title="Experiment 4: complete-path baselines")
        fig.tight_layout()
        fig.savefig(directory / "quality_cost.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for group in self.settings["snapshot_groups"]:
            values = [v for v in local["comparisons"] if v["group"] == group]
            if values:
                ax.plot([v["lambda"] for v in values], [v["utility_gap"] for v in values], "o-", label=group)
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set(xlabel="Cost weight lambda", ylabel="Held-seed utility gap vs uniform reference",
               title="Local timing oracle (selection seeds 17/29, evaluation 43)")
        if local["comparisons"]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "oracle_gap.png", dpi=160)
        plt.close(fig)
        lines = ["# 实验四：检查时机", "", "工程状态：完成。研究结论依据下列表格，不预设正收益。", "",
                 f"源运行：`{self.source.name}`；本轮运行：`{self.root.name}`。",
                 f"开发集冻结固定间隔 `{self.fixed}`，随机检查概率 `{self.probability}`；成本单位 {self.cost_unit:.3f} 秒。", "",
                 "## 同一批 100 题的完整路径", "",
                 "| 策略 | 答对 | 检查次数 | 未完成 | 平均服务秒数 |", "| --- | ---: | ---: | ---: | ---: |"]
        for policy, value in summary.items():
            lines.append(f"| {policy} | {value['correct']}/100 | {value['checks']} | {value['unfinished']} | {value['mean_service_seconds']:.3f} |")
        lines += ["", "不检查与终点检查引用原运行，未重复计为新样本；新路径沿用同一模型、题目、初始种子和预算。",
                  "服务成本扣除了离线组织所需的前缀恢复，FAIL 后机制本身要求的重算仍计入；跨运行测时可能受设备状态影响。", "",
                  "## 同一快照的局部时机比较", "",
                  "每个动作在 17、29 号种子上的平均效用用于选择，43 号种子独立评价。两类快照分别报告。",
                  "统一动作参考也用 17、29 的标签选择，只是带事后信息的局部参考，不是开发集训练的可部署策略。", "",
                  "| 快照类型 | 可用/选题数 | 说明 |", "| --- | ---: | --- |"]
        for group in self.settings["snapshot_groups"]:
            relevant = [r for r in coverage if r["group"] == group]
            lines.append(f"| {group} | {sum(r['available'] for r in relevant)}/{len(relevant)} | 缺少快照的题保留原因，不补选有利题 |")
        lines += ["", "| 类型 | λ | 快照数 | 所选动作答对 | 统一参考 | 参考答对 | 独立种子效用差 | 95% 区间 |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for item in local["comparisons"]:
            lines.append(f"| {item['group']} | {item['lambda']} | {item['count']} | {item['oracle_correct']} | {item['uniform_action']} | {item['uniform_correct']} | {item['utility_gap']:.3f} | {item['bootstrap_95_interval']} |")
        lines += ["", "## 如何解读", "",
                  "首先比较固定间隔、随机检查与终点检查的质量和实际成本；检查更频繁并不自动代表更好。",
                  "局部 oracle 仅在相同快照、四个候选动作和固定后续策略范围内提供有先知信息的参考；不是可部署方法，也不是全局最优。",
                  "若独立种子的差距很小、区间跨零或覆盖不足，应报告证据不足，不扩大为已证明自适应有收益。",
                  "初始检查点组与真实历史 PASS 组分开分析；未完成和预算耗尽均保留。",
                  "实验二的同一批测试题已被观察，因此这是沿用冻结协议的后续诊断，不是全新未见测试集。", "",
                  "## 产物与复现", "",
                  "`config.json` 固定模型、配置、依赖哈希、题目、策略、种子与成本权重；`snapshots/` 保存完整历史。",
                  "`full_paths.csv`、`oracle_branches.csv`、`oracle_comparison.csv`、`quality_cost.png`、`oracle_gap.png` 在本目录。",
                  "`snapshot_coverage.json` 保留所有未形成快照的题；`reused_trajectories.json` 标明旧路径来源与哈希。",
                  "`phase_times.jsonl` 记录波次墙钟时间，含模型加载与卸载，不当作线上排队延迟。"]
        report = "\n".join(lines) + "\n"
        (directory / "report.md").write_text(report)
        (self.root / "report.md").write_text(report)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_oracle(rows: list[dict], settings: dict, cost_unit: float) -> dict:
    indexed = {(r["snapshot_id"], r["action"], r["replicate"]): r for r in rows}
    actions = settings["actions"]
    comparisons, selections, action_summaries = [], [], []
    held = settings["evaluation_seed"]
    for group in settings["snapshot_groups"]:
        identifiers = sorted({r["snapshot_id"] for r in rows if r["group"] == group})
        if not identifiers:
            continue
        for action in actions:
            values = [indexed[s, action, held] for s in identifiers]
            action_summaries.append({"group": group, "action": action, "count": len(values),
                                     "correct": sum(r["correct"] for r in values),
                                     "unfinished": sum(r["unfinished"] for r in values),
                                     "mean_suffix_seconds": sum(r["suffix_service_seconds"] for r in values)/len(values)})
        for weight in settings["cost_weights"]:
            def utility(row):
                return int(row["correct"]) - weight * row["suffix_service_seconds"] / cost_unit
            estimates = {(s, a): sum(utility(indexed[s, a, seed]) for seed in settings["selection_seeds"])
                         / len(settings["selection_seeds"]) for s in identifiers for a in actions}
            uniform = max(actions, key=lambda a: sum(estimates[s, a] for s in identifiers))
            chosen = {s: max(actions, key=lambda a: estimates[s, a]) for s in identifiers}
            oracle_rows = [indexed[s, chosen[s], held] for s in identifiers]
            uniform_rows = [indexed[s, uniform, held] for s in identifiers]
            differences = [utility(a)-utility(b) for a,b in zip(oracle_rows,uniform_rows)]
            rng = random.Random(settings["seed"])
            bootstrap = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                               for _ in range(2000))
            comparisons.append({"group": group, "lambda": weight, "count": len(identifiers),
                                "oracle_correct": sum(r["correct"] for r in oracle_rows),
                                "uniform_action": uniform, "uniform_correct": sum(r["correct"] for r in uniform_rows),
                                "utility_gap": sum(differences)/len(differences),
                                "bootstrap_95_interval": [round(bootstrap[50], 4), round(bootstrap[1949], 4)],
                                "oracle_mean_suffix_seconds": sum(r["suffix_service_seconds"] for r in oracle_rows)/len(identifiers),
                                "uniform_mean_suffix_seconds": sum(r["suffix_service_seconds"] for r in uniform_rows)/len(identifiers)})
            selections.extend({"snapshot_id": s, "lambda": weight, "chosen_action": chosen[s],
                               "uniform_action": uniform, "selection_seed_estimates": {a: estimates[s,a] for a in actions}}
                              for s in identifiers)
    return {"comparisons": comparisons, "selections": selections, "held_seed_actions": action_summaries}


def run(config: Path, resume: bool = True) -> Path:
    experiment = Experiment4(config, resume)
    print(f"Experiment 4 run: {experiment.root}", flush=True)
    print(f"Frozen: fixed={experiment.fixed}, random probability={experiment.probability}", flush=True)
    with (experiment.root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute()
            (experiment.root / "run_error.json").unlink(missing_ok=True)
        except Exception as exc:
            atomic_json(experiment.root / "run_error.json", {"type": type(exc).__name__, "message": str(exc),
                                                             "stage": experiment.progress_stage})
            raise
    return experiment.root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp4_4060.yaml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
