from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import psutil
from transformers import AutoTokenizer

from .common import atomic_json, load_config, read_json, stable_hash
from .data import is_correct, prepare_splits
from .model import (GENERATOR_TEMPLATE, VERIFIER_TEMPLATE, LanguageModel, feedback_text,
                    parse_verdict, verification_prompt)
from .prm import ProcessRewardModel
from .state import Budget, PathState


def request_seed(seed: int, request_id: str) -> int:
    return int(stable_hash((seed, request_id))[:12], 16)


def parse_steps(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


class RunManager:
    def __init__(self, config_path: Path, resume: bool = False):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.resume = resume
        splits_path = Path(self.config.get("splits_path", ".cache/splits.json"))
        if self.config.get("require_frozen_splits") and not splits_path.exists():
            raise RuntimeError(f"Frozen split manifest is missing: {splits_path}")
        self.splits = prepare_splits(splits_path, int(self.config["seed"]))
        self.assets = read_json(Path(".cache/assets.json"), {})
        if not all(role in self.assets for role in ("generator", "verifier")):
            raise RuntimeError("Run python -m scripts.fetch_assets before experiments")
        for role in ("generator", "verifier"):
            if self.assets[role]["id"] != self.config[role]["id"]:
                raise RuntimeError(f"Cached {role} does not match config; run fetch_assets for this config")
        if self.config["verifier"].get("type") == "process_reward":
            choice = read_json(Path(".cache/prm_dev_threshold.json"))
            if (not choice or not choice.get("selected") or
                choice.get("model_revision") != self.assets["verifier"]["revision"] or
                choice.get("splits_hash") != stable_hash(self.splits) or
                float(choice["selected"]["threshold"]) != float(self.config["verifier"]["threshold"])):
                raise RuntimeError("PRM threshold is not frozen for this model and split manifest")
        digest = hashlib.sha256()
        for source in sorted(Path("veriserve").glob("*.py")) + [Path("run_experiments.py")]:
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        self.run_hash = stable_hash({
            "config": self.config, "data_revisions": self.splits["revisions"],
            "splits_hash": stable_hash(self.splits),
            "model_revisions": {key: value["revision"] for key, value in self.assets.items()},
            "generator_prompt": GENERATOR_TEMPLATE, "verifier_prompt": VERIFIER_TEMPLATE,
            "code_hash": digest.hexdigest(),
        })
        self.root = Path("runs") / self.run_hash[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.assets["generator"]["path"], use_fast=True)
        atomic_json(self.root / "config.json", {
            "hash": self.run_hash, "config": self.config,
            "assets": self.assets, "revisions": self.splits["revisions"],
        })
        atomic_json(self.root / "splits.json", self.splits)
        if self.config["verifier"].get("type") == "process_reward":
            for name in ("prm_dev_scores.json", "prm_dev_threshold.json", "prm_smoke.json"):
                value = read_json(Path(".cache") / name)
                if value is not None:
                    atomic_json(self.root / name, value)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "run_hash": self.run_hash,
            "stages": ["exp1", "exp2a", "exp2b_dev", "exp2b_eval", "exp3"],
            "counts": {"debug": 10, "development": 40, "evaluation": 100,
                       "diagnostic_raw": 60, "exp3_fail_max": 40},
            "planned_paths": 10 + 40 * 5 + 100 * 3 + 40 * 2,
            "machine_hours_first_round_excluding_evaluation": [1.8, 8.0],
            "disk_available_gib": round(__import__("shutil").disk_usage(self.root).free / 2**30, 2),
        }
        atomic_json(self.root / "run_manifest.json", manifest)

    def budget(self) -> Budget:
        raw = self.config["budget"]
        return Budget(generated_tokens=int(raw["generated_tokens"]),
                      verifier_calls=int(raw["verifier_calls"]),
                      verifier_output_tokens=int(raw["verifier_output_tokens"]),
                      prefill_tokens=int(raw["prefill_tokens"]),
                      context_tokens=int(self.config["context_limit"]))

    def prompt_ids(self, question: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": GENERATOR_TEMPLATE.format(question=question)}],
            tokenize=True, add_generation_prompt=True,
        )

    def state_path(self, stage: str, policy: str, request_id: str) -> Path:
        return self.root / "states" / stage / policy / f"{request_id}.json"

    def new_state(self, row: dict[str, str], policy: str) -> PathState:
        return PathState.create(row["id"], row["question"], row["gold"], policy,
                                request_seed(int(self.config["seed"]), row["id"]),
                                self.prompt_ids(row["question"]), self.budget(), self.run_hash)

    def load_states(self, stage: str, policies: list[str], rows: list[dict[str, str]]) -> dict[Path, PathState]:
        states = {}
        for policy in policies:
            for row in rows:
                path = self.state_path(stage, policy, row["id"])
                existing = read_json(path) if self.resume else None
                state = PathState.from_dict(existing) if existing else self.new_state(row, policy)
                states[path] = state
        return states

    @staticmethod
    def _save_state(path: Path, state: PathState) -> None:
        atomic_json(path, state.to_dict())

    def _generate_wave(self, states: dict[Path, PathState]) -> None:
        pending = [(path, state) for path, state in states.items() if state.status == "WAIT_GENERATION"]
        if not pending:
            return
        started = time.perf_counter()
        model = LanguageModel(self.assets["generator"]["path"])
        try:
            for path, state in pending:
                if not state.can_generate():
                    self._save_state(path, state)
                    continue
                needs_logical_prefill = not state.checks or state.checks[-1]["effective_verdict"] == "FAIL"
                input_tokens = len(state.context_ids) + len(state.pending_ids)
                if needs_logical_prefill and state.prefill_tokens + input_tokens > state.budget.prefill_tokens:
                    state.finish("BUDGET_PREFILL", state.candidate)
                    self._save_state(path, state)
                    continue
                try:
                    result = model.generate_path(state, self.config)
                except torch.cuda.OutOfMemoryError:
                    atomic_json(self.root / "oom.json", {"stage": str(path),
                                                          "allocated": torch.cuda.memory_allocated(),
                                                          "reserved": torch.cuda.memory_reserved()})
                    raise
                state.rng_state = result.rng_state
                state.accept_generated(result.kept_ids, result.pending_ids,
                                       result.reason, result.answer, input_tokens,
                                       result.prefill_seconds, result.decode_seconds,
                                       current_step=result.current_step)
                self._save_state(path, state)
        finally:
            model.close()
            self._append_phase("generator", len(pending), time.perf_counter() - started)

    def _append_phase(self, role: str, count: int, seconds: float) -> None:
        path = self.root / "phase_times.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": role, "count": count, "seconds": seconds}) + "\n")

    def _verify_wave(self, states: dict[Path, PathState]) -> None:
        pending = [(path, state) for path, state in states.items() if state.status == "WAIT_CHECK"]
        if not pending:
            return
        started = time.perf_counter()
        is_prm = self.config["verifier"].get("type") == "process_reward"
        model = (ProcessRewardModel(self.assets["verifier"]["path"],
                                    float(self.config["verifier"]["threshold"])) if is_prm else
                 LanguageModel(self.assets["verifier"]["path"], "nf4"))
        try:
            for path, state in pending:
                accepted = self.tokenizer.decode(state.checkpoint_ids[len(state.prompt_ids):],
                                                 skip_special_tokens=True)
                new = self.tokenizer.decode(state.context_ids[state.segment_start:],
                                            skip_special_tokens=True)
                base_prompt = verification_prompt(state.question, accepted, new)
                parsed = None
                format_error = False
                for attempt in range(2):
                    prompt = base_prompt if attempt == 0 else (
                        base_prompt + "\nYour previous response was malformed. Return a raw JSON object. "
                        "Do not use Markdown fences; diagnosis and hint must be strings.")
                    prompt_ids = (model.input_ids(state.question, accepted, new)[0] if is_prm else
                                  model.prompt_ids(prompt))
                    if state.verifier_calls >= state.budget.verifier_calls:
                        break
                    if len(prompt_ids) > state.budget.context_tokens or (
                        state.prefill_tokens + len(prompt_ids) > state.budget.prefill_tokens
                    ):
                        state.finish("BUDGET_PREFILL_OR_CONTEXT", state.candidate)
                        break
                    output_cap = min(state.budget.verifier_output_tokens,
                                     state.budget.context_tokens - len(prompt_ids))
                    if output_cap <= 0 and not is_prm:
                        state.finish("CONTEXT_LIMIT", state.candidate)
                        break
                    if is_prm:
                        raw, input_tokens, output_tokens, seconds = model.verify(
                            state.question, accepted, new)
                    else:
                        raw, input_tokens, output_tokens, seconds = model.generate_verification(
                            prompt, output_cap)
                    state.charge_verification(input_tokens, output_tokens, seconds, raw,
                                              retry=attempt > 0)
                    parsed = parse_verdict(raw)
                    if parsed is not None:
                        break
                    format_error = True
                    if is_prm:
                        break
                if state.finished:
                    self._save_state(path, state)
                    continue
                if parsed is None:
                    parsed = {"verdict": "UNCERTAIN", "first_error_step": None,
                              "diagnosis": "FORMAT_ERROR", "hint": ""}
                feedback = []
                if parsed["verdict"] == "FAIL":
                    feedback = self.tokenizer.encode(
                        feedback_text(parsed["diagnosis"], parsed["hint"], state.checkpoint_step),
                        add_special_tokens=False)
                state.apply_verdict(parsed["verdict"], parsed["first_error_step"],
                                    parsed["diagnosis"], parsed["hint"], feedback,
                                    state.current_step, format_error=format_error)
                self._save_state(path, state)
        finally:
            model.close()
            self._append_phase("verifier", len(pending), time.perf_counter() - started)

    def run_paths(self, states: dict[Path, PathState]) -> dict[Path, PathState]:
        for _ in range(50):
            if all(state.finished for state in states.values()):
                break
            before = sum(state.finished for state in states.values())
            self._generate_wave(states)
            self._verify_wave(states)
            after = sum(state.finished for state in states.values())
            print(f"Paths: {after}/{len(states)} finished (previous {before})", flush=True)
        else:
            raise RuntimeError("More than 50 execution waves; inspect paths for non-termination")
        return states

    def _collect_logs(self, stage: str, states: dict[Path, PathState]) -> None:
        stage_dir = self.root / stage
        stage_dir.mkdir(exist_ok=True)
        files = {name: (stage_dir / f"{name}.jsonl").open("w", encoding="utf-8")
                 for name in ("events", "traces", "checkpoints", "verifications")}
        try:
            for state in states.values():
                for event in state.events:
                    files["events"].write(json.dumps(event, ensure_ascii=False) + "\n")
                for checkpoint in state.checkpoints:
                    files["checkpoints"].write(json.dumps({"request_id": state.request_id,
                                                             "policy": state.policy, **checkpoint},
                                                            ensure_ascii=False) + "\n")
                for check in state.checks:
                    files["verifications"].write(json.dumps({"request_id": state.request_id,
                                                               "policy": state.policy, **check},
                                                              ensure_ascii=False) + "\n")
                trace = {"request_id": state.request_id, "policy": state.policy,
                         "question": state.question, "gold": state.gold,
                         "final_answer": state.final_answer,
                         "correct": is_correct(state.final_answer, state.gold),
                         "termination": state.termination,
                         "context_ids": state.context_ids,
                         "generated_tokens": state.generated_tokens,
                         "verifier_calls": state.verifier_calls,
                         "prefill_tokens": state.prefill_tokens,
                         "generator_seconds": state.generator_seconds,
                         "verifier_seconds": state.verifier_seconds,
                         "offline_restore_seconds": state.offline_restore_seconds}
                files["traces"].write(json.dumps(trace, ensure_ascii=False) + "\n")
        finally:
            for handle in files.values():
                handle.close()

    def _summary(self, stage: str, states: dict[Path, PathState]) -> dict[str, Any]:
        self._collect_logs(stage, states)
        by_policy: dict[str, list[PathState]] = {}
        for state in states.values():
            by_policy.setdefault(state.policy, []).append(state)
        summary = {}
        for policy, group in by_policy.items():
            n = len(group)
            summary[policy] = {
                "count": n,
                "correct": sum(is_correct(s.final_answer, s.gold) for s in group),
                "accuracy": sum(is_correct(s.final_answer, s.gold) for s in group) / n,
                "unfinished": sum(s.final_answer is None for s in group),
                "checks": sum(s.verifier_calls for s in group),
                "failures": sum(c["effective_verdict"] == "FAIL" for s in group for c in s.checks),
                "generator_seconds": sum(s.generator_seconds for s in group),
                "verifier_seconds": sum(s.verifier_seconds for s in group),
                "offline_restore_seconds": sum(s.offline_restore_seconds for s in group),
                "termination_counts": {reason: sum(s.termination == reason for s in group)
                                       for reason in sorted({s.termination for s in group})},
            }
        atomic_json(self.root / stage / "summary.json", summary)
        with (self.root / stage / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["policy", "count", "correct", "accuracy", "unfinished",
                             "checks", "failures", "generator_seconds", "verifier_seconds"])
            for policy, values in summary.items():
                writer.writerow([policy] + [values[key] for key in (
                    "count", "correct", "accuracy", "unfinished", "checks", "failures",
                    "generator_seconds", "verifier_seconds")])
        return summary

    def probe_hardware(self) -> dict[str, Any]:
        path = self.root / "hardware_profile.json"
        existing = read_json(path) if self.resume else None
        if existing is not None:
            return existing
        profile: dict[str, Any] = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "config_hash": self.run_hash,
            "host_memory_total_bytes": psutil.virtual_memory().total,
            "host_memory_available_bytes": psutil.virtual_memory().available,
            "roles": {},
        }
        for role in ("generator", "verifier"):
            started = time.perf_counter()
            is_prm = role == "verifier" and self.config["verifier"].get("type") == "process_reward"
            model = (ProcessRewardModel(self.assets[role]["path"],
                                        float(self.config["verifier"]["threshold"])) if is_prm else
                     LanguageModel(self.assets[role]["path"],
                                   "nf4" if role == "verifier" else None))
            role_profile: dict[str, Any] = {"load_seconds": time.perf_counter() - started,
                                            "lengths": {}}
            try:
                if is_prm:
                    filler = model.tokenizer.encode(" calculate carefully and continue.\n",
                                                    add_special_tokens=False)
                    initial = model.input_ids("Profile arithmetic reasoning length and memory.",
                                              "", "Step 1: calculate carefully.")[0]
                else:
                    filler = model.text_ids(" Step 1: calculate carefully and continue.\n")
                    initial = model.prompt_ids("Profile arithmetic reasoning length and memory.")
                for length in (512, 1024, 2048, 3072):
                    ids = (initial + filler * ((length - len(initial)) // len(filler) + 1))[:length]
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        if is_prm:
                            prefill_seconds = model.profile(ids)
                            started_decode = time.perf_counter()
                        else:
                            cache, logits, prefill_seconds = model._prefill(ids)
                            started_decode = time.perf_counter()
                            for _ in range(8):
                                token = int(torch.argmax(logits, dim=-1).item())
                                logits = model._next_logits(token, cache)
                        torch.cuda.synchronize()
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                        headroom_ok = free_bytes >= float(self.config["gpu_headroom_gib"]) * 2**30
                        role_profile["lengths"][str(length)] = {
                            "status": "PASS" if headroom_ok else "LOW_HEADROOM",
                            "prefill_seconds": prefill_seconds,
                            "decode_seconds": time.perf_counter() - started_decode,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                            "free_bytes_after_decode": free_bytes,
                            "total_bytes_reported": total_bytes,
                        }
                        if not is_prm:
                            del cache, logits
                    except torch.cuda.OutOfMemoryError as exc:
                        role_profile["lengths"][str(length)] = {"status": "OOM", "error": str(exc),
                                                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                        gc.collect()
                        torch.cuda.empty_cache()
                    atomic_json(path, {**profile, "roles": {**profile["roles"], role: role_profile}})
            finally:
                model.close()
            profile["roles"][role] = role_profile
            atomic_json(path, profile)
        for role in ("generator", "verifier"):
            if profile["roles"][role]["lengths"][str(self.config["context_limit"])]["status"] != "PASS":
                raise RuntimeError(f"{role} failed capacity test at configured context length")
        return profile

    def run_experiment1(self) -> dict[str, Any]:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        checks = {"passed": result.returncode == 0, "output": result.stdout}
        atomic_json(self.root / "execution_checks.json", checks)
        if not checks["passed"]:
            raise RuntimeError("Deterministic execution checks failed")
        self.probe_hardware()
        states = self.load_states("exp1", ["k2"], self.splits["debug"])
        self.run_paths(states)
        summary = self._summary("exp1", states)
        mean_service = (summary["k2"]["generator_seconds"] +
                        summary["k2"]["verifier_seconds"]) / len(states)
        atomic_json(self.root / "time_estimate.json", {
            "observed_debug_paths": len(states), "mean_path_service_seconds": mean_service,
            "development_200_paths_service_hours_placeholder": mean_service * 200 / 3600,
            "evaluation_300_paths_service_hours_placeholder": mean_service * 300 / 3600,
            "note": "Rough estimate; excludes model switches and path-length variation",
        })
        report = ["# 实验一：执行器、检查点与预算", "", "状态：PASS（工程检查）", "",
                  f"确定性测试：{result.stdout.strip()}", "",
                  f"真实题：{summary['k2']['count']}，答对 {summary['k2']['correct']}，"
                  f"未完成 {summary['k2']['unfinished']}。", "",
                  "该结果只检验执行链与粗略资源需求，不作为质量收益结论。"]
        (self.root / "exp1" / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return summary

    def _diagnostic_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for row in self.splits["diagnostic"]:
            label = row["label"]
            steps = row["steps"]
            if label == -1:
                valid = steps[:min(2, len(steps))]
                jobs.append({"id": row["id"] + "-valid", "case_id": row["id"],
                             "partition": row["partition"], "kind": "valid",
                             "question": row["problem"], "accepted": "",
                             "new": "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(valid)),
                             "expected_step": None})
            else:
                accepted = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps[:label]))
                if label > 0:
                    jobs.append({"id": row["id"] + "-valid", "case_id": row["id"],
                                 "partition": row["partition"], "kind": "valid",
                                 "question": row["problem"], "accepted": "",
                                 "new": accepted, "expected_step": None})
                jobs.append({"id": row["id"] + "-error", "case_id": row["id"],
                             "partition": row["partition"], "kind": "error",
                             "question": row["problem"], "accepted": accepted,
                             "new": f"Step {label+1}: {steps[label]}",
                             "expected_step": label + 1})
        return jobs

    def run_diagnostic(self) -> dict[str, Any]:
        directory = self.root / "exp2a"
        directory.mkdir(exist_ok=True)
        jobs = self._diagnostic_jobs()
        missing = [job for job in jobs if not (directory / f"{job['id']}.json").exists() or not self.resume]
        if missing:
            started = time.perf_counter()
            is_prm = self.config["verifier"].get("type") == "process_reward"
            model = (ProcessRewardModel(self.assets["verifier"]["path"],
                                        float(self.config["verifier"]["threshold"])) if is_prm else
                     LanguageModel(self.assets["verifier"]["path"], "nf4"))
            try:
                for index, job in enumerate(missing, 1):
                    prompt = verification_prompt(job["question"], job["accepted"], job["new"])
                    attempts = []
                    parsed = None
                    for attempt in range(2):
                        actual = prompt if attempt == 0 else (
                            prompt + "\nYour previous response was malformed. Return a raw JSON object "
                            "without Markdown fences; diagnosis and hint must be strings.")
                        input_ids = (model.input_ids(job["question"], job["accepted"], job["new"])[0]
                                     if is_prm else model.prompt_ids(actual))
                        if len(input_ids) > int(self.config["context_limit"]):
                            attempts.append({"error": "CONTEXT_LIMIT"})
                            break
                        output_cap = min(int(self.config["budget"]["verifier_output_tokens"]),
                                         int(self.config["context_limit"]) - len(input_ids))
                        if output_cap <= 0 and not is_prm:
                            attempts.append({"error": "CONTEXT_LIMIT"})
                            break
                        if is_prm:
                            raw, input_tokens, output_tokens, seconds = model.verify(
                                job["question"], job["accepted"], job["new"])
                        else:
                            raw, input_tokens, output_tokens, seconds = model.generate_verification(
                                actual, output_cap)
                        attempts.append({"raw": raw, "input_tokens": input_tokens,
                                         "output_tokens": output_tokens, "seconds": seconds})
                        parsed = parse_verdict(raw)
                        if parsed is not None:
                            break
                        if is_prm:
                            break
                    result = {"job": job, "attempts": attempts, "parsed": parsed,
                              "json_valid": parsed is not None}
                    atomic_json(directory / f"{job['id']}.json", result)
                    if index % 10 == 0:
                        print(f"Diagnostic: {index}/{len(missing)}", flush=True)
            finally:
                model.close()
                self._append_phase("diagnostic_verifier", len(missing), time.perf_counter() - started)
        results = [read_json(directory / f"{job['id']}.json") for job in jobs]
        metrics: dict[str, Any] = {}
        for partition in ("dev", "holdout", "all"):
            rows = [r for r in results if partition == "all" or r["job"]["partition"] == partition]
            valid = [r for r in rows if r["job"]["kind"] == "valid"]
            error = [r for r in rows if r["job"]["kind"] == "error"]
            metrics[partition] = {
                "count": len(rows), "valid_count": len(valid), "error_count": len(error),
                "json_valid_rate": sum(r["json_valid"] for r in rows) / len(rows),
                "false_positive_rate": sum(r["parsed"] is not None and
                                           r["parsed"]["verdict"] == "FAIL" for r in valid) / len(valid),
                "error_detection_rate": sum(r["parsed"] is not None and
                                            r["parsed"]["verdict"] == "FAIL" and
                                            r["parsed"]["first_error_step"] == r["job"]["expected_step"]
                                            for r in error) / len(error),
                "uncertain_rate": sum(r["parsed"] is not None and
                                      r["parsed"]["verdict"] == "UNCERTAIN" for r in rows) / len(rows),
            }
        threshold = self.config["diagnostic"]
        h = metrics["holdout"]
        gate = (h["json_valid_rate"] >= threshold["min_json_valid"] and
                h["false_positive_rate"] <= threshold["max_false_positive"] and
                h["error_detection_rate"] >= threshold["min_error_detection"])
        report = {"metrics": metrics, "gate_passed": gate, "thresholds": threshold,
                  "verifier_output_kind": ("deterministic_adapter_from_process_reward"
                                           if self.config["verifier"].get("type") == "process_reward"
                                           else "model_generated_json")}
        atomic_json(directory / "summary.json", report)
        (directory / "report.md").write_text(
            "# 实验二 A：验证器诊断\n\n" +
            f"工程状态：{'PASS' if gate else 'BLOCKED'}\n\n" +
            "```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n",
            encoding="utf-8")
        return report

    def run_experiment2(self) -> dict[str, Any]:
        diagnostic = self.run_diagnostic()
        if not diagnostic["gate_passed"]:
            (self.root / "exp2_report.md").write_text(
                "# 实验二\n\n工程状态：BLOCKED。留出诊断集未达到预注册门槛；"
                "未运行依赖的反馈闭环和正式评估。详见 exp2a/report.md。\n",
                encoding="utf-8")
            return {"status": "BLOCKED", "diagnostic": diagnostic}
        development = self.load_states("exp2b_dev", ["unchecked", "endpoint", "k2", "k1", "k4"],
                                       self.splits["development"])
        self.run_paths(development)
        dev_summary = self._summary("exp2b_dev", development)
        atomic_json(self.root / "frozen_evaluation.json", {
            "run_hash": self.run_hash, "main_fixed_interval": 2,
            "evaluation_ids": [row["id"] for row in self.splits["evaluation"]],
            "policies": ["unchecked", "endpoint", "k2"],
        })
        evaluation = self.load_states("exp2b_eval", ["unchecked", "endpoint", "k2"],
                                      self.splits["evaluation"])
        self.run_paths(evaluation)
        eval_summary = self._summary("exp2b_eval", evaluation)
        by_policy = {(state.policy, state.request_id): state for state in evaluation.values()}
        pairs = []
        examples = {"improved": [], "worsened": [], "unchanged": []}
        for row in self.splits["evaluation"]:
            qid = row["id"]
            base = by_policy[("unchecked", qid)]
            checked = by_policy[("k2", qid)]
            base_ok = is_correct(base.final_answer, base.gold)
            checked_ok = is_correct(checked.final_answer, checked.gold)
            pairs.append(int(checked_ok) - int(base_ok))
            category = "improved" if checked_ok and not base_ok else (
                "worsened" if base_ok and not checked_ok else "unchanged")
            if len(examples[category]) < 5:
                examples[category].append({
                    "question_id": qid, "question": row["question"], "gold": row["gold"],
                    "unchecked_answer": base.final_answer, "checked_answer": checked.final_answer,
                    "checked_context": self.tokenizer.decode(checked.context_ids[len(checked.prompt_ids):],
                                                              skip_special_tokens=True),
                    "checks": checked.checks,
                })
        rng = random.Random(int(self.config["seed"]))
        bootstrap = [sum(pairs[rng.randrange(len(pairs))] for _ in pairs) / len(pairs)
                     for _ in range(2000)]
        bootstrap.sort()
        paired = {
            "k2_minus_unchecked_accuracy": sum(pairs) / len(pairs),
            "bootstrap_95_interval": [bootstrap[50], bootstrap[1949]],
            "wrong_to_right": sum(value == 1 for value in pairs),
            "right_to_wrong": sum(value == -1 for value in pairs),
            "question_count": len(pairs),
        }
        atomic_json(self.root / "exp2b_eval" / "paired.json", paired)
        atomic_json(self.root / "exp2b_eval" / "examples.json", examples)
        report = {"status": "PASS", "diagnostic": diagnostic,
                  "development": dev_summary, "evaluation": eval_summary,
                  "paired": paired}
        atomic_json(self.root / "exp2_summary.json", report)
        lines = ["# 实验二：验证器与反馈", "", "工程状态：PASS", "",
                 "| 集合 | 策略 | 答对/总数 | 检查次数 | 未完成 |", "|---|---|---:|---:|---:|"]
        for label, summary in (("开发", dev_summary), ("评估", eval_summary)):
            for policy, values in summary.items():
                lines.append(f"| {label} | {policy} | {values['correct']}/{values['count']} | "
                             f"{values['checks']} | {values['unfinished']} |")
        lines += ["", f"固定间隔相对不检查：错改对 {paired['wrong_to_right']}，"
                  f"对改错 {paired['right_to_wrong']}；准确率差 {paired['k2_minus_unchecked_accuracy']:.3f}，"
                  f"按题目重采样的 95% 区间 {paired['bootstrap_95_interval']}。", "",
                  "质量结论需结合配对差异、失败类型与成本表解释；工程 PASS 不表示收益为正。"]
        (self.root / "exp2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def run_experiment3(self) -> dict[str, Any]:
        from .exp3 import run_experiment3

        return run_experiment3(self)
