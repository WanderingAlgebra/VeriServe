from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .common import stable_hash


Verdict = Literal["PASS", "FAIL", "UNCERTAIN"]


@dataclass
class Budget:
    generated_tokens: int = 2048
    verifier_calls: int = 6
    verifier_output_tokens: int = 256
    prefill_tokens: int = 16384
    context_tokens: int = 2048


@dataclass
class PathState:
    request_id: str
    question: str
    gold: str
    policy: str
    seed: int
    config_hash: str
    prompt_ids: list[int]
    context_ids: list[int]
    checkpoint_ids: list[int]
    budget: Budget
    status: str = "WAIT_GENERATION"
    checkpoint_id: str = "initial"
    history_id: str = "root"
    checkpoint_step: int = 0
    current_step: int = 0
    segment_start: int = 0
    marker_baseline: int = 0
    pending_ids: list[int] = field(default_factory=list)
    candidate: str | None = None
    final_answer: str | None = None
    termination: str | None = None
    generated_tokens: int = 0
    verifier_calls: int = 0
    verifier_output_tokens: int = 0
    prefill_tokens: int = 0
    generator_seconds: float = 0.0
    verifier_seconds: float = 0.0
    offline_restore_seconds: float = 0.0
    feedback_prefill_seconds: float = 0.0
    rng_state: list[int] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    fail_snapshots: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, request_id: str, question: str, gold: str, policy: str,
               seed: int, prompt_ids: list[int], budget: Budget,
               config_hash: str = "test") -> "PathState":
        return cls(request_id=request_id, question=question, gold=gold,
                   policy=policy, seed=seed, config_hash=config_hash,
                   prompt_ids=prompt_ids.copy(),
                   context_ids=prompt_ids.copy(), checkpoint_ids=prompt_ids.copy(),
                   budget=copy.deepcopy(budget), segment_start=len(prompt_ids))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PathState":
        payload = dict(value)
        payload["budget"] = Budget(**payload["budget"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.status == "FINISHED"

    def record(self, name: str, **values: Any) -> None:
        action_id = len(self.events)
        self.events.append({"event": name, "request_id": self.request_id,
                            "policy": self.policy, "history_id": self.history_id,
                            "action_id": action_id,
                            "task_key": stable_hash((self.config_hash, self.request_id,
                                                     self.history_id, action_id, self.seed)),
                            **values})

    def finish(self, reason: str, answer: str | None = None) -> None:
        self.final_answer = answer
        self.termination = reason
        self.status = "FINISHED"
        self.record("terminate", reason=reason, answer=answer)

    def can_generate(self) -> bool:
        if self.generated_tokens >= self.budget.generated_tokens:
            self.finish("BUDGET_GENERATION", self.candidate)
            return False
        if len(self.context_ids) + len(self.pending_ids) >= self.budget.context_tokens:
            self.finish("CONTEXT_LIMIT", self.candidate)
            return False
        if self.policy != "unchecked" and self.verifier_calls >= self.budget.verifier_calls:
            self.finish("BUDGET_VERIFIER", self.candidate)
            return False
        return True

    def merge_pending(self) -> None:
        if self.pending_ids:
            self.context_ids.extend(self.pending_ids)
            self.record("pending_merged", tokens=len(self.pending_ids))
            self.pending_ids = []

    def accept_generated(self, kept_ids: list[int], pending_ids: list[int],
                         reason: str, answer: str | None, prefill_tokens: int,
                         prefill_seconds: float, decode_seconds: float,
                         current_step: int = 0) -> None:
        if self.status != "WAIT_GENERATION":
            raise RuntimeError("Generate called outside generation state")
        charged = len(kept_ids) + len(pending_ids)
        if self.generated_tokens + charged > self.budget.generated_tokens:
            raise ValueError("Generation over budget")
        self.generated_tokens += charged
        self.current_step = current_step
        self.context_ids.extend(kept_ids)
        self.pending_ids = pending_ids
        self.generator_seconds += prefill_seconds + decode_seconds
        if len(self.context_ids) - len(self.prompt_ids) == len(kept_ids):
            self.prefill_tokens += prefill_tokens
            prefill_kind = "INITIAL"
        elif self.checks and self.checks[-1]["effective_verdict"] == "FAIL":
            self.prefill_tokens += prefill_tokens
            self.feedback_prefill_seconds += prefill_seconds
            prefill_kind = "FAIL_RESTORE"
        else:
            self.offline_restore_seconds += prefill_seconds
            prefill_kind = "OFFLINE_RESTORE"
        self.record("generate", kept_tokens=len(kept_ids), pending_tokens=len(pending_ids),
                    reason=reason, prefill_tokens=prefill_tokens, prefill_kind=prefill_kind,
                    prefill_seconds=prefill_seconds, decode_seconds=decode_seconds)
        if reason == "FINAL":
            self.candidate = answer
            if self.policy == "unchecked":
                self.finish("FINAL_UNCHECKED", answer)
            else:
                self.status = "WAIT_CHECK"
        elif reason == "STEP":
            if self.policy == "unchecked":
                raise RuntimeError("Unchecked path should never stop at a step")
            self.status = "WAIT_CHECK"
        elif reason in {"GEN_BUDGET", "CONTEXT_LIMIT", "EOS", "FORMAT_ERROR"}:
            self.finish(reason, self.candidate)
        else:
            raise ValueError(reason)

    def charge_verification(self, input_tokens: int, output_tokens: int,
                            seconds: float, raw: str, retry: bool = False) -> None:
        if self.status != "WAIT_CHECK":
            raise RuntimeError("Verify called outside check state")
        if self.verifier_calls >= self.budget.verifier_calls:
            raise ValueError("Verifier calls exhausted")
        if self.prefill_tokens + input_tokens > self.budget.prefill_tokens:
            raise ValueError("Prefill budget exhausted")
        self.verifier_calls += 1
        self.prefill_tokens += input_tokens
        self.verifier_output_tokens += output_tokens
        self.verifier_seconds += seconds
        self.record("verify_call", input_tokens=input_tokens, output_tokens=output_tokens,
                    seconds=seconds, retry=retry, raw=raw)

    def apply_verdict(self, verdict: Verdict, first_error_step: int | None,
                      diagnosis: str, hint: str, feedback_ids: list[int],
                      current_step: int, format_error: bool = False) -> str:
        if self.status != "WAIT_CHECK":
            raise RuntimeError("Verdict outside check state")
        effective = verdict
        conflict = verdict == "FAIL" and first_error_step is not None and first_error_step <= self.checkpoint_step
        if conflict:
            effective = "UNCERTAIN"
        check = {
            "raw_verdict": verdict, "effective_verdict": effective,
            "first_error_step": first_error_step, "diagnosis": diagnosis,
            "hint": hint, "checkpoint_conflict": conflict,
            "format_error": format_error, "candidate": self.candidate,
            "checkpoint_id_before": self.checkpoint_id,
        }
        if effective == "FAIL":
            snapshot_state = self.to_dict()
            snapshot_state["fail_snapshots"] = []
            for old_check in snapshot_state["checks"]:
                old_check.pop("snapshot", None)
            snapshot = {
                "state": snapshot_state, "diagnosis": diagnosis, "hint": hint,
                "first_error_step": first_error_step,
            }
            self.fail_snapshots.append(snapshot)
            self.context_ids = self.checkpoint_ids.copy() + feedback_ids
            self.pending_ids = []
            self.segment_start = len(self.context_ids)
            self.candidate = None
            self.current_step = self.checkpoint_step
            self.history_id = stable_hash((self.history_id, self.verifier_calls, len(self.fail_snapshots)))[:16]
            self.record("rollback", checkpoint_id=self.checkpoint_id,
                        checkpoint_tokens=len(self.checkpoint_ids),
                        feedback_tokens=len(feedback_ids))
        elif effective == "PASS":
            self.checkpoint_ids = self.context_ids.copy()
            self.checkpoint_step = current_step
            self.checkpoint_id = stable_hash((self.history_id, len(self.checks), self.checkpoint_ids))[:16]
            self.segment_start = len(self.context_ids)
            self.checkpoints.append({"checkpoint_id": self.checkpoint_id,
                                     "history_id": self.history_id,
                                     "token_count": len(self.checkpoint_ids),
                                     "token_hash": stable_hash(self.checkpoint_ids),
                                     "step": self.checkpoint_step,
                                     "generated_tokens": self.generated_tokens,
                                     "verifier_calls": self.verifier_calls})
            self.record("checkpoint", checkpoint_id=self.checkpoint_id,
                        tokens=len(self.checkpoint_ids), step=current_step)
        self.checks.append(check)
        self.marker_baseline = 0
        self.record("verdict", verdict=verdict, effective=effective,
                    conflict=conflict, format_error=format_error)
        if self.candidate is not None and effective in {"PASS", "UNCERTAIN"}:
            self.finish("FINAL_PASS" if effective == "PASS" else "FINAL_UNCERTAIN", self.candidate)
        elif self.verifier_calls >= self.budget.verifier_calls:
            self.finish("BUDGET_VERIFIER", None)
        else:
            self.status = "WAIT_GENERATION"
        return effective
