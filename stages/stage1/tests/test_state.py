import copy

import pytest

from veriserve.state import Budget, PathState


def state(policy="k2", calls=6):
    return PathState.create("q1", "1+1?", "2", policy, 7, [10, 11],
                            Budget(generated_tokens=50, verifier_calls=calls,
                                   prefill_tokens=200, context_tokens=100))


def generated(path, ids, final=False):
    path.accept_generated(ids, [], "FINAL" if final else "STEP", "2" if final else None,
                          len(path.context_ids), 0.01, 0.02)


def checked(path, verdict, step=None, feedback=None, current_step=1):
    path.charge_verification(3, 2, 0.01, "{}")
    return path.apply_verdict(verdict, step, "wrong", "retry",
                              feedback if feedback is not None else [99], current_step)


def test_first_fail_returns_to_exact_prompt_and_keeps_cost():
    path = state()
    generated(path, [1, 2, 3])
    checked(path, "FAIL", 1)
    assert path.context_ids == [10, 11, 99]
    assert path.checkpoint_ids == [10, 11]
    assert path.generated_tokens == 3
    assert path.verifier_calls == 1


def test_pass_then_fail_uses_historical_checkpoint():
    path = state()
    generated(path, [1, 2])
    checked(path, "PASS", current_step=2)
    accepted = copy.deepcopy(path.checkpoint_ids)
    generated(path, [3, 4, 5])
    checked(path, "FAIL", step=4, current_step=4)
    assert path.context_ids == accepted + [99]
    assert path.checkpoint_ids == accepted
    assert path.checkpoint_step == 2
    assert path.generated_tokens == 5


def test_repeated_fail_does_not_advance_checkpoint():
    path = state()
    for _ in range(2):
        generated(path, [1, 2])
        checked(path, "FAIL", 1)
    assert path.checkpoint_ids == [10, 11]
    assert path.verifier_calls == 2
    assert path.generated_tokens == 4
    assert len(path.fail_snapshots) == 2


def test_uncertain_and_checkpoint_conflict_do_not_advance():
    path = state()
    generated(path, [1])
    checked(path, "PASS", current_step=2)
    accepted = path.checkpoint_ids.copy()
    generated(path, [2])
    effective = checked(path, "FAIL", 1, current_step=3)
    assert effective == "UNCERTAIN"
    assert path.checkpoint_ids == accepted
    assert path.context_ids == accepted + [2]
    assert path.checks[-1]["checkpoint_conflict"]
    generated(path, [3])
    checked(path, "UNCERTAIN", current_step=4)
    assert path.checkpoint_ids == accepted


def test_final_fail_revokes_candidate():
    path = state()
    generated(path, [1, 2], final=True)
    checked(path, "FAIL", 1)
    assert path.candidate is None
    assert path.final_answer is None
    assert path.status == "WAIT_GENERATION"


def test_final_pass_and_uncertain_have_distinct_reasons():
    for verdict, reason in [("PASS", "FINAL_PASS"), ("UNCERTAIN", "FINAL_UNCERTAIN")]:
        path = state()
        generated(path, [1], final=True)
        checked(path, verdict)
        assert path.termination == reason
        assert path.final_answer == "2"


def test_verifier_budget_exhaustion_and_prefill_rejection():
    path = state(calls=1)
    generated(path, [1])
    checked(path, "FAIL", 1)
    assert path.termination == "BUDGET_VERIFIER"
    path = state()
    generated(path, [1])
    with pytest.raises(ValueError, match="Prefill budget"):
        path.charge_verification(1000, 1, 0.0, "")
    assert path.verifier_calls == 0


def test_two_requests_and_resume_are_isolated():
    left, right = state(), state()
    left.request_id = "left"
    right.request_id = "right"
    generated(left, [1])
    generated(right, [7])
    checked(left, "FAIL", 1)
    restored = PathState.from_dict(left.to_dict())
    assert restored.context_ids == left.context_ids
    assert restored.generated_tokens == left.generated_tokens
    assert right.context_ids == [10, 11, 7]
    assert right.verifier_calls == 0


def test_pending_tokens_count_once():
    path = state()
    path.accept_generated([1, 2], [3], "STEP", None, 2, 0.0, 0.0)
    checked(path, "PASS", current_step=1)
    path.merge_pending()
    assert path.context_ids == [10, 11, 1, 2, 3]
    assert path.generated_tokens == 3
    assert path.checkpoint_ids == [10, 11, 1, 2]
