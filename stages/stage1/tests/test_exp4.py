"""Checks for the new intervention semantics and independent-seed evaluation."""
import copy
from types import SimpleNamespace

from scripts.run_exp4 import (Experiment4, accept_generation, analyze_oracle,
                             branch_state, geometric_interval, logical_prefill_needed)
from veriserve.state import Budget, PathState


def snapshot():
    state = PathState.create("q", "question", "2", "history_initial", 11, [10, 11],
                             Budget(generated_tokens=50, prefill_tokens=100, context_tokens=100))
    state.accept_generated([20, 21], [22], "STEP", None, 2, .1, .2, current_step=1)
    return {"id": "initial/q", "group": "initial", "question_id": "q", "state": state.to_dict()}


def test_actions_share_history_budget_seed_and_change_only_first_decision():
    source = snapshot()
    untouched = copy.deepcopy(source)
    run = object.__new__(Experiment4)
    run.fixed = "k4"
    run.settings = {"actions": ["now", "delay1", "delay2", "endpoint_only"]}
    states = [branch_state(source, a, 17, "new", 20260922) for a in run.settings["actions"]]
    assert source == untouched
    assert len({s.seed for s in states}) == 1
    assert states[0].status == "WAIT_CHECK"
    for state in states:
        assert state.context_ids == source["state"]["context_ids"]
        assert state.pending_ids == [22]
        assert state.generated_tokens == 3 and state.prefill_tokens == 2
        assert not logical_prefill_needed(state)
    assert run.effective_policy(states[1]) == "k1"
    assert run.effective_policy(states[2]) == "k2"
    assert run.effective_policy(states[3]) == "endpoint"
    for state in states:
        state.checks.append({"effective_verdict": "PASS"})
    assert [run.effective_policy(s) for s in states] == ["k4", "k4", "k4", "endpoint"]
    restored = PathState.from_dict(states[1].to_dict())
    assert run.effective_policy(restored) == "k4"
    assert branch_state(source, "now", 43, "new", 20260922).seed != states[0].seed


def test_delayed_snapshot_restore_does_not_charge_a_second_fail_prefill():
    state = PathState.from_dict(snapshot()["state"])
    state.charge_verification(3, 0, .1, "{}")
    state.apply_verdict("FAIL", 1, "wrong", "retry", [99], 1)
    assert logical_prefill_needed(state)
    result = SimpleNamespace(kept_ids=[30, 31], pending_ids=[32], reason="STEP", answer=None,
                             prefill_seconds=.2, decode_seconds=.3, current_step=1, rng_state=[1])
    accept_generation(state, result, 3, logical=True)
    paid = state.prefill_tokens
    state.status = "WAIT_GENERATION"  # delay the pending check without a new verdict
    state.merge_pending()
    assert not logical_prefill_needed(state)
    accept_generation(state, result, len(state.context_ids), logical=False)
    assert state.prefill_tokens == paid
    assert state.feedback_prefill_seconds == .2
    assert state.offline_restore_seconds == .2
    assert state.generated_tokens == 9


def test_random_interval_is_resumable_and_does_not_consume_generator_rng():
    run = object.__new__(Experiment4)
    run.probability = .25
    state = PathState.from_dict(snapshot()["state"])
    state.policy = "random"
    state.rng_state = [5, 6]
    before = copy.deepcopy(state.rng_state)
    first = run.effective_policy(state)
    assert run.effective_policy(PathState.from_dict(state.to_dict())) == first
    assert state.rng_state == before
    assert geometric_interval(1, 123) == 1


def test_oracle_selection_does_not_read_held_seed_answers():
    settings = {"actions": ["now", "delay1", "delay2", "endpoint_only"],
                "snapshot_groups": ["initial"], "selection_seeds": [17, 29],
                "evaluation_seed": 43, "cost_weights": [0], "seed": 9}
    rows = [{"snapshot_id": "initial/q", "group": "initial", "action": action,
             "replicate": seed, "correct": action == ("delay1" if seed != 43 else "now"),
             "suffix_service_seconds": 1, "unfinished": False}
            for action in settings["actions"] for seed in [17, 29, 43]]
    result = analyze_oracle(rows, settings, 1)
    assert result["selections"][0]["chosen_action"] == "delay1"
    assert result["comparisons"][0]["oracle_correct"] == 0


def test_history_snapshot_requires_natural_pass_and_excludes_early_final(tmp_path):
    run = object.__new__(Experiment4)
    run.root = tmp_path
    state = PathState.from_dict(snapshot()["state"])
    state.policy = "history_historical_pass"
    run.capture_snapshot(state)
    assert state.status == "WAIT_CHECK"  # A step without a PASS is not a historical checkpoint.
    state.charge_verification(3, 0, .1, "{}")
    state.apply_verdict("PASS", None, "", "", [], 1)
    state.merge_pending()
    state.accept_generated([30], [31], "STEP", None, len(state.context_ids), .1, .2, current_step=2)
    run.capture_snapshot(state)
    assert state.status == "SNAPSHOT_READY"
    assert (tmp_path / "snapshots/historical_pass/q.json").exists()
    terminal = PathState.from_dict(snapshot()["state"])
    terminal.policy = "history_initial"
    terminal.candidate = "2"
    run.capture_snapshot(terminal)
    assert terminal.termination == "HISTORY_FINAL_BEFORE_SNAPSHOT"
    assert not (tmp_path / "snapshots/initial/q.json").exists()
