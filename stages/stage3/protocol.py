"""Small CPU-only protocol helpers; original generated token IDs stay authoritative."""
from __future__ import annotations

import math
import re

from stages.stage2.run_probe import FINAL, STEP, boundaries, boxed, decode


def boundary(tokenizer, generated_ids, target_step=2):
    """Find a *closed* ordinary step without generating its following trajectory.

    The next complete marker is lookahead. Tokens crossing the content boundary
    belong entirely to pending, exactly as in stage2.aligned_end.
    """
    text = decode(tokenizer, generated_ids)
    matches = list(STEP.finditer(text))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("FORMAT_ERROR: missing, repeated or non-sequential Step N markers")
    target = next((i for i, number in enumerate(numbers) if number == target_step), None)
    if target is None:
        return None
    final = FINAL.search(text)
    if final is not None and final.start() < matches[target].start():
        raise ValueError("FORMAT_ERROR: step after Final answer")
    next_marker = matches[target + 1] if target + 1 < len(matches) else None
    final_closes = final is not None and final.start() > matches[target].start()
    if next_marker is None and not final_closes:
        return None  # Current content, trailing whitespace and partial headers are not closure.
    _, parsed = boundaries(tokenizer, generated_ids)
    ordinary = next((step for step in parsed["steps"] if step["step_number"] == target_step), None)
    excluded = [step for step in parsed["excluded_steps"]
                if (step["step_number"] == 2 if target_step == 2 else 2 < step["step_number"] <= target_step)]
    if excluded:
        if target_step == 2:
            raise ValueError("NO_ELIGIBLE_ANCHOR: EXPLICIT_FINAL_OR_BOXED in Step 2")
        return None  # DELAY reaches the endpoint if two more ordinary steps never exist.
    if ordinary is None:
        raise ValueError("NO_ELIGIBLE_ANCHOR: " + "; ".join(parsed["format_errors"]))
    # A pending empty Step N is expected; errors in the accepted part are not.
    for error in parsed["format_errors"]:
        if "empty Step" in error and next_marker and error.endswith(f"Step {numbers[target + 1]}"):
            continue
        if "empty Step" in error or "step after Final answer" in error or "interior control token" in error:
            raise ValueError(error)
    count = ordinary["prefix_token_count"]
    near_end = final_closes and (next_marker is None or final.start() < next_marker.start())
    return {"reason": "STEP", "prefix_ids": list(generated_ids[:count]),
            "pending_ids": list(generated_ids[count:]), "current_step": target_step,
            "near_end": bool(near_end), "delay_collapsed": bool(near_end),
            "step": {**ordinary, "near_end": bool(near_end)}}


def terminal(tokenizer, generated_ids, eos=False, text=None):
    """Submit a complete standalone Final answer only after EOS, using stage2 format."""
    if not eos:
        return None
    ids = list(generated_ids)
    specials = set(tokenizer.all_special_ids)
    while ids and ids[-1] in specials:
        ids.pop()
    if text is None:
        text = decode(tokenizer, ids)
    else:
        # Dedicated reasoning buffers may need a newline where feedback was
        # removed. Strip EOS by its original decoded suffix, never re-tokenize.
        original = decode(tokenizer, generated_ids)
        content = decode(tokenizer, ids)
        suffix = original[len(content):] if original.startswith(content) else ""
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
    finals = list(FINAL.finditer(text))
    matches = list(STEP.finditer(text))
    numbers = [int(match.group(1)) for match in matches]
    error = None
    if any(token in specials for token in ids):
        error = "INTERIOR_CONTROL_TOKEN"
    elif numbers != list(range(1, len(numbers) + 1)):
        error = "NONSEQUENTIAL_STEPS"
    elif not finals:
        error = "NO_FINAL_ANSWER"
    elif len(finals) != 1 or text[finals[-1].end():].strip():
        error = "FINAL_FORMAT_ERROR"
    answer = boxed(finals[-1].group()) if finals else None
    if not error and answer is None:
        error = "MISSING_OR_INCOMPLETE_BOXED"
    return {"reason": "FORMAT_ERROR" if error else "FINAL", "termination": error,
            "answer": None if error else answer, "text": text,
            "current_step": numbers[-1] if numbers else 0,
            "prefix_ids": ids, "pending_ids": []}


_FEEDBACK = re.compile(
    r"(?ms)^\[Verification feedback\]\n.*?^\[Continue\]\n"
    r"(?:Keep the accepted Steps 1–\d+\. The later attempt has been discarded\."
    r"|No reasoning steps have been accepted\. Restart from Step 1\.)\n"
    r"Continue again from Step \d+\.\n")


def prm_steps(accepted, new):
    """Use stage1's step adapter, retaining Final answer and excluding its feedback wrapper."""
    from stages.stage1.veriserve.prm import split_steps
    previous = split_steps(_FEEDBACK.sub("", accepted))
    following = split_steps(_FEEDBACK.sub("", new), start=previous[-1][0] + 1 if previous else 1)
    return previous + following, len(previous)


def effective_verdict(verdict, first_error_step, checkpoint_step):
    if verdict not in {"PASS", "FAIL", "UNCERTAIN"}:
        raise ValueError(f"Invalid verdict {verdict!r}")
    conflict = verdict == "FAIL" and first_error_step is not None and first_error_step <= checkpoint_step
    return "UNCERTAIN" if conflict else verdict


def verdict_from_scores(scores, numbers, prior_count, threshold=.35):
    """Stage1 deterministic feedback, with the accepted-prefix raw conflict preserved."""
    if len(scores) != len(numbers) or not 0 <= prior_count <= len(numbers):
        raise ValueError("PRM score/step counts do not agree")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Invalid PRM scores")
    failing = next((i for i, score in enumerate(scores) if score < threshold), None)
    conflict = failing is not None and failing < prior_count
    if not numbers or len(numbers) == prior_count:
        raw, effective, first = "UNCERTAIN", "UNCERTAIN", None
        diagnosis, hint = "No complete step to score.", ""
    elif failing is None:
        raw, effective, first, diagnosis, hint = "PASS", "PASS", None, "", ""
    elif conflict:
        raw, effective, first = "FAIL", "UNCERTAIN", numbers[failing]
        diagnosis, hint = "An earlier accepted step received a low process score.", ""
    else:
        raw, effective, first = "FAIL", "FAIL", numbers[failing]
        diagnosis = f"Step {first} has a low process score ({scores[failing]:.3f})."
        hint = f"Recheck the reasoning and arithmetic in Step {first} from the accepted prefix."
    return {"verdict": effective, "raw_verdict": raw, "effective_verdict": effective,
            "first_error_step": first, "diagnosis": diagnosis, "hint": hint,
            "checkpoint_conflict": bool(conflict),
            "step_scores": [{"step": number, "score": round(score, 6)}
                            for number, score in zip(numbers, scores)],
            "verifier_output_kind": "deterministic_adapter_from_process_reward"}


def feedback_text(diagnosis, hint, checkpoint_step):
    from stages.stage1.veriserve.model import feedback_text as stage1_feedback
    value = stage1_feedback(diagnosis, hint, checkpoint_step)
    if checkpoint_step == 0:
        value = value.replace("Keep the accepted Steps 1–0. The later attempt has been discarded.",
                              "No reasoning steps have been accepted. Restart from Step 1.")
    return value


def budget_reason(used, limits, *, generation_tokens=0, context_tokens=None,
                  prm_input_tokens=0, prefill_tokens=0, prm_calls=0, rollbacks=0):
    """Check an action's cost *before* executing it; discarded and pending costs stay used."""
    if any(value < 0 for value in (generation_tokens, context_tokens or 0,
                                  prm_input_tokens, prefill_tokens, prm_calls, rollbacks)):
        raise ValueError("Budget costs cannot be negative")
    checks = ((used.get("attempt_tokens", 0) + generation_tokens, limits["attempt_tokens"], "BUDGET_ATTEMPT"),
              (used.get("generated_tokens", 0) + generation_tokens, limits["generated_tokens"], "BUDGET_GENERATION"),
              (context_tokens or 0, limits["context_tokens"], "CONTEXT_LIMIT"),
              (prm_input_tokens, limits["prm_input_tokens"], "PRM_INPUT_LIMIT"),
              (used.get("prefill_tokens", 0) + prefill_tokens + prm_input_tokens,
               limits["prefill_tokens"], "BUDGET_PREFILL"),
              (used.get("prm_calls", 0) + prm_calls, limits["prm_calls"], "BUDGET_VERIFIER"),
              (used.get("rollbacks", 0) + rollbacks, limits["rollbacks"], "BUDGET_ROLLBACK"))
    return next((reason for count, limit, reason in checks if count > limit), None)


def self_check():
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)

    class Tokenizer:
        all_special_ids = [999]
        def __init__(self, pieces):
            self.pieces = pieces
        def decode(self, ids, **kwargs):
            return "".join("<eos>" if i == 999 else self.pieces[i] for i in ids)

    pieces = ["Step 1:", " Work", ".\n\nSt", "ep 2:", " More", ".", "\n\nStep 3:"]
    tok = Tokenizer(pieces)
    check("unfinished Step 2 is not closed", boundary(tok, list(range(6))) is None)
    found = boundary(tok, list(range(7)))
    check("Step 2 original prefix and pending are separated", found["prefix_ids"] == list(range(6))
          and found["pending_ids"] == [6] and not found["near_end"])
    check("Step 1 cross-marker token aligns backward", boundaries(tok, list(range(7)))[1]["steps"][0]["end_token_index"] == 1)
    crossed = Tokenizer(["Step 1: x\n\nStep 2:", " y", ".\n\nSt", "ep 3:"])
    found = boundary(crossed, [0, 1, 2, 3])
    check("Step 2 crossing token remains pending", found["prefix_ids"] == [0, 1] and found["pending_ids"] == [2, 3]
          and found["step"]["boundary_aligned_back"])
    class ByteTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")
    byte_pieces = [b"Step 1:", b" x", b"\nStep 2: "] + [bytes([b]) for b in "∛".encode()] + [b"\nStep 3:"]
    found = boundary(ByteTokenizer(byte_pieces), list(range(len(byte_pieces))))
    check("Unicode boundary keeps every byte of complete content", found["prefix_ids"] == list(range(len(byte_pieces) - 1)))
    near = Tokenizer(["Step 1:", " x", "\n\nStep 2:", " y", "\n\nFinal answer:", r" \boxed{\frac{1}{2}", "}"])
    found = boundary(near, list(range(5)))
    check("early endpoint anchor retained", found["near_end"] and found["delay_collapsed"] and found["pending_ids"] == [4])
    check("incomplete boxed cannot terminate", terminal(near, list(range(6)), eos=True)["reason"] == "FORMAT_ERROR")
    check("complete box waits for EOS", terminal(near, list(range(7))) is None)
    check("complete final retained before EOS", terminal(near, list(range(7)) + [999], eos=True)["answer"] == r"\frac{1}{2}")
    partial = Tokenizer(["Step 1: x\n\nStep 2: y\n\nStep 3"])
    check("partial next header is pending until colon", boundary(partial, [0]) is None)
    delay = Tokenizer(["Step 1:", " a", "\nStep 2:", " b", "\nStep 3:", " c", "\nStep 4:", " d", "\nFinal answer:"])
    check("DELAY closes fourth ordinary step", boundary(delay, list(range(9)), 4)["current_step"] == 4)
    prior_box = Tokenizer(["Step 1:", r" \boxed{1}", "\nStep 2:", " y", "\nStep 3:"])
    check("anchor eligibility only excludes boxed in Step 2", boundary(prior_box, list(range(5)))["current_step"] == 2)
    for invalid in ("Step 1: a\nStep 3: b", "Step 1: a\nStep 2: b\nStep 2: c",
                    "Step 1: a\nStep 2: \\boxed{2}\nFinal answer:"):
        try:
            boundary(Tokenizer([invalid]), [0])
        except ValueError:
            pass
        else:
            raise AssertionError("illegal or boxed anchor must be excluded")
    check("illegal and boxed anchors excluded", True)
    final_only = Tokenizer([r"Final answer: \boxed{2}", "\nmore reasoning"])
    check("post-final reasoning is format failure", terminal(final_only, [0, 1], True)["reason"] == "FORMAT_ERROR")
    joined = Tokenizer(["Step 1: x\nStep 2: y", "Step 3: z\nFinal answer: \\boxed{2}"])
    joined_text = joined.decode([0]) + "\n\n" + joined.decode([1, 999])
    check("feedback removal preserves reasoning chunk boundaries", terminal(joined, [0, 1, 999], True, text=joined_text)["current_step"] == 3)
    feedback = feedback_text("Step 3 diagnosis", "Recheck arithmetic.", 0)
    check("initial feedback explicitly restarts Step 1", "No reasoning steps have been accepted. Restart from Step 1." in feedback
          and "Keep the accepted Steps 1–0" not in feedback)
    steps, prior = prm_steps("Step 1: accepted", feedback + "Step 2: new\nFinal answer: \\boxed{2}")
    check("PRM retains final and excludes feedback", prior == 1 and len(steps) == 2
          and r"Final answer: \boxed{2}" in steps[-1][1] and all("diagnosis" not in body for _, body in steps))
    steps, prior = prm_steps("Step 1: accepted\nStep 2: accepted", r"Final answer: \boxed{2}")
    check("final-only new segment receives a PRM step", prior == 2 and steps[-1] == (3, r"Final answer: \boxed{2}"))
    conflict = verdict_from_scores([.1, .9], [1, 2], 1)
    check("accepted prefix conflict preserves raw FAIL and effective UNCERTAIN",
          conflict["raw_verdict"] == "FAIL" and conflict["effective_verdict"] == "UNCERTAIN"
          and effective_verdict("FAIL", 1, 1) == "UNCERTAIN")
    check("threshold equality passes", verdict_from_scores([.35], [1], 0)["verdict"] == "PASS")
    check("new low score returns deterministic FAIL", verdict_from_scores([.9, .2], [1, 2], 1)["first_error_step"] == 2)
    limits = dict(attempt_tokens=4, generated_tokens=8, context_tokens=8,
                  prm_input_tokens=8, prefill_tokens=16, prm_calls=4, rollbacks=2)
    used = dict(attempt_tokens=4, generated_tokens=8, prefill_tokens=8, prm_calls=3, rollbacks=2)
    check("exact budget permits necessary terminal verification", budget_reason(used, limits, prm_input_tokens=8, prm_calls=1) is None)
    check("lookahead and revoked tokens cannot be refunded", budget_reason(used, limits, generation_tokens=1) == "BUDGET_ATTEMPT")
    check("FAIL with exhausted rollback budget cannot retry", budget_reason(used, limits, rollbacks=1) == "BUDGET_ROLLBACK")
    check("PRM length has independent tokenizer budget", budget_reason({}, limits, prm_input_tokens=9) == "PRM_INPUT_LIMIT")
    return {"passed": True, "checks": checks, "count": len(checks)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), ensure_ascii=False))
