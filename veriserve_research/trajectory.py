"""Original-token boundaries and Math-Verify grading, without model loading."""
from __future__ import annotations
import re
from .artifacts.io import safe_error
STEP = re.compile(r"(?m)^Step ([1-9]\d*):")
FINAL = re.compile(r"(?m)^Final answer:[^\n]*")
ANSWER_STEP = re.compile(r"(?i)\b(?:final answer|(?:the|our) answer\s+(?:is|equals)|answer is)\b")


def boxed(text):
    """Last complete boxed expression; nested and escaped braces are handled, never math itself."""
    found = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            escapes = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                escapes += 1
                j -= 1
            if escapes % 2:
                continue
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                found.append(text[start:i])
                break
    return found[-1] if found else None


def grade(text, gold, completion, timeout):
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.utils import TimeoutException
    result = {"answer": None, "label": None, "exclusion": None}
    lines = list(FINAL.finditer(text))
    if lines:
        result["answer"] = boxed(lines[-1].group())
    if completion != "EOS":
        result["exclusion"] = completion
        return result
    if not lines:
        result["exclusion"] = "NO_FINAL_ANSWER"
        return result
    if len(lines) != 1 or text[lines[-1].end():].strip():
        result["exclusion"] = "FINAL_FORMAT_ERROR"
        return result
    if result["answer"] is None:
        result["exclusion"] = "MISSING_OR_INCOMPLETE_BOXED"
        return result
    options = {"extraction_config": [LatexExtractionConfig(boxed_match_priority=0)],
               "fallback_mode": "no_fallback", "extraction_mode": "first_match",
               "parsing_timeout": timeout, "raise_on_error": True}
    phase = "GOLD"
    try:
        expected = parse(f"${gold}$", **options)
        if not expected:
            result["exclusion"] = "GOLD_UNPARSEABLE"
            return result
        phase = "PREDICTION"
        predicted = parse(f"${result['answer']}$", **options)
        if not predicted:
            result["exclusion"] = "PREDICTION_UNPARSEABLE"
            return result
        phase = "VERIFY"
        correct = verify(expected, predicted, timeout_seconds=timeout, raise_on_error=True)
        result["label"] = int(not correct)
    except TimeoutException:
        result["exclusion"] = f"{phase}_TIMEOUT"
    except Exception as exc:
        result["exclusion"] = f"{phase}_ERROR"
        result["grading_error"] = safe_error(exc)
    return result


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def aligned_end(tokenizer, ids, text, char_end, char_start=0):
    """Align a character boundary to original token IDs, without re-tokenizing any text."""
    cache = {0: ""}
    def prefix(n):
        if n not in cache:
            cache[n] = decode(tokenizer, ids[:n])
        return cache[n]
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(prefix(mid)) <= char_end:
            lo = mid
        else:
            hi = mid - 1
    n = lo
    previous = 0
    while n:
        if not text.startswith(prefix(n)) or len(prefix(n)) > char_end:
            n -= 1
            continue
        previous = n - 1
        while previous and not text.startswith(prefix(previous)):
            previous -= 1  # A preceding token may contain only a partial UTF-8 character.
        if text[len(prefix(previous)):len(prefix(n))].strip():
            break
        n -= 1
    if not n or len(prefix(previous)) < char_start:
        raise ValueError("FORMAT_ERROR: no complete content token within step")
    if "\ufffd" in prefix(n)[-1:]:
        raise ValueError("FORMAT_ERROR: incomplete UTF-8 token")
    return {"end_token_index": n - 1, "prefix_token_count": n,
            "char_end": char_end, "aligned_char_end": len(prefix(n)),
            "boundary_aligned_back": len(prefix(n)) != char_end}


def boundaries(tokenizer, generated_ids):
    special = set(tokenizer.all_special_ids)
    content_ids = list(generated_ids)
    while content_ids and content_ids[-1] in special:
        content_ids.pop()
    text = decode(tokenizer, content_ids)
    result = {"steps": [], "excluded_steps": [], "format_errors": [], "end": None}
    if not text.rstrip():
        result["format_errors"].append("FORMAT_ERROR: empty generation")
        return text, result
    if any(t in special for t in content_ids):
        result["format_errors"].append("FORMAT_ERROR: interior control token")
        return text, result
    try:
        result["end"] = aligned_end(tokenizer, content_ids, text, len(text))
    except ValueError as exc:
        result["format_errors"].append(str(exc))
    matches = list(STEP.finditer(text))
    final = list(FINAL.finditer(text))
    final_start = final[0].start() if final else len(text)
    numbers = [int(m.group(1)) for m in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        result["format_errors"].append("FORMAT_ERROR: missing, repeated or non-sequential Step N markers")
        return text, result
    if not matches:
        result["format_errors"].append("FORMAT_ERROR: no Step N marker")
    for i, match in enumerate(matches):
        stop = min(matches[i + 1].start() if i + 1 < len(matches) else len(text), final_start)
        if stop <= match.start():
            result["format_errors"].append("FORMAT_ERROR: step after Final answer")
            continue
        segment = text[match.start():stop].rstrip()
        step = {"step_number": int(match.group(1)), "text": segment, "char_start": match.start()}
        if re.search(r"\\boxed\s*\{", segment) or ANSWER_STEP.search(segment):
            step["reason"] = "EXPLICIT_FINAL_OR_BOXED"
            result["excluded_steps"].append(step)
            continue
        if not text[match.end():stop].strip():
            result["format_errors"].append(f"FORMAT_ERROR: empty Step {step['step_number']}")
            continue
        try:
            step.update(aligned_end(tokenizer, content_ids, text,
                                    match.start() + len(segment), match.end()))
            step["effective_step"] = len(result["steps"]) + 1
            step["near_end"] = False
            result["steps"].append(step)
        except ValueError as exc:
            result["format_errors"].append(str(exc))
    # Keep reliable positions even when a separate step failed, but never guess an anomalous marker.
    if result["steps"]:
        result["steps"][-1]["near_end"] = True
    return text, result
