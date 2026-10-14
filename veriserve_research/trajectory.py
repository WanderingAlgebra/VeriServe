"""Original-token boundaries and Math-Verify grading, without model loading."""
from __future__ import annotations
import re
from .artifacts.io import safe_error
STEP = re.compile(r"(?m)^Step ([1-9]\d*):")
FINAL = re.compile(r"(?m)^Final answer:[^\n]*")
ANSWER_STEP = re.compile(r"(?i)\b(?:final answer|(?:the|our) answer\s+(?:is|equals)|answer is)\b")
PARAGRAPHS = 'thinking_paragraphs'
EXPLANATION_FINAL = 'explanation_then_final_line'
BLANK = re.compile(r'\r?\n[ \t]*(?:\r?\n[ \t]*)+')


def paragraph_spans(raw, include_tail=False):
    """Original character spans; only real blank lines close offline checkpoints."""
    opening = re.match(r'\s*<think>', raw)
    cursor = opening.end() if opening else 0
    stop = raw.find('</think>', cursor)
    stop = len(raw) if stop < 0 else stop
    spans = []
    for delimiter in BLANK.finditer(raw, cursor, stop):
        segment = raw[cursor:delimiter.start()]
        if segment.strip():
            spans.append(dict(char_start=cursor+len(segment)-len(segment.lstrip()),
                              char_end=cursor+len(segment.rstrip()),
                              delimiter_char_start=delimiter.start(), delimiter_char_end=delimiter.end()))
        cursor = delimiter.end()
    tail = raw[cursor:stop]
    if include_tail and tail.strip():
        spans.append(dict(char_start=cursor+len(tail)-len(tail.lstrip()),
                          char_end=cursor+len(tail.rstrip())))
    return spans


def verification_units(raw, mode='steps', terminal=False):
    """External checkpoint labels for the verifier; never alter generator token IDs."""
    if mode != PARAGRAPHS:
        return [int(m.group(1)) for m in STEP.finditer(thinking_view(raw))], None
    spans = paragraph_spans(raw, include_tail=True)
    if terminal and '</think>' in raw:
        start = raw.index('</think>')+len('</think>')
        if raw[start:].strip():
            spans.append(dict(char_start=start))
    numbers = list(range(1, len(spans)+1))
    # Retain mathematical text and offsets; mask known wrappers only in the verifier view.
    starts = [0]+[span['char_start'] for span in spans[1:]]+[len(raw)] if spans else []
    units = [dict(checkpoint_number=n, quoted_text=thinking_view(raw[a:b]))
             for n,a,b in zip(numbers,starts,starts[1:])]
    assert ''.join(u['quoted_text'] for u in units) == thinking_view(raw) or not spans
    return numbers, units


def boxed_spans(text):
    """Complete boxed expressions and their original character spans."""
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
                found.append((text[start:i], match.start(), i + 1))
                break
    return found


def boxed(text):
    found = boxed_spans(text)
    return found[-1][0] if found else None


def final_answer_line(raw, completion='EOS'):
    """The revised native final-response contract, shared by grading and submission."""
    result = dict(answer=None, exclusion=None)
    if completion != 'EOS':
        return dict(result, exclusion=completion)
    raw = re.sub(r'(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)+$', '', raw).rstrip()
    error = thinking_error(raw, terminal=True, mode=PARAGRAPHS)
    if error:
        return dict(result, exclusion=error)
    final = raw.split('</think>', 1)[1]
    lines = list(re.finditer(r'(?m)^[ \t]*Final answer:[^\r\n]*', final))
    if not lines:
        return dict(result, exclusion='NO_FINAL_ANSWER')
    if len(lines) != 1 or final[lines[0].end():].strip():
        return dict(result, exclusion='FINAL_FORMAT_ERROR')
    payload = lines[0].group().strip()[len('Final answer:'):].strip()
    boxes = boxed_spans(payload)
    if not boxes:
        return dict(result, exclusion='MISSING_OR_INCOMPLETE_BOXED')
    if len(boxes) != 1 or boxes[0][1:] != (0, len(payload)):
        return dict(result, exclusion='FINAL_FORMAT_ERROR')
    return dict(result, answer=boxes[0][0])


def grade(text, gold, completion, timeout, mode='steps', final_response_mode=None):
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.utils import TimeoutException
    if final_response_mode == EXPLANATION_FINAL:
        result = dict(label=None, **final_answer_line(text, completion))
        if result['exclusion']:
            return result
        text = 'Final answer: \\boxed{' + result['answer'] + '}'
    elif mode == PARAGRAPHS:
        text = text.split('</think>', 1)[1].strip() if '</think>' in text else ''
        if completion == 'EOS' and text and not FINAL.fullmatch(text):
            return dict(answer=None,label=None,exclusion='FINAL_FORMAT_ERROR')
    text = thinking_view(text)
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
    phase = "PREDICTION" if final_response_mode == EXPLANATION_FINAL else "GOLD"
    try:
        predicted = None
        if final_response_mode == EXPLANATION_FINAL:
            predicted = parse(f"${result['answer']}$", **options)
            result['answer_parseable'] = bool(predicted)
            if not predicted:
                result['exclusion'] = 'PREDICTION_UNPARSEABLE'
                return result
        phase = 'GOLD'
        expected = parse(f"${gold}$", **options)
        if not expected:
            result["exclusion"] = "GOLD_UNPARSEABLE"
            return result
        phase = "PREDICTION"
        if predicted is None:
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


def thinking_view(text):
    """Mask known wrappers at their original character offsets; never re-tokenize."""
    return re.sub(r"</?think>", lambda m: "\n" * len(m.group()), text)


def thinking_error(raw, terminal=False, mode='steps'):
    if raw.count('<think>') > 1 or raw.count('</think>') > 1:
        return 'MULTIPLE_THINKING_WRAPPERS'
    if '<think>' in raw and raw[:raw.index('<think>')].strip():
        return 'MISPLACED_THINKING_OPEN'
    if terminal and raw.count('</think>') != 1:
        return 'UNCLOSED_THINKING'
    if mode == PARAGRAPHS:
        return None  # Native thoughts may quote final-answer syntax or restart numbering.
    if '</think>' in raw:
        closed = raw.index('</think>')
        view = thinking_view(raw)
        if any(m.start() < closed for m in FINAL.finditer(view)) or any(m.start() > closed for m in STEP.finditer(view)):
            return 'STEPS_OR_FINAL_OUTSIDE_THINKING_CONTRACT'
    return None


def control_ids(tokenizer):
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
    return set(tokenizer.all_special_ids) - {vocab.get('<think>'), vocab.get('</think>')}


def boundaries(tokenizer, generated_ids, mode='steps'):
    if mode not in ('steps',PARAGRAPHS):
        raise ValueError(f'Unknown boundary mode: {mode}')
    special = control_ids(tokenizer)
    content_ids = list(generated_ids)
    while content_ids and content_ids[-1] in special:
        content_ids.pop()
    raw = decode(tokenizer, content_ids)
    text = thinking_view(raw)
    result = {"steps": [], "excluded_steps": [], "format_errors": [], "end": None}
    if not text.rstrip():
        result["format_errors"].append("FORMAT_ERROR: empty generation")
        return text, result
    if any(t in special for t in content_ids):
        result["format_errors"].append("FORMAT_ERROR: interior control token")
        return text, result
    try:
        result["end"] = aligned_end(tokenizer, content_ids, raw, len(raw))
    except ValueError as exc:
        result["format_errors"].append(str(exc))
    if mode == PARAGRAPHS:
        result['mode'] = mode
        result['selection_semantics'] = 'offline reference paragraph endpoints; delimiter observed later'
        for number,span in enumerate(paragraph_spans(raw),1):
            segment = raw[span['char_start']:span['char_end']]
            point = dict(step_number=number,text=segment,near_end=False,**span)
            if re.search(r'\\boxed\s*\{',segment) or ANSWER_STEP.search(segment):
                point['reason'] = 'EXPLICIT_FINAL_OR_BOXED'
                result['excluded_steps'].append(point)
                continue
            try:
                point.update(aligned_end(tokenizer,content_ids,raw,span['char_end'],span['char_start']))
                if raw[point['aligned_char_end']:span['char_end']].strip():
                    raise ValueError('CHECKPOINT_TOKEN_CROSSES_CONTENT')
                if result['steps'] and point['prefix_token_count'] <= result['steps'][-1]['prefix_token_count']:
                    raise ValueError('NONINCREASING_CHECKPOINT_TOKENS')
                point['effective_step'] = len(result['steps'])+1
                result['steps'].append(point)
            except ValueError as exc:
                point['reason'] = str(exc)
                result['excluded_steps'].append(point)
        if result['steps']:
            result['steps'][-1]['near_end'] = True
        return raw, result
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
            step.update(aligned_end(tokenizer, content_ids, raw,
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
