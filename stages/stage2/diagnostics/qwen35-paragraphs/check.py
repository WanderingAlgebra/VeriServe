"""Offline smoke-only checkpoint diagnostics; never edits scientific records.

Run with environments/qwen35/.venv/bin/python stages/stage2/diagnostics/qwen35-paragraphs/check.py
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, digest, now, read_json, stable_hash
from veriserve_research.probe.materials import record_path
from veriserve_research.trajectory import ANSWER_STEP, aligned_end, decode

BLANK = re.compile(r"\r?\n[ \t]*(?:\r?\n[ \t]*)+")
MATH = re.compile(r"```|\$\$|\$|\\[\[\]()]|\\(?:begin|end)\{(?:equation|align|gather|multline)\*?\}")


def paragraphs(raw):
    start = raw.index('<think>') + len('<think>') if '<think>' in raw else 0
    stop = raw.find('</think>', start)
    stop = len(raw) if stop < 0 else stop
    spans, cursor = [], start
    for delimiter in BLANK.finditer(raw, start, stop):
        segment = raw[cursor:delimiter.start()]
        if segment.strip():
            left = cursor + len(segment) - len(segment.lstrip())
            right = cursor + len(segment.rstrip())
            spans.append((left, right, delimiter.start(), delimiter.end()))
        cursor = delimiter.end()
    return spans, raw[cursor:stop].strip()


def lexical_flags(prefix):
    # ponytail: lexical risk hints only; semantic completeness still needs review.
    opened, flags = set(), []
    for match in MATH.finditer(prefix):
        backslashes, pos = 0, match.start() - 1
        while pos >= 0 and prefix[pos] == '\\':
            backslashes, pos = backslashes + 1, pos - 1
        if backslashes % 2:
            continue
        marker = match.group()
        if marker == '```':
            opened.symmetric_difference_update({marker})
        elif '```' in opened:
            continue
        elif marker in ('$', '$$'):
            opened.symmetric_difference_update({marker})
        elif marker in ('\\[', '\\('):
            opened.add(marker)
        elif marker in ('\\]', '\\)'):
            opened.discard({'\\]': '\\[', '\\)': '\\('}[marker])
        elif marker.startswith('\\begin'):
            opened.add(marker[6:])
        elif marker.startswith('\\end'):
            opened.discard(marker[4:])
    if opened:
        flags.append('OPEN_MATH_OR_CODE_AT_BOUNDARY')
    if re.search(r'[,;:=+*/-]\s*$', prefix):
        flags.append('CONTINUATION_PUNCTUATION')
    return flags


def observed_count(tokenizer, ids, raw, char_end):
    """First stable original prefix containing the entire real delimiter."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode(tokenizer, ids[:mid])) < char_end:
            lo = mid + 1
        else:
            hi = mid
    while lo <= len(ids):
        prefix = decode(tokenizer, ids[:lo])
        if len(prefix) >= char_end and raw.startswith(prefix):
            return lo
        lo += 1
    raise ValueError('No stable original token prefix contains delimiter')


def self_check(tokenizer):
    raw = '<think>计算 $1+1=2$。\n\nReview:\n\nFinal answer: \\boxed{2}\n\n</think>answer'
    spans, tail = paragraphs(raw)
    assert len(spans) == 3 and not tail
    assert [raw[a:b] for a,b,_,_ in spans] == ['计算 $1+1=2$。', 'Review:', 'Final answer: \\boxed{2}']
    assert len(paragraphs('a\n\nb unfinished')[0]) == 1
    assert len(paragraphs('<think>\n</think>')[0]) == 0
    assert len(paragraphs('a\r\n \r\nb')[0]) == 1
    assert lexical_flags('inside $x\n\n') == ['OPEN_MATH_OR_CODE_AT_BOUNDARY']
    assert not lexical_flags(r'cost \$3 and $x$')
    assert lexical_flags('```python\nx=1\n\n') == ['OPEN_MATH_OR_CODE_AT_BOUNDARY']
    assert not lexical_flags(r'\begin{align}x=1\end{align}')
    ids = tokenizer.encode(raw, add_special_tokens=False)
    for a,b,_,end in spans:
        aligned = aligned_end(tokenizer, ids, raw, b, a)
        prefix = decode(tokenizer, ids[:aligned['prefix_token_count']])
        assert raw.startswith(prefix) and len(prefix) <= b
        count = observed_count(tokenizer, ids, raw, end)
        assert count >= aligned['prefix_token_count']
        assert len(decode(tokenizer, ids[:count])) >= end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'stages/stage2/runs/v2-20261013-2f15553ff1e5')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_json(run/'manifest.json')
    cfg = manifest['config']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revisions']['tokenizer'],
        cache_dir=cfg.get('model_cache_dir'), local_files_only=True)
    self_check(tokenizer)
    paths = [run/'manifest.json', *(record_path(run,'smoke',q['unique_id']) for q in manifest['splits']['smoke'])]
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    results = []
    for row in manifest['splits']['smoke']:
        record = read_json(record_path(run,'smoke',row['unique_id']))
        ids, raw = record['generated_ids'], record['raw_text']
        assert decode(tokenizer, ids) == raw
        assert record['token_sha256'] == stable_hash({'prompt_ids':record['prompt_ids'],'generated_ids':ids})
        spans, tail = paragraphs(raw)
        points, previous = [], 0
        for number,(a,b,c,d) in enumerate(spans,1):
            segment = raw[a:b]
            point = dict(paragraph_number=number,char_start=a,char_end=b,delimiter_char_start=c,
                delimiter_char_end=d,tail=segment[-240:],flags=lexical_flags(raw[:b]),near_end=False)
            if re.search(r'\b(?:format(?:ting)?|instructions?|system prompt|step numbering)\b', segment,re.I):
                point['flags'].append('POSSIBLE_FORMAT_DISCUSSION')
            point['excluded_reason'] = ('EXPLICIT_FINAL_OR_BOXED' if re.search(r'\\boxed\s*\{',segment)
                or ANSWER_STEP.search(segment) else None)
            try:
                point.update(aligned_end(tokenizer,ids,raw,b,a))
                point['observed_prefix_token_count'] = observed_count(tokenizer,ids,raw,d)
                point['unaligned_content_tail'] = raw[point['aligned_char_end']:b]
                point['gap_from_previous_paragraph_tokens'] = point['prefix_token_count'] - previous
                previous = point['prefix_token_count']
            except ValueError as exc:
                point['alignment_error'] = str(exc)
            points.append(point)
        candidates = [p for p in points if not p['excluded_reason'] and not p.get('alignment_error')]
        if candidates:
            candidates[-1]['near_end'] = True
        counts = [p['prefix_token_count'] for p in candidates]
        results.append(dict(unique_id=row['unique_id'],completion=record['completion'],
            generated_tokens=len(ids),thinking_closed='</think>' in raw,closed_paragraphs=len(points),
            candidates=len(candidates),distinct_candidate_tokens=len(set(counts)),
            can_supply_two_distinct_candidates=len(set(counts))>=2,undelimited_tail_chars=len(tail),
            undelimited_tail_preview=tail[-240:],points=points))
        print(f"{row['unique_id'][:12]} paragraphs={len(points)} candidates={len(candidates)}",flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert before == after, 'Source records changed during diagnostics'
    points = [p for r in results for p in r['points']]
    gaps = [p['gap_from_previous_paragraph_tokens'] for p in points if 'prefix_token_count' in p]
    result = dict(validation_only=True,scientific_results=False,new_smoke_passed=False,created=now(),
        rule='Original Thinking text only; actual blank line; no wrapper-induced or EOF boundary; explicit-answer paragraphs excluded.',
        causal_semantics='Offline reference-selected content token endpoints. Delimiters are observed later; no claim of online discovery.',
        limitations=['Old Step-constrained prompts, not natural-prompt trajectories.',
            'Lexical flags are risk hints, not mathematical correctness or independent human audit.',
            'Changing the production protocol requires new identity, shared stage2/stage3 boundaries and fresh smoke.'],
        self_check='PASS',script_sha256=digest(__file__),tokenizer_revision=cfg['revisions']['tokenizer'],
        source_hashes=before,sources_unchanged=True,questions=results,
        summary=dict(questions=len(results),closed_paragraphs=len(points),candidates=sum(r['candidates'] for r in results),
            questions_with_two_distinct_candidates=sum(r['can_supply_two_distinct_candidates'] for r in results),
            completion_counts=dict(Counter(r['completion'] for r in results)),
            lexical_flags=dict(Counter(f for p in points for f in p['flags'])),
            alignment_errors=sum('alignment_error' in p for p in points),
            content_tail_loss=sum(bool(p.get('unaligned_content_tail','').strip()) for p in points),
            paragraph_gap_median_tokens=statistics.median(gaps),paragraph_gap_max_tokens=max(gaps)))
    atomic_json(Path(__file__).with_name('offline_check.json'),result)
    print(result['summary'],flush=True)


if __name__ == '__main__':
    main()
