"""Two fresh requests; reuse resident models, token boundaries and native KV crop."""
from __future__ import annotations

import copy
import gc
import math
import time
from pathlib import Path

from stages.stage1.veriserve.common import atomic_json, stable_hash
from stages.stage2 import run_probe as old
from stages.stage3 import protocol, run_timing as legacy, storage
from . import source_hashes

Session = legacy.Session
clock = legacy.clock


class ReferenceMismatch(RuntimeError):
    """The original trajectory cannot select positions for this streaming backend."""
    def __init__(self, details):
        self.details = details
        super().__init__(str(details))


def limits(cfg):
    return dict(attempt_tokens=cfg['max_new_tokens'], generated_tokens=cfg['max_request_tokens'],
                context_tokens=cfg['max_total_tokens'], prm_input_tokens=cfg['max_prm_input_tokens'],
                prefill_tokens=cfg['max_forward_tokens'], prm_calls=cfg['max_prm_calls'],
                rollbacks=cfg['max_rollbacks'])


def budget(cfg, state, **cost):
    used = {**state['budget'], 'prefill_tokens': state['budget']['forward_tokens']}
    return protocol.budget_reason(used, limits(cfg), **cost)


def closed_boundary(tokenizer, ids, target):
    """Stage2's original token endpoint, plus the complete text of crossing tokens."""
    text = old.decode(tokenizer, ids)
    markers = list(old.STEP.finditer(text))
    if [int(m.group(1)) for m in markers] != list(range(1, len(markers) + 1)):
        raise ValueError('NONSEQUENTIAL_STEPS')
    selected = next((i for i, m in enumerate(markers) if int(m.group(1)) == target), None)
    if selected is None:
        return None
    final = old.FINAL.search(text)
    if final and final.start() < markers[selected].start():
        raise ValueError('STEP_AFTER_FINAL')
    following = markers[selected + 1] if selected + 1 < len(markers) else None
    closures = [m.start() for m in (following, final) if m and m.start() > markers[selected].start()]
    if not closures:
        return None
    stop = min(closures)
    _, parsed = old.boundaries(tokenizer, ids)
    # The following marker/body is pending lookahead and can have no complete
    # content token yet. Validate every closed marker through our target only.
    parsed_numbers = {s['step_number'] for name in ('steps', 'excluded_steps') for s in parsed[name]}
    if not set(range(1, target + 1)).issubset(parsed_numbers):
        raise ValueError('; '.join(parsed['format_errors']) or 'INCOMPLETE_CLOSED_STEP')
    step = next((s for s in parsed['steps'] if s['step_number'] == target), None)
    if step is None:
        raise ValueError('INELIGIBLE_SELECTED_STEP')
    count = step['prefix_token_count']
    return dict(prefix_ids=list(ids[:count]), pending_ids=list(ids[count:]),
                current_step=target, step=step, prm_text=text[:stop].rstrip())


def initial_state(row, selection, arm, backend):
    prompt = selection['prompt_ids']
    snapshot = dict(unique_id=row['unique_id'], phase=selection['phase'], prompt_ids=prompt,
                    prefix_ids=[], pending_ids=[], token_hash=stable_hash(prompt), backend=backend,
                    budget=dict(generated_tokens=0, attempt_tokens=0, forward_tokens=len(prompt)))
    state = legacy.arm_state(snapshot, arm)
    state.update(experiment_type='within_question_high_low', problem=row['problem'],
                 selected_position=copy.deepcopy(selection['positions'][arm]),
                 selection_hash=stable_hash(selection), source_hashes=source_hashes(),
                 intermediate_checks=0, reference_prefix_match=None)
    return state


def reasoning_text(state, tokenizer, start=0, end=None):
    """After crop, show checked content without the crossing token's next header."""
    if not state.get('checkpoint_reworked'):
        return legacy.reasoning_text(state, tokenizer, start, end)
    count = len(state['checkpoint_reasoning_ids'])
    end = len(state['reasoning_ids']) if end is None else end
    if start >= count:
        return legacy.reasoning_text(state, tokenizer, start, end)
    if start != 0 or end < count:
        raise ValueError('A logical accepted checkpoint cannot be sliced inside its raw token span')
    tail = legacy.reasoning_text(state, tokenizer, count, end)
    return state['checkpoint_text'] + ('\n\n' + tail if tail else '')


def checkpoint_span(tokenizer, ids, floor, checked_text):
    """Keep original tokens through all scored characters, including a crossing tail."""
    full = old.decode(tokenizer, ids)
    if not full.startswith(checked_text):
        raise RuntimeError('Scored first-check text is not the original reasoning prefix')
    for count in range(floor, len(ids) + 1):
        text = old.decode(tokenizer, ids[:count])
        if text.startswith(checked_text) and full.startswith(text):
            return count, text[len(checked_text):]
    raise RuntimeError('No original token prefix covers the complete checked step')


def strict_result(result, numbers, prior, threshold, synthetic=False):
    """Keep every low score a FAIL, including accepted-prefix conflicts."""
    result = copy.deepcopy(result)
    scores = result.get('unrounded_scores', [s['score'] for s in result.get('step_scores', [])])
    valid = bool(numbers) and len(scores) == len(numbers) and all(
        math.isfinite(s) and 0 <= s <= 1 for s in scores)
    if valid:
        first = next((i for i, score in enumerate(scores) if score < threshold), None)
        verdict = 'FAIL' if first is not None else 'PASS'
        result['first_error_step'] = numbers[first] if first is not None else None
        result['checkpoint_conflict'] = first is not None and first < prior
    elif synthetic and result.get('verdict') in ('PASS', 'FAIL'):
        verdict = result['verdict']
        result['checkpoint_conflict'] = bool(verdict == 'FAIL' and result.get('first_error_step')
                                             in numbers[:prior])
    else:
        verdict = 'UNAVAILABLE'
        result['checkpoint_conflict'] = False
    result.update(raw_verdict=verdict, effective_verdict=verdict, verdict=verdict)
    return result


def check(cfg, state, session, verifier, tokenizer, prefix_count, final, *, prm_text=None, forced=None):
    """One strict check; crop only to the most recent PASS and prefill feedback."""
    if not final and state['intermediate_checks']:
        raise AssertionError('Second intermediate check is forbidden')
    prefix = state['reasoning_ids'][:prefix_count]
    accepted_count = len(state['checkpoint_reasoning_ids'])
    if prefix[:accepted_count] != state['checkpoint_reasoning_ids']:
        raise RuntimeError('Accepted reasoning prefix changed')
    full_text = reasoning_text(state, tokenizer, 0, prefix_count)
    accepted = state.get('checkpoint_text', '') if accepted_count else ''
    if not full_text.startswith(accepted):
        raise RuntimeError('Logical accepted reasoning prefix changed')
    new = full_text[len(accepted):]
    if prm_text is not None:
        # First check only: include the target step tail in a crossing pending token,
        # while retaining that token and its KV for continuation after a PASS.
        if accepted_count:
            raise AssertionError('Text override is only valid before the first checkpoint')
        new = prm_text
    ids, numbers, prior = verifier.input_ids(state['problem'], accepted, new)
    if not ids or not numbers:
        return 'NO_SCORABLE_STEPS'
    reason = budget(cfg, state, prm_input_tokens=len(ids), prm_calls=1)
    if reason:
        return reason
    start = clock()
    try:
        result = legacy.verify_prm(verifier, ids, numbers, prior) if forced is None else forced
    except ValueError as exc:
        result = dict(verdict='UNAVAILABLE', diagnosis=old.safe_error(exc), hint='')
    result = strict_result(result, numbers, prior, verifier.threshold, synthetic=forced is not None)
    seconds = clock() - start
    verdict = result['effective_verdict']
    state['budget']['prm_calls'] += 1
    state['budget']['forward_tokens'] += len(ids)
    current = max([int(m.group(1)) for m in old.STEP.finditer(
        full_text)] or [state['checkpoint_step']])
    feedback = protocol.feedback_text(result.get('diagnosis', ''), result.get('hint', ''),
                                      state['checkpoint_step']) if verdict == 'FAIL' else ''
    feedback_ids = tokenizer.encode(feedback, add_special_tokens=False) if feedback else []
    entry = dict(kind='CHECK', position='FINAL' if final else 'STEP',
                 current_step=current, reasoning_tokens=prefix_count, prefix_ids=prefix.copy(),
                 is_first=not state['first_check_done'], input_ids=ids, input_tokens=len(ids),
                 accepted_steps=prior, seconds=seconds, feedback_text=feedback,
                 feedback_ids=feedback_ids, synthetic=forced is not None, **result)
    state['checks'].append(entry)
    state['events'].append(entry.copy())
    state['first_check_done'] = True
    if not final:
        state['intermediate_checks'] += 1
    if verdict == 'PASS':
        checkpoint_count, suffix = (prefix_count, '') if final or prm_text is None else checkpoint_span(
            tokenizer, state['reasoning_ids'], prefix_count, prm_text)
        checked_text = full_text if final or prm_text is None else prm_text
        end = state['reasoning_positions'][checkpoint_count - 1] + 1 if checkpoint_count else len(state['prompt_ids'])
        state['checkpoint_ids'] = state['context_ids'][:end]
        state['checkpoint_reasoning_ids'] = state['reasoning_ids'][:checkpoint_count]
        state['checkpoint_positions'] = state['reasoning_positions'][:checkpoint_count]
        state['checkpoint_step'] = current
        state['checkpoint_breaks'] = [b for b in state['reasoning_breaks'] if b < checkpoint_count]
        state['checkpoint_text'] = checked_text
        state['checkpoint_unscored_suffix'] = suffix
        metadata = dict(checkpoint_reasoning_tokens=checkpoint_count,
                        checkpoint_extra_token_ids=state['reasoning_ids'][prefix_count:checkpoint_count],
                        checkpoint_unscored_suffix=suffix, checkpoint_text=checked_text)
        entry.update(metadata)
        state['events'][-1].update(metadata)
        if final:
            state['submitted_reasoning_ids'] = prefix.copy()
            return 'FINAL_PASS'
        return None
    if verdict != 'FAIL':
        return 'PRM_UNAVAILABLE'
    checkpoint = state['checkpoint_ids']
    missing = max(0, len(checkpoint) - session.cache.get_seq_length()) + len(feedback_ids)
    reason = budget(cfg, state, rollbacks=1, prefill_tokens=missing,
                    context_tokens=len(checkpoint) + len(feedback_ids))
    if reason:
        return reason
    if (state['budget']['generated_tokens'] >= cfg['max_request_tokens']
            or state['budget']['prm_calls'] >= cfg['max_prm_calls']):
        return 'BUDGET_NO_REWORK'
    revoked = len(state['reasoning_ids']) - len(state['checkpoint_reasoning_ids'])
    cache_before = session.cache.get_seq_length()
    start = clock()
    charged = session.rollback(checkpoint, feedback_ids)
    if charged != missing:
        raise RuntimeError('Feedback prefill accounting differs from actual input')
    state['budget']['rollbacks'] += 1
    state['budget']['revoked_tokens'] += revoked
    state['budget']['forward_tokens'] += charged
    state['budget']['feedback_tokens'] += len(feedback_ids)
    state['budget']['attempt_tokens'] = 0
    state['context_ids'] = session.ids.copy()
    state['reasoning_ids'] = state['checkpoint_reasoning_ids'].copy()
    state['reasoning_positions'] = state['checkpoint_positions'].copy()
    state['reasoning_breaks'] = state['checkpoint_breaks'].copy() + [len(state['reasoning_ids'])]
    state['checkpoint_reworked'] = bool(state['checkpoint_reasoning_ids'])
    state['feedback_tokens'].append(feedback_ids)
    state['eos'] = False
    state['events'].append(dict(kind='ROLLBACK', checkpoint_step=state['checkpoint_step'],
                              checkpoint_length=len(checkpoint), cache_before=cache_before,
                              cache_after=session.cache.get_seq_length(), feedback_prefill_tokens=charged,
                              revoked_tokens=revoked, feedback_ids=feedback_ids, seconds=clock() - start))
    return None


def eos_tokens(model):
    configured = model.generation_config.eos_token_id
    return set(configured if isinstance(configured, list) else [configured])


def execute_path(cfg, run, row, selection, arm, model, tokenizer, verifier, backend):
    """T_request starts before fresh prompt prefill; gold grading follows the timer."""
    state = initial_state(row, selection, arm, backend)
    path = Path(run) / 'arms' / f'{legacy.key(row["unique_id"])}.{arm}.json'
    selected = selection['positions'][arm]
    session = None
    start = clock()
    try:
        reason = budget(cfg, state, context_tokens=len(state['prompt_ids']))
        if not reason:
            session = Session(model, state['prompt_ids'])
            atomic_json(path, state)
            while True:
                reason = budget(cfg, state, generation_tokens=1, context_tokens=len(session.ids) + 1)
                if reason:
                    break
                token = session.sample()
                state['context_ids'] = session.ids.copy()
                state['reasoning_positions'].append(len(session.ids) - 1)
                state['reasoning_ids'].append(token)
                state['budget']['generated_tokens'] += 1
                state['budget']['attempt_tokens'] += 1
                if not state['generated_segments'] or state['generated_segments'][-1]['kind'] != 'DECODE':
                    state['generated_segments'].append(dict(kind='DECODE', ids=[]))
                state['generated_segments'][-1]['ids'].append(token)
                if not state['first_check_done']:
                    i = len(state['reasoning_ids']) - 1
                    reference = selection['generated_ids']
                    if i >= len(reference) or token != reference[i]:
                        raise ReferenceMismatch(dict(arm=arm, unique_id=row['unique_id'], token_index=i,
                                                     expected=reference[i] if i < len(reference) else None,
                                                     actual=token, before_first_check=True))
                if token in eos_tokens(model):
                    state['eos'] = True
                    endpoint = protocol.terminal(tokenizer, state['reasoning_ids'], True,
                                                 text=reasoning_text(state, tokenizer))
                    if not endpoint or endpoint['reason'] != 'FINAL':
                        reason = endpoint.get('termination', 'FINAL_FORMAT_ERROR') if endpoint else 'EOS_FORMAT_FAILURE'
                        break
                    if not state['first_check_done']:
                        reason = 'EOS_BEFORE_SELECTED_STEP'
                        break
                    reason = check(cfg, state, session, verifier, tokenizer,
                                   len(endpoint['prefix_ids']), True)
                    atomic_json(path, state)
                    if reason:
                        break
                    continue
                if not state['first_check_done']:
                    try:
                        boundary = closed_boundary(tokenizer, state['reasoning_ids'], selected['step_number'])
                    except ValueError as exc:
                        reason = 'STEP_FORMAT_ERROR: ' + str(exc)
                        break
                    if boundary:
                        if boundary['prefix_ids'] != selected['prefix_ids']:
                            raise ReferenceMismatch(dict(arm=arm, unique_id=row['unique_id'],
                                                         reason='SELECTED_ENDPOINT_MISMATCH'))
                        state['reference_prefix_match'] = True
                        state['first_check_pending_ids'] = boundary['pending_ids']
                        state['first_check_generated_ids'] = state['reasoning_ids'].copy()
                        reason = check(cfg, state, session, verifier, tokenizer,
                                       len(boundary['prefix_ids']), False, prm_text=boundary['prm_text'])
                        atomic_json(path, state)
                        if reason:
                            break
                if state['budget']['generated_tokens'] % 32 == 0:
                    atomic_json(path, state)
        state.update(status='TIMED_COMPLETE', termination=reason)
        if state.get('submitted_reasoning_ids') is not None:
            state['submitted_text'] = reasoning_text(state, tokenizer, 0,
                                                    len(state['submitted_reasoning_ids']))
        atomic_json(path, state)
        state['timing']['T_request'] = clock() - start
        state['timing']['definition'] = ('fresh prompt prefill through submission/termination; includes '
                                         'in-path atomic saves; excludes grading and final timing metadata save')
        legacy.finish_arm(cfg, state, row)
        exclusion = state['grading'].get('exclusion') or ''
        if state.get('submitted_reasoning_ids') is not None and exclusion.endswith('_ERROR'):
            state['Y'] = None
        atomic_json(path, state)
        storage.event(run, 'HIGH_LOW_PATH_COMPLETED', unique_id=row['unique_id'], arm=arm,
                      Y=state['Y'], termination=reason, T_request=state['timing']['T_request'])
        print(f"{state['phase']} {row['unique_id']} {arm} Y={state['Y']} {reason} "
              f"T_request={state['timing']['T_request']:.3f}", flush=True)
        return state
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status='REFERENCE_MISMATCH' if isinstance(exc, ReferenceMismatch) else 'INFRA_INTERRUPTED',
                     error=old.safe_error(exc), completed=False)
        if isinstance(exc, ReferenceMismatch):
            state['mismatch'] = exc.details
        state['timing']['interrupted_wall'] = clock() - start
        atomic_json(path, state)
        raise
    finally:
        del session
        gc.collect()


def collect_reference(cfg, row, model, tokenizer):
    """Unintervened streaming reference; offline cost is never a path's T_request."""
    prompt = cfg['prompt'].replace('{problem}', row['problem'])
    prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               tokenize=True, add_generation_prompt=True)
    generated = []
    start = clock()
    session = None
    completion = 'MAX_NEW_TOKENS'
    try:
        if len(prompt_ids) > cfg['max_total_tokens'] or len(prompt_ids) > cfg['max_forward_tokens']:
            completion = 'INITIAL_CONTEXT_OR_PREFILL_LIMIT'
        else:
            session = Session(model, prompt_ids)
            for _ in range(cfg['max_new_tokens']):
                if len(session.ids) >= cfg['max_total_tokens']:
                    completion = 'CONTEXT_LIMIT'
                    break
                token = session.sample()
                generated.append(token)
                if token in eos_tokens(model):
                    completion = 'EOS'
                    break
        return dict(prompt_ids=prompt_ids, generated_ids=generated, completion=completion,
                    raw_text=old.decode(tokenizer, generated),
                    token_sha256=stable_hash(dict(prompt_ids=prompt_ids, generated_ids=generated)),
                    reference_origin='stage3_stream_unintervened_greedy',
                    offline_reference_seconds=clock() - start)
    finally:
        del session
        gc.collect()


def self_check(cfg):
    """CPU assertions use the same state transition functions as real requests."""
    from unittest.mock import patch
    checks = []
    class Tokenizer:
        all_special_ids = [999]
        pieces = {10: 'prompt', 15: 'Step 1:', 11: ' x', 12: '.\n\nStep 2:', 16: 'Step 2:', 13: ' y',
                  14: '\n\nFinal answer: \\boxed{2}', 999: '<eos>', 88: '<feedback>', 89: '<continue>'}
        def decode(self, ids, **kwargs):
            return ''.join(self.pieces[i] for i in ids)
        def encode(self, text, **kwargs):
            return [88, 89]
    class Cache:
        def __init__(self, length):
            self.length = length
        def get_seq_length(self):
            return self.length
    class FakeSession:
        def __init__(self, ids):
            self.ids = list(ids)
            self.cache = Cache(max(0, len(ids) - 1))
        def rollback(self, checkpoint, feedback):
            assert self.ids[:len(checkpoint)] == checkpoint
            charged = max(0, len(checkpoint) - self.cache.length) + len(feedback)
            self.crop_to = min(self.cache.length, len(checkpoint))
            self.ids = list(checkpoint) + list(feedback)
            self.cache.length = len(self.ids)
            return charged
    class Verifier:
        threshold = .35
        def input_ids(self, question, accepted, new):
            steps, prior = protocol.prm_steps(accepted, new)
            self.steps = steps
            return list(range(len(steps))), [n for n, _ in steps], prior
    tokenizer, verifier = Tokenizer(), Verifier()
    selection = dict(prompt_ids=[10], generated_ids=[15, 11, 12, 13, 14, 999], phase='pilot',
                     positions={a: dict(step_number=1, prefix_ids=[15, 11], q=.5, z=0., near_end=False)
                                for a in ('HIGH', 'LOW')})
    row = dict(unique_id='synthetic', problem='synthetic')
    state = initial_state(row, selection, 'HIGH', {})
    other = initial_state(row, selection, 'LOW', {})
    state.update(context_ids=[10, 15, 11, 12], reasoning_ids=[15, 11, 12], reasoning_positions=[1, 2, 3])
    state['budget'].update(generated_tokens=3, attempt_tokens=3)
    session = FakeSession(state['context_ids'])
    boundary = closed_boundary(tokenizer, state['reasoning_ids'], 1)
    assert boundary['prefix_ids'] == [15, 11] and boundary['pending_ids'] == [12]
    assert boundary['prm_text'] == 'Step 1: x.'
    checks.append('arbitrary target and crossing-token endpoint align; PRM receives full step tail')
    with patch(__name__ + '.clock', return_value=0.):
        assert check(cfg, state, session, verifier, tokenizer, 2, False,
                     prm_text=boundary['prm_text'], forced=dict(verdict='PASS')) is None
        assert verifier.steps == [(1, 'x.')]
        assert state['checkpoint_ids'] == [10, 15, 11, 12] and state['checkpoint_step'] == 1
        assert state['checkpoint_text'] == 'Step 1: x.' and state['checkpoint_unscored_suffix'] == '\n\nStep 2:'
        assert session.ids == [10, 15, 11, 12] and state['context_ids'] == session.ids
        assert state['intermediate_checks'] == 1 and state['reference_prefix_match'] is None
        checks.append('PASS stores exact token checkpoint and retains KV/pending token')
        try:
            check(cfg, state, session, verifier, tokenizer, 2, False, forced=dict(verdict='PASS'))
        except AssertionError:
            pass
        else:
            raise AssertionError('Second intermediate check was allowed')
        checks.append('exactly one intermediate check even after rework')
        state['reasoning_ids'] += [13, 14, 999]
        state['reasoning_positions'] += [4, 5, 6]
        state['context_ids'] += [13, 14, 999]
        state['budget'].update(generated_tokens=6, attempt_tokens=6)
        session = FakeSession(state['context_ids'])
        assert check(cfg, state, session, verifier, tokenizer, 5, True,
                     forced=dict(verdict='FAIL', first_error_step=1,
                                 unrounded_scores=[.1, .9])) is None
        assert state['checks'][-1]['effective_verdict'] == 'FAIL'
        assert state['checks'][-1]['checkpoint_conflict']
        assert session.crop_to == 4 and session.ids == [10, 15, 11, 12, 88, 89]
        assert state['checkpoint_ids'] == [10, 15, 11, 12] and state['reasoning_ids'] == [15, 11, 12]
        assert reasoning_text(state, tokenizer) == 'Step 1: x.'
        assert state['budget']['generated_tokens'] == 6 and state['budget']['revoked_tokens'] == 3
        assert state['budget']['attempt_tokens'] == 0 and state['budget']['feedback_tokens'] == 2
        assert state['budget']['forward_tokens'] == 4 + 2  # prompt + two PRM inputs + feedback
        assert state.get('submitted_reasoning_ids') is None
        checks.append('accepted-prefix FAIL crops to latest PASS including scored crossing tail; hides unscored header fragment')
        untouched = copy.deepcopy(state)
        untouched['budget']['rollbacks'] = cfg['max_rollbacks']
        before_ids = session.ids.copy()
        assert check(cfg, untouched, session, verifier, tokenizer, 3, True,
                     forced=dict(verdict='FAIL', first_error_step=1)) == 'BUDGET_ROLLBACK'
        assert session.ids == before_ids
        checks.append('budget checked before crop; generated/revoked costs never refunded')
        no_score = copy.deepcopy(state)
        assert check(cfg, no_score, session, verifier, tokenizer, 3, True,
                     forced=dict(verdict='UNCERTAIN')) == 'PRM_UNAVAILABLE'
        assert no_score.get('submitted_reasoning_ids') is None
        submitted = copy.deepcopy(state)
        assert check(cfg, submitted, session, verifier, tokenizer, 3, True,
                     forced=dict(verdict='PASS')) == 'FINAL_PASS'
        assert submitted['submitted_reasoning_ids'] == [15, 11, 12]
        checks.append('only terminal PASS can submit; unavailable PRM cannot pass')
    assert other['budget']['generated_tokens'] == 0 and other['reasoning_ids'] == []
    assert budget(cfg, state, generation_tokens=cfg['max_request_tokens']) in ('BUDGET_ATTEMPT', 'BUDGET_GENERATION')
    assert strict_result(dict(unrounded_scores=[.35]), [1], 0, .35)['verdict'] == 'PASS'
    assert strict_result(dict(unrounded_scores=[float('nan')]), [1], 0, .35)['verdict'] == 'UNAVAILABLE'
    checks.append('independent budgets; threshold equality passes; invalid scores unavailable')
    import contextlib
    import io
    import tempfile
    ticks, sessions, force_terminal_fail = [0.], [], [False]
    class StreamingSession(FakeSession):
        def __init__(self, model, ids):
            super().__init__(ids)
            self.iterator = iter(selection['generated_ids'])
            ticks[0] += 2.  # Real request timing must include prompt prefill.
            sessions.append(self)
        def sample(self):
            ticks[0] += 1.
            token = next(self.iterator)
            self.ids.append(token)
            self.cache.length = len(self.ids) - 1
            return token
        def rollback(self, checkpoint, feedback):
            charged = super().rollback(checkpoint, feedback)
            self.iterator = iter([16, 13, 14, 999])
            ticks[0] += charged
            return charged
    class Model:
        class generation_config:
            eos_token_id = 999
    def fake_verify(verifier, ids, numbers, prior):
        ticks[0] += 3.
        if force_terminal_fail[0] and prior and len(numbers) > prior:
            force_terminal_fail[0] = False
            return dict(verdict='FAIL', unrounded_scores=[.9] * prior + [.1] * (len(numbers) - prior))
        return dict(verdict='PASS', unrounded_scores=[.9] * len(numbers))
    real_finish = legacy.finish_arm
    def grade_after_timer(cfg, state, row):
        ticks[0] += 100.
        return real_finish(cfg, state, row)
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
            patch(__name__ + '.Session', StreamingSession), patch(__name__ + '.clock', side_effect=lambda: ticks[0]), \
            patch.object(legacy, 'verify_prm', side_effect=fake_verify), \
            patch.object(legacy, 'finish_arm', side_effect=grade_after_timer):
        pair = [execute_path(cfg, folder, {**row, 'answer': '2'}, selection, arm,
                             Model(), tokenizer, verifier, {}) for arm in ('HIGH', 'LOW')]
        assert len(sessions) == 2 and sessions[0] is not sessions[1]
        assert all(a['Y'] == 1 and a['termination'] == 'FINAL_PASS' and a['intermediate_checks'] == 1
                   and a['reference_prefix_match'] and a['budget']['generated_tokens'] == 6
                   and a['budget']['prm_calls'] == 2 and a['timing']['T_request'] == 14. for a in pair)
        denied = execute_path({**cfg, 'max_prm_calls': 1}, folder, {**row, 'answer': '2'}, selection,
                              'HIGH', Model(), tokenizer, verifier, {})
        assert denied['Y'] == 0 and denied['termination'] == 'BUDGET_VERIFIER'
        assert denied.get('submitted_reasoning_ids') is None and denied['budget']['prm_calls'] == 1
        force_terminal_fail[0] = True
        reworked = execute_path(cfg, folder, {**row, 'answer': '2'}, selection,
                                'HIGH', Model(), tokenizer, verifier, {})
        assert reworked['Y'] == 1 and reworked['termination'] == 'FINAL_PASS'
        assert reworked['intermediate_checks'] == 1 and reworked['budget']['rollbacks'] == 1
        assert reworked['budget']['generated_tokens'] == 10 and reworked['budget']['revoked_tokens'] == 3
        assert 'Step 1: x.' in reworked['submitted_text'] and reworked['submitted_text'].count('Step 2:') == 1
        assert old.decode(tokenizer, reworked['submitted_reasoning_ids']).count('Step 2:') == 2
        assert reworked['timing']['T_request'] == 23.
    checks.append('full two-path loop starts fresh even at same position; prompt timed/gold excluded; terminal budget failure Y=0')
    checks.append('complete terminal FAIL/crop/rework/PASS preserves scored tail, raw crossing token, logical markers and final Math-Verify')
    return dict(status='PASS', count=len(checks), checks=checks)


def gpu_checks(cfg, run, row, selection, model, tokenizer, verifier):
    """Pilot-only real streaming, pause/PASS and forced crop checks; no scientific Y."""
    import torch
    target = max(selection['positions'].values(), key=lambda p: p['step_number'])
    state = initial_state(row, selection, 'HIGH', {})
    session = Session(model, state['prompt_ids'])
    try:
        boundary = None
        while boundary is None:
            reason = budget(cfg, state, generation_tokens=1, context_tokens=len(session.ids) + 1)
            if reason:
                raise RuntimeError('Pilot prefix generation stopped: ' + reason)
            token = session.sample()
            i = len(state['reasoning_ids'])
            reference = selection['generated_ids']
            if i >= len(reference) or token != reference[i]:
                raise ReferenceMismatch(dict(unique_id=row['unique_id'], token_index=i,
                                             expected=reference[i] if i < len(reference) else None,
                                             actual=token, gpu_check=True))
            state['reasoning_ids'].append(token)
            state['reasoning_positions'].append(len(session.ids) - 1)
            state['context_ids'] = session.ids.copy()
            state['budget']['generated_tokens'] += 1
            state['budget']['attempt_tokens'] += 1
            if token in eos_tokens(model):
                raise RuntimeError('Pilot EOS before selected closed step')
            boundary = closed_boundary(tokenizer, state['reasoning_ids'], target['step_number'])
        if boundary['prefix_ids'] != target['prefix_ids']:
            raise ReferenceMismatch(dict(unique_id=row['unique_id'], reason='PILOT_ENDPOINT_MISMATCH'))
        continuous, paused = session.clone(), session.clone()
        pause_state = copy.deepcopy(state)
        pause_state['selected_position'] = copy.deepcopy(target)
        cache_before = paused.cache.get_seq_length()
        # Force PASS in control state, but actually run the resident PRM as the pause.
        ids, numbers, prior = verifier.input_ids(row['problem'], '', boundary['prm_text'])
        reason = budget(cfg, pause_state, prm_input_tokens=len(ids), prm_calls=1)
        if reason or not ids:
            raise RuntimeError('Pilot pause PRM input unavailable: ' + str(reason))
        natural_result = legacy.verify_prm(verifier, ids, numbers, prior)
        assert check(cfg, pause_state, paused, verifier, tokenizer, len(boundary['prefix_ids']),
                     False, prm_text=boundary['prm_text'], forced=dict(verdict='PASS')) is None
        assert paused.cache.get_seq_length() == cache_before and paused.ids == continuous.ids
        expected, actual = [], []
        for _ in range(32):
            if len(continuous.ids) >= cfg['max_total_tokens']:
                break
            token = continuous.sample()
            expected.append(token)
            actual.append(paused.sample())
            if token in eos_tokens(model):
                break
        if expected != actual:
            raise RuntimeError('Pilot retained-KV pause changed continuation tokens')
        del continuous, paused
        forced = copy.deepcopy(state)
        forced_session = session.clone()
        original_generated = forced['budget']['generated_tokens']
        assert check(cfg, forced, forced_session, verifier, tokenizer, len(boundary['prefix_ids']),
                     False, prm_text=boundary['prm_text'],
                     forced=dict(verdict='FAIL', first_error_step=1,
                                 diagnosis='SYNTHETIC SELF-CHECK forced FAIL.', hint='Restart.')) is None
        event = forced['events'][-1]
        assert event['kind'] == 'ROLLBACK' and forced['checkpoint_step'] == 0
        assert forced_session.ids == state['prompt_ids'] + forced['feedback_tokens'][-1]
        assert forced_session.cache.get_seq_length() == len(forced_session.ids)
        assert event['feedback_prefill_tokens'] == len(forced['feedback_tokens'][-1])
        assert forced['budget']['generated_tokens'] == original_generated
        assert forced['budget']['revoked_tokens'] == original_generated
        assert forced['budget']['attempt_tokens'] == 0 and forced['intermediate_checks'] == 1
        del forced_session
        # Also exercise a real crop that preserves an accepted generated prefix.
        accepted = copy.deepcopy(pause_state)
        accepted_session = session.clone()
        assert check(cfg, accepted, accepted_session, verifier, tokenizer, len(accepted['checkpoint_reasoning_ids']),
                     True, forced=dict(verdict='FAIL', first_error_step=target['step_number'])) is None
        assert accepted['checks'][-1]['checkpoint_conflict']
        assert accepted_session.ids[:len(accepted['checkpoint_ids'])] == accepted['checkpoint_ids']
        assert accepted_session.cache.get_seq_length() == len(accepted_session.ids)
        assert accepted['checkpoint_text'] == boundary['prm_text']
        assert reasoning_text(accepted, tokenizer) == boundary['prm_text']
        assert old.decode(tokenizer, accepted['checkpoint_reasoning_ids']).startswith(boundary['prm_text'])
        result = dict(status='PASS', unique_id=row['unique_id'], source_hashes=source_hashes(),
                      selected_step=target['step_number'], reference_prefix_match=True,
                      generated_before_check=state['reasoning_ids'], pending_ids=boundary['pending_ids'],
                      pause=dict(token_equal=True, continuous_ids=expected, paused_ids=actual,
                                 natural_prm=natural_result, control_pass_synthetic=True),
                      forced_fail=dict(synthetic=True, excluded_from_statistics=True,
                                       no_pass_state=forced, accepted_prefix_state=accepted), created=old.now())
        atomic_json(Path(run) / 'pilot_checks' / f'{legacy.key(row["unique_id"])}.json', result)
        del accepted_session
        return result
    finally:
        del session
        gc.collect()
        torch.cuda.empty_cache()
