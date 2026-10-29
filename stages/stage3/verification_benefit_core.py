"""Pure CPU rules for the Stage3 verification-benefit experiment."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

LIMIT = 65536
THRESHOLDS = (8192, 16384, 32768)


def paired_seed(seed, uid, count):
    return int.from_bytes(hashlib.sha256(f'{seed}:{uid}:{count}'.encode()).digest()[:4], 'big') & 0x7fffffff


def choose_checkpoints(record, seed):
    """Use only saved, complete, non-answer aligned Thinking paragraph endpoints."""
    ids = record['generated_ids']
    bounds = record['boundaries']
    if bounds.get('mode') != 'thinking_paragraphs' or bounds.get('format_errors'):
        return {'thresholds': {str(t): {'reason': 'INVALID_BOUNDARIES'} for t in THRESHOLDS}, 'checkpoints': []}
    eligible = [s for s in bounds['steps'] if 0 < s['prefix_token_count'] < LIMIT
                and s['prefix_token_count'] <= len(ids)
                and not s.get('boundary_aligned_back', False)
                and s.get('aligned_char_end') == s['char_end']]
    chosen, thresholds = {}, {}
    for t in THRESHOLDS:
        options = [s for s in eligible if s['prefix_token_count'] >= t]
        if not options:
            thresholds[str(t)] = {'reason': 'ENDED_BEFORE_THRESHOLD' if len(ids) < t else 'NO_ELIGIBLE_CHECKPOINT'}
            continue
        point = min(options, key=lambda s: s['prefix_token_count'])
        count = point['prefix_token_count']
        thresholds[str(t)] = {'prefix_token_count': count}
        if count not in chosen:
            chosen[count] = {'prefix_token_count': count, 'step_number': point['step_number'],
                             'thresholds': [], 'pair_seed': paired_seed(seed, record['unique_id'], count),
                             'probe_B_score': None}
        chosen[count]['thresholds'].append(t)
    return {'thresholds': thresholds, 'checkpoints': list(chosen.values())}


def arm_rule(prefix_count, verdict):
    if type(prefix_count) is not int or not 0 < prefix_count < LIMIT:
        raise ValueError('Invalid checkpoint budget')
    if verdict not in ('NO_CHECK', 'PASS', 'FAIL', 'UNCERTAIN'):
        raise ValueError('Invalid verdict')
    return {'restart': verdict == 'FAIL', 'budget': LIMIT if verdict == 'FAIL' else LIMIT-prefix_count,
            'revoked_tokens': prefix_count if verdict == 'FAIL' else 0}


def token_accounting(first_count, prefix_count, verdict, generated_count):
    rule = arm_rule(prefix_count, verdict)
    if not 0 <= first_count <= LIMIT or not 0 <= generated_count <= rule['budget'] or prefix_count > first_count:
        raise ValueError('Output budget or prefix invalid')
    answer_count = generated_count if rule['restart'] else prefix_count + generated_count
    combined = first_count + answer_count
    return {'first_round_output_tokens': first_count, 'second_round_new_tokens': generated_count,
            'second_round_answer_tokens': answer_count, 'answer_length_sum': combined,
            'answer_length_sum_exceeds_limit': combined > LIMIT,
            'actual_4b_generated_tokens': first_count + generated_count,
            'revoked_tokens': rule['revoked_tokens']}


def summarize(pairs, bootstrap=1000, seed=20261013):
    """Question-weighted paired means and question-bootstrap interval."""
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    all_pairs = defaultdict(list)
    by_question = defaultdict(list)
    for pair in pairs:
        a, b = pair['A'], pair['B']
        all_pairs[pair['unique_id']].append(pair)
        if a['label'] is None or b['label'] is None:
            continue
        by_question[pair['unique_id']].append((int(a['label'] == 1 and b['label'] == 0),
            int(a['label'] == 0 and b['label'] == 1), int(a['label'] == 1)-int(b['label'] == 1)))
    questions = list(by_question.values())
    def estimate(values):
        if not values:
            return {'mean': None, 'interval': None}
        rng = random.Random(seed)
        draws = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(bootstrap))
        return {'mean': sum(values)/len(values),
                'interval': [draws[int(.025*(bootstrap-1))], draws[int(.975*(bootstrap-1))]]}
    means = [sum(x[2] for x in rows)/len(rows) for rows in questions]
    paired = estimate(means)
    result = {'paired_questions': len(questions), 'paired_checkpoints': sum(map(len, questions)),
            'all_completed_questions': len(all_pairs), 'all_completed_checkpoints': len(pairs),
            'unscorable_checkpoints': len(pairs)-sum(map(len, questions)),
            'a_wrong_b_right_checkpoints': sum(x[0] for rows in questions for x in rows),
            'a_right_b_wrong_checkpoints': sum(x[1] for rows in questions for x in rows),
            'accuracy_difference': paired['mean'], 'interval': paired['interval']}
    if questions:
        for arm in ('A', 'B'):
            scores = []
            for rows in all_pairs.values():
                valid = [p for p in rows if p['A']['label'] is not None and p['B']['label'] is not None]
                if valid:
                    scores.append(sum(p[arm]['label'] == 0 for p in valid)/len(valid))
            result[f'{arm}_accuracy'] = estimate(scores)
    for field, getter in {
        'new_generation_tokens': lambda x: x['tokens']['second_round_new_tokens'],
        'answer_length_sum': lambda x: x['tokens']['answer_length_sum'],
        'answer_length_sum_exceeds_limit': lambda x: x['tokens']['answer_length_sum_exceeds_limit'],
        'recomputed_input_tokens': lambda x: x['recomputed_input_tokens'],
        'verifier_input_tokens': lambda x: x['verifier_input_tokens'],
        'verifier_output_tokens': lambda x: x['verifier_output_tokens'],
        'second_round_forward_tokens': lambda x: (x['recomputed_input_tokens']
            + x['tokens']['second_round_new_tokens'] + x['verifier_input_tokens'] + x['verifier_output_tokens']),
        'branch_seconds': lambda x: x['branch_seconds'],
    }.items():
        values, a_values, b_values, paired_count = [], [], [], 0
        for rows in all_pairs.values():
            differences, aval, bval = [], [], []
            for pair in rows:
                try:
                    a, b = getter(pair['A']), getter(pair['B'])
                except KeyError:
                    continue
                if a is not None and b is not None:
                    differences.append(float(b)-float(a))
                    aval.append(float(a))
                    bval.append(float(b))
            if differences:
                paired_count += len(differences)
                values.append(sum(differences)/len(differences))
                a_values.append(sum(aval)/len(aval))
                b_values.append(sum(bval)/len(bval))
        result[field+'_B_minus_A'] = {**estimate(values), 'questions':len(values), 'checkpoints':paired_count}
        result[field+'_A'] = estimate(a_values)
        result[field+'_B'] = estimate(b_values)
    return result
