"""Prepare, execute, and analyze a new Stage3 verification-benefit run."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, digest, read_json, stable_hash
from veriserve_research.inference import GenerativeVerifier, generation_seed, prompt_tokens
from veriserve_research.intervention.protocol import feedback_text, parse_verdict
from veriserve_research.trajectory import boundaries, decode, grade, verification_units
from stages.stage3.verification_benefit_core import LIMIT, arm_rule, choose_checkpoints, summarize, token_accounting

CONFIG = ROOT/'stages/stage3/verification_benefit_config.json'
SOURCE_FILES = [Path(__file__), ROOT/'stages/stage3/verification_benefit_core.py',
                ROOT/'stages/stage3/verification_benefit_logits.py', ROOT/'stages/stage3/verification_benefit_worker.py',
                ROOT/'veriserve_research/trajectory.py', ROOT/'veriserve_research/inference.py',
                ROOT/'veriserve_research/intervention/protocol.py', ROOT/'veriserve_research/artifacts/io.py',
                ROOT/'veriserve_research/artifacts/legacy.py', ROOT/'environments/qwen35/uv.lock',
                ROOT/'environments/vllm/requirements.freeze.txt']
VLLM_PYTHON = '/root/veriserve-vllm-env/bin/python'


def identity(cfg):
    manifest = ROOT/cfg['question_manifest']
    return {'config': cfg, 'question_manifest_sha256': digest(manifest),
            'sources': {str(p.relative_to(ROOT)): digest(p) for p in SOURCE_FILES}}


def prepare(config_path=CONFIG):
    cfg = read_json(config_path)
    original = read_json(ROOT/cfg['question_manifest'])
    train = original['splits']['train']
    test = {r['unique_id'] for r in original['splits']['test']}
    audit = original['splits']['verifier_audit']
    if (len(train) != 100 or len(audit) != 20 or len({r['unique_id'] for r in train}) != 100
            or test & {r['unique_id'] for r in train[:25]}
            or {r['unique_id'] for r in audit} & (test | {r['unique_id'] for r in train})):
        raise ValueError('Source splits are invalid or overlap')
    frozen = identity(cfg)
    run = ROOT/'stages/stage3/runs'/('verification-benefit-'+stable_hash(frozen)[:12])
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run/'manifest.json'
    manifest = {'identity': frozen, 'splits': {'pilot': train[:5], 'exploratory': train[5:25], 'verifier_audit': audit},
                'generator_prompt_config': {k: original['config'][k] for k in
                    ('system_prompt', 'prompt', 'format_examples', 'enable_thinking')},
                'status': 'PREPARED_NO_GENERATION'}
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError('Frozen run identity differs; do not overwrite')
    else:
        atomic_json(manifest_path, manifest)
    return run, cfg, manifest


def snapshots(cfg, kind):
    model = cfg['generator_model'] if kind == 'generator' else cfg['verifier_model']
    revision = cfg['generator_revision'] if kind == 'generator' else cfg['verifier_revision']
    path = Path(cfg['model_cache_dir'])/('models--'+model.replace('/', '--'))/'snapshots'/revision
    if not path.is_dir():
        raise FileNotFoundError(f'Model snapshot is missing: {path}')
    return path


def load_tokenizer(cfg, kind):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshots(cfg, kind)), local_files_only=True)


def eos_ids(tokenizer, snapshot):
    model_config = read_json(snapshot/'config.json')
    token = model_config.get('text_config', {}).get('eos_token_id', model_config.get('eos_token_id'))
    values = token if isinstance(token, list) else [token]
    values += [tokenizer.eos_token_id, tokenizer.get_vocab().get('<|im_end|>')]
    return list(dict.fromkeys(int(x) for x in values if x is not None))


def worker_call(run, cfg, kind, input_ids, history, budget, seed, tokenizer, key):
    snapshot = snapshots(cfg, kind)
    maximum = cfg['max_model_len'] if kind == 'generator' else cfg['max_verifier_model_len']
    if budget <= 0 or len(input_ids)+budget > maximum:
        raise ValueError(f'{kind} context or output limit exceeded')
    request = {'engine': {'snapshot': str(snapshot), 'max_model_len': maximum,
               'gpu_memory_utilization': cfg['gpu_memory_utilization'],
               'prefill_chunk_size': cfg[kind+'_prefill_chunk_size']},
               'sampling': cfg[kind+'_sampling'], 'input_ids': input_ids, 'presence_history': history,
               'max_tokens': budget, 'seed': seed, 'eos_ids': eos_ids(tokenizer, snapshot)}
    directory = run/'worker'/key
    directory.mkdir(parents=True, exist_ok=True)
    req, result_path = directory/'request.json', directory/'result.json'
    if result_path.exists():
        if read_json(req) != request:
            raise ValueError('Saved request differs; preserve and investigate')
        saved = read_json(result_path)
        if 'worker_wall_seconds' not in saved:
            raise RuntimeError('Worker completed without a recorded end-to-end time; retain result and inspect')
        return {**saved, '_reused_after_interruption': True}
    atomic_json(req, request)
    started = time.perf_counter()
    with (directory/'worker.log').open('a') as log:
        completed = subprocess.run([VLLM_PYTHON, '-u', '-m', 'stages.stage3.verification_benefit_worker',
                                    str(req), str(result_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'VLLM_WORKER_MULTIPROC_METHOD':'spawn'}, timeout=14400)
    if completed.returncode or not result_path.exists():
        raise RuntimeError(f'vLLM worker failed; see {directory}/worker.log')
    result = read_json(result_path)
    result['worker_wall_seconds'] = time.perf_counter()-started
    ids = result['generated_ids']
    if len(ids) > budget or (result['finish_reason'] == 'stop' and (not ids or ids[-1] not in request['eos_ids'])):
        raise ValueError('Invalid output budget or EOS; retained worker result')
    atomic_json(result_path, result)
    return {**result, '_reused_after_interruption': False}


def completion(result):
    return 'EOS' if result['finish_reason'] == 'stop' else 'LENGTH_TRUNCATED'


def record_path(run, group, uid):
    return run/group/(stable_hash(uid)[:20]+'.json')


def first_round(run, cfg, manifest, group, row, tokenizer):
    path = record_path(run, group, row['unique_id'])
    if path.exists():
        result = read_json(path)
        if result['token_sha256'] != stable_hash({'prompt_ids':result['prompt_ids'], 'generated_ids':result['generated_ids']}):
            raise ValueError('First-round token checksum mismatch')
        if result['selection'] != choose_checkpoints(result, cfg['seed']):
            raise ValueError('Saved selection differs from the frozen first-round boundaries')
        return result
    prompt = prompt_tokens(manifest['generator_prompt_config'], tokenizer, row['problem'])
    generated = worker_call(run, cfg, 'generator', prompt, [], LIMIT,
                            generation_seed(cfg['seed'], row['unique_id']), tokenizer, group+'-'+stable_hash(row['unique_id'])[:16]+'-first')
    ids = generated['generated_ids']
    raw = decode(tokenizer, ids)
    _, bounds = boundaries(tokenizer, ids, 'thinking_paragraphs')
    score = grade(raw, row['answer'], completion(generated), 5, 'thinking_paragraphs', 'explanation_then_final_line')
    result = {'unique_id': row['unique_id'], 'prompt_ids': prompt, 'generated_ids': ids, 'raw_text': raw,
              'completion': completion(generated), 'grade': score, 'boundaries': bounds,
              'engine': generated, 'token_sha256': stable_hash({'prompt_ids':prompt, 'generated_ids':ids})}
    result['selection'] = choose_checkpoints(result, cfg['seed'])
    atomic_json(path, result)
    return result


def branch(run, cfg, group, row, first, point, arm, generator_tokenizer, verifier_tokenizer):
    uid, n = row['unique_id'], point['prefix_token_count']
    key = group+'-'+stable_hash(uid)[:16]+'-'+str(n)+'-'+arm
    path = run/'branches'/(key+'.json')
    if path.exists():
        result = read_json(path)
        checksum = result.pop('record_sha256', None)
        if (checksum != stable_hash(result) or result['first_token_sha256'] != first['token_sha256']
                or result['checkpoint'] != point or result['unique_id'] != uid or result['arm'] != arm):
            raise ValueError('Saved branch does not match first-round tokens')
        result['record_sha256'] = checksum
        return result
    prefix = first['generated_ids'][:n]
    verdict, verifier_output, feedback = 'NO_CHECK', None, []
    start = time.perf_counter()
    if arm == 'B':
        prefix_text = decode(generator_tokenizer, prefix)
        numbers, _ = verification_units(prefix_text, 'thinking_paragraphs')
        if not numbers:
            raise ValueError('Selected checkpoint has no verifier units')
        verifier = object.__new__(GenerativeVerifier)
        verifier.tokenizer = verifier_tokenizer
        verifier.cfg = {'boundary_mode':'thinking_paragraphs', 'verifier':{'enable_thinking':False}}
        verifier_ids = verifier.input_ids(row['problem'], prefix_text, numbers[-1], [], 'intermediate')
        if len(verifier_ids) > cfg['max_verifier_input_tokens'] or len(verifier_ids)+cfg['max_verifier_output_tokens'] > cfg['max_verifier_model_len']:
            checked = parse_verdict('', numbers, 0, 'INPUT_LIMIT', False)
            verifier_output = {'input_ids':verifier_ids, 'generated_ids':[], 'result':checked, 'engine':None}
        else:
            vout = worker_call(run, cfg, 'verifier', verifier_ids, [], cfg['max_verifier_output_tokens'],
                               generation_seed(cfg['seed'], key+'-verifier'), verifier_tokenizer, key+'-verifier')
            raw = decode(verifier_tokenizer, vout['generated_ids'])
            # Remove the end-of-turn token only; all other output must satisfy strict JSON parsing.
            terminal = verifier_tokenizer.convert_ids_to_tokens(vout['generated_ids'][-1]) if completion(vout) == 'EOS' else ''
            clean = raw[:-len(terminal)] if terminal and raw.endswith(terminal) else raw
            checked = parse_verdict(clean, numbers, 0, completion(vout), False)
            verifier_output = {'input_ids':verifier_ids, 'generated_ids':vout['generated_ids'],
                               'raw_text':raw, 'result':checked, 'engine':vout}
        verdict = checked['effective_verdict']
        if verdict == 'FAIL':
            text = feedback_text(checked['diagnosis'], checked['hint'], 0, 'thinking_paragraphs')
            feedback = generator_tokenizer.encode(text, add_special_tokens=False)
    rule = arm_rule(n, verdict)
    input_ids = first['prompt_ids'] + (feedback if rule['restart'] else prefix)
    history = feedback if rule['restart'] else prefix
    generated = worker_call(run, cfg, 'generator', input_ids, history, rule['budget'], point['pair_seed'],
                            generator_tokenizer, key+'-generator')
    output = generated['generated_ids']
    timing_complete = not generated['_reused_after_interruption'] and not (
        verifier_output and verifier_output['engine'] and verifier_output['engine']['_reused_after_interruption'])
    branch_seconds = time.perf_counter()-start if timing_complete else None
    answer_ids = output if rule['restart'] else prefix+output
    raw_answer = decode(generator_tokenizer, answer_ids)
    score = grade(raw_answer, row['answer'], completion(generated), 5, 'thinking_paragraphs', 'explanation_then_final_line')
    result = {'unique_id':uid, 'arm':arm, 'checkpoint':point, 'first_token_sha256':first['token_sha256'],
              'verdict':verdict, 'verification':verifier_output, 'feedback_ids':feedback,
              'input_ids':input_ids, 'generated_ids':output, 'answer_ids':answer_ids, 'raw_answer':raw_answer,
              'completion':completion(generated), 'grade':score, 'label':score.get('label'),
              'tokens':token_accounting(len(first['generated_ids']), n, verdict, len(output)),
              'recomputed_input_tokens':len(input_ids), 'verifier_input_tokens':len(verifier_output['input_ids']) if verifier_output else 0,
              'verifier_output_tokens':len(verifier_output['generated_ids']) if verifier_output else 0,
              'engine':generated, 'branch_seconds':branch_seconds,
              'branch_time_excludes_scoring_and_persistence':True,
              'timing_incomplete_after_interruption':not timing_complete}
    result['record_sha256'] = stable_hash(result)
    atomic_json(path, result)
    return result


def require_audit(run, cfg):
    path = run/'verifier_audit_gate.json'
    gate = read_json(path, {})
    if (gate.get('status') != 'PASS' or gate.get('verifier_revision') != cfg['verifier_revision']
            or gate.get('engine') != 'vllm' or gate.get('mode') != 'direct'
            or gate.get('generator_revision') != cfg['generator_revision']
            or gate.get('verifier_sampling') != cfg['verifier_sampling']
            or gate.get('verifier_output_limit') != cfg['max_verifier_output_tokens']
            or gate.get('config_hash') != stable_hash(cfg)
            or not gate.get('labels_sha256')
            or not gate.get('audit_report_sha256')):
        raise RuntimeError('A matching independent-question verifier quality audit is required before pilot; see Stage3 README')
    report_path = run/'verifier_audit_report.json'
    labels_path = run/'verifier_audit_labels.json'
    if (not report_path.is_file() or not labels_path.is_file()
            or digest(report_path) != gate['audit_report_sha256']
            or digest(labels_path) != gate['labels_sha256']):
        raise RuntimeError('Audit report or reference labels are missing or mismatched')
    report = read_json(report_path)
    metrics = report.get('metrics', {})
    if (report.get('status') != 'PASS' or report.get('split') != 'verifier_audit'
            or report.get('labels_sha256') != digest(labels_path)
            or not report.get('collection_complete')
            or report.get('correct_prefixes', 0) < 20 or report.get('incorrect_prefixes', 0) < 20
            or any(metrics.get(k, {}).get('value') is None or metrics[k]['value'] > .2 for k in
                   ('incorrect_pass_rate', 'correct_fail_rate', 'uncertain_or_format_or_truncated_rate'))
            or metrics.get('balanced_accuracy', {}).get('value') is None
            or metrics['balanced_accuracy']['value'] < .8):
        raise RuntimeError('Independent verifier quality audit does not satisfy Stage3 thresholds')


def backup(run, label):
    """Stage only this experiment; a failed ordinary push stops collection."""
    def git(*args):
        value = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, 'GIT_TERMINAL_PROMPT':'0'}, timeout=600)
        if value.returncode:
            raise RuntimeError('Git backup failed; local files retained; see repository transport configuration')
        return value.stdout.strip()
    if git('branch', '--show-current') != 'experiment/step-hidden-probe':
        raise RuntimeError('Backup requires the existing experiment/step-hidden-probe branch')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Unrelated staged changes exist; refusing to commit them')
    if git('diff', '--name-only', '--', *(str(p.relative_to(ROOT)) for p in SOURCE_FILES[4:])):
        raise RuntimeError('A shared dependency has uncommitted changes; refusing an incomplete backup')
    git('lfs', 'version')
    paths = [str(p.relative_to(ROOT)) for p in SOURCE_FILES[:4]] + [
        'stages/stage3/verification_benefit_config.json', 'stages/stage3/verification_benefit_plan.md',
        'stages/stage3/README.md', 'tests/test_verification_benefit_cpu.py',
        'tests/test_verification_benefit_logits.py', str(run.relative_to(ROOT))]
    registry = ROOT/'stages/stage2/runs/v2-20261013-e136a41fea81/verifier_audit_reservations.json'
    if registry.exists():
        paths.append(str(registry.relative_to(ROOT)))
    git('add', '--', *paths)
    if git('diff', '--cached', '--name-only', '--', *paths):
        git('commit', '--only', '-m', f'Stage3 verification benefit: {label}', '--', *paths)
    commit = git('rev-parse', 'HEAD')
    git('lfs', 'push', 'origin', 'HEAD')
    git('push', '--porcelain', 'origin', 'HEAD:refs/heads/experiment/step-hidden-probe')
    remote = git('ls-remote', 'origin', 'refs/heads/experiment/step-hidden-probe').split()
    if not remote or remote[0] != commit:
        raise RuntimeError('Remote backup commit mismatch; stop collection')


def reserve_audit(run, cfg, manifest, extra):
    """Reserve fixed original audit questions then unused odd-index questions, without reuse."""
    if extra < 0:
        raise ValueError('audit-extra must be nonnegative')
    if (run/'verifier_audit_gate.json').exists():
        raise ValueError('A passed audit is frozen and cannot be expanded')
    source_path = ROOT/cfg['question_manifest']
    source = read_json(source_path)
    stage2 = source_path.parent
    dataset_path = stage2/'omni_math_rule.jsonl'
    if digest(dataset_path) != source['dataset_integrity']['sha256']:
        raise ValueError('Audit dataset bytes differ from source manifest')
    lookup = {r['unique_id']:r for r in source['splits']['verifier_audit']}
    for line in dataset_path.read_text().splitlines():
        row = json.loads(line)
        uid = hashlib.sha256(row['problem'].encode()).hexdigest()
        lookup.setdefault(uid, {**row, 'unique_id':uid})
    pool = [r['unique_id'] for r in source['splits']['verifier_audit']] + source['unused_ids'][1::2]
    registry_path = stage2/'verifier_audit_reservations.json'
    lock_path = ROOT/'.cache'/'stage3-verifier-audit-reservations.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read_json(registry_path, {})
        key = stable_hash(manifest['identity'])
        reserved = registry.get(key, [])
        occupied = {uid for k, ids in registry.items() if k != key for uid in ids}
        if len(reserved) > 20+extra:
            raise ValueError('Reserved audit cohort cannot shrink; use the original audit-extra value')
        available = [uid for uid in pool if uid not in occupied and uid not in reserved]
        missing = 20+extra-len(reserved)
        if len(available) < missing or any(uid not in lookup for uid in reserved+available[:missing]):
            raise ValueError('Not enough independent unused audit questions')
        reserved = reserved + available[:missing]
        registry[key] = reserved
        atomic_json(registry_path, registry)
    rows = [lookup[uid] for uid in reserved]
    current = run/'verifier_audit_cohort.json'
    if current.exists():
        old = read_json(current)
        if [r['unique_id'] for r in rows[:len(old)]] != [r['unique_id'] for r in old]:
            raise ValueError('Reserved audit cohort changed; retain prior records')
    atomic_json(current, rows)
    return rows


def collect_audit(run, cfg, manifest, extra=0):
    """Collect fixed independent questions and blind labeling inputs."""
    cohort = reserve_audit(run, cfg, manifest, extra)
    backup(run, 'verifier-audit-before-start')
    generator_tokenizer = load_tokenizer(cfg, 'generator')
    verifier_tokenizer = load_tokenizer(cfg, 'verifier')
    review = []
    for row in cohort:
        first = first_round(run, cfg, manifest, 'verifier_audit', row, generator_tokenizer)
        steps = first['boundaries'].get('steps', [])
        counts = []
        for fraction in (.25, .75):
            if steps:
                target = fraction * len(first['generated_ids'])
                counts.append((min(steps, key=lambda x:abs(x['prefix_token_count']-target))['prefix_token_count'], 'intermediate'))
        if first['completion'] == 'EOS' and first['boundaries'].get('end'):
            counts.append((first['boundaries']['end']['prefix_token_count'], 'terminal'))
        # The terminal kind overrides an intermediate kind at the same endpoint.
        for index, (n, kind) in enumerate(dict(counts).items()):
            prefix_ids = first['generated_ids'][:n]
            text = decode(generator_tokenizer, prefix_ids)
            numbers, units = verification_units(text, 'thinking_paragraphs', terminal=kind == 'terminal')
            if not numbers:
                continue
            verifier = object.__new__(GenerativeVerifier)
            verifier.tokenizer = verifier_tokenizer
            verifier.cfg = {'boundary_mode':'thinking_paragraphs', 'verifier':{'enable_thinking':False}}
            ids = verifier.input_ids(row['problem'], text, numbers[-1], [], kind)
            review_key = stable_hash([row['unique_id'], first['token_sha256'], n, kind])
            name = 'verifier-audit-'+stable_hash(row['unique_id'])[:16]+'-'+str(n)
            path = run/'audit_checks'/(name+'.json')
            existing = read_json(path) if path.exists() else None
            if existing:
                if (existing.get('input_sha256') != stable_hash(ids) or existing.get('review_key') != review_key
                        or existing.get('reference_token_sha256') != first['token_sha256']
                        or existing.get('check_type') != kind or existing.get('checkpoint_numbers') != numbers):
                    raise ValueError('Saved audit input differs; retained existing record')
                results = existing['results']
            else:
                results = []
                for repeat in range(2 if index == 0 else 1):
                    if len(ids) > cfg['max_verifier_input_tokens'] or len(ids)+cfg['max_verifier_output_tokens'] > cfg['max_verifier_model_len']:
                        result = parse_verdict('', numbers, 0, 'INPUT_LIMIT', False)
                        result.update(input_tokens=len(ids), output_tokens=0, seconds=0.)
                    else:
                        out = worker_call(run, cfg, 'verifier', ids, [], cfg['max_verifier_output_tokens'],
                                          generation_seed(cfg['seed'], name+'-'+str(repeat)), verifier_tokenizer, name+'-'+str(repeat))
                        raw = decode(verifier_tokenizer, out['generated_ids'])
                        terminal = verifier_tokenizer.convert_ids_to_tokens(out['generated_ids'][-1]) if completion(out) == 'EOS' else ''
                        clean = raw[:-len(terminal)] if terminal and raw.endswith(terminal) else raw
                        result = parse_verdict(clean, numbers, 0, completion(out), False)
                        result.update(input_tokens=len(ids), output_tokens=len(out['generated_ids']),
                                      seconds=out['worker_wall_seconds'], raw_text=raw, generated_ids=out['generated_ids'], engine=out)
                    results.append(result)
            item = {'unique_id':row['unique_id'], 'review_key':review_key, 'input_ids':ids,
                    'input_sha256':stable_hash(ids), 'checkpoint_numbers':numbers, 'check_type':kind,
                    'reference_token_sha256':first['token_sha256'], 'prefix_token_count':n, 'results':results}
            item['record_sha256'] = stable_hash(item)
            if path.exists() and read_json(path) != item:
                raise ValueError('Saved audit input or output differs; retained existing record')
            atomic_json(path, item)
            # No verdicts in the labeling template; gold never enters online verifier input.
            review.append({'review_key':review_key, 'unique_id':row['unique_id'], 'problem':row['problem'],
                'quoted_prefix':text, 'checkpoint_units':units, 'check_type':kind,
                'reference_answer':row['answer'], 'reference_solution':row['solution'],
                'label':None, 'first_error_step':None, 'label_source':None, 'labeler':None})
        backup(run, 'verifier-audit-question')
    atomic_json(run/'verifier_audit_review_inputs.json', review)
    atomic_json(run/'verifier_audit_collection.json', {'status':'AWAITING_LABELS',
        'ordered_ids':[r['unique_id'] for r in cohort],
        'checks':len(review), 'review_inputs_sha256':digest(run/'verifier_audit_review_inputs.json'),
        'config_hash':stable_hash(cfg)})
    backup(run, 'verifier-audit-review-export')


def parse_model_label(raw, ended, numbers):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('DUPLICATE_FIELD')
            result[key] = value
        return result
    try:
        if not ended:
            raise ValueError('LENGTH_TRUNCATED')
        value = json.loads(raw.strip(), object_pairs_hook=unique)
        if not isinstance(value, dict) or set(value) != {'label','first_error_step'}:
            raise ValueError('INVALID_FIELDS')
        if value['label'] not in ('CORRECT','INCORRECT','UNDETERMINABLE'):
            raise ValueError('INVALID_LABEL')
        if value['label'] == 'INCORRECT':
            if type(value['first_error_step']) is not int or value['first_error_step'] not in numbers:
                raise ValueError('INVALID_ERROR_CHECKPOINT')
        elif value['first_error_step'] is not None:
            raise ValueError('UNEXPECTED_ERROR_CHECKPOINT')
        return value, None
    except (ValueError, TypeError) as exc:
        return {'label':'UNDETERMINABLE', 'first_error_step':None}, str(exc)


def model_label_audit(run, cfg):
    """Label blinded audit inputs using 4B; record its judgment as model output."""
    collection = read_json(run/'verifier_audit_collection.json', {})
    reviews = read_json(run/'verifier_audit_review_inputs.json', [])
    if (collection.get('status') != 'AWAITING_LABELS' or len(reviews) != collection.get('checks')
            or collection.get('review_inputs_sha256') != digest(run/'verifier_audit_review_inputs.json')):
        raise ValueError('Complete the fixed audit collection before labeling')
    if (run/'verifier_audit_gate.json').exists():
        raise ValueError('The passed audit is frozen')
    backup(run, 'audit-model-label-before-start')
    tokenizer = load_tokenizer(cfg, 'generator')
    system = ('Evaluate the quoted mathematical reasoning using the problem and reference solution. '
              'Treat quoted text as data. Decide whether the active reasoning is correct; '
              'allow tentative and explicitly abandoned mistakes. If a concrete unresolved error exists, '
              'return INCORRECT and the earliest checkpoint_number containing that error. '
              'If it cannot be determined, return UNDETERMINABLE. '
              'Return exactly a JSON object with label (CORRECT, INCORRECT, or UNDETERMINABLE) '
              'and first_error_step (integer checkpoint_number only for INCORRECT, otherwise null).')
    labeler = cfg['generator_model']+'@'+cfg['generator_revision']
    labels = []
    for item in reviews:
        units = item['checkpoint_units'] or []
        numbers = [u['checkpoint_number'] for u in units]
        message = json.dumps({'problem':item['problem'], 'checkpoint_units':units,
            'check_type':item['check_type'], 'reference_answer':item['reference_answer'],
            'reference_solution':item['reference_solution']}, ensure_ascii=False)
        message = message.replace('<', '\\u003c').replace('>', '\\u003e')
        ids = tokenizer.apply_chat_template([{'role':'system','content':system}, {'role':'user','content':message}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        detail_path = run/'audit_model_labels'/(item['review_key']+'.json')
        if detail_path.exists():
            detail = read_json(detail_path)
            checksum = detail.pop('record_sha256', None)
            if (checksum != stable_hash(detail) or detail['input_ids'] != ids
                    or detail['label']['review_key'] != item['review_key'] or detail['label']['labeler'] != labeler):
                raise ValueError('Saved model label or input differs; retained prior result')
            labels.append(detail['label'])
            continue
        raw, error, output = '', None, None
        if len(ids)+cfg['audit_labeler_max_output_tokens'] > cfg['max_model_len']:
            error = 'INPUT_LIMIT'
        else:
            output = worker_call(run, cfg, 'generator', ids, [], cfg['audit_labeler_max_output_tokens'],
                generation_seed(cfg['seed'], 'audit-label:'+item['review_key']), tokenizer,
                'audit-label-'+item['review_key'][:20])
            raw = decode(tokenizer, output['generated_ids'])
            terminal = tokenizer.convert_ids_to_tokens(output['generated_ids'][-1]) if completion(output) == 'EOS' else ''
            clean = raw[:-len(terminal)] if terminal and raw.endswith(terminal) else raw
            value, error = parse_model_label(clean, completion(output) == 'EOS', numbers)
        if error:
            value = {'label':'UNDETERMINABLE', 'first_error_step':None}
        label = {'review_key':item['review_key'], **value, 'label_source':'model', 'labeler':labeler}
        detail = {'label':label, 'input_ids':ids, 'raw_text':raw, 'parse_error':error, 'engine':output}
        detail['record_sha256'] = stable_hash(detail)
        atomic_json(detail_path, detail)
        labels.append(label)
        backup(run, 'audit-model-label')
    atomic_json(run/'verifier_audit_model_labels.json', labels)
    backup(run, 'audit-model-label-complete')
    return labels


def audit_report(run, cfg, manifest, label_file):
    labels = read_json(label_file)
    if not isinstance(labels, list):
        raise ValueError('Reference labels must be a JSON array')
    by_key = {}
    seen = set()
    for label in labels:
        key = label.get('review_key')
        if not key or key in seen:
            raise ValueError('Missing or duplicate review_key')
        seen.add(key)
        if label.get('label') is not None:
            if label.get('label_source') not in ('human', 'model') or not isinstance(label.get('labeler'), str) or not label['labeler'].strip():
                raise ValueError('Each label must record its human/model source and labeler')
            if label.get('label') not in ('CORRECT', 'INCORRECT', 'UNDETERMINABLE'):
                raise ValueError('Invalid reference label')
            by_key[key] = label
    collection = read_json(run/'verifier_audit_collection.json', {})
    records = [read_json(p) for p in sorted((run/'audit_checks').glob('*.json'))]
    cohort = read_json(run/'verifier_audit_cohort.json', [])
    expected_ids = [r['unique_id'] for r in cohort]
    complete = (len(expected_ids) >= 20 and len(expected_ids) == len(set(expected_ids))
                and collection.get('ordered_ids') == expected_ids and collection.get('config_hash') == stable_hash(cfg)
                and collection.get('status') == 'AWAITING_LABELS' and collection.get('checks') == len(records))
    if not complete:
        raise ValueError('Audit collection is incomplete or its identity differs')
    items = []
    for record in records:
        checksum = record.pop('record_sha256', None)
        if (checksum != stable_hash(record) or record['input_sha256'] != stable_hash(record['input_ids'])
                or record['unique_id'] not in expected_ids):
            raise ValueError('Audit input integrity or cohort mismatch')
        label = by_key.get(record['review_key'])
        if label and label['label'] == 'INCORRECT' and (type(label.get('first_error_step')) is not int
                or label['first_error_step'] not in record['checkpoint_numbers']):
            raise ValueError('Incorrect reference label requires a valid first error checkpoint')
        items.append({'unique_id':record['unique_id'], 'reference':label['label'] if label else None, 'result':record['results'][0]})
    if set(by_key)-{r['review_key'] for r in records}:
        raise ValueError('Labels contain unknown review keys')
    def mean(rows, predicate):
        by_question = {}
        for item in rows:
            by_question.setdefault(item['unique_id'], []).append(float(predicate(item)))
        return (sum(sum(v)/len(v) for v in by_question.values())/len(by_question)) if by_question else None
    correct = [x for x in items if x['reference'] == 'CORRECT']
    incorrect = [x for x in items if x['reference'] == 'INCORRECT']
    pass_recall = mean(correct, lambda x:x['result']['verdict'] == 'PASS')
    fail_recall = mean(incorrect, lambda x:x['result']['verdict'] == 'FAIL')
    values = {'incorrect_pass_rate':mean(incorrect, lambda x:x['result']['verdict'] == 'PASS'),
              'correct_fail_rate':mean(correct, lambda x:x['result']['verdict'] == 'FAIL'),
              'uncertain_or_format_or_truncated_rate':mean(items, lambda x:x['result']['verdict'] == 'UNCERTAIN'
                  or bool(x['result'].get('parser_error')) or x['result']['completion'] != 'EOS'),
              'balanced_accuracy':(pass_recall+fail_recall)/2 if pass_recall is not None and fail_recall is not None else None}
    passed = (len(correct) >= 20 and len(incorrect) >= 20 and all(values[k] is not None and values[k] <= .2 for k in
              ('incorrect_pass_rate','correct_fail_rate','uncertain_or_format_or_truncated_rate'))
              and values['balanced_accuracy'] is not None and values['balanced_accuracy'] >= .8)
    target = run/'verifier_audit_labels.json'
    if (run/'verifier_audit_gate.json').exists() and target.exists() and read_json(target) != labels:
        raise ValueError('Passed audit labels are frozen; do not overwrite them')
    atomic_json(run/'audit_label_versions'/(stable_hash(labels)+'.json'), labels)
    atomic_json(target, labels)
    report = {'status':'PASS' if passed else 'INSUFFICIENT_OR_FAILED', 'split':'verifier_audit',
        'collection_complete':complete, 'correct_prefixes':len(correct), 'incorrect_prefixes':len(incorrect),
        'checks':len(items), 'unlabeled':sum(x['reference'] is None for x in items),
        'undeterminable':sum(x['reference'] == 'UNDETERMINABLE' for x in items),
        'label_sources':{kind:sum(label.get('label_source') == kind for label in by_key.values()) for kind in ('human','model')},
        'interpretation':'agreement with supplied reference labels, not independently established ground truth',
        'labels_sha256':digest(target), 'metrics':{k:{'value':v} for k,v in values.items()}}
    report_path = run/'verifier_audit_report.json'
    atomic_json(run/'audit_report_versions'/(stable_hash(report)+'.json'), report)
    atomic_json(report_path, report)
    if passed:
        atomic_json(run/'verifier_audit_gate.json', {'status':'PASS', 'config_hash':stable_hash(cfg),
            'generator_revision':cfg['generator_revision'], 'verifier_revision':cfg['verifier_revision'],
            'verifier_sampling':cfg['verifier_sampling'], 'verifier_output_limit':cfg['max_verifier_output_tokens'],
            'engine':'vllm', 'mode':'direct', 'audit_report_sha256':digest(report_path),
            'labels_sha256':digest(target), 'label_sources':report['label_sources']})
    return report


def execute(run, cfg, manifest, group):
    """Each completed request is saved and backed up before the next starts."""
    require_audit(run, cfg)
    if group == 'exploratory':
        gate = read_json(run/'pilot_acceptance.json', {})
        protocol = run/'protocol_frozen.json'
        pilot_analysis = run/'pilot-analysis.json'
        if (gate.get('status') != 'PASS' or gate.get('config_hash') != stable_hash(cfg)
                or not protocol.is_file() or not pilot_analysis.is_file()
                or gate.get('protocol_sha256') != digest(protocol)
                or gate.get('pilot_analysis_sha256') != digest(pilot_analysis)):
            raise RuntimeError('Exploratory collection requires the validated pilot acceptance record')
    backup(run, group+'-before-start')
    generator_tokenizer = load_tokenizer(cfg, 'generator')
    verifier_tokenizer = load_tokenizer(cfg, 'verifier')
    for row in manifest['splits'][group]:
        first = first_round(run, cfg, manifest, group, row, generator_tokenizer)
        backup(run, group+'-first-round')
        for point in first['selection']['checkpoints']:
            for arm in ('A','B'):
                branch(run, cfg, group, row, first, point, arm, generator_tokenizer, verifier_tokenizer)
                backup(run, group+'-'+arm)
    result = analyze(run, group, manifest)
    backup(run, group+'-analysis')
    return result


def capacity_check(run, cfg, manifest):
    """Explicit GPU diagnostic: one token at each model's configured full context."""
    backup(run, 'capacity-before-start')
    results = {}
    for kind in ('generator', 'verifier'):
        tokenizer = load_tokenizer(cfg, kind)
        max_length = cfg['max_model_len'] if kind == 'generator' else cfg['max_verifier_model_len']
        if kind == 'generator':
            prompt = prompt_tokens(manifest['generator_prompt_config'], tokenizer, manifest['splits']['pilot'][0]['problem'])
        else:
            prompt = tokenizer.encode('Capacity diagnostic. ', add_special_tokens=False)
        filler = tokenizer.encode(' x', add_special_tokens=False)
        if not filler or len(prompt) >= max_length:
            raise ValueError('Invalid synthetic capacity diagnostic prompt')
        ids = (prompt + (filler * (max_length//len(filler)+1)))[:max_length-1]
        out = worker_call(run, cfg, kind, ids, [], 1, cfg['seed'], tokenizer, 'capacity-'+kind)
        results[kind] = {'input_tokens':len(ids), 'output_tokens':len(out['generated_ids']), 'engine':out}
        if (len(out['generated_ids']) != 1 or not out.get('device_total_bytes')
                or out['device_total_bytes'] > 32*1024**3):
            raise ValueError('Capacity diagnostic did not complete a real GPU token')
        backup(run, 'capacity-'+kind)
    atomic_json(run/'capacity_check.json', {'status':'PASS', 'config_hash':stable_hash(cfg),
        'scope':'synthetic full-context single-token generation; not a substitute for real pilot', 'models':results})
    backup(run, 'capacity-complete')


def forced_check(run, cfg, manifest):
    """Explicit GPU diagnostic for both original-prefix and prompt-plus-feedback inputs."""
    require_audit(run, cfg)
    backup(run, 'forced-check-before-start')
    tokenizer = load_tokenizer(cfg, 'generator')
    selected = None
    for row in manifest['splits']['pilot']:
        path = record_path(run, 'pilot', row['unique_id'])
        if path.exists():
            first = read_json(path)
            if first['selection']['checkpoints']:
                selected = (row, first, first['selection']['checkpoints'][0])
                break
    if not selected:
        raise ValueError('Pilot has no saved eligible checkpoint for the forced GPU diagnostic')
    row, first, point = selected
    n = point['prefix_token_count']
    prefix = first['generated_ids'][:n]
    feedback = tokenizer.encode(feedback_text('Synthetic diagnostic.', 'Recheck the reasoning.', 0, 'thinking_paragraphs'),
                                add_special_tokens=False)
    checks = {}
    for name, ids, history in [('continue', first['prompt_ids']+prefix, prefix),
                               ('restart', first['prompt_ids']+feedback, feedback)]:
        out = worker_call(run, cfg, 'generator', ids, history, min(64, LIMIT-n) if name == 'continue' else 64,
                          point['pair_seed'], tokenizer, 'forced-'+name)
        if not out['generated_ids'] or not out.get('device_total_bytes'):
            raise ValueError('Forced GPU diagnostic failed to generate')
        checks[name] = {'input_ids':ids, 'history_ids':history, 'engine':out}
        backup(run, 'forced-check-'+name)
    atomic_json(run/'forced_check.json', {'status':'PASS', 'config_hash':stable_hash(cfg),
        'unique_id':row['unique_id'], 'first_token_sha256':first['token_sha256'],
        'checkpoint':point, 'feedback_ids':feedback, 'checks':checks,
        'scope':'synthetic forced continuation/restart; excluded from experiment outcomes'})
    backup(run, 'forced-check-complete')


def certify_pilot(run, cfg, manifest):
    require_audit(run, cfg)
    capacity = read_json(run/'capacity_check.json', {})
    forced = read_json(run/'forced_check.json', {})
    if any(x.get('status') != 'PASS' or x.get('config_hash') != stable_hash(cfg) for x in (capacity, forced)):
        raise ValueError('Full-context synthetic capacity and forced GPU continuation/restart checks are required')
    for kind in ('generator','verifier'):
        item = capacity.get('models', {}).get(kind, {})
        maximum = cfg['max_model_len'] if kind == 'generator' else cfg['max_verifier_model_len']
        engine = item.get('engine', {})
        if (item.get('input_tokens') != maximum-1 or item.get('output_tokens') != 1
                or not engine.get('device_total_bytes') or engine['device_total_bytes'] > 32*1024**3
                or not engine.get('device_peak_bytes')):
            raise ValueError('Capacity evidence does not cover the configured context on the 32GB device')
    forced_first_path = record_path(run, 'pilot', forced.get('unique_id', ''))
    if not forced_first_path.exists():
        raise ValueError('Forced-check source first round is missing')
    forced_first = read_json(forced_first_path)
    n = forced.get('checkpoint', {}).get('prefix_token_count', 0)
    checks = forced.get('checks', {})
    if (forced_first.get('token_sha256') != forced.get('first_token_sha256')
            or checks.get('continue', {}).get('input_ids') != forced_first['prompt_ids']+forced_first['generated_ids'][:n]
            or checks.get('restart', {}).get('input_ids') != forced_first['prompt_ids']+forced.get('feedback_ids', [])
            or any(not checks.get(k, {}).get('engine', {}).get('device_total_bytes') for k in ('continue','restart'))):
        raise ValueError('Forced-check evidence does not use the original prefix and restart prompt')
    result = analyze(run, 'pilot', manifest)
    if result['questions_with_first_round'] != 5 or result['incomplete_pairs'] or not result['overall']['all_completed_checkpoints']:
        raise ValueError('Pilot needs five first rounds and all selected paired branches')
    branch_files = sorted((run/'branches').glob('pilot-*.json'))
    records = [read_json(p) for p in branch_files]
    if any(r.get('branch_seconds') is None or not r.get('engine', {}).get('device_total_bytes')
           or r['engine'].get('prefill_first_token_seconds') is None for r in records):
        raise ValueError('Pilot has missing GPU, timing, or prefix prefill evidence')
    result_hash = digest(run/'pilot-analysis.json')
    atomic_json(run/'protocol_frozen.json', {'config_hash':stable_hash(cfg), 'identity':manifest['identity'],
        'pilot_analysis_sha256':result_hash, 'capacity_sha256':digest(run/'capacity_check.json'),
        'forced_sha256':digest(run/'forced_check.json'), 'verifier_audit_sha256':digest(run/'verifier_audit_gate.json')})
    atomic_json(run/'pilot_acceptance.json', {'status':'PASS', 'config_hash':stable_hash(cfg),
        'pilot_analysis_sha256':result_hash, 'protocol_sha256':digest(run/'protocol_frozen.json')})
    backup(run, 'pilot-accepted')


def analyze(run, group, manifest):
    pairs, coverage = [], {}
    for row in manifest['splits'][group]:
        first_path = record_path(run, group, row['unique_id'])
        if not first_path.exists():
            continue
        first = read_json(first_path)
        if (first['token_sha256'] != stable_hash({'prompt_ids':first['prompt_ids'], 'generated_ids':first['generated_ids']})
                or first['selection'] != choose_checkpoints(first, manifest['identity']['config']['seed'])):
            raise ValueError('First-round evidence or selection checksum mismatch')
        coverage[row['unique_id']] = first['selection']
        for point in first['selection']['checkpoints']:
            prefix = group+'-'+stable_hash(row['unique_id'])[:16]+'-'+str(point['prefix_token_count'])
            a, b = (run/'branches'/(prefix+'-'+arm+'.json') for arm in ('A','B'))
            if a.exists() and b.exists():
                arms = {}
                for name, path in [('A', a), ('B', b)]:
                    arm = read_json(path)
                    checksum = arm.pop('record_sha256', None)
                    if (checksum != stable_hash(arm) or arm['first_token_sha256'] != first['token_sha256']
                            or arm['unique_id'] != row['unique_id'] or arm['checkpoint'] != point or arm['arm'] != name):
                        raise ValueError('Branch evidence checksum or source mismatch')
                    arm['record_sha256'] = checksum
                    arms[name] = arm
                pairs.append({'unique_id':row['unique_id'], 'checkpoint':point, **arms})
    result = {'group':group, 'coverage':coverage, 'overall':summarize(pairs),
              'by_threshold': {str(t): summarize([p for p in pairs if t in p['checkpoint']['thresholds']]) for t in (8192,16384,32768)},
              'coverage_by_threshold': {str(t):sum('prefix_token_count' in v['thresholds'][str(t)] for v in coverage.values()) for t in (8192,16384,32768)},
              'questions_with_first_round':len(coverage), 'planned_questions':len(manifest['splits'][group]),
              'probe_score_grouping':'NA: new trajectories have no reused hidden features',
              'forward_token_note':'second-round model input and output token counts are a compute proxy, not FLOPs; feedback is included in generator input',
              'incomplete_pairs':sum(len(v['checkpoints']) for v in coverage.values())-len(pairs)}
    atomic_json(run/(group+'-analysis.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare','audit-collect','audit-label','audit-report','capacity','pilot','forced-check','certify-pilot','exploratory','analyze'), required=True)
    parser.add_argument('--group', choices=('pilot','exploratory'), default='pilot')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--audit-extra', type=int, default=0)
    args = parser.parse_args()
    if args.config.resolve() != CONFIG.resolve():
        raise ValueError('Use the experiment configuration at '+str(CONFIG)+' so backup includes it')
    run, cfg, manifest = prepare(args.config)
    print('run:', run)
    if args.phase == 'prepare':
        return
    if args.phase == 'analyze':
        print(json.dumps(analyze(run,args.group,manifest), ensure_ascii=False))
        return
    lock_path = ROOT/'.cache'/(run.name+'.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another run is already active') from None
        if args.phase == 'audit-collect':
            collect_audit(run, cfg, manifest, args.audit_extra)
        elif args.phase == 'audit-label':
            model_label_audit(run, cfg)
        elif args.phase == 'audit-report':
            label_path = args.labels or run/'verifier_audit_model_labels.json'
            if not label_path.is_file():
                raise ValueError('audit-report requires model labels from audit-label or an explicit --labels file')
            print(json.dumps(audit_report(run, cfg, manifest, label_path), ensure_ascii=False))
            backup(run, 'verifier-audit-report')
        elif args.phase == 'capacity':
            capacity_check(run, cfg, manifest)
        elif args.phase == 'forced-check':
            forced_check(run, cfg, manifest)
        elif args.phase == 'certify-pilot':
            certify_pilot(run, cfg, manifest)
        else:
            execute(run, cfg, manifest, args.phase)


if __name__ == '__main__':
    main()
