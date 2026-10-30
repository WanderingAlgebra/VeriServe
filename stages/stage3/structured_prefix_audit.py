"""Isolated, resumable JSON-Schema rerun of the saved 57 verifier inputs.

Run with /root/veriserve-vllm-env/bin/python -m stages.stage3.structured_prefix_audit.
This diagnostic never writes a verifier quality gate or modifies source records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'stages/stage3/runs/verification-benefit-c905eb4ccd85'
CODEX = Path('/root/project/VeriServe/stages/stage3/diagnostics/gpt61sol-prefix-audit/codex-direct-review-v1')
FOCUS = [10, 20, 21, 23, 25, 42, 48, 51, 52, 54]
MAJOR = [23, 25, 52, 54]
SMOKE = [4, 23, 34]  # short, formerly missed main error, longest exported prefix
PRIOR_CAPPED = ROOT / 'stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-v1'
DOCS = 'https://docs.vllm.ai/en/v0.20.0/features/structured_outputs/'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from veriserve_research.artifacts.io import atomic_json
    atomic_json(Path(path), value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema(numbers):
    def branch(verdict, first, hint):
        return dict(type='object', additionalProperties=False,
                    required=['verdict', 'first_error_step', 'diagnosis', 'hint'],
                    properties=dict(verdict={'const': verdict}, first_error_step=first,
                                    diagnosis={'type': 'string', 'minLength': 1}, hint=hint))
    return {'anyOf': [branch('PASS', {'type': 'null'}, {'const': ''}),
                      branch('FAIL', {'type': 'integer', 'enum': numbers},
                             {'type': 'string', 'minLength': 1})]}


def parse_output(raw, finish_reason, numbers):
    """Completion is checked before parsing; valid-looking truncated JSON is unfinished."""
    if finish_reason != 'stop':
        return None, 'truncation' if finish_reason == 'length' else 'incomplete_generation', str(finish_reason)
    try:
        from jsonschema import Draft202012Validator
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON field: ' + key)
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique)
        Draft202012Validator(schema(numbers)).validate(value)
        if not value['diagnosis'].strip() or value['verdict'] == 'FAIL' and not value['hint'].strip():
            raise ValueError('Feedback must contain non-whitespace text')
        return value, 'completed', None
    except (ValueError, TypeError) as exc:
        return None, 'format_error', str(exc)
    except Exception as exc:
        # jsonschema.ValidationError is an output validation failure, not a math label.
        from jsonschema import ValidationError
        if isinstance(exc, ValidationError):
            return None, 'format_error', exc.message
        raise


def make_prompt(old):
    before = 'Otherwise use UNCERTAIN.'
    after = ('Only PASS or FAIL is allowed. If you cannot establish a concrete unresolved '
             'mathematical error, return PASS under this operational rule.')
    assert old.count(before) == 1
    lines = old.replace(before, after).splitlines()
    assert lines[4].startswith('Emit exactly one JSON object')
    lines[4] = ('Emit exactly one JSON object with required fields verdict, first_error_step, diagnosis, hint. '
                'verdict is PASS or FAIL. PASS: first_error_step=null, a brief concrete explanation in '
                'diagnosis of what was checked, and hint="". FAIL: first_error_step is a current '
                'checkpoint number, diagnosis explains the concrete unresolved error, and hint gives '
                'actionable guidance for rework. Be concise; there is no character-count limit. '
                'Do not output markdown fences or other text outside the JSON.')
    return '\n'.join(lines)


def prepare(run):
    from transformers import AutoTokenizer
    import vllm
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    if (run / 'manifest.json').exists():
        raise FileExistsError('Use the existing immutable manifest; do not prepare twice')
    rows = read(SOURCE / 'verifier_audit_review_inputs.json')
    with (CODEX / 'labels.csv').open(encoding='utf-8-sig') as f:
        codex = list(csv.DictReader(f))
    assert len(rows) == len(codex) == 57
    cfg = read(SOURCE / 'manifest.json')['identity']['config']
    snapshot = (Path(cfg['model_cache_dir']) / ('models--' + cfg['verifier_model'].replace('/', '--'))
                / 'snapshots' / cfg['verifier_revision'])
    assert snapshot.is_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    checks = {read(p)['review_key']: (p, read(p)) for p in (SOURCE / 'audit_checks').glob('*.json')}
    assert len(checks) == 57 and len(set(r['review_key'] for r in rows)) == 57
    manifest_rows = []
    original_prompt = None
    for index, (row, reference) in enumerate(zip(rows, codex), 1):
        assert int(reference['source_index']) == index
        assert all(row[k] == reference[k] for k in ('review_key', 'unique_id', 'check_type'))
        path, check = checks[row['review_key']]
        request_path = SOURCE / 'worker' / (path.stem + '-0') / 'request.json'
        request = read(request_path)
        assert request['input_ids'] == check['input_ids']
        assert request['sampling'] == cfg['verifier_sampling'] and request['max_tokens'] == 1024
        assert not request.get('presence_history')
        decoded = tok.decode(request['input_ids'], skip_special_tokens=False)
        assert tok.encode(decoded, add_special_tokens=False) == request['input_ids']
        head = '<|im_start|>system\n'
        sep = '<|im_end|>\n<|im_start|>user\n'
        assert decoded.startswith(head)
        old_system, rest = decoded[len(head):].split(sep, 1)
        user_json = rest.split('<|im_end|>\n<|im_start|>assistant\n', 1)[0]
        payload = json.loads(user_json)
        expected = dict(problem=row['problem'], quoted_prefix=row['checkpoint_units'],
                        current_step=check['checkpoint_numbers'][-1], retained_feedback=[],
                        check_type=row['check_type'])
        assert payload == expected  # no gold, old verdicts, Codex labels, or future text
        original_prompt = original_prompt or old_system
        assert original_prompt == old_system
        new_system = make_prompt(old_system)
        new_text = head + new_system + sep + rest
        request['input_ids'] = tok.encode(new_text, add_special_tokens=False)
        # User override: no explicit output-token cap. vLLM0.20 resolves None to
        # max_model_len - input_tokens; context exhaustion remains unfinished.
        request['max_tokens'] = None
        request['context_output_capacity'] = request['engine']['max_model_len'] - len(request['input_ids'])
        request.update(source_index=index, review_key=row['review_key'],
                       checkpoint_numbers=check['checkpoint_numbers'], schema=schema(check['checkpoint_numbers']))
        save(run / 'requests' / f'{index:03}.json', request)
        manifest_rows.append(dict(source_index=index, review_key=row['review_key'], unique_id=row['unique_id'],
                                  check_type=row['check_type'], original_file=str(path),
                                  original_request=str(request_path), codex_label=reference['label'],
                                  codex_reason=reference['reason'], codex_quote=reference['first_error_quote'],
                                  codex_caveat=reference.get('caveat', ''),
                                  original_result=check['results'][0],
                                  prompt_token_difference=len(request['input_ids']) - len(check['input_ids'])))
    run.mkdir(parents=True, exist_ok=True)
    (run / 'original_system_prompt.txt').write_text(original_prompt)
    (run / 'structured_system_prompt.txt').write_text(make_prompt(original_prompt))
    save(run / 'schema_example.json', schema([1, 2, 3]))
    save(run / 'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(), source=str(SOURCE),
         source_inputs_sha256=sha(SOURCE / 'verifier_audit_review_inputs.json'),
         codex_csv=str(CODEX / 'labels.csv'), codex_csv_sha256=sha(CODEX / 'labels.csv'),
         original_config=cfg, rows=manifest_rows,
         implementation_sha256=sha(__file__), vllm_version=vllm.__version__,
         environment={k: importlib.metadata.version(k) for k in ['torch','transformers','xgrammar','jsonschema']},
         api_signature=str(inspect.signature(StructuredOutputsParams)), docs=DOCS,
         differences=['JSON Schema via SamplingParams.structured_outputs; backend=xgrammar.',
           'UNCERTAIN removed; no concrete unresolved error -> PASS. This changes the decision space.',
           'PASS diagnosis now contains a concrete explanation; hint remains empty.',
           'Removed300-character diagnosis/hint cap; fields remain mandatory.',
           'Resident sequential engine per invocation, rather than one model initialization per sample. '
           'max_num_seqs=1 and prefix caching disabled; requests have independent original seeds.',
           'Original eos IDs, sampling, model revision, precision and context retained.',
           'User override: max_tokens=None, no fixed output cap. vLLM resolves it to the remaining '
           '148480-token model context; actual output tokens and context exhaustion are recorded. '
           'This replaces the originally planned1024-token cap and changes the budget comparison.'],
         input_audit=dict(count=57, ordered_keys_match=True, user_payloads_exact=True,
                          original_token_roundtrip=True, seed_source='original per-prefix worker request'),
         smoke_indices=SMOKE, focus_indices=FOCUS, major_indices=MAJOR,
         output_budget_policy='no_fixed_output_cap', prior_capped_run=str(PRIOR_CAPPED),
         truncation_probe=read(PRIOR_CAPPED/'smoke_truncation/004/attempt_01.json'),
         vllm_none_semantics_source='/root/veriserve-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/input_processor.py:307'))
    print(f'PREPARED {run}:57 exact user payloads and reference joins', flush=True)


def backup(run, reason):
    """Native scoped Git/LFS backup; never stage pre-existing user changes."""
    def git(*args):
        p = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True,
                           env={**os.environ, 'GIT_TERMINAL_PROMPT':'0'}, timeout=600)
        if p.returncode:
            raise RuntimeError('Git/LFS backup failed; local artifacts retained; inspect transport configuration')
        return p.stdout.strip()
    try:
        assert git('branch', '--show-current') == 'experiment/step-hidden-probe'
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Unrelated staged files; refusing to include them')
        paths = [str(Path(__file__).relative_to(ROOT)), 'tests/test_structured_prefix_audit.py',
                 str(run.relative_to(ROOT))]
        git('lfs','version')
        git('add','--',*paths)
        if git('diff','--cached','--name-only','--',*paths):
            git('commit','--only','-m','Stage3 structured prefix audit: '+reason,'--',*paths)
        commit = git('rev-parse','HEAD')
        git('lfs','push','origin','HEAD')
        git('push','--porcelain','origin','HEAD:refs/heads/experiment/step-hidden-probe')
        assert git('ls-remote','origin','refs/heads/experiment/step-hidden-probe').split()[0] == commit
        save(run/'backup_status.json',dict(status='PUSHED',commit=commit,reason=reason))
        print('BACKUP',reason,commit,flush=True)
    except Exception as exc:
        save(run/'backup_status.json',dict(status='FAILED',reason=reason,error=str(exc)))
        raise


def attempts(run, phase, index):
    return sorted((run/phase/f'{index:03}').glob('attempt_*.json'))


def run_phase(run, phase, stop_after=None):
    manifest = read(run/'manifest.json')
    assert manifest['implementation_sha256'] == sha(__file__), 'Changed runner: create a new run'
    assert manifest['source_inputs_sha256'] == sha(SOURCE/'verifier_audit_review_inputs.json')
    assert manifest['codex_csv_sha256'] == sha(manifest['codex_csv'])
    if phase == 'first':
        for i in SMOKE:
            records=attempts(run,'smoke',i)
            assert records and read(records[-1])['status']=='completed', 'Complete real smoke before first round'
        probe=manifest['truncation_probe']
        assert probe['status']=='truncation' and probe['judgment'] is None
        assert (run/'resume_check.json').is_file(), 'Verify real resume before first round'
    selected = SMOKE if phase=='smoke' else list(range(1,58))
    if phase == 'retry':
        selected=[i for i in selected if attempts(run,'first',i) and read(attempts(run,'first',i)[-1])['status']!='completed']
    # Preserve an attempt whose process died between intent and result persistence.
    for intent in (run/phase).glob('*/intent_*.json'):
        target=intent.with_name(intent.name.replace('intent_','attempt_'))
        if not target.exists():
            prior=read(intent)
            save(target,{**prior,'status':'process_interrupted','judgment':None,
                         'error':'Process ended without a persisted response; token cost unknown',
                         'raw_output':None,'raw_output_with_special_tokens':None,
                         'generated_ids':None,'output_tokens':None,'input_tokens':None,
                         'request_seconds':None,'finish_reason':None,'stop_reason':None})
    pending=[]; skipped=[]
    for i in selected:
        completed=attempts(run,phase,i)
        if completed:
            last=read(completed[-1])
            if last['status']=='completed' or phase=='first':
                skipped.append(i);continue
            if phase=='retry':
                skipped.append(i);continue
        pending.append(i)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session_dir=run/'sessions'/stamp
    save(session_dir/'start.json',dict(phase=phase,pending=pending,skipped=skipped,max_tokens=None))
    print('PENDING',phase,pending,'SKIPPED',skipped,flush=True)
    if not pending:
        return
    backup(run,phase+'-before-start')
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from veriserve_research.vllm_backend import DevicePeak
    from transformers import AutoTokenizer
    assert torch.cuda.device_count()==1
    request0=read(run/'requests'/f'{pending[0]:03}.json');cfg=request0['engine']
    peak=DevicePeak();started=time.perf_counter()
    try:
        model=LLM(model=cfg['snapshot'],tokenizer=cfg['snapshot'],dtype='bfloat16',
                  max_model_len=cfg['max_model_len'],max_num_seqs=1,
                  gpu_memory_utilization=cfg['gpu_memory_utilization'],
                  max_num_batched_tokens=cfg['prefill_chunk_size'],enable_chunked_prefill=True,
                  mamba_ssm_cache_dtype='float32',enable_prefix_caching=False,
                  generation_config='vllm',language_model_only=True,disable_log_stats=False,
                  limit_mm_per_prompt={'image':0,'video':0},
                  logits_processors=['stages.stage3.verification_benefit_logits:RestoredPresence'],
                  structured_outputs_config={'backend':'xgrammar'})
        torch.cuda.synchronize();init=time.perf_counter()-started
        tok=AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True)
        save(session_dir/'engine.json',dict(init_seconds=init,vllm_version=vllm.__version__,
            torch_version=torch.__version__,device=torch.cuda.get_device_name(),
            sampling_signature=str(inspect.signature(SamplingParams))))
    except Exception as exc:
        save(session_dir/'engine_failure.json',dict(status='service_error',error=str(exc),traceback=traceback.format_exc()))
        peak.stop();raise
    count=0
    for index in pending:
        request=read(run/'requests'/f'{index:03}.json')
        cap=request['max_tokens']
        assert cap is None, 'User requested no fixed output cap'
        directory=run/phase/f'{index:03}';directory.mkdir(parents=True,exist_ok=True)
        number=max([int(p.stem.split('_')[-1]) for p in directory.glob('attempt_*.json')]+[0])+1
        path=directory/f'attempt_{number:02}.json'
        intent=directory/f'intent_{number:02}.json'
        save(intent,dict(source_index=index,phase=phase,max_tokens=cap,seed=request['seed'],session=stamp))
        start=time.perf_counter();record=dict(source_index=index,review_key=request['review_key'],
             phase=phase,attempt=number,max_tokens=cap,seed=request['seed'],session=stamp,
             status='service_error',judgment=None,raw_output='',raw_output_with_special_tokens='',
             generated_ids=[],output_tokens=0,input_tokens=len(request['input_ids']),
             finish_reason=None,stop_reason=None)
        record['context_output_capacity']=request['context_output_capacity']
        try:
            params=SamplingParams(**request['sampling'],frequency_penalty=0.,max_tokens=cap,
                seed=request['seed'],detokenize=False,stop_token_ids=request['eos_ids'],
                extra_args={'restored_presence_history':[]},
                structured_outputs=StructuredOutputsParams(json=request['schema']))
            response=model.generate([{'prompt_token_ids':request['input_ids']}],params,use_tqdm=False)[0]
            result=response.outputs[0];torch.cuda.synchronize()
            ids=list(result.token_ids)
            generation_seconds=time.perf_counter()-start
            clean_ids=ids[:-1] if ids and ids[-1] in request['eos_ids'] else ids
            raw=tok.decode(clean_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            record.update(raw_output=raw,
                raw_output_with_special_tokens=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                generated_ids=ids,output_tokens=len(ids),finish_reason=result.finish_reason,
                stop_reason=result.stop_reason,generation_seconds=generation_seconds)
            save(directory/f'response_{number:02}.json',record)
            parsed,status,error=parse_output(raw,result.finish_reason,request['checkpoint_numbers'])
            # Only a completed valid object supplies a mathematical label.
            record.update(status=status,judgment=parsed,error=error,
                metrics={name:getattr(response.metrics,name) for name in
                         ['arrival_time','first_token_ts','scheduled_ts','last_token_ts','finished_time']
                         if response.metrics and hasattr(response.metrics,name)},
                legacy_parser=legacy_diagnostic(raw,result.finish_reason,request['checkpoint_numbers']))
        except Exception as exc:
            record.update(error=str(exc),traceback=traceback.format_exc())
        record['request_seconds']=time.perf_counter()-start
        save(path,record);count+=1
        print(json.dumps(dict(index=index,status=record['status'],verdict=(record['judgment'] or {}).get('verdict'),
                   output_tokens=record['output_tokens'],seconds=round(record['request_seconds'],3))),flush=True)
        if record['status']=='service_error':
            backup(run,phase+'-service-error');raise RuntimeError(record['error'])
        if count%10==0:
            backup(run,phase+f'-{count}-complete')
        if stop_after and count>=stop_after:
            break
    save(session_dir/'completed.json',dict(phase=phase,completed_this_session=count,
         model_init_seconds=init,device_peak_bytes=peak.stop(),skipped=skipped))
    backup(run,phase+'-session-complete')


def legacy_diagnostic(raw, finish, numbers):
    from veriserve_research.intervention.protocol import parse_verdict
    return parse_verdict(raw,numbers,0,'EOS' if finish=='stop' else 'LENGTH_TRUNCATED',False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','smoke','first','retry','backup'],required=True)
    p.add_argument('--stop-after',type=int)
    args=p.parse_args();run=args.run.resolve()
    if args.phase=='prepare':prepare(run)
    elif args.phase=='backup':backup(run,'manual')
    else:run_phase(run,args.phase,args.stop_after)


if __name__=='__main__':
    main()
