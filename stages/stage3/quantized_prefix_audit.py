"""27B quantized verifier comparison. Never edits source prefixes or reference labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from stages.stage3.structured_prefix_audit import ROOT, read, save, sha, parse_output

BASE = ROOT / 'stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-unbounded-v2'
REF = BASE / 'full-review-v2-20261101'
SOURCE = ROOT / 'stages/stage3/runs/verification-benefit-c905eb4ccd85/verifier_audit_review_inputs.json'
RUN = ROOT / 'stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v1'
REPO = 'cyankiwi/Qwen3.5-27B-AWQ-4bit'
REV = '9d8d63632afafe90bdfcb70164d3e2a0955619f7'
SNAPSHOT = Path('/root/autodl-tmp/qwen35-27b-awq') / REV


def digest_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def backup(reason):
    """Scope commits to this diagnostic; preserve all pre-existing dirty files."""
    def git(*args):
        r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}, timeout=600)
        if r.returncode:
            raise RuntimeError('Scoped Git/LFS backup failed; local evidence retained')
        return r.stdout.strip()
    try:
        assert git('branch', '--show-current') == 'experiment/step-hidden-probe'
        assert not git('diff', '--cached', '--name-only'), 'Unrelated staged changes'
        paths = [str(Path(__file__).relative_to(ROOT)), str(RUN.relative_to(ROOT))]
        git('add', '--', *paths)
        if git('diff', '--cached', '--name-only'):
            git('commit', '--only', '-m', 'Stage3 27B quantized audit: ' + reason, '--', *paths)
        commit = git('rev-parse', 'HEAD')
        git('lfs', 'push', 'origin', 'HEAD')
        git('push', '--porcelain', 'origin', 'HEAD:refs/heads/experiment/step-hidden-probe')
        assert git('ls-remote', 'origin', 'refs/heads/experiment/step-hidden-probe').split()[0] == commit
        save(RUN / 'backup_status.json', dict(status='PUSHED', commit=commit, reason=reason))
        print('BACKUP', reason, commit, flush=True)
    except Exception as e:
        save(RUN / 'backup_status.json', dict(status='FAILED', reason=reason, error=str(e)))
        raise


def prepare():
    from transformers import AutoTokenizer
    assert not (RUN / 'manifest.json').exists(), 'Manifest is immutable'
    source = read(SOURCE)
    refs = read(REF / 'labels.json')
    old_manifest = read(BASE / 'manifest.json')
    cfg = old_manifest['original_config']
    old_snapshot = Path(cfg['model_cache_dir']) / ('models--' + cfg['verifier_model'].replace('/', '--')) / 'snapshots' / cfg['verifier_revision']
    old_tok = AutoTokenizer.from_pretrained(old_snapshot, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    inventory, payloads = [], []
    for i, (s, ref) in enumerate(zip(source, refs), 1):
        req = read(BASE / 'requests' / f'{i:03}.json')
        assert s['review_key'] == ref['review_key'] == req['review_key']
        text = old_tok.decode(req['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        system, rest = text[len('<|im_start|>system\n'):].split('<|im_end|>\n<|im_start|>user\n', 1)
        user = rest.split('<|im_end|>\n<|im_start|>assistant\n', 1)[0]
        payload = json.loads(user)
        expected = dict(problem=s['problem'], quoted_prefix=s['checkpoint_units'],
                        current_step=req['checkpoint_numbers'][-1], retained_feedback=[], check_type=s['check_type'])
        assert payload == expected
        assert system == (BASE / 'structured_system_prompt.txt').read_text()
        ids = tok.encode(text, add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
        req['input_ids'] = ids
        req['context_output_capacity'] = 81920 - len(ids)
        assert req['context_output_capacity'] >= 16384
        req['engine'] = dict(snapshot=str(SNAPSHOT), max_model_len=81920,
                             gpu_memory_utilization=.90, prefill_chunk_size=1024)
        save(RUN / 'requests' / f'{i:03}.json', req)
        inventory.append(dict(index=i, question_number=ref['question_number'], unique_id=s['unique_id'],
            review_key=s['review_key'], check_type=s['check_type'], checkpoint_count=len(s['checkpoint_units']),
            input_tokens=len(ids), old_input_tokens=len(read(BASE / 'requests' / f'{i:03}.json')['input_ids']),
            token_ids_equal_to_9b=ids == read(BASE / 'requests' / f'{i:03}.json')['input_ids'],
            context_output_capacity=req['context_output_capacity'], user_payload_sha256=digest_text(user),
            prefix_sha256=digest_text(s['quoted_prefix']), prompt_sha256=digest_text(text)))
        payloads.append(dict(index=i, review_key=s['review_key'], payload=payload))
    assert len(inventory) == len(source) == len(refs) == 57
    ordered = sorted(inventory, key=lambda x:x['input_tokens'])
    capacity = [4, ordered[len(ordered)//2]['index'], ordered[-1]['index']]
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'system_prompt.txt').write_text((BASE / 'structured_system_prompt.txt').read_text())
    save(RUN / 'payloads.json', payloads)
    save(RUN / 'input_inventory.json', inventory)
    protected = [SOURCE, BASE/'manifest.json', BASE/'structured_system_prompt.txt',
                 REF/'labels.json', REF/'labels.csv', REF/'report.md', REF/'disagreements.json', REF/'mathematical_notes.md']
    protected += sorted((BASE/'first').glob('*/attempt_01.json'))
    save(RUN / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model=REPO, revision=REV, snapshot=str(SNAPSHOT), quantization=read(SNAPSHOT/'config.json')['quantization_config'],
        source=str(SOURCE), reference=str(REF/'labels.json'), reference_is_provisional=True,
        reference_is_independent_expert_gold=False, baseline=str(BASE), capacity_indices=capacity,
        implementation_sha256=sha(__file__), protected_sources={str(p.relative_to(ROOT)):sha(p) for p in protected},
        versions={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','compressed-tensors','xgrammar']},
        config=dict(max_model_len=81920,max_tokens=None,gpu_memory_utilization=.90,
                    prefill_chunk_size=1024, dtype='bfloat16',kv_cache_dtype='auto',
                    mamba_ssm_cache_dtype='float32',enforce_eager=True,max_num_seqs=1,
                    enable_prefix_caching=False,enable_thinking=False,sampling=cfg['verifier_sampling']),
        differences_from_9b=['27B community AWQ W4A16 compressed-tensors group32 vs 9B BF16; selected after official GPTQ footprint inspection.',
          'Total context81920 vs148480; no fixed output cap in either; every full prompt fits with >=16384 output tokens.',
          'Prefill chunks1024 vs2048; eager execution vs default compilation, for VRAM margin. Runtime comparison includes these differences.',
          'Same system prompt, exact user text, JSON schema, per-prefix seeds, EOS IDs and non-thinking assistant prefix.',
          'Verifier only, no 4B generator residency, no rework calls.'],
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],text=True),
        initial_git_status=subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True),
        sources=['https://huggingface.co/'+REPO+'/tree/'+REV,
          'https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4/blob/8f0c09f227ae570e79617c6d9172b59df9c16081/config.json']))
    print('PREPARED', len(inventory), 'CAPACITY', capacity, 'MAX_INPUT', ordered[-1]['input_tokens'], flush=True)


def check_sources(m):
    assert m['implementation_sha256'] == sha(__file__), 'Runner changed; create a new run'
    for p, h in m['protected_sources'].items():
        assert sha(ROOT/p) == h, 'Protected source changed: ' + p


def execute(phase, stop_after=None):
    import torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer
    from veriserve_research.vllm_backend import DevicePeak
    m = read(RUN/'manifest.json'); check_sources(m)
    if phase == 'first':
        assert all(read(RUN/'capacity'/f'{i:03}.json')['status']=='completed' for i in m['capacity_indices'])
    selected = m['capacity_indices'] if phase=='capacity' else list(range(1,58))
    pending = [i for i in selected if not (RUN/phase/f'{i:03}.json').exists()]
    if not pending:
        print('NOTHING_PENDING',phase,flush=True);return
    backup(phase+'-before-start')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session=RUN/'sessions'/stamp
    save(session/'start.json',dict(phase=phase,pending=pending,config=m['config']))
    init_peak=DevicePeak(); start=time.perf_counter()
    try:
        model=LLM(model=str(SNAPSHOT),tokenizer=str(SNAPSHOT),dtype='bfloat16',
            max_model_len=81920,max_num_seqs=1,gpu_memory_utilization=.90,
            max_num_batched_tokens=1024,enable_chunked_prefill=True,mamba_ssm_cache_dtype='float32',
            enable_prefix_caching=False,generation_config='vllm',language_model_only=True,
            limit_mm_per_prompt={'image':0,'video':0},enforce_eager=True,
            logits_processors=['stages.stage3.verification_benefit_logits:RestoredPresence'],
            structured_outputs_config={'backend':'xgrammar'})
        torch.cuda.synchronize()
        save(session/'engine.json',dict(load_seconds=time.perf_counter()-start,
            device_peak_bytes=init_peak.stop(),device_peak_sampling_ms=20,
            vllm=vllm.__version__,device=torch.cuda.get_device_name(),cuda_capability=torch.cuda.get_device_capability()))
    except Exception as e:
        save(session/'engine_failure.json',dict(error=str(e),traceback=traceback.format_exc(),
            load_seconds=time.perf_counter()-start,device_peak_bytes=init_peak.stop()))
        backup(phase+'-engine-failure');raise
    tok=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
    completed=0
    for i in pending:
        req=read(RUN/'requests'/f'{i:03}.json')
        save(RUN/phase/f'{i:03}.intent.json',dict(index=i,session=stamp,seed=req['seed']))
        peak=DevicePeak();start=time.perf_counter()
        record=dict(index=i,review_key=req['review_key'],phase=phase,session=stamp,
            input_tokens=len(req['input_ids']),max_tokens=None,context_output_capacity=req['context_output_capacity'],
            status='service_error',judgment=None,output_tokens=None,seed=req['seed'])
        try:
            params=SamplingParams(**req['sampling'],frequency_penalty=0.,max_tokens=None,seed=req['seed'],
                detokenize=False,stop_token_ids=req['eos_ids'],extra_args={'restored_presence_history':[]},
                structured_outputs=StructuredOutputsParams(json=req['schema']))
            response=model.generate([{'prompt_token_ids':req['input_ids']}],params,use_tqdm=False)[0]
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            result=response.outputs[0];ids=list(result.token_ids)
            clean=ids[:-1] if ids and ids[-1] in req['eos_ids'] else ids
            raw=tok.decode(clean,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            record.update(raw_output=raw,raw_output_with_special_tokens=tok.decode(ids,skip_special_tokens=False),
                generated_ids=ids,output_tokens=len(ids),finish_reason=result.finish_reason,stop_reason=result.stop_reason,
                generation_seconds=elapsed,metrics={k:getattr(response.metrics,k) for k in
                    ['arrival_time','scheduled_ts','first_token_ts','last_token_ts'] if response.metrics and hasattr(response.metrics,k)})
            save(RUN/phase/f'{i:03}.response.json',record)
            parsed,status,error=parse_output(raw,result.finish_reason,req['checkpoint_numbers'])
            record.update(judgment=parsed,status=status,error=error)
        except Exception as e:
            record.update(error=str(e),traceback=traceback.format_exc())
        record.update(request_seconds=time.perf_counter()-start,device_peak_bytes=peak.stop(),device_peak_sampling_ms=20)
        save(RUN/phase/f'{i:03}.json',record);completed+=1
        print(json.dumps({k:record[k] for k in ['index','status','output_tokens','request_seconds','device_peak_bytes']}),
              (record['judgment'] or {}).get('verdict'),flush=True)
        if record['status']=='service_error':
            backup(phase+'-request-error');raise RuntimeError(record['error'])
        if completed%10==0:backup(phase+f'-{completed}-complete')
        if stop_after and completed>=stop_after:break
    save(session/'completed.json',dict(phase=phase,completed_this_session=completed))
    check_sources(m);backup(phase+'-session-complete')


def self_check():
    """Reuse production parser: truncated JSON must never supply a label."""
    raw=json.dumps(dict(verdict='FAIL',first_error_step=2,diagnosis='Wrong sum',hint='Recompute sum'))
    assert parse_output(raw,'stop',[1,2])[1]=='completed'
    assert parse_output(raw,'length',[1,2])[0] is None
    assert parse_output(raw,'stop',[1])[1]=='format_error'
    print('SELF_CHECK_PASSED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['prepare','capacity','first','backup','check'])
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='backup':backup('manual')
    elif a.phase=='check':self_check()
    else:execute(a.phase,a.stop_after)
