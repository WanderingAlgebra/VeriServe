"""Paired Step-2 intervention; ordinary files, resident models, native KV crop."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from stages.stage1.veriserve.common import atomic_json, stable_hash
from stages.stage2 import run_probe as old
from . import protocol, storage

ROOT = old.ROOT
HERE = Path(__file__).resolve().parent


def key(uid):
    return stable_hash(uid)[:20]


def clock():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def probe_files(cfg):
    folder = ROOT / cfg['stage2_run']
    result = {}
    for name in ('B', 'C'):
        meta = storage.safe_read_json(folder / f'probe_{name}.json')
        with np.load(folder / f'probe_{name}.npz', allow_pickle=False) as data:
            probe = {k: data[k].copy() for k in ('mean', 'scale', 'w', 'b')}
        if not np.all(probe['scale'] > 0) or not np.isclose(float(probe['b']), meta['b'], atol=0, rtol=0):
            raise ValueError('Probe JSON/NPZ mismatch')
        if name == 'B' and meta['layer'] != 19:
            raise ValueError('B must be hidden_states[19]')
        result[name] = probe
    return result


class Session:
    """cache holds ids[:cache_length]; logits are valid only once all ids are fed."""
    def __init__(self, model, ids):
        self.model, self.ids = model, list(ids)
        self.cache, self.logits = None, None
        self.forward(self.ids)

    def forward(self, ids):
        import torch
        if not ids:
            return
        with torch.inference_mode():
            x = torch.tensor([ids], device='cuda:0')
            output = self.model.model(input_ids=x, past_key_values=self.cache, use_cache=True,
                                      return_dict=True)
            self.cache = output.past_key_values
            self.logits = self.model.lm_head(output.last_hidden_state[:, -1:])[0, -1]
        if self.cache.get_seq_length() != len(self.ids):
            raise RuntimeError('KV/context length mismatch')

    def ensure(self):
        missing = self.ids[self.cache.get_seq_length():]
        if missing:
            self.forward(missing)

    def sample(self):
        self.ensure()
        token = int(self.logits.argmax())
        self.ids.append(token)  # last sampled token may still be absent from KV
        return token

    def rollback(self, checkpoint, feedback):
        checkpoint = list(checkpoint)
        if self.ids[:len(checkpoint)] != checkpoint:
            raise RuntimeError('Rollback checkpoint is not an original token prefix')
        self.cache.crop(min(self.cache.get_seq_length(), len(checkpoint)))
        self.ids = checkpoint + list(feedback)
        missing = self.ids[self.cache.get_seq_length():]
        self.forward(missing)
        return len(missing)

    def clone(self):
        other = object.__new__(Session)
        other.model, other.ids = self.model, self.ids.copy()
        other.cache, other.logits = copy.deepcopy(self.cache), self.logits.clone()
        return other


def load_resident(cfg, run):
    import torch
    from huggingface_hub import snapshot_download
    from stages.stage1.veriserve.prm import ProcessRewardModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Required BF16 CUDA unavailable; no model/precision substitution')
    generator, tokenizer = old.load_model(cfg)
    path = snapshot_download(cfg['verifier']['id'], revision=cfg['revisions']['verifier'])
    verifier = ProcessRewardModel(path, threshold=cfg['verifier']['threshold'])
    if verifier.tokenizer.name_or_path != path:
        raise RuntimeError('Verifier tokenizer revision not fixed')
    start = clock()
    warm = Session(generator, [tokenizer.eos_token_id] * 16)
    warm.sample()
    warm.ensure()
    verifier.verify('Compute 1+1.', '', 'Step 1: 1+1=2.\n\nFinal answer: \\boxed{2}')
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    # Real simultaneous residency and cache/headroom at the frozen limits.
    torch.cuda.reset_peak_memory_stats()
    big = Session(generator, [tokenizer.eos_token_id] * cfg['max_total_tokens'])
    prm_ids = [verifier.tokenizer.eos_token_id] * cfg['max_prm_input_tokens']
    with torch.inference_mode():
        output = verifier.model(input_ids=torch.tensor([prm_ids], device=verifier.device), use_cache=False)
        if not torch.isfinite(output.logits).all():
            raise RuntimeError('PRM capacity forward produced non-finite logits')
    capacity = {'status': 'PASS', 'warmup_capacity_seconds': clock()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'free_total_bytes': list(torch.cuda.mem_get_info()),
                'generator_context_tokens': cfg['max_total_tokens'],
                'prm_input_tokens': cfg['max_prm_input_tokens'],
                'simultaneous_residency': True, 'source_hashes': storage.source_hashes()}
    atomic_json(run / 'capacity_check.json', capacity)
    del big, output, prm_ids
    gc.collect()
    torch.cuda.empty_cache()
    return generator, tokenizer, verifier


def feature(model, ids):
    """Save only the selected layer; hook layer 18 gives HF hidden_states[19]."""
    import torch
    values = []
    def capture(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        values.append(h[0, -1].float().cpu().numpy().copy())
    hook = model.model.layers[18].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            x = torch.tensor([ids], device='cuda:0')
            model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                        output_hidden_states=False, return_dict=True)
    finally:
        hook.remove()
    if len(values) != 1:
        raise RuntimeError('Selected hidden layer was not captured exactly once')
    return values[0]


def score_snapshot(cfg, run, snapshot, model, probes):
    if snapshot.get('q_B') is not None:
        path = run / 'snapshots' / f"{key(snapshot['unique_id'])}.npz"
        if old.digest(path) != snapshot['feature_sha256']:
            raise RuntimeError('Snapshot feature hash mismatch; recover committed backup')
        return snapshot
    ids = snapshot['prompt_ids'] + snapshot['prefix_ids']
    start = clock()
    h = feature(model, ids)
    feature_seconds = clock()-start
    start = time.perf_counter()
    q_b = float(old.predict(probes['B'], h[None])[0])
    q_c = float(old.predict(probes['C'], np.array([[2, len(snapshot['prefix_ids'])]]))[0])
    probe_seconds = time.perf_counter()-start
    path = run / 'snapshots' / f"{key(snapshot['unique_id'])}.npz"
    old.save_npz(path, h=h, layer=np.array(19))
    snapshot.update(q_B=q_b, q_C=q_c, feature_sha256=old.digest(path),
                    timing={**snapshot['timing'], 'T_offline_feature':feature_seconds, 'T_probe':probe_seconds},
                    offline_feature_input_tokens=len(ids))
    atomic_json(path.with_suffix('.json'), snapshot)
    storage.event(run, 'SNAPSHOT_SCORED', unique_id=snapshot['unique_id'], q_B=q_b, q_C=q_c)
    return snapshot


def generator_limit(cfg, state):
    if state['budget']['generated_tokens'] >= cfg['max_request_tokens']:
        return 'BUDGET_GENERATION_REQUEST'
    if state['budget']['attempt_tokens'] >= cfg['max_new_tokens']:
        return 'BUDGET_GENERATION_ATTEMPT'
    if len(state['context_ids']) >= cfg['max_total_tokens']:
        return 'CONTEXT_LIMIT'
    return None


def initial_snapshot(cfg, run, phase, row, model, tokenizer, probes, backend):
    uid = row['unique_id']
    path = run / 'snapshots' / f'{key(uid)}.json'
    saved = storage.safe_read_json(path)
    if saved and saved.get('status') in ('ELIGIBLE', 'NO_ELIGIBLE_ANCHOR'):
        if stable_hash(saved['prompt_ids'] + saved.get('prefix_ids', []) + saved.get('pending_ids', [])) != saved['token_hash']:
            raise RuntimeError('Snapshot raw token hash mismatch')
        return score_snapshot(cfg, run, saved, model, probes) if saved['status']=='ELIGIBLE' else saved
    if saved:
        snapshot = saved
    else:
        prompt = cfg['prompt'].replace('{problem}', row['problem'])
        prompt_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True)
        snapshot = {'unique_id':uid, 'phase':phase, 'status':'GENERATING', 'problem':row['problem'],
                    'prompt_ids':prompt_ids, 'generated_ids':[], 'budget':{'generated_tokens':0,'attempt_tokens':0,
                    'forward_tokens':len(prompt_ids)}, 'timing':{'T_common_generation':0.0},
                    'backend':backend, 'source_hashes':storage.source_hashes(), 'created':old.now()}
        atomic_json(path,snapshot)
    if len(snapshot['prompt_ids']) > cfg['max_total_tokens'] or snapshot['budget']['forward_tokens'] > cfg['max_forward_tokens']:
        snapshot['reason']='INITIAL_CONTEXT_OR_PREFILL_LIMIT'
        session=None
    else:
        start=clock()
        session=Session(model,snapshot['prompt_ids'])
        if snapshot['generated_ids']:
            for token in snapshot['generated_ids']:
                session.ids.append(token)
                session.ensure()
            snapshot['timing']['T_common_infra_restore']=clock()-start
            start=clock()
        eos_cfg = model.generation_config.eos_token_id
        eos = set(eos_cfg if isinstance(eos_cfg,list) else [eos_cfg])
        while True:
            state={'context_ids':session.ids,'budget':snapshot['budget']}
            reason=generator_limit(cfg,state)
            if reason:
                snapshot['reason']=reason
                break
            token=session.sample()
            snapshot['generated_ids'].append(token)
            snapshot['budget']['generated_tokens']+=1
            snapshot['budget']['attempt_tokens']+=1
            if token in eos:
                snapshot['reason']='EOS_BEFORE_ELIGIBLE_STEP_2'
                break
            try:
                bound=protocol.boundary(tokenizer,snapshot['generated_ids'],target_step=2)
            except ValueError as exc:
                snapshot['reason']=str(exc)
                break
            if bound:
                snapshot.update(status='ELIGIBLE', prefix_ids=bound['prefix_ids'], pending_ids=bound['pending_ids'],
                                step=bound['step'], step_number=2, prefix_length=len(bound['prefix_ids']),
                                near_end=bound['near_end'], delay_collapsed=bound['near_end'],
                                checkpoint_ids=snapshot['prompt_ids'].copy(), checkpoint_step=0)
                break
            if len(snapshot['generated_ids']) % 32 == 0:
                elapsed=clock()-start
                snapshot['timing']['T_common_generation']+=elapsed
                atomic_json(path,snapshot)
                start=clock()
        snapshot['timing']['T_common_generation']+=clock()-start
    if snapshot['status']!='ELIGIBLE':
        snapshot.update(status='NO_ELIGIBLE_ANCHOR', prefix_ids=snapshot['generated_ids'].copy(),pending_ids=[])
    snapshot['token_hash']=stable_hash(snapshot['prompt_ids']+snapshot['prefix_ids']+snapshot['pending_ids'])
    snapshot['cache_state']={'cache_length':session.cache.get_seq_length() if session else None,
                             'context_length':len(session.ids) if session else len(snapshot['prompt_ids']),
                             'last_sampled_token_cached':bool(session and session.cache.get_seq_length()==len(session.ids))}
    atomic_json(path,snapshot)
    storage.event(run,'SNAPSHOT_COMPLETED',unique_id=uid,status=snapshot['status'],reason=snapshot.get('reason'))
    return score_snapshot(cfg,run,snapshot,model,probes) if snapshot['status']=='ELIGIBLE' else snapshot


def arm_state(snapshot, arm):
    reasoning=snapshot['prefix_ids']+snapshot['pending_ids']
    prompt=snapshot['prompt_ids']
    return {'unique_id':snapshot['unique_id'],'phase':snapshot['phase'],'arm':arm,'status':'RUNNING','completed':False,
            'prompt_ids':prompt.copy(),'context_ids':prompt+reasoning,'reasoning_ids':reasoning.copy(),
            'reasoning_positions':list(range(len(prompt),len(prompt)+len(reasoning))),
            'checkpoint_ids':prompt.copy(),'checkpoint_reasoning_ids':[],'checkpoint_positions':[],
            'checkpoint_step':0,'first_check_done':False,'eos':False,'events':[],'checks':[],
            'reasoning_breaks':[],'checkpoint_breaks':[],
            'generated_segments':[{'kind':'COMMON','ids':reasoning.copy()}], 'feedback_tokens':[],
            'budget':{**copy.deepcopy(snapshot['budget']),'prm_calls':0,'rollbacks':0,'revoked_tokens':0,
                      'feedback_tokens':0},'timing':{},'backend':snapshot['backend'], 'Y':None,
            'snapshot_token_hash':snapshot['token_hash'],'created':old.now()}


def reasoning_text(state, tokenizer, start=0, end=None):
    ids=state['reasoning_ids']
    end=len(ids) if end is None else end
    breaks=[start]+[b for b in state.get('reasoning_breaks',[]) if start<b<end]+[end]
    return '\n\n'.join(old.decode(tokenizer,ids[a:b]) for a,b in zip(breaks,breaks[1:]))


def verify_prm(verifier, ids, numbers, prior):
    import torch
    with torch.inference_mode():
        output=verifier.model(input_ids=torch.tensor([ids],device=verifier.device),use_cache=False)
        positions=[i for i,t in enumerate(ids) if t==verifier.separator_id]
        scores=torch.softmax(output.logits[0,positions].float(),dim=-1)[:,1].tolist()
    result=protocol.verdict_from_scores(scores,numbers,prior,verifier.threshold)
    result['unrounded_scores']=scores
    return result


def prm_check(cfg, state, session, verifier, tokenizer, prefix_count, final, forced=None):
    prefix=state['reasoning_ids'][:prefix_count]
    n_accepted=len(state['checkpoint_reasoning_ids'])
    if prefix[:n_accepted]!=state['checkpoint_reasoning_ids']:
        raise RuntimeError('Accepted reasoning prefix changed')
    accepted=reasoning_text(state,tokenizer,0,n_accepted)
    new=reasoning_text(state,tokenizer,n_accepted,prefix_count)
    ids,numbers,prior=verifier.input_ids(state['problem'],accepted,new)
    reason = ('BUDGET_PRM_CALLS' if state['budget']['prm_calls']>=cfg['max_prm_calls'] else
              'PRM_INPUT_LIMIT' if len(ids)>cfg['max_prm_input_tokens'] else
              'BUDGET_FORWARD_INPUT' if state['budget']['forward_tokens']+len(ids)>cfg['max_forward_tokens'] else None)
    if reason:
        return reason
    start=clock()
    result=verify_prm(verifier,ids,numbers,prior) if forced is None else forced
    raw=json.dumps(result,ensure_ascii=False)
    input_tokens=len(ids)
    elapsed=clock()-start
    state['budget']['prm_calls']+=1
    state['budget']['forward_tokens']+=input_tokens
    effective=protocol.effective_verdict(result['verdict'],result.get('first_error_step'),state['checkpoint_step'])
    current=max([int(m.group(1)) for m in old.STEP.finditer(reasoning_text(state,tokenizer,0,prefix_count))] or [state['checkpoint_step']])
    feedback=protocol.feedback_text(result.get('diagnosis',''),result.get('hint',''),state['checkpoint_step']) if effective=='FAIL' else ''
    feedback_ids=tokenizer.encode(feedback,add_special_tokens=False) if feedback else []
    check={'kind':'CHECK','position':'FINAL' if final else 'STEP','is_first':not state['first_check_done'],
           'raw_verdict':result.get('raw_verdict',result['verdict']),'effective_verdict':effective,'first_error_step':result.get('first_error_step'),
           'diagnosis':result.get('diagnosis',''),'hint':result.get('hint',''),'step_scores':result.get('step_scores',[]),
           'raw':raw,'input_ids':ids,'input_tokens':input_tokens,'accepted_steps':prior,
           'current_step':current,'reasoning_tokens':prefix_count,'seconds':elapsed,
           'feedback_ids':feedback_ids,'feedback_text':feedback,'synthetic':forced is not None,
           'extra_steps':max(0,current-2),'early_endpoint':bool(final and not state['first_check_done'] and current<4)}
    state['checks'].append(check)
    state['events'].append(check.copy())
    if not state['first_check_done']:
        state['first_check']={'extra_steps':check['extra_steps'],'early_endpoint':check['early_endpoint'],
                              'position':check['position'],'input_tokens':input_tokens}
    state['first_check_done']=True
    state['timing']['verification_seconds']=state['timing'].get('verification_seconds',0)+elapsed
    if effective=='PASS':
        end=state['reasoning_positions'][prefix_count-1]+1 if prefix_count else len(state['prompt_ids'])
        state['checkpoint_ids']=state['context_ids'][:end]
        state['checkpoint_reasoning_ids']=prefix.copy()
        state['checkpoint_positions']=state['reasoning_positions'][:prefix_count]
        state['checkpoint_step']=current
        state['checkpoint_breaks']=[b for b in state.get('reasoning_breaks',[]) if b<prefix_count]
    if final and effective in ('PASS','UNCERTAIN'):
        state['submitted_reasoning_ids']=prefix.copy()
        return 'FINAL_'+effective
    if effective=='FAIL':
        if state['budget']['rollbacks']>=cfg['max_rollbacks']:
            return 'BUDGET_ROLLBACKS'
        cp=state['checkpoint_ids']
        missing=max(0,len(cp)-session.cache.get_seq_length())+len(feedback_ids)
        if len(cp)+len(feedback_ids)>=cfg['max_total_tokens']:
            return 'CONTEXT_LIMIT_FEEDBACK'
        if state['budget']['forward_tokens']+missing>cfg['max_forward_tokens']:
            return 'BUDGET_FORWARD_INPUT_FEEDBACK'
        if state['budget']['generated_tokens']>=cfg['max_request_tokens'] or state['budget']['prm_calls']>=cfg['max_prm_calls']:
            return 'BUDGET_NO_REWORK'
        revoked=len(state['reasoning_ids'])-len(state['checkpoint_reasoning_ids'])
        cache_before=session.cache.get_seq_length()
        start=clock()
        charged=session.rollback(cp,feedback_ids)
        rollback_seconds=clock()-start
        state['budget']['rollbacks']+=1
        state['budget']['revoked_tokens']+=revoked
        state['budget']['forward_tokens']+=charged
        state['budget']['feedback_tokens']+=len(feedback_ids)
        state['budget']['attempt_tokens']=0
        state['context_ids']=session.ids.copy()
        state['reasoning_ids']=state['checkpoint_reasoning_ids'].copy()
        state['reasoning_positions']=state['checkpoint_positions'].copy()
        state['reasoning_breaks']=state['checkpoint_breaks'].copy()+[len(state['reasoning_ids'])]
        state['feedback_tokens'].append(feedback_ids)
        state['eos']=False
        state['timing']['rollback_feedback_seconds']=state['timing'].get('rollback_feedback_seconds',0)+rollback_seconds
        state['events'].append({'kind':'ROLLBACK','checkpoint_step':state['checkpoint_step'],
                                'checkpoint_length':len(cp),'cache_before':cache_before,
                                'cache_after':session.cache.get_seq_length(),'feedback_prefill_tokens':charged,
                                'revoked_tokens':revoked,'feedback_ids':feedback_ids,'seconds':rollback_seconds})
    elif state['budget']['prm_calls']>=cfg['max_prm_calls']:
        return 'BUDGET_PRM_CALLS'
    return None


def finish_arm(cfg, state, row):
    state['status']='COMPLETE'
    state['completed']=True
    state['finished']=old.now()
    submitted=state.get('submitted_reasoning_ids')
    if submitted is None:
        state['Y']=0
        state['grading']={'answer':None,'label':1,'exclusion':state['termination']}
    else:
        text=state['submitted_text']
        grade=old.grade(text,row['answer'],'EOS',cfg['score_timeout_seconds'])
        state['grading']=grade
        reason=grade.get('exclusion')
        state['Y']=None if reason and (reason.startswith('GOLD_') or reason.endswith('_TIMEOUT') or reason=='VERIFY_ERROR') else (
            int(grade['label']==0) if grade['label'] is not None else 0)
    return state


def execute_arm(cfg, run, row, snapshot, arm, model, tokenizer, verifier, backend):
    import torch
    state=arm_state(snapshot,arm)
    state['backend']=backend
    state['problem']=row['problem']  # gold is used only in finish_arm, after timing ends
    path=run/'arms'/f'{key(row["unique_id"])}.{arm}.json'
    start=clock()
    session=Session(model,state['context_ids'])
    restore=clock()-start
    state['timing'].update(T_snapshot_restore=restore,snapshot_restore_input_tokens=len(state['context_ids']),
                          snapshot_restore_kind='experiment_only_full_original_context_prefill')
    atomic_json(path,state)
    torch.cuda.reset_peak_memory_stats()
    start=clock()
    generation_start=start
    eos_cfg=model.generation_config.eos_token_id
    eos=set(eos_cfg if isinstance(eos_cfg,list) else [eos_cfg])
    reason=None
    check_pending=(len(snapshot['prefix_ids']),False) if arm=='NOW' else None
    try:
        while True:
            if check_pending is not None:
                count,final=check_pending
                state['timing']['generation_control_seconds']=state['timing'].get('generation_control_seconds',0)+(clock()-generation_start)
                reason=prm_check(cfg,state,session,verifier,tokenizer,count,final)
                atomic_json(path,state)
                storage.event(run,'CHECK_SAVED',unique_id=row['unique_id'],arm=arm,check=len(state['checks']),
                              verdict=state['checks'][-1]['effective_verdict'] if state['checks'] else None)
                generation_start=clock()
                check_pending=None
                if reason:
                    break
            reason=generator_limit(cfg,state)
            if reason:
                break
            token=session.sample()
            state['context_ids']=session.ids.copy()
            state['reasoning_positions'].append(len(session.ids)-1)
            state['reasoning_ids'].append(token)
            state['budget']['generated_tokens']+=1
            state['budget']['attempt_tokens']+=1
            if not state['generated_segments'] or state['generated_segments'][-1]['kind']!='DECODE':
                state['generated_segments'].append({'kind':'DECODE','ids':[]})
            state['generated_segments'][-1]['ids'].append(token)
            if token in eos:
                state['eos']=True
                end=protocol.terminal(tokenizer,state['reasoning_ids'],eos=True,text=reasoning_text(state,tokenizer))
                if end and end['reason']=='FINAL':
                    check_pending=(len(end['prefix_ids']),True)
                    continue
                reason=end.get('termination','FINAL_FORMAT_ERROR') if end else 'EOS_FORMAT_FAILURE'
                break
            if not state['first_check_done'] and arm=='DELAY_2':
                try:
                    bound=protocol.boundary(tokenizer,state['reasoning_ids'],target_step=4)
                except ValueError as exc:
                    reason='STEP_FORMAT_ERROR: '+str(exc)
                    break
                if bound:
                    check_pending=(len(bound['prefix_ids']),False)
                    state['delay_boundary']=bound['step']
            if state['budget']['generated_tokens']%32==0:
                atomic_json(path,state)
        state['timing']['generation_control_seconds']=state['timing'].get('generation_control_seconds',0)+(clock()-generation_start)
        state['termination']=reason
        state['timing']['T_postfork_wall']=clock()-start
        state['timing']['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        state['timing']['common_generated_tokens_inherited']=snapshot['budget']['generated_tokens']
        if state.get('submitted_reasoning_ids') is not None:
            ids=list(state['submitted_reasoning_ids'])
            while ids and ids[-1] in tokenizer.all_special_ids:
                ids.pop()
            state['submitted_text']=reasoning_text(state,tokenizer,0,len(ids))
        finish_arm(cfg,state,row)
        atomic_json(path,state)
        storage.event(run,'ARM_COMPLETED',unique_id=row['unique_id'],arm=arm,Y=state['Y'],termination=reason,
                      seconds=state['timing']['T_postfork_wall'])
        print(f"{state['phase']} {row['unique_id']} {arm} Y={state['Y']} {reason} T={state['timing']['T_postfork_wall']:.3f}",flush=True)
        return state
    except Exception as exc:
        state.update(status='INFRA_INTERRUPTED',error=old.safe_error(exc),completed=False)
        state['timing']['interrupted_wall']=clock()-start
        atomic_json(path,state)
        raise
    finally:
        del session
        gc.collect()


def pilot_checks(cfg,run,snapshot,model,tokenizer,verifier):
    saved=storage.safe_read_json(run/'pilot_checks.json')
    if saved and saved.get('source_hashes')==storage.source_hashes() and saved.get('status')=='PASS':
        return saved
    import torch
    ids=snapshot['prompt_ids']+snapshot['prefix_ids']
    h=feature(model,ids)
    reference=old.extract(model,ids,[len(ids)-1])[18,0]
    max_abs=float(np.max(np.abs(h-reference)))
    if not np.allclose(h,reference,atol=cfg['causal_atol'],rtol=cfg['causal_rtol']):
        raise RuntimeError('Pilot hidden_states[19] prefix extraction differs from stage2; tolerance unchanged')
    base=Session(model,snapshot['prompt_ids'])
    for token in snapshot['prefix_ids']+snapshot['pending_ids']:
        base.ids.append(token)
        base.ensure()
    continuous=base.clone()
    paused=base.clone()
    a,b=[],[]
    max_logits=float((continuous.logits-paused.logits).abs().max().item())
    eos_cfg=model.generation_config.eos_token_id
    eos=set(eos_cfg if isinstance(eos_cfg,list) else [eos_cfg])
    for _ in range(32):
        token=continuous.sample()
        a.append(token)
        if token in eos:
            break
    raw,_,_,_=verifier.verify(snapshot['problem'],'',old.decode(tokenizer,snapshot['prefix_ids']))
    for index in range(len(a)):
        token=paused.sample()
        b.append(token)
    # Same retained cache with a paused PRM call, no control changes.
    pause_match=a==b
    forced=arm_state(snapshot,'SYNTHETIC_SELF_CHECK')
    forced['problem']=snapshot['problem']
    force_session=Session(model,forced['context_ids'])
    forced['timing']={}
    result=prm_check(cfg,forced,force_session,verifier,tokenizer,len(snapshot['prefix_ids']),False,
                     forced={'verdict':'FAIL','first_error_step':1,'diagnosis':'SYNTHETIC SELF-CHECK forced FAIL.',
                             'hint':'SYNTHETIC restart.','step_scores':[]})
    retry=[]
    for _ in range(64):
        token=force_session.sample()
        retry.append(token)
        if token in eos:
            break
    text=old.decode(tokenizer,retry)
    forced_ok=(result is None and forced['reasoning_ids']==[] and forced['checkpoint_step']==0
               and force_session.ids[:len(snapshot['prompt_ids'])]==snapshot['prompt_ids']
               and re.search(r'(?m)^Step 1:',text) is not None
               and forced['budget']['generated_tokens']==snapshot['budget']['generated_tokens'])
    # Document the experiment-only full-prefill numerical effect separately.
    restore=Session(model,snapshot['prompt_ids']+snapshot['prefix_ids']+snapshot['pending_ids'])
    delta=float((base.logits-restore.logits).abs().max().item())
    check={'status':'PASS' if pause_match and forced_ok else 'FAILED','source_hashes':storage.source_hashes(),
           'unique_id':snapshot['unique_id'],'feature':{'layer':19,'max_abs':max_abs,'atol':cfg['causal_atol'],
           'rtol':cfg['causal_rtol'],'pass':True},'pause':{'continuous_ids':a,'paused_ids':b,
           'token_equal':pause_match,'initial_logit_max_abs':max_logits,'prm_output':json.loads(raw)},
           'snapshot_prefill_diagnostic':{'logit_max_abs':delta,'note':'full prefix shape may differ in BF16; both arms use identical restore'},
           'forced_fail':{'pass':forced_ok,'synthetic':True,'excluded_from_statistics':True,'state':forced,
           'retry_ids':retry,'retry_text':text},'created':old.now()}
    atomic_json(run/'pilot_checks.json',check)
    del continuous,paused,base,force_session,restore
    torch.cuda.empty_cache()
    if check['status']!='PASS':
        raise RuntimeError('Pilot GPU logic checks failed; results retained')
    return check


def cpu_checks(cfg,run):
    checks=protocol.self_check()
    storage_checks=storage.self_check()
    if checks is None:
        checks={}
    probes=probe_files(cfg)
    random=np.random.default_rng(cfg['seed'])
    for name,size in (('B',len(probes['B']['mean'])),('C',2)):
        x=random.standard_normal((3,size))
        direct=1/(1+np.exp(-(((x-probes[name]['mean'])/probes[name]['scale'])@probes[name]['w']+probes[name]['b'])))
        assert np.allclose(old.predict(probes[name],x),direct,atol=1e-12)
    equal=old.grade('Step 1: Compute.\n\nFinal answer: \\boxed{2\\sqrt{2}}',r'\sqrt{8}','EOS',5)
    assert equal['label']==0
    malformed=old.grade('Step 1: \\boxed{2}\nFinal answer: nope','2','EOS',5)
    assert malformed['label'] is None and malformed['exclusion']=='MISSING_OR_INCOMPLETE_BOXED'
    snapshot={'unique_id':'synthetic','phase':'pilot','prompt_ids':[10],'prefix_ids':[11,12],'pending_ids':[13],
              'budget':{'generated_tokens':3,'attempt_tokens':3,'forward_tokens':1},'backend':{},'token_hash':'synthetic'}
    a,b=arm_state(snapshot,'NOW'),arm_state(snapshot,'DELAY_2')
    a['reasoning_ids'].clear()
    a['budget']['generated_tokens']+=5
    assert b['reasoning_ids']==[11,12,13] and b['budget']['generated_tokens']==3
    b['budget']['attempt_tokens']=cfg['max_new_tokens']
    assert generator_limit(cfg,b)=='BUDGET_GENERATION_ATTEMPT'
    b.update(termination='BUDGET_NO_REWORK')
    finish_arm(cfg,b,{'answer':'should not be scored'})
    assert b['Y']==0 and b.get('submitted_text') is None
    from unittest.mock import patch
    class TinyTokenizer:
        all_special_ids=[999]
        def decode(self, ids, **kwargs):
            return ''.join({10:'prompt',11:'Step 1: accepted\n',12:'Step 2: new',13:'\nStep 3:',14:' next\nFinal answer: \\boxed{2}',999:'<eos>'}[i] for i in ids)
        def encode(self,text,**kwargs):
            return [88,89]
    class FakeCache:
        def get_seq_length(self):
            return 4
    class FakeSession:
        cache=FakeCache()
        def __init__(self, ids):
            self.ids=list(ids)
        def rollback(self,cp,fb):
            self.ids=list(cp)+list(fb)
            return len(fb)
    class FakeVerifier:
        def input_ids(self,question,accepted,new):
            assert 'feedback' not in accepted+new
            steps,prior=protocol.prm_steps(accepted,new)
            return list(range(len(steps))),[n for n,_ in steps],prior
    tok=TinyTokenizer();ver=FakeVerifier()
    st=arm_state(snapshot,'NOW');st['problem']='synthetic'
    session=FakeSession(st['context_ids'])
    with patch(__name__+'.clock',return_value=0.0):
        assert prm_check(cfg,st,session,ver,tok,2,False,forced={'verdict':'PASS','first_error_step':None}) is None
        assert st['checkpoint_ids']==[10,11,12] and st['checkpoint_step']==2 and st['context_ids']==[10,11,12,13]
        checkpoint=st['checkpoint_ids'].copy()
        assert prm_check(cfg,st,session,ver,tok,3,False,forced={'verdict':'FAIL','first_error_step':1}) is None
        assert st['checks'][-1]['effective_verdict']=='UNCERTAIN' and st['checkpoint_ids']==checkpoint
        assert st['budget']['rollbacks']==0 and st['first_check_done']
        assert prm_check(cfg,st,session,ver,tok,3,False,forced={'verdict':'FAIL','first_error_step':3}) is None
        assert st['context_ids']==checkpoint+[88,89] and st['reasoning_ids']==[11,12]
        assert st['budget']['generated_tokens']==3 and st['budget']['attempt_tokens']==0 and st['budget']['revoked_tokens']==1
        st['reasoning_ids'].append(14);st['reasoning_positions'].append(len(st['context_ids']))
        st['context_ids'].append(14);session.ids.append(14)
        assert 'Step 2: new\n\n next' in reasoning_text(st,tok)
        endpoint=arm_state(snapshot,'DELAY_2');endpoint['problem']='synthetic'
        endpoint['reasoning_ids']=[11,12,14]
        endpoint['context_ids']=[10,11,12,14]
        assert prm_check(cfg,endpoint,FakeSession(endpoint['context_ids']),ver,tok,3,True,
                         forced={'verdict':'UNCERTAIN','first_error_step':None})=='FINAL_UNCERTAIN'
        assert endpoint['first_check_done'] and len(endpoint['checks'])==1 and endpoint['submitted_reasoning_ids']==[11,12,14]
    from .analyze import self_check as analysis_check
    analysis_check()
    output={'status':'PASS','protocol':checks,'storage':storage_checks,'probe_B_layer':19,'probe_predictions_agree':True,
            'math_expression_equivalence':True,'independent_arm_state':True,'actual_check_state_transitions':True,'budget_rejected_answer_not_submitted':True,
            'source_hashes':storage.source_hashes(),'environment':old.environment(False),'created':old.now()}
    atomic_json(run/'self_check.json',output)
    print('SELF_CHECK PASS',flush=True)
    return output


def artifact_hashes(run,phase):
    return {str(p.relative_to(run)):old.digest(p) for folder in ('snapshots','arms')
            for p in sorted((run/folder).glob('*')) if p.suffix in ('.json','.npz')
            and (p.suffix=='.npz' or storage.safe_read_json(p).get('phase')==phase)}


def validate_evaluation_data(cfg,run,row):
    # Post-hoc only, after both timed arms have ended, including algorithm failures.
    from math_verify import LatexExtractionConfig,parse
    from math_verify.utils import TimeoutException
    data=storage.safe_read_json(run/'evaluation_data.json',{})
    uid=row['unique_id']
    if uid not in data:
        try:
            expected=parse('$'+row['answer']+'$',extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                           fallback_mode='no_fallback',extraction_mode='first_match',
                           parsing_timeout=cfg['score_timeout_seconds'],raise_on_error=True)
            result={'status':'VALID' if expected else 'GOLD_UNPARSEABLE'}
        except TimeoutException:
            result={'status':'GOLD_TIMEOUT'}
        except Exception as exc:
            result={'status':'GOLD_ERROR','error':old.safe_error(exc)}
        data[uid]=result
        atomic_json(run/'evaluation_data.json',data)
    if data[uid]['status']!='VALID':
        for arm in ('NOW','DELAY_2'):
            path=run/'arms'/f'{key(uid)}.{arm}.json'
            state=storage.safe_read_json(path)
            state['grading_original']=state['grading']
            state['grading']={**state['grading'],'exclusion':data[uid]['status']}
            state['Y']=None
            atomic_json(path,state)
    return data[uid]


def freeze(cfg,run,manifest):
    from .analyze import freeze_groups
    frozen=storage.safe_read_json(run/'protocol_frozen.json')
    if frozen:
        if frozen['source_hashes']!=storage.source_hashes() or frozen['config_hash']!=manifest['config_hash']:
            raise RuntimeError('Frozen protocol source changed; use new run_id, preserve old run')
        return frozen
    groups=freeze_groups(cfg,run,manifest)
    frozen={'source_hashes':storage.source_hashes(),'config_hash':manifest['config_hash'],
            'probe_hashes':manifest.get('probe_hashes'),'code_commit':old.git('rev-parse','HEAD'), 'verifier_revision':cfg['revisions']['verifier'],
            'dev_groups_sha256':old.digest(run/'dev_groups.json'),'created':old.now(),
            'feedback_adaptation':'checkpoint 0: No reasoning steps have been accepted. Restart from Step 1.; preserve diagnosis/hint',
            'cache_implementation':'DynamicCache.crop; feedback incremental prefill; experimental initial full prefix restore only'}
    atomic_json(run/'protocol_frozen.json',frozen)
    storage.backup(run,'freeze dev groups and protocol before independent test')
    return frozen


def run_phase(cfg,run,manifest,phase,stop_after=None):
    from .analyze import analyze
    check=storage.safe_read_json(run/'self_check.json')
    if not check or check['status']!='PASS' or check['source_hashes']!=storage.source_hashes():
        raise RuntimeError('Current implementation requires passing CPU self-check')
    backup=storage.safe_read_json(run/'backup.json',{})
    if backup.get('status')!='PUSHED':
        storage.backup(run,'resume pending stage3 backup before generation')
    if phase in ('dev','test'):
        pilot=storage.safe_read_json(run/'pilot_checks.json',{})
        resume=storage.safe_read_json(run/'resume_check.json',{})
        if pilot.get('status')!='PASS' or resume.get('status')!='PASS':
            raise RuntimeError('Pilot GPU and independent process resume checks required')
        if pilot.get('source_hashes')!=storage.source_hashes():
            raise RuntimeError('Pilot source changed; rerun pilot checks before dev')
        for row in manifest['splits']['pilot']:
            s=storage.safe_read_json(run/'snapshots'/f'{key(row["unique_id"])}.json',{})
            if s.get('status') not in ('ELIGIBLE','NO_ELIGIBLE_ANCHOR'):
                raise RuntimeError('Complete all pilot IDs before dev/test')
            if s['status']=='ELIGIBLE' and not all(storage.safe_read_json(run/'arms'/f'{key(row["unique_id"])}.{arm}.json',{}).get('completed') for arm in ('NOW','DELAY_2')):
                raise RuntimeError('Incomplete pilot pair')
    if phase=='test':
        freeze(cfg,run,manifest)
    pending_resume=storage.safe_read_json(run/'resume_probe.json')
    if pending_resume and pending_resume['phase']==phase:
        current=artifact_hashes(run,phase)
        unchanged=all(current.get(p)==h for p,h in pending_resume['hashes'].items())
        if not unchanged:
            raise RuntimeError('Completed artifacts changed across independent processes')
        atomic_json(run/'resume_check.json',{'status':'PASS','verified_hashes':pending_resume['hashes'],
                    'independent_process':pending_resume['pid']!=os.getpid(),'source_hashes':storage.source_hashes(),
                    'created':old.now()})
        if pending_resume['pid']==os.getpid():
            raise RuntimeError('Resume verification requires independent process')
    remaining=[]
    for index,row in enumerate(manifest['splits'][phase]):
        snap=storage.safe_read_json(run/'snapshots'/f'{key(row["unique_id"])}.json',{})
        completed=(snap.get('status')=='NO_ELIGIBLE_ANCHOR' or (snap.get('status')=='ELIGIBLE' and all(
            storage.safe_read_json(run/'arms'/f'{key(row["unique_id"])}.{arm}.json',{}).get('completed') for arm in ('NOW','DELAY_2'))))
        if not completed:
            remaining.append((index,row))
    if not remaining:
        analyze(cfg,run,manifest)
        print(f'{phase} all saved results skipped; no generation',flush=True)
        if phase=='dev':
            freeze(cfg,run,manifest)
            run_phase(cfg,run,manifest,'test')
        return
    hardware=old.environment(gpu=True)
    import torch
    backend={'gpu':torch.cuda.get_device_name(0),'total_memory':torch.cuda.get_device_properties(0).total_memory,
             'uuid':str(torch.cuda.get_device_properties(0).uuid),'implementation':cfg['cache_backend']}
    model,tokenizer,verifier=load_resident(cfg,run)
    probes=probe_files(cfg)
    atomic_json(run/'runtime_environment.json',{'environment':hardware,'bitsandbytes':__import__('importlib.metadata',fromlist=['version']).version('bitsandbytes'),
                'backend':backend,'created':old.now()})
    completed_now=0
    try:
        for index,row in remaining:
            snap=initial_snapshot(cfg,run,phase,row,model,tokenizer,probes,backend)
            if snap['status']=='ELIGIBLE':
                if phase=='pilot':
                    pilot_checks(cfg,run,snap,model,tokenizer,verifier)
                paths=[run/'arms'/f'{key(row["unique_id"])}.{arm}.json' for arm in ('NOW','DELAY_2')]
                existing=[storage.safe_read_json(p,{}) for p in paths]
                if any(existing):
                    trial=run/'interrupted_pairs'/key(row['unique_id'])/str(time.time_ns())
                    trial.mkdir(parents=True)
                    for p in paths:
                        if p.exists():
                            p.rename(trial/p.name)
                    storage.event(run,'PAIR_RETIMING_AFTER_INTERRUPTION',unique_id=row['unique_id'],archive=str(trial.relative_to(run)))
                order=('NOW','DELAY_2') if index%2==0 else ('DELAY_2','NOW')
                for arm in order:
                    execute_arm(cfg,run,row,snap,arm,model,tokenizer,verifier,backend)
                validate_evaluation_data(cfg,run,row)
            completed_now+=1
            storage.event(run,'PROBLEM_COMPLETED',unique_id=row['unique_id'],phase=phase,completed_this_process=completed_now)
            if completed_now%cfg['backup_every']==0:
                analyze(cfg,run,manifest)
                storage.backup(run,f'{phase} progress {completed_now} new problems')
            if stop_after and completed_now>=stop_after:
                analyze(cfg,run,manifest)
                atomic_json(run/'resume_probe.json',{'phase':phase,'hashes':artifact_hashes(run,phase),'pid':os.getpid(),'created':old.now()})
                storage.backup(run,f'{phase} stop-after {stop_after}, resume checkpoint')
                return
        analyze(cfg,run,manifest)
        storage.backup(run,f'{phase} complete')
    finally:
        del model,tokenizer,verifier
        gc.collect()
        torch.cuda.empty_cache()
    if phase=='dev':
        freeze(cfg,run,manifest)
        run_phase(cfg,run,manifest,'test')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--phase',choices=['prepare','self-check','pilot','dev','test','analyze','backup'],required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    cfg,run,manifest=storage.prepare(args.config,resume=args.resume)
    print(f'RUN {run}',flush=True)
    try:
        if args.phase=='prepare':
            return
        if args.phase=='self-check':
            cpu_checks(cfg,run)
        elif args.phase=='backup':
            storage.backup(run,'stage3 code and experiment artifacts')
        elif args.phase=='analyze':
            from .analyze import analyze
            analyze(cfg,run,manifest)
        else:
            run_phase(cfg,run,manifest,args.phase,args.stop_after)
    except Exception as exc:
        storage.event(run,'PHASE_INTERRUPTED',phase=args.phase,error=old.safe_error(exc))
        atomic_json(run/'interruption.json',{'phase':args.phase,'error':old.safe_error(exc),'created':old.now(),
                    'resume_command':f'/root/miniconda3/bin/python -m stages.stage3.run_timing --config {args.config} --phase {args.phase} --resume'})
        raise


if __name__=='__main__':
    main()
