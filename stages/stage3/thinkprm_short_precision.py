"""One fixed eight-input BF16/NF4 synthetic comparison, with a short context."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from stages.stage3.thinkprm_prefix_trial import (
    ROOT, MODEL, REV, SCHEMA, read, save, sha, parse_final, inspect_quantized_model)

RUN=ROOT/'stages/stage3/diagnostics/thinkprm-14b-bnb4-prefix-audit/20261110-bf16-nf4-short-control'
OUTPUT_BUDGET=2048
CASES=[
    ('division','Solve for x: 2x + 3 = 7',
     'Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 2',
     'Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 1',
     'Identify Step 2 and explain that 4/2=2, not 1.'),
    ('sign','Simplify -(x - 2) for real x.',
     'Step 1: Distribute the minus sign: -(x - 2) = -x + 2.',
     'Step 1: Distribute the minus sign: -(x - 2) = -x - 2.',
     'Identify the constant sign error: the second term must be +2.'),
    ('possible_zero','Solve x^2 = x over the real numbers.',
     'Step 1: Rewrite the equation as x(x - 1) = 0.\nStep 2: Therefore, x = 0 or x = 1.',
     'Step 1: Divide both sides by x to obtain x = 1.\nStep 2: Therefore, the only solution is x = 1.',
     'Identify unjustified division by possibly-zero x and the lost solution x=0.'),
    ('invalid_inference','An integer n is known to be even. Must n be divisible by 4?',
     'Step 1: Take n = 2. It is even but is not divisible by 4.\nStep 2: Therefore, the answer is no.',
     'Step 1: Every integer divisible by 4 is even.\nStep 2: Since n is even, it follows that n is divisible by 4, so the answer is yes.',
     'Identify the invalid converse; an even integer need not be divisible by 4, e.g. n=2.'),
]


def card_prompt(tok, problem, prefix):
    user=f'''You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{prefix}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps
'''
    text=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,
        add_generation_prompt=True)+"\nLet's verify step by step:"
    ids=tok.encode(text,add_special_tokens=False)
    assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text
    return text,ids


def prepare(compact_final=False):
    from transformers import AutoTokenizer
    assert not (RUN/'manifest.json').exists(), 'Do not overwrite a prepared comparison'
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    inventory=[];rubric=[]
    for group,problem,correct,incorrect,criterion in CASES:
        for variant,prefix in [('correct',correct),('incorrect',incorrect)]:
            key=f'{len(inventory)+1:02}_{group}_{variant}'
            text,ids=card_prompt(tok,problem,prefix)
            req=dict(key=key,problem=problem,prefix=prefix,prompt=text,input_ids=ids,
                input_tokens=len(ids),schema=SCHEMA)
            save(RUN/'requests'/f'{key}.json',req)
            inventory.append(dict(key=key,input_tokens=len(ids),request_sha256=sha(RUN/'requests'/f'{key}.json')))
            rubric.append(dict(key=key,group=group,variant=variant,
                expected='PASS' if variant=='correct' else 'FAIL',
                reasoning_criterion=criterion if variant=='incorrect' else 'Do not report a nonexistent mathematical error.'))
    assert len(inventory)==8 and len({x['key'] for x in inventory})==8
    config=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',
        max_model_len=max(x['input_tokens'] for x in inventory)+OUTPUT_BUDGET,
        max_num_seqs=1,max_num_batched_tokens=256,enable_chunked_prefill=True,
        kv_cache_memory_bytes=768*1024**2,gpu_memory_utilization=.97,
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',
        structured_outputs_config=dict(backend='guidance',reasoning_parser='deepseek_r1',
            enable_in_reasoning=False,disable_any_whitespace=compact_final))
    sampling=dict(temperature=0.,top_p=1.,top_k=0,min_p=0.,repetition_penalty=1.,
        presence_penalty=0.,frequency_penalty=0.,max_tokens=OUTPUT_BUDGET,seed=20261110)
    save(RUN/'rubric.json',dict(cases=rubric,
        primary='Assess the reasoning body before reading final JSON. FAIL alone is insufficient: require the concrete actual error.',
        correct_case='Record false alarms, ambiguous reasoning, and incomplete output separately.',
        endpoint='One BF16 initialization attempt and at most eight requests; one NF4 initialization and at most eight requests. No adaptive prompting or repeated seeds.',
        finite_synthetic_diagnostic=True,source_audit=False))
    save(RUN/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model='launch/ThinkPRM-14B',revision=REV,config=config,sampling=sampling,
        inventory=inventory,implementation_sha256=sha(__file__),helper_sha256=sha(ROOT/'stages/stage3/thinkprm_prefix_trial.py'),
        rubric_sha256=sha(RUN/'rubric.json'),
        versions={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','bitsandbytes','llguidance']},
        precisions=dict(bf16=dict(quantization=None,load_format='safetensors'),
                        nf4=dict(quantization='bitsandbytes',load_format='bitsandbytes')),
        prompt='Original model-card template; no added JSON instructions. Same final-only guidance Schema for both precisions.',
        memory='Fixed 768 MiB BF16 KV cache for both precisions; explicit KV bytes override utilization for KV allocation.',
        changes_from_previous_trial='Short context, prefill256, output2048, fixed KV768MiB. Original division cases retained; no source-data collection.',
        technical_repair=('One compact-final repair after preliminary final-whitespace loops; original evidence and source retained in parent directory. No further adaptive retries.' if compact_final else None),
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],text=True)))
    print('PREPARED',len(inventory),'CONTEXT',config['max_model_len'],'MAX_INPUT',config['max_model_len']-OUTPUT_BUDGET,flush=True)


def execute(precision):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.reasoning import ReasoningParserManager
    from veriserve_research.vllm_backend import DevicePeak
    m=read(RUN/'manifest.json');folder=RUN/precision
    assert sha(__file__)==m['implementation_sha256']
    assert sha(ROOT/'stages/stage3/thinkprm_prefix_trial.py')==m['helper_sha256']
    assert sha(RUN/'rubric.json')==m['rubric_sha256']
    assert not (folder/'start.json').exists(), 'One attempt only; preserve existing evidence'
    config={**m['config'],**m['precisions'][precision]}
    save(folder/'start.json',dict(config=config,sampling=m['sampling'],started_utc=datetime.now(timezone.utc).isoformat()))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    parser=ReasoningParserManager.get_reasoning_parser('deepseek_r1')(tok)
    peak=DevicePeak();start=time.perf_counter()
    try:
        llm=LLM(**config);torch.cuda.synchronize()
        evidence=llm.apply_model(inspect_quantized_model)
        for item in evidence:
            if precision=='nf4':
                assert item['quantized_linear_count']==192 and item['quant_types']==['nf4']
            else:
                assert item['quantized_linear_count']==0 and list(item['parameter_storage_dtypes'])==['torch.bfloat16']
        save(folder/'engine.json',dict(load_seconds=time.perf_counter()-start,
            actual_weights=evidence,device_peak_bytes=peak.stop(),device_peak_sampling_ms=20))
    except Exception as e:
        save(folder/'engine_failure.json',dict(error=str(e),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-start,device_peak_bytes=peak.stop()))
        raise
    completed=0
    for item in m['inventory']:
        key=item['key'];path=RUN/'requests'/f'{key}.json'
        assert sha(path)==item['request_sha256']
        req=read(path);ids=req['input_ids']
        options=SamplingParams(**m['sampling'],skip_special_tokens=False,
            structured_outputs=StructuredOutputsParams(json=SCHEMA))
        peak=DevicePeak();start=time.perf_counter()
        record=dict(key=key,precision=precision,input_tokens=len(ids),status='service_error')
        try:
            response=llm.generate([dict(prompt_token_ids=ids)],options,use_tqdm=False)[0]
            torch.cuda.synchronize();out=response.outputs[0];generated=list(out.token_ids)
            raw=tok.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            eos=bool(generated and generated[-1]==tok.eos_token_id)
            clean=generated[:-1] if eos else generated
            reasoning,final=parser.extract_reasoning(tok.decode(clean,skip_special_tokens=False,
                clean_up_tokenization_spaces=False),None)
            judgment,status=parse_final(final,out.finish_reason)
            end=parser.end_token_id in generated;matched=list(response.prompt_token_ids)==ids
            if judgment is not None and not (eos and end and matched):
                judgment,status=None,'invalid_termination_or_input'
            reasoning_tokens=generated.index(parser.end_token_id) if end else len(generated)
            record.update(raw_output=raw,generated_ids=generated,reasoning=reasoning,final_output=final,
                output_tokens=len(generated),reasoning_tokens=reasoning_tokens,
                final_tokens_including_eos=max(0,len(generated)-reasoning_tokens-1),
                judgment=judgment,status=status,finish_reason=out.finish_reason,stop_reason=out.stop_reason,
                eos_observed=eos,reasoning_end_observed=end,returned_prompt_ids_match=matched)
        except Exception as e:
            record.update(error=str(e),traceback=traceback.format_exc())
        record.update(seconds=time.perf_counter()-start,device_peak_bytes=peak.stop(),device_peak_sampling_ms=20)
        save(folder/f'{key}.json',record);completed+=1
        print(json.dumps({k:v for k,v in record.items() if k not in
            ['raw_output','generated_ids','reasoning','final_output','traceback']}),flush=True)
        if record['status']=='service_error':
            break
    save(folder/'completed.json',dict(requests=completed,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','bf16','nf4'])
    p.add_argument('--compact-final',action='store_true');args=p.parse_args()
    if args.compact_final:
        RUN=RUN/'compact-final'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION','1')
    prepare(args.compact_final) if args.phase=='prepare' else execute(args.phase)
