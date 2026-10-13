"""Real-model interface/restore diagnostics; no scientific trajectories or backup bypass.

python tests/qwen35_migration_check.py --gpu [--capacity]
"""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from veriserve_research import ROOT
from veriserve_research.artifacts.io import read_json,atomic_json,now,digest,safe_error,stable_hash
from veriserve_research.artifacts.provenance import environment,source_hashes
from veriserve_research.inference import (load_model,Session,GenerativeVerifier,prompt_tokens,migration_self_check,
    snapshot_gpu_check,extract,clock,cache_state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--capacity',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    sources={k:source_hashes(k) for k in ('probe','high-low')}
    output=args.output or ROOT/'.cache/qwen35-validation'/stable_hash({'sources':sources,'validator':digest(__file__)})[:12]
    output.mkdir(parents=True,exist_ok=True)
    result=dict(status='RUNNING',created=now(),validation_only=True,scientific_results=False,
                sources=sources,validator_sha256=digest(__file__),cpu=migration_self_check())
    atomic_json(output/'validation.json',result)
    if not args.gpu:
        result.update(status='CPU_PASS_GPU_NOT_RUN',environment=environment())
        atomic_json(output/'validation.json',result)
        print(output,flush=True)
        return
    import torch
    cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_config.json')
    model=tokenizer=verifier=session=None
    try:
        result['environment']=environment(True)
        model,tokenizer=load_model(cfg)
        result['loading']=model.research_loading_info
        result['model_config']=model.config.to_dict()
        ids=prompt_tokens(cfg,tokenizer,'Compute 1 + 1.')
        session=Session(model,ids,cfg,1234)
        generated=[]
        for _ in range(16):
            generated.append(session.sample())
        result['synthetic_interface']=dict(prompt_ids=ids,generated_ids=generated,
            raw_text=tokenizer.decode(generated,skip_special_tokens=False),generation_seed=1234)
        a,b=session.clone(),session.clone()
        expected=[a.sample() for _ in range(8)]
        observed=[b.sample() for _ in range(8)]
        assert expected==observed
        result['pause_same_rng']=dict(passed=True,expected=expected,observed=observed)
        del a,b
        feedback=tokenizer.encode('\nSynthetic verification feedback: recompute.\n',add_special_tokens=False)
        result['restore']=snapshot_gpu_check(session,feedback,[7,8],cfg)
        atomic_json(output/'validation.json',result)
        if result['restore']['status']!='PASS':
            raise RuntimeError('Restore/prefix comparison failed; raw errors retained without tolerance changes')
        prefix=session.ids.copy()
        hidden=extract(model,prefix,[len(prefix)-1])
        result['hidden']=dict(shape=list(hidden.shape),layers=model.config.num_hidden_layers,hidden_size=model.config.hidden_size)
        assert hidden.shape==(32,1,2560)
        del session
        session=None
        gc.collect()
        if args.capacity:
            high_cfg=read_json(ROOT/'stages/stage3/qwen35_4b_omni_high_low_config.json')
            verifier=GenerativeVerifier(high_cfg)
            result['verifier_loading']=verifier.model.research_loading_info
            # Actual simultaneous residency at fixed context/output budgets. No quantization.
            torch.cuda.reset_peak_memory_stats()
            start=clock()
            short=Session(model,ids,cfg,0)
            prompt_checkpoint=short.snapshot()
            del short
            active=Session(model,[tokenizer.eos_token_id]*cfg['max_total_tokens'],cfg,0)
            pass_checkpoint=active.snapshot()
            verifier_ids=[verifier.tokenizer.eos_token_id]*high_cfg['verifier']['max_input_tokens']
            # Capacity requires all 4096 decode tokens, regardless of early synthetic EOS.
            verifier_session=Session(verifier.model,verifier_ids,high_cfg,0)
            for _ in range(high_cfg['verifier']['max_new_tokens']):
                verifier_session.sample()
            result['capacity']=dict(status='PASS',seconds=clock()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),free_total_bytes=list(torch.cuda.mem_get_info()),
                generator_context=cfg['max_total_tokens'],verifier_input=len(verifier_ids),
                verifier_output=high_cfg['verifier']['max_new_tokens'],prefill_chunk_size=cfg['prefill_chunk_size'],
                simultaneous_residency=True,two_immutable_checkpoints=True,precision='bfloat16')
            del active,verifier_session,prompt_checkpoint,pass_checkpoint
        result.update(status='PASS',finished=now())
        atomic_json(output/'validation.json',result)
        print(f'QWEN35 GPU CHECK PASS {output}',flush=True)
    except BaseException as exc:
        result.update(status='FAILED',error=safe_error(exc),finished=now())
        atomic_json(output/'validation.json',result)
        raise
    finally:
        del model,tokenizer,verifier,session
        gc.collect()
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
