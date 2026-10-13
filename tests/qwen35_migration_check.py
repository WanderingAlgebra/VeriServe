"""Real-model interface/restore diagnostics; no scientific trajectories or backup bypass.

python tests/qwen35_migration_check.py --gpu [--capacity] [--saved-run RUN] [--benchmark-run RUN]
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
    parser.add_argument('--saved-run',type=Path,help='Compare first smoke record NPZ endpoints with independent original prefixes')
    parser.add_argument('--benchmark-run',type=Path,help='Time fixed original prefixes under the active backend; diagnostic only')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if (args.saved_run or args.benchmark_run) and not args.gpu:
        parser.error('--saved-run and --benchmark-run require --gpu')
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
        if args.saved_run:
            import numpy as np
            from veriserve_research.probe.materials import record_path,read_record
            run=args.saved_run.resolve()
            manifest=read_json(run/'manifest.json')
            if manifest['config']!=cfg or manifest['source_hashes']!=sources['probe']:
                raise RuntimeError('Saved feature run differs from the frozen model/config/source')
            path=record_path(run,'smoke',manifest['splits']['smoke'][0]['unique_id'])
            before=digest(path)
            record=read_record(path,run,recover=False)
            steps=record['boundaries']['steps']
            positions=[len(record['prompt_ids'])+s['end_token_index'] for s in steps]
            if record['boundaries'].get('end') is not None:
                positions.append(len(record['prompt_ids'])+record['boundaries']['end']['end_token_index'])
            if not steps or not record.get('feature_files'):
                raise RuntimeError('First smoke has no saved intermediate features to verify')
            targets=sorted({0,len(positions)-1})
            saved={}
            for entry in record['feature_files']:
                file=path.parent/entry['name']
                if digest(file)!=entry['sha256']:
                    raise RuntimeError('Saved NPZ hash mismatch')
                with np.load(file,allow_pickle=False) as data:
                    assert data['positions'].tolist()==positions
                    for layer,values in zip(data['layers'],data['hidden']):
                        assert int(layer) not in saved
                        saved[int(layer)]=values[targets].copy()
            assert sorted(saved)==list(range(1,model.config.num_hidden_layers+1))
            actual=extract(model,record['prompt_ids']+record['generated_ids'],[positions[i] for i in targets])
            stored=np.stack([saved[layer] for layer in sorted(saved)])
            delta=np.abs(actual-stored)
            check=dict(passed=bool(np.allclose(actual,stored,atol=cfg['causal_atol'],rtol=cfg['causal_rtol'])),
                record_sha256=before,unique_id=record['unique_id'],positions=[positions[i] for i in targets],
                layers=sorted(saved),shape=list(actual.shape),max_abs=float(delta.max()),
                rms=float(np.sqrt(np.mean(delta**2))),atol=cfg['causal_atol'],rtol=cfg['causal_rtol'],
                source_unchanged=digest(path)==before,feature_hashes=record['feature_files'])
            result['saved_features']=check
            atomic_json(output/'validation.json',result)
            if not check['passed'] or not check['source_unchanged']:
                raise RuntimeError('Disk features differ from original-prefix reconstruction; original tolerances retained')
        if args.benchmark_run:
            import inspect
            import numpy as np
            from transformers.models.qwen3_5 import modeling_qwen3_5 as kernels
            from veriserve_research.probe.materials import record_path,read_record
            run=args.benchmark_run.resolve()
            manifest=read_json(run/'manifest.json')
            if manifest['config']!=cfg or manifest['source_hashes']!=sources['probe']:
                raise RuntimeError('Benchmark source differs from the frozen model/config/source')
            path=record_path(run,'smoke',manifest['splits']['smoke'][0]['unique_id'])
            before=digest(path)
            record=read_record(path,run,recover=False)
            backend={}
            for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
                fn=getattr(kernels,name)
                while fn is not None:
                    closure=inspect.getclosurevars(fn).nonlocals
                    if 'implementation' in closure:
                        break
                    fn=getattr(fn,'__wrapped__',None)
                if fn is None:
                    raise RuntimeError('Cannot identify the pinned Transformer kernel dispatcher')
                backend[name]=dict(module=closure['implementation'].__module__,
                                   optimized=closure['is_new_implementation'])
            forced=record['generated_ids'][:256]
            assert len(forced)==256
            repetitions=[]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                start=clock()
                bench=Session(model,record['prompt_ids'],cfg,record['generation_seed'])
                prefill=clock()-start
                start=clock()
                for token in forced:
                    bench.ids.append(token)
                    assert bench.ensure()==1
                seconds=clock()-start
                repetitions.append(dict(prefill_seconds=prefill,teacher_forced_decode_seconds=seconds,
                                        tokens_per_second=len(forced)/seconds))
                del bench
            steps=record['boundaries']['steps']
            targets=[steps[0],steps[len(steps)//2],record['boundaries'].get('end') or steps[-1]]
            positions=sorted({len(record['prompt_ids'])+s['end_token_index'] for s in targets})
            reforward=[]
            for _ in range(2):
                start=clock()
                hidden=extract(model,record['prompt_ids']+record['generated_ids'],positions)
                reforward.append(clock()-start)
                assert hidden.shape==(32,len(positions),2560) and np.isfinite(hidden).all()
                del hidden
            result['benchmark']=dict(unique_id=record['unique_id'],record_sha256=before,backend=backend,
                forced_token_sha256=stable_hash(forced),decode_tokens=len(forced),repetitions=repetitions,
                independent_prefix_positions=positions,reforward_seconds=reforward,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_unchanged=digest(path)==before,
                scope='Fixed-token decoder and original-prefix extraction only; no sampling, labels or T_request comparison.')
            assert result['benchmark']['source_unchanged']
            atomic_json(output/'validation.json',result)
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
