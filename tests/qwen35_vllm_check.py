"""Fixed original-token feature comparison; never writes old runs."""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from veriserve_research import ROOT
from veriserve_research.inference import load_model, extract, extract_incremental, clock, Session, snapshot_gpu_check
from veriserve_research.artifacts.io import read_json, atomic_json, digest, now
from veriserve_research.artifacts.provenance import environment, source_hashes


def errors(a,b,cfg):
    d=np.abs(a-b)
    relative=np.linalg.norm(a-b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1),1e-12)
    cosine=np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
    return dict(max_abs=float(d.max()), rms=float(np.sqrt(np.mean(d*d))),
        max_relative_l2=float(relative.max()),min_cosine=float(cosine.min()),
        max_abs_per_layer=d.max(axis=(1,2)).tolist(), relative_l2_per_layer=relative.max(axis=1).tolist(),
        cosine_per_layer=cosine.min(axis=1).tolist(),
        passed=bool(np.allclose(a,b,atol=cfg['causal_atol'],rtol=cfg['causal_rtol']) and cosine.min()>=.999))


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'stages/stage2/diagnostics/qwen35-vllm-migration')
    args=parser.parse_args(); folder=args.output; folder.mkdir(parents=True,exist_ok=True)
    cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_config.json')
    old=ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9'
    manifest=read_json(old/'manifest.json')
    from veriserve_research.probe.materials import record_path
    path=record_path(old,'smoke',manifest['splits']['smoke'][0]['unique_id'])
    r=read_json(path); ids=r['prompt_ids']+r['generated_ids']
    # Determined before model execution; preserve the existing elementwise tolerance.
    result=dict(status='RUNNING',created=now(),acceptance=dict(atol=cfg['causal_atol'],rtol=cfg['causal_rtol'],
        min_cosine=.999,rule='all coordinates allclose AND every layer/endpoint cosine >= .999; no post-hoc relaxation'),
        environment=environment(True),sources=source_hashes('probe'),original_record_sha256=digest(path),
        mapping={str(i):('HF hidden_states[%d], decoder[%d] output before final norm'%(i,i-1) if i<32
            else 'HF hidden_states[32], decoder[31] output AFTER final RMSNorm') for i in range(1,33)})
    atomic_json(folder/'features.json',result)
    model,tok=load_model(cfg)
    cases=[('short',[127,255,511]),('real_paragraphs', [len(r['prompt_ids'])+s['end_token_index'] for s in r['boundaries']['steps'][:3]]),
           ('complete_A',[len(r['prompt_ids'])+r['boundaries']['end']['end_token_index']])]
    result['cases']=[]
    for name,positions in cases:
        torch.cuda.reset_peak_memory_stats(); start=clock(); a=extract(model,ids,positions); baseline=clock()-start
        baseline_peak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats(); start=clock(); b=extract_incremental(model,ids,positions,chunk_size=cfg['prefill_chunk_size']); seconds=clock()-start
        metrics=errors(a,b,cfg)
        changed=ids[:positions[-1]+1]+[42]*17
        future=extract_incremental(model,changed,positions,chunk_size=cfg['prefill_chunk_size'])
        metrics.update(name=name,positions=positions,original_prefix_seconds=baseline,incremental_seconds=seconds,
            original_peak_bytes=baseline_peak,incremental_peak_bytes=torch.cuda.max_memory_allocated(),
            future_exact_equal=bool(np.array_equal(b,future)))
        result['cases'].append(metrics); atomic_json(folder/'features.json',result)
        print(name,json.dumps(metrics),flush=True)
    short=Session(model,ids[:128],cfg,1234)
    result['restore']=snapshot_gpu_check(short,tok.encode('\nCheck again.\n',add_special_tokens=False),[7,8],cfg)
    result['old_record_unchanged']=digest(path)==result['original_record_sha256']
    result.update(status='PASS' if all(c['passed'] and c['future_exact_equal'] for c in result['cases']) and result['restore']['status']=='PASS' else 'FAILED',finished=now())
    atomic_json(folder/'features.json',result)

if __name__=='__main__': main()
