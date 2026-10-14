"""Public worker RPC diagnostic, no vLLM engine edits; one endpoint per layer."""
import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np,torch
from veriserve_research import ROOT
from veriserve_research.artifacts.io import read_json,atomic_json,now
from veriserve_research.inference import load_model,extract
from veriserve_research.vllm_backend import VLLMGenerator
from veriserve_research.probe.materials import record_path
folder=Path(__file__).parent;cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_vllm_config.json')
cfg.update(vllm_enforce_eager=True,vllm_async_scheduling=False,vllm_diagnostic_callbacks=True)
old=ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9';m=read_json(old/'manifest.json')
r=read_json(record_path(old,'smoke',m['splits']['smoke'][0]['unique_id']));ids=r['prompt_ids']+r['generated_ids']
positions=[127,len(r['prompt_ids'])+r['boundaries']['steps'][1]['end_token_index'],len(r['prompt_ids'])+r['boundaries']['end']['end_token_index']]
result=dict(status='RUNNING',created=now(),acceptance=dict(atol=cfg['causal_atol'],rtol=cfg['causal_rtol'],min_cosine=.999),
 implementation='public LLM.apply_model hooks, eager diagnostic, overwrite only latest endpoint per layer',positions=positions,rows=[])
atomic_json(folder/'direct_hidden.json',result)
model,tok=load_model(cfg);engine=VLLMGenerator(cfg,folder/'direct_hidden_engine')
try:
 for i,p in enumerate(positions):
  path=folder/f'direct-endpoint-{p}.npz'
  out=engine.generate(ids[:p+1],1,1234,model.generation_config.eos_token_id,ignore_eos=True,capture_path=path.resolve())
  capture=out['hidden_capture'][0];assert capture['position']==p and capture['token_id']==ids[p],capture
  a=extract(model,ids,[p]);b=np.load(path)['hidden'];d=np.abs(a-b)
  relative=np.linalg.norm(a-b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1),1e-12)
  cosine=np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
  result['rows'].append(dict(position=p,token_aligned=True,max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),
   max_relative_l2=float(relative.max()),min_cosine=float(cosine.min()),max_abs_per_layer=d.max(axis=(1,2)).tolist(),
   passed=bool(np.allclose(a,b,atol=cfg['causal_atol'],rtol=cfg['causal_rtol']) and cosine.min()>=.999)))
  atomic_json(folder/'direct_hidden.json',result);print(result['rows'][-1],flush=True)
 result.update(status='PASS' if all(row['passed'] for row in result['rows']) else 'REJECTED_NUMERICAL_EQUIVALENCE',finished=now())
 atomic_json(folder/'direct_hidden.json',result)
finally:engine.close()
