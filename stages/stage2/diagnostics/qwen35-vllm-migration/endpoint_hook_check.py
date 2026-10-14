import sys,time,json
from pathlib import Path
sys.path.insert(0,'/root/project/VeriServe-step-hidden-probe')
import numpy as np,torch
from veriserve_research.inference import load_model,full_forward,clock
from veriserve_research.artifacts.io import read_json,atomic_json,now
from veriserve_research import ROOT
cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_config.json');model,tok=load_model(cfg)
from veriserve_research.probe.materials import record_path
old=ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9';m=read_json(old/'manifest.json')
r=read_json(record_path(old,'smoke',m['splits']['smoke'][0]['unique_id']))
ids=r['prompt_ids']+r['generated_ids'];rows=[]
for p in [127, len(r['prompt_ids'])+r['boundaries']['steps'][1]['end_token_index'],len(r['prompt_ids'])+r['boundaries']['end']['end_token_index']]:
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=clock()
 with torch.inference_mode():
  x=torch.tensor([ids[:p+1]],device='cuda');o=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,output_hidden_states=True,return_dict=True)
  a=torch.stack([h[0,-1] for h in o.hidden_states[1:]]).float().cpu().numpy()
 seconds=clock()-start;peak=torch.cuda.max_memory_allocated();del o,x
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=clock();b=full_forward(model,ids[:p+1],[p])[:,0]
 d=np.abs(a-b);rows.append(dict(position=p,max_abs=float(d.max()),exact_equal=bool(np.array_equal(a,b)),old_seconds=seconds,new_seconds=clock()-start,old_peak_bytes=peak,new_peak_bytes=torch.cuda.max_memory_allocated(),min_cosine=float(np.min(np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))))))
 atomic_json(ROOT/'stages/stage2/diagnostics/qwen35-vllm-migration/endpoint_hooks.json',dict(status='PASS' if all(r['exact_equal'] for r in rows) else 'FAILED',rows=rows,complete=len(rows)==3,updated=now()))
 print(rows[-1],flush=True)
