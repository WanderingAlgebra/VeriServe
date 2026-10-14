"""Original BF16 sampler, fixed 512 new tokens including EOS; no scientific record."""
import sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch,numpy as np
from veriserve_research import ROOT
from veriserve_research.inference import load_model,Session,clock,extract
from veriserve_research.artifacts.io import read_json,atomic_json,now
from veriserve_research.probe.materials import record_path
folder=Path(__file__).parent;cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_config.json')
old=ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9';manifest=read_json(old/'manifest.json')
r=read_json(record_path(old,'smoke',manifest['splits']['smoke'][0]['unique_id']))
start=time.perf_counter();model,tok=load_model(cfg);load=time.perf_counter()-start
rows=[]
for rep in range(2):
 torch.cuda.reset_peak_memory_stats();start=clock();s=Session(model,r['prompt_ids'],cfg,r['generation_seed'])
 generated=[s.sample() for _ in range(512)];gen=clock()-start;genpeak=torch.cuda.max_memory_allocated();del s
 ids=r['prompt_ids']+generated;p=[len(r['prompt_ids'])+k-1 for k in [128,256,512]]
 torch.cuda.reset_peak_memory_stats();start=clock();h=extract(model,ids,p);hidden=clock()-start
 rows.append(dict(repetition=rep,new_tokens=512,generation_seconds=gen,tokens_per_second=512/gen,
  feature_seconds=hidden,combined_seconds=gen+hidden,generation_peak_bytes=genpeak,
  feature_peak_bytes=torch.cuda.max_memory_allocated(),generated_ids=generated,positions=p))
 atomic_json(folder/'baseline_benchmark.json',dict(status='PASS',model_load_seconds=load,rows=rows,updated=now()))
 print({k:v for k,v in rows[-1].items() if k not in ('generated_ids','positions')},flush=True)
