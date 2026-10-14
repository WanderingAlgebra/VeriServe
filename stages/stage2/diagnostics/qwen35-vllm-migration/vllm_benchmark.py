"""Fixed 512-token vLLM + original-model endpoint extraction benchmark."""
import sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch,numpy as np
from veriserve_research import ROOT
from veriserve_research.vllm_backend import VLLMGenerator
from veriserve_research.inference import load_model,clock,extract
from veriserve_research.artifacts.io import read_json,atomic_json,now
from veriserve_research.probe.materials import record_path
folder=Path(__file__).parent;cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_vllm_config.json')
old=ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9';manifest=read_json(old/'manifest.json')
r=read_json(record_path(old,'smoke',manifest['splits']['smoke'][0]['unique_id']))
# Engine and HF model are distinct processes with fixed independent environments.
model,tok=load_model(cfg);engine=VLLMGenerator(cfg,folder/'benchmark_engine')
rows=[]
try:
 for rep in range(2):
  start=clock();out=engine.generate(r['prompt_ids'],512,r['generation_seed'],model.generation_config.eos_token_id,ignore_eos=True)
  gen=clock()-start;generated=out['generated_ids'];assert len(generated)==512
  ids=r['prompt_ids']+generated;p=[len(r['prompt_ids'])+k-1 for k in [128,256,512]]
  torch.cuda.reset_peak_memory_stats();start=clock();h=extract(model,ids,p);hidden=clock()-start
  rows.append(dict(repetition=rep,new_tokens=512,generation_including_wake_sleep_ipc_seconds=gen,
   pure_generation_seconds=out['pure_generation_seconds'],tokens_per_second=512/out['pure_generation_seconds'],
   feature_seconds=hidden,combined_seconds=gen+hidden,feature_peak_bytes=torch.cuda.max_memory_allocated(),
   engine_result=out,positions=p))
  atomic_json(folder/'vllm_benchmark.json',dict(status='PASS',engine_initialization=engine.ready,rows=rows,updated=now()))
  print({k:v for k,v in rows[-1].items() if k not in ('engine_result','positions')},flush=True)
 torch.cuda.reset_peak_memory_stats()
 capacity=engine.generate([tok.encode('a',add_special_tokens=False)[0]]*(cfg['max_total_tokens']-16),16,1234,model.generation_config.eos_token_id,ignore_eos=True)
 assert len(capacity['generated_ids'])==16
 saved=read_json(folder/'vllm_benchmark.json');saved['same_seed_repeat_equal']=rows[0]['engine_result']['generated_ids']==rows[1]['engine_result']['generated_ids'];assert saved['same_seed_repeat_equal'];saved['context_capacity']=dict(status='PASS',context_tokens=cfg['max_total_tokens'],result=capacity)
 atomic_json(folder/'vllm_benchmark.json',saved)
finally:engine.close()
