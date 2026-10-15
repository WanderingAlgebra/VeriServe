"""Operational CUDA allocator test; no frozen execution source/config edits."""
import sys,os,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np,torch
from veriserve_research import ROOT
from veriserve_research.artifacts.io import read_json,atomic_json,now,digest,safe_error
from veriserve_research.inference import load_model,extract,clock
from veriserve_research.vllm_backend import VLLMGenerator
folder=Path(__file__).parent
profile=sys.argv[1]
if profile=='gc01fraction1':
 torch.cuda.set_per_process_memory_fraction(1.0)
run=ROOT/'stages/stage2/runs/v2-20261013-854c63b5c942';cfg=read_json(ROOT/'stages/stage2/qwen35_4b_omni_vllm_config.json')
p=next((run/'records/train').glob('*.json'));r=read_json(p);original_hash=digest(p)
result=dict(status='RUNNING',created=now(),profile=profile,allocator_env=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
 original_record_sha256=original_hash,source_config_unchanged=True,trials=[])
output=folder/f'allocator-{profile}.json';atomic_json(output,result)
model,tok=load_model(cfg);engine=VLLMGenerator(cfg,folder/f'allocator-{profile}-engine')
def memory():
 return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0])
try:
 first=engine.generate(r['prompt_ids'],256,r['generation_seed'],model.generation_config.eos_token_id,ignore_eos=True)
 ids=r['prompt_ids']+r['generated_ids'];positions=[len(r['prompt_ids'])+s['end_token_index'] for s in r['boundaries']['steps']+[r['boundaries']['end']]]
 torch.cuda.empty_cache();start=clock();hidden=extract(model,ids,positions);seconds=clock()-start
 saved=np.concatenate([np.load(p.parent/e['name'])['hidden'] for e in r['feature_files']],axis=0)
 exact=np.array_equal(saved,hidden);assert exact
 result['saved_feature_exact']=exact;result['first_feature_seconds']=seconds
 del hidden,saved
 result['before_wake_after_real_features']=memory();atomic_json(output,result)
 second=engine.generate(r['prompt_ids'],256,r['generation_seed'],model.generation_config.eos_token_id,ignore_eos=True)
 assert first['generated_ids']==second['generated_ids']
 result['trials'].append(dict(name='real_46_endpoints_then_wake',passed=True,raw_tokens_repeat_exact=True,engine=second))
 atomic_json(output,result)
 if profile!='default':
  # One full large prefix deliberately raises HF workspace needs, then wake again.
  limit=read_json(folder/'fixed_prompt_capacity.json')['max_single_trajectory_tokens']
  long_ids=(ids*((limit+len(ids)-1)//len(ids)))[:limit]
  torch.cuda.empty_cache();start=clock();hidden=extract(model,long_ids,[len(long_ids)-1]);seconds=clock()-start
  assert np.isfinite(hidden).all();del hidden,long_ids
  result['before_wake_after_long_prefix']=memory();result['long_prefix_tokens']=limit;atomic_json(output,result)
  third=engine.generate(r['prompt_ids'],256,r['generation_seed'],model.generation_config.eos_token_id,ignore_eos=True)
  assert third['generated_ids']==first['generated_ids']
  result['trials'].append(dict(name='maximum_fixed_cohort_prefix_then_wake',passed=True,feature_seconds=seconds,engine=third))
 result.update(status='PASS',original_record_unchanged=digest(p)==original_hash,finished=now())
except BaseException as exc:
 result.update(status='FAILED',error=safe_error(exc),finished=now())
finally:
 atomic_json(output,result);engine.close();print({k:v for k,v in result.items() if k!='trials'},flush=True)
