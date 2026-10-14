"""Stage3 capacity only: official BF16 4B + 9B and two immutable full snapshots.

Synthetic teacher-forced verifier context covers full 147456+16384 tokens;
this checks capacity, not verifier correctness or a real HIGH/LOW pilot.
"""
import sys,gc,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from veriserve_research import ROOT
from veriserve_research.inference import load_model,Session,GenerativeVerifier,prompt_tokens,clock,snapshot_gpu_check
from veriserve_research.artifacts.io import read_json,atomic_json,now,safe_error
folder=Path(__file__).parent;cfg=read_json(ROOT/'stages/stage3/qwen35_4b_omni_high_low_config.json')
chunk=int(sys.argv[1]) if len(sys.argv)>1 else cfg['prefill_chunk_size'];cfg['prefill_chunk_size']=chunk
output=folder/('stage3_capacity.json' if chunk==2048 else f'stage3_capacity_chunk{chunk}.json')
result=dict(status='RUNNING',created=now(),precision='bfloat16',validation_only=True,formal_stage3_started=False,
 verifier_input=147456,verifier_output_capacity=16384,generator_context=131072,prefill_chunk_size=cfg['prefill_chunk_size'])
atomic_json(output,result)
try:
 model,tok=load_model(cfg);verifier=GenerativeVerifier(cfg)
 torch.cuda.reset_peak_memory_stats();start=clock()
 prompt=Session(model,prompt_tokens(cfg,tok,'Compute 1+1.'),cfg,0);prompt_snapshot=prompt.snapshot();del prompt
 active=Session(model,[tok.encode('a',add_special_tokens=False)[0]]*cfg['max_total_tokens'],cfg,0)
 pass_snapshot=active.snapshot()
 result['phase']='GENERATOR_FULL_WITH_TWO_SNAPSHOTS_VERIFIER_PENDING';atomic_json(output,result)
 vc=cfg['verifier'];v=Session(verifier.model,[verifier.tokenizer.encode('a',add_special_tokens=False)[0]]*vc['max_total_tokens'],cfg,0)
 result.update(status='PASS',seconds=clock()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  free_total_bytes=list(torch.cuda.mem_get_info()),simultaneous_residency=True,two_immutable_checkpoints=True,
  verifier_full_budget_teacher_forced=True,not_sampled_verifier_output=True)
except BaseException as exc:
 result.update(status='FAILED',error=safe_error(exc),peak_allocated_bytes=torch.cuda.max_memory_allocated())
finally:
 result['finished']=now();atomic_json(output,result);print(result,flush=True)
