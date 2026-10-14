"""Existing Stage2 CLI sequence; smoke gates first, formal + A/B/C next, no Stage3 run."""
import os,sys,subprocess,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from veriserve_research.artifacts.io import atomic_json,read_json,now,safe_error
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.probe.collect import prepare
PYTHON=ROOT/'environments/qwen35/.venv/bin/python'
CONFIG=ROOT/'stages/stage2/qwen35_4b_omni_vllm_config.json'
DIAG=ROOT/'stages/stage2/diagnostics/qwen35-vllm-migration'
FROZEN=source_hashes('probe')
RUN=Path(sys.argv[1]).resolve()
STATE=RUN/'pipeline_status.json'
BASE=[str(PYTHON),'-m','veriserve_research','probe','--config',str(CONFIG)]

def status(phase,**extra):
 counts={split:len(list((RUN/'records'/split).glob('*.json'))) for split in ('smoke','train','test')}
 state=dict(phase=phase,pid=os.getpid(),updated=now(),run=str(RUN),source_hashes=FROZEN,counts=counts,**extra)
 atomic_json(STATE,state);print(state,flush=True)
 p=ROOT/'stages/stage2/qwen35_status.json';old=read_json(p,{})
 old.update(status=phase,run=str(RUN.relative_to(ROOT)),updated=now(),pipeline_pid=os.getpid(),
  pipeline_status=str(STATE.relative_to(ROOT)),formal_generated=counts['train']+counts['test'],
  smoke_generated=counts['smoke'],migration_diagnostics=str(DIAG.relative_to(ROOT)),
  old_run_preserved='stages/stage2/runs/v2-20261013-78a47750b3b9')
 atomic_json(p,old)

def command(phase,args):
 if source_hashes('probe')!=FROZEN:raise RuntimeError('Execution code changed; preserve run and start a new identity')
 status(phase)
 subprocess.run(args,cwd=ROOT,check=True,env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})

try:
 cfg,actual,manifest=prepare(CONFIG)
 assert actual.resolve()==RUN
 # Regenerate budget/restore checks under the final frozen source; no verifier requests.
 command('GENERATOR_CAPACITY_AND_RESTORE',[str(PYTHON),'tests/qwen35_migration_check.py','--gpu','--generator-only','--output',str(RUN/'gpu_validation')])
 hooks=read_json(DIAG/'endpoint_hooks.json');bench=read_json(DIAG/'vllm_benchmark.json');base=read_json(DIAG/'baseline_benchmark.json')
 assert read_json(DIAG/'sampling.json')['status']=='PASS'
 assert hooks['complete'] and all(r['exact_equal'] for r in hooks['rows'])
 assert bench['status']=='PASS' and bench['context_capacity']['status']=='PASS'
 assert bench['rows'][-1]['combined_seconds']<base['rows'][-1]['combined_seconds']
 atomic_json(RUN/'migration_gate.json',dict(status='SHORT_CHECKS_PASS_SMOKE_PENDING',source_hashes=FROZEN,
  exact_endpoint_hooks=hooks,benchmark=bench,baseline=base,
  incremental_feature_candidate='REJECTED; original allclose gate failed; independent prefixes explicitly retained',updated=now()))
 for phase,flags in [('SMOKE_FIRST_QUESTION',['--smoke','--resume','--stop-after','1']),
                     ('SMOKE_RESTART_AND_COMPLETE',['--smoke','--resume']),
                     ('SMOKE_ALL_RECORDS_RESUME_AUDIT',['--smoke','--resume'])]:
  command(phase,BASE+flags)
 smoke=read_json(RUN/'smoke_diagnostics.json');check=read_json(RUN/'smoke_checks.json');resume=read_json(RUN/'resume_check.json')
 original=read_json(ROOT/'stages/stage2/runs/v2-20261013-78a47750b3b9/smoke_diagnostics.json')
 if not check['passed'] or not resume['passed']:
  raise RuntimeError(f"Smoke engineering gate failed: {check.get('completed_parseable_questions',0)}/10; formal not started")
 if smoke['generation_and_extraction_seconds']>=original['generation_and_extraction_seconds']:
  raise RuntimeError('Fixed-cohort smoke generation + features shows no end-to-end benefit; formal not started')
 gate=read_json(RUN/'migration_gate.json');gate.update(status='PASS',smoke=smoke,
  original_fixed_smoke_seconds=original['generation_and_extraction_seconds'],
  measured_smoke_speedup=original['generation_and_extraction_seconds']/smoke['generation_and_extraction_seconds'],
  comparison='Same fixed questions and budgets, independent engine trajectories; lengths may differ. No mixed statistics.',updated=now())
 atomic_json(RUN/'migration_gate.json',gate)
 command('STAGE2_FORMAL_COLLECTION_AND_FIT',BASE+['--resume'])
 status('STAGE2_COMPLETE_STAGE3_NOT_STARTED',probe_B_frozen=(RUN/'probe_B.json').exists())
except BaseException as exc:
 status('STOPPED',error=safe_error(exc),exception_type=type(exc).__name__);traceback.print_exc();raise
