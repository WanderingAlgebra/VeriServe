"""One-shot sequential execution of the authorized migration, stopping for human audit."""
import os,sys,time,subprocess,traceback,json
from pathlib import Path
ROOT=Path('/root/project/VeriServe-step-hidden-probe')
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from veriserve_research.artifacts.io import read_json,atomic_json,now,safe_error,digest
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.probe.collect import prepare
PYTHON=ROOT/'environments/qwen35/.venv/bin/python'
PROBE=ROOT/'stages/stage2/qwen35_4b_omni_config.json'
HIGH=ROOT/'stages/stage3/qwen35_4b_omni_high_low_config.json'
RUN=ROOT/read_json(HIGH)['stage2_run']
FROZEN=source_hashes('probe')
STATE=RUN/'pipeline_status.json'
BASE=[str(PYTHON),'-m','veriserve_research']

def status(phase,**extra):
    state=dict(phase=phase,pid=os.getpid(),updated=now(),run=str(RUN),source_hashes=FROZEN,**extra)
    atomic_json(STATE,state)
    print(json.dumps(state,ensure_ascii=False),flush=True)
    p=ROOT/'stages/stage2/qwen35_status.json'
    s=read_json(p,{})
    s.update(status=phase,updated=now(),pipeline_pid=os.getpid(),pipeline_status=str(STATE.relative_to(ROOT)),
             formal_generated=len(list((RUN/'records/train').glob('*.json')))+len(list((RUN/'records/test').glob('*.json'))),
             smoke_generated=len(list((RUN/'records/smoke').glob('*.json'))))
    atomic_json(p,s)

def command(phase,args):
    if source_hashes('probe')!=FROZEN:
        raise RuntimeError('Generator execution source changed while pipeline was waiting; preserve run and restart with new identity')
    status(phase)
    subprocess.run(args,cwd=ROOT,check=True,env={**os.environ,'HF_HUB_DISABLE_XET':'1','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1'})

def weights_ready(model,revision):
    folder=Path('/root/autodl-tmp/hf-qwen35')/('models--Qwen--'+model)/'snapshots'/revision
    names=['config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json']
    if not all((folder/n).exists() for n in names):return False
    names=set(json.loads((folder/'model.safetensors.index.json').read_text())['weight_map'].values())
    return all((folder/n).exists() for n in names)

try:
    cfg,actual,manifest=prepare(PROBE)
    if actual.resolve()!=RUN.resolve():
        raise RuntimeError('Configured stage2 run differs from actual source/config identity')
    status('WAITING_FOR_4B_WEIGHTS')
    while not weights_ready('Qwen3.5-4B',cfg['revisions']['model']):
        os.kill(int(os.environ['QWEN_DOWNLOAD_PID']),0)
        time.sleep(10)
    validation=read_json(RUN/'gpu_validation/validation.json',{})
    if validation.get('status')!='PASS' or validation.get('sources',{}).get('probe')!=FROZEN:
        command('REAL_MODEL_INTERFACE_AND_RESTORE_CHECK',[str(PYTHON),'tests/qwen35_migration_check.py','--gpu','--output',str(RUN/'gpu_validation')])
    for phase,flags in [('SMOKE_FIRST_QUESTION',['--smoke','--resume','--stop-after','1']),
                        ('SMOKE_RESTART_AND_COMPLETE',['--smoke','--resume']),
                        ('SMOKE_ALL_RECORDS_RESUME_AUDIT',['--smoke','--resume']),
                        ('STAGE2_FORMAL_COLLECTION_AND_FIT',['--resume'])]:
        command(phase,BASE+['probe','--config',str(PROBE)]+flags)
    if not (RUN/'probe_B.json').exists():
        raise RuntimeError('Stage2 did not fit probe B; retain insufficient-training diagnosis')
    high=read_json(HIGH)
    status('STAGE2_COMPLETE_WAITING_FOR_9B_WEIGHTS')
    while not weights_ready('Qwen3.5-9B',high['revisions']['verifier']):
        os.kill(int(os.environ['QWEN_DOWNLOAD_PID']),0)
        time.sleep(10)
    for phase in ('prepare','capacity','verifier-smoke','verifier-dev','verifier-audit','pilot'):
        command('STAGE3_'+phase.upper(),BASE+['intervene','--experiment','high-low','--config',str(HIGH),'--phase',phase,'--resume'])
    stage3=read_json(ROOT/'stages/stage3/qwen35_status.json',{})
    stage3.update(status='AWAITING_INDEPENDENT_HUMAN_AUDIT',updated=now(),formal_pairs=0,
                  note='Review exported verifier_audit_review_inputs.json; no automatic human labels or formal test')
    atomic_json(ROOT/'stages/stage3/qwen35_status.json',stage3)
    status('STAGE2_COMPLETE_STAGE3_AWAITING_HUMAN_AUDIT')
except BaseException as exc:
    status('STOPPED',error=safe_error(exc),exception_type=type(exc).__name__)
    traceback.print_exc()
    raise
