"""Serial reference/isolated-FLA diagnostics after the live format trial exits."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from veriserve_research import ROOT
from veriserve_research.artifacts.backup import backup_probe
from veriserve_research.artifacts.io import read_json,atomic_json,now,safe_error
from veriserve_research.artifacts.provenance import source_hashes

HERE=Path(__file__).resolve().parent
RUN=ROOT/'stages/stage2/runs/v2-20261013-da5c802342ee'
STATE=HERE/'kernel_check_status.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    frozen=source_hashes('probe')
    assert frozen==read_json(RUN/'manifest.json')['source_hashes']
    trial=HERE.parent/'qwen35-paragraphs/final-format-retry'
    process_state=runpy.run_path(str(trial/'trial.py'))['process_state']
    metadata=dict(created=now(),pid=os.getpid(),wait_pid=args.wait_pid,source_hashes=frozen,
                  validation_only=True,scientific_results=False)
    def status(phase,**extra):
        atomic_json(STATE,dict(metadata,status=phase,updated=now(),**extra))
    status('WAITING_FOR_FORMAT_TRIAL')
    prior=process_state(args.wait_pid)
    if prior:
        while ((current:=process_state(args.wait_pid)) and current[19]==prior[19]
               and current[0] not in ('Z','X')):
            time.sleep(30)
    assert read_json(trial/'trial_status.json')['status']=='DIAGNOSTIC_COMPLETE'
    assert source_hashes('probe')==frozen
    backup_probe(RUN,'Prepare serial isolated GDN kernel validation')
    results={}
    for name in ('reference','fla'):
        output=HERE/name
        if (output/'validation.json').exists():
            raise FileExistsError(f'Preserve existing diagnostic: {output}')
        output.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_XET':'1','PYTHONUNBUFFERED':'1',
             'PYTHONPATH':str(ROOT)}
        if name=='fla':
            target=ROOT/'.cache/qwen35-fla-candidate/site-packages'
            env['PYTHONPATH']=str(target)+os.pathsep+str(ROOT)
        command=[str(ROOT/'environments/qwen35/.venv/bin/python'),'-u',
                 str(ROOT/'tests/qwen35_migration_check.py'),'--gpu','--benchmark-run',str(RUN),'--output',str(output)]
        assert source_hashes('probe')==frozen
        with (HERE/f'{name}.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(name.upper()+'_GPU_CHECK',child_pid=child.pid)
            code=child.wait()
        results[name]=read_json(output/'validation.json',{})
        backup_probe(RUN,f'Save {name} GDN validation and fixed-prefix timing')
        if code or results[name].get('status')!='PASS':
            raise RuntimeError(f'{name} GPU check failed; retain {output}')
    reference=results['reference']['benchmark']
    candidate=results['fla']['benchmark']
    optimized=all(item['optimized'] and item['module'].startswith('fla.') for item in candidate['backend'].values())
    comparison=dict(status='MEASURED' if optimized else 'CANDIDATE_FELL_BACK',created=now(),
        validation_only=True,scientific_results=False,optimized_gdn_active=optimized,
        fixed_prefix_sha256=reference['forced_token_sha256'],
        decode_speedup=reference['repetitions'][1]['teacher_forced_decode_seconds']/candidate['repetitions'][1]['teacher_forced_decode_seconds'],
        warm_reforward_speedup=reference['reforward_seconds'][1]/candidate['reforward_seconds'][1],
        reference=reference,candidate=candidate,
        scope='Synthetic full-cache restoration plus fixed-token and three original-prefix timings. Not a formal trajectory, ten-question smoke, pilot or capacity gate.')
    assert reference['forced_token_sha256']==candidate['forced_token_sha256']
    assert reference['independent_prefix_positions']==candidate['independent_prefix_positions']
    atomic_json(HERE/'comparison.json',comparison)
    status('DIAGNOSTICS_COMPLETE',comparison=str(HERE/'comparison.json'))
    backup_probe(RUN,'Save isolated GDN kernel comparison without changing active environment')


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        previous=read_json(STATE,{})
        atomic_json(STATE,dict(previous,status='FAILED',error=safe_error(exc),updated=now()))
        raise
