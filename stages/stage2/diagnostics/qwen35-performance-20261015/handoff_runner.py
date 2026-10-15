"""Run the authorized isolated benchmark and restore the unchanged formal CLI.

Run after the boundary watcher reports COLLECTOR_STOPPED_AT_SAVED_BOUNDARY.
The continuation monitor stays alive until formal collection and fitting finish.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from veriserve_research.artifacts.io import atomic_json, digest, now, read_json
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.probe.materials import valid_features

RUN = ROOT / 'stages/stage2/runs/v2-20261013-e136a41fea81'
CONFIG = ROOT / 'stages/stage2/qwen35_4b_omni_vllm_config.json'
PYTHON = ROOT / 'environments/qwen35/.venv/bin/python'
FORMAL = [str(PYTHON), '-m', 'veriserve_research', 'probe', '--config', str(CONFIG), '--resume']


def alive(pid):
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def protected_files():
    paths = [CONFIG, ROOT / 'stages/stage2/qwen35_4b_omni_config.json']
    for path in sorted((RUN / 'records').glob('*/*.json')):
        record = read_json(path)
        if record.get('stage') == 'FEATURES':
            assert valid_features(path, record), path
            paths.append(path)
    paths.extend(sorted((RUN / 'records').glob('*/*.npz')))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def verify(snapshot):
    assert source_hashes('probe') == snapshot['execution_sources']
    for name, expected in snapshot['protected_sha256'].items():
        assert digest(ROOT / name) == expected, name


def counts():
    return {split: sum(read_json(p).get('stage') == 'FEATURES'
                       for p in (RUN / 'records' / split).glob('*.json'))
            for split in ('smoke', 'train', 'test')}


def monitor():
    snapshot = read_json(HERE / 'pause_snapshot.json')
    verify(snapshot)
    state = read_json(HERE / 'handoff.json')
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    with (RUN / 'pipeline.log').open('a', buffering=1) as log:
        child = subprocess.Popen(FORMAL, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='FORMAL_RESUMED', resumed_at=now(), continuation_pid=os.getpid(),
                     formal_pid=child.pid, formal_command=FORMAL)
        atomic_json(HERE / 'handoff.json', state)
        atomic_json(RUN / 'launch.json', dict(command=[str(PYTHON), str(Path(__file__)), '--monitor'],
            pid=os.getpid(), formal_pid=child.pid, created=now(), driver_sha256=digest(Path(__file__)),
            previous_launch=str((HERE / 'previous_launch.json').relative_to(ROOT)), stage3_formal=False))
        last = None
        while True:
            rc = child.poll()
            current = counts()
            if current != last or rc is not None:
                phase = ('STAGE2_FORMAL_COLLECTION_AND_FIT' if rc is None else
                         'STAGE2_COMPLETE_STAGE3_NOT_STARTED' if rc == 0 else 'STOPPED')
                status = dict(phase=phase, pid=os.getpid(), formal_pid=child.pid, updated=now(),
                              run=str(RUN), source_hashes=snapshot['execution_sources'], counts=current,
                              continuation_after_performance_benchmark=True, returncode=rc)
                atomic_json(RUN / 'pipeline_status.json', status)
                p = ROOT / 'stages/stage2/qwen35_status.json'
                old = read_json(p, {})
                old.update(status=phase, updated=now(), pipeline_pid=os.getpid(),
                    formal_generated=sum(len(list((RUN / 'records' / s).glob('*.json'))) for s in ('train', 'test')),
                    formal_features_completed=current['train'] + current['test'],
                    current_gpu_job={'kind': 'stage2_pipeline', 'pipeline_pid': os.getpid()})
                atomic_json(p, old)
                last = current
            if rc is not None:
                state.update(formal_returncode=rc, formal_finished_at=now())
                atomic_json(HERE / 'handoff.json', state)
                return rc
            time.sleep(10)


def main():
    state = read_json(HERE / 'handoff.json')
    assert state['status'] == 'COLLECTOR_STOPPED_AT_SAVED_BOUNDARY'
    assert read_json(ROOT / state['record'])['stage'] == 'FEATURES'
    for key, needle in [('collector_pid', b'\x00veriserve_research\x00probe\x00'),
                        ('driver_pid', b'pipeline_driver.py')]:
        assert needle in Path(f'/proc/{state[key]}/cmdline').read_bytes()
    snapshot = dict(created=now(), execution_sources=source_hashes('probe'), protected_sha256=protected_files())
    atomic_json(HERE / 'pause_snapshot.json', snapshot)
    atomic_json(HERE / 'previous_launch.json', read_json(RUN / 'launch.json'))
    collector, driver = state['collector_pid'], state['driver_pid']
    os.kill(collector, signal.SIGINT)
    os.kill(collector, signal.SIGCONT)
    deadline = time.monotonic() + 120
    while alive(collector):
        if time.monotonic() >= deadline:
            raise TimeoutError('Collector cleanup did not finish; do not start benchmark')
        time.sleep(.5)
    os.kill(driver, signal.SIGTERM)
    os.kill(driver, signal.SIGCONT)
    state.update(status='FORMAL_PAUSED_GPU_RELEASING', stopped_at=now())
    atomic_json(HERE / 'handoff.json', state)
    from runpy import run_path
    benchmark = run_path(str(HERE / 'benchmark.py'))
    deadline = time.monotonic() + 60
    while benchmark['busy_gpu']():
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU not released; do not start benchmark or competing formal process')
        time.sleep(1)
    try:
        verify(snapshot)
        from veriserve_research.artifacts.backup import backup_probe
        backup_probe(RUN, 'stage2: checkpoint before isolated performance benchmark')
        state.update(status='GPU_BENCHMARK_RUNNING', benchmark_started_at=now())
        atomic_json(HERE / 'handoff.json', state)
        with (HERE / 'benchmark.log').open('a', buffering=1) as log:
            child = subprocess.Popen(['/root/veriserve-vllm-env/bin/python', '-u', str(HERE / 'benchmark.py'), '--run'],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['benchmark_pid'] = child.pid
            atomic_json(HERE / 'handoff.json', state)
            try:
                rc = child.wait(timeout=2100)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                rc = -signal.SIGTERM
        state.update(benchmark_returncode=rc, benchmark_finished_at=now())
    finally:
        verify(snapshot)
        state.update(status='BENCHMARK_FINISHED_RESTORING_FORMAL', preserved_files_verified=True)
        atomic_json(HERE / 'handoff.json', state)
        deadline = time.monotonic() + 120
        while benchmark['busy_gpu']():
            if time.monotonic() >= deadline:
                raise TimeoutError('Benchmark GPU processes remain; refuse competing formal restart')
            time.sleep(1)
        with (HERE / 'continuation.log').open('a', buffering=1) as log:
            child = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__)), '--monitor'], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Formal continuation monitor launched:', child.pid, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.monitor:
        sys.exit(monitor())
    main()
