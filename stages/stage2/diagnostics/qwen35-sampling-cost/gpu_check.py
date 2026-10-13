"""Compare one suffix-view substitution in isolation; never change live sampling."""
import inspect
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, digest, now, read_json
from veriserve_research.inference import sample_token, sampling_config


def main():
    folder = Path(__file__).resolve().parent
    output = folder/'gpu_cost.json'
    if output.exists():
        raise FileExistsError('Preserve the existing benchmark')
    original = inspect.getsource(sample_token)
    before = 'torch.tensor(ids[prompt_length:], device=scores.device).unique()'
    after = 'tokens[0, prompt_length:].unique()'
    assert original.count(before) == 1
    namespace = dict(sample_token.__globals__)
    exec(compile(original.replace(before, after), '<suffix-view-candidate>', 'exec'), namespace)
    candidate = namespace['sample_token']
    run = ROOT/'stages/stage2/runs/v2-20261013-da5c802342ee'
    cfg = read_json(run/'manifest.json')['config']
    records = sorted([read_json(p) for p in (run/'records/smoke').glob('*.json')],
                     key=lambda r: len(r['generated_ids']))
    selected = [records[0], next(r for r in records if len(r['generated_ids']) == 20287), records[-1]]
    logits = torch.randn(248320, device='cuda', generator=torch.Generator(device='cuda').manual_seed(101))
    rows = []
    for record in selected:
        prompt = len(record['prompt_ids'])
        ids = record['prompt_ids'] + record['generated_ids']
        for seed in range(32):
            a = torch.Generator(device='cuda').manual_seed(seed)
            b = torch.Generator(device='cuda').manual_seed(seed)
            assert sample_token(logits, ids, prompt, sampling_config(cfg), a) == candidate(logits, ids, prompt, sampling_config(cfg), b)
            assert torch.equal(a.get_state(), b.get_state())
        def measure(fn):
            times = []
            for _ in range(5):
                rng = torch.Generator(device='cuda').manual_seed(0)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(64):
                    fn(logits, ids, prompt, sampling_config(cfg), rng)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)/64)
            return statistics.median(times)
        old, new = measure(sample_token), measure(candidate)
        rows.append(dict(unique_id=record['unique_id'],generated_tokens=len(record['generated_ids']),
                         original_ms=1000*old,candidate_ms=1000*new,sampler_speedup=old/new,
                         token_and_rng_equal_seeds=32))
    atomic_json(output,dict(status='PASS',created=now(),adopted=False,torch_version=torch.__version__,
        source_sha256=digest(ROOT/'veriserve_research/inference.py'),before=before,after=after,rows=rows,
        scope='Isolated CUDA sampler microbenchmark on frozen reference token histories; no model generation, scientific protocol change or whole-request speedup claim.'))
    print('SAMPLER_GPU_CHECK_PASS', rows, flush=True)


if __name__ == '__main__':
    main()
