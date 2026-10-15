"""Isolated vLLM benchmark; refuses to load weights while another GPU job exists.

Run with the existing vLLM Python. All outputs remain in this diagnostic folder.
The benchmark changes output budgets only for diagnostics and never grades answers.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from veriserve_research.artifacts.io import atomic_json
from veriserve_research.artifacts.provenance import source_hashes


def windows(milestones):
    edges = [1, 8192, 16384, 32768, 65536]
    return [dict(start_tokens=a, end_tokens=b, seconds=milestones[b] - milestones[a],
                 tokens_per_second=(b - a) / (milestones[b] - milestones[a]))
            for a, b in zip(edges, edges[1:]) if a in milestones and b in milestones]


def busy_gpu():
    # Fail closed: a failed NVML query must never authorize an extra GPU workload.
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                      '--format=csv,noheader'], text=True).strip()
    return result


def preemptions(llm):
    return sum(m.value for m in llm.get_metrics() if m.name == 'vllm:num_preemptions')


def stream(llm, records, budget, cfg, eos, tag, deadline):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    before = preemptions(llm)
    state = {}
    start = time.perf_counter()
    for i, record in enumerate(records):
        key = f'{tag}-{i}'
        params = SamplingParams(temperature=cfg['temperature'], top_p=cfg['top_p'], top_k=cfg['top_k'],
            min_p=cfg['min_p'], presence_penalty=cfg['presence_penalty'],
            repetition_penalty=cfg['repetition_penalty'], frequency_penalty=0,
            max_tokens=budget, seed=record['generation_seed'], stop_token_ids=eos,
            ignore_eos=False, detokenize=False, output_kind=RequestOutputKind.DELTA)
        state[key] = dict(unique_id=record['unique_id'], token_ids=[], milestones={}, finished=False)
        engine.add_request(key, {'prompt_token_ids': record['prompt_ids']}, params)
    while engine.has_unfinished_requests():
        if time.monotonic() >= deadline:
            engine.abort_request(list(state))
            raise TimeoutError('Diagnostic wall-time budget reached; partial timings are not final results')
        for output in engine.step():
            row = state[output.request_id]
            old = len(row['token_ids'])
            row['token_ids'].extend(output.outputs[0].token_ids)
            count = len(row['token_ids'])
            elapsed = time.perf_counter() - start
            for edge in (1, 8192, 16384, 32768, 65536):
                if old < edge <= count:
                    row['milestones'][edge] = elapsed
            if output.finished:
                row.update(finished=True, seconds=elapsed, finish_reason=output.outputs[0].finish_reason)
    elapsed = time.perf_counter() - start
    rows = []
    for row in state.values():
        ids = row.pop('token_ids')
        row.update(tokens=len(ids), token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   decode_windows=windows(row['milestones']))
        rows.append(row)
    return dict(tag=tag, seconds=elapsed, tokens=sum(r['tokens'] for r in rows),
                aggregate_tokens_per_second=sum(r['tokens'] for r in rows) / elapsed,
                preemptions=preemptions(llm) - before, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Load GPU weights only if the GPU is idle')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error('--max-seconds must be positive')
    cfg_path = ROOT / 'stages/stage2/qwen35_4b_omni_vllm_config.json'
    cfg = json.loads(cfg_path.read_text())
    run = ROOT / 'stages/stage2/runs/v2-20261013-e136a41fea81'
    manifest = json.loads((run / 'manifest.json').read_text())
    saved = {r['unique_id']: r for r in (json.loads(p.read_text()) for p in (run / 'records/smoke').glob('*.json'))}
    records = [saved[r['unique_id']] for r in manifest['splits']['smoke'][:4]]
    result = dict(status='PLANNED', created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fixed_smoke_ids=[r['unique_id'] for r in records], config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        execution_sources=source_hashes('probe'), max_seconds=args.max_seconds,
        plan=['First fixed smoke prompt: stream up to 65536 tokens, preserve sampling/EOS, record actual decode windows.',
              'First four fixed smoke prompts: same work at concurrency 1, 2, 4, up to 2048 new tokens per request.'],
        limitations=['GPU-only decoding; no resident HF model, hidden extraction, scoring or formal backup costs.',
                     'Short concurrency tests do not establish feasibility or performance of four simultaneous 81920-token outputs.',
                     'Changing engine max_num_seqs or batch composition can change sampled tokens despite identical seeds.',
                     'Milestone times are host-observed stream arrival times; first window excludes TTFT and uses 8191 intervals.',
                     'No speculative decoding, prompt edits, new dependencies or writes to formal run.'], measurements=[])
    if not args.run:
        atomic_json(HERE / 'benchmark_plan.json', result)
        print('PLAN_ONLY; pass --run on an idle GPU. No CUDA import or weight loading.')
        return
    busy = busy_gpu()
    if busy:
        result.update(status='DEFERRED_GPU_BUSY', active_compute_apps=busy)
        atomic_json(HERE / 'benchmark_status.json', result)
        print('DEFERRED_GPU_BUSY: formal collection continues. No CUDA import or weight loading.')
        return
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    import vllm
    from vllm import LLM
    if vllm.__version__ != cfg['vllm_version']:
        raise RuntimeError('Installed vLLM version differs from the diagnosed formal environment')
    snapshot = Path(cfg['model_cache_dir']) / ('models--' + cfg['model'].replace('/', '--')) / 'snapshots' / cfg['revisions']['model']
    deadline = time.monotonic() + args.max_seconds
    llm = None
    try:
        for concurrency in (1, 2, 4):
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic budget reached before engine initialization')
            llm = LLM(model=str(snapshot), tokenizer=str(snapshot), dtype='bfloat16',
                max_model_len=cfg['max_total_tokens'], max_num_seqs=concurrency,
                gpu_memory_utilization=cfg['vllm_gpu_memory_utilization'],
                max_num_batched_tokens=cfg['prefill_chunk_size'], enable_chunked_prefill=True,
                mamba_ssm_cache_dtype='float32', enable_prefix_caching=False,
                seed=cfg['seed'], generation_config='vllm', language_model_only=True,
                limit_mm_per_prompt={'image': 0, 'video': 0}, async_scheduling=True,
                enforce_eager=False, disable_log_stats=False)
            tokenizer = llm.get_tokenizer()
            eos = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>'),
                        tokenizer.convert_tokens_to_ids('<|endoftext|>')})
            stream(llm, records[:1], 128, cfg, eos, f'warmup-{concurrency}', deadline)
            if concurrency == 1:
                result['measurements'].append(stream(llm, records[:1], 65536, cfg, eos, 'length-profile', deadline))
                atomic_json(HERE / 'benchmark_result.json', {**result, 'status': 'RUNNING'})
            batches = [stream(llm, records[i:i + concurrency], 2048, cfg, eos,
                              f'batch-{concurrency}-{i}', deadline) for i in range(0, 4, concurrency)]
            seconds = sum(b['seconds'] for b in batches)
            tokens = sum(b['tokens'] for b in batches)
            result['measurements'].append(dict(concurrency=concurrency, batches=batches, seconds=seconds,
                tokens=tokens, aggregate_tokens_per_second=tokens / seconds))
            atomic_json(HERE / 'benchmark_result.json', {**result, 'status': 'RUNNING'})
            llm.llm_engine.engine_core.shutdown()
            llm = None
        assert result['execution_sources'] == source_hashes('probe')
        assert result['config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        result['status'] = 'GPU_BENCHMARK_COMPLETE'
    except Exception as exc:
        result.update(status='FAILED_OR_TIMED_OUT', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        atomic_json(HERE / 'benchmark_result.json', result)


if __name__ == '__main__':
    assert windows({1: 1.0, 8192: 8192.0})[0]['tokens_per_second'] == 1.0
    assert windows({1: 1.0}) == []
    main()
