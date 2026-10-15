"""Read-only performance audit. No model forward, regrading, or formal writes."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from veriserve_research.artifacts.io import atomic_json, stable_hash
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.inference import generation_seed, prompt_tokens

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'stages/stage2/runs/v2-20261013-78a47750b3b9'
RUN = ROOT / 'stages/stage2/runs/v2-20261013-e136a41fea81'
CONFIG = ROOT / 'stages/stage2/qwen35_4b_omni_vllm_config.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repetition(ids, width=64, stride=16):
    # Diagnostic heuristic only: fixed-stride exact matches miss shifted/paraphrased loops.
    counts = collections.Counter(tuple(ids[i:i + width])
                                 for i in range(0, len(ids) - width + 1, stride))
    return sum(n - 1 for n in counts.values()) / max(1, sum(counts.values()))


def describe(record, tokenizer=None, cfg=None):
    ids = record['generated_ids']
    text = record['raw_text']
    selected = record['boundaries']['steps'] + [record['boundaries']['end']]
    selected = [s for s in selected if s is not None]
    prefixes = [len(record['prompt_ids']) + s['end_token_index'] + 1 for s in selected]
    timings = record['timings']
    lines = collections.Counter(line.strip() for line in text.splitlines() if len(line.strip()) >= 60)
    duplicate_lines = [(line, count) for line, count in lines.most_common() if count > 1]
    row = dict(unique_id=record['unique_id'], tokens=len(ids), completion=record['completion'],
               stage=record['stage'], answer_parseable=record.get('answer_parseable'),
               generation_seconds=timings['generation_seconds'],
               feature_seconds=timings.get('reforward_seconds'),
               boundary_and_grading_seconds=timings.get('boundary_and_grading_seconds'),
               extraction_endpoints=len(prefixes), independently_forwarded_input_tokens=sum(prefixes),
               prefix_replay_factor=sum(prefixes) / (len(ids) + len(record['prompt_ids'])),
               tps_including_request_overhead=len(ids) / timings['generation_seconds'],
               exact_64token_stride16_repeat_fraction=repetition(ids),
               repeated_long_lines=duplicate_lines[:5], wait_mentions=len(re.findall(r'\bwait\b', text, re.I)),
               thinking_closed=text.count('</think>'), final_answer_mentions=text.count('Final answer:'),
               tail_excerpt=text[-1000:], token_sha256_matches=record['token_sha256'] == stable_hash(
                   dict(prompt_ids=record['prompt_ids'], generated_ids=ids)))
    if tokenizer is not None:
        eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>'),
                   tokenizer.convert_tokens_to_ids('<|endoftext|>')}
        row.update(prompt_rebuild_exact=prompt_tokens(cfg, tokenizer, record['source']['problem']) == record['prompt_ids'],
                   seed_rebuild_exact=generation_seed(cfg['seed'], record['unique_id']) == record['generation_seed'],
                   interior_eos_positions=[i for i, token in enumerate(ids[:-1]) if token in eos_ids],
                   last_token=ids[-1], ends_with_eos=ids[-1] in eos_ids,
                   budget_matches=(len(ids) == min(cfg['max_new_tokens'], cfg['max_total_tokens'] - len(record['prompt_ids'])))
                   if record['completion'] != 'EOS' else None)
        closing = tokenizer.encode('</think>', add_special_tokens=False)
        pos = next((i for i in range(len(ids) - len(closing) + 1) if ids[i:i + len(closing)] == closing), None)
        row['thinking_close_token_offset'] = pos
        row['tokens_after_thinking_close'] = len(ids) - pos - len(closing) if pos is not None else None
    return row


def aggregate(rows):
    complete = [r for r in rows if r['feature_seconds'] is not None]
    gen = sum(r['generation_seconds'] for r in complete)
    feature = sum(r['feature_seconds'] for r in complete)
    tokens = sum(r['tokens'] for r in complete)
    return dict(questions=len(rows), complete_features=len(complete), tokens=tokens,
                mean_tokens=tokens / max(1, len(complete)), generation_seconds=gen, feature_seconds=feature,
                generation_fraction=gen / (gen + feature) if gen + feature else None,
                aggregate_tps_including_request_overhead=tokens / gen if gen else None,
                generation_plus_features_seconds=gen + feature,
                projected_300_hours=(gen + feature) / max(1, len(complete)) * 300 / 3600,
                completion_counts=dict(collections.Counter(r['completion'] for r in rows)),
                independently_forwarded_input_tokens=sum(r['independently_forwarded_input_tokens'] for r in complete))


def main():
    cfg = read(CONFIG)
    protected = [CONFIG, ROOT / 'stages/stage2/qwen35_4b_omni_config.json']
    for run in (OLD, RUN):
        protected.extend(sorted((run / 'records/smoke').glob('*')))
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected if p.is_file()}
    sources = source_hashes('probe')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revisions']['tokenizer'],
                                             cache_dir=cfg['model_cache_dir'], local_files_only=True)
    cohorts = {}
    for name, run in [('old_transformers_smoke', OLD), ('current_vllm_smoke', RUN)]:
        manifest = read(run / 'manifest.json')
        records = {r['unique_id']: r for r in map(read, sorted((run / 'records/smoke').glob('*.json')))}
        rows = [describe(records[r['unique_id']], tokenizer, cfg) for r in manifest['splits']['smoke']]
        assert len(rows) == 10 and all(r['token_sha256_matches'] and r['prompt_rebuild_exact'] and
            r['seed_rebuild_exact'] and not r['interior_eos_positions'] for r in rows)
        assert all(r['ends_with_eos'] == (r['completion'] == 'EOS') for r in rows)
        assert all(r['budget_matches'] for r in rows if r['completion'] != 'EOS')
        cohorts[name] = dict(summary=aggregate(rows), rows=rows)
    formal = [describe(read(p)) for split in ('train', 'test') for p in sorted((RUN / 'records' / split).glob('*.json'))]
    log = (RUN / 'engine/worker.log').read_text()
    evidence = [line for line in log.splitlines() if any(s in line for s in (
        'Using Triton/FLA GDN', 'Using FlashAttention', 'GPU KV cache size:',
        'Available KV cache memory:', 'Capturing CUDA graphs', 'torch.compile took'))]
    process = subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,args'], text=True)
    live_jobs = [line for line in process.splitlines() if any(s in line for s in (
        'pipeline_driver.py', '-m veriserve_research probe', '-m veriserve_research.vllm_backend'))]
    fields = ('name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,'
              'clocks.current.sm,clocks.current.memory,clocks_event_reasons.sw_power_cap,'
              'clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown')
    telemetry = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv'], text=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected if p.is_file()}
    assert before == after and sources == source_hashes('probe')
    report = dict(status='OFFLINE_AUDIT_COMPLETE_GPU_BENCHMARK_DEFERRED_BUSY',
        observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        config_path=str(CONFIG.relative_to(ROOT)), current_run=str(RUN.relative_to(ROOT)),
        cohorts=cohorts, current_formal_snapshot=dict(summary=aggregate(formal), rows=formal),
        runtime=dict(live_jobs=live_jobs, nvidia_smi=telemetry, engine_log_evidence=evidence,
                     preemption_warning_lines=[line for line in log.splitlines() if re.search(r'preempt', line, re.I)]),
        preservation=dict(protected_sha256=before, execution_sources_sha256=sources,
                          all_protected_files_unchanged=True),
        limits=['Generation timer includes prefill, wake/sleep and IPC; pure_generation_seconds still includes prefill.',
                'Old and new engines have distinct trajectories despite fixed prompts/seeds. Ratio is observed cohort cost, not isolated engine speedup.',
                'Feature timings can be re-extraction measurements while generation timings come from preserved trajectories; sum is workload cost, not one uninterrupted wall-clock run.',
                'No per-token timestamps in saved records: 0-8K/8-16K/16-32K/32-64K decode rates cannot be recovered.',
                'No preemption metrics were enabled in the formal worker. Missing warnings is not proof of zero preemptions.',
                'Exact repetition and Wait counts are heuristics, not mathematical or semantic-loop judgments.',
                'GPU currently busy with formal collection. No concurrent GPU benchmark, formal pause, resampling, or configuration change.',
                'Projection from ten smoke questions excludes loading, engineering checks, backup, interruptions and probe fitting.'])
    atomic_json(HERE / 'audit.json', report)
    print(json.dumps({k: v['summary'] for k, v in cohorts.items()}, indent=2))
    print('Protected records/features/configs and execution source hashes unchanged.')


if __name__ == '__main__':
    assert repetition([1, 2] * 80, 64, 16) > 0.8
    assert repetition(list(range(160)), 64, 16) == 0
    assert aggregate([])['generation_plus_features_seconds'] == 0
    main()
