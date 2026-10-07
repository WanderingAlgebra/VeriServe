"""Thin HIGH/LOW runner using stage2 materials and stage3's resident runtime."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import time
from pathlib import Path

from stages.stage1.veriserve.common import atomic_json, stable_hash
from stages.stage2 import run_probe as old
from stages.stage3 import storage, run_timing as legacy
from . import source_hashes, selection, runtime, analyze

HERE = Path(__file__).resolve().parents[1]
ARMS = ('HIGH', 'LOW')


def prepare(config_path):
    cfg = storage.safe_read_json(config_path)
    original, splits = selection.load_source(old.ROOT / cfg['stage2_run'])
    if cfg.get('experiment_type') != 'within_question_high_low' or cfg['split_sizes'] != {'pilot': 4, 'test': 200}:
        raise ValueError('This entry only supports the frozen 4-pilot/200-test HIGH/LOW diagnostic')
    baseline = original['config']
    for field in ('model', 'prompt', 'dataset', 'dataset_split', 'attention', 'dtype', 'batch_size', 'do_sample'):
        if cfg[field] != baseline[field]:
            raise ValueError(f'Original generator setting changed: {field}')
    for field in ('model', 'tokenizer', 'dataset'):
        if cfg['revisions'][field] != baseline['revisions'][field]:
            raise ValueError(f'Original revision changed: {field}')
    legacy_cfg = storage.safe_read_json(HERE / 'config.json')
    for field in ('max_new_tokens', 'max_request_tokens', 'max_total_tokens', 'max_forward_tokens',
                  'max_prm_calls', 'max_prm_input_tokens', 'max_rollbacks', 'verifier'):
        if cfg[field] != legacy_cfg[field]:
            raise ValueError(f'Original stage3 budget/verifier changed: {field}')
    if cfg['revisions']['verifier'] != legacy_cfg['revisions']['verifier'] or cfg['revisions']['verifier_tokenizer'] != cfg['revisions']['verifier']:
        raise ValueError('PRM model/tokenizer revision changed')
    config_hash = stable_hash(cfg)
    run = HERE / 'runs' / f"{cfg['seed']}-{config_hash[:12]}"
    run.mkdir(parents=True, exist_ok=True)
    material = old.ROOT / cfg['stage2_run']
    hashes = {str(p.relative_to(old.ROOT)): old.digest(p)
              for p in [material / 'manifest.json', material / 'probe_B.json', material / 'probe_B.npz']}
    record_hashes = {str(old.record_path(material, 'smoke' if phase == 'pilot' else phase, row['unique_id']).relative_to(old.ROOT)):
                     old.digest(old.record_path(material, 'smoke' if phase == 'pilot' else phase, row['unique_id']))
                     for phase, rows in splits.items() for row in rows}
    current = storage.safe_read_json(run / 'manifest.json')
    if current:
        if (current['config_hash'] != config_hash or current['material_hashes'] != hashes
                or current['record_hashes'] != record_hashes or current['splits'] != splits):
            raise RuntimeError('Manifest/source changed; preserve this run and create a separate run')
        if current['source_hashes'] != source_hashes():
            if any((run / 'arms').glob('*.json')) or (run / 'protocol_frozen.json').exists():
                raise RuntimeError('Execution source changed; preserve timed pairs and use a new run_note/run')
            current.update(source_hashes=source_hashes(), code_commit=old.git('rev-parse', 'HEAD'))
            atomic_json(run / 'manifest.json', current)
        if not any((run / 'arms').glob('*.json')) and not (run / 'protocol_frozen.json').exists():
            current.update(code_commit=old.git('rev-parse', 'HEAD'))
            atomic_json(run / 'manifest.json', current)
        return cfg, run, current
    current = {'experiment_type': cfg['experiment_type'], 'run_id': run.name, 'created': old.now(),
               'config': cfg, 'config_hash': config_hash, 'splits': splits,
               'stage2_manifest': str(material.relative_to(old.ROOT) / 'manifest.json'),
               'material_hashes': hashes, 'record_hashes': record_hashes, 'source_hashes': source_hashes(),
               'code_commit': old.git('rev-parse', 'HEAD'), 'checked_head': 'eeb47f1f0b28ee70992d2201ad1db8f5481baaaa',
               'environment': storage.environment(), 'grading': original['grading'],
               'probe_interpretation': 'Unintervened complete-trajectory final-error risk; not local error or check benefit',
               'sampling': 'Original smoke first4 for self-check; original held-out test200 in manifest order; exploratory reuse, no label filtering',
               'timing': cfg['timing']}
    atomic_json(run / 'manifest.json', current)
    storage.event(run, 'PREPARED', experiment_type=cfg['experiment_type'], sizes=cfg['split_sizes'])
    return cfg, run, current


def paths(run, uid):
    return [run / 'arms' / f'{legacy.key(uid)}.{arm}.json' for arm in ARMS]


def complete_pair(run, row):
    pair = [storage.safe_read_json(p, {}) for p in paths(run, row['unique_id'])]
    if not all(a.get('completed') and a.get('status') == 'COMPLETE' for a in pair):
        return False
    if any(a.get('source_hashes') != source_hashes() for a in pair):
        raise RuntimeError('Saved pair source differs from current execution')
    if pair[0]['backend'] != pair[1]['backend']:
        raise RuntimeError('Saved pair was not measured with one backend')
    selected = storage.safe_read_json(run / 'selections' / f'{legacy.key(row["unique_id"])}.json')
    if (selected.get('selection_sha256') != stable_hash({k: v for k, v in selected.items() if k != 'selection_sha256'})
            or any(a.get('selection_hash') != stable_hash(selected) for a in pair)):
        raise RuntimeError('Saved pair/frozen selection hash mismatch')
    return True


def archive_pair(run, uid, reason):
    existing = [p for p in paths(run, uid) if p.exists()]
    if existing:
        folder = run / 'interrupted_pairs' / legacy.key(uid) / str(time.time_ns())
        folder.mkdir(parents=True)
        for path in existing:
            path.rename(folder / path.name)
        storage.event(run, 'PAIR_RETIMING', unique_id=uid, reason=reason, archive=str(folder.relative_to(run)))


def counts(run, manifest):
    return {phase: sum(complete_pair(run, row) for row in rows)
            for phase, rows in manifest['splits'].items()}


def backup(run, manifest, reason):
    """Same ordinary Git/LFS backup, scoped to this experiment and its two READMEs."""
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    scoped = ['stages/stage3/high_low', 'stages/stage3/high_low_config.json',
              str(run.relative_to(old.ROOT)), 'stages/stage3/README.md', 'README.md']
    status = {'status': 'PENDING', 'reason': reason, 'created': old.now(), 'completed_pairs': counts(run, manifest)}
    atomic_json(run / 'backup.json', status)
    try:
        if old.git('branch', '--show-current') != old.BRANCH:
            raise RuntimeError('Backup must use the authorized existing branch')
        old.git('add', '--', *scoped)
        if old.git('diff', '--cached', '--name-only', '--', *scoped):
            old.git('commit', '--only', '-m', f'Stage3 HIGH/LOW {reason}', '--', *scoped)
        commit = old.git('rev-parse', 'HEAD')
        if list(run.rglob('*.npz')):
            old.git('lfs', 'push', 'origin', 'HEAD', env=env)
        old.git('push', '--porcelain', 'origin', f'HEAD:refs/heads/{old.BRANCH}', env=env)
        if old.git('ls-remote', 'origin', f'refs/heads/{old.BRANCH}', env=env).split()[0] != commit:
            raise RuntimeError('Remote did not confirm the pushed commit')
        status.update(status='PUSHED', commit=commit, pushed_at=old.now())
    except Exception as exc:
        # The protocol explicitly permits local saving when remote access is unavailable.
        status.update(status='LOCAL_ONLY', error=old.safe_error(exc))
        print(f'BACKUP LOCAL_ONLY: {status["error"]}', flush=True)
    atomic_json(run / 'backup.json', status)
    return status


def cpu_checks(cfg, run):
    original = legacy.cpu_checks(cfg, run)
    result = {**original, 'experiment_type': cfg['experiment_type'], 'source_hashes': source_hashes(),
              'high_low_selection': selection.self_check(), 'high_low_runtime': runtime.self_check(cfg),
              'high_low_analysis': analyze.self_check()}
    atomic_json(run / 'self_check.json', result)
    return result


def freeze(cfg, run, manifest):
    pilot = storage.safe_read_json(run / 'pilot_checks.json', {})
    check = storage.safe_read_json(run / 'self_check.json', {})
    if pilot.get('status') != 'PASS' or check.get('status') != 'PASS':
        raise RuntimeError('Current CPU and all four real GPU pilot checks must pass before freeze/test')
    if pilot.get('source_hashes') != source_hashes() or check.get('source_hashes') != source_hashes():
        raise RuntimeError('Pilot/self-check source mismatch')
    result = {'experiment_type': cfg['experiment_type'], 'source_hashes': source_hashes(),
              'config_hash': manifest['config_hash'], 'material_hashes': manifest['material_hashes'],
              'record_hashes': manifest['record_hashes'], 'ordered_ids': {p: [r['unique_id'] for r in rows] for p, rows in manifest['splits'].items()},
              'selection': 'Offline full-reference z maximum/minimum, earliest tie; no gold or PRM; positions saved before path execution',
              'intervention': 'One active intermediate check, then terminal-only checks/rework; PASS checkpoint; all low steps FAIL including checkpoint_conflict',
              'submission': 'Terminal PASS only; no empty/unavailable PRM result is PASS',
              'reference': cfg['reference_policy'], 'timing': cfg['timing'],
              'bootstrap': {'replicates': cfg['bootstrap'], 'seed': cfg['seed'], 'delta_Y': 'HIGH-LOW', 'delta_T': 'LOW-HIGH'},
              'pilot_checks_sha256': old.digest(run / 'pilot_checks.json'), 'code_commit': old.git('rev-parse', 'HEAD'), 'created': old.now()}
    saved = storage.safe_read_json(run / 'protocol_frozen.json')
    if saved:
        for field in ('source_hashes', 'config_hash', 'material_hashes', 'record_hashes', 'ordered_ids', 'pilot_checks_sha256'):
            if saved[field] != result[field]:
                raise RuntimeError(f'Frozen protocol changed: {field}')
        return saved
    atomic_json(run / 'protocol_frozen.json', result)
    backup(run, manifest, 'freeze protocol after four GPU pilots')
    return result


def freeze_selection(cfg, run, row, phase, tokenizer, model, reference=None):
    path = run / 'selections' / f'{legacy.key(row["unique_id"])}.json'
    saved = storage.safe_read_json(path)
    if saved and saved.get('source_hashes') != source_hashes():
        if any((run / 'arms').glob('*.json')) or (run / 'protocol_frozen.json').exists():
            raise RuntimeError('Frozen selection source changed; preserve it and use a new run')
        atomic_json(run / 'preflight_history' / f'{time.time_ns()}-{path.name}', saved)
        saved = None
    if saved and reference is None and saved.get('status') in ('ELIGIBLE', 'EXCLUDED'):
        if saved.get('selection_sha256') != stable_hash({k: v for k, v in saved.items() if k != 'selection_sha256'}):
            raise RuntimeError('Frozen selection hash mismatch')
        if any(not Path(p).exists() or old.digest(p) != digest
               for p, digest in saved.get('source_material_hashes', {}).items()):
            raise RuntimeError('Frozen selection material hash mismatch')
        return saved
    selected = selection.freeze_reference(old.ROOT / cfg['stage2_run'], 'smoke' if phase == 'pilot' else phase,
                                          row, tokenizer=tokenizer, model=model, reference=reference,
                                          feature_cache_path=run / 'features' / f'{legacy.key(row["unique_id"])}.npz')
    selected.pop('selection_sha256', None)
    selected['timing'] = selected.pop('timings', selected.get('timing', {}))
    selected.update(phase=phase, source_hashes=source_hashes(), frozen_at=old.now())
    selected['selection_sha256'] = stable_hash(selected)
    atomic_json(path, selected)
    storage.event(run, 'POSITIONS_FROZEN', unique_id=row['unique_id'], phase=phase, status=selected['status'],
                  selection_sha256=selected['selection_sha256'])
    return selected


def recollect(cfg, run, row, phase, selected, model, tokenizer, mismatch):
    uid = row['unique_id']
    if selected.get('reference_source') == 'stage3_streaming' or selected.get('recollected_streaming'):
        raise RuntimeError('A fresh streamed reference is not reproducible; stop without selecting another outcome')
    folder = run / 'reference_history' / legacy.key(uid)
    atomic_json(folder / f'{time.time_ns()}-stage2-selection.json', selected)
    archive_pair(run, uid, 'original reference token mismatch; recollect this question only')
    start = legacy.clock()
    reference = runtime.collect_reference(cfg, row, model, tokenizer)
    reference.update(recollection_reason=old.safe_error(mismatch), offline_wall_seconds=legacy.clock() - start)
    reference_path = run / 'references' / f'{legacy.key(uid)}.json'
    atomic_json(reference_path, reference)
    selected = freeze_selection(cfg, run, row, phase, tokenizer, model, reference)
    selected['recollected_streaming'] = True
    selected['reference_source'] = 'stage3_streaming'
    selected['source_material_hashes'][str(reference_path)] = old.digest(reference_path)
    selected.setdefault('timing', {})['offline_reference_seconds'] = reference['offline_wall_seconds']
    selected['selection_sha256'] = stable_hash({k: v for k, v in selected.items() if k != 'selection_sha256'})
    atomic_json(run / 'selections' / f'{legacy.key(uid)}.json', selected)
    return selected


def validate_gold(cfg, run, row):
    # Reuse stage3 gold validation, then apply its result to the new arm names.
    from math_verify import LatexExtractionConfig, parse
    from math_verify.utils import TimeoutException
    data = storage.safe_read_json(run / 'evaluation_data.json', {})
    uid = row['unique_id']
    if uid not in data:
        try:
            expected = parse('$' + row['answer'] + '$', extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                             fallback_mode='no_fallback', extraction_mode='first_match',
                             parsing_timeout=cfg['score_timeout_seconds'], raise_on_error=True)
            result = {'status': 'VALID' if expected else 'GOLD_UNPARSEABLE'}
        except TimeoutException:
            result = {'status': 'GOLD_TIMEOUT'}
        except Exception as exc:
            result = {'status': 'GOLD_ERROR', 'error': old.safe_error(exc)}
        data[uid] = result
        atomic_json(run / 'evaluation_data.json', data)
    if data[uid]['status'] != 'VALID':
        for path in paths(run, uid):
            state = storage.safe_read_json(path)
            state.setdefault('grading_original', state['grading'])
            state['grading'] = {**state['grading'], 'exclusion': data[uid]['status']}
            state['Y'] = None
            atomic_json(path, state)


def run_phase(cfg, run, manifest, phase, model, tokenizer, verifier, backend, stop_after=None):
    check = storage.safe_read_json(run / 'self_check.json', {})
    if check.get('status') != 'PASS' or check.get('source_hashes') != source_hashes():
        raise RuntimeError('Current CPU self-check required')
    if phase == 'test':
        freeze(cfg, run, manifest)
    checks = storage.safe_read_json(run / 'pilot_checks.json', {'questions': {}})
    if phase == 'pilot' and checks.get('status') == 'PASS' and all(complete_pair(run, row) for row in manifest['splits']['pilot']):
        print('pilot all valid completed pairs skipped; frozen GPU checks preserved', flush=True)
        return
    done = 0
    for index, row in enumerate(manifest['splits'][phase]):
        if complete_pair(run, row):
            validate_gold(cfg, run, row)
            continue
        selected = freeze_selection(cfg, run, row, phase, tokenizer, model)
        if selected['status'] == 'ELIGIBLE':
            archive_pair(run, row['unique_id'], 'incomplete infrastructure pair: rerun both request clocks')
            while True:
                try:
                    if phase == 'pilot':
                        checked = runtime.gpu_checks(cfg, run, row, selected, model, tokenizer, verifier)
                        checks['questions'][row['unique_id']] = checked
                        checks.update(source_hashes=source_hashes(), status='RUNNING', created=old.now())
                        atomic_json(run / 'pilot_checks.json', checks)
                    order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
                    for arm in order:
                        runtime.execute_path(cfg, run, row, selected, arm, model, tokenizer, verifier, backend)
                    validate_gold(cfg, run, row)
                    break
                except runtime.ReferenceMismatch as exc:
                    selected = recollect(cfg, run, row, phase, selected, model, tokenizer, exc)
                    if selected['status'] != 'ELIGIBLE':
                        break
        elif phase == 'pilot':
            checks['questions'][row['unique_id']] = {'status': 'EXCLUDED', 'reason': selected.get('reason')}
            atomic_json(run / 'pilot_checks.json', checks)
        new_pair = selected['status'] == 'ELIGIBLE' and complete_pair(run, row)
        done += int(new_pair)
        total = counts(run, manifest)[phase]
        if new_pair and total % cfg['backup_every'] == 0:
            print(f'PROGRESS {phase}: {total} complete HIGH/LOW pairs', flush=True)
            analyze.analyze(cfg, run, manifest)
            backup(run, manifest, f'{phase} {total} complete pairs')
        if stop_after and done >= stop_after:
            break
    if phase == 'pilot':
        all_checked = len(checks['questions']) == 4 and all(v.get('status') == 'PASS' for v in checks['questions'].values())
        checks.update(status='PASS' if all_checked else 'INCOMPLETE', source_hashes=source_hashes(), updated=old.now())
        atomic_json(run / 'pilot_checks.json', checks)
    analyze.analyze(cfg, run, manifest)
    backup(run, manifest, f'{phase} checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'high_low_config.json')
    parser.add_argument('--phase', choices=['prepare', 'self-check', 'pilot', 'freeze', 'test', 'analyze', 'backup', 'all'])
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Keep valid completed pairs; restart both arms of incomplete pairs')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    phase = 'self-check' if args.self_check else args.phase
    if phase is None:
        parser.error('--phase or --self-check is required')
    cfg, run, manifest = prepare(args.config)
    print(f'RUN {run}', flush=True)
    try:
        if phase in ('self-check', 'all'):
            cpu_checks(cfg, run)
        if phase in ('prepare', 'self-check', 'analyze'):
            if phase != 'analyze':
                for split, rows in manifest['splits'].items():
                    for row in rows:
                        freeze_selection(cfg, run, row, split, None, None)
            analyze.analyze(cfg, run, manifest)
            return
        if phase == 'backup':
            backup(run, manifest, 'manual backup')
            return
        if phase == 'freeze':
            freeze(cfg, run, manifest)
            return
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            atomic_json(run / 'execution_status.json', {'status': 'NOT_EXECUTED', 'reason': 'BF16 CUDA unavailable', 'created': old.now()})
            analyze.analyze(cfg, run, manifest)
            print('GPU EXPERIMENT NOT EXECUTED: BF16 CUDA unavailable', flush=True)
            return
        backend = {'gpu': torch.cuda.get_device_name(0), 'uuid': str(torch.cuda.get_device_properties(0).uuid),
                   'total_memory': torch.cuda.get_device_properties(0).total_memory,
                   'implementation': cfg['cache_backend'], 'attention': cfg['attention'], 'dtype': cfg['dtype'],
                   'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                   'bitsandbytes': importlib.metadata.version('bitsandbytes'), 'source_sha256': stable_hash(source_hashes())}
        model, tokenizer, verifier = legacy.load_resident(cfg, run)
        storage.archive_diagnostic(run, 'runtime_environment.json')
        atomic_json(run / 'runtime_environment.json', {'environment': old.environment(True), 'backend': backend,
                    'effective_generation_config': model.generation_config.to_dict(),
                    'execution_commit': old.git('rev-parse', 'HEAD'), 'created': old.now()})
        try:
            for step in ('pilot', 'test') if phase == 'all' else (phase,):
                run_phase(cfg, run, manifest, step, model, tokenizer, verifier, backend, args.stop_after)
        finally:
            del model, tokenizer, verifier
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        storage.event(run, 'PHASE_INTERRUPTED', phase=phase, error=old.safe_error(exc))
        atomic_json(run / 'interruption.json', {'phase': phase, 'error': old.safe_error(exc), 'created': old.now(),
                    'resume_command': f'python -m stages.stage3.high_low --config {args.config} --phase {phase} --resume'})
        analyze.analyze(cfg, run, manifest)
        raise


if __name__ == '__main__':
    main()
