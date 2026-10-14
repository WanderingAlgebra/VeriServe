from __future__ import annotations
import argparse
import gc
import importlib.metadata
import time
import tempfile
from pathlib import Path
from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, atomic_bytes, read_json, stable_hash, digest, now, git, safe_error
from veriserve_research.artifacts.provenance import environment
from veriserve_research.probe.materials import record_path
from veriserve_research.artifacts import storage
from veriserve_research.intervention import fixed_step as fixed
from . import source_hashes
from ...artifacts.provenance import run_identity, manifest_version, require_current
from ...artifacts.materials import material_key, validate_materials
from ...artifacts.backup import backup_high_low
from . import selection, runtime
from ...analysis import high_low as analyze
HERE = ROOT / "stages/stage3"
ARMS = ("HIGH", "LOW")


def prepare(config_path):
    cfg = storage.safe_read_json(config_path)
    status = cfg.get('verifier',{}).get('selection_status')
    if status is not None and status != 'SELECTED_BY_DEVELOPMENT':
        raise RuntimeError('Select a verifier from the fixed development comparison before Stage3 prepare; Stage2 can proceed independently')
    if status == 'SELECTED_BY_DEVELOPMENT':
        proof=ROOT/cfg['verifier_development_comparison']
        decision=read_json(proof)
        mode='thinking' if cfg['verifier']['enable_thinking'] else 'direct'
        if (digest(proof)!=cfg['verifier_development_comparison_sha256']
                or decision.get('status')!='SELECTED_BY_DEVELOPMENT'
                or decision.get('selected_mode')!=mode
                or decision.get('source_hashes')!=source_hashes()):
            raise ValueError('Verifier development selection provenance mismatch')
    if not cfg.get('stage2_run'):
        raise ValueError('stage2_run must point to the completed new stage2 run (with probe_B.json)')
    original, splits = selection.load_source(ROOT / cfg['stage2_run'])
    if cfg.get('experiment_type') != 'within_question_high_low' or cfg['split_sizes'] != {'pilot': 4, 'test': 200}:
        raise ValueError('This entry only supports the frozen 4-pilot/200-test HIGH/LOW diagnostic')
    baseline = original['config']
    if baseline.get('engine','transformers') != 'transformers' or cfg.get('engine','transformers') != 'transformers':
        raise ValueError('Stage3 Session currently requires Transformers references and full hybrid snapshots; vLLM Stage2 references cannot be replayed or treated as prefix-cache snapshots')
    for field in ('model', 'prompt', 'dataset', 'dataset_split', 'attention', 'dtype', 'batch_size', 'do_sample'):
        if cfg[field] != baseline[field]:
            raise ValueError(f'Original generator setting changed: {field}')
    for field in ('model', 'tokenizer', 'dataset'):
        if cfg['revisions'][field] != baseline['revisions'][field]:
            raise ValueError(f'Original revision changed: {field}')
    if cfg.get('verifier', {}).get('kind') == 'generative':
        from ...inference import sampling_config
        if sampling_config(cfg) != sampling_config(baseline):
            raise ValueError('stage2/stage3 sampling mismatch')
        if cfg.get('system_prompt') != baseline.get('system_prompt'):
            raise ValueError('stage2/stage3 system prompt mismatch')
        if cfg.get('format_examples',[]) != baseline.get('format_examples',[]):
            raise ValueError('stage2/stage3 format examples mismatch')
        if cfg.get('boundary_mode','steps') != baseline.get('boundary_mode','steps'):
            raise ValueError('stage2/stage3 checkpoint rule mismatch')
        for field in ('enable_thinking','prefill_chunk_size','max_new_tokens','max_total_tokens','final_response_mode'):
            if cfg.get(field) != baseline.get(field):
                raise ValueError(f'stage2/stage3 setting mismatch: {field}')
        legacy_cfg = cfg
    else:
        legacy_cfg = storage.safe_read_json(HERE / 'config.json')
    for field in ('max_new_tokens', 'max_request_tokens', 'max_total_tokens', 'max_forward_tokens',
                  'max_prm_calls', 'max_prm_input_tokens', 'max_rollbacks', 'verifier'):
        if cfg[field] != legacy_cfg[field]:
            raise ValueError(f'Original stage3 budget/verifier changed: {field}')
    if cfg['revisions']['verifier'] != legacy_cfg['revisions']['verifier'] or cfg['revisions']['verifier_tokenizer'] != cfg['revisions']['verifier']:
        raise ValueError('PRM model/tokenizer revision changed')
    config_hash = stable_hash(cfg)
    material = ROOT / cfg['stage2_run']
    hashes = {str(p.relative_to(ROOT)): digest(p)
              for p in [material / 'manifest.json', material / 'probe_B.json', material / 'probe_B.npz']}
    record_hashes = {str(record_path(material, 'smoke' if phase == 'pilot' else phase, row['unique_id']).relative_to(ROOT)):
                     digest(record_path(material, 'smoke' if phase == 'pilot' else phase, row['unique_id']))
                     for phase, rows in splits.items() for row in rows}
    run = HERE / 'runs' / run_identity(cfg, 'high-low', {"materials": hashes, "records": record_hashes})
    run.mkdir(parents=True, exist_ok=True)
    current = storage.safe_read_json(run / 'manifest.json')
    if current:
        require_current(current, 'high-low')
        if (current['config_hash'] != config_hash or current['material_hashes'] != hashes
                or current['record_hashes'] != record_hashes or current['splits'] != splits):
            raise RuntimeError('Manifest/source changed; preserve this run and create a separate run')
        if current['source_hashes'] != source_hashes():
            if any((run / 'arms').glob('*.json')) or (run / 'protocol_frozen.json').exists():
                raise RuntimeError('Execution source changed; preserve timed pairs and use a new run_note/run')
            current.update(source_hashes=source_hashes(), code_commit=git('rev-parse', 'HEAD'))
            atomic_json(run / 'manifest.json', current)
        if not any((run / 'arms').glob('*.json')) and not (run / 'protocol_frozen.json').exists():
            current.update(code_commit=git('rev-parse', 'HEAD'))
            atomic_json(run / 'manifest.json', current)
        return cfg, run, current
    current = {**manifest_version('high-low'), 'run_id': run.name, 'created': now(),
               'config': cfg, 'config_hash': config_hash, 'splits': splits,
               'stage2_manifest': str(material.relative_to(ROOT) / 'manifest.json'),
               'material_hashes': hashes, 'record_hashes': record_hashes, 'source_hashes': source_hashes(),
               'code_commit': git('rev-parse', 'HEAD'),
               'environment': storage.environment(), 'grading': original['grading'],
               'probe_interpretation': 'Unintervened complete-trajectory final-error risk; not local error or check benefit',
               'sampling': 'Original smoke first4 for self-check; original held-out test200 in manifest order; exploratory reuse, no label filtering',
               'timing': cfg['timing']}
    atomic_json(run / 'manifest.json', current)
    storage.event(run, 'PREPARED', experiment_type=cfg['experiment_type'], sizes=cfg['split_sizes'])
    return cfg, run, current


def paths(run, uid):
    return [run / 'arms' / f'{fixed.key(uid)}.{arm}.json' for arm in ARMS]


def complete_pair(run, row):
    pair = [storage.safe_read_json(p, {}) for p in paths(run, row['unique_id'])]
    if not all(a.get('completed') and a.get('status') == 'COMPLETE' for a in pair):
        return False
    if any(a.get('source_hashes') != source_hashes() for a in pair):
        raise RuntimeError('Saved pair source differs from current execution')
    if pair[0]['backend'] != pair[1]['backend']:
        raise RuntimeError('Saved pair was not measured with one backend')
    selected = storage.safe_read_json(run / 'selections' / f'{fixed.key(row["unique_id"])}.json')
    if (selected.get('selection_sha256') != stable_hash({k: v for k, v in selected.items() if k != 'selection_sha256'})
            or any(a.get('selection_hash') != stable_hash(selected) for a in pair)):
        raise RuntimeError('Saved pair/frozen selection hash mismatch')
    return True


def archive_pair(run, uid, reason):
    existing = [p for p in paths(run, uid) if p.exists()]
    if existing:
        folder = run / 'interrupted_pairs' / fixed.key(uid) / str(time.time_ns())
        folder.mkdir(parents=True)
        for path in existing:
            path.rename(folder / path.name)
        storage.event(run, 'PAIR_RETIMING', unique_id=uid, reason=reason, archive=str(folder.relative_to(run)))


def counts(run, manifest):
    return {phase: sum(complete_pair(run, row) for row in rows)
            for phase, rows in manifest['splits'].items()}


def backup(run, manifest, reason):
    status=backup_high_low(run, counts(run, manifest), reason)
    if (manifest.get('config',{}).get('verifier',{}).get('kind')=='generative'
            and status.get('status')!='PUSHED'):
        raise RuntimeError('Generative HIGH/LOW remote backup failed; local data retained, collection stopped')
    return status


def cpu_checks(cfg, run):
    if cfg.get('verifier', {}).get('kind') == 'generative':
        from ...inference import migration_self_check
        result = dict(status='PASS', migration=migration_self_check(), source_hashes=source_hashes())
        atomic_json(run / 'self_check.json', result)
        return result
    original = fixed.cpu_checks(cfg, run)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='high-low-recovery-') as tmp:
        folder = Path(tmp)
        row = {'unique_id': 'synthetic-recovery'}
        selected = {'unique_id': row['unique_id'], 'phase': 'pilot', 'positions': {}}
        selected['selection_sha256'] = stable_hash(selected)
        atomic_json(folder / 'selections' / f'{fixed.key(row["unique_id"])}.json', selected)
        state = {'completed': True, 'status': 'COMPLETE', 'source_hashes': source_hashes(),
                 'backend': {'gpu': 'synthetic'}, 'selection_hash': stable_hash(selected)}
        pair_paths = paths(folder, row['unique_id'])
        atomic_json(pair_paths[0], state)
        assert not complete_pair(folder, row)
        atomic_json(pair_paths[1], state)
        assert complete_pair(folder, row)
        archive_pair(folder, row['unique_id'], 'synthetic interruption')
        assert not any(p.exists() for p in pair_paths)
        assert len(list((folder / 'interrupted_pairs').rglob('*.json'))) == 2
        for path in pair_paths:
            atomic_json(path, state)
        selected['positions'] = {'corrupt': True}
        atomic_json(folder / 'selections' / f'{fixed.key(row["unique_id"])}.json', selected)
        try:
            complete_pair(folder, row)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Corrupted selection must not skip a timed pair')
    result = {**original, 'experiment_type': cfg['experiment_type'], 'source_hashes': source_hashes(),
              'high_low_selection': selection.self_check(), 'high_low_runtime': runtime.self_check(cfg),
              'high_low_analysis': analyze.self_check(), 'high_low_recovery': 'PASS'}
    atomic_json(run / 'self_check.json', result)
    return result


def freeze(cfg, run, manifest):
    if cfg.get('verifier', {}).get('kind') == 'generative':
        capacity = storage.safe_read_json(run/'capacity_check.json',{})
        if capacity.get('status') != 'PASS' or capacity.get('source_hashes') != source_hashes():
            raise RuntimeError('Real simultaneous residency and full-budget capacity check required')
        audit = storage.safe_read_json(run / 'verifier_audit_gate.json', {})
        if (audit.get('status') != 'PASS' or audit.get('source_hashes') != source_hashes()
                or audit.get('config_hash') != manifest['config_hash']):
            raise RuntimeError('Independent human verifier audit must pass the prespecified quality gate')
    pilot = storage.safe_read_json(run / 'pilot_checks.json', {})
    check = storage.safe_read_json(run / 'self_check.json', {})
    if pilot.get('status') != 'PASS' or check.get('status') != 'PASS':
        raise RuntimeError('Current CPU and all four real GPU pilot checks must pass before freeze/test')
    if not all(complete_pair(run, row) for row in manifest['splits']['pilot']):
        raise RuntimeError('All four pilot questions require complete real HIGH/LOW pairs before freeze/test')
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
              'pilot_checks_sha256': digest(run / 'pilot_checks.json'), 'code_commit': git('rev-parse', 'HEAD'), 'created': now()}
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
    path = run / 'selections' / f'{fixed.key(row["unique_id"])}.json'
    saved = storage.safe_read_json(path)
    if saved and saved.get('source_hashes') != source_hashes():
        if any((run / 'arms').glob('*.json')) or (run / 'protocol_frozen.json').exists():
            raise RuntimeError('Frozen selection source changed; preserve it and use a new run')
        atomic_json(run / 'preflight_history' / f'{time.time_ns()}-{path.name}', saved)
        saved = None
    if saved and reference is None and saved.get('status') in ('ELIGIBLE', 'EXCLUDED'):
        if saved.get('selection_sha256') != stable_hash({k: v for k, v in saved.items() if k != 'selection_sha256'}):
            raise RuntimeError('Frozen selection hash mismatch')
        validate_materials(saved)
        return saved
    selected = selection.freeze_reference(ROOT / cfg['stage2_run'], 'smoke' if phase == 'pilot' else phase,
                                          row, tokenizer=tokenizer, model=model, reference=reference,
                                          feature_cache_path=run / 'features' / f'{fixed.key(row["unique_id"])}.npz')
    selected.pop('selection_sha256', None)
    selected['timing'] = selected.pop('timings', selected.get('timing', {}))
    selected.update(phase=phase, source_hashes=source_hashes(), frozen_at=now())
    selected['selection_sha256'] = stable_hash(selected)
    atomic_json(path, selected)
    storage.event(run, 'POSITIONS_FROZEN', unique_id=row['unique_id'], phase=phase, status=selected['status'],
                  selection_sha256=selected['selection_sha256'])
    return selected


def recollect(cfg, run, row, phase, selected, model, tokenizer, mismatch):
    uid = row['unique_id']
    if cfg.get('do_sample'):
        raise RuntimeError('Sampled reference mismatch: preserve diagnosis, never recollect or change seed') from mismatch
    if selected.get('reference_source') == 'stage3_streaming' or selected.get('recollected_streaming'):
        raise RuntimeError('A fresh streamed reference is not reproducible; stop without selecting another outcome')
    folder = run / 'reference_history' / fixed.key(uid)
    atomic_json(folder / f'{time.time_ns()}-stage2-selection.json', selected)
    archive_pair(run, uid, 'original reference token mismatch; recollect this question only')
    start = fixed.clock()
    reference = runtime.collect_reference(cfg, row, model, tokenizer)
    reference.update(recollection_reason=safe_error(mismatch), offline_wall_seconds=fixed.clock() - start)
    reference_path = run / 'references' / f'{fixed.key(uid)}.json'
    atomic_json(reference_path, reference)
    selected = freeze_selection(cfg, run, row, phase, tokenizer, model, reference)
    selected['recollected_streaming'] = True
    selected['reference_source'] = 'stage3_streaming'
    selected['source_material_hashes'][material_key(reference_path)] = digest(reference_path)
    selected.setdefault('timing', {})['offline_reference_seconds'] = reference['offline_wall_seconds']
    selected['selection_sha256'] = stable_hash({k: v for k, v in selected.items() if k != 'selection_sha256'})
    atomic_json(run / 'selections' / f'{fixed.key(uid)}.json', selected)
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
            result = {'status': 'GOLD_ERROR', 'error': safe_error(exc)}
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
    if cfg.get('verifier',{}).get('kind')=='generative':
        backup(run,manifest,f'{phase} before paired requests')
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
                        checks.update(source_hashes=source_hashes(), status='RUNNING', created=now())
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
        checks.update(status='PASS' if all_checked else 'INCOMPLETE', source_hashes=source_hashes(), updated=now())
        atomic_json(run / 'pilot_checks.json', checks)
    analyze.analyze(cfg, run, manifest)
    backup(run, manifest, f'{phase} checkpoint')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'high_low_config.json')
    parser.add_argument('--phase', choices=['prepare', 'self-check', 'pilot', 'freeze', 'test', 'analyze', 'backup', 'all', 'verifier-smoke', 'verifier-dev', 'verifier-audit', 'dev-report', 'audit-report', 'compare-verifiers', 'capacity'])
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Keep valid completed pairs; restart both arms of incomplete pairs')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--audit-extra','--review-extra',type=int,default=0,help='Append fixed unused questions; development and audit use disjoint alternating subsequences')
    parser.add_argument('--human-labels',type=Path,help='Confirmed independent human audit labels JSON array')
    parser.add_argument('--review-output',type=Path,help='Verifier review folder')
    parser.add_argument('--compare-review',type=Path,help='Second development candidate folder for compare-verifiers')
    parser.add_argument('--run', type=Path, help='Saved run for --phase analyze, including legacy runs')
    args = parser.parse_args(argv)
    phase = 'self-check' if args.self_check else args.phase
    if phase is None:
        parser.error('--phase or --self-check is required')
    if args.run is not None and phase != 'analyze':
        parser.error('--run is only for analysis; legacy execution uses its recorded commit')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if phase == 'compare-verifiers':
        if not args.review_output or not args.compare_review:
            parser.error('compare-verifiers requires --review-output and --compare-review')
        return compare_verifiers(args.review_output,args.compare_review,args.human_labels,args.config)
    if phase in ('verifier-smoke','verifier-dev','verifier-audit','dev-report','audit-report'):
        return verifier_review(read_json(args.config),phase,args.review_output,args.human_labels,args.audit_extra)
    if phase == 'self-check':
        with tempfile.TemporaryDirectory(prefix='veriserve-high-low-check-') as folder:
            return cpu_checks(read_json(args.config), Path(folder))
    if phase == 'analyze':
        from ...analysis import analyze_run, find_run
        return analyze_run(args.run or find_run(read_json(args.config), 'high-low'))
    cfg, run, manifest = prepare(args.config)
    print(f'RUN {run}', flush=True)
    try:
        if phase in ('pilot', 'all'):
            cpu_checks(cfg, run)
        if phase in ('prepare', 'self-check', 'analyze'):
            if phase != 'analyze':
                for split, rows in manifest['splits'].items():
                    for row in rows:
                        freeze_selection(cfg, run, row, split, None, None)
            analyze.analyze(cfg, run, manifest)
            if cfg.get('verifier',{}).get('kind')=='generative':
                backup(run,manifest,'prepared frozen inputs before model requests')
            return
        if phase == 'backup':
            backup(run, manifest, 'manual backup')
            return
        if phase == 'freeze':
            freeze(cfg, run, manifest)
            return
        if phase in ('pilot','all') and cfg.get('final_response_mode')=='explanation_then_final_line':
            audit=read_json(run/'verifier_audit_gate.json',{})
            if (audit.get('status')!='PASS' or audit.get('source_hashes')!=source_hashes()
                    or audit.get('config_hash')!=manifest['config_hash']):
                raise RuntimeError('Revised Stage3 pilots require the selected verifier independent human audit first')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            atomic_json(run / 'execution_status.json', {'status': 'NOT_EXECUTED', 'reason': 'BF16 CUDA unavailable', 'created': now()})
            analyze.analyze(cfg, run, manifest)
            print('GPU EXPERIMENT NOT EXECUTED: BF16 CUDA unavailable', flush=True)
            return
        backend = {'gpu': torch.cuda.get_device_name(0), 'uuid': str(torch.cuda.get_device_properties(0).uuid),
                   'total_memory': torch.cuda.get_device_properties(0).total_memory,
                   'implementation': cfg['cache_backend'], 'attention': cfg['attention'], 'dtype': cfg['dtype'],
                   'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                   'bitsandbytes': importlib.metadata.version('bitsandbytes'), 'source_sha256': stable_hash(source_hashes())}
        model, tokenizer, verifier = fixed.load_resident(cfg, run)
        if phase == 'capacity':
            return
        storage.archive_diagnostic(run, 'runtime_environment.json')
        atomic_json(run / 'runtime_environment.json', {'environment': environment(True), 'backend': backend,
                    'effective_generation_config': model.generation_config.to_dict(),
                    'execution_commit': git('rev-parse', 'HEAD'), 'created': now()})
        try:
            for step in ('pilot', 'test') if phase == 'all' else (phase,):
                run_phase(cfg, run, manifest, step, model, tokenizer, verifier, backend, args.stop_after)
        finally:
            del model, tokenizer, verifier
            gc.collect()
            torch.cuda.empty_cache()
    except (Exception, KeyboardInterrupt) as exc:
        storage.event(run, 'PHASE_INTERRUPTED', phase=phase, error=safe_error(exc))
        atomic_json(run / 'interruption.json', {'phase': phase, 'error': safe_error(exc), 'created': now(),
                    'resume_command': f'python -m stages.stage3.high_low --config {args.config} --phase {phase} --resume'})
        analyze.analyze(cfg, run, manifest)
        raise


def backup_verifier_review(stage2, folder, reason):
    """Both reference tokens and verifier judgments must reach the native remote."""
    from ...artifacts.backup import backup_probe
    backup_probe(stage2, f'Qwen3.5 verifier references: {reason}')
    status = backup_high_low(folder, {}, f'verifier review {reason}')
    if status.get('status') != 'PUSHED':
        raise RuntimeError('Verifier review remote backup failed; local data retained, stop before next question')


def review_manifest_hash(stage2,folder):
    """Keep original bytes while Stage2 adds formal-freeze bookkeeping."""
    path=stage2/'manifest.json'
    snapshot=folder/'stage2_manifest_snapshot.json'
    if not snapshot.exists():
        atomic_bytes(snapshot,path.read_bytes())
    original,current=read_json(snapshot),read_json(path)
    for key in ('schema_version','experiment_type','run_id','config','config_hash','source_hashes',
                'revisions','splits','unused_ids','dataset_integrity','split_manifest_sha256',
                'checkpoint_definition','hidden_index_convention','environment','grading'):
        if original.get(key)!=current.get(key):
            raise ValueError(f'Frozen Stage2 scientific inputs changed: {key}')
    return digest(snapshot)


def review_collection_complete(folder,split,records):
    status=read_json(folder/f'{split}_status.json',{})
    cohort=read_json(folder/f'{split}_ordered_ids.json')
    return (isinstance(cohort,list) and bool(cohort)
            and status.get('status') in ('COLLECTED','AWAITING_HUMAN_REVIEW')
            and status.get('questions')==len(cohort) and status.get('checks')==len(records))


def verifier_review(cfg, phase, output=None, human_labels=None, audit_extra=0):
    """Fixed length-position samples, blind human review export and held-out gate."""
    import json
    import numpy as np
    from ...inference import load_model, GenerativeVerifier
    from ...probe.collect import collect_one
    from ...trajectory import decode, verification_units
    if not cfg.get('stage2_run'):
        raise ValueError('Prepare new stage2 and set its actual stage2_run first')
    stage2 = ROOT/cfg['stage2_run']
    source = read_json(stage2/'manifest.json')
    if not source or source['config']['model'] != cfg['model']:
        raise ValueError('New stage2 manifest required')
    if cfg.get('final_response_mode')=='explanation_then_final_line':
        require_current(source,'probe')
    folder = output or HERE/'runs'/('verifier-'+run_identity(cfg,'high-low',{'stage2_run_id':source['run_id']}))
    folder=Path(folder).resolve()
    if not folder.is_relative_to(ROOT.resolve()):
        raise ValueError('Verifier review output must be inside the repository for required native backup')
    folder.mkdir(parents=True,exist_ok=True)
    identity = dict(config_hash=stable_hash(cfg),source_hashes=source_hashes(),stage2_manifest_sha256=review_manifest_hash(stage2,folder))
    saved = read_json(folder/'identity.json')
    if saved and saved != identity:
        raise ValueError('Verifier review execution changed; use new review folder')
    atomic_json(folder/'identity.json',identity)
    atomic_json(folder/'review_config.json',cfg)
    if phase in ('dev-report','audit-report'):
        return verifier_audit_report(cfg,folder,human_labels,'verifier_dev' if phase=='dev-report' else 'verifier_audit')
    split = {'verifier-smoke':'smoke','verifier-dev':'verifier_dev','verifier-audit':'verifier_audit'}[phase]
    cohort = list(source['splits'][split])
    if audit_extra < 0 or audit_extra and split not in ('verifier_dev','verifier_audit'):
        raise ValueError('review-extra must be nonnegative and only used for development or audit')
    if audit_extra:
        import hashlib
        unused=source['unused_ids'][0 if split=='verifier_dev' else 1::2]
        if audit_extra > len(unused):
            raise ValueError('Not enough unused audit questions')
        lookup = {}
        for line in (stage2/'omni_math_rule.jsonl').read_text().splitlines():
            row = json.loads(line)
            uid = hashlib.sha256(row['problem'].encode()).hexdigest()
            lookup.setdefault(uid,dict(row,unique_id=uid))
        cohort += [lookup[uid] for uid in unused[:audit_extra]]
    if split=='verifier_audit' and cfg.get('final_response_mode')=='explanation_then_final_line':
        if cfg.get('verifier',{}).get('selection_status')!='SELECTED_BY_DEVELOPMENT':
            raise ValueError('Independent audit is reserved for the development-selected verifier')
        # Reserve held-out questions once, so changing a failed verifier cannot reuse its audit.
        import hashlib
        pool=list(source['splits']['verifier_audit'])
        lookup={}
        for line in (stage2/'omni_math_rule.jsonl').read_text().splitlines():
            row=json.loads(line)
            uid=hashlib.sha256(row['problem'].encode()).hexdigest()
            lookup.setdefault(uid,dict(row,unique_id=uid))
        pool += [lookup[uid] for uid in source['unused_ids'][1::2]]
        registry_path=stage2/'verifier_audit_reservations.json'
        registry=read_json(registry_path,{})
        audit_key=stable_hash(identity)
        reserved=registry.get(audit_key,[])
        others={uid for key,ids in registry.items() if key!=audit_key for uid in ids}
        available=[row for row in pool if row['unique_id'] not in others and row['unique_id'] not in reserved]
        required=len(source['splits']['verifier_audit'])+audit_extra
        if required<len(reserved):
            raise ValueError('Audit cohort cannot shrink after inputs have been reserved')
        if len(available)<required-len(reserved):
            raise ValueError('Insufficient fresh independent audit questions')
        reserved += [row['unique_id'] for row in available[:required-len(reserved)]]
        registry[audit_key]=reserved
        atomic_json(registry_path,registry)
        all_rows={row['unique_id']:row for row in pool}
        cohort=[all_rows[uid] for uid in reserved]
    atomic_json(folder/f'{split}_ordered_ids.json',[r['unique_id'] for r in cohort])
    generator=tokenizer=verifier=None
    try:
        backup_verifier_review(stage2,folder,f'{phase} prepared inputs')
        generator,tokenizer=load_model(source['config'])
        verifier=GenerativeVerifier(cfg)
        review=[]
        for index,row in enumerate(cohort,1):
            record,_=collect_one(source['config'],stage2,split,row,generator,tokenizer,environment(True),features=False)
            steps=record['boundaries']['steps']
            counts=[]
            for fraction in (.25,.75):
                if steps:
                    target=fraction*len(record['generated_ids'])
                    counts.append((min(steps,key=lambda s:abs(s['prefix_token_count']-target))['prefix_token_count'],'intermediate'))
            if record['completion']=='EOS' and record['boundaries'].get('end'):
                counts.append((record['boundaries']['end']['prefix_token_count'],'terminal'))
            # The terminal check takes precedence when its exact prefix duplicates an intermediate one.
            for check_index,(count,kind) in enumerate(dict(counts).items()):
                prefix=decode(tokenizer,record['generated_ids'][:count])
                numbers,units=verification_units(prefix,cfg.get('boundary_mode','steps'),terminal=kind=='terminal')
                if not numbers:
                    continue
                ids=verifier.input_ids(row['problem'],prefix,max(numbers),[],kind)
                review_key=stable_hash([row['unique_id'],record['token_sha256'],count,kind])
                item=dict(unique_id=row['unique_id'],split=split,problem=row['problem'],prefix=prefix,
                          review_key=review_key,checkpoint_numbers=numbers,
                          prefix_ids=record['generated_ids'][:count],check_type=kind,input_ids=ids,
                          input_sha256=stable_hash(ids),identity=identity,reference_token_sha256=record['token_sha256'])
                path=folder/'checks'/f'{split}-{fixed.key(row["unique_id"])}-{count}.json'
                existing=read_json(path)
                if existing:
                    if existing['input_sha256'] != item['input_sha256'] or existing['identity'] != identity:
                        raise ValueError('Saved verifier input changed')
                    item=existing
                else:
                    repetitions=2 if split=='verifier_audit' and check_index==0 else 1
                    results=[]
                    for _ in range(repetitions):
                        started=fixed.clock()
                        result=verifier.verify(ids,row['unique_id'],numbers,0,kind,audit_repeat=len(results))
                        result['seconds']=fixed.clock()-started
                        results.append(result)
                    item['results']=results
                    atomic_json(path,item)
                review.append(dict(review_key=review_key,input_sha256=item['input_sha256'],unique_id=row['unique_id'],problem=row['problem'],
                                   quoted_prefix=prefix,checkpoint_units=units,check_type=kind,reference_answer=row['answer'],
                                   reference_solution=row['solution'],label=None,first_error_step=None,reviewer=None,confirmed=False))
            backup_verifier_review(stage2,folder,f'{phase} question {index}/{len(cohort)}')
        # Keep judgments separate from blind review inputs and from online model inputs.
        atomic_json(folder/f'{split}_review_inputs.json',review)
        result=dict(status='AWAITING_HUMAN_REVIEW' if split!='smoke' else 'COLLECTED',split=split,
                    questions=len(cohort),checks=len(review),folder=str(folder),**identity)
        atomic_json(folder/f'{split}_status.json',result)
        backup_verifier_review(stage2,folder,f'{phase} review export')
        print(result,flush=True)
        return result
    finally:
        del generator,tokenizer,verifier
        gc.collect()


def verifier_audit_report(cfg,folder,human_labels,split='verifier_audit'):
    import numpy as np
    from collections import Counter
    from ...artifacts.io import write_csv
    records=[read_json(p) for p in sorted((folder/'checks').glob(f'{split}-*.json'))]
    labels=read_json(human_labels,[]) if human_labels else []
    if not isinstance(labels,list):
        raise ValueError('Human labels must be a JSON array with explicit reviewer and confirmed=true')
    by_hash={}
    for label in labels:
        key=label.get('review_key') or label.get('input_sha256')
        if not key or key in by_hash:
            raise ValueError('Missing or duplicate human label key')
        if label.get('confirmed') is True and isinstance(label.get('reviewer'),str) and label['reviewer'].strip():
            if label.get('label') not in ('CORRECT','INCORRECT','UNDETERMINABLE'):
                raise ValueError('Invalid human label enum')
            if label['label']=='INCORRECT' and type(label.get('first_error_step')) is not int:
                raise ValueError('Human INCORRECT needs error step')
            by_hash[key]=label
    items=[]
    for record in records:
        label=by_hash.get(record.get('review_key')) or by_hash.get(record['input_sha256'])
        if (label and label['label']=='INCORRECT' and record.get('checkpoint_numbers')
                and label['first_error_step'] not in record['checkpoint_numbers']):
            raise ValueError('Human error step is outside the external checkpoint numbers')
        results=record['results']
        items.append(dict(unique_id=record['unique_id'],human=label['label'] if label else None,
                          result=results[0],consistent=results[0]['verdict']==results[1]['verdict'] if len(results)>1 else None))
    def mean(values):
        # Each question has equal weight within the metric's own denominator.
        counts=Counter(item['unique_id'] for item,_ in values)
        pairs=[(float(value),1/counts[item['unique_id']]) for item,value in values]
        return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None
    def metrics(rows):
        valid=[x for x in rows if x['human'] in ('CORRECT','INCORRECT')]
        correct=[x for x in valid if x['human']=='CORRECT']
        wrong=[x for x in valid if x['human']=='INCORRECT']
        recalls=[mean([(x,x['result']['verdict']==verdict) for x in cohort]) for cohort,verdict in ((correct,'PASS'),(wrong,'FAIL'))]
        return dict(accuracy=mean([(x,x['result']['verdict']==('PASS' if x['human']=='CORRECT' else 'FAIL')) for x in valid]),
                    balanced_accuracy=sum(recalls)/2 if all(r is not None for r in recalls) else None,
                    incorrect_pass_rate=mean([(x,x['result']['verdict']=='PASS') for x in wrong]),
                    correct_fail_rate=mean([(x,x['result']['verdict']=='FAIL') for x in correct]),
                    uncertain_or_format_or_truncated_rate=mean([(x,x['result']['verdict']=='UNCERTAIN' or bool(x['result'].get('parser_error')) or x['result'].get('completion','EOS')!='EOS') for x in rows]),
                    consistency=mean([(x,x['consistent']) for x in rows if x['consistent'] is not None]),
                    **{key:mean([(x,x['result'][key]) for x in rows]) for key in ('input_tokens','output_tokens','seconds')})
    point=metrics(items)
    rng=np.random.default_rng(cfg['seed'])
    ids=sorted({x['unique_id'] for x in items})
    samples={k:[] for k in point}
    if ids:
        for _ in range(cfg['bootstrap']):
            sampled=rng.choice(ids,len(ids),replace=True)
            replica=[dict(x,unique_id=str(i)) for i,uid in enumerate(sampled) for x in items if x['unique_id']==uid]
            for k,v in metrics(replica).items():
                if v is not None:samples[k].append(v)
    correct=sum(x['human']=='CORRECT' for x in items)
    wrong=sum(x['human']=='INCORRECT' for x in items)
    confirmed=sum(x['human'] is not None for x in items)
    minimum=10 if split=='verifier_dev' else 20
    collection_complete=review_collection_complete(folder,split,records)
    passed=(collection_complete and len(records)>0 and correct>=minimum and wrong>=minimum
            and all(point[k] is not None and point[k]<=.2 for k in ('incorrect_pass_rate','correct_fail_rate','uncertain_or_format_or_truncated_rate'))
            and point['balanced_accuracy'] is not None and point['balanced_accuracy']>=.8)
    seconds=[x['result']['seconds'] for x in items]
    repeated_seconds=[r['seconds'] for record in records for r in record['results'][1:]]
    result=dict(status='PASS' if passed else 'INSUFFICIENT_OR_FAILED',split=split,source_hashes=source_hashes(),
                config_hash=stable_hash(cfg),human_labels_sha256=digest(human_labels) if human_labels else None,
                review_identity=read_json(folder/'identity.json'),questions=len(ids),checks=len(records),confirmed=confirmed,
                collection_complete=collection_complete,
                unconfirmed=len(records)-confirmed,required_each_class=minimum,
                correct_prefixes=correct,incorrect_prefixes=wrong,undeterminable=sum(x['human']=='UNDETERMINABLE' for x in items),
                repeated_prefixes=sum(x['consistent'] is not None for x in items),
                repeat_full_call_seconds=repeated_seconds,
                raw_counts=dict(incorrect_pass=sum(x['human']=='INCORRECT' and x['result']['verdict']=='PASS' for x in items),
                    correct_fail=sum(x['human']=='CORRECT' and x['result']['verdict']=='FAIL' for x in items),
                    uncertain_or_format_or_truncated=sum(x['result']['verdict']=='UNCERTAIN' or bool(x['result'].get('parser_error')) or x['result'].get('completion','EOS')!='EOS' for x in items)),
                completion_counts=dict(Counter(x['result'].get('completion','EOS') for x in items)),
                verdict_counts=dict(Counter(x['result']['verdict'] for x in items)),
                parser_error_counts=dict(Counter(x['result'].get('parser_error') for x in items if x['result'].get('parser_error'))),
                full_call_seconds=dict(mean=float(np.mean(seconds)) if seconds else None,
                    quantiles=dict(zip(('p50','p90','p95'),np.quantile(seconds,[.5,.9,.95]).tolist())) if seconds else {}),
                metrics={k:dict(value=v,ci95=np.quantile(samples[k],[.025,.975]).tolist() if samples[k] else None,
                                bootstrap_valid=len(samples[k])) for k,v in point.items()},
                prespecified_gate=f'Complete fixed collection; >={minimum} confirmed correct and >={minimum} confirmed incorrect; incorrect PASS, correct FAIL and all-input UNCERTAIN/format/truncation each <=20%; Balanced Accuracy >=80%; unconfirmed inputs reported separately')
    name='verifier_dev_gate' if split=='verifier_dev' else 'verifier_audit_gate'
    atomic_json(folder/f'{name}.json',result)
    write_csv(folder/f'{split}_checks.csv',[dict(unique_id=x['unique_id'],human=x['human'],verdict=x['result']['verdict'],
        input_tokens=x['result']['input_tokens'],output_tokens=x['result']['output_tokens'],seconds=x['result']['seconds']) for x in items],
        ['unique_id','human','verdict','input_tokens','output_tokens','seconds'])
    if passed and split=='verifier_audit':
        _,run,manifest=prepare_config_for_review_gate(cfg)
        atomic_json(run/'verifier_audit_gate.json',result)
    print(result,flush=True)
    return result


def compare_verifiers(first,second,human_labels,config_path):
    folders=[Path(first).resolve(),Path(second).resolve()]
    configs=[read_json(p/'review_config.json') for p in folders]
    records=[[read_json(p) for p in sorted((folder/'checks').glob('verifier_dev-*.json'))] for folder in folders]
    if any(not rows for rows in records) or {c['verifier']['enable_thinking'] for c in configs}!={False,True}:
        raise ValueError('Collect both fixed development candidates first')
    keyed=[{r['review_key']:(r['unique_id'],r['prefix_ids'],r['check_type']) for r in rows} for rows in records]
    if keyed[0]!=keyed[1]:
        raise ValueError('Candidates did not check identical mathematical inputs')
    cohorts=[read_json(p/'verifier_dev_ordered_ids.json') for p in folders]
    if cohorts[0]!=cohorts[1]:
        raise ValueError('Development cohorts differ')
    for folder,cfg,cohort,rows in zip(folders,configs,cohorts,records):
        if not review_collection_complete(folder,'verifier_dev',rows):
            raise ValueError('Complete both fixed development cohorts before mode selection')
        expected=dict(config_hash=stable_hash(cfg),source_hashes=source_hashes(),
                      stage2_manifest_sha256=review_manifest_hash(ROOT/cfg['stage2_run'],folder))
        if read_json(folder/'identity.json')!=expected:
            raise ValueError('Candidate review config/source provenance changed')
    table=folders[0].parent/'development_human_review.json'
    existing={r['review_key']:r for r in read_json(table,[])}
    atomic_json(table,[existing.get(r['review_key'],r) for r in read_json(folders[0]/'verifier_dev_review_inputs.json')])
    reports=[verifier_audit_report(cfg,folder,human_labels,'verifier_dev') for cfg,folder in zip(configs,folders)]
    eligible=[i for i,r in enumerate(reports) if r['status']=='PASS']
    winner=min(eligible,key=lambda i:reports[i]['full_call_seconds']['mean']) if eligible else None
    report_path=folders[0].parent/'development_comparison.json'
    result=dict(status='SELECTED_BY_DEVELOPMENT' if winner is not None else
                ('AWAITING_HUMAN_LABELS' if any(r['confirmed']<r['checks'] for r in reports) else
                 'INSUFFICIENT_CLASS_LABELS' if any(min(r['correct_prefixes'],r['incorrect_prefixes'])<10 for r in reports) else 'NO_CANDIDATE_PASSED'),
                selected_mode=('thinking' if configs[winner]['verifier']['enable_thinking'] else 'direct') if winner is not None else None,
                source_hashes=source_hashes(),candidate_folders=[str(p) for p in folders],candidate_reports=reports,
                human_review_table=str(table),selection_rule='Quality gates first; mean full call time on all inputs breaks ties',created=now())
    atomic_json(report_path,result)
    if winner is not None:
        cfg=read_json(config_path)
        cfg['verifier']=dict(configs[winner]['verifier'],selection_status='SELECTED_BY_DEVELOPMENT')
        cfg['verifier_development_comparison']=str(report_path.relative_to(ROOT))
        cfg['verifier_development_comparison_sha256']=digest(report_path)
        atomic_json(config_path,cfg)
    print(result,flush=True)
    return result


def prepare_config_for_review_gate(cfg):
    # Reuse the runner's exact manifest identity; never create a separate gate identity.
    with tempfile.TemporaryDirectory(prefix='verifier-gate-config-') as folder:
        path=Path(folder)/'config.json'
        atomic_json(path,cfg)
        return prepare(path)
