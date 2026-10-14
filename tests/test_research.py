"""Research boundaries, CPU checks and regression on the retained experiment materials."""
import ast
import json
import shutil
import subprocess
import sys

import pytest

from veriserve_research import ROOT
from veriserve_research.artifacts import backup, provenance
from veriserve_research.artifacts.io import atomic_json, digest, read_json, stable_hash
from veriserve_research.artifacts.materials import material_key, material_path, validate_materials


def test_revised_final_response_shared_contract():
    from veriserve_research.trajectory import final_answer_line,grade,PARAGRAPHS,EXPLANATION_FINAL
    from veriserve_research.intervention.protocol import terminal
    class Tokenizer:
        all_special_ids=[0]
        def get_vocab(self):return {}
        def decode(self,ids,**kwargs):return ''.join(chr(i) if i else '<|im_end|>' for i in ids)
    tok=Tokenizer()
    raw='A draft \\boxed{999}.\n</think>\nAn explanation with \\boxed{7}.\nFinal answer: \\boxed{\\frac{1}{2}}'
    assert final_answer_line(raw+'<|im_end|>')['answer']==r'\frac{1}{2}'
    assert final_answer_line(raw+'<|im_end|>\n<|endoftext|>')['answer']==r'\frac{1}{2}'
    assert final_answer_line(raw+'<|unknown_control|><|im_end|>')['exclusion']
    assert grade(raw,'0.5','EOS',5,PARAGRAPHS,EXPLANATION_FINAL)['label']==0
    assert grade(raw,'0.5','EOS',5,PARAGRAPHS)['exclusion']=='FINAL_FORMAT_ERROR'
    assert terminal(tok,[ord(c) for c in raw]+[0],True,mode=PARAGRAPHS,final_response_mode=EXPLANATION_FINAL)['answer']==r'\frac{1}{2}'
    for bad in (raw+'\nExtra.',raw+'\nFinal answer: \\boxed{1}',
                raw+' prose',raw.replace(r'\boxed{\frac{1}{2}}',r'\boxed{\frac{1}{2}'),
                raw.replace(r'\boxed{\frac{1}{2}}',r'\boxed{1} \\boxed{2}'),
                raw.split('</think>')[0],raw.replace('</think>','')):
        assert final_answer_line(bad)['exclusion']
    assert final_answer_line(raw,'LENGTH_TRUNCATED')['exclusion']=='LENGTH_TRUNCATED'
    from veriserve_research.trajectory import boundaries
    from veriserve_research.intervention.high_low.selection import _reference_bounds
    reference='First assertion.\n\nSecond assertion.\n\n'+raw
    ids=[ord(c) for c in reference]+[0]
    _,bounds=boundaries(tok,ids,PARAGRAPHS)
    record=dict(enable_thinking=True,boundary_mode=PARAGRAPHS,completion='EOS',prompt_ids=[1],
                generated_ids=ids,raw_text=tok.decode(ids),boundaries=bounds,
                token_sha256=stable_hash({'prompt_ids':[1],'generated_ids':ids}))
    assert _reference_bounds(record,tok,PARAGRAPHS,EXPLANATION_FINAL)==bounds
    with pytest.raises(ValueError,match='REFERENCE_FINAL_FORMAT_ERROR'):
        _reference_bounds(record,tok,PARAGRAPHS)
    for suffix in ('\n\n','\r\n\t'):
        ids=[ord(c) for c in reference+suffix]+[0]
        _,bounds=boundaries(tok,ids,PARAGRAPHS)
        record.update(generated_ids=ids,raw_text=tok.decode(ids),boundaries=bounds,
                      token_sha256=stable_hash({'prompt_ids':[1],'generated_ids':ids}))
        assert _reference_bounds(record,None,PARAGRAPHS,EXPLANATION_FINAL)==bounds


def test_independent_verifier_modes_and_rng(monkeypatch):
    from types import SimpleNamespace
    import torch
    from veriserve_research import inference
    from veriserve_research.intervention.protocol import parse_verdict
    payload='{"verdict":"PASS","first_error_step":null,"diagnosis":"","hint":""}'
    assert parse_verdict(payload,[1],enable_thinking=False)['verdict']=='PASS'
    assert parse_verdict('Reasoning</think>'+payload,[1])['verdict']=='PASS'
    for raw in ('prefix'+payload,'```json\n'+payload+'\n```','Reasoning</think>'+payload,
                payload.replace('"verdict":"PASS"','"verdict":"PASS","verdict":"FAIL"')):
        assert parse_verdict(raw,[1],enable_thinking=False)['parser_error']
    assert parse_verdict(payload,[1],completion='LENGTH_TRUNCATED',enable_thinking=False)['verdict']=='UNCERTAIN'
    calls=[]
    class FakeSession:
        def __init__(self,model,ids,cfg,seed):calls.append((cfg,seed))
        def sample(self):return 0
    class Tokenizer:
        def decode(self,ids,**kwargs):return payload
        def apply_chat_template(self,messages,**kwargs):calls.append(kwargs);return [1]
    v=object.__new__(inference.GenerativeVerifier);v.model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[0]));v.tokenizer=Tokenizer()
    v.cfg={'seed':7,'temperature':1.,'verifier':{'enable_thinking':False,'max_new_tokens':1,'max_total_tokens':10,'sampling':{'do_sample':True,'temperature':.7,'top_p':.8}}}
    monkeypatch.setattr(inference,'Session',FakeSession);monkeypatch.setattr(inference,'clock',lambda:0.)
    before=torch.random.get_rng_state().clone();v.input_ids('problem','Step 1: 2+2=4',1,[],'intermediate')
    assert calls[-1]['enable_thinking'] is False
    a=v.verify([1],'uid',[1],0,'intermediate');b=v.verify([1],'uid',[1],0,'intermediate')
    assert a['generation_seed']==b['generation_seed'] and a['sampling']['temperature']==.7
    assert torch.equal(before,torch.random.get_rng_state())
    v.cfg['verifier']['enable_thinking']=True
    c=v.verify([1],'uid',[1],0,'intermediate')
    assert c['generation_seed']!=a['generation_seed']


@pytest.mark.parametrize("experiment", ("probe", "fixed-step", "high-low"))
def test_cpu_checks(experiment):
    from veriserve_research.__main__ import self_check
    result = self_check(experiment)[experiment]
    assert result.get("status") == "PASS" or result.get("passed") is True
    if experiment == "probe":
        assert len(result["checks"]) == 34


def test_execution_and_analysis_identity_are_separate(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "veriserve_research", tmp_path / "veriserve_research",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "stages/stage1/veriserve", tmp_path / "stages/stage1/veriserve",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    cfg = {"seed": 7}
    before = provenance.source_hashes("high-low")
    analysis = provenance.analysis_hashes()
    identity = provenance.run_identity(cfg, "high-low", {"material": "a"})
    report = tmp_path / "veriserve_research/analysis/high_low.py"
    report.write_text(report.read_text() + "\n# report-only change\n")
    assert provenance.source_hashes("high-low") == before
    assert provenance.analysis_hashes() != analysis
    assert provenance.run_identity(cfg, "high-low", {"material": "a"}) == identity
    assert provenance.run_identity(cfg, "high-low", {"material": "b"}) != identity
    manifest = provenance.manifest_version("high-low")
    scientific = tmp_path / "veriserve_research/trajectory.py"
    scientific.write_text(scientific.read_text() + "\n# execution change\n")
    assert provenance.run_identity(cfg, "high-low", {"material": "a"}) != identity
    with pytest.raises(ValueError, match="Execution source changed"):
        provenance.require_current(manifest, "high-low")
    with pytest.raises(ValueError, match="Legacy execution"):
        provenance.require_current({"schema_version": 1}, "high-low")
    shared = tmp_path / "veriserve_research/probe/__init__.py"
    before = {name: provenance.source_hashes(name) for name in ("probe", "fixed-step", "high-low")}
    shared.write_text(shared.read_text() + "\n# shared package change\n")
    assert all(provenance.source_hashes(name) != hashes for name, hashes in before.items())


def test_material_relocation_and_corruption(tmp_path):
    relative = "stages/stage2/runs/example/probe_B.json"
    current = tmp_path / relative
    atomic_json(current, {"b": 1.0})
    old = "/another/machine/VeriServe/" + relative
    record = {"source_material_hashes": {old: digest(current)}}
    original = json.dumps(record)
    assert material_path(old, root=tmp_path) == current
    assert material_key(current, root=tmp_path) == relative
    validate_materials(record, root=tmp_path)
    assert json.dumps(record) == original
    current.write_text("changed")
    with pytest.raises(ValueError, match="changed source material"):
        validate_materials(record, root=tmp_path)
    current.unlink()
    with pytest.raises(ValueError, match="Missing"):
        validate_materials(record, root=tmp_path)
    with pytest.raises(ValueError, match="parent directories"):
        material_path("../outside", root=tmp_path)


def test_analysis_never_repairs_original_tokens(tmp_path):
    from veriserve_research.probe.materials import load_examples, record_path
    row = {"unique_id": "broken"}
    path = record_path(tmp_path, "test", row["unique_id"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{broken")
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="leaves source files unchanged"):
        load_examples(tmp_path, {"splits": {"test": [row]}}, "test")
    assert path.read_bytes() == b"{broken"
    assert set(tmp_path.rglob("*")) == before


def test_fixed_step_archives_partial_pair_and_skips_completed_pair(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from veriserve_research.artifacts import storage
    from veriserve_research.intervention import fixed_step
    from veriserve_research.analysis import fixed_step as reports
    cfg = read_json(ROOT / "stages/stage3/config.json")
    row = {"unique_id": "interrupted-pair"}
    manifest = {"splits": {"pilot": [row]}}
    snapshot = {**row, "phase": "pilot", "status": "ELIGIBLE"}
    hashes = provenance.source_hashes()
    atomic_json(tmp_path / "self_check.json", {"status": "PASS", "source_hashes": hashes})
    atomic_json(tmp_path / "backup.json", {"status": "PUSHED", "source_hashes": hashes})
    stem = fixed_step.key(row["unique_id"])
    atomic_json(tmp_path / "snapshots" / f"{stem}.json", snapshot)
    first = tmp_path / "arms" / f"{stem}.NOW.json"
    atomic_json(first, {**row, "phase": "pilot", "completed": True, "source_hashes": hashes})
    original = first.read_bytes()
    model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {}),
                            config=SimpleNamespace(_name_or_path="synthetic-model-not-downloaded"))
    monkeypatch.setattr(fixed_step, "environment", lambda **kwargs: {})
    monkeypatch.setattr(fixed_step, "load_resident", lambda *args: (model, None, None))
    monkeypatch.setattr(fixed_step, "probe_files", lambda *args: {})
    monkeypatch.setattr(fixed_step, "initial_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(fixed_step, "pilot_checks", lambda *args: None)
    monkeypatch.setattr(fixed_step, "validate_evaluation_data", lambda *args: None)
    monkeypatch.setattr(storage, "backup", lambda *args: None)
    monkeypatch.setattr(reports, "analyze", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "synthetic")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=1, uuid="cpu"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    executed = []

    def execute(config, run, item, saved, arm, *args):
        executed.append(arm)
        atomic_json(run / "arms" / f"{stem}.{arm}.json", {
            **row, "phase": "pilot", "completed": True, "status": "COMPLETE", "source_hashes": hashes})

    monkeypatch.setattr(fixed_step, "execute_arm", execute)
    fixed_step.run_phase(cfg, tmp_path, manifest, "pilot")
    assert executed == ["NOW", "DELAY_2"]
    archived = list((tmp_path / "interrupted_pairs").rglob("*.json"))
    assert len(archived) == 1 and archived[0].read_bytes() == original
    before = fixed_step.artifact_hashes(tmp_path, "pilot")
    monkeypatch.setattr(fixed_step, "load_resident", lambda *args: pytest.fail("Completed pair loaded GPU models"))
    fixed_step.run_phase(cfg, tmp_path, manifest, "pilot")
    assert executed == ["NOW", "DELAY_2"]
    assert fixed_step.artifact_hashes(tmp_path, "pilot") == before


def test_prepare_v2_manifests_and_resume(tmp_path, monkeypatch):
    import datasets
    from veriserve_research.probe import collect
    from veriserve_research.artifacts import storage
    from veriserve_research.intervention.high_low import run as high_low
    source = read_json(ROOT / "stages/stage2/runs/20261004-36fe9009f1f5/manifest.json")
    old_fixed = read_json(ROOT / "stages/stage3/runs/20261004-109173095d08/manifest.json")
    rows = [row for split in source["splits"].values() for row in split]
    rows += [row for split in old_fixed["splits"].values() for row in split]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(collect, "HERE", tmp_path / "stage2")
    monkeypatch.setattr(collect, "environment", lambda **kwargs: {})
    config = tmp_path / "probe.json"
    shutil.copyfile(ROOT / "stages/stage2/config.json", config)
    original = config.read_bytes()
    cfg, run, manifest = collect.prepare(config)
    assert run.name.startswith("v2-") and manifest["schema_version"] == 2
    assert config.read_bytes() == original
    assert collect.prepare(config)[1:] == (run, manifest)
    monkeypatch.setattr(storage, "HERE", tmp_path / "stage3")
    _, fixed_run, fixed_manifest = storage.prepare(ROOT / "stages/stage3/config.json")
    assert fixed_run.name.startswith("v2-") and fixed_manifest["schema_version"] == 2
    assert [r["unique_id"] for split in fixed_manifest["splits"].values() for r in split] == source["unused_ids"]
    assert storage.prepare(ROOT / "stages/stage3/config.json")[1] == fixed_run
    monkeypatch.setattr(high_low, "HERE", tmp_path / "high_low")
    (tmp_path / "high_low").mkdir()
    shutil.copyfile(ROOT / "stages/stage3/config.json", tmp_path / "high_low/config.json")
    _, high_run, high_manifest = high_low.prepare(ROOT / "stages/stage3/high_low_config.json")
    assert high_run.name.startswith("v2-") and high_manifest["schema_version"] == 2
    assert high_low.prepare(ROOT / "stages/stage3/high_low_config.json")[1] == high_run


@pytest.mark.parametrize("args", (
    ["probe", "--help"],
    ["intervene", "--experiment", "fixed-step", "--phase", "self-check"],
    ["intervene", "--experiment", "high-low", "--self-check"],
))
def test_functional_cli_dispatch(args):
    from veriserve_research.__main__ import main
    if "--help" in args:
        with pytest.raises(SystemExit) as result:
            main(args)
        assert result.value.code == 0
    else:
        assert main(args)["status"] == "PASS"


@pytest.mark.parametrize("module", (
    "veriserve_research", "stages.stage2.run_probe", "stages.stage3.run_timing", "stages.stage3.high_low",
))
def test_command_help(module, tmp_path):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("run_name", (
    "20261004-109173095d08", "20261004-bcab71671427",
    "20261007-9fe830084094", "20261007-acde3028d49c",
))
def test_intervention_metrics_match_legacy(run_name, tmp_path, monkeypatch):
    from veriserve_research.analysis import analyze_run
    from veriserve_research import inference
    monkeypatch.setattr(inference, "load_model", lambda *args: pytest.fail("Analysis loaded a model"))
    run = ROOT / "stages/stage3/runs" / run_name
    baseline = read_json(ROOT / "tests/fixtures/research_baseline.json")
    original_metrics = (run / "metrics.json").read_bytes()
    actual = analyze_run(run, tmp_path / "analysis")
    scientific = {k: v for k, v in actual.items() if k != "analysis_output"}
    assert stable_hash(scientific) == baseline["scientific_metrics_sha256"][run_name]
    if run_name in ("20261004-109173095d08", "20261007-9fe830084094"):
        assert scientific == read_json(run / "metrics.json")
    provenance_data = read_json(tmp_path / "analysis/analysis_provenance.json")
    assert all(digest(run / path) == sha for path, sha in provenance_data["input_hashes"].items())
    assert (run / "metrics.json").read_bytes() == original_metrics
    with pytest.raises(ValueError, match="separate"):
        analyze_run(run, run)


def test_probe_metrics_and_predictions_match_legacy(tmp_path):
    from veriserve_research.analysis import analyze_run
    run = ROOT / "stages/stage2/runs/20261004-36fe9009f1f5"
    expected = read_json(run / "metrics.json")
    actual = analyze_run(run, tmp_path / "analysis")
    for key, value in expected.items():
        if key not in ("analysis_environment", "scoring_seconds", "counts"):
            assert actual[key] == value, key
    for split, counts in expected["counts"].items():
        assert actual["counts"][split] == counts
    assert (tmp_path / "analysis/test_predictions.csv").read_bytes() == (run / "test_predictions.csv").read_bytes()


def test_reference_positions_match_all_legacy_selections():
    from veriserve_research.intervention.high_low.selection import freeze_reference, load_source
    from veriserve_research.intervention.fixed_step import key
    source = ROOT / "stages/stage2/runs/20261004-36fe9009f1f5"
    run = ROOT / "stages/stage3/runs/20261007-9fe830084094"
    _, splits = load_source(source)
    for phase, rows in splits.items():
        for row in rows:
            saved = read_json(run / "selections" / f"{key(row['unique_id'])}.json")
            reference = read_json(run / "references" / f"{key(row['unique_id'])}.json")
            actual = freeze_reference(source, "smoke" if phase == "pilot" else phase, row,
                                      reference=reference, feature_cache_path=run / "features" / f"{key(row['unique_id'])}.npz")
            for field in ("status", "positions", "exclusion_reasons"):
                assert actual[field] == saved[field], (row["unique_id"], field)


def test_backup_failure_policies(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("remote unavailable https://secret@example.invalid/repo")
    monkeypatch.setattr(backup, "git", unavailable)
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    run = tmp_path / "stages/stage3/runs/v2-fixture"
    with pytest.raises(RuntimeError, match="collection stopped"):
        backup.backup_probe(run, "test")
    assert read_json(run / "backup.json")["status"] == "FAILED"
    atomic_json(run / "backup.json", {"status": "PUSHED", "commit": "a" * 40})
    with pytest.raises(RuntimeError, match="stop before next batch"):
        backup.backup_fixed_step(run, "test")
    assert read_json(run / "backup.json")["last_successful_commit"] == "a" * 40
    result = backup.backup_high_low(run, {"pilot": 0, "test": 0}, "test")
    assert result["status"] == "LOCAL_ONLY"
    assert "secret@" not in result["error"]


def test_no_research_library_imports_stage_runners():
    for path in (ROOT / "veriserve_research").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("stages.stage2", "stages.stage3")), path


def test_qwen35_shared_sampling_thinking_and_full_snapshots():
    from veriserve_research.inference import migration_self_check
    assert migration_self_check()['status'] == 'PASS'


def test_rule_dataset_deduplicates_and_preserves_conflicts(tmp_path):
    import hashlib
    from veriserve_research.probe.collect import rule_rows
    rows = [dict(problem='Question '+str(i),answer='2',solution='Offline only') for i in range(5)]
    rows.append(rows[0])
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    atomic_json(tmp_path/'not-used.json',{})
    (tmp_path/'omni_math_rule.jsonl').write_bytes(raw)
    cfg = dict(dataset_url='https://example.invalid/fixed',dataset_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
               revisions={'dataset':'a'*40},dataset_expected_rows=6,dataset_expected_unique=5,score_timeout_seconds=5)
    unique,info = rule_rows(cfg,tmp_path)
    assert info['rows'] == 6 and info['unique'] == 5 and not info['invalid_gold_count']
    assert unique[0]['unique_id'] == hashlib.sha256(rows[0]['problem'].encode()).hexdigest()
    rows[-1] = {**rows[0],'answer':'3'}
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    (tmp_path/'omni_math_rule.jsonl').write_bytes(raw)
    cfg['dataset_blob'] = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    with pytest.raises(ValueError,match='counts/conflicts'):
        rule_rows(cfg,tmp_path)
    assert read_json(tmp_path/'dataset_integrity.json')['conflicting_answers']


def test_high_low_layer_comes_from_probe():
    from veriserve_research.intervention.high_low.selection import self_check
    assert self_check(layer=7)['status'] == 'PASS'


@pytest.mark.parametrize('boundary_mode',('steps','thinking_paragraphs'))
def test_generative_check_pass_fail_uncertain_and_accounting(tmp_path,monkeypatch,boundary_mode):
    from veriserve_research.intervention.high_low import runtime
    from veriserve_research.inference import Session
    from types import SimpleNamespace
    cfg=read_json(ROOT/'stages/stage3/qwen35_4b_omni_high_low_config.json')
    cfg['boundary_mode']=boundary_mode
    monkeypatch.setattr(runtime,'clock',lambda:0.)
    class Tok:
        def decode(self,ids,**kwargs):
            parts={1:'prompt',2:'Step 1:',3:' x',4:'\nStep 2:',5:' y',8:'feedback'}
            if boundary_mode == 'thinking_paragraphs':
                parts.update({2:'Compute',3:' 1+1=2.',4:'\n\n',5:'Verify 2=2.'})
            return ''.join(parts[i] for i in ids)
        def encode(self,text,**kwargs):
            return [8]
    class Verifier:
        def input_ids(self,problem,prefix,current_step,feedback,kind):
            assert problem == 'problem' and kind in ('intermediate','terminal')
            assert 'GOLD' not in prefix
            return [1,2]
        def verify(self,ids,uid,numbers,checkpoint_step,kind):
            payload=json.dumps(dict(verdict='PASS',first_error_step=None,diagnosis='',hint=''))
            result=runtime.protocol.parse_verdict('Checked.</think>'+payload,numbers,checkpoint_step)
            result.update(input_tokens=len(ids),output_tokens=7,seconds=.25,prefill_seconds=.05,decode_seconds=.2)
            return result
    class Active:
        def __init__(self):
            self.ids=[1,2,3]
            self.cached_length=2
            self.prompt_checkpoint={'ids':[1]}
            self.pass_checkpoint=None
        def ensure(self):
            charge=len(self.ids)-self.cached_length
            self.cached_length=len(self.ids)
            return charge
        def snapshot(self):
            return {'ids':self.ids.copy()}
        def restore(self,snapshot,feedback):
            self.ids=snapshot['ids']+feedback
            self.cached_length=len(self.ids)
            return len(feedback)
    selected={'phase':'pilot','prompt_ids':[1],'positions':{'HIGH':{}}}
    state=runtime.initial_state({'unique_id':'id','problem':'problem'},selected,'HIGH',{})
    state.update(reasoning_ids=[2,3],reasoning_positions=[1,2],context_ids=[1,2,3])
    state['budget'].update(generated_tokens=2,attempt_tokens=2)
    session=Active()
    assert runtime.check_generative(cfg,state,session,Verifier(),Tok(),2,False) is None
    assert state['checkpoint_ids']==[1,2,3] and state['budget']['forward_tokens']==4
    assert state['checks'][0]['input_tokens']==2 and state['checks'][0]['prefill_seconds']==.05
    assert state['budget']['verifier_generated_tokens']==7
    # A later FAIL contradicting the PASS becomes UNCERTAIN and cannot submit.
    state['reasoning_ids'] += [4,5]
    state['reasoning_positions'] += [3,4]
    state['context_ids'] += [4,5]
    session.ids += [4,5]
    state['budget'].update(generated_tokens=4,attempt_tokens=4)
    conflict={'verdict':'FAIL','first_error_step':1,'diagnosis':'bad','hint':'fix'}
    assert runtime.check_generative(cfg,state,session,Verifier(),Tok(),4,True,forced=conflict)=='TERMINAL_UNCERTAIN'
    assert state['budget']['rollbacks']==0 and 'submitted_reasoning_ids' not in state
    assert runtime.check_generative(cfg,state,session,Verifier(),Tok(),4,True,
        forced={**conflict,'first_error_step':2}) is None
    assert session.ids==[1,2,3,8] and state['budget']['rollbacks']==1
    assert state['budget']['generated_tokens']==4 and state['budget']['attempt_tokens']==0
    assert state['budget']['revoked_tokens']==2 and state['budget']['feedback_tokens']==1
    assert session.pass_checkpoint['ids']==[1,2,3]


def test_paragraph_checkpoint_selection_verifier_and_terminal_contract():
    from veriserve_research.trajectory import boundaries,decode,grade,verification_units,PARAGRAPHS,thinking_view
    from veriserve_research.intervention import protocol
    from veriserve_research.intervention.high_low.selection import _reference_bounds
    from veriserve_research.inference import GenerativeVerifier
    class Tok:
        all_special_ids=[0]
        def encode(self,text,**kwargs):return [ord(c)+1 for c in text]
        def decode(self,ids,**kwargs):return ''.join(chr(i-1) if i else '<|im_end|>' for i in ids)
        def apply_chat_template(self,messages,**kwargs):
            self.messages=messages
            return [1,2]
    tok=Tok()
    raw='计算 $1+1=2$。\n\nStep 1: Verify 2=2.\n\nFinal answer: \\boxed{2}\n\n</think>\nFinal answer: \\boxed{2}'
    ids=tok.encode(raw)+[0]
    text,bounds=boundaries(tok,ids,PARAGRAPHS)
    assert len(bounds['steps'])==2 and len(bounds['excluded_steps'])==1
    assert bounds['steps'][-1]['near_end'] and not bounds['format_errors']
    for point in bounds['steps']:
        prefix=decode(tok,ids[:point['prefix_token_count']])
        assert len(prefix)==point['char_end'] and raw.startswith(prefix)
        assert point['delimiter_char_start']>=len(prefix)
    prefix=decode(tok,ids[:bounds['steps'][1]['prefix_token_count']])
    numbers,units=verification_units(prefix,PARAGRAPHS)
    assert numbers==[1,2] and ''.join(u['quoted_text'] for u in units)==thinking_view(prefix)
    verifier=object.__new__(GenerativeVerifier)
    verifier.cfg={'boundary_mode':PARAGRAPHS}
    verifier.tokenizer=tok
    assert verifier.input_ids('problem',prefix,2,[],'intermediate')==[1,2]
    payload=json.loads(tok.messages[1]['content'])
    assert payload['quoted_prefix']==units and 'HIGH' not in json.dumps(payload)
    terminal_numbers,terminal_units=verification_units(raw,PARAGRAPHS,terminal=True)
    verifier.input_ids('Check x < 2.',raw,max(terminal_numbers),[],'terminal')
    encoded=tok.messages[1]['content']
    restored=json.loads(encoded)
    assert '<' not in encoded and '>' not in encoded
    assert restored['problem']=='Check x < 2.' and restored['quoted_prefix']==terminal_units
    assert ''.join(unit['quoted_text'] for unit in restored['quoted_prefix'])==thinking_view(raw)
    assert len(''.join(unit['quoted_text'] for unit in restored['quoted_prefix']))==len(raw)
    endpoint=protocol.terminal(tok,ids,True,mode=PARAGRAPHS)
    assert endpoint['reason']=='FINAL' and endpoint['answer']=='2'
    assert grade(text,'2','EOS',5,PARAGRAPHS)['label']==0
    bad=raw.replace('</think>\n','</think>\nHere is my solution.\n')
    assert protocol.terminal(tok,tok.encode(bad)+[0],True,mode=PARAGRAPHS)['reason']=='FORMAT_ERROR'
    assert grade(bad,'2','EOS',5,PARAGRAPHS)['exclusion']=='FINAL_FORMAT_ERROR'
    assert grade(bad,'2','LENGTH_TRUNCATED',5,PARAGRAPHS)['exclusion']=='LENGTH_TRUNCATED'
    record=dict(prompt_ids=[1],generated_ids=ids,raw_text=decode(tok,ids),completion='EOS',
                boundaries=bounds,enable_thinking=True,boundary_mode=PARAGRAPHS)
    record['token_sha256']=stable_hash({'prompt_ids':[1],'generated_ids':ids})
    assert _reference_bounds(record,tok,PARAGRAPHS)==bounds
    unfinished=tok.encode('A complete paragraph.\n\nUnfinished $x')
    assert len(boundaries(tok,unfinished,PARAGRAPHS)[1]['steps'])==1
    assert not boundaries(tok,tok.encode('<think>\n</think>'),PARAGRAPHS)[1]['steps']
    assert 'Step ' not in protocol.feedback_text('bad','recompute',2,PARAGRAPHS)


def test_verifier_quality_gate_requires_confirmed_independent_humans(tmp_path):
    from veriserve_research.intervention.high_low.run import verifier_audit_report
    cfg=read_json(ROOT/'stages/stage3/qwen35_4b_omni_high_low_config.json')
    cfg['bootstrap']=10
    labels=[]
    for i in range(40):
        result=dict(verdict='PASS',input_tokens=10,output_tokens=2,seconds=.1)
        atomic_json(tmp_path/'checks'/f'verifier_audit-{i}.json',dict(unique_id=str(i//2),
            input_sha256=str(i),results=[result,result]))
        labels.append(dict(input_sha256=str(i),reviewer='independent human',confirmed=False,
                           label='CORRECT' if i%2==0 else 'INCORRECT',first_error_step=1))
    path=tmp_path/'human.json'
    atomic_json(path,labels)
    absent=verifier_audit_report(cfg,tmp_path,path)
    assert absent['status']=='INSUFFICIENT_OR_FAILED' and absent['confirmed']==0
    for item in labels:item['confirmed']=True
    atomic_json(path,labels)
    failed=verifier_audit_report(cfg,tmp_path,path)
    assert failed['correct_prefixes']==failed['incorrect_prefixes']==20
    assert failed['metrics']['incorrect_pass_rate']['value']==1.
    assert failed['status']=='INSUFFICIENT_OR_FAILED'
    # All three error-rate gates pass, but UNCERTAIN must also reduce balanced accuracy.
    for i in range(40):
        verdict=('UNCERTAIN' if i//2<6 else 'PASS') if i%2==0 else ('PASS' if i//2<4 else 'FAIL')
        result=dict(verdict=verdict,input_tokens=10,output_tokens=2,seconds=.1)
        atomic_json(tmp_path/'checks'/f'verifier_audit-{i}.json',dict(unique_id=str(i//2),
            input_sha256=str(i),results=[result] if i%2 else [result,result]))
    low_balanced=verifier_audit_report(cfg,tmp_path,path)
    assert low_balanced['metrics']['balanced_accuracy']['value']==pytest.approx(.75)
    assert low_balanced['metrics']['uncertain_or_format_or_truncated_rate']['value']==pytest.approx(.15)
    assert low_balanced['repeated_prefixes']==20
    assert low_balanced['status']=='INSUFFICIENT_OR_FAILED'
    # Unknown human labels do not remove uncertain calls from the all-input denominator.
    atomic_json(path,[])
    unlabelled=verifier_audit_report(cfg,tmp_path,path)
    assert unlabelled['confirmed']==0
    assert unlabelled['metrics']['uncertain_or_format_or_truncated_rate']['value']==pytest.approx(.15)


def test_development_comparison_preserves_human_table_and_selects_by_full_cost(tmp_path,monkeypatch):
    from veriserve_research.intervention.high_low import run as high_low
    monkeypatch.setattr(high_low,'ROOT',tmp_path)
    atomic_json(tmp_path/'stage2/manifest.json',{})
    official=tmp_path/'official.json'
    atomic_json(official,{'verifier':{'selection_status':'AWAITING_DEVELOPMENT_COMPARISON'}})
    folders=[tmp_path/'modes/direct',tmp_path/'modes/thinking']
    for mode,folder in enumerate(folders):
        cfg=dict(seed=7,bootstrap=5,stage2_run='stage2',verifier={'enable_thinking':bool(mode)})
        atomic_json(folder/'review_config.json',cfg)
        atomic_json(folder/'identity.json',dict(config_hash=stable_hash(cfg),source_hashes=high_low.source_hashes(),
            stage2_manifest_sha256=digest(tmp_path/'stage2/manifest.json')))
        atomic_json(folder/'verifier_dev_ordered_ids.json',[str(i) for i in range(20)])
        table=[]
        for i in range(21):
            result=dict(verdict='UNCERTAIN' if i==20 else 'PASS' if i<10 else 'FAIL',input_tokens=10,output_tokens=2,seconds=2.-mode)
            atomic_json(folder/'checks'/f'verifier_dev-{i}.json',dict(unique_id=str(i%20),review_key=str(i),
                input_sha256=f'{mode}-{i}',prefix_ids=[i],check_type='intermediate',results=[result]))
            table.append(dict(review_key=str(i),label=None,confirmed=False,reviewer=None))
        atomic_json(folder/'verifier_dev_review_inputs.json',table)
    waiting=high_low.compare_verifiers(*folders,None,official)
    assert waiting['status']=='AWAITING_HUMAN_LABELS'
    path=tmp_path/'modes/development_human_review.json'
    labels=read_json(path)
    for i,item in enumerate(labels):
        if i<20:
            item.update(label='CORRECT' if i<10 else 'INCORRECT',confirmed=True,reviewer='human',first_error_step=1 if i>=10 else None)
    atomic_json(path,labels)
    selected=high_low.compare_verifiers(*folders,path,official)
    assert selected['selected_mode']=='thinking'
    assert selected['candidate_reports'][0]['confirmed']==20
    assert selected['candidate_reports'][0]['unconfirmed']==1
    assert read_json(path)==labels
    assert read_json(official)['verifier']['selection_status']=='SELECTED_BY_DEVELOPMENT'


@pytest.mark.parametrize('fail_at',(1,2))
def test_verifier_review_backup_failure_stops_before_next_question(tmp_path,monkeypatch,fail_at):
    from veriserve_research.intervention.high_low import run as high_low
    from veriserve_research import inference
    from veriserve_research.probe import collect
    monkeypatch.setattr(high_low,'ROOT',tmp_path)
    monkeypatch.setattr(high_low,'environment',lambda *args,**kwargs:{})
    rows=[dict(unique_id='first'),dict(unique_id='second')]
    atomic_json(tmp_path/'stage2/manifest.json',dict(config={'model':'fixture'},splits={'smoke':rows}))
    cfg=dict(stage2_run='stage2',model='fixture')
    cfg['final_response_mode']='explanation_then_final_line'
    with pytest.raises(ValueError,match='Legacy execution'):
        high_low.verifier_review(cfg,'verifier-smoke',tmp_path/'review')
    cfg.pop('final_response_mode')
    collected=[]
    def collect_row(*args,**kwargs):
        collected.append(args[3]['unique_id'])
        return dict(boundaries={'steps':[]},completion='LENGTH_TRUNCATED',generated_ids=[1]),True
    monkeypatch.setattr(collect,'collect_one',collect_row)
    monkeypatch.setattr(inference,'load_model',lambda cfg:(object(),object()))
    monkeypatch.setattr(inference,'GenerativeVerifier',lambda cfg:object())
    pushed=[]
    monkeypatch.setattr(backup,'backup_probe',lambda *args:pushed.append('references'))
    def push_review(*args):
        pushed.append('judgments')
        count=pushed.count('judgments')
        return {'status':'LOCAL_ONLY' if count==fail_at else 'PUSHED'}
    monkeypatch.setattr(high_low,'backup_high_low',push_review)
    with pytest.raises(RuntimeError,match='remote backup failed'):
        high_low.verifier_review(cfg,'verifier-smoke',tmp_path/'review')
    assert collected==([] if fail_at==1 else ['first'])
    assert pushed==['references','judgments']*fail_at


def test_generative_pilot_rejects_local_only_backup_before_requests(tmp_path,monkeypatch):
    from veriserve_research.intervention.high_low import run as high_low
    cfg={'verifier':{'kind':'generative'}}
    manifest={'config':cfg,'splits':{'pilot':[{'unique_id':'first'}]}}
    atomic_json(tmp_path/'self_check.json',{'status':'PASS','source_hashes':high_low.source_hashes()})
    monkeypatch.setattr(high_low,'backup_high_low',lambda *args:{'status':'LOCAL_ONLY'})
    monkeypatch.setattr(high_low,'freeze_selection',lambda *args:pytest.fail('Backup failure started a request'))
    with pytest.raises(RuntimeError,match='remote backup failed'):
        high_low.run_phase(cfg,tmp_path,manifest,'pilot',None,None,None,{})


def test_text_loading_unpacks_model_and_serializes_loading_info(monkeypatch):
    import torch
    import transformers
    from types import SimpleNamespace
    from veriserve_research.inference import load_model
    sample=torch.ones((1,1,2),dtype=torch.bfloat16)
    moves=[]
    model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1,hidden_size=2),
        parameters=lambda:iter([sample]),model=lambda **kwargs:SimpleNamespace(hidden_states=[sample,sample],last_hidden_state=sample))
    model.to=lambda device:(moves.append(device) or model)
    model.eval=lambda:model
    model.generation_config=SimpleNamespace(eos_token_id=98)
    tokenizer=SimpleNamespace(eos_token_id=0,get_vocab=lambda:{'<|im_end|>':99})
    info=dict(missing_keys=set(),unexpected_keys=set(),mismatched_keys=set(),error_msgs=[])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    original_tensor=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *args,**kwargs:original_tensor(*args,**{**kwargs,'device':'cpu'}))
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *args,**kwargs:SimpleNamespace(model_type='fixture'))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *args,**kwargs:(model,info))
    cfg=dict(model='fixture',enable_thinking=True,revisions={'model':'a'*40,'tokenizer':'a'*40})
    assert load_model(cfg)==(model,tokenizer) and moves==['cuda:0']
    json.dumps(model.research_loading_info)
    assert model.generation_config.eos_token_id==[98,0,99]
    moves.clear()
    info['missing_keys']={'model.layers.0.weight'}
    with pytest.raises(RuntimeError,match='Incomplete text checkpoint'):
        load_model(cfg)
    assert not moves


def test_chat_templates_explicitly_request_original_id_lists():
    from veriserve_research.inference import prompt_tokens,GenerativeVerifier
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            self.messages=messages
            assert kwargs['return_dict'] is False and kwargs['tokenize'] is True
            return [1,2,3]
    tokenizer=Tokenizer()
    assert prompt_tokens({'prompt':'Problem: {problem}','enable_thinking':True},tokenizer,'1+1')==[1,2,3]
    assert prompt_tokens({'prompt':'{problem}','enable_thinking':True,'system_prompt':'Strict format'},tokenizer,'1+1')==[1,2,3]
    assert tokenizer.messages==[{'role':'system','content':'Strict format'},{'role':'user','content':'1+1'}]
    examples=[{'role':'user','content':'2+2?'},{'role':'assistant','content':'Final answer: \\boxed{4}'}]
    prompt_tokens({'prompt':'{problem}','format_examples':examples},tokenizer,'question')
    assert tokenizer.messages==examples+[{'role':'user','content':'question'}]
    verifier=object.__new__(GenerativeVerifier)
    verifier.tokenizer=tokenizer
    assert verifier.input_ids('1+1','Step 1: 1+1=2',1,[],'intermediate')==[1,2,3]
