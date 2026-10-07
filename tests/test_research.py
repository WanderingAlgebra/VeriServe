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
