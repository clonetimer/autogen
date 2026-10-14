from __future__ import annotations
import copy
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from phm_design.data import Dataset, validate_contract, sequence_from_csv, fit_scaler, transform, reject_overlap
from phm_design.engine import fit, evaluate, predict, load_ensemble, config_check, ensemble_scores, deterministic_cpu
from phm_design.io import write_json, read_json, sha_file, run_lock, verify
from phm_design.models import Shape, Registry, default_registry


def fixture_dataset(root: Path, channels: int = 2, classes: int = 3) -> tuple[Path, Path]:
    rng = np.random.default_rng(9182)
    contract = {"device_family": "software.fixture.sequence", "task": "fixed_length_case_classification",
                "channels": [{"name": f"sensor.x{i}", "unit": "1", "role": "deployment_input", "provenance": "synthetic software fixture; not a device"} for i in range(channels)],
                "classes": [f"C{i}" for i in range(classes)], "sample_s": 1.0,
                "evidence_type": "software_synthetic", "scope": "software contract test only"}
    records = []
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        part = "test" if split == "test" else "dev"
        for label in range(classes):
            for n in range(count):
                identifier = f"{split}-{label}-{n}"
                path = root / part / f"{identifier}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                values = rng.normal(0, 0.1, (12, channels)) + label*2 + rng.normal(0, 0.02)
                with path.open("w", newline="") as stream:
                    w = csv.writer(stream)
                    w.writerow(["time_s", *[c["name"] for c in contract["channels"]]])
                    w.writerows([[i, *v] for i,v in enumerate(values)])
                records.append({"id": identifier, "group_id": identifier, "lineage_id": identifier,
                                "split": split, "label": f"C{label}", "path": path.name, "sha256": sha_file(path)})
    for part, splits in (("dev", {"train", "validation"}), ("test", {"test"})):
        write_json(root / part / "dataset.json", {"schema": "phm.sequence-dataset.v1", "dataset_id": f"fixture-{part}", "contract": contract,
            "samples": [r for r in records if r["split"] in splits], "provenance": {"source": "software_fixture"}})
    return root / "dev/dataset.json", root / "test/dataset.json"


def config() -> dict:
    return {"schema": "phm.search.v1", "window": 8, "candidates": [
            {"family": "mlp", "parameters": {"widths": [8], "representation": "raw"}},
            {"family": "cnn1d", "parameters": {"widths": [8], "kernel": 3}},
            {"family": "gru", "parameters": {"hidden": 8, "layers": 1}}],
            "seeds": [3, 5], "training": {"epochs": 4, "batch_size": 6, "learning_rate": 0.01, "weight_decay": 0.001, "patience": 2},
            "budget": {"max_parameters_per_model": 5000, "max_ensemble_parameters": 10000, "max_trials": 6, "max_trial_seconds": 30},
            "selection": "mean_seed_validation_macro_f1", "abstention": {"min_uncalibrated_score": 0.6}}


@pytest.fixture
def inputs(tmp_path):
    dev, test = fixture_dataset(tmp_path / "data")
    cfg = tmp_path / "config.json"
    write_json(cfg, config())
    return dev, test, cfg


@pytest.mark.parametrize("spec", config()["candidates"] + [{"family": "mlp", "parameters": {"widths": [8, 4], "representation": "statistics"}}])
def test_architecture_shape_backward_and_parameter_count(spec):
    r, shape = default_registry(), Shape(8, 2, 3)
    model = r.build(spec, shape, 3, 5000)
    x, y = torch.randn(4, 8, 2), torch.tensor([0, 1, 2, 0])
    out = model(x)
    assert out.shape == (4, 3)
    nn.functional.cross_entropy(out, y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == r.describe(spec, shape)["parameter_count"]


def test_initialization_seed_and_rng_restoration():
    r, shape, spec = default_registry(), Shape(8, 2, 3), config()["candidates"][0]
    before = torch.get_rng_state().clone()
    a, b = r.build(spec, shape, 3, 5000), r.build(spec, shape, 3, 5000)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(), b.parameters()))
    c = r.build(spec, shape, 5, 5000)
    assert any(not torch.equal(x,y) for x,y in zip(a.parameters(), c.parameters()))


@pytest.mark.parametrize("name", ["label.class", "truth.state", "estimate.torque", "sensor.actual_torque", "adcs.control.applied_torque_nm_0", "status.health"])
def test_truth_contract_rejected(inputs, name):
    payload = read_json(inputs[0])["contract"]
    payload["channels"][0]["name"] = name
    with pytest.raises(ValueError, match="TRUTH_INPUT_FORBIDDEN"):
        validate_contract(payload)


def test_unknown_or_duplicate_architecture():
    r = default_registry()
    with pytest.raises(ValueError, match="UNREGISTERED"):
        r.build({"family": "unknown", "parameters": {}}, Shape(8,2,3), 1, 5000)
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE"):
        c = config(); c["candidates"].append(c["candidates"][0])
        config_check(c, 2, 3, r)


@pytest.mark.parametrize("change, message", [
    (lambda c: c["seeds"].__setitem__(1, 3), "DISTINCT_SEEDS"),
    (lambda c: c["budget"].__setitem__("max_trials", 1), "TRIAL_BUDGET"),
    (lambda c: c["training"].__setitem__("learning_rate", float("nan")), "LEARNING_RATE"),
    (lambda c: c.__setitem__("selection", "test_f1"), "SELECTION"),
])
def test_invalid_search_contract(change, message):
    c = config(); change(c)
    with pytest.raises(ValueError, match=message):
        config_check(c, 2, 3, default_registry())


def test_fit_rejects_test_role_manifest(inputs, tmp_path):
    with pytest.raises(ValueError, match="DATA_ROLE_VIOLATION"):
        fit(inputs[1], inputs[2], tmp_path / "run")


def test_group_leakage_detected(inputs):
    p = read_json(inputs[0])
    a = next(s for s in p["samples"] if s["split"] == "train")
    b = next(s for s in p["samples"] if s["split"] == "validation")
    b["group_id"] = a["group_id"]
    write_json(inputs[0], p)
    with pytest.raises(ValueError, match="CROSS_SPLIT_LEAKAGE"):
        Dataset(inputs[0], allowed_splits={"train", "validation"})


def test_file_tamper_blocked(inputs):
    d = Dataset(inputs[0], allowed_splits={"train", "validation"})
    p = d.root / d.samples[0]["path"]
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError, match="DATA_FILE_CHANGED"):
        d.verify_files()


def test_path_escape(inputs):
    p = read_json(inputs[0]); p["samples"][0]["path"] = "../../outside.csv"; write_json(inputs[0], p)
    with pytest.raises(ValueError, match="PATH_ESCAPES_ROOT"):
        Dataset(inputs[0], allowed_splits={"train", "validation"})


@pytest.mark.parametrize("failure", ["time", "nan", "missing", "extra", "short"])
def test_bad_input_sequences(inputs, tmp_path, failure):
    d = Dataset(inputs[0], allowed_splits={"train", "validation"})
    rows = list(csv.reader((d.root/d.samples[0]["path"]).open()))
    if failure == "time": rows[2][0] = rows[1][0]
    if failure == "nan": rows[1][1] = "nan"
    if failure == "missing": rows[0][1] = "wrong"
    if failure == "extra":
        rows = [r+["label.truth" if i == 0 else "0"] for i,r in enumerate(rows)]
    if failure == "short": rows = rows[:3]
    p = tmp_path / "bad.csv"
    with p.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError):
        sequence_from_csv(p, d.contract, 8)


def test_standardization_fitted_only_on_train(inputs):
    d = Dataset(inputs[0], allowed_splits={"train", "validation"})
    x = d.load("train", 8).x
    s = fit_scaler(x)
    assert s["fit_role"] == "train_only"
    np.testing.assert_allclose(transform(x,s).mean((0,1)), np.zeros(2), atol=1e-6)
    transform(x+1000, s)
    assert s == fit_scaler(x)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("design-complete")
    dev, holdout = fixture_dataset(root / "data")
    cfg = root / "config.json"; write_json(cfg, config())
    report = fit(dev, cfg, root / "run")
    return root, dev, holdout, cfg, report


def test_real_training_and_separate_neural_winner(completed):
    _,_,_,_,report = completed
    assert report["status"] == "COMPLETED"
    assert len(report["candidate_results"]) == 3
    assert all(len(c["trials"]) == 2 for c in report["candidate_results"])
    assert report["best_neural_candidate"]
    assert not report["neural_output_replaced_by_baseline"]
    assert not report["test_used"] and not report["production_release_qualified"]
    assert all(t["history"] for c in report["candidate_results"] for t in c["trials"])


def test_prediction_restore_units_and_no_labels(completed):
    root, dev, _, _, _ = completed
    d = Dataset(dev, allowed_splits={"train", "validation"})
    units = {c["name"]: c["unit"] for c in d.contract["channels"]}
    p = d.root / d.samples[0]["path"]
    a, b = predict(root/"run", p, units), predict(root/"run", p, units)
    assert a == b
    assert not a["probability_calibrated"]
    units[next(iter(units))] = "wrong"
    with pytest.raises(ValueError, match="UNITS_MISMATCH"):
        predict(root/"run", p, units)


def test_holdout_once_and_idempotent_report(completed):
    root, _, test, _, _ = completed
    a = evaluate(root/"run", test, root/"evaluation")
    b = evaluate(root/"run", test, root/"unused-destination")
    assert a == b
    assert a["open_count_in_this_run"] == 1
    assert a["counts"]["test"]["groups"] == 6
    assert not a["test_used_for_selection"]
    assert not a["production_release_qualified"]


def test_completed_run_resume_reuses_verified_trials(completed):
    root,dev,_,cfg,report = completed
    a = fit(dev, cfg, root/"run", resume=True)
    assert a == report


def test_changed_search_contract_rejects_resume(completed, tmp_path):
    root,dev,_,cfg,_ = completed
    changed = read_json(cfg); changed["training"]["epochs"] += 1
    changed_path = tmp_path/"changed.json"; write_json(changed_path, changed)
    with pytest.raises(ValueError, match="RESUME_INPUT_OR_IMPLEMENTATION_CHANGED"):
        fit(dev, changed_path, root/"run", resume=True)


def test_weight_tamper_rejected(completed, tmp_path):
    root,_,_,_,_ = completed
    copied = tmp_path/"run"; shutil.copytree(root/"run", copied)
    bundle = read_json(copied/"model_bundle.json")
    with (copied/bundle["weights"][0]).open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="INTEGRITY_FAILURE"):
        load_ensemble(copied)


def test_trial_boundary_interruption_and_resume(inputs, tmp_path):
    dev,_,cfg = inputs
    def interrupt(record):
        raise KeyboardInterrupt("simulated process interruption after commit")
    output = tmp_path/"interrupted"
    with pytest.raises(KeyboardInterrupt):
        fit(dev,cfg,output,on_trial_complete=interrupt)
    checkpoints = list(output.glob("trials/*/*/model.pt"))
    assert len(checkpoints) == 1
    before = sha_file(checkpoints[0]); before_mtime = checkpoints[0].stat().st_mtime_ns
    report = fit(dev,cfg,output,resume=True)
    assert report["status"] == "COMPLETED"
    assert before == sha_file(checkpoints[0])
    assert before_mtime == checkpoints[0].stat().st_mtime_ns


def test_all_budget_rejections_do_not_substitute_baseline(inputs, tmp_path):
    dev,_,cfg = inputs
    c = read_json(cfg); c["budget"]["max_parameters_per_model"] = 1; write_json(cfg,c)
    report = fit(dev,cfg,tmp_path/"run")
    assert report["status"] == "NO_FEASIBLE_NEURAL_CANDIDATE"
    assert report["best_neural_candidate"] is None
    assert not (tmp_path/"run/model_bundle.json").exists()
    assert report["best_non_neural_baseline"]


def test_fit_does_not_need_or_open_holdout(inputs, tmp_path):
    dev,test,cfg = inputs
    shutil.rmtree(test.parent)
    report = fit(dev,cfg,tmp_path/"run")
    assert report["status"] == "COMPLETED" and not report["test_used"]


def test_changed_shape_second_contract(tmp_path):
    dev,test = fixture_dataset(tmp_path/"second", channels=4, classes=2)
    cfg = tmp_path/"config.json"; write_json(cfg, config())
    report = fit(dev,cfg,tmp_path/"run")
    ev = evaluate(tmp_path/"run",test,tmp_path/"eval")
    assert report["status"] == "COMPLETED"
    bundle = read_json(tmp_path/"run/model_bundle.json")
    assert bundle["shape"] == {"window":8,"channels":4,"classes":2}
    assert ev["claim_status"] == "SOFTWARE_OR_UNREVIEWED_EVIDENCE"


def custom_estimate(shape, p):
    return shape.window*shape.channels*shape.classes+shape.classes


def custom_build(shape, p):
    return nn.Sequential(nn.Flatten(1), nn.Linear(shape.window*shape.channels, shape.classes))


def test_external_family_registers_without_engine_change(inputs, tmp_path):
    r = Registry(); r.register("custom_linear_neural", "test1", custom_build, custom_estimate)
    dev,_,cfg = inputs
    c = config(); c["candidates"] = [{"family":"custom_linear_neural","parameters":{}}]
    write_json(cfg,c)
    report = fit(dev,cfg,tmp_path/"run",registry=r)
    assert report["status"] == "COMPLETED"
    bundle, models = load_ensemble(tmp_path/"run",registry=r)
    assert len(models) == 2


def test_concurrent_write_is_locked(tmp_path):
    with run_lock(tmp_path/"run"):
        with pytest.raises(RuntimeError, match="RUN_ALREADY_LOCKED"):
            with run_lock(tmp_path/"run"):
                pass


def test_search_budget_accepts_optional_generated_graph_mac_ceiling():
    c=config();c["budget"]["max_multiply_accumulate_per_case"]=60_000_000
    records=config_check(c,2,3,default_registry())
    assert len(records)==len(c["candidates"])


def test_sealed_failure_is_terminal_on_resume(monkeypatch, inputs, tmp_path):
    import phm_design.engine as eng
    c=config();c["candidates"]=[c["candidates"][0]];c["seeds"]=[3,5];c["budget"]["max_trials"]=2
    cfg=tmp_path/"failure-config.json";write_json(cfg,c)
    calls=[]
    original=eng.train_trial
    def forced_failure(registry,spec,shape,seed,config,x,y,vx,vy,destination,classes):
        calls.append(seed)
        if seed==3:
            raise TimeoutError("TRIAL_TIME_BUDGET_EXCEEDED")
        return original(registry,spec,shape,seed,config,x,y,vx,vy,destination,classes)
    monkeypatch.setattr(eng,"train_trial",forced_failure)
    run=tmp_path/"failure-run"
    first=fit(inputs[0],cfg,run)
    assert first["candidate_results"][0]["status"]=="INCOMPLETE_SEEDS"
    assert (run/"trials"/first["candidate_results"][0]["id"]/"seed_3"/"manifest.json").exists()
    assert calls.count(3)==1
    # A sealed terminal failure is replayed, not retrained.
    second=fit(inputs[0],cfg,run,resume=True)
    assert second==first and calls.count(3)==1
