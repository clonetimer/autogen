"""R82.3 operator launcher contract tests — no actual Basilisk is executed.

The test runtime and campaign producer are deliberately mocked; validation
only covers safe refusal, source identity and run-receipt plumbing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.native_campaign_execution import (
    execute_native_rw_campaign, inspect_native_campaign_prerequisites,
)


def fixture_source(tmp_path, monkeypatch):
    source = tmp_path / "producer"
    (source / "scripts").mkdir(parents=True)
    (source / "src/sat_sim/bsk_engine").mkdir(parents=True)
    for relative in (
        "scripts/generate_astrograph_rw_fault_dataset.py",
        "src/sat_sim/bsk_engine/unified_native.py",
        "src/sat_sim/bsk_engine/project_native_modules.py",
    ):
        (source / relative).write_text("# TEST STUB, NO PHYSICS\n")
    config = source / "config.json"
    write_json(config, {
        "schema_version": "sat-sim.astrograph-fault-dataset-factory.v3",
        "campaign_id": "rw_intake_test", "base_seed": 42,
        "counts": {"train": 3, "validation": 1, "test": 1},
        "fault_ids": [
            "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM",
            "ADCS_RW_TORQUE_AUTHORITY_LOSS"],
        "duration_s": 60., "sample_s": 1., "solver_step_s": .2,
        "capability_id": "subsystem.adcs_unified_native.v1",
        "simulation_backend": "basilisk", "allow_test_proxy": False,
        "output_root": "fake-campaign",
    })
    wheel = tmp_path / "simIncludeRW.py"
    wheel.write_text("# MOCK WHEEL SOURCE, NO BASILISK\n")
    pkg = types.ModuleType("Basilisk")
    pkg.__path__ = []
    utilities = types.ModuleType("Basilisk.utilities")
    utilities.__path__ = []
    sim = types.ModuleType("Basilisk.utilities.simIncludeRW")
    sim.__file__ = str(wheel)
    utilities.simIncludeRW = sim
    pkg.utilities = utilities
    for name, module in (
        ("Basilisk", pkg),
        ("Basilisk.utilities", utilities),
        ("Basilisk.utilities.simIncludeRW", sim),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "phm_design.industrial.native_campaign_execution.importlib.metadata.version",
        lambda name: "2.11.0+satfix1" if name == "bsk" else "0")
    return source, config


def mock_git(monkeypatch, *, dirty=False, producer_exit=1):
    def fake_run(argv, *, cwd=None, text=None, capture_output=None,
                 check=None, timeout=None, stdout=None, stderr=None):
        if argv[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
        if argv[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, " M tracked.py" if dirty else "", "")
        if argv and argv[0] == sys.executable:
            if stdout is not None:
                stdout.write("MOCK SUBPROCESS; NO BSK RUN\n")
            return subprocess.CompletedProcess(argv, producer_exit, "", "")
        raise AssertionError("unexpected process call")
    monkeypatch.setattr(
        "phm_design.industrial.native_campaign_execution.subprocess.run",
        fake_run)


def test_native_launcher_requires_approved_runtime_and_source_revision(tmp_path, monkeypatch):
    root, config = fixture_source(tmp_path, monkeypatch)
    mock_git(monkeypatch)
    receipt = inspect_native_campaign_prerequisites(root, config)
    assert receipt["bsk_version"] == "2.11.0+satfix1"
    assert receipt["source_revision"] == "a" * 40
    assert receipt["native_execution_performed"] is False
    assert receipt["proxy_fallback_allowed"] is False
    assert receipt["independent_qualification"] is False


@pytest.mark.parametrize("field,value", [
    ("allow_test_proxy", True),
    ("simulation_backend", "python"),
    ("capability_id", "proxy.rw.mock"),
])
def test_native_launcher_rejects_proxy_contract_before_launch(
        tmp_path, monkeypatch, field, value):
    source, config = fixture_source(tmp_path, monkeypatch)
    raw = read_json(config)
    raw[field] = value
    write_json(config, raw)
    mock_git(monkeypatch)
    with pytest.raises(ValueError, match="R82_NATIVE_ONLY_FACTORY_CONFIG_REQUIRED"):
        inspect_native_campaign_prerequisites(source, config)


def test_native_launcher_rejects_dirty_source_or_unapproved_version(tmp_path, monkeypatch):
    source, config = fixture_source(tmp_path, monkeypatch)
    mock_git(monkeypatch, dirty=True)
    with pytest.raises(RuntimeError, match="R82_NATIVE_PRODUCER_CHECKOUT_NOT_CLEAN"):
        inspect_native_campaign_prerequisites(source, config)
    mock_git(monkeypatch)
    monkeypatch.setattr(
        "phm_design.industrial.native_campaign_execution.importlib.metadata.version",
        lambda name: "2.10.2")
    with pytest.raises(RuntimeError, match="R82_BASILISK_RUNTIME_VERSION_NOT_APPROVED"):
        inspect_native_campaign_prerequisites(source, config)


def test_failed_native_producer_writes_explicit_refusal_no_model(tmp_path, monkeypatch):
    source, config = fixture_source(tmp_path, monkeypatch)
    mock_git(monkeypatch, producer_exit=4)
    logs = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="R82_NATIVE_CAMPAIGN_PRODUCER_FAILED"):
        execute_native_rw_campaign(
            source_checkout=source, config=config,
            output=tmp_path / "campaign", evidence_dir=logs)
    receipt = read_json(logs / "execution_receipt.json")
    assert receipt["status"] == "REFUSED"
    assert receipt["native_execution_performed"] is False
    assert receipt["producer_exit_code"] == 4
    assert receipt["independent_qualification"] is False
    assert (logs / "producer.log").exists()


def test_source_checkout_and_artifact_paths_must_be_isolated(tmp_path, monkeypatch):
    source, config = fixture_source(tmp_path, monkeypatch)
    mock_git(monkeypatch)
    with pytest.raises(ValueError, match="R82_NATIVE_EXECUTION_OUTPUT_PATH_ISOLATION"):
        execute_native_rw_campaign(
            source_checkout=source, config=config,
            output=source / "campaign", evidence_dir=tmp_path / "other")


def test_no_success_if_producer_claims_pass_without_index(tmp_path, monkeypatch):
    source, config = fixture_source(tmp_path, monkeypatch)
    mock_git(monkeypatch, producer_exit=0)
    with pytest.raises(RuntimeError, match="R82_NATIVE_CAMPAIGN_INDEX_MISSING"):
        execute_native_rw_campaign(
            source_checkout=source, config=config,
            output=tmp_path / "campaign", evidence_dir=tmp_path / "evidence")
    receipt = read_json(tmp_path / "evidence" / "execution_receipt.json")
    assert receipt["status"] == "REFUSED"
    assert receipt["native_execution_performed"] is False
