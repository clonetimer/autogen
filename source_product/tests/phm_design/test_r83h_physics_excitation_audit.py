"""R83h physics-only source QA. Every fixture is SYNTHETIC, not native."""

import copy
import csv
import hashlib
from pathlib import Path

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.r83h_physics_excitation_audit import (
    ORIGINAL_SHA, MECHANISMS, REGIMES, audit_native_train_excitation,
)


def _case_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path):
    native = root / "source"
    original = {"schema": "phm.r83e.original-native-paired-observability.v1",
                "source_role": "TRUSTED_NATIVE_TRAIN_SOURCE_PHYSICS_QA_ONLY",
                "paired_cases": 45,
                "test_final_or_validation_traces_accessed": False,
                "truth_available_to_model_inference": False,
                "model_trained_or_selected": False, "per_pair": []}
    plan = root / "frozen_plan.audit.json"
    write_json(plan, {"frozen_plan_canonical_sha256": ORIGINAL_SHA,
                      "planned_cases": 126})
    for mechanism in MECHANISMS:
        for inclination in REGIMES:
            for index in range(5):
                pair = f"train-{mechanism.lower()}-{list(REGIMES).index(inclination)*5+index:03d}"
                traces = {}
                for role in ("fault", "nominal"):
                    case = native / "cases" / (
                        f"r83d_balanced_native_rw-{pair}-{role}")
                    write_json(case / "astrograph/dataset_contract.json", {
                        "split": "train",
                        "pairing": {"pair_id": pair, "case_role": role},
                        "simulation": {"backend": "basilisk",
                                       "bsk_version": "2.11.0+satfix1"},
                        "ground_truth": {
                            "class_id": mechanism,
                            "onset_time_s": 30.5}})
                    write_json(case / "task_spec.json", {
                        "parameters": {"inclination_deg": inclination,
                                       "controller_k": 1+index},
                        "simulation": {"seed": 9000+index}})
                    output=case / "trace.csv"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    fields=("time_s", "adcs.rw.speed_rad_s_0",
                            "adcs.rw.command_torque_nm_0",
                            "adcs.control.applied_torque_nm_0",
                            "adcs.rw.effective_max_torque_nm_0")
                    with output.open("w", newline="", encoding="utf-8") as out:
                        wr = csv.DictWriter(out, fieldnames=fields)
                        wr.writeheader()
                        for t in range(61):
                            degraded=role=="fault" and t>=31
                            speed=10+t*.05+(0.2 if degraded else 0)
                            cmd=0.01
                            applied=cmd/2 if degraded else cmd
                            wr.writerow(dict(zip(fields,(t,speed,cmd,applied,.2))))
                    traces[role]=_case_sha(output)
                original["per_pair"].append({
                    "source_pair_id": pair,
                    "fault_class": mechanism,
                    "physical_wheel_index_audit_only": 0,
                    "fault_onset_s_audit_only": 30.5,
                    "max_pre_onset_abs_speed_delta_rad_s": 0.,
                    "max_post_onset_abs_speed_delta_rad_s": .2,
                    "original_fault_trace_sha256": traces["fault"],
                    "original_nominal_trace_sha256": traces["nominal"]})
    original_path=root / "original_train_pairs.json"
    write_json(original_path,original)
    return native,original_path,plan


def test_frozen_45_real_source_contract_simulated_fixture_only(tmp_path):
    native,original,plan=_fixture(tmp_path)
    # An unreadable unrelated VALIDATION file must not influence TRAIN QA.
    foreign=native / "cases/r83d_balanced_native_rw-validation-foo/trace.csv"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("CORRUPTED_AND_SHOULD_NEVER_BE_READ",encoding="utf-8")
    output=tmp_path / "excitation.json"
    a=audit_native_train_excitation(
        source_root=native, frozen_45_pair_audit=original,
        frozen_plan_audit=plan, output=output)
    assert a["schema"].endswith("native-train-physics-excitation-audit.v1")
    assert a["source_validation_test_or_final_files_read"] is False
    assert a["new_diagnostic_model_trained"] is False
    assert a["source_internal_applied_torque_used_for_PHYSICS_QA_ONLY"] is True
    assert a["original_case_traces_read"]==90
    assert a["original_source_train_independent_groups"]==45
    assert len(a["per_original_pair"])==45
    assert len(a["per_fault_regime"])==9
    for f in MECHANISMS:
        row=a["per_fault_mechanism"][f]
        assert row["independent_original_pairs"]==15
        assert row["post_max_abs_target_speed_delta_rad_s"]["median"]==pytest.approx(.2)
        assert row["nominal_post_mean_abs_command_to_rating"]["median"]==pytest.approx(.05)
        assert row["fault_post_mean_abs_command_applied_gap_nm"]["median"]==pytest.approx(.005)
    assert read_json(output)==a
    with pytest.raises(FileExistsError,match="PHYSICS_QA_ALREADY_EXISTS"):
        audit_native_train_excitation(
            source_root=native,frozen_45_pair_audit=original,
            frozen_plan_audit=plan,output=output)


def test_tampered_native_train_csv_and_protected_pair_refused(tmp_path):
    native,original,plan=_fixture(tmp_path)
    source=read_json(original)
    pair=source["per_pair"][0]["source_pair_id"]
    path=native/"cases"/f"r83d_balanced_native_rw-{pair}-fault"/"trace.csv"
    with path.open("a",encoding="utf-8") as f:
        f.write("tampered")
    with pytest.raises(ValueError,match="TRACE_HASH_CHANGED"):
        audit_native_train_excitation(source_root=native,
            frozen_45_pair_audit=original,frozen_plan_audit=plan,
            output=tmp_path/"no_tamper")
    assert not (tmp_path/"no_tamper").exists()
    native,original,plan=_fixture(tmp_path)
    source=read_json(original)
    source["per_pair"][0]["source_pair_id"]="validation-leaked-source"
    write_json(original,source)
    with pytest.raises(ValueError,match="FORBIDDEN_NONTRAIN_SOURCE_PAIR"):
        audit_native_train_excitation(source_root=native,
            frozen_45_pair_audit=original,frozen_plan_audit=plan,
            output=tmp_path/"no_validation")
