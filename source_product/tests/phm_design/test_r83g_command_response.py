"""R83g: causal command-response features and strict source equivalence.

All fixtures in this file are SYNTHETIC and must never be labeled a
Basilisk diagnostic-performance experiment.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from phm_design.io import read_json
from phm_design.industrial.design_engine import _observations_digest, _sha
from phm_design.industrial.taskspec import TaskSpec, TaskSpecRejected
from phm_design.industrial import r83g_command_response as g


def _spec():
    return read_json(
        Path(__file__).resolve().parents[2] /
        "examples/industrial/satellite_rw_native_r83_matched_train.json")


def _one():
    return [
        {"time_s": 0., **{n: 10.0 + i for i, n in enumerate(g.SOURCE_SPEED)},
         **{n: 0.1 + i*.1 for i,n in enumerate(g.SOURCE_COMMAND)}},
        {"time_s": 1., **{n: 12. + i for i,n in enumerate(g.SOURCE_SPEED)},
         **{n: 0.2 + i*.1 for i,n in enumerate(g.SOURCE_COMMAND)}},
        {"time_s": 2., **{n: 15. + i for i,n in enumerate(g.SOURCE_SPEED)},
         **{n: 0.3 + i*.1 for i,n in enumerate(g.SOURCE_COMMAND)}},
    ]


def _sixty():
    spec=_spec()
    labels=spec["task"]["labels"]
    out=[]
    for env in range(3):
        for cls in labels:
            for index in range(5):
                measurements = []
                shift = float(1000*env+100*labels.index(cls)+index)
                for t in range(61):
                    row = {"time_s":float(t)}
                    for i, name in enumerate(g.SOURCE_SPEED):
                        row[name] = 10.0 + shift*0.01+i + 0.04*t+0.002*t*t
                    for i, name in enumerate(g.SOURCE_COMMAND):
                        row[name] = 0.01*(i+1)+shift*.0001+.0005*t
                    measurements.append(row)
                out.append({"metadata":{
                    "role":"TRAIN","label":cls,
                    "rw_run_lineage_id":f"synthetic-{env}-{labels[1] if cls==labels[0] else cls}-{index}",
                    "operating_regime":f"regime_{env}"},
                    "measurements":measurements})
    return out


def _synthetic_receipt(monkeypatch, rows):
    raw = _spec()
    meta = _sha([r["metadata"] for r in rows])
    obs = _observations_digest([
        np.asarray([[x["time_s"], *[x[n] for n in g.SOURCE_ORDER]]
                    for x in r["measurements"]], dtype=np.float64)
        for r in rows])
    monkeypatch.setattr(g,"NATIVE_TRAIN_METADATA_SHA",meta)
    monkeypatch.setattr(g,"NATIVE_TRAIN_OBSERVATIONS_SHA",obs)
    return {
        "schema":"phm.r83.matched-source-train-study.v1",
        "source_train_test_or_final_accessed":False,
        "independent_qualification":False,
        "source_task_spec_sha256":TaskSpec.from_dict(raw).sha256,
        "source_train_manifest_sha256":meta,
        "source_train_observations_sha256":obs,
    }


def test_causal_derivatives_lags_and_zero_padding_have_exact_equal_shape():
    spec=_spec()
    source=_one()
    original=deepcopy(source)
    expanded, raw=g.make_matched_representation(
        spec,[{"metadata":{"role":"TRAIN"},"measurements":source}],
        "zero_padded_raw")
    _,phys=g.make_matched_representation(
        spec,[{"metadata":{"role":"TRAIN"},"measurements":source}],
        "causal_command_response")
    assert source == original
    assert len(expanded["observations"]["channels"]) == 18
    assert [c["name"] for c in expanded["observations"]["channels"][:6]] == list(g.SOURCE_ORDER)
    assert list(raw[0]["measurements"][0]) == list(phys[0]["measurements"][0])
    assert all(row[name]==0.0 for row in raw[0]["measurements"]
               for name in g.DERIVED_NAMES)
    x=phys[0]["measurements"]
    assert x[0]["r83g.rw.speed_slope_rad_s2_0"] == 0.0
    assert x[0]["r83g.rw.command_impulse_nms_0"] == 0.0
    assert x[1]["r83g.rw.speed_slope_rad_s2_0"] == pytest.approx(2.0)
    assert x[1]["r83g.rw.command_impulse_nms_0"] == pytest.approx(0.1)
    assert x[2]["r83g.rw.command_impulse_nms_0"] == pytest.approx(0.3)
    assert x[2]["r83g.rw.speed_slope_rad_s2_0"] == pytest.approx(3.0)
    assert x[2]["r83g.rw.speed_delta_rad_s_0"] == pytest.approx(5.0)
    assert x[2]["r83g.rw.command_speed_power_w_0"] == pytest.approx(.2*12)
    assert list(x[2]) == list(raw[0]["measurements"][2])
    altered=deepcopy(source)
    altered[2][g.SOURCE_SPEED[0]]=1000000.
    earlier=g._features(altered,mode="causal_command_response",time_field="time_s")[1]
    assert earlier == x[1], "Future measurements may NEVER affect prior features."


def test_fault_truth_and_protected_roles_cannot_enter_exploratory_model():
    spec=_spec()
    rows=_sixty()
    _, output=g.make_matched_representation(spec,rows,"causal_command_response")
    assert len(output)==60
    assert len({r["metadata"]["rw_run_lineage_id"] for r in output})==45
    assert all(len(rec["measurements"])==61 for rec in output)
    assert all(len(rec["measurements"][0]) == 19 for rec in output)
    assert all(set(rec["measurements"][0])=={"time_s",*g.SOURCE_ORDER,*g.DERIVED_NAMES}
               for rec in output)
    bad=deepcopy(rows)
    bad[0]["measurements"][0]["fault_onset_s"]=14.
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED"):
        g.make_matched_representation(spec,bad,"causal_command_response")
    bad=deepcopy(rows)
    bad[0]["metadata"]["role"]="VALIDATION"
    with pytest.raises(TaskSpecRejected, match="VALIDATION_OR_PROTECTED"):
        g._source_integrity(g._spec(spec), bad, _synthetic_receipt(
            pytest.MonkeyPatch(),rows))


def test_same_train_source_same_group_folds_and_equal_real_budget_contract(monkeypatch):
    spec=_spec()
    rows=_sixty()
    frozen=_synthetic_receipt(monkeypatch,rows)
    submitted=[]
    def mock_train(task_spec,records):
        submitted.append((task_spec,records))
        return {
            "source_train_manifest_sha256":_sha([r["metadata"] for r in records]),
            "source_environment_count":3,"candidate_budget_per_arm":3,
            "fit_budget_per_arm":10,
            "max_wall_seconds_per_arm":1800,
            "acceptance_policy":{"accuracy_min":.9},
            "arms":{a:{"source_train_candidate_count":3} for a in
                    ("fixed_portfolio","static_topology","self_organizing")},
            "fairness_gate":{"source_train_comparison_allowed":True},
        }
    monkeypatch.setattr(g,"run_matched_source_train_study",mock_train)
    study=g.run_train_only_representations(spec,rows,frozen)
    assert len(submitted)==2
    assert study["fairness_gate"]["comparison_allowed_train_research_only"]
    assert study["frozen_r83f_source"]["source_train_records"] == 60
    assert study["frozen_r83f_source"]["source_train_physical_groups"] == 45
    assert study["frozen_r83f_source"]["paired_nominal_fault_groups"] == 15
    assert not study["fairness_gate"]["performance_claim_allowed"]
    assert not study["validation_test_or_final_telemetry_read"]
    first,second=submitted
    assert first[0] == second[0]
    assert len(first[0]["observations"]["channels"]) == 18
    assert [rec["metadata"] for rec in first[1]] == [
        rec["metadata"] for rec in second[1]]
    for a,b in zip(first[1],second[1]):
        for ra,rb in zip(a["measurements"],b["measurements"]):
            assert all(ra[key] == rb[key] for key in ("time_s",*g.SOURCE_ORDER))
    copied=deepcopy(rows)
    copied[0]["measurements"][-1][g.SOURCE_SPEED[0]]+=1.
    with pytest.raises(TaskSpecRejected, match="TELEMETRY_CHANGED"):
        g.run_train_only_representations(spec,copied,frozen)
    copied=deepcopy(rows)
    copied[0]["metadata"]["rw_run_lineage_id"]="tampered"
    with pytest.raises(TaskSpecRejected,match="GROUPS_OR_LABELS_CHANGED"):
        g.run_train_only_representations(spec,copied,frozen)


def test_old_six_channel_contract_cannot_silently_accept_truth_or_unknown_basis():
    spec=_spec()
    old=deepcopy(spec)
    old["observations"]["channels"][0]["name"]="truth_fault_id"
    with pytest.raises(TaskSpecRejected):
        g.make_matched_representation(old,_sixty(),"causal_command_response")
    with pytest.raises(ValueError,match="UNKNOWN_PREREGISTERED_REPRESENTATION"):
        g.make_matched_representation(spec,_sixty(),"posthoc_best")
