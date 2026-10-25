"""R92 product guard: no synthetic score or Draft code may claim deployment."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from phm_design.strategy.scope_gate import (
    CLAIMS, ScopeClaimRejected, assess_claim, read_scope, main)

REGISTRY = (Path(__file__).resolve().parents[2] /
            "configs/product/r92_scope.json")


def test_stable_vs_research_vs_planned_scope_is_real_and_separate():
    catalog, digest = read_scope(REGISTRY)
    assert len(digest) == 64
    assert catalog["stable_version"] == "0.7.0"
    ids = {c["id"] for c in catalog["capabilities"]}
    assert len(ids) >= 12
    assert {"r60_sat_offline", "r91_cwru",
            "cross_domain_qualification", "industrial_online_ot"} <= ids
    caps = {c["id"]: c for c in catalog["capabilities"]}
    assert caps["r60_sat_offline"]["delivery"] == "stable"
    assert caps["r91_cwru"]["delivery"] == "draft"
    assert caps["cross_domain_qualification"]["delivery"] == "planned"
    assert caps["industrial_online_ot"]["evidence"] == "none"


def test_only_specific_offline_sat_stable_claim_is_admitted():
    catalog, _ = read_scope(REGISTRY)
    assert assess_claim(catalog, capability_id="r60_sat_offline",
        claim="limited_simulation_offline_qualified")["admitted"]
    assert not assess_claim(catalog, capability_id="r91_cwru",
        claim="limited_simulation_offline_qualified")["admitted"]
    assert not assess_claim(catalog, capability_id="r80_taskspec",
        claim="limited_simulation_offline_qualified")["admitted"]


def test_real_cwru_ingestion_never_means_model_accuracy_or_production():
    catalog, _ = read_scope(REGISTRY)
    assert assess_claim(catalog, capability_id="r91_cwru",
        claim="real_measured_signal_ingested")["admitted"]
    for claim in ("independent_cross_device_benefit",
                  "industrial_production_ready", "online_safety_approved"):
        result = assess_claim(catalog, capability_id="r91_cwru", claim=claim)
        assert not result["admitted"]
        assert result["reason"] == "R92_CLAIM_EXCEEDS_AVAILABLE_EVIDENCE"
    assert not assess_claim(catalog, capability_id="r90_paderborn",
        claim="real_measured_signal_ingested")["admitted"]


def test_synthetic_ci_and_fuzzy_implemented_are_not_effectiveness_evidence():
    catalog, _ = read_scope(REGISTRY)
    assert assess_claim(catalog, capability_id="r86_r87_fuzzy",
        claim="synthetic_ci_passed")["admitted"]
    assert not assess_claim(catalog, capability_id="r86_r87_fuzzy",
        claim="independent_cross_device_benefit")["admitted"]
    for cap in ("cross_domain_qualification", "generic_rul",
                "industrial_online_ot", "operator_workspace"):
        assert not assess_claim(catalog, capability_id=cap,
            claim="code_present")["admitted"]


def test_unknown_claims_and_capabilities_fail_closed():
    catalog, _ = read_scope(REGISTRY)
    with pytest.raises(ScopeClaimRejected, match="R92_UNKNOWN_CAPABILITY"):
        assess_claim(catalog, capability_id="my_magic_device", claim="code_present")
    with pytest.raises(ScopeClaimRejected, match="R92_UNKNOWN_OR_UNAPPROVED_CLAIM_TYPE"):
        assess_claim(catalog, capability_id="r60_sat_offline",
                     claim="universal_industrial_qualification")


@pytest.mark.parametrize("mutation,reason", [
    (lambda x: x["capabilities"].append(copy.deepcopy(x["capabilities"][0])),
     "R92_DUPLICATE_OR_INVALID_CAPABILITY_ID"),
    (lambda x: x["capabilities"].__setitem__(1, dict(
        x["capabilities"][1], delivery="stable")), "R92_STABLE_SCOPE_EXCEEDS_R60"),
    (lambda x: x["capabilities"].__setitem__(-1, dict(
        x["capabilities"][-1], evidence="real_measurement_ingested")),
     "R92_UNBUILT_FEATURE_HAS_EVIDENCE"),
    (lambda x: x["capabilities"].__setitem__(1, dict(
        x["capabilities"][1], evidence="limited_stable_offline")),
     "R92_DRAFT_CANNOT_INHERIT_STABLE_QUALIFICATION"),
])
def test_untrusted_registry_can_not_expand_main_or_unbuilt_scope(
        tmp_path, mutation, reason):
    doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutation(doc)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ScopeClaimRejected, match=reason):
        read_scope(path)


def test_duplicate_json_keys_refused(tmp_path):
    raw = REGISTRY.read_text(encoding="utf-8")
    # One JSON object, with an exact duplicated schema key.
    raw = raw.replace(
        '"schema": "phm.autogen.scope.v1",',
        '"schema": "phm.autogen.scope.v1", "schema": "phm.autogen.scope.v1",', 1)
    path = tmp_path / "dup.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ScopeClaimRejected, match="R92_DUPLICATE_JSON_PROPERTY"):
        read_scope(path)


def test_operator_cli_lists_capabilities_and_returns_nonzero_for_unproved_claim(
        capsys):
    assert main(["--register", str(REGISTRY)]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert len(obj["capabilities"]) >= 12
    assert main(["--register", str(REGISTRY), "--capability",
                 "r91_cwru", "--claim", "industrial_production_ready"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["admitted"] is False
    assert main(["--register", str(REGISTRY), "--capability",
                 "r91_cwru", "--claim", "real_measured_signal_ingested"]) == 0
    admission = json.loads(capsys.readouterr().out)
    assert admission["admitted"] is True
    assert len(admission["capability_register_sha256"]) == 64
