"""R92 cross-domain product boundary and evidence-claim admission gate.

This gate is intentionally independent of numerical libraries, network access,
the research PR stack and model weights. It prevents product/CLI claims from
being promoted merely because synthetic CI passed or measured inputs loaded.
It does NOT independently attest truthfulness of registry maintainers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

SCHEMA = "phm.autogen.scope.v1"
DELIVERY = frozenset({"stable", "draft", "planned"})
EVIDENCE = frozenset({
    "none", "synthetic_ci", "real_measurement_ingested",
    "independent_model_validated", "limited_stable_offline",
    "industrial_production_approved",
})
CLAIMS = frozenset({
    "code_present", "synthetic_ci_passed", "real_measured_signal_ingested",
    "limited_simulation_offline_qualified",
    "independent_cross_device_benefit",
    "industrial_production_ready",
    "online_safety_approved",
})
REQUIRED = frozenset({"id", "delivery", "evidence", "note"})


class ScopeClaimRejected(ValueError):
    """A claim or register is unsupported; never coerce into a PASS."""


def _reject(code):
    raise ScopeClaimRejected(code)


def _unique_pairs(pairs):
    doc = {}
    for k, v in pairs:
        if k in doc:
            _reject("R92_DUPLICATE_JSON_PROPERTY:" + str(k))
        doc[k] = v
    return doc


def read_scope(path):
    """Read and validate the exact audited catalog; no arbitrary imports."""
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) > 1024 * 1024:
        _reject("R92_REGISTER_TOO_LARGE")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScopeClaimRejected("R92_INVALID_REGISTER_JSON") from exc
    if not isinstance(obj, Mapping) or obj.get("schema") != SCHEMA:
        _reject("R92_UNRECOGNIZED_REGISTER_SCHEMA")
    if set(obj) != {
        "schema", "as_of", "stable_version", "stable_main_sha",
        "levels", "note", "capabilities", "forbidden_general_claims",
    }:
        _reject("R92_UNDECLARED_REGISTER_KEYS")
    if (obj["stable_version"] != "0.7.0" or
            not isinstance(obj["stable_main_sha"], str) or
            len(obj["stable_main_sha"]) != 40 or
            any(c not in "0123456789abcdef" for c in obj["stable_main_sha"])):
        _reject("R92_UNVERIFIED_STABLE_MAIN_REFERENCE")
    if (not isinstance(obj["capabilities"], list) or
            not obj["capabilities"]):
        _reject("R92_EMPTY_CAPABILITY_REGISTER")
    if (not isinstance(obj["levels"], list) or
            set(obj["levels"]) != EVIDENCE or
            len(obj["levels"]) != len(EVIDENCE)):
        _reject("R92_INVALID_EVIDENCE_VOCABULARY")
    if (not isinstance(obj["forbidden_general_claims"], list) or
            len(obj["forbidden_general_claims"]) != len(set(obj["forbidden_general_claims"]))):
        _reject("R92_INVALID_FORBIDDEN_CLAIMS")
    by_id = {}
    for item in obj["capabilities"]:
        if not isinstance(item, Mapping) or set(item) != REQUIRED:
            _reject("R92_INVALID_CAPABILITY_ENTRY")
        ident = item["id"]
        if (not isinstance(ident, str) or
                not ident or not ident.replace("_", "").isalnum() or
                ident in by_id):
            _reject("R92_DUPLICATE_OR_INVALID_CAPABILITY_ID")
        if (item["delivery"] not in DELIVERY or
                item["evidence"] not in EVIDENCE or
                not isinstance(item["note"], str) or not item["note"].strip()):
            _reject("R92_INVALID_CAPABILITY_SCOPE")
        if item["delivery"] == "planned" and item["evidence"] != "none":
            _reject("R92_UNBUILT_FEATURE_HAS_EVIDENCE")
        if item["delivery"] == "stable" and (
                ident != "r60_sat_offline" or
                item["evidence"] != "limited_stable_offline"):
            _reject("R92_STABLE_SCOPE_EXCEEDS_R60")
        if item["delivery"] == "draft" and item["evidence"] in (
                "limited_stable_offline", "industrial_production_approved"):
            _reject("R92_DRAFT_CANNOT_INHERIT_STABLE_QUALIFICATION")
        by_id[ident] = dict(item)
    if ("r60_sat_offline" not in by_id or
            by_id["r60_sat_offline"]["delivery"] != "stable" or
            "r91_cwru" not in by_id or
            by_id["r91_cwru"]["evidence"] != "real_measurement_ingested"):
        _reject("R92_REQUIRED_EVIDENCE_ANCHOR_MISSING")
    return obj, hashlib.sha256(raw).hexdigest()


def assess_claim(catalog, *, capability_id, claim):
    """Fail closed for cross-domain efficacy / real-time / promotion claims."""
    if claim not in CLAIMS:
        _reject("R92_UNKNOWN_OR_UNAPPROVED_CLAIM_TYPE")
    matches = [x for x in catalog["capabilities"] if x["id"] == capability_id]
    if not matches:
        _reject("R92_UNKNOWN_CAPABILITY")
    cap = matches[0]
    allowed = (
        claim == "code_present" and cap["delivery"] in {"stable", "draft"}
        or claim == "synthetic_ci_passed" and
        cap["delivery"] == "draft" and
        cap["evidence"] in {"synthetic_ci", "real_measurement_ingested"}
        or claim == "real_measured_signal_ingested" and
        cap["delivery"] == "draft" and
        cap["evidence"] == "real_measurement_ingested"
        or claim == "limited_simulation_offline_qualified" and
        cap["id"] == "r60_sat_offline" and cap["delivery"] == "stable"
        and cap["evidence"] == "limited_stable_offline"
    )
    return {
        "schema": "phm.autogen.scope-claim-decision.v1",
        "capability_id": capability_id, "requested_claim": claim,
        "admitted": bool(allowed),
        "evidence_level": cap["evidence"],
        "delivery": cap["delivery"],
        "scope": cap["note"],
        "industrial_production_qualified": False,
        "independent_cross_device_generalization_proven": False,
        "reason": "R92_SCOPE_DOCUMENTED_ONLY" if allowed
                  else "R92_CLAIM_EXCEEDS_AVAILABLE_EVIDENCE",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="List autogen capability boundary or inspect one claim.")
    parser.add_argument("--register", required=True,
                        help="R92 capability register JSON, kept in repo.")
    parser.add_argument("--capability")
    parser.add_argument("--claim")
    args = parser.parse_args(argv)
    if bool(args.capability) != bool(args.claim):
        parser.error("--capability and --claim must be supplied together")
    catalog, sha = read_scope(args.register)
    if args.capability:
        result = assess_claim(catalog,
                              capability_id=args.capability, claim=args.claim)
        result["capability_register_sha256"] = sha
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["admitted"] else 2
    print(json.dumps({
        "schema": SCHEMA, "register_sha256": sha,
        "stable_main_sha": catalog["stable_main_sha"],
        "capabilities": catalog["capabilities"],
    }, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
