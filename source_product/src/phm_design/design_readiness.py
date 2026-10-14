"""Evidence-bound *development* readiness and next-study planner.

No fit, search, calibration, or test-set access takes place in this module.
Recommendations are research hypotheses, not proof that proposed sensors or
sampling changes will improve diagnosis. A passed development gate is not a
release/RC qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "phm.design-readiness.development-evidence.v1"
OUTPUT_SCHEMA = "phm.design-readiness.recommendation.v1"
REQUIRED_GATES = ("min_macro_f1", "min_class_recall", "max_normal_false_alarm", "min_coverage")


class EvidenceError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise EvidenceError(f"INVALID_FINITE_NUMBER:{name}")
    out = float(value)
    if out < 0 or out > 1:
        raise EvidenceError(f"RATE_OUT_OF_RANGE:{name}")
    return out


def _strict(obj: Any, allowed: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise EvidenceError(f"EXPECTED_OBJECT:{where}")
    extra = set(obj) - allowed
    if extra:
        raise EvidenceError(f"UNSUPPORTED_FIELD:{where}:{sorted(extra)[0]}")
    return obj


def _evidence_sources(record: dict, source_root: Path) -> list[dict]:
    root = source_root.resolve(strict=True)
    bindings = record.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise EvidenceError("MISSING_SOURCE_BINDINGS")
    verified = []
    for entry in bindings:
        _strict(entry, {"path", "sha256"}, "source_binding")
        rel = entry.get("path")
        declared = entry.get("sha256")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise EvidenceError("UNSAFE_SOURCE_PATH")
        if not isinstance(declared, str) or len(declared) != 64 or any(x not in "0123456789abcdef" for x in declared):
            raise EvidenceError("INVALID_SOURCE_SHA")
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise EvidenceError("UNSAFE_OR_MISSING_SOURCE")
        actual = sha_file(target)
        if declared != actual:
            raise EvidenceError(f"SOURCE_SHA_MISMATCH:{rel}")
        verified.append({"path": rel, "sha256": declared})
    if len({x["path"] for x in verified}) != len(verified):
        raise EvidenceError("DUPLICATE_SOURCE_BINDING")
    return sorted(verified, key=lambda x: x["path"])


def analyze(record: Mapping[str, Any], *, source_root: str | Path) -> dict:
    """Evaluate an immutable aggregate development evidence manifest.

    Inputs are aggregate per-environment metrics. If support or groups are
    missing, they stay UNKNOWN. This module NEVER reads raw fault labels and
    NEVER grants a release qualification.
    """
    record = _strict(dict(record), {
        "schema", "domain", "task", "data_role", "protected_splits_opened",
        "environments", "source_bindings", "gates", "physics_pair_evidence",
        "training_provenance", "observed_channels", "candidate_status"
    }, "manifest")
    if record.get("schema") != SCHEMA:
        raise EvidenceError("SCHEMA_MISMATCH")
    if record.get("data_role") != "DEVELOPMENT_ONLY" or record.get("protected_splits_opened") is not False:
        raise EvidenceError("PROTECTED_DATA_ROLE_REJECTED")
    for key in ("domain", "task"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise EvidenceError("MISSING_DOMAIN_OR_TASK")
    gates = _strict(record.get("gates"), set(REQUIRED_GATES) | {"min_balanced_accuracy", "min_fault_detection_recall", "min_component_localization_accuracy"}, "gates")
    for key in REQUIRED_GATES:
        if key not in gates:
            raise EvidenceError(f"MISSING_GATE:{key}")
    for key, value in gates.items():
        _number(value, f"gate.{key}")
    chans = record.get("observed_channels")
    if not isinstance(chans, list) or not chans or any(not isinstance(c, str) or not c.strip() for c in chans):
        raise EvidenceError("OBSERVATION_CONTRACT_UNVERIFIED")
    if any("truth" in c.lower() or "fault_label" in c.lower() for c in chans):
        raise EvidenceError("ORACLE_CHANNEL_REJECTED")
    prov = _strict(record.get("training_provenance"), {"model_family", "candidate_count", "training_count", "model_fits", "source_report_role"}, "training_provenance")
    if prov.get("source_report_role") != "DEVELOPMENT_ONLY":
        raise EvidenceError("UNVERIFIED_TRAIN_PROVENANCE")
    for key in ("candidate_count", "training_count", "model_fits"):
        val = prov.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise EvidenceError(f"INVALID_TRAINING_COST:{key}")
    status = record.get("candidate_status")
    if status not in ("NO_ELIGIBLE", "DEVELOPMENT_ELIGIBLE_NOT_BLIND"):
        raise EvidenceError("INVALID_CANDIDATE_STATUS")
    physics = _strict(record.get("physics_pair_evidence"), {"pair_count", "status"}, "physics")
    if physics.get("status") not in ("PHYSICAL_PAIRS_PASS", "NOT_ASSESSED"):
        raise EvidenceError("INVALID_PHYSICS_STATUS")
    if isinstance(physics.get("pair_count"), bool) or not isinstance(physics.get("pair_count"), int) or physics["pair_count"] < 0:
        raise EvidenceError("INVALID_PHYSICS_PAIR_COUNT")
    if physics["status"] == "PHYSICAL_PAIRS_PASS" and physics["pair_count"] == 0:
        raise EvidenceError("PHYSICS_PAIR_COUNT_MISSING")
    if physics["status"] == "NOT_ASSESSED" and physics["pair_count"]:
        raise EvidenceError("PHYSICS_PAIR_UNVERIFIED")

    envs = record.get("environments")
    if not isinstance(envs, list) or len(envs) < 1:
        raise EvidenceError("NO_DEVELOPMENT_ENVIRONMENTS")
    parsed = []
    observed_ids = set()
    class_names = None
    for env in envs:
        env = _strict(env, {"environment_id", "case_count", "world_group_count", "support_by_class", "metrics"}, "environment")
        name = env.get("environment_id")
        if not isinstance(name, str) or not name or name in observed_ids:
            raise EvidenceError("DUPLICATE_OR_INVALID_ENVIRONMENT")
        observed_ids.add(name)
        n = env.get("case_count")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise EvidenceError("INVALID_ENVIRONMENT_CASE_COUNT")
        groups = env.get("world_group_count")
        if groups is not None and (isinstance(groups, bool) or not isinstance(groups, int) or not 1 <= groups <= n):
            raise EvidenceError("INVALID_WORLD_GROUP_COUNT")
        metrics = _strict(env.get("metrics"), {"macro_f1", "balanced_accuracy", "normal_false_alarm", "coverage", "recall_by_class", "fault_detection_recall", "component_localization_accuracy"}, "metrics")
        for k in ("macro_f1", "normal_false_alarm", "coverage"):
            if k not in metrics:
                raise EvidenceError(f"MISSING_METRIC:{k}")
            _number(metrics[k], k)
        for k in ("balanced_accuracy", "fault_detection_recall", "component_localization_accuracy"):
            if k in metrics and metrics[k] is not None:
                _number(metrics[k], k)
        recalls = metrics.get("recall_by_class")
        if not isinstance(recalls, dict) or "NOMINAL" not in recalls or len(recalls) < 2:
            raise EvidenceError("MISSING_PER_CLASS_RECALL")
        for cls, r in recalls.items():
            if not isinstance(cls, str) or not cls:
                raise EvidenceError("INVALID_CLASS_NAME")
            _number(r, f"recall.{cls}")
        if class_names is not None and class_names != set(recalls):
            raise EvidenceError("CLASS_CONTRACT_DRIFT")
        class_names = set(recalls)
        support = env.get("support_by_class")
        if support is not None:
            if not isinstance(support, dict) or set(support) != set(recalls):
                raise EvidenceError("CLASS_SUPPORT_MISMATCH")
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in support.values()) or sum(support.values()) != n:
                raise EvidenceError("INVALID_CLASS_SUPPORT")
            for cls, count in support.items():
                # A recall measured on n records must correspond to an integer
                # number of correctly classified records (within rounding).
                if abs(round(recalls[cls] * count) - recalls[cls] * count) > 1e-3:
                    raise EvidenceError("RECALL_SUPPORT_INCONSISTENT")
        missing = []
        if metrics["coverage"] >= 1.0 and abs(metrics["normal_false_alarm"] - (1.0 - recalls["NOMINAL"])) > 1e-3:
            raise EvidenceError("NOMINAL_RECALL_FPR_INCONSISTENT")
        if metrics["macro_f1"] < gates["min_macro_f1"]:
            missing.append("MACRO_F1")
        if metrics["normal_false_alarm"] > gates["max_normal_false_alarm"]:
            missing.append("NORMAL_FALSE_ALARM")
        if metrics["coverage"] < gates["min_coverage"]:
            missing.append("LOW_COVERAGE")
        for cls, r in sorted(recalls.items()):
            if cls != "NOMINAL" and r < gates["min_class_recall"]:
                missing.append("LOW_CLASS_RECALL:" + cls)
        for key, metric, fail_code in (
            ("min_balanced_accuracy", "balanced_accuracy", "BALANCED_ACCURACY"),
            ("min_fault_detection_recall", "fault_detection_recall", "FAULT_DETECTION_RECALL"),
            ("min_component_localization_accuracy", "component_localization_accuracy", "LOCALIZATION"),
        ):
            if key in gates:
                if metrics.get(metric) is None:
                    missing.append("METRIC_NOT_REPORTED:" + metric)
                elif metrics[metric] < gates[key]:
                    missing.append(fail_code)
        parsed.append({"environment_id": name, "case_count": n,
                       "world_group_count": groups, "support_by_class": support,
                       "metrics": metrics, "failed_gates": missing})

    verified = _evidence_sources(record, Path(source_root))
    failures = [x for p in parsed for x in p["failed_gates"]]
    class_deficits = {cls: round(min(p["metrics"]["recall_by_class"][cls] for p in parsed), 6)
                      for cls in sorted(class_names) if cls != "NOMINAL"}
    max_fpr = max(p["metrics"]["normal_false_alarm"] for p in parsed)
    min_coverage = min(p["metrics"]["coverage"] for p in parsed)
    all_dev_metrics_pass = not failures
    required_environment_count = 3  # Research evidence sufficiency flag, never substitutes predeclared domain gate
    gaps = []
    if len(parsed) < required_environment_count:
        gaps.append({"code": "LIMITED_INDEPENDENT_ENVIRONMENTS", "observed": len(parsed),
                     "required_research_floor": required_environment_count})
    if any(p["world_group_count"] is None for p in parsed):
        gaps.append({"code": "WORLD_GROUP_COUNTS_UNVERIFIED"})
    if any(p["support_by_class"] is None for p in parsed):
        gaps.append({"code": "CLASS_SUPPORT_UNVERIFIED"})
    else:
        sparse = {cls:min(p["support_by_class"][cls] for p in parsed)
                  for cls in sorted(class_names)}
        sparse = {cls:n for cls,n in sparse.items() if n < 5}
        if sparse:
            gaps.append({"code":"SMALL_CLASS_SUPPORT",
                         "minimum_cases_by_class":sparse,
                         "research_floor_per_environment":5})
    if max_fpr > gates["max_normal_false_alarm"]:
        gaps.append({"code": "NORMAL_FALSE_ALARM_EXCEEDS_GATE", "observed": round(max_fpr, 6),
                     "limit": gates["max_normal_false_alarm"]})
    if min_coverage < gates["min_coverage"]:
        gaps.append({"code": "INSUFFICIENT_COVERAGE", "observed": round(min_coverage, 6)})
    for cls, low in class_deficits.items():
        if low < gates["min_class_recall"]:
            gaps.append({"code": "WEAK_CLASS", "class": cls, "observed_min_recall": low,
                         "required": gates["min_class_recall"]})
    if any(p["metrics"]["macro_f1"] < gates["min_macro_f1"] for p in parsed):
        gaps.append({"code": "MACRO_F1_BELOW_GATE", "observed": round(min(p["metrics"]["macro_f1"] for p in parsed), 6)})
    if physics["status"] == "NOT_ASSESSED":
        gaps.append({"code": "PHYSICAL_OBSERVABILITY_NOT_ASSESSED"})

    experiments = []
    if any(x["code"] == "NORMAL_FALSE_ALARM_EXCEEDS_GATE" for x in gaps):
        experiments.append({"priority": "P0", "study": "NEW_HEALTHY_ENVIRONMENT_AND_CONDITIONING",
            "purpose": "Collect distinct healthy world groups across forcing/thermal regimes and audit normal false alarms before model selection.",
            "qualification_effect": "HYPOTHESIS_ONLY"})
    if any(x["code"] == "WEAK_CLASS" for x in gaps):
        experiments.append({"priority": "P0", "study": "GROUP_BALANCED_FAULT_AND_OBSERVATION_STUDY",
            "weak_classes": sorted(x["class"] for x in gaps if x["code"] == "WEAK_CLASS"),
            "purpose": "Test observability and competing fault hypotheses under independent world groups; do not infer benefit from group counts alone.",
            "qualification_effect": "HYPOTHESIS_ONLY"})
    if any(x["code"] in ("LIMITED_INDEPENDENT_ENVIRONMENTS", "WORLD_GROUP_COUNTS_UNVERIFIED", "CLASS_SUPPORT_UNVERIFIED", "SMALL_CLASS_SUPPORT") for x in gaps):
        experiments.append({"priority": "P0", "study": "NEW_INDEPENDENT_DEVELOPMENT_ENVIRONMENTS",
            "purpose": "Predefine genuinely unseen scene/lineage groups, per-class support and search budget before data generation.",
            "qualification_effect": "HYPOTHESIS_ONLY"})
    if physics["status"] == "PHYSICAL_PAIRS_PASS" and failures:
        experiments.append({"priority": "P1", "study": "OBSERVATION_MECHANISM_DISCRIMINABILITY_AUDIT",
            "purpose": "Contrast measured multi-channel fault responses with healthy variability and competing fault hypotheses; paired changes are not diagnosis proof.",
            "qualification_effect": "HYPOTHESIS_ONLY"})
    if not experiments:
        experiments.append({"priority": "P0", "study": "FRESH_CALIBRATION_AND_SEALED_TEST_PROTOCOL",
            "purpose": "Development-only results never qualify a release. Freeze model, data, risk policy and release checks before opening new cohorts.",
            "qualification_effect": "HYPOTHESIS_ONLY"})
    result = {
        "schema": OUTPUT_SCHEMA,
        "domain": record["domain"], "task": record["task"],
        "source_role": "DEVELOPMENT_ONLY", "candidate_status": status,
        "development_gate_metrics_pass": all_dev_metrics_pass,
        "independent_qualification": False,
        "release_decision": "HOLD_DO_NOT_PROMOTE",
        "physics_status": ("PAIRED_SIGNAL_CHANGE_ONLY_NOT_DIAGNOSIS" if physics["status"] == "PHYSICAL_PAIRS_PASS" else "OBSERVABILITY_UNKNOWN"),
        "class_minimum_recall": class_deficits,
        "maximum_normal_false_alarm": round(max_fpr, 6),
        "minimum_coverage": round(min_coverage, 6),
        "environment_audit": parsed,
        "evidence_gaps": gaps,
        "proposed_studies": experiments,
        "verified_source_bindings": verified,
        "source_manifest_sha256": _sha_bytes(_json_bytes(record)),
        "forbidden_claims": ["fresh_blind_pass", "calibration_qualified", "deployment_qualified", "automatic_rc_promotion"],
    }
    result["recommendation_fingerprint"] = _sha_bytes(_json_bytes(result))
    return result
