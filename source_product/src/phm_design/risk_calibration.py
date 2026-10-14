"""Independent selective-risk calibration for an already frozen PHM route.

The route/model is fixed before this stage. A distinct calibration dataset is used
only to choose an abstention threshold. Statistical statements are deliberately
narrow: the exact binomial upper bound applies to the event that a calibration
*world group* contains at least one accepted classification error, under an iid
group assumption. Bonferroni correction covers the predeclared threshold grid.
It is not probability calibration, domain-shift coverage, unknown-fault detection,
or production/safety qualification.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

from .data import Dataset, reject_overlap, transform
from .engine import deterministic_cpu, metrics
from .evidence_ledger import EvidenceLedger
from .io import digest, read_json, seal, sha_file, verify, write_json
from .route_selection import _selection, _no_numeric_reuse, selected_scores


POLICY = {
    "schema": "phm.selective-risk-policy.v1",
    "thresholds": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    "alpha": 0.05,
    "max_group_error_upper_bound": 0.20,
    "min_case_coverage": 0.25,
}


def _policy(policy=None) -> dict:
    p = dict(POLICY if policy is None else policy)
    v1={"schema", "thresholds", "alpha", "max_group_error_upper_bound", "min_case_coverage"}
    v2=v1|{"min_class_case_coverage"}
    if set(p) not in {frozenset(v1),frozenset(v2)}:
        raise ValueError("RISK_POLICY_SCHEMA")
    if p["schema"] not in {"phm.selective-risk-policy.v1","phm.selective-risk-policy.v2"}:
        raise ValueError("RISK_POLICY_SCHEMA")
    if p["schema"]=="phm.selective-risk-policy.v1" and set(p)!=v1:
        raise ValueError("RISK_POLICY_SCHEMA")
    if p["schema"]=="phm.selective-risk-policy.v2" and set(p)!=v2:
        raise ValueError("RISK_POLICY_SCHEMA")
    t = p["thresholds"]
    if (not isinstance(t, list) or not t or len(t) > 50 or
        any(type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) or not 1/3 <= v <= 1 for v in t)):
        raise ValueError("RISK_THRESHOLD_GRID")
    t = [float(v) for v in t]
    if t != sorted(set(t)):
        raise ValueError("RISK_THRESHOLD_GRID")
    numeric_names=["alpha", "max_group_error_upper_bound", "min_case_coverage"]
    if p["schema"]=="phm.selective-risk-policy.v2":numeric_names.append("min_class_case_coverage")
    for name in numeric_names:
        v = p[name]
        if type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("RISK_POLICY_VALUE:" + name)
    if not 0 < p["alpha"] < 1:
        raise ValueError("RISK_POLICY_ALPHA")
    if not 0 < p["max_group_error_upper_bound"] < 1:
        raise ValueError("RISK_POLICY_TARGET")
    if not 0 <= p["min_case_coverage"] <= 1:
        raise ValueError("RISK_POLICY_COVERAGE")
    if p["schema"]=="phm.selective-risk-policy.v2" and not 0 <= p["min_class_case_coverage"] <= 1:
        raise ValueError("RISK_POLICY_CLASS_COVERAGE")
    p["thresholds"] = t
    p["alpha"] = float(p["alpha"])
    p["max_group_error_upper_bound"] = float(p["max_group_error_upper_bound"])
    p["min_case_coverage"] = float(p["min_case_coverage"])
    if p["schema"]=="phm.selective-risk-policy.v2":
        p["min_class_case_coverage"] = float(p["min_class_case_coverage"])
    return p


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n <= 0:
        raise ValueError("BINOMIAL_COUNTS")
    if not 0 <= p <= 1 or not math.isfinite(p):
        raise ValueError("BINOMIAL_PROBABILITY")
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided exact binomial upper confidence limit."""
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n <= 0:
        raise ValueError("BINOMIAL_COUNTS")
    if type(alpha) not in (int, float) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("BINOMIAL_ALPHA")
    if k == n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def selective_report(y, scores, samples, classes, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(y), len(classes)) or len(samples) != len(y) or not np.isfinite(scores).all():
        raise ValueError("SELECTIVE_SCORE_SHAPE")
    if type(threshold) not in (int, float) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("SELECTIVE_THRESHOLD")
    pred = scores.argmax(1)
    confidence = scores.max(1)
    accepted = confidence >= float(threshold)
    correct = pred == y
    groups = {}
    for i, sample in enumerate(samples):
        gid = sample["group_id"]
        groups.setdefault(gid, []).append(i)
    failures = 0
    accepted_groups = 0
    for ids in groups.values():
        ids = np.asarray(ids, dtype=int)
        if accepted[ids].any():
            accepted_groups += 1
        if (accepted[ids] & ~correct[ids]).any():
            failures += 1
    accepted_count = int(accepted.sum())
    accepted_errors = int((accepted & ~correct).sum())
    class_coverage={}
    class_accepted={}
    class_support={}
    for i,name in enumerate(classes):
        mask=y==i;n=int(mask.sum());a=int((accepted & mask).sum())
        class_support[name]=n;class_accepted[name]=a
        class_coverage[name]=(float(a/n) if n else None)
    supported_coverage=[v for v in class_coverage.values() if v is not None]
    closed = metrics(y, pred, classes)
    return {
        **closed,
        "threshold": float(threshold),
        "cases": int(len(y)),
        "groups": int(len(groups)),
        "accepted_cases": accepted_count,
        "accepted_correct": int((accepted & correct).sum()),
        "accepted_errors": accepted_errors,
        "case_coverage": float(accepted.mean()),
        "class_support":class_support,
        "accepted_cases_by_class":class_accepted,
        "case_coverage_by_class":class_coverage,
        "minimum_class_case_coverage":(min(supported_coverage) if supported_coverage else None),
        "accepted_case_error_rate": (float(accepted_errors / accepted_count) if accepted_count else None),
        "accepted_groups": int(accepted_groups),
        "group_failure_events": int(failures),
        "group_failure_rate": float(failures / len(groups)),
    }


def choose_threshold(y, scores, samples, classes, policy=None) -> dict:
    p = _policy(policy)
    corrected_alpha = p["alpha"] / len(p["thresholds"])
    candidates = []
    for threshold in p["thresholds"]:
        report = selective_report(y, scores, samples, classes, threshold)
        upper = clopper_pearson_upper(report["group_failure_events"], report["groups"], corrected_alpha)
        eligible = (upper <= p["max_group_error_upper_bound"] and
                    report["case_coverage"] >= p["min_case_coverage"])
        if p["schema"]=="phm.selective-risk-policy.v2":
            all_classes_supported=all(report["class_support"][c]>0 for c in classes)
            eligible = (eligible and all_classes_supported and
                        report["minimum_class_case_coverage"] is not None and
                        report["minimum_class_case_coverage"] >= p["min_class_case_coverage"])
        candidates.append({**report, "bonferroni_alpha": corrected_alpha,
            "group_failure_upper_bound": upper, "eligible": bool(eligible)})
    eligible = [c for c in candidates if c["eligible"]]
    if eligible:
        # Maximize usable coverage after the risk constraint, then prefer smaller
        # upper bound and lower threshold for deterministic ties.
        chosen = sorted(eligible, key=lambda c: (-c["case_coverage"], c["group_failure_upper_bound"], c["threshold"]))[0]
        status = "QUALIFIED_THRESHOLD"
        threshold = chosen["threshold"]
    else:
        chosen = None
        status = "NO_THRESHOLD_QUALIFIED_ABSTAIN_ALL"
        threshold = None
    return {
        "schema": "phm.selective-risk-calibration-result.v1",
        "status": status,
        "threshold": threshold,
        "policy": p,
        "candidates": candidates,
        "selected": chosen,
        "statistical_scope": {
            "unit": "world_group",
            "failure_event": "at_least_one_accepted_classification_error_in_group",
            "bound": "one_sided_exact_binomial_clopper_pearson",
            "multiple_threshold_adjustment": "bonferroni_over_predeclared_grid",
            "assumption": "iid calibration world groups from the target calibration distribution",
            "not_claimed": ["casewise iid", "probability calibration", "domain-shift guarantee",
                            "unknown-fault guarantee", "hardware or production qualification"],
        },
    }


def calibrate_risk(run, selection, calibration, output, *, registry, policy=None):
    output = Path(output)
    p = _policy(policy)
    run, dep, source, rec = _selection(run, selection)
    if output.exists():
        raise FileExistsError("RISK_CALIBRATION_OUTPUT_EXISTS")
    data = Dataset(calibration, allowed_splits={"calibration"})
    if data.contract != dep["contract"]:
        raise ValueError("RISK_CALIBRATION_CONTRACT_MISMATCH")
    development = read_json(run / "development_snapshot.json")["samples"]
    reject_overlap(development, data.samples)
    reject_overlap(rec["calibration_samples"], data.samples)
    data.verify_files()
    block = data.load("calibration", dep["shape"]["window"])
    used = set(read_json(run / "observation_hashes.json")) | set(rec["calibration_numeric_signatures"])
    signatures = _no_numeric_reuse(block.x, used)
    selection_manifest = sha_file(Path(selection) / "manifest.json")
    key = digest({"run": source, "selection": selection_manifest, "risk_calibration": data.hash, "policy": p})
    with EvidenceLedger(registry) as ledger:
        cached = ledger.claim(key, "risk_calibration", selection_manifest, output, dep["contract"]["device_family"],
                              data.samples, [(x, "calibration") for x in signatures])
        if cached is not None:
            if cached.resolve() != output.resolve():
                raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:" + str(cached))
            return read_json(cached / "risk_calibration.json")
        try:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", {"schema": "phm.selective-risk-calibration-plan.v1",
                "operation": key, "run_manifest_sha256": source, "selection_manifest_sha256": selection_manifest,
                "calibration_hash": data.hash, "policy": p, "implementation_sha256": sha_file(Path(__file__))})
            x = transform(block.x, read_json(run / "preprocessing.json"))
            with deterministic_cpu():
                scores = selected_scores(run, selection, x)
            result = choose_threshold(block.y, scores, block.samples, data.contract["classes"], p)
            result.update({
                "operation": key,
                "run_manifest_sha256": source,
                "selection_manifest_sha256": selection_manifest,
                "route": rec["chosen_route"],
                "contract": dep["contract"],
                "shape": dep["shape"],
                "risk_calibration_dataset_hash": data.hash,
                "risk_calibration_samples": data.samples,
                "risk_calibration_numeric_signatures": signatures,
                "selection_calibration_reused": False,
                "test_used": False,
                "group_risk_bound_qualified": result["status"] == "QUALIFIED_THRESHOLD",
                "risk_certified": False,
                "production_qualified": False,
            })
            write_json(output / "risk_calibration.json", result)
            write_json(output / "scores.json", {"samples": block.samples, "truth": block.y.tolist(), "scores": scores.tolist()})
            seal(output, ["plan.json", "risk_calibration.json", "scores.json"])
            ledger.complete(key, output)
            return result
        except Exception as exc:
            ledger.fail(key, exc)
            raise


def _risk(run, selection, risk):
    run, dep, source, rec = _selection(run, selection)
    risk = Path(risk)
    included = verify(risk)
    if not {"plan.json", "risk_calibration.json", "scores.json"} <= set(included):
        raise ValueError("RISK_MANIFEST_MISSING_RECORD")
    rr = read_json(risk / "risk_calibration.json")
    if rr.get("schema") != "phm.selective-risk-calibration-result.v1":
        raise ValueError("RISK_CALIBRATION_SCHEMA")
    if rr["run_manifest_sha256"] != source or rr["selection_manifest_sha256"] != sha_file(Path(selection) / "manifest.json"):
        raise ValueError("RISK_CALIBRATION_SOURCE_CHANGED")
    if rr["contract"] != dep["contract"] or rr["shape"] != dep["shape"] or rr["route"] != rec["chosen_route"]:
        raise ValueError("RISK_CALIBRATION_BINDING_CHANGED")
    return run, dep, source, rec, rr


def evaluate_risk(run, selection, risk, holdout, output, *, registry):
    output = Path(output)
    run, dep, source, rec, rr = _risk(run, selection, risk)
    if output.exists():
        raise FileExistsError("RISK_EVALUATION_OUTPUT_EXISTS")
    data = Dataset(holdout, allowed_splits={"test"})
    if data.contract != dep["contract"]:
        raise ValueError("HOLDOUT_CONTRACT_MISMATCH")
    development = read_json(run / "development_snapshot.json")["samples"]
    reject_overlap(development, data.samples)
    reject_overlap(rec["calibration_samples"], data.samples)
    reject_overlap(rr["risk_calibration_samples"], data.samples)
    data.verify_files()
    block = data.load("test", dep["shape"]["window"])
    used = (set(read_json(run / "observation_hashes.json")) |
            set(rec["calibration_numeric_signatures"]) |
            set(rr["risk_calibration_numeric_signatures"]))
    _no_numeric_reuse(block.x, used)
    risk_manifest = sha_file(Path(risk) / "manifest.json")
    plan = {"schema": "phm.selective-risk-evaluation-plan.v1", "risk_manifest_sha256": risk_manifest,
            "holdout_hash": data.hash, "implementation_sha256": sha_file(Path(__file__))}
    key = digest(plan)
    with EvidenceLedger(registry) as ledger:
        cached = ledger.claim(key, "evaluation", risk_manifest, output, dep["contract"]["device_family"], data.samples)
        if cached is not None:
            if cached.resolve() != output.resolve():
                raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:" + str(cached))
            return read_json(cached / "evaluation_report.json")
        try:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            x = transform(block.x, read_json(run / "preprocessing.json"))
            with deterministic_cpu():
                scores = selected_scores(run, selection, x)
            classes = data.contract["classes"]
            frozen = selective_report(block.y, scores, block.samples, classes, dep["threshold"])
            if rr["status"] == "QUALIFIED_THRESHOLD":
                calibrated = selective_report(block.y, scores, block.samples, classes, rr["threshold"])
            else:
                # Same closed-set predictions, but the deployed risk policy abstains all.
                calibrated = selective_report(block.y, scores, block.samples, classes, 1.0)
                calibrated["accepted_cases"] = 0
                calibrated["accepted_correct"] = 0
                calibrated["accepted_errors"] = 0
                calibrated["case_coverage"] = 0.0
                calibrated["accepted_case_error_rate"] = None
                calibrated["accepted_groups"] = 0
                calibrated["group_failure_events"] = 0
                calibrated["group_failure_rate"] = 0.0
                calibrated["threshold"] = None
            result = {"schema": "phm.selective-risk-evaluation.v1", "operation": key,
                "route": rec["chosen_route"], "risk_status": rr["status"],
                "risk_threshold": rr["threshold"], "frozen_design_threshold": dep["threshold"],
                "holdout_hash": data.hash, "counts": data.counts(),
                "frozen_threshold_metrics": frozen, "risk_threshold_metrics": calibrated,
                "test_used_for_threshold_selection": False, "probability_calibrated": False,
                "risk_bound_revalidated_on_test": False, "production_qualified": False}
            write_json(output / "evaluation_report.json", result)
            write_json(output / "predictions.json", {"samples": block.samples, "truth": block.y.tolist(), "scores": scores.tolist()})
            seal(output, ["plan.json", "evaluation_report.json", "predictions.json"])
            ledger.complete(key, output)
            return result
        except Exception as exc:
            ledger.fail(key, exc)
            raise


def predict_risk(run, selection, risk, csv_path, passport_path, domain):
    from .family_pipeline import compute_features, feature_contract, implementation_sha, TRANSFORM
    from .adapters.sat_nonadcs import contract
    from .data import sequence_from_csv
    run, dep, _, rec, rr = _risk(run, selection, risk)
    if dep.get("commissioning_profile_required"):
        raise ValueError("COMMISSIONING_PROFILE_REQUIRED_USE_COMMISSIONED_PREDICTOR")
    if dep.get("raw_feature_transform") not in (None, TRANSFORM) or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("REFERENCE_FEATURE_IMPLEMENTATION_CHANGED")
    if dep["contract"] != feature_contract(domain, "mechanism"):
        raise ValueError("REFERENCE_RAW_PIPELINE_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), contract(domain), dep["shape"]["window"])
    features = compute_features(domain, raw, read_json(passport_path), "mechanism")
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu():
        scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax())
    if rr["status"] != "QUALIFIED_THRESHOLD":
        accepted = False
    else:
        accepted = bool(scores[j] >= rr["threshold"])
    return {"route": rec["chosen_route"], "risk_status": rr["status"],
        "status": "PREDICTED" if accepted else "ABSTAIN_RISK_THRESHOLD",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": scores.tolist(),
        "threshold": rr["threshold"], "probability_calibrated": False,
        "group_risk_bound_qualified": rr["status"] == "QUALIFIED_THRESHOLD",
        "risk_certified": False, "claim_status": "RESEARCH_ONLY"}
