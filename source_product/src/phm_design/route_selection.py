"""Recovered dev5 route selection with dev6 per-label and evidence-use guards.

Calibration is a second selection environment, not a probability/risk calibration
sample. The empirical criteria below do NOT guarantee nondegradation on new data.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .data import Dataset, transform, reject_overlap, sequence_from_csv
from .engine import softmax, load_ensemble, ensemble_scores, deterministic_cpu
from .reference import reference_logits
from .reference_design import score_report
from .evidence_ledger import EvidenceLedger
from .io import read_json, write_json, digest, seal, verify, sha_file, run_lock

ROUTES = ("reference", "fixed", "bounded", "raw", "unbounded")
POLICY = {"schema": "phm.route-policy.v2", "recall_guard": "each_class",
    "strict_improvement": "worst_environment_f1_or_minimum_recall",
    "tie_preference": list(ROUTES), "score_threshold_source": "frozen_design"}
REQUIRED_RUN = {"deployment.json", "design_report.json", "reference.json", "preprocessing.json",
                "development_snapshot.json", "observation_hashes.json"}


def _policy(policy):
    p = copy.deepcopy(POLICY if policy is None else policy)
    if set(p) != set(POLICY) or any(p[k] != POLICY[k] for k in p if k != "recall_guard"):
        raise ValueError("ROUTE_POLICY_SCHEMA")
    if p["recall_guard"] not in {"each_class", "weakest_class"}:
        raise ValueError("ROUTE_RECALL_GUARD")
    return p


def _metrics(metric, classes):
    if set(metric.get("recall_by_class", {})) != set(classes):
        raise ValueError("ROUTE_CLASS_ORDER_OR_COVERAGE")
    vals = [metric.get("macro_f1"), *metric["recall_by_class"].values()]
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in vals):
        raise ValueError("ROUTE_METRIC_INVALID")
    n = metric.get("cases")
    if type(n) is not int or n <= 0:
        raise ValueError("ROUTE_CASE_COUNT_INVALID")
    for k in ("accepted_cases", "accepted_errors", "accepted_correct"):
        if type(metric.get(k)) is not int or not 0 <= metric[k] <= n:
            raise ValueError("ROUTE_ACCEPTED_COUNTS_INVALID")
    if metric["accepted_errors"] + metric["accepted_correct"] != metric["accepted_cases"]:
        raise ValueError("ROUTE_ACCEPTED_COUNTS_INCONSISTENT")


def route_eligible(dev, cal, ref_dev, ref_cal, route, policy=None):
    policy = _policy(policy)
    classes = tuple(ref_dev.get("recall_by_class", {}))
    if not classes or route not in ROUTES:
        raise ValueError("ROUTE_OR_CLASSES_INVALID")
    for m in (dev, cal, ref_dev, ref_cal):
        _metrics(m, classes)
    if route == "reference":
        return True
    eps = 1e-12
    cf = min(dev["macro_f1"], cal["macro_f1"])
    rf = min(ref_dev["macro_f1"], ref_cal["macro_f1"])
    cr = min(min(m["recall_by_class"].values()) for m in (dev, cal))
    rr = min(min(m["recall_by_class"].values()) for m in (ref_dev, ref_cal))
    if policy["recall_guard"] == "each_class":
        recall_ok = all(m["recall_by_class"][c] >= r["recall_by_class"][c] - eps
                        for m, r in ((dev, ref_dev), (cal, ref_cal)) for c in classes)
    else:
        recall_ok = all(min(m["recall_by_class"].values()) >= min(r["recall_by_class"].values()) - eps
                        for m, r in ((dev, ref_dev), (cal, ref_cal)))
    return bool(cf >= rf - eps and recall_ok and (cf > rf + eps or cr > rr + eps))


def route_rank(dev, cal, route):
    return (-min(dev["macro_f1"], cal["macro_f1"]), -(dev["macro_f1"] + cal["macro_f1"]) / 2,
        max(m["accepted_errors"] / m["cases"] for m in (dev, cal)),
        -min(m["accepted_correct"] / m["cases"] for m in (dev, cal)), ROUTES.index(route))


def choose_metrics(dev, cal, policy=None):
    policy = _policy(policy)
    if "reference" not in dev or set(dev) != set(cal) or not set(dev) <= set(ROUTES):
        raise ValueError("ROUTE_METRICS_INCOMPLETE")
    eligible = {r: route_eligible(dev[r], cal[r], dev["reference"], cal["reference"], r, policy) for r in dev}
    ranks = {r: route_rank(dev[r], cal[r], r) for r in dev}
    chosen = min((r for r in dev if eligible[r]), key=lambda r: ranks[r])
    return chosen, eligible, ranks


def _run(run):
    run = Path(run)
    included = verify(run)
    if not REQUIRED_RUN <= set(included):
        raise ValueError("RUN_MANIFEST_MISSING_REQUIRED_RECORD")
    dep = read_json(run / "deployment.json")
    if dep.get("schema") != "phm.reference-deployment.v1":
        raise ValueError("REFERENCE_DEPLOYMENT_SCHEMA")
    return run, dep, sha_file(run / "manifest.json")


def _scores(run, x):
    result = {"reference": softmax(reference_logits(read_json(run / "reference.json"), x))}
    for r in ROUTES[1:]:
        p = run / ("selected_" + r)
        if p.exists():
            _, models = load_ensemble(p)
            result[r] = ensemble_scores(models, x)
    return result


def _signature(x):
    return digest(np.asarray(x).tolist())


def _no_numeric_reuse(x, used):
    signatures = [_signature(row) for row in x]
    if set(signatures) & set(used):
        raise ValueError("IDENTICAL_OBSERVATIONS_ACROSS_ROLES")
    return signatures


def _budget_terminal_design(run, rep):
    """True only when every non-completed planned candidate terminated by the frozen trial-time budget.

    This does not convert a failed model into a valid candidate. It only permits the
    completed candidates from an otherwise fully executed design run to proceed to
    downstream selection. Interrupted, crashed, missing, or numerically failed trials
    remain fatal.
    """
    if rep.get("status") == "COMPLETED":
        return True
    if rep.get("status") != "INCOMPLETE_TRIALS":
        return False
    saw_budget_failure = False
    reports = sorted(Path(run).glob("generation_*/run/search_report.json"))
    if not reports:
        return False
    for path in reports:
        report = read_json(path)
        for candidate in report.get("candidate_results", []):
            if candidate.get("status") == "COMPLETED":
                continue
            trials = candidate.get("trials")
            if not isinstance(trials, list) or not trials:
                return False
            for trial in trials:
                if trial.get("status") != "FAILED" or trial.get("reason") != "TRIAL_TIME_BUDGET_EXCEEDED":
                    return False
                saw_budget_failure = True
    return saw_budget_failure


def _development(run):
    rep = read_json(run / "design_report.json")
    if not _budget_terminal_design(run, rep):
        raise ValueError("INCOMPLETE_SOURCE_DESIGN")
    dev = {"reference": rep["reference_validation"]}
    for r in ROUTES[1:]:
        w = rep["winners"].get(r)
        if w and w["status"] == "COMPLETED":
            dev[r] = w["ensemble_validation"]
    return dev


def select_route(run, calibration, output, *, registry, policy=None):
    policy = _policy(policy)
    output = Path(output)
    with run_lock(Path(run)):
        run, dep, source = _run(run)
        if (run / "evaluation_ledger.json").exists():
            raise ValueError("SOURCE_HOLDOUT_ALREADY_OPENED")
        data = Dataset(calibration, allowed_splits={"calibration"})
        if data.contract != dep["contract"]:
            raise ValueError("CALIBRATION_CONTRACT_MISMATCH")
        development = read_json(run / "development_snapshot.json")["samples"]
        reject_overlap(development, data.samples)
        data.verify_files()
        block = data.load("calibration", dep["shape"]["window"])
        signatures = _no_numeric_reuse(block.x, read_json(run / "observation_hashes.json"))
        key = digest({"source": source, "calibration": data.hash, "policy": policy})
        with EvidenceLedger(registry) as ledger:
            cached = ledger.claim(key, "selection", source, output, data.contract["device_family"],
                                  [*development, *data.samples], [(x, "calibration") for x in signatures])
            if cached is not None:
                if cached.resolve() != output.resolve():
                    raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:" + str(cached))
                return read_json(cached / "selection.json")
            try:
                output.mkdir(parents=True, exist_ok=False)
                plan = {"schema": "phm.route-selection-plan.v2", "operation": key,
                    "source_manifest_sha256": source, "policy": policy, "calibration_hash": data.hash,
                    "implementation_sha256": sha_file(Path(__file__))}
                write_json(output / "plan.json", plan)
                x = transform(block.x, read_json(run / "preprocessing.json"))
                with deterministic_cpu():
                    scores = _scores(run, x)
                dev = _development(run)
                cal = {r: score_report(block.y, s, data.contract["classes"], dep["threshold"]) for r, s in scores.items()}
                chosen, eligible, ranks = choose_metrics(dev, cal, policy)
                record = {"schema": "phm.reference-route-selection.v2", "operation": key,
                    "source_run_manifest_sha256": source, "source_design_fingerprint": dep["fingerprint"],
                    "contract": dep["contract"], "shape": dep["shape"], "policy": policy,
                    "calibration_dataset_hash": data.hash, "calibration_samples": data.samples,
                    "calibration_numeric_signatures": signatures,
                    "calibration_role_consumed_for_route_selection": True, "probability_calibration_performed": False,
                    "test_used": False, "development_metrics": dev, "calibration_metrics": cal,
                    "rank_keys": {r: list(k) for r, k in ranks.items()}, "eligibility": eligible,
                    "chosen_route": chosen, "risk_certified": False, "production_qualified": False}
                write_json(output / "selection.json", record)
                write_json(output / "scores.json", {"samples": data.samples, "truth": block.y.tolist(),
                    "routes": {r: s.tolist() for r, s in scores.items()}})
                seal(output, ["plan.json", "selection.json", "scores.json"])
                ledger.complete(key, output)
                return record
            except Exception as exc:
                ledger.fail(key, exc)
                raise


def _selection(run, selection):
    run, dep, source = _run(run)
    selection = Path(selection)
    included = verify(selection)
    if not {"plan.json", "selection.json", "scores.json"} <= set(included):
        raise ValueError("SELECTION_MANIFEST_MISSING_RECORD")
    rec = read_json(selection / "selection.json")
    if rec.get("schema") not in {"phm.reference-route-selection.v2","phm.multi-environment-route-selection.v1"} or rec.get("chosen_route") not in ROUTES:
        raise ValueError("ROUTE_SELECTION_SCHEMA")
    if rec["source_run_manifest_sha256"] != source or rec["source_design_fingerprint"] != dep["fingerprint"]:
        raise ValueError("ROUTE_SELECTION_RUN_CHANGED")
    if rec["contract"] != dep["contract"] or rec["shape"] != dep["shape"]:
        raise ValueError("ROUTE_SELECTION_CONTRACT_CHANGED")
    return run, dep, source, rec


def selected_scores(run, selection, x):
    run, dep, source, rec = _selection(run, selection)
    r = rec["chosen_route"]
    if r == "reference":
        return softmax(reference_logits(read_json(run / "reference.json"), x))
    _, models = load_ensemble(run / ("selected_" + r))
    return ensemble_scores(models, x)


def evaluate_suite(run, selection, holdouts, output, *, registry):
    """One predeclared multi-cohort opening; exact repeats return verified cache.

    Repeated world IDs across cohorts are expected paired stress views. The union
    count, not the sum of cohort groups, is reported as the evidence unit count.
    """
    output = Path(output)
    if not isinstance(holdouts, dict) or not holdouts or any(not isinstance(k, str) or not k.isidentifier() for k in holdouts):
        raise ValueError("EVALUATION_SUITE_NAMES")
    with run_lock(Path(run)):
        run, dep, source, rec = _selection(run, selection)
        data = {k: Dataset(v, allowed_splits={"test"}) for k, v in sorted(holdouts.items())}
        development = read_json(run / "development_snapshot.json")["samples"]
        for d in data.values():
            if d.contract != dep["contract"]:
                raise ValueError("HOLDOUT_CONTRACT_MISMATCH")
            reject_overlap(development, d.samples)
            reject_overlap(rec["calibration_samples"], d.samples)
            d.verify_files()
        plan = {"schema": "phm.frozen-evaluation-suite.v1", "source": source,
            "selection": sha_file(Path(selection) / "manifest.json"),
            "holdouts": {k: d.hash for k, d in data.items()},
            "evaluation_implementation_sha256": sha_file(Path(__file__))}
        key = digest(plan)
        with EvidenceLedger(registry) as ledger:
            cached = ledger.claim(key, "evaluation", source, output, dep["contract"]["device_family"],
                [s for d in data.values() for s in d.samples])
            if cached is not None:
                if cached.resolve() != output.resolve():
                    raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:" + str(cached))
                return read_json(cached / "evaluation_report.json")
            try:
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / "plan.json", plan)
                used = set(read_json(run / "observation_hashes.json")) | set(rec["calibration_numeric_signatures"])
                cohorts = {}
                with deterministic_cpu():
                    for name, d in data.items():
                        block = d.load("test", dep["shape"]["window"])
                        _no_numeric_reuse(block.x, used)
                        x = transform(block.x, read_json(run / "preprocessing.json"))
                        scores = _scores(run, x)
                        report = {r: score_report(block.y, s, d.contract["classes"], dep["threshold"]) for r, s in scores.items()}
                        cohorts[name] = {"counts": d.counts(), "metrics": report,
                            "deployment": report[rec["chosen_route"]], "dataset_hash": d.hash}
                        write_json(output / (name + "_predictions.json"), {"samples": block.samples,
                            "truth": block.y.tolist(), "scores": {r: v.tolist() for r, v in scores.items()}})
                groups = {s["group_id"] for d in data.values() for s in d.samples}
                result = {"schema": "phm.selected-route-suite-evaluation.v1", "operation": key,
                    "chosen_route": rec["chosen_route"], "union_test_groups": len(groups),
                    "cohort_case_views": sum(len(d.samples) for d in data.values()), "cohorts": cohorts,
                    "test_used_for_selection": False, "probability_calibrated": False,
                    "risk_certified": False, "production_qualified": False}
                write_json(output / "evaluation_report.json", result)
                seal(output, [p.name for p in output.iterdir() if p.is_file() and p.name != "manifest.json"])
                ledger.complete(key, output)
                return result
            except Exception as exc:
                ledger.fail(key, exc)
                raise


def evaluate_selected(run, selection, holdout, output, *, registry):
    return evaluate_suite(run, selection, {"test": holdout}, output, registry=registry)


def predict_selected(run, selection, csv_path, passport_path, domain):
    from .family_pipeline import feature_contract, compute_features, implementation_sha, TRANSFORM
    from .adapters.sat_nonadcs import contract
    run, dep, _, rec = _selection(run, selection)
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
        s = selected_scores(run, selection, x)[0]
    j = int(s.argmax()); accepted = bool(s[j] >= dep["threshold"])
    return {"route": rec["chosen_route"], "status": "PREDICTED" if accepted else "ABSTAIN_LOW_SCORE",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": s.tolist(),
        "probability_calibrated": False, "claim_status": "RESEARCH_ONLY"}
