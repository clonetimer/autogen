from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

from .algorithm_dag import (
    _eligible,
    _metrics,
    execute,
    records_to_arrays,
    search,
    search_multi_environment,
    search_physical_rule_tree_multi_environment,
)


def _read(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _records(path: str | Path) -> list[dict]:
    payload = _read(path)
    records = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("ALGORITHM_DAG_RECORDS_JSON")
    return records


def _named_paths(values: list[str] | None, *, reason: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(reason)
        name, path = value.split("=", 1)
        name = name.strip(); p = Path(path.strip())
        if not name or name in result or not p.is_file():
            raise ValueError(reason)
        result[name] = p
    return result


def _require(value, reason: str):
    if value in (None, "", []):
        raise ValueError(reason)
    return value


def _single(args) -> dict:
    records = _records(_require(args.records, "ALGORITHM_DAG_RECORDS_REQUIRED"))
    config = _read(_require(args.config, "ALGORITHM_DAG_CONFIG_REQUIRED"))
    report = search(records, config)
    out = Path(_require(args.output, "ALGORITHM_DAG_OUTPUT_REQUIRED")); out.mkdir(parents=True, exist_ok=False)
    _write(out / "search_report.json", report)
    if report.get("chosen_candidate") is not None:
        _write(out / "selected_algorithm_dag.json", report["chosen_candidate"])
    return report


def _multi(args) -> dict:
    config_path = _require(args.config, "ALGORITHM_DAG_CONFIG_REQUIRED")
    env_paths = _named_paths(args.environment, reason="ALGORITHM_DAG_ENVIRONMENT_NAME_PATH")
    if len(env_paths) < 2:
        raise ValueError("ALGORITHM_DAG_MULTI_ENVIRONMENTS_REQUIRED")
    environments = {name: _records(path) for name, path in env_paths.items()}
    config = _read(config_path)
    if args.search_kind == "physical-rule-tree":
        report = search_physical_rule_tree_multi_environment(environments, config)
    else:
        report = search_multi_environment(environments, config)
    out = Path(_require(args.output, "ALGORITHM_DAG_OUTPUT_REQUIRED")); out.mkdir(parents=True, exist_ok=False)
    _write(out / "search_report.json", report)
    if report.get("chosen_candidate") is not None:
        _write(out / "selected_algorithm_dag.json", report["chosen_candidate"])
    _write(out / "environment_bindings.json", {
        "data_role": config.get("data_role"),
        "test_or_holdout_used_for_selection": False,
        "environments": {name: {"path": str(path), "sha256": _sha(path)} for name, path in env_paths.items()},
    })
    return report


def _evaluate(args) -> dict:
    records_path = Path(_require(args.records, "ALGORITHM_DAG_RECORDS_REQUIRED"))
    candidate_path = Path(_require(args.candidate, "ALGORITHM_DAG_CANDIDATE_REQUIRED"))
    config_path = Path(_require(args.config, "ALGORITHM_DAG_CONFIG_REQUIRED"))
    records = _records(records_path)
    candidate = _read(candidate_path); config = _read(config_path)
    signals, truth, truth_location = records_to_arrays(records, candidate["classes"])
    runtime = execute(candidate, signals)
    metrics = _metrics(truth, runtime["class_index"], truth_location, runtime["location_index"],
                       candidate["nominal_index"], candidate["classes"])
    eligible, reasons = _eligible(metrics, config.get("constraints", {}))
    report = {
        "schema": "phm.algorithm-dag-frozen-evaluation.v1",
        "status": "PASS" if eligible else "FAIL_CLOSED",
        "candidate_fingerprint": candidate.get("fingerprint"),
        "candidate_sha256": _sha(candidate_path),
        "records_sha256": _sha(records_path),
        "config_sha256": _sha(config_path),
        "cases": len(records),
        "groups": len({r.get("group_id") for r in records if r.get("group_id") is not None}),
        "metrics": metrics,
        "strict_gate_pass": bool(eligible),
        "strict_gate_reasons": reasons,
        "truth_inputs_used_at_runtime": candidate.get("truth_inputs_used_at_runtime"),
        "test_used_for_selection": False,
    }
    report["fingerprint"] = _fingerprint(report)
    out = Path(_require(args.output, "ALGORITHM_DAG_OUTPUT_REQUIRED")); out.mkdir(parents=True, exist_ok=False)
    _write(out / "evaluation.json", report)
    return report


def _freeze(args) -> dict:
    candidate_path = Path(_require(args.candidate, "ALGORITHM_DAG_CANDIDATE_REQUIRED"))
    search_path = Path(_require(args.search_report, "ALGORITHM_DAG_SEARCH_REPORT_REQUIRED"))
    config_path = Path(_require(args.config, "ALGORITHM_DAG_CONFIG_REQUIRED"))
    seed = int(_require(args.blind_seed, "ALGORITHM_DAG_BLIND_SEED_REQUIRED"))
    if seed <= 0:
        raise ValueError("ALGORITHM_DAG_BLIND_SEED")
    campaign_id = str(_require(args.campaign_id, "ALGORITHM_DAG_CAMPAIGN_ID_REQUIRED"))
    env_paths = _named_paths(args.environment, reason="ALGORITHM_DAG_ENVIRONMENT_NAME_PATH")
    if not env_paths:
        raise ValueError("ALGORITHM_DAG_DEVELOPMENT_ENVIRONMENTS_REQUIRED")
    bind_paths = _named_paths(args.bind_file, reason="ALGORITHM_DAG_BIND_NAME_PATH")
    candidate = _read(candidate_path); report = _read(search_path); config = _read(config_path)
    if config.get("data_role") != "development" or bool(config.get("retrospective_replay")):
        raise ValueError("ALGORITHM_DAG_FREEZE_REQUIRES_FRESH_DEVELOPMENT")
    if report.get("qualification_eligible") is not True or report.get("chosen_is_eligible") is False:
        raise ValueError("ALGORITHM_DAG_FREEZE_REQUIRES_ELIGIBLE_SEARCH")
    chosen = report.get("chosen_candidate")
    if not isinstance(chosen, dict) or chosen.get("fingerprint") != candidate.get("fingerprint"):
        raise ValueError("ALGORITHM_DAG_FREEZE_CANDIDATE_MISMATCH")
    # A reserved blind seed must not already occur in a development source-case ID.
    seed_text = str(seed)
    for path in env_paths.values():
        for row in _records(path):
            if seed_text in str(row.get("source_case", "")):
                raise ValueError("ALGORITHM_DAG_BLIND_SEED_ALREADY_PRESENT")
    payload = {
        "schema": "phm.algorithm-dag-preblind-freeze.v1",
        "status": "FROZEN",
        "blind_seed": seed,
        "blind_campaign_id": campaign_id,
        "claim_boundary": "candidate selected from development environments only; reserved blind not used for selection",
        "candidate_fingerprint": candidate.get("fingerprint"),
        "selection_report_fingerprint": report.get("fingerprint"),
        "files": {
            "candidate": {"path": str(candidate_path), "sha256": _sha(candidate_path)},
            "search_report": {"path": str(search_path), "sha256": _sha(search_path)},
            "search_config": {"path": str(config_path), "sha256": _sha(config_path)},
            **{name: {"path": str(path), "sha256": _sha(path)} for name, path in bind_paths.items()},
        },
        "development_environments": {name: {"path": str(path), "sha256": _sha(path)} for name, path in env_paths.items()},
        "test_used_for_selection": False,
        "post_blind_changes_permitted_for_this_qualification": False,
    }
    payload["fingerprint"] = _fingerprint(payload)
    out = Path(_require(args.output, "ALGORITHM_DAG_OUTPUT_REQUIRED")); out.mkdir(parents=True, exist_ok=False)
    _write(out / "preblind_freeze.json", payload)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(prog="phm-design-algorithm-dag")
    p.add_argument("--mode", choices=("single", "multi", "evaluate", "freeze"), default="single")
    p.add_argument("--records", help="JSON list or evaluation JSON containing a predictions list")
    p.add_argument("--config")
    p.add_argument("--output")
    p.add_argument("--environment", action="append", default=[], help="repeat NAME=records.json for multi search/freeze")
    p.add_argument("--search-kind", choices=("generic", "physical-rule-tree"), default="generic")
    p.add_argument("--candidate")
    p.add_argument("--search-report")
    p.add_argument("--blind-seed", type=int)
    p.add_argument("--campaign-id")
    p.add_argument("--bind-file", action="append", default=[], help="repeat NAME=path to bind source/model/provenance files")
    a = p.parse_args()
    if a.mode == "single":
        report = _single(a)
        summary = {
            "status": report["status"], "candidate_count": report["candidate_count"],
            "chosen_grammar": report.get("chosen_grammar"), "chosen_metrics": report.get("chosen_metrics"),
            "qualification_eligible": report.get("qualification_eligible"), "claim_status": report.get("claim_status"),
        }
    elif a.mode == "multi":
        report = _multi(a)
        summary = {
            "status": report["status"], "candidate_count": report["candidate_count"],
            "eligible_candidate_count": report.get("eligible_candidate_count"),
            "chosen_grammar": report.get("chosen_grammar"),
            "qualification_eligible": report.get("qualification_eligible"),
            "test_or_holdout_used_for_selection": report.get("test_or_holdout_used_for_selection", False),
        }
    elif a.mode == "evaluate":
        report = _evaluate(a)
        summary = {"status": report["status"], "strict_gate_pass": report["strict_gate_pass"], "metrics": report["metrics"]}
    else:
        report = _freeze(a)
        summary = {"status": report["status"], "blind_seed": report["blind_seed"], "fingerprint": report["fingerprint"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
