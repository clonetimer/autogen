#!/usr/bin/env python3
"""Read-only evidence-locked reviewer for PHM AutoDesign strategic direction.

Not an independently trained LLM, independent human panel or live sealed-test
steward. It is a separately executable adversarial audit agent: immutable
source input, bounded evidence policy, repeatable machine findings, no writes
to reviewed target and no model-training side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


class ReviewIntegrityError(RuntimeError):
    pass


def git_blob_sha1(contents: bytes) -> str:
    """Exact Git blob object identity, not SHA256 of a display string."""
    return hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\x00" + contents).hexdigest()


def validate_source_lock(root: Path, lock: dict, *, check_git_head: bool = True) -> dict[str, str]:
    """Reject a changed, incomplete or differently checked-out reviewed source.

    No raw fault trajectories or protected FINAL datasets are accessed.
    """
    root = root.resolve(strict=True)
    if not isinstance(lock, dict) or lock.get("schema") != "phm.independent-review.source-lock.v1":
        raise ReviewIntegrityError("SOURCE_LOCK_SCHEMA")
    head = lock.get("target_commit")
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ReviewIntegrityError("INVALID_TARGET_COMMIT")
    if check_git_head:
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.PIPE, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReviewIntegrityError("UNVERIFIED_GIT_HEAD") from exc
        if actual != head:
            raise ReviewIntegrityError("TARGET_COMMIT_MISMATCH")
    files = lock.get("files")
    if not isinstance(files, list) or len(files) < 10:
        raise ReviewIntegrityError("INSUFFICIENT_LOCKED_SOURCE_EVIDENCE")
    seen = set()
    sources = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "blob_sha1"}:
            raise ReviewIntegrityError("MALFORMED_EVIDENCE_ENTRY")
        name, expected = entry["path"], entry["blob_sha1"]
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ReviewIntegrityError("MALFORMED_EVIDENCE_IDENTITY")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or name in seen
                or re.fullmatch("[0-9a-f]{40}", expected) is None):
            raise ReviewIntegrityError("UNSAFE_OR_DUPLICATE_EVIDENCE_PATH")
        seen.add(name)
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ReviewIntegrityError("EVIDENCE_OUTSIDE_TARGET_OR_MISSING:" + name)
        data = candidate.read_bytes()
        if git_blob_sha1(data) != expected:
            raise ReviewIntegrityError("SOURCE_BLOB_CHANGED:" + name)
        try:
            sources[name] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReviewIntegrityError("NOT_UTF8_SOURCE:" + name) from exc
    return sources


def _evidence_url(sha: str, relative: str) -> str:
    return f"https://github.com/clonetimer/autogen/blob/{sha}/{relative}"


def assess(lock: dict, sources: dict[str, str]) -> dict:
    """Dispute-friendly source observations; no inference from test labels."""
    def source(key: str) -> str:
        path = "source_product/" + key if not key.startswith("RELEASE") else key
        if path not in sources:
            raise ReviewIntegrityError("REQUIRED_EVIDENCE_NOT_LOCKED:" + path)
        return sources[path]

    contract = json.loads(source("src/phm_design/limited_models/scope_contract.json"))
    ontology = source("src/phm_design/task_architectures.py")
    graph = source("src/phm_design/task_graph.py")
    registry = source("src/phm_design/task_candidate_registry.py")
    generic = source("src/phm_design/open_design_fit.py")
    domains = source("src/phm_design/platform_features.py")
    neural = source("src/phm_design/neural_domain_generalization.py")
    provenance = source("src/phm_design/leakage_control.py")
    r78 = source("README_R78.md")
    stable = source("RELEASE_DECISION_R60.md")
    retrospective = source("docs/DIAGNOSIS_DESIGN_TESTING_LESSONS_AND_LEAKAGE_PREVENTION.md")
    vals = {
        "stable_limited_satellite_scope": (
            contract.get("release_tier") == "STABLE"
            and "Basilisk" in contract.get("release_domain", "")
            and len(contract.get("proposed_certification_cells", [])) == 15
            and "离线" in stable
        ),
        "fixed_five_class_task_graph": (
            "len(self.labels)!=5" in ontology and "len(labels)!=5" in ontology
            and "UNSUPPORTED_TASK_ONTOLOGY" in ontology
        ),
        "satellite_specific_schema": all(
            f"'{domain}':" in domains
            for domain in ("ADCS", "EPS", "Thermal", "Data")
        ),
        "generic_design_path_exists": (
            "def design(examples,spec,classes,healthy_label" in generic
            and "FeedbackPlanner" in generic
        ),
        "neural_design_profile_bounded": (
            "def _auto_neural_candidate_specs(" in graph
            and "compact_domain_generalization" in graph
            and "'detection':" in graph and "'localization':" in graph
            and "'mechanism':" in graph
        ),
        "trainable_neural_objectives": (
            "class DGSequenceEstimator(" in neural
            and "class RiskExtrapolationEstimator(" in neural
            and "'rex_tcn':" in registry
            and "loss.backward()" in neural
        ),
        "family_group_holdout": (
            "LeaveOneGroupOut" in graph
            and "leave_environment_family_out" in graph
            and "GRAPH_OUTER_ENVIRONMENT_LEAKAGE" in graph
        ),
        "requires_healthy_prefix": (
            "HEALTHY_PREFIX_NOT_CONFIRMED" in graph
            and "healthy_prefix_confirmed=True" not in graph
        ),
        "evaluation_protection_documented": (
            "authorize_design" in provenance
            and "FINAL_FEEDBACK_TAINTED_DESIGN" in provenance
            and "提示词不是强制门禁" in retrospective
        ),
        "r78_stress_revealed_failure": (
            "5.36%" in r78 and "1.79%" in r78
            and "100%" in r78 and "REVEALED_SYNTHETIC_DEV_STRESS" in r78
        ),
    }
    # If source evidence does not pass the expected conditions, the agent
    # records not verified rather than treating it as a passing diagnosis.
    sha = lock["target_commit"]
    entries = [
      ("D01", "P0", "Satellite validation mistaken for generic industrial design",
       ["stable_limited_satellite_scope", "satellite_specific_schema"],
       "A generic task adapter must configure an unseen equipment family without changing the AutoDesign core.",
       ["RELEASE_DECISION_R60.md", "source_product/src/phm_design/platform_features.py"]),
      ("D02", "P0", "Generic program optimizer and five-class neural task graph remain distinct",
       ["fixed_five_class_task_graph", "generic_design_path_exists", "neural_design_profile_bounded"],
       "A unified TaskSpec must generate an audited pipeline for two different task/ontology shapes.",
       ["source_product/src/phm_design/task_architectures.py", "source_product/src/phm_design/open_design_fit.py", "source_product/src/phm_design/task_graph.py"]),
      ("D03", "P0", "Neural objectives lack proven unseen-equipment robustness",
       ["r78_stress_revealed_failure", "trainable_neural_objectives"],
       "Pre-register and pass paired strong-baseline trials on independent unseen physical families.",
       ["source_product/README_R78.md", "source_product/src/phm_design/neural_domain_generalization.py"]),
      ("D04", "P0", "Offline healthy-prefix interface cannot establish causal industrial monitoring",
       ["stable_limited_satellite_scope", "requires_healthy_prefix"],
       "Run causal replay with fixed decision timestamps, lead time and missed/false alarm costs.",
       ["RELEASE_DECISION_R60.md", "source_product/src/phm_design/task_graph.py"]),
      ("D05", "P1", "Fault observability cannot be inferred from model score alone",
       ["satellite_specific_schema"],
       "Verify sensor sensitivity and sufficient fault/healthy physical separation before model search.",
       ["source_product/src/phm_design/platform_features.py"]),
      ("D06", "P1", "Metadata gates are not an independent sealed-test steward",
       ["evaluation_protection_documented"],
       "Demonstrate separate access controls, fixed model hashes and one-time FINAL scoring.",
       ["source_product/src/phm_design/leakage_control.py", "source_product/docs/DIAGNOSIS_DESIGN_TESTING_LESSONS_AND_LEAKAGE_PREVENTION.md"]),
    ]
    findings = []
    for ident, priority, title, flags, challenge, refs in entries:
        findings.append({
            "id": ident, "priority": priority, "finding": title,
            "source_support": "SUPPORTED" if all(vals[k] for k in flags) else "NOT_VERIFIED",
            "checked_observations": {name: vals[name] for name in flags},
            "falsification_test": challenge,
            "source_urls": [_evidence_url(sha, p) for p in refs],
        })
    return {
        "schema": "phm.independent-review.r79.v1",
        "reviewed_commit": sha,
        "product_under_review": "industrial_equipment_diagnostic_algorithm_autodesign",
        "current_test_adapter": "satellite_basilisk",
        "verdict": "KEEP_DIRECTION_WITH_PIVOT",
        "broad_industrial_qualification": False,
        "research_continue": True,
        "reviewer_type": "DETERMINISTIC_READ_ONLY_AGENT_NOT_INDEPENDENT_LLM",
        "not_inspected": ["sealed test labels", "real industrial equipment signals",
                          "live hardware", "private agent development feedback"],
        "observations": vals, "findings": findings,
        "unknown_is_not_pass": True,
        "no_model_training_or_final_data_opened": True,
    }


def render_markdown(report: dict) -> str:
    parts = [
        "# Independent industrial AutoDesign review — machine audit\n",
        f"- Reviewed commit: \`{report['reviewed_commit']}\`",
        f"- Verdict: **{report['verdict']}**",
        f"- Broad industrial qualification: **{report['broad_industrial_qualification']}**",
        f"- Reviewer implementation: {report['reviewer_type']}",
        "- Full human-readable adversarial analysis: \`R79_DIRECTION_REVIEW.md\`",
        "- These are source-level flags, NOT a new diagnostic accuracy test.\n",
        "| Priority | Finding | Evidence status | Falsification test |",
        "|---|---|---|---|",
    ]
    for f in report["findings"]:
        parts.append(f"| {f['priority']} | {f['finding']} | {f['source_support']} | {f['falsification_test']} |")
    parts.append("\nUnverified observations are not accepted as successful diagnostics.\n")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only independent direction review")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--lock", default=str(Path(__file__).with_name("source_lock.json")))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-git-head", action="store_true", help="For synthetic unit fixtures ONLY")
    args = parser.parse_args(argv)
    root = Path(args.source_root).resolve(strict=True)
    out = Path(args.out_dir).resolve()
    if out == root or out.is_relative_to(root):
        raise ReviewIntegrityError("REVIEW_OUTPUT_MUST_BE_OUTSIDE_TARGET")
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    files = validate_source_lock(root, lock, check_git_head=not args.skip_git_head)
    report = assess(lock, files)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (out / "review.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "findings": len(report["findings"]),
                      "verified_source_files": len(files), "target": report["reviewed_commit"],
                      "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
