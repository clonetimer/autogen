"""R95 operator CLI for frozen diagnostic DESIGN and SOURCE-TRAIN research.

The CLI intentionally has no protected VALIDATION/FINAL input argument.
Actual estimator execution is delegated to the installed trusted torch_like
provider; no model persistence or algorithm library is reimplemented.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .__main__ import _read
from .design_engine import (
    DesignRejected, propose_diagnostic_study, execute_source_design_study)


def _save_unique(path, document):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False,
                  sort_keys=True, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="autogen R95: automatic diagnosis design, not deployment.")
    cmd = parser.add_subparsers(dest="phase", required=True)
    proposal = cmd.add_parser("propose", help="TRAIN-only design space, no fits")
    evaluation = cmd.add_parser("source-study",
                                help="source TRAIN internal OOF; not external VAL")
    for p in (proposal, evaluation):
        p.add_argument("--equipment", required=True)
        p.add_argument("--train-manifest", required=True)
        p.add_argument("--trial", required=True)
        p.add_argument("--output", required=True)
    evaluation.add_argument("--source-features", required=True)
    evaluation.add_argument("--frozen-study", required=True)
    evaluation.add_argument("--expected-study-sha256", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("R95_REFUSE_TO_OVERWRITE_DESIGN_EVIDENCE")
    try:
        equipment = _read(args.equipment)
        train = _read(args.train_manifest)
        trial = _read(args.trial)
        if args.phase == "propose":
            result = propose_diagnostic_study(equipment, train, trial)
        else:
            if (type(args.expected_study_sha256) is not str
                    or len(args.expected_study_sha256) != 64
                    or any(x not in "0123456789abcdef"
                           for x in args.expected_study_sha256)):
                raise DesignRejected("R95_ARCHIVED_STUDY_SHA256_REQUIRED")
            study = _read(args.frozen_study)
            if study.get("study_sha256") != args.expected_study_sha256:
                raise DesignRejected("R95_ARCHIVED_STUDY_DIGEST_CHANGED")
            result = execute_source_design_study(
                study, equipment, train, trial, _read(args.source_features))
        _save_unique(args.output, result)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "artifact": args.output,
        "phase": args.phase,
        "source_only": True,
        "independent_model_validation": False,
        "industrial_production_approved": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
