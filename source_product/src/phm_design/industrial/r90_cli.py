"""Windows/Linux command-line entry for R90 preregistered Paderborn pilot.

Run explicitly separated stages and write new-only JSON receipts; never
auto-read VALIDATION on source freeze, and never fit during review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .r86_neuro_fuzzy import FuzzyRulePolicy
from .r90_paderborn_pilot import (
    freeze_paderborn_source, evaluate_paderborn_validation)


def _receipt(data, destination: str):
    p = Path(destination)
    # Fail closed rather than overwrite an archived receipt.
    with p.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True,
                  allow_nan=False, indent=2)
        f.write("\n")


def _parser():
    parser = argparse.ArgumentParser(
        description="R90 research-only Paderborn heldout cohort pilot; NEVER Stable")
    sub = parser.add_subparsers(dest="stage", required=True)
    first = sub.add_parser("freeze", help="TRAIN only, do not read VALIDATION")
    first.add_argument("--raw-root", required=True)
    first.add_argument("--frozen-dir", required=True)
    first.add_argument("--task-spec", required=True)
    first.add_argument("--receipt", required=True)
    first.add_argument("--max-rules", type=int, default=2)
    first.add_argument("--premise-steps", type=int, default=2)

    review = sub.add_parser("review", help="frozen VALIDATION only; no refitting")
    review.add_argument("--raw-root", required=True)
    review.add_argument("--frozen-dir", required=True)
    review.add_argument("--manifest-sha", required=True)
    review.add_argument("--receipt", required=True)
    return parser


def main(args=None):
    opt = _parser().parse_args(args)
    if Path(opt.receipt).exists():
        _parser().error("receipt already exists: append a new version filename")
    if opt.stage == "freeze":
        spec = json.loads(Path(opt.task_spec).read_text(encoding="utf-8"))
        result = freeze_paderborn_source(
            spec, opt.raw_root, opt.frozen_dir,
            policy=FuzzyRulePolicy(max_rules=opt.max_rules,
                                   premise_steps=opt.premise_steps))
    else:
        result = evaluate_paderborn_validation(
            opt.frozen_dir, opt.raw_root,
            expected_manifest_sha256=opt.manifest_sha)
    _receipt(result, opt.receipt)
    print(json.dumps({
        "stage": opt.stage, "receipt": opt.receipt,
        "stable_qualified": False,
        "independent_generalization_proven": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
