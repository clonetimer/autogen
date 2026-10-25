"""R94 portable, new-only CLI: TRAIN fit and unlabeled model replay.

No search, model-selection via VALIDATION, or FINAL access is provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .__main__ import _read
from .candidate_executor import CandidateAdapter, FAMILIES, load_model


def _write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2,
                  ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="R94 research-only unified candidate fit/replay; no VAL.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    for name in ("equipment", "dataset", "trial", "features", "family",
                 "model", "resource"):
        fit.add_argument("--" + name, required=True)
    replay = commands.add_parser("predict")
    for name in ("model", "observations", "output"):
        replay.add_argument("--" + name, required=True)
    a = parser.parse_args(argv)
    try:
        if a.command == "fit":
            if a.family not in FAMILIES:
                parser.error("R94_UNIMPLEMENTED_EXECUTOR_FAMILY")
            if a.model == a.resource or Path(a.model).exists() or Path(a.resource).exists():
                parser.error("R94_REFUSE_TO_OVERWRITE_TRAINING_OUTPUT")
            m = CandidateAdapter(a.family).fit(
                _read(a.equipment), _read(a.dataset), _read(a.trial),
                _read(a.features))
            _write_new(a.model, m.export_model())
            _write_new(a.resource, m.resource_receipt())
            print(json.dumps({"model_path": a.model, "resource_path": a.resource,
                              "model_sha256": m.export_model()["sha256"],
                              "qualified": False}, sort_keys=True))
        else:
            if Path(a.output).exists():
                parser.error("R94_REFUSE_TO_OVERWRITE_INFERENCE_OUTPUT")
            m = load_model(_read(a.model))
            results = m.predict(_read(a.observations))
            _write_new(a.output, {
                "schema": "phm.autogen.candidate-inference.v1",
                "model_sha256": m.export_model()["sha256"],
                "predictions": results,
                "independent_validation_performed": False,
                "industrial_production_approved": False,
            })
            print(json.dumps({"output": a.output, "qualified": False}))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
