"""Pure contract compilation CLI. Never fits a model or reads target data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .taskspec import TaskSpecRejected, compile_task_spec


def _reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise TaskSpecRejected("DUPLICATE_JSON_KEY", key)
        out[key] = value
    return out


def _read_contract(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=_reject_duplicate_keys)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R80 industrial diagnostic TaskSpec compiler")
    parser.add_argument("--spec", type=Path, required=True, help="TaskSpec JSON")
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="TRAIN-only group/class provenance JSON list, never FINAL")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output plan JSON path; default stdout")
    args = parser.parse_args(argv)
    try:
        spec = _read_contract(args.spec)
        records = (_read_contract(args.train_manifest)
                   if args.train_manifest is not None else None)
        plan = compile_task_spec(spec, train_manifest=records)
    except (TaskSpecRejected, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "TASKSPEC_REJECTED", "reason": str(exc),
                          "trained": False, "qualified": False}, sort_keys=True),
              file=sys.stderr)
        return 2
    text = json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # No unsafe construction or execution based on fields in TaskSpec.
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
