"""R93 minimal CLI to audit trusted manifests or compile TRAIN-only designs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .contracts import validate_dataset, compile_source_plan, ContractRejected


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="P1 industrial metadata contracts; no model is trained.")
    p.add_argument("command", choices=("audit", "source-plan"))
    p.add_argument("--equipment", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    destination = Path(a.output)
    if destination.exists():
        p.error("R93_REFUSE_TO_OVERWRITE_EXISTING_AUDIT")
    equipment, dataset, trial = (_read(a.equipment),
                                 _read(a.dataset), _read(a.trial))
    try:
        if a.command == "audit":
            result = validate_dataset(dataset, equipment, trial)
        else:
            result = compile_source_plan(equipment, dataset, trial)
    except ContractRejected as exc:
        p.error(str(exc))
    with destination.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        out.write("\n")
    print(f"{result['schema']} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
