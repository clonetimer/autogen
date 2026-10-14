"""Evidence-bound diagnosis-design next-study CLI (research only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .design_readiness import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit development evidence and suggest research studies (never promote release).")
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text(encoding="utf-8"))
    report = analyze(data, source_root=args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(f"{report['release_decision']}: {len(report['evidence_gaps'])} evidence gaps; saved {args.output}")


if __name__ == "__main__":
    main()
