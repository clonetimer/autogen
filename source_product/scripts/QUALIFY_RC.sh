#!/usr/bin/env bash
# Compatibility shim for legacy QUALIFY_RC.sh entry point.
# Run ONLY after installing the candidate WHEEL and pinned REAL torch_like.
# A broader 0.8.0rc1 release gate, including complete regression tests,
# lives in .github/workflows/stable-candidate-080.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/scripts/stable_candidate_smoke.py"
