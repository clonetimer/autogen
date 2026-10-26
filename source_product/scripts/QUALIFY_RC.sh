#!/usr/bin/env bash
# HISTORICAL COMPATIBILITY SCRIPT: OPTIONAL REAL PRIVATE torch_like acceptance ONLY.
# 0.8.0 Stable Core does NOT require native private torch_like acceptance.
# Invoke manually only when validating the optional provider integration after installation.
# A broader 0.8.0 release gate, including complete regression tests,
# lives in .github/workflows/stable-candidate-080.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/scripts/stable_candidate_smoke.py"
