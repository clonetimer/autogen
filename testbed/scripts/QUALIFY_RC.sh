#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DEPS="${PHM_RUNTIME_DEPS:-/mnt/data/work/runtime_deps}"
OUT="${PHM_RC_QUAL_OUT:-$ROOT/.rc_qualification}"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH="$ROOT/src:$RUNTIME_DEPS" python - <<'PY' > "$OUT/import_smoke.txt"
import phm_design
assert phm_design.__version__ == '0.6.0rc1', phm_design.__version__
print(phm_design.__version__)
PY
PYTHONPATH="$ROOT/src:$RUNTIME_DEPS" python -m phm_design --help > "$OUT/cli_help.txt"
PYTHONPATH="$ROOT/src:$RUNTIME_DEPS" pytest -q "$ROOT/tests" > "$OUT/pytest.log"
python -m pip install --no-deps --no-build-isolation --target "$OUT/install" "$ROOT" > "$OUT/pip_install.log" 2>&1
PYTHONPATH="$OUT/install:$RUNTIME_DEPS" python - <<'PY' > "$OUT/installed_import.txt"
import phm_design
assert phm_design.__version__ == '0.6.0rc1'
print(phm_design.__version__)
PY
printf 'PASS\n' > "$OUT/status.txt"
