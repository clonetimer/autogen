#!/usr/bin/env bash
set -Eeuo pipefail
R=/mnt/data/PHM_R60_LIMITED_15_20261011
ART=$R/product/phm-design-0.7.0.dev13/dist/phm_design-0.7.0.dev13
mkdir -p "$R/evidence" "$R/clean_envs"
for ver in 311 312; do
  BIN=/mnt/data/r57_work/runtime/cp$ver/python/bin/python3.${ver: -2}
  DEPS=/mnt/data/r57_work/deps$([ "$ver" = 312 ] && echo 312 || true)
  for form in wheel sdist; do
    NAME=${ver}_${form};VENV=$R/clean_envs/$NAME
    echo "BEGIN $NAME $(date -u +'%FT%TZ')"
    "$BIN" -m venv "$VENV"
    PY="$VENV/bin/python"
    if [ "$form" = wheel ]; then PKG="${ART}-py3-none-any.whl[dev,simulation,automl]"; else PKG="${ART}.tar.gz[dev,simulation,automl]"; fi
    "$PY" -m pip install --no-index --find-links="$DEPS" --disable-pip-version-check --no-cache-dir "$PKG" > "$R/evidence/${NAME}_INSTALL.log" 2>&1
    "$PY" -m pip check > "$R/evidence/${NAME}_PIP_CHECK.log" 2>&1
    "$PY" - <<'PY' > "$R/evidence/${NAME}_PACKAGE_CHECK.json"
import json,sys,importlib.metadata as m
from phm_design.platform_limited import contract,model_manifest,open_bundled,allowed_use_case
import Basilisk
assert m.version('phm-design')=='0.7.0.dev13'
c=contract();assert len(c['proposed_certification_cells'])==15
assert len(c['out_of_claim_unqualified_cells'])==8
assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
assert set(model_manifest()['models'])=={'ADCS','EPS','Data','EPS_localization','Data_localization'}
for dom in model_manifest()['models']:open_bundled(dom)
print(json.dumps({'python':sys.version,'product':m.version('phm-design'),'source':__import__('phm_design').__file__,'contract':c['contract_id'],'models':5,'inherited_scientific_site':False,'basilisk':'PASS'}))
PY
    cd "$R/testbed"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest tests/phm_design -q --disable-warnings -rs -p no:cacheprovider > "$R/evidence/${NAME}_FULL_TEST.log" 2>&1
    echo PASS > "$R/evidence/${NAME}.status"
    echo "PASS $NAME $(tail -1 "$R/evidence/${NAME}_FULL_TEST.log")"
  done
done
