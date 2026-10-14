#!/usr/bin/env bash
set -Eeuo pipefail
R=/mnt/data/PHM_R60_LIMITED_15_20261011
TIER="$1"
if [ "$TIER" = RC ]; then VER=0.7.0rc1; else VER=0.7.0; fi
PROD="$R/product/phm-design-$VER"
ART="$PROD/dist/phm_design-$VER"
ROOT="$R/release_validation/$TIER";mkdir -p "$ROOT/evidence" "$ROOT/clean_envs"
# Fixtures copy is not copied into the installed site; executes without source path.
if [ ! -d "$ROOT/testbed" ]; then
 mkdir -p "$ROOT/testbed"
 cp -a "$PROD/tests" "$ROOT/testbed/"
 cp -a "$PROD/configs" "$ROOT/testbed/"
 cp -a "$PROD/scripts" "$ROOT/testbed/"
 cp "$PROD/paired_observation_audit.py" "$ROOT/testbed/"
 echo -e '[pytest]\ntestpaths = tests/phm_design' > "$ROOT/testbed/pytest.ini"
fi
for n in 311 312;do
 RUNTIME=/mnt/data/r57_work/runtime/cp$n/python/bin/python3.${n: -2}
 WH=/mnt/data/r57_work/deps$([ "$n" = 312 ] && echo 312 || true)
 for FORM in wheel sdist;do
  name=${n}_${FORM};envpath=$ROOT/clean_envs/$name
  echo "BEGIN $TIER $name $(date -u +'%FT%TZ')"
  "$RUNTIME" -m venv "$envpath"
  PY=$envpath/bin/python
  if [ "$FORM" = wheel ];then DIST="${ART}-py3-none-any.whl[dev,simulation,automl]";else DIST="${ART}.tar.gz[dev,simulation,automl]";fi
  "$PY" -m pip install --no-index --find-links="$WH" --disable-pip-version-check --no-cache-dir "$DIST" > "$ROOT/evidence/${name}_INSTALL.log" 2>&1
  "$PY" -m pip check > "$ROOT/evidence/${name}_PIP_CHECK.log" 2>&1
  "$PY" - <<'PYCHECK' > "$ROOT/evidence/${name}_PACKAGE_CHECK.json"
import json,sys,importlib.metadata as m
from phm_design.platform_limited import contract,model_manifest,open_bundled,allowed_use_case,_checked_release_certificate
import Basilisk
assert m.version('phm-design') in ('0.7.0rc1','0.7.0')
c=contract();assert len(c['proposed_certification_cells'])==15
assert len(c['out_of_claim_unqualified_cells'])==8
assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
assert set(model_manifest()['models'])=={'ADCS','EPS','Data','EPS_localization','Data_localization'}
cert=_checked_release_certificate();assert cert['sealed_passed']==15
for dom in model_manifest()['models']:open_bundled(dom)
print(json.dumps({'runtime':sys.version,'product':m.version('phm-design'),'source':__import__('phm_design').__file__,'contract':c['contract_id'],'models':5,'inherited_scientific_site':False,'basilisk':'PASS','tier':c['release_tier']}))
PYCHECK
  cd "$ROOT/testbed"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest tests/phm_design -q --disable-warnings -rs -p no:cacheprovider > "$ROOT/evidence/${name}_FULL_TEST.log" 2>&1
  echo PASS > "$ROOT/evidence/${name}.status"
  echo "PASS $TIER $name $(tail -1 "$ROOT/evidence/${name}_FULL_TEST.log")"
 done
done
