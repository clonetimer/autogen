#!/usr/bin/env bash
set -Eeuo pipefail
R=/mnt/data/PHM_R60_LIMITED_15_20261011
D=/mnt/data/PHM_R60_RELEASE_CLOSURE_20261011
PROD=$R/product/phm-design-0.7.0
DIST=$PROD/dist/phm_design-0.7.0
mkdir -p "$D/evidence" "$D/isolated_work"
for n in 311 312; do
  RUNTIME=/mnt/data/r57_work/runtime/cp$n/python/bin/python3.${n: -2}
  WH=/mnt/data/r57_work/deps$([ "$n" = 312 ] && echo 312 || true)
  for FORM in wheel sdist; do
    NAME=${n}_${FORM}; VENV=$D/isolated_work/$NAME
    if [ -e "$D/evidence/$NAME.status" ]; then echo "EXISTS_NOT_RERUN $NAME";continue;fi
    echo "BEGIN_STABLE $NAME $(date -u +'%FT%TZ')" | tee -a "$D/STABLE_MATRIX.log"
    "$RUNTIME" -m venv "$VENV"
    PY="$VENV/bin/python"
    if [ "$FORM" = wheel ];then PKG="${DIST}-py3-none-any.whl[dev,simulation,automl]";else PKG="${DIST}.tar.gz[dev,simulation,automl]";fi
    "$PY" -m pip install --no-index --find-links="$WH" --disable-pip-version-check --no-cache-dir "$PKG" > "$D/evidence/${NAME}_INSTALL.log" 2>&1
    "$PY" -m pip check > "$D/evidence/${NAME}_PIP_CHECK.log" 2>&1
    "$PY" - <<'PYCHECK' > "$D/evidence/${NAME}_PACKAGE_CHECK.json"
import json,sys,importlib.metadata as m
from phm_design.platform_limited import contract,model_manifest,open_bundled,allowed_use_case,_checked_release_certificate
import Basilisk
assert m.version('phm-design')=='0.7.0'
c=contract();assert len(c['proposed_certification_cells'])==15 and len(c['out_of_claim_unqualified_cells'])==8
assert c['release_tier']=='STABLE' and c['formal_stable'].startswith('0.7.0') and c['previous_formal_stable'].startswith('0.6.0')
cert=_checked_release_certificate();assert cert['calibration_passed']==cert['sealed_passed']==15
assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
assert set(model_manifest()['models'])=={'ADCS','EPS','Data','EPS_localization','Data_localization'}
for dom in model_manifest()['models']:open_bundled(dom)
print(json.dumps({'runtime':sys.version,'product':m.version('phm-design'),'source':__import__('phm_design').__file__,
 'contract':c['contract_id'],'models':5,'inherited_scientific_site':False,'basilisk':'IMPORT_PASS','tier':c['release_tier']}))
PYCHECK
    "$PY" - <<'PYSMOKE' > "$D/evidence/${NAME}_BASILISK_SMOKE.json"
import json
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft
sim=SimulationBaseClass.SimBaseClass()
proc=sim.CreateNewProcess('phm_stable_smoke_process')
proc.addTask(sim.CreateNewTask('phm_stable_smoke_task',macros.sec2nano(0.1)))
vehicle=spacecraft.Spacecraft();vehicle.ModelTag='stable_qualification_vehicle';vehicle.hub.mHub=10.0
sim.AddModelToTask('phm_stable_smoke_task',vehicle)
sim.InitializeSimulation();sim.ConfigureStopTime(macros.sec2nano(0.3));sim.ExecuteSimulation()
print(json.dumps({'module':'Basilisk.simulation.spacecraft','runtime_nanos':int(sim.TotalSim.CurrentNanos),'status':'PASS'}))
PYSMOKE
    (
     cd "$D/STABLE_TESTBED"
     PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest tests/phm_design -q --disable-warnings -rs -p no:cacheprovider
    ) > "$D/evidence/${NAME}_FULL_TEST.log" 2>&1
    if grep -E '([1-9][0-9]* failed|[1-9][0-9]* skipped|[1-9][0-9]* errors)' "$D/evidence/${NAME}_FULL_TEST.log";then echo "NONZERO_BAD_TEST $NAME";exit 3;fi
    grep -E '^[0-9]+ passed,.*warnings in' "$D/evidence/${NAME}_FULL_TEST.log" | tail -1
    (
     cd "$D"
     OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" stable_operations.py run --cell "$NAME"
    ) > "$D/evidence/${NAME}_OPERATIONS.log" 2>&1
    if [ "$NAME" = 312_sdist ]; then
      # Verify rollback from the newly installed Stable to the exact frozen 0.6.0 wheel.
      STABLE_DIST="$DIST-py3-none-any.whl" OLD="$R/rollback/phm_design-0.6.0-py3-none-any.whl" "$PY" - <<'ROLLPRE' > "$D/evidence/${NAME}_ROLLBACK_PRE.json"
from pathlib import Path
import os,json,hashlib,importlib.metadata as m
p=Path(os.environ['OLD']);h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='91c0501f03d88dc92d20d3a0e33dd556dd833d250f1a1027aa7dfa34a01e3421'
assert m.version('phm-design')=='0.7.0'
print(json.dumps({'source_version':'0.7.0','target_version':'0.6.0','official_prior_stable_sha256':h,'status':'PRECHECK_PASS'}))
ROLLPRE
      "$PY" -m pip uninstall -y phm-design > "$D/evidence/${NAME}_ROLLBACK_UNINSTALL.log" 2>&1
      "$PY" -m pip install --no-index --no-deps "$R/rollback/phm_design-0.6.0-py3-none-any.whl" > "$D/evidence/${NAME}_ROLLBACK_INSTALL.log" 2>&1
      "$PY" -m pip check > "$D/evidence/${NAME}_ROLLBACK_PIP_CHECK.log" 2>&1
      "$PY" - <<'ROLLCHECK' > "$D/evidence/${NAME}_ROLLBACK_CHECK.json"
import json,importlib.metadata as m,phm_design
assert m.version('phm-design')=='0.6.0' and phm_design.__version__=='0.6.0'
from phm_design import __version__
print(json.dumps({'rollback_version':__version__,'status':'PASS'}))
ROLLCHECK
    fi
    printf 'PASS\n' > "$D/evidence/$NAME.status"
    echo "PASS_STABLE $NAME $(tail -1 "$D/evidence/${NAME}_FULL_TEST.log")" | tee -a "$D/STABLE_MATRIX.log"
    rm -rf "$VENV" # preserve build/test/operation/rollback logs; clean env was verified independently
    df -h /mnt/data | tail -1 | tee -a "$D/STABLE_MATRIX.log"
  done
done
