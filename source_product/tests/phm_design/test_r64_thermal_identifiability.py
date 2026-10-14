"""R64 audit tests: synthetic fixtures only, no learning or qualification."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from phm_design.thermal_identifiability import (
    EXPECTED_HEADER, AuditContractError, audit_world, audit_worlds, main,
)


def _write_world(root: Path) -> Path:
    w = root / 'r62-independent-thermal-synthetic-fixture'
    w.mkdir(parents=True)
    t = np.arange(21) * 20.
    light = np.where(np.arange(21) % 5 < 3, 1., 0.)
    a_cmd = np.where(np.arange(21) % 4, 3., 4.)
    b_cmd = np.where(np.arange(21) % 3, 2., 3.)
    nominal = np.column_stack([t, light, a_cmd, b_cmd,
                               300. + .1*np.arange(21), 310. + .2*np.arange(21)])
    onset = 12
    for k in range(5):
        trace = nominal.copy()
        if k in (1, 3):
            trace[onset:, 4] += np.arange(1, 22-onset) * (1 if k == 1 else 2.)
        if k in (2, 4):
            trace[onset:, 5] += np.arange(1, 22-onset) * (1 if k == 2 else 2.)
        f = w / f'L{k}_DEPLOYMENT.csv'
        with f.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(EXPECTED_HEADER)
            writer.writerows(trace)
        (w / f'L{k}_CASE.json').write_text(json.dumps({
            'offline_label': k, 'domain':'Thermal_localization',
            'world_id': w.name, 'role':'new_development_evaluation',
            'observation_sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
            'rows':len(t)}))
        fault_active = np.where(np.arange(21) >= onset, 1, 0) if k else np.zeros(21)
        (w / f'L{k}_SIMULATOR_ONLY.json').write_text(json.dumps({
            'native_audit': True, 'truth':np.column_stack([t, fault_active]).tolist()}))
    return w


def _rehash(world, k):
    path=world / f'L{k}_DEPLOYMENT.csv'
    meta=world / f'L{k}_CASE.json'
    values=json.loads(meta.read_text())
    values['observation_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    meta.write_text(json.dumps(values))


def test_offline_mechanism_vs_localization_and_documented_claim_limit(tmp_path):
    w=_write_world(tmp_path)
    x=audit_world(w)
    assert x['labelled_fault_pair_count']==4
    assert x['prefault_byte_equivalent_numeric_rows']
    assert x['within_component_signature_similarity']['a']['pearson']==pytest.approx(1.)
    assert x['within_component_signature_similarity']['b']['pearson']==pytest.approx(1.)
    assert max(a['unaffected_component_max_abs_change_k'] for a in x['mechanism_deltas'])==0
    result=audit_worlds(tmp_path)
    assert result['world_count']==1
    assert result['new_worlds_generated']==0
    assert result['diagnostic_qualification']=='NOT_EVALUATED'
    assert result['oracle_nominal_and_simulator_fault_onset_used'] is True
    assert result['training_or_model_selection_performed'] is False


def test_detect_observation_tamper_by_sha(tmp_path):
    w=_write_world(tmp_path)
    p=w/'L1_DEPLOYMENT.csv'
    p.write_text(p.read_text().replace('300.0', '303.0', 1))
    with pytest.raises(AuditContractError,match='SHA256'):
        audit_world(w)


def test_reject_pair_mismatch_even_if_csv_rehashed(tmp_path):
    w=_write_world(tmp_path)
    p=w/'L1_DEPLOYMENT.csv'
    s=p.read_text().splitlines()
    row=s[4].split(',')
    row[4]=str(float(row[4])+1.)
    s[4]=','.join(row)
    p.write_text('\n'.join(s)+'\n')
    _rehash(w,1)
    with pytest.raises(AuditContractError,match='PREFAULT_PAIR_MISMATCH'):
        audit_world(w)


def test_reject_inconsistent_truth_onsets(tmp_path):
    w=_write_world(tmp_path)
    p=w/'L2_SIMULATOR_ONLY.json'
    x=json.loads(p.read_text())
    x['truth'][13][1]=0
    p.write_text(json.dumps(x))
    with pytest.raises(AuditContractError,match='PAIR_ONSET_NOT_EQUAL'):
        audit_world(w)


def test_reject_extra_deployment_field_even_with_matching_hash(tmp_path):
    w=_write_world(tmp_path)
    p=w/'L3_DEPLOYMENT.csv'
    text=p.read_text().replace('temperature_a_k', 'simulator_truth_temperature_a_k')
    p.write_text(text)
    _rehash(w,3)
    with pytest.raises(AuditContractError,match='DEPLOYMENT_FIELDS_NOT_EXACT'):
        audit_world(w)


def test_cli_immutable_output(tmp_path):
    _write_world(tmp_path)
    out=tmp_path/'report.json'
    assert main(['--native-root',str(tmp_path), '--output',str(out)])==0
    before=out.read_bytes()
    assert json.loads(before)['world_count']==1
    with pytest.raises(SystemExit):
        main(['--native-root',str(tmp_path), '--output',str(out)])
    assert out.read_bytes()==before
