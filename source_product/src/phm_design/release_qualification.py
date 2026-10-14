"""Evidence aggregation without relabeling a known capability gap as a PASS.

This function is a release-policy decision aid, not an authenticity verifier or
a diagnosis rule. Evidence providers must separately verify their source files.
"""
from __future__ import annotations

LEGACY_CELLS=tuple(f'{d}:{t}' for d in ('ADCS','EPS','Thermal','Data') for t in ('cross_scenario','weak_fault','progressive_degradation'))

def assess_release(evidence,*,accepted_limitations=()):
    allowed=set(accepted_limitations)
    if allowed-{'ADCS:weak_fault'}:raise ValueError('LIMITATION_NOT_AUTHORIZED_BY_CURRENT_RELEASE_SCOPE')
    cells=evidence.get('capabilities',{})
    missing=[c for c in LEGACY_CELLS if c not in cells];blockers=[]
    if missing:blockers.append({'gate':'scope','missing':missing})
    for key,item in cells.items():
        if key in allowed:
            # Keep every metric and failed status in the support table.
            if item.get('status')!='SUPPORTED':continue
        if item.get('status')!='SUPPORTED':blockers.append({'gate':'diagnostic_capability','cell':key,'status':item.get('status','MISSING')})
    for version in ('3.11','3.12','3.13'):
        e=evidence.get('runtime_matrix',{}).get(version,{})
        for gate in ('clean_wheel_install','clean_sdist_install','full_regression_no_native_skips'):
            if e.get(gate) is not True:blockers.append({'gate':gate,'python':version})
    for gate in ('source_full_regression','installed_full_regression','integrity_verified','diagnostic_qualification','independent_calibration','sealed_qualification'):
        if evidence.get(gate) is not True:blockers.append({'gate':gate})
    return {'schema':'phm.release-qualification.v1','decision':'ELIGIBLE_FOR_REVIEW' if not blockers else 'HOLD_DO_NOT_PROMOTE',
        'accepted_limitations':sorted(allowed),'known_limitations_count_as_pass':False,'legacy_12_cells_retained':not missing,
        'capabilities':cells,'blockers':blockers,'historical_hold_records_must_be_preserved':True,'in_orbit_qualification':False}
