"""Verify R65 dev-only native pilot archive; no simulation or inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_HEADER=['time_s','illumination','heat_a_command_w','heat_b_command_w','temperature_a_k','temperature_b_k']


def verify(root:Path)->dict:
    m=json.loads((root/'MANIFEST_SHA256.json').read_text())
    if m.get('schema')!='phm.r65.dev-evidence-manifest.v1':raise ValueError('BAD_MANIFEST')
    files=m['files']; expected=set(files)|{'MANIFEST_SHA256.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError(f'FILE_INVENTORY_DIFFERENT: missing={sorted(expected-actual)} extra={sorted(actual-expected)}')
    for name,digest in files.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('SHA256_MISMATCH:'+name)
    protocol=json.loads((root/'PROTOCOL_FROZEN_COPY.json').read_text())
    from run_r65_dev_experiment import validate_protocol
    validate_protocol(protocol)
    result=json.loads((root/'R65_DEV_RESULTS.json').read_text())
    if (result['protocol_sha256']!=files['PROTOCOL_FROZEN_COPY.json']
        or result['blind_worlds_opened']!=0
        or result['model_training_performed'] is not False
        or result['diagnostic_qualification']!='NOT_EVALUATED'
        or result['development_worlds_per_arm']!=4
        or result['paired_native_trajectories']!=40
        or result['independent_noise_native_trajectories']!=40):
        raise ValueError('DEV_ONLY_CONTRACT_VIOLATION')
    labels=set(range(5));worlds=set(protocol['world_ids']);arms=set(protocol['arms'])
    for category in ('research_oracle_pairs','deployment_observations_dev_only'):
        for arm in arms:
            for world in worlds:
                for label in labels:
                    filename=f'{category}/{arm}/{world}/L{label}.csv'
                    with (root/filename).open(newline='') as f:
                        r=csv.reader(f)
                        if next(r)!=EXPECTED_HEADER:raise ValueError('DEPLOYMENT_FIELDS_INVALID')
                        rows=list(r)
                    if len(rows)!=181 or any(len(row)!=6 for row in rows):raise ValueError('TRAJECTORY_SHAPE_INVALID')
    for item in result['worlds']:
        if item['world_id'] not in worlds or set(a['arm'] for a in item['arms'])!=arms:
            raise ValueError('WORLD_REPORT_WRONG')
    return {'status':'PASS','verified_files':len(files),'development_worlds':len(worlds),
        'arms':len(arms),'trajectory_csv_count':len(worlds)*len(arms)*len(labels)*2,
        'blind_worlds_opened':0,'new_qualified_cells':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.root),indent=2))

if __name__=='__main__':main()
