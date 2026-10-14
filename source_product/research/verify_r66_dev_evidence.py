"""Strict checksum, inventory, protocol and deployment-field verification for R66."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
try:
    from research.run_r66_dev_experiment import validate_protocol,HEADER
except ModuleNotFoundError:
    from run_r66_dev_experiment import validate_protocol,HEADER

def verify(root:Path):
    doc=json.loads((root/'MANIFEST_SHA256.json').read_text())
    if doc['schema']!='phm.r66.dev-evidence-manifest.v1':raise ValueError('BAD_SCHEMA')
    m=doc['files'];expected=set(m)|{'MANIFEST_SHA256.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('INVENTORY_MISMATCH')
    for name,hash_ in m.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=hash_:raise ValueError('FILE_HASH_MISMATCH:'+name)
    p=json.loads((root/'PROTOCOL_FROZEN_COPY.json').read_text());validate_protocol(p)
    r=json.loads((root/'R66_DEV_RESULTS.json').read_text())
    if (r['protocol_sha256']!=m['PROTOCOL_FROZEN_COPY.json'] or
        r['blind_worlds_opened']!=0 or r['model_training_performed'] is not False or
        r['diagnostic_qualification']!='NOT_EVALUATED' or
        r['paired_native_trajectories']!=40 or r['independent_noise_native_trajectories']!=40):
        raise ValueError('QUALIFICATION_BOUNDARY')
    for cat in ('research_oracle_pairs','deployment_observations_dev_only'):
        for arm in ('A','B'):
            for world in p['world_ids']:
                for label in range(5):
                    with (root/cat/arm/world/f'L{label}.csv').open(newline='') as f:
                        rr=csv.reader(f)
                        if next(rr)!=list(HEADER):raise ValueError('DEPLOYMENT_FIELDS_INCORRECT')
                        rows=list(rr)
                        if len(rows)!=721 or any(len(row)!=6 for row in rows):raise ValueError('TRAJECTORY_INVALID')
                        if any(float(row[0])!=i*20 for i,row in enumerate(rows)):raise ValueError('TIMES_INVALID')
    return {'status':'PASS','sha256_files_verified':len(m),'native_trajectories':80,'blind_worlds_opened':0,'new_diagnostic_qualifications':0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().root),indent=2))

if __name__=='__main__':main()
