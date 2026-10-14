"""R67 immutable inventory, checksum, native timestamps and role-boundary verification."""
from __future__ import annotations
import argparse,csv,hashlib,json
import numpy as np
from pathlib import Path
try:
    from research.run_r67_dev_experiment import validate_protocol,HEADER
except ModuleNotFoundError:
    from run_r67_dev_experiment import validate_protocol,HEADER


def verify(root:Path)->dict:
    manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
    if manifest['schema']!='phm.r67.dev-evidence-manifest.v1':raise ValueError('R67_BAD_MANIFEST_SCHEMA')
    files=manifest['files'];actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(files)|{'MANIFEST_SHA256.json'}:raise ValueError('R67_INVENTORY_MISMATCH')
    for name,expected in files.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('R67_CHECKSUM_FAIL:'+name)
    p=json.loads((root/'PROTOCOL_FROZEN_COPY.json').read_text());validate_protocol(p)
    r=json.loads((root/'R67_DEV_RESULTS.json').read_text())
    if (r['protocol_sha256'] != files['PROTOCOL_FROZEN_COPY.json']
        or r['status']!='DEVELOPMENT_ONLY_NOT_QUALIFIED'
        or r['blind_worlds_opened']!=0 or r['model_training_performed'] is not False
        or r['diagnostic_qualification']!='NOT_EVALUATED'
        or r['native_trajectories']!=120
        or r['paired_oracle_trajectories']!=60
        or r['independent_noise_trajectories']!=60
        or len(r['worlds'])!=4):raise ValueError('R67_QUALIFICATION_BOUNDARY')
    sim_params=json.loads((root/'SIMULATOR_PARAMETERS_ONLY.json').read_text())
    if sim_params.get('role')!='SIMULATOR_ONLY_NOT_DEPLOYMENT':raise ValueError('R67_SIMULATOR_ROLE')
    for category in ('research_oracle_pairs','deployment_observations_dev_only'):
        for arm in p['arms']:
            for world_id in p['world_ids']:
                for label in range(5):
                    with (root/category/arm/world_id/f'L{label}.csv').open(newline='') as f:
                        csv_rows=csv.reader(f)
                        if next(csv_rows)!=list(HEADER):raise ValueError('R67_HEADER')
                        rows=list(csv_rows)
                        if len(rows)!=721 or any(len(row)!=6 for row in rows):raise ValueError('R67_CSV_ROWS')
                        if any(float(row[0])!=i*20 for i,row in enumerate(rows)):raise ValueError('R67_CSV_TIMES')
                        arr=np.asarray(rows,dtype=float)
                        if not np.isfinite(arr).all() or arr[:,4:6].min()<250 or arr[:,4:6].max()>450:
                            raise ValueError('R67_NUMERICAL_SAFETY')
                        params=sim_params['parameters'][world_id]
                        for comp in (0,1):
                            base=params['heat_a_w' if comp==0 else 'heat_b_w']
                            for start in (0,360):
                                if not np.isclose(20*arr[start+1:start+361,comp+2].sum(),7200*base,rtol=1e-10,atol=1e-7):
                                    raise ValueError('R67_COMMAND_ENERGY')
    # Reconstruct all 24 mechanism-separation residuals from immutable native CSV
    # research oracle ONLY.  Never look at the independent-noise deployment files.
    for world_report in r['worlds']:
        world_id=world_report['world_id']
        if world_id not in p['world_ids']:raise ValueError('R67_WORLD_ID')
        for arm,arm_report in world_report['arms'].items():
            origin=root/'research_oracle_pairs'/arm/world_id
            def load(label):
                with (origin/f'L{label}.csv').open(newline='') as f:
                    return np.asarray(list(csv.reader(f))[1:],dtype=float)
            nominal=load(0)
            for component,emiss,heat,index in [('a',1,3,4),('b',2,4,5)]:
                a=load(emiss)[361:,index]-nominal[361:,index]
                b=load(heat)[361:,index]-nominal[361:,index]
                scale=np.dot(a,b)/np.dot(a,a)
                residual=np.linalg.norm(b-scale*a)/np.linalg.norm(b)
                expected=next(c for c in arm_report['components'] if c['component']==component)['postfault_scaled_residual_fraction']
                if not np.isclose(residual,expected,rtol=1e-8,atol=1e-10):
                    raise ValueError('R67_ORACLE_RESIDUAL_NOT_REPRODUCIBLE')
    return {'status':'PASS','sha256_files_verified':len(files),'native_trajectories':120,
            'blind_worlds_opened':0,'diagnostic_qualifications_added':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
