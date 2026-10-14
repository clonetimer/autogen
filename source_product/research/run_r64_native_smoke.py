"""R64 Basilisk native-only smoke, not a diagnostic qualification experiment.

Never reads R62 evaluation traces or trains/selects models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path
from phm_design.adapters.sat_multicomponent import parameters,run_native,pair_audit,FAMILIES


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('OUTPUT_ALREADY_EXISTS')
    args.output.mkdir(parents=True)
    p=parameters('r64-native-physics-smoke-001','thermal')
    reference=run_native('thermal',p,0)
    records=[]
    for label in range(5):
        sim=reference if label==0 else run_native('thermal',p,label)
        data=json.dumps({'observations':sim['observations'],'truth':sim['truth'],
                         'output_times_ns':sim['output_times_ns']},
                        separators=(',',':'),allow_nan=False).encode()
        # Truth and simulator parameters are in separate research-only files,
        # never merged into deployed observations.
        file=args.output/f'simulator_case_{label}.json';file.write_bytes(data)
        audit=pair_audit(reference,sim,p,'thermal',label)
        records.append({'label':FAMILIES['thermal']['classes'][label],
                        'sha256':hashlib.sha256(data).hexdigest(),
                        'rows':len(sim['observations']),
                        'native_timestamp_consistency': all(a==b for a,b in sim['output_times_ns']),
                        'pair_audit':audit})
    result={'schema':'phm.r64.native-smoke.v1','role':'DEVELOPMENT_NATIVE_SMOKE_NOT_QUALIFICATION',
            'bsk_version':md.version('bsk'),'python':sys.version.split()[0],
            'domain':'thermal','world_key':'r64-native-physics-smoke-001',
            'normal_and_four_fault_cases':5,'records':records,
            'not_used_for_model_selection':True,'new_certification_worlds':0}
    (args.output/'SMOKE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'SIMULATOR_PARAMETERS_ONLY.json').write_text(json.dumps(p,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
