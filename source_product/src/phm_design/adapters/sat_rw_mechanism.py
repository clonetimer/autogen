"""Opt-in nine-channel ADCS adapter, bound to inspected native construction.

This does not modify the simulation or silently repair historical metadata.
The factory's rw_wheel_inertia_kg_m2 is an envelope parameter, NOT the inertia
set on Honeywell_HR16 by UnifiedNativeRuntime. We query the actual factory.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
from ..data import Dataset,validate_contract,reject_overlap
from ..io import read_json,write_json,sha_file,safe_child,digest
from ..physics import analyze_program
from .sat_rw import prepare

GYRO = [{'name':f'adcs.sensor.gyro_measured_rad_s_{axis}','unit':'rad/s','role':'deployment_input',
         'provenance':'Basilisk ImuSensor.AngVelPlatform recorder, body-aligned; simulated sensing, not real hardware'} for axis in 'xyz']


def momentum_program(contract:dict, inertias:list[float], limits:list[float], source:dict)->dict:
    """Build reviewed fixed-axis balance equations, not inferred physical laws."""
    if len(inertias)!=3 or len(limits)!=3 or len(contract['channels'])!=9:
        raise ValueError('ORTHOGONAL_THREE_WHEEL_NINE_CHANNEL_CONTRACT_REQUIRED')
    nodes=[];constants=[];outputs=[]
    def op(name,*refs):
        key=f'p{len(nodes)}';nodes.append({'id':key,'op':name,'inputs':list(refs)});return key
    for i in range(3):
        constants += [{'id':f'k{2*i}','value':inertias[i],'unit':'kg*m^2','role':'design_constant','source':source},
                      {'id':f'k{2*i+1}','value':limits[i],'unit':'N*m','role':'design_constant','source':source}]
        inertial_rate=op('add',f'x{i}',f'x{6+i}')
        delta=op('delta',inertial_rate)
        momentum=op('mul',f'k{2*i}',delta)
        normal_torque=op('symmetric_clip',f'x{3+i}',f'k{2*i+1}')
        impulse=op('trapezoid_integral',normal_torque)
        residual=op('sub',momentum,impulse)
        outputs.append({'name':f'wheel_{i}_momentum_innovation','node':residual,'unit':'N*m*s','component':f'wheel_{i}'})
    return {'schema':'phm.physical-program.v1','inputs':contract['channels'],'sample_s':contract['sample_s'],
            'constants':constants,'nodes':nodes,'outputs':outputs,'source':source,'evidence_scope':'simulation_research',
            'assumptions':['Balanced rigid fixed-axis rotors aligned with body x/y/z, not gimballed or jitter-coupled.',
                           'Normal rotor friction is zero in inspected factory configuration; fault friction is not used in inputs.',
                           'Body angular-rate projection is included; gyroscope bias/noise, discretization and model mismatch remain in the innovation.',
                           'Trapezoid command integration at telemetry rate is an approximation; commands also update between telemetry samples.',
                           'Window integral starts at the first observed sample, NOT at the fault onset.',
                           'Model innovation is not the isolated fault torque and is not ground-truth cause.',
                           'Inertias come from the instantiated Basilisk wheel factory, not the inconsistent envelope metadata.']}


def prepare_mechanism(campaign:str|Path,output:str|Path,project_root:str|Path)->dict:
    campaign,output,project_root=Path(campaign),Path(output),Path(project_root)
    if output.exists(): raise FileExistsError('OUTPUT_EXISTS')
    stage=output.with_name(output.name+'.preparing')
    if stage.exists(): raise FileExistsError('PREPARATION_INCOMPLETE_REVIEW_REQUIRED')
    source=project_root/'src/sat_sim/bsk_engine/unified_native.py'
    effect=project_root/'src/sat_sim/bsk_engine/project_native_modules.py'
    if not source.is_file() or not effect.is_file(): raise ValueError('SOURCE_AUDIT_FILES_REQUIRED')
    # Exact source anchors fail closed on construction changes; no assumption
    # that a similarly named wheel_inertia parameter reaches this constructor.
    text=source.read_text();effect_text=effect.read_text()
    required=['"Honeywell_HR16", list(axis), maxMomentum=50.0', 'u_max=0.2, label=f"RW{index + 1}"', 'imu.setBodyToPlatformDCM(0.0, 0.0, 0.0)']
    if not all(s in text for s in required): raise ValueError('NATIVE_CONSTRUCTION_CHANGED_REVIEW_REQUIRED')
    from Basilisk.utilities import simIncludeRW
    factory=simIncludeRW.rwFactory()
    wheel=factory.create('Honeywell_HR16',[1.,0.,0.],maxMomentum=50.,Omega=100.,u_max=0.2,label='RW1')
    if wheel.RWModel!=0 or any(abs(v)>0 for v in (wheel.fCoulomb,wheel.fStatic,wheel.cViscous)):
        raise ValueError('UNREVIEWED_ROTOR_OR_NOMINAL_FRICTION_MODEL')
    native={'Js_kg_m2':float(wheel.Js),'u_max_nm':float(wheel.u_max),'RWModel':int(wheel.RWModel),
            'fCoulomb':float(wheel.fCoulomb),'cViscous':float(wheel.cViscous)}
    audit={'scope':'simulation_source_binding_only','native_wheel':native,
           'runtime_source':{'reference':'src/sat_sim/bsk_engine/unified_native.py','sha256':sha_file(source)},
           'fault_effect_source':{'reference':'src/sat_sim/bsk_engine/project_native_modules.py','sha256':sha_file(effect)},
           'wheel_factory_source':{'reference':'Basilisk/utilities/simIncludeRW.py','sha256':sha_file(Path(simIncludeRW.__file__))},
           'torque_authority_implementation':'requested command multiplied by scale, then clipped to nominal max_torque_nm',
           'reduced_limit_is_used_for_clipping':False,'independent_hardware_verification':False,
           'known_discrepancy':'Metadata/envelope rw_wheel_inertia_kg_m2 does not configure native wheel.Js; source checked, simulation unchanged.'}
    expected = 'output[index] *= scale'
    if expected not in effect_text or 'min(self.max_torque_nm, float(output[index]))' not in effect_text:
        raise ValueError('FAULT_EFFECT_CHANGED_REVIEW_REQUIRED')
    old=prepare(campaign,stage)
    by_id={}
    metadata_inertias=set()
    for cp in sorted((campaign/'cases').glob('*/astrograph/dataset_contract.json')):
        c=read_json(cp);spec=read_json(cp.parent.parent/'task_spec.json')
        values=spec['parameters'];values=values.get('values',values)
        if values.get('rw_configuration','orthogonal_3')!='orthogonal_3':
            raise ValueError('UNREVIEWED_WHEEL_AXES')
        if c['pairing']['case_role']=='fault' and c['ground_truth']['class_id'] not in {'ADCS_RW_FRICTION_INCREASE','ADCS_RW_JAM','ADCS_RW_TORQUE_AUTHORITY_LOSS'}:
            raise ValueError('UNREVIEWED_FAULTS')
        metadata_inertias.add(values.get('rw_wheel_inertia_kg_m2'))
        by_id[c['dataset_id']]=safe_child(cp.parent.parent,c['telemetry']['source_trace_file'])
    paths={r['id']:by_id[r['source_case']] for r in old['source_records']}
    for part in ('development','holdout'):
        path=stage/part/'dataset.json';data=read_json(path)
        data['contract']['channels']+=GYRO
        data['contract']['device_family']='satellite.adcs.reaction_wheel.fixed_three_wheels.with_body_gyro'
        data['contract']['scope']+='; body-aligned simulated gyro for balanced-rotor momentum innovation; unreviewed physical fidelity'
        data['dataset_id']=f'sat-rw-v04-nine-{part}'
        data['provenance']['adapter']='sat_rw_mechanism.v1'
        for sample in data['samples']:
            dest=safe_child(stage/part,sample['path']); fields=['time_s']+[c['name'] for c in data['contract']['channels']]
            with paths[sample['id']].open(newline='') as stream:
                reader=csv.DictReader(stream)
                if not set(fields).issubset(reader.fieldnames or []):raise ValueError('GYRO_MEASUREMENT_COLUMNS_MISSING')
                rows=[{k:r[k] for k in fields} for r in reader]
            with dest.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            sample['sha256']=sha_file(dest)
        validate_contract(data['contract']);write_json(path,data)
    write_json(stage/'runtime_binding.json',{**audit,'declared_envelope_inertias_kg_m2':sorted(metadata_inertias,key=str)})
    binding={'reference':'runtime_binding.json','sha256':sha_file(stage/'runtime_binding.json')}
    contract=read_json(stage/'development/dataset.json')['contract']
    program=momentum_program(contract,[native['Js_kg_m2']]*3,[native['u_max_nm']]*3,binding)
    program_audit=analyze_program(program,contract)
    write_json(stage/'physical_program.json',program);write_json(stage/'physical_program_audit.json',program_audit)
    dev=Dataset(stage/'development/dataset.json',allowed_splits={'train','validation'})
    test=Dataset(stage/'holdout/dataset.json',allowed_splits={'test'})
    reject_overlap(dev.samples,test.samples);dev.verify_files();test.verify_files()
    report={**old,'channel_count':9,'adapter':'sat_rw_mechanism.v1','runtime_binding':'runtime_binding.json',
            'physical_program_sha256':digest(program),'source_hash_verified':True,'physical_qualification':False}
    write_json(stage/'preparation_report.json',report)
    write_json(stage/'units.json',{c['name']:c['unit'] for c in contract['channels']})
    os.replace(stage,output)
    return report


def generate_mechanism_campaign(config_path:str|Path,output:str|Path)->dict:
    """Reuse the existing factory/runner; only extend the observation manifest."""
    from sat_sim.fault_dataset_factory import (FaultDatasetFactoryConfig,build_fault_dataset_campaign,run_fault_dataset_campaign)
    cfg=FaultDatasetFactoryConfig.from_json(config_path)
    if Path(output).exists(): raise FileExistsError('CAMPAIGN_OUTPUT_EXISTS')
    plan=build_fault_dataset_campaign(cfg)
    fields=plan['campaign']['base_spec']['outputs']['record_fields']
    for c in GYRO:
        if c['name'] not in fields:fields.append(c['name'])
    result=run_fault_dataset_campaign(plan,output_root=Path(output),continue_on_error=False,dry_run=False)
    write_json(Path(output)/'mechanism_campaign_request.json',plan)
    return result.to_dict()
