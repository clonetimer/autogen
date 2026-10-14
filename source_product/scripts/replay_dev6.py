#!/usr/bin/env python3
# 用途：重放已冻结模型搜索并核对权重和选择。
# 参数：--study --output。
# 输出：重放模型及audit.json，不打开新的测试数据。
"""Replay complete designs without consuming a fresh holdout or calibration."""
from pathlib import Path
import argparse,json
from phm_design.reference_design import design_reference
from phm_design.route_selection import choose_metrics
from phm_design.io import read_json,write_json,sha_file

def replay(study,output):
    b,out=Path(study),Path(output);out.mkdir(parents=True,exist_ok=False)
    audit={}
    for d in ('eps','thermal','data'):
        orig=b/'designs'/d;new=out/d;data=b/'prepared'/d/'development'
        design_reference(data/'dataset.json',data/'device_binding.json',b/'policy.json',new)
        weights=sorted(orig.glob('generation_*/run/trials/*/seed_*/model.pt'))
        checks=[{'file':str(p.relative_to(orig)),'sha256':sha_file(p),
                 'equal':sha_file(p)==sha_file(new/p.relative_to(orig))} for p in weights]
        js={name:read_json(orig/name)==read_json(new/name) for name in ('lineage.json','design_report.json','deployment.json')}
        before={str(p.relative_to(orig)):(sha_file(p),p.stat().st_mtime_ns) for p in weights}
        design_reference(data/'dataset.json',data/'device_binding.json',b/'policy.json',orig,resume=True)
        unchanged=all((sha_file(orig/n),(orig/n).stat().st_mtime_ns)==v for n,v in before.items())
        sel=read_json(b/'selections'/d/'selection.json')
        route,eligible,rank=choose_metrics(sel['development_metrics'],sel['calibration_metrics'],sel['policy'])
        same_selection=(route==sel['chosen_route'] and eligible==sel['eligibility'] and {r:list(v) for r,v in rank.items()}==sel['rank_keys'])
        audit[d]={'weights':checks,'json_equal':js,'completed_resume_unchanged':unchanged,
                  'selection_metrics_recomputed_equal':same_selection}
        assert all(x['equal'] for x in checks) and all(js.values()) and unchanged and same_selection
        print('REPLAY_COMPLETE',d,len(checks),flush=True)
    write_json(out/'audit.json',{'schema':'phm.dev6-replay.v1','domains':audit,'all_equal':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();replay(a.study,a.output)
