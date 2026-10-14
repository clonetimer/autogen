"""Active package scope and frozen legacy model access; qualification is version-bound."""
import argparse, json
from .platform_limited import contract, allowed_use_case, predict_bundle, design_extended

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('scope')
    c=sub.add_parser('check-use-case');c.add_argument('--case',required=True)
    d=sub.add_parser('design');d.add_argument('--dataset',required=True);d.add_argument('--output',required=True);d.add_argument('--trials',type=int,default=24)
    r=sub.add_parser('predict-bundled');r.add_argument('--domain',required=True);r.add_argument('--case',required=True);r.add_argument('--observations',required=True);r.add_argument('--healthy-prefix-confirmed',action='store_true')
    a=p.parse_args(argv)
    if a.action=='scope':result=contract()
    elif a.action=='check-use-case':result=allowed_use_case(a.case)
    elif a.action=='design':result=design_extended(a.dataset,a.output,max_trials=a.trials)
    else:result=predict_bundle(a.domain,a.case,a.observations,healthy_prefix_confirmed=a.healthy_prefix_confirmed)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
