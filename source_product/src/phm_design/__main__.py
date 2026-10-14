"""Linux CPU reference CLI. Nonzero exit on invalid input or incomplete search."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .io import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phm_design")
    sub = parser.add_subparsers(dest="command", required=True)
    pcd = sub.add_parser("plan-codesign-arms", help="map legal representations to executable cross-domain co-design arms")
    pcd.add_argument("--task", choices=["multicomponent","family"], default="multicomponent")
    pcd.add_argument("--capabilities", help="optional JSON mapping domain to frozen prerequisite artifact names")
    pcd.add_argument("--output", required=True)
    prepr = sub.add_parser("plan-representations", help="enumerate legal representation arms for automatic cross-domain co-design")
    prepr.add_argument("--task", choices=["multicomponent","family"], default="multicomponent")
    prepr.add_argument("--capabilities", help="optional JSON mapping domain to frozen prerequisite artifact names")
    prepr.add_argument("--output", required=True)
    sharm = sub.add_parser("select-hybrid-arm", help="choose mechanism versus mechanism+commissioning hybrid without reading risk/test data")
    for flag in ("mechanism-run","mechanism-selection","hybrid-run","hybrid-selection","output"):
        sharm.add_argument("--"+flag, required=True)
    gh2 = sub.add_parser("generate-hybrid-commissioning", help="fit independent healthy regression profiles for mechanism+commissioning hybrid")
    for flag in ("campaign","output"): gh2.add_argument("--"+flag, required=True)
    ph2 = sub.add_parser("prepare-hybrid", help="preserve mechanism features and append healthy commissioning residuals")
    for flag in ("campaign","commissioning","domain","section","output"): ph2.add_argument("--"+flag, required=True)
    ph2s = sub.add_parser("predict-hybrid-selected", help="raw prediction using frozen hybrid route and commissioning profile")
    for flag in ("run","selection","csv","passport","profile","domain","output"): ph2s.add_argument("--"+flag, required=True)
    ph2r = sub.add_parser("predict-hybrid-risk", help="raw prediction using frozen hybrid route, commissioning profile and risk gate")
    for flag in ("run","selection","risk","csv","passport","profile","domain","output"): ph2r.add_argument("--"+flag, required=True)
    sfa = sub.add_parser("select-feature-arm", help="freeze a representation using development plus route-calibration metrics only")
    sfa.add_argument("--config",required=True);sfa.add_argument("--output",required=True)
    thm = sub.add_parser("fit-thermal-health-models", help="fit healthy thermal nuisance regressions from separate commissioning telemetry")
    for flag in ("campaign","commissioning","output"): thm.add_argument("--"+flag, required=True)
    thp = sub.add_parser("prepare-thermal-v2", help="prepare thermal diagnosis features corrected by frozen healthy regression")
    for flag in ("campaign","commissioning","health-models","section","output"): thp.add_argument("--"+flag, required=True)
    thps = sub.add_parser("predict-thermal-v2-selected", help="raw thermal prediction with frozen route and healthy regression")
    for flag in ("run","selection","csv","passport","profile","health-model","output"): thps.add_argument("--"+flag, required=True)
    thpr = sub.add_parser("predict-thermal-v2-risk", help="raw thermal prediction with risk gate and healthy regression")
    for flag in ("run","selection","risk","csv","passport","profile","health-model","output"): thpr.add_argument("--"+flag, required=True)
    gh = sub.add_parser("generate-commissioning", help="generate separate nominal commissioning profiles for an existing non-ADCS campaign")
    for flag in ("campaign","output"): gh.add_argument("--"+flag, required=True)
    ph = sub.add_parser("prepare-commissioned-family", help="prepare diagnosis features centered by separate healthy commissioning telemetry")
    for flag in ("campaign","commissioning","domain","section","output"): ph.add_argument("--"+flag, required=True)
    pcs = sub.add_parser("predict-commissioned-selected", help="raw prediction using selected route plus commissioning profile")
    for flag in ("run","selection","csv","passport","profile","domain","output"): pcs.add_argument("--"+flag, required=True)
    pcr = sub.add_parser("predict-commissioned-risk", help="raw prediction using selected route, risk gate and commissioning profile")
    for flag in ("run","selection","risk","csv","passport","profile","domain","output"): pcr.add_argument("--"+flag, required=True)
    cr = sub.add_parser("calibrate-risk", help="choose abstention threshold from a distinct calibration cohort")
    for flag in ("run","selection","calibration","policy","output","registry"):cr.add_argument("--"+flag, required=True)
    erisk = sub.add_parser("evaluate-risk", help="evaluate frozen route plus independent risk threshold")
    for flag in ("run","selection","risk","holdout","output","registry"):erisk.add_argument("--"+flag, required=True)
    prisk = sub.add_parser("predict-risk", help="raw input prediction using frozen route and risk threshold")
    for flag in ("run","selection","risk","csv","passport","domain","output"):prisk.add_argument("--"+flag, required=True)
    srm = sub.add_parser("select-reference-route-multi", help="freeze route across multiple independent selection environments")
    for flag in ("run","output","registry"): srm.add_argument("--"+flag, required=True)
    srm.add_argument("--policy")
    srm.add_argument("--calibration", action="append", required=True, help="NAME=dataset.json; specify at least twice")
    sma = sub.add_parser("select-multi-arm", help="freeze representation/search arm from multi-environment route selections")
    sma.add_argument("--config", required=True); sma.add_argument("--output", required=True)
    smm = sub.add_parser("select-matched-multi", help="one-shot matched multi-environment comparison across routes/representations/search strategies")
    for flag in ("config","output","registry"): smm.add_argument("--"+flag, required=True)
    sr = sub.add_parser("select-reference-route", help="consume separate calibration data for frozen route selection")
    for flag in ("run", "calibration", "output", "registry"):sr.add_argument("--"+flag, required=True)
    sr.add_argument("--policy")
    es = sub.add_parser("evaluate-selected-reference", help="one-shot evaluation of a frozen route")
    for flag in ("run", "selection", "holdout", "output", "registry"):es.add_argument("--"+flag, required=True)
    ps = sub.add_parser("predict-selected-reference", help="raw input prediction with a frozen selected route")
    for flag in ("run", "selection", "csv", "passport", "domain", "output"):ps.add_argument("--"+flag, required=True)
    rdr = sub.add_parser("design-reference-random", help="matched-budget feedback-free random legal graph control")
    for flag in ("dataset","binding","policy","output"): rdr.add_argument("--"+flag,required=True)
    rdr.add_argument("--resume",action="store_true")
    rdo = sub.add_parser("design-reference-optuna", help="matched-budget Optuna TPE legal graph proposal backend")
    for flag in ("dataset","binding","policy","output"): rdo.add_argument("--"+flag,required=True)
    rdo.add_argument("--resume",action="store_true")
    rdp = sub.add_parser("design-reference-portfolio", help="fixed-budget alternating Native Feedback + Optuna TPE proposal portfolio")
    for flag in ("dataset","binding","policy","output"): rdp.add_argument("--"+flag,required=True)
    rdp.add_argument("--resume",action="store_true")
    rd = sub.add_parser("design-reference", help="generate matched raw/bounded/unbounded correction DAGs")
    for flag in ("dataset","binding","policy","output"):rd.add_argument("--"+flag,required=True)
    rd.add_argument("--resume",action="store_true")
    er = sub.add_parser("evaluate-reference", help="one-shot evaluation of frozen comparison and deployment routes")
    for flag in ("run","holdout","output"):er.add_argument("--"+flag,required=True)
    prf = sub.add_parser("predict-reference", help="raw CSV plus nominal passport inference")
    for flag in ("run","csv","passport","domain","output"):prf.add_argument("--"+flag,required=True)
    gm = sub.add_parser("generate-multicomponent", help="generate native two-component localization benchmark")
    gm.add_argument("--config", required=True); gm.add_argument("--output", required=True)
    pm = sub.add_parser("prepare-multicomponent", help="prepare causal component-wise mechanism features")
    pm.add_argument("--campaign", required=True); pm.add_argument("--domain", choices=["eps","thermal","data"], required=True)
    pm.add_argument("--section", choices=["development","calibration","holdout"], required=True); pm.add_argument("--output", required=True)
    pm.add_argument("--representation-version", choices=["v1","v2"], default="v1")
    hd = sub.add_parser("evaluate-hierarchical-diagnosis", help="derive detection/location/mechanism metrics from a frozen risk evaluation")
    hd.add_argument("--domain", choices=["eps","thermal","data"], required=True)
    hd.add_argument("--dataset", required=True); hd.add_argument("--risk-evaluation", required=True); hd.add_argument("--output", required=True)
    gc = sub.add_parser("generate-nonadcs-context", help="freeze normal passports then generate native research cases")
    gc.add_argument("--config", required=True); gc.add_argument("--output", required=True)
    fp = sub.add_parser("prepare-family", help="trusted per-role causal feature preparation")
    fp.add_argument("--campaign", required=True); fp.add_argument("--domain", choices=["eps","thermal","data"], required=True)
    fp.add_argument("--section", choices=["development","calibration","holdout"], required=True)
    fp.add_argument("--mode", choices=["context","mechanism"], required=True); fp.add_argument("--output", required=True)
    tf2 = sub.add_parser("prepare-family-thermal-v2", help="prepare causal rolling thermal family v2 features; research-only")
    for flag in ("campaign", "section", "output"):
        tf2.add_argument("--"+flag, required=True)
    tf2.add_argument("--no-blind", action="store_true", help="reject test/holdout roles explicitly")
    ftfit = sub.add_parser("fit-thermal-v2-forest", help="fit frozen JSON forest using train-only Thermal v2 datasets")
    ftfit.add_argument("--train-dataset",action="append",required=True)
    ftfit.add_argument("--policy",required=True)
    ftfit.add_argument("--output",required=True)
    ftpr = sub.add_parser("predict-thermal-v2-forest", help="raw/passport prediction from a sealed, research-only Thermal v2 JSON forest")
    for flag in ("model", "csv", "passport", "output"):ftpr.add_argument("--"+flag,required=True)
    xp = sub.add_parser("export-family-pipeline", help="bundle raw-to-feature transform with selected neural ensemble")
    xp.add_argument("--selected", required=True); xp.add_argument("--domain", choices=["eps","thermal","data"], required=True)
    xp.add_argument("--mode", choices=["context","mechanism"], required=True); xp.add_argument("--output", required=True)
    pp = sub.add_parser("predict-family-pipeline", help="predict from raw CSV and pre-injection normal design passport")
    pp.add_argument("--pipeline", required=True); pp.add_argument("--csv", required=True)
    pp.add_argument("--passport", required=True); pp.add_argument("--output", required=True)
    fd = sub.add_parser("design-family", help="same compositional designer, device-bound arbitrary component groups")
    fd.add_argument("--dataset", required=True)
    fd.add_argument("--binding", required=True)
    fd.add_argument("--policy", required=True)
    fd.add_argument("--output", required=True)
    fd.add_argument("--resume", action="store_true")
    nd = sub.add_parser("generate-nonadcs", help="native EPS/thermal/data research datasets; not a certified capability")
    nd.add_argument("--config", required=True)
    nd.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare-sat", help="trusted raw campaign-to-dataset adapter")
    prepare.add_argument("--campaign", required=True)
    prepare.add_argument("--output", required=True)
    for name in ("plan", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, help="development/dataset.json only")
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name == "fit":
            p.add_argument("--resume", action="store_true")
    gm = sub.add_parser("generate-sat-mechanism", help="existing Basilisk factory with gyro observation export")
    gm.add_argument("--config", required=True)
    gm.add_argument("--output", required=True)
    pm = sub.add_parser("prepare-sat-mechanism", help="nine-channel raw adapter plus sourced momentum program")
    pm.add_argument("--campaign", required=True)
    pm.add_argument("--output", required=True)
    pm.add_argument("--project-root", required=True)
    md = sub.add_parser("design-mechanism", help="dimension-checked physical/data graph comparison")
    md.add_argument("--dataset", required=True)
    md.add_argument("--program", required=True)
    md.add_argument("--request", required=True)
    md.add_argument("--output", required=True)
    md.add_argument("--resume", action="store_true")
    dg = sub.add_parser("design", help="synthesize and evolve compositional operator graphs")
    dg.add_argument("--dataset", required=True, help="development dataset only")
    dg.add_argument("--request", required=True)
    dg.add_argument("--output", required=True)
    dg.add_argument("--resume", action="store_true")
    ab = sub.add_parser("ablate", help="development-only structural ablations")
    ab.add_argument("--dataset", required=True)
    ab.add_argument("--design", required=True)
    ab.add_argument("--output", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run", required=True)
    ev.add_argument("--holdout", required=True)
    ev.add_argument("--output", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--run", required=True)
    pr.add_argument("--csv", required=True)
    pr.add_argument("--units", required=True, help="JSON name-to-unit contract; values must match the frozen bundle")
    pr.add_argument("--output", required=True)
    rb = sub.add_parser("design-robust", help="synthesize quality-aware compositional graphs with development stresses")
    rb.add_argument("--dataset", required=True)
    rb.add_argument("--request", required=True)
    rb.add_argument("--binding", required=True)
    rb.add_argument("--output", required=True)
    rb.add_argument("--resume", action="store_true")
    re = sub.add_parser("evaluate-robust", help="frozen stress evaluation; missing cases remain in denominators")
    re.add_argument("--run", required=True)
    re.add_argument("--holdout", required=True)
    re.add_argument("--output", required=True)
    rp = sub.add_parser("predict-robust", help="mask/timestamp-carrying observation JSON, not legacy CSV")
    rp.add_argument("--run", required=True)
    rp.add_argument("--packet", required=True)
    rp.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan-codesign-arms":
            from .representation_registry import multidomain_codesign_plan
            caps=read_json(args.capabilities) if args.capabilities else {}
            r=multidomain_codesign_plan(task=args.task,available_artifacts_by_domain=caps)
            write_json(args.output,r)
            summary={"status":"PLANNED","domains":r["domains"],"test_used":False,"production_qualified":False}
        elif args.command == "plan-representations":
            from .representation_registry import multidomain_representation_plan
            caps=read_json(args.capabilities) if args.capabilities else {}
            if not isinstance(caps,dict): raise ValueError("REPRESENTATION_CAPABILITIES_SCHEMA")
            r=multidomain_representation_plan(task=args.task,available_artifacts_by_domain=caps)
            write_json(args.output,r); summary=r
        elif args.command == "select-hybrid-arm":
            from .feature_arm_selection import select_hybrid_arm
            summary=select_hybrid_arm(args.mechanism_run,args.mechanism_selection,args.hybrid_run,args.hybrid_selection,args.output)
        elif args.command == "generate-hybrid-commissioning":
            from .commissioning_hybrid import generate_profiles
            summary=generate_profiles(args.campaign,args.output)
        elif args.command == "prepare-hybrid":
            from .commissioning_hybrid import prepare_hybrid_section
            r=prepare_hybrid_section(args.campaign,args.commissioning,args.domain,args.section,args.output)
            summary={"status":"PREPARED","cases":len(r["samples"]),"dataset_id":r["dataset_id"]}
        elif args.command == "predict-hybrid-selected":
            from .commissioning_hybrid import predict_hybrid_selected
            summary=predict_hybrid_selected(args.run,args.selection,args.csv,args.passport,args.profile,args.domain);write_json(args.output,summary)
        elif args.command == "predict-hybrid-risk":
            from .commissioning_hybrid import predict_hybrid_risk
            summary=predict_hybrid_risk(args.run,args.selection,args.risk,args.csv,args.passport,args.profile,args.domain);write_json(args.output,summary)
        elif args.command == "select-feature-arm":
            from .feature_selection import select_feature_arm_config
            r=select_feature_arm_config(args.config,args.output)
            summary={k:r[k] for k in ("chosen_arm","risk_calibration_used","test_used","production_qualified")}
        elif args.command == "fit-thermal-health-models":
            from .thermal_commissioning_v2 import fit_registry
            r=fit_registry(args.campaign,args.commissioning,args.output)
            summary={k:r[k] for k in ("completed_models","fault_information_used","diagnosis_time_paired_nominal_used","production_qualified")}
        elif args.command == "prepare-thermal-v2":
            from .thermal_commissioning_v2 import prepare_section
            r=prepare_section(args.campaign,args.commissioning,args.health_models,args.section,args.output)
            summary={"status":"PREPARED","cases":len(r["samples"]),"dataset_id":r["dataset_id"]}
        elif args.command == "predict-thermal-v2-selected":
            from .thermal_commissioning_v2 import predict_selected
            summary=predict_selected(args.run,args.selection,args.csv,args.passport,args.profile,args.health_model);write_json(args.output,summary)
        elif args.command == "predict-thermal-v2-risk":
            from .thermal_commissioning_v2 import predict_risk
            summary=predict_risk(args.run,args.selection,args.risk,args.csv,args.passport,args.profile,args.health_model);write_json(args.output,summary)
        elif args.command == "generate-commissioning":
            from .commissioning import generate_commissioning
            def progress(record):
                print(json.dumps(record),file=sys.stderr,flush=True)
            r=generate_commissioning(args.campaign,args.output,on_profile=progress)
            summary={k:r[k] for k in ("completed_profiles","separate_nominal_runs","fault_information_used","production_qualified")}
        elif args.command == "prepare-commissioned-family":
            from .commissioning import prepare_commissioned_section
            r=prepare_commissioned_section(args.campaign,args.commissioning,args.domain,args.section,args.output)
            summary={"status":"PREPARED","cases":len(r["samples"]),"dataset_id":r["dataset_id"]}
        elif args.command == "predict-commissioned-selected":
            from .commissioning import predict_commissioned_selected
            summary=predict_commissioned_selected(args.run,args.selection,args.csv,args.passport,args.profile,args.domain);write_json(args.output,summary)
        elif args.command == "predict-commissioned-risk":
            from .commissioning import predict_commissioned_risk
            summary=predict_commissioned_risk(args.run,args.selection,args.risk,args.csv,args.passport,args.profile,args.domain);write_json(args.output,summary)
        elif args.command == "calibrate-risk":
            from .risk_calibration import calibrate_risk
            r=calibrate_risk(args.run,args.selection,args.calibration,args.output,registry=args.registry,policy=read_json(args.policy))
            summary={k:r[k] for k in ("status","route","threshold","group_risk_bound_qualified","risk_certified","test_used")}
        elif args.command == "evaluate-risk":
            from .risk_calibration import evaluate_risk
            summary=evaluate_risk(args.run,args.selection,args.risk,args.holdout,args.output,registry=args.registry)
        elif args.command == "predict-risk":
            from .risk_calibration import predict_risk
            summary=predict_risk(args.run,args.selection,args.risk,args.csv,args.passport,args.domain);write_json(args.output,summary)
        elif args.command == "select-reference-route-multi":
            from .multi_environment_selection import select_route_multi
            envs={}
            for item in args.calibration:
                if "=" not in item: raise ValueError("CALIBRATION_MUST_BE_NAME_EQUALS_PATH")
                name,path=item.split("=",1)
                if not name.isidentifier() or not path or name in envs: raise ValueError("CALIBRATION_NAME_OR_DUPLICATE")
                envs[name]=path
            r=select_route_multi(args.run,envs,args.output,registry=args.registry,
                policy=read_json(args.policy) if args.policy else None)
            summary={k:r[k] for k in ("chosen_route","eligibility","selection_environment_count","test_used","risk_certified")}
        elif args.command == "select-multi-arm":
            from .multi_environment_selection import select_arm_multi
            cfg=read_json(args.config)
            if set(cfg)!={"schema","baseline","tie_preference","arms"} or cfg["schema"]!="phm.multi-environment-arm-config.v1":
                raise ValueError("MULTI_ARM_CONFIG_SCHEMA")
            r=select_arm_multi(cfg["arms"],cfg["baseline"],args.output,tie_preference=cfg["tie_preference"])
            summary={k:r[k] for k in ("baseline_arm","chosen_arm","eligibility","test_used","risk_certified")}
        elif args.command == "select-matched-multi":
            from .multi_environment_selection import select_matched_multi
            cfg=read_json(args.config)
            if cfg.get("schema")=="phm.matched-multi-environment-config.v1":
                if set(cfg)!={"schema","baseline","tie_preference","arms","calibrations"}:
                    raise ValueError("MATCHED_MULTI_CONFIG_SCHEMA")
                selection_rule="reference_noninferiority"
            elif cfg.get("schema")=="phm.matched-multi-environment-config.v2":
                if set(cfg)!={"schema","baseline","tie_preference","arms","calibrations","selection_rule"}:
                    raise ValueError("MATCHED_MULTI_CONFIG_SCHEMA")
                selection_rule=cfg["selection_rule"]
            else:
                raise ValueError("MATCHED_MULTI_CONFIG_SCHEMA")
            r=select_matched_multi(cfg["arms"],cfg["calibrations"],cfg["baseline"],args.output,
                registry=args.registry,tie_preference=cfg["tie_preference"],selection_rule=selection_rule)
            summary={k:r[k] for k in ("baseline_arm","chosen_arm","chosen_route","eligibility","selection_environment_count","test_used")}
            summary["selection_rule"]=r.get("selection_rule",selection_rule)
        elif args.command == "select-reference-route":
            from .route_selection import select_route
            r=select_route(args.run,args.calibration,args.output,registry=args.registry,policy=read_json(args.policy) if args.policy else None)
            summary={k:r[k] for k in ("chosen_route","eligibility","test_used","risk_certified")}
        elif args.command == "evaluate-selected-reference":
            from .route_selection import evaluate_selected
            summary=evaluate_selected(args.run,args.selection,args.holdout,args.output,registry=args.registry)
        elif args.command == "predict-selected-reference":
            from .route_selection import predict_selected
            summary=predict_selected(args.run,args.selection,args.csv,args.passport,args.domain);write_json(args.output,summary)
        elif args.command == "design-reference-random":
            from .reference_design import design_reference
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report=design_reference(args.dataset,args.binding,args.policy,args.output,resume=args.resume,
                on_trial_complete=progress,proposal_strategy="random_control")
            summary={k:report[k] for k in ("status","fingerprint","candidate_count","deployment_route","proposal_strategy","test_used")}
        elif args.command == "design-reference-optuna":
            from .reference_design import design_reference
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report=design_reference(args.dataset,args.binding,args.policy,args.output,resume=args.resume,
                on_trial_complete=progress,proposal_strategy="optuna_tpe")
            summary={k:report[k] for k in ("status","fingerprint","candidate_count","deployment_route","proposal_strategy","test_used")}
        elif args.command == "design-reference-portfolio":
            from .reference_design import design_reference
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report=design_reference(args.dataset,args.binding,args.policy,args.output,resume=args.resume,
                on_trial_complete=progress,proposal_strategy="portfolio_feedback_optuna")
            summary={k:report[k] for k in ("status","fingerprint","candidate_count","deployment_route","proposal_strategy","test_used")}
        elif args.command == "design-reference":
            from .reference_design import design_reference
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report=design_reference(args.dataset,args.binding,args.policy,args.output,resume=args.resume,on_trial_complete=progress)
            summary={k:report[k] for k in ("status","fingerprint","candidate_count","deployment_route","test_used")}
        elif args.command == "evaluate-reference":
            from .reference_design import evaluate_reference
            report=evaluate_reference(args.run,args.holdout,args.output)
            summary={k:report[k] for k in ("deployment_route","metrics","counts")}
        elif args.command == "predict-reference":
            from .reference_design import predict_reference
            summary=predict_reference(args.run,args.csv,args.passport,args.domain);write_json(args.output,summary)
        elif args.command == "generate-multicomponent":
            from .adapters.sat_multicomponent import generate
            def progress(record):
                print(json.dumps({"domain":record["domain"],"split":record["sample"]["split"],"case":record["sample"]["id"],
                                  "component":record["root_cause_truth"]["component"],"mechanism":record["root_cause_truth"]["mechanism"]}),
                      file=sys.stderr,flush=True)
            summary=generate(args.config,args.output,on_case=progress)
        elif args.command == "prepare-multicomponent":
            from .multicomponent_features import prepare_section
            report=prepare_section(args.campaign,args.domain,args.section,args.output,args.representation_version)
            summary={"status":"PREPARED","cases":len(report["samples"]),"dataset_id":report["dataset_id"],"representation_version":args.representation_version}
        elif args.command == "evaluate-hierarchical-diagnosis":
            from .hierarchical_diagnosis import evaluate_feature_dataset
            summary=evaluate_feature_dataset(args.domain,args.dataset,args.risk_evaluation,args.output)
        elif args.command == "generate-nonadcs-context":
            from .family_pipeline import generate_with_passports
            def progress(record):
                print(json.dumps({"domain":record["domain"],"split":record["sample"]["split"],"case":record["sample"]["id"]}),file=sys.stderr,flush=True)
            summary=generate_with_passports(args.config,args.output,on_case=progress)
        elif args.command == "prepare-family":
            from .family_pipeline import prepare_family_section
            report=prepare_family_section(args.campaign,args.domain,args.section,args.mode,args.output)
            summary={"status":"PREPARED","cases":len(report["samples"]),"dataset_id":report["dataset_id"]}
        elif args.command == "prepare-family-thermal-v2":
            from .thermal_family_v2 import prepare_section
            if args.no_blind and args.section != "development":
                raise ValueError("THERMAL_V2_DEVELOPMENT_ONLY")
            report=prepare_section(args.campaign,args.section,args.output)
            summary={"status":"PREPARED","cases":len(report["samples"]),"dataset_id":report["dataset_id"],"representation":"family.thermal.rolling.v2"}
        elif args.command == "fit-thermal-v2-forest":
            from .thermal_family_v2_model import fit_json_forest
            summary=fit_json_forest(args.train_dataset,args.policy,args.output)
        elif args.command == "predict-thermal-v2-forest":
            from .thermal_family_v2_model import predict_raw
            summary=predict_raw(args.model,args.csv,args.passport)
            write_json(args.output,summary)
        elif args.command == "export-family-pipeline":
            from .family_pipeline import attach_pipeline
            summary=attach_pipeline(args.selected,args.domain,args.mode,args.output)
        elif args.command == "predict-family-pipeline":
            from .family_pipeline import predict_pipeline
            summary=predict_pipeline(args.pipeline,args.csv,args.passport);write_json(args.output,summary)
        elif args.command == "design-family":
            from .family_design import design_family
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report = design_family(args.dataset,args.binding,args.policy,args.output,resume=args.resume,on_trial_complete=progress)
            summary = {k:report[k] for k in ("status","fingerprint","candidate_count","generated_graphs","distinct_topologies","test_used","claim_status")}
        elif args.command == "generate-nonadcs":
            from .adapters.sat_nonadcs import generate_nonadcs
            def progress(record):
                print(json.dumps({"domain":record["domain"], "split":record["sample"]["split"], "label":record["sample"]["label"]}),file=sys.stderr,flush=True)
            summary = generate_nonadcs(args.config,args.output,on_case=progress)
        elif args.command == "design-robust":
            from .robust_engine import design_robust
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            report = design_robust(args.dataset,args.request,args.binding,args.output,resume=args.resume,on_trial_complete=progress)
            summary = {k:report[k] for k in ("status","fingerprint","generated_candidates","training_trials","selected_id","test_used","claim_status")}
        elif args.command == "evaluate-robust":
            from .robust_engine import evaluate_robust
            report = evaluate_robust(args.run,args.holdout,args.output)
            summary = {k:v for k,v in report.items() if k not in {"predictions","base_samples"}}
        elif args.command == "predict-robust":
            from .robust_engine import predict_robust
            summary = predict_robust(args.run,args.packet)
            write_json(args.output,summary)
        elif args.command == "generate-sat-mechanism":
            from .adapters.sat_rw_mechanism import generate_mechanism_campaign
            summary = generate_mechanism_campaign(args.config,args.output)
        elif args.command == "prepare-sat-mechanism":
            from .adapters.sat_rw_mechanism import prepare_mechanism
            report = prepare_mechanism(args.campaign,args.output,args.project_root)
            summary = {k:v for k,v in report.items() if k != "source_records"}
        elif args.command == "design-mechanism":
            from .mechanism_design import mechanism_design
            def progress(record):
                print(json.dumps({k:record[k] for k in ("candidate","seed","status")}),file=sys.stderr,flush=True)
            summary = mechanism_design(args.dataset,args.program,args.request,args.output,resume=args.resume,on_trial_complete=progress)
        elif args.command == "ablate":
            from .ablation import ablate
            summary = ablate(args.dataset, args.design, args.output)
        elif args.command == "design":
            from .synthesis import design
            def progress(record):
                print(json.dumps({k: record[k] for k in ("candidate", "seed", "status")}), file=sys.stderr, flush=True)
            report = design(args.dataset, args.request, args.output, resume=args.resume, on_trial_complete=progress)
            summary = {k: report[k] for k in ("status", "fingerprint", "candidate_count", "generated_graphs", "distinct_topologies", "test_used", "claim_status")}
        elif args.command == "prepare-sat":
            from .adapters.sat_rw import prepare as run
            report = run(args.campaign, args.output)
            summary = {k: v for k,v in report.items() if k != "source_records"}
        elif args.command == "plan":
            from .data import Dataset
            from .engine import config_check
            from .registry import default_registry
            d = Dataset(args.dataset, allowed_splits={"train", "validation"})
            c = read_json(args.config)
            result = config_check(c, len(d.contract["channels"]), len(d.contract["classes"]), default_registry())
            summary = {"status": "PLANNED", "candidate_count": len(result),
                       "trial_count": sum(x["feasible"] for x in result)*len(c["seeds"]),
                       "candidates": result, "training_executed": False}
            if Path(args.output).exists():
                raise FileExistsError("PLAN_OUTPUT_EXISTS")
            write_json(args.output, summary)
        elif args.command == "fit":
            from .engine import fit
            def progress(record: dict) -> None:
                print(json.dumps({k: record[k] for k in ("candidate", "seed", "status")}), file=sys.stderr, flush=True)
            report = fit(args.dataset, args.config, args.output, resume=args.resume, on_trial_complete=progress)
            summary = {k: report[k] for k in ("status", "fingerprint", "best_neural_candidate", "best_non_neural_baseline", "test_used", "claim_status")}
        elif args.command == "evaluate":
            from .engine import evaluate
            report = evaluate(args.run, args.holdout, args.output)
            summary = {k: v for k,v in report.items() if k != "predictions"}
        else:
            from .engine import predict
            summary = predict(args.run, args.csv, read_json(args.units))
            write_json(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if summary.get("status") in {"NO_FEASIBLE_NEURAL_CANDIDATE", "NO_FEASIBLE_DESIGNED_CANDIDATE", "INCOMPLETE_TRIALS", "INCOMPLETE_SEARCH"} else 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
