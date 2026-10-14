import json, sys
from pathlib import Path

from phm_design.algorithm_dag_cli import main

CLASSES=["NOMINAL","FRICTION","JAM","TORQUE_LOSS"]


def _rows(tag="a"):
    base=[
      ("NOMINAL",[.95,.02,.02,.01],[.001,.002,.001],-1),
      ("NOMINAL",[.92,.03,.03,.02],[.002,.001,.001],-1),
      ("FRICTION",[.01,.96,.02,.01],[.001,.08,.002],1),
      ("JAM",[.01,.02,.96,.01],[.09,.001,.001],0),
      ("TORQUE_LOSS",[.72,.02,.01,.25],[.001,.002,.06],2),
      ("TORQUE_LOSS",[.64,.03,.02,.31],[.001,.055,.002],1),
    ]
    return [{"truth":t,"neural_scores":s,"mechanism_values":m,"truth_location":l,
             "group_id":f"{tag}-{i}","source_case":f"development-{tag}-{i}"}
            for i,(t,s,m,l) in enumerate(base)]


def _config():
    return {
      "schema":"phm.algorithm-dag-search.v1","classes":CLASSES,"nominal_class":"NOMINAL",
      "data_role":"development","retrospective_replay":False,
      "topologies":["physics_gate_neural_fault_physical_localizer"],
      "physical_gate_thresholds":[.005,.01,.02],
      "constraints":{"min_macro_f1":.9,"min_fault_detection_recall":.95,
                     "max_nominal_false_alarm_rate":.1,"min_component_localization_accuracy":.95},
      "objective_weights":{"macro_f1":1.0,"fault_detection_recall":.5,
                           "component_localization_accuracy":.25,"complexity_penalty":1e-4},
    }


def _write(path: Path, obj):
    path.write_text(json.dumps(obj),encoding="utf-8")


def test_legacy_single_cli_is_backward_compatible(tmp_path, monkeypatch):
    rec=tmp_path/"records.json"; cfg=tmp_path/"config.json"; out=tmp_path/"single"
    _write(rec,_rows()); _write(cfg,_config())
    monkeypatch.setattr(sys,"argv",["phm-design-algorithm-dag","--records",str(rec),"--config",str(cfg),"--output",str(out)])
    assert main()==0
    assert json.loads((out/"search_report.json").read_text())["qualification_eligible"] is True
    assert (out/"selected_algorithm_dag.json").is_file()


def test_multi_freeze_and_frozen_evaluate_workflow(tmp_path, monkeypatch):
    a=tmp_path/"a.json"; b=tmp_path/"b.json"; cfg=tmp_path/"config.json"
    _write(a,_rows("a")); _write(b,_rows("b")); _write(cfg,_config())
    multi=tmp_path/"multi"
    monkeypatch.setattr(sys,"argv",["phm-design-algorithm-dag","--mode","multi",
        "--environment",f"a={a}","--environment",f"b={b}","--config",str(cfg),"--output",str(multi)])
    assert main()==0
    search=json.loads((multi/"search_report.json").read_text())
    assert search["qualification_eligible"] is True
    assert search["test_or_holdout_used_for_selection"] is False

    freeze=tmp_path/"freeze"
    monkeypatch.setattr(sys,"argv",["phm-design-algorithm-dag","--mode","freeze",
        "--environment",f"a={a}","--environment",f"b={b}","--config",str(cfg),
        "--candidate",str(multi/"selected_algorithm_dag.json"),"--search-report",str(multi/"search_report.json"),
        "--blind-seed","9990001","--campaign-id","unit-blind-9990001","--output",str(freeze)])
    assert main()==0
    frozen=json.loads((freeze/"preblind_freeze.json").read_text())
    assert frozen["status"]=="FROZEN" and frozen["test_used_for_selection"] is False

    evaluation=tmp_path/"evaluation"
    monkeypatch.setattr(sys,"argv",["phm-design-algorithm-dag","--mode","evaluate",
        "--records",str(a),"--candidate",str(multi/"selected_algorithm_dag.json"),
        "--config",str(cfg),"--output",str(evaluation)])
    assert main()==0
    result=json.loads((evaluation/"evaluation.json").read_text())
    assert result["strict_gate_pass"] is True
    assert result["truth_inputs_used_at_runtime"] is False
