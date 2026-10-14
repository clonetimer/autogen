import json
from pathlib import Path
from phm_design.route_selection import _budget_terminal_design


def _write(root:Path, status, trials):
    root.mkdir(parents=True,exist_ok=True)
    (root/'design_report.json').write_text(json.dumps({'status':status}))
    p=root/'generation_00/run';p.mkdir(parents=True)
    (p/'search_report.json').write_text(json.dumps({'candidate_results':[{'status':'INCOMPLETE_SEEDS','trials':trials}]}))
    return json.loads((root/'design_report.json').read_text())


def test_budget_terminal_timeout_is_eligible_source_state(tmp_path):
    rep=_write(tmp_path/'r','INCOMPLETE_TRIALS',[
        {'status':'FAILED','reason':'TRIAL_TIME_BUDGET_EXCEEDED'},
        {'status':'FAILED','reason':'TRIAL_TIME_BUDGET_EXCEEDED'}])
    assert _budget_terminal_design(tmp_path/'r',rep) is True


def test_non_budget_failure_remains_fatal(tmp_path):
    rep=_write(tmp_path/'r','INCOMPLETE_TRIALS',[{'status':'FAILED','reason':'NONFINITE_LOSS'}])
    assert _budget_terminal_design(tmp_path/'r',rep) is False


def test_interrupted_or_missing_trials_remain_fatal(tmp_path):
    rep=_write(tmp_path/'r','INCOMPLETE_TRIALS',[{'status':'RUNNING','reason':None}])
    assert _budget_terminal_design(tmp_path/'r',rep) is False


def test_completed_design_remains_accepted(tmp_path):
    rep={'status':'COMPLETED'}
    assert _budget_terminal_design(tmp_path,rep) is True
