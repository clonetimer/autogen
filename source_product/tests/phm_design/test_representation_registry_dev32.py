from phm_design.representation_registry import representation_candidates, multidomain_representation_plan


def ids(domain, **kw):
    return [x.id for x in representation_candidates(domain, **kw)]


def test_multicomponent_capability_matrix_is_domain_legal():
    assert ids("eps") == ["raw", "mechanism.v1"]
    assert ids("data") == ["raw", "mechanism.v1"]
    assert ids("thermal") == ["raw", "mechanism.v1", "mechanism.v2"]


def test_family_prerequisites_are_explicit_not_inferred():
    assert ids("eps", task="family") == ["family.context.v1", "family.mechanism.v1"]
    assert ids("eps", task="family", available_artifacts=["commissioning_profile"]) == [
        "family.context.v1", "family.mechanism.v1", "family.commissioned.v1"
    ]
    assert ids("eps", task="family", available_artifacts=["commissioning_profile", "hybrid_profile"]) == [
        "family.context.v1", "family.mechanism.v1", "family.commissioned.v1", "family.hybrid.v1"
    ]


def test_plan_is_cross_domain_and_does_not_select_a_winner():
    p = multidomain_representation_plan(task="multicomponent")
    assert p["domains"] == ["eps", "thermal", "data"]
    assert p["winner_selected"] is False
    assert p["test_used"] is False
    assert [x["id"] for x in p["arms"]["thermal"]] == ["raw", "mechanism.v1", "mechanism.v2"]


def test_invalid_domain_or_task_fail_closed():
    import pytest
    with pytest.raises(ValueError, match="UNKNOWN_DOMAIN"):
        representation_candidates("rw")
    with pytest.raises(ValueError, match="UNKNOWN_REPRESENTATION_TASK"):
        representation_candidates("eps", task="unknown")


def test_cli_plans_multidomain_representations(tmp_path, monkeypatch):
    import json, sys
    from phm_design.__main__ import main
    out=tmp_path/'plan.json'
    monkeypatch.setattr(sys,'argv',['phm_design','plan-representations','--task','multicomponent','--output',str(out)])
    assert main()==0
    p=json.loads(out.read_text())
    assert [x['id'] for x in p['arms']['eps']] == ['raw','mechanism.v1']
    assert [x['id'] for x in p['arms']['thermal']] == ['raw','mechanism.v1','mechanism.v2']
    assert p['winner_selected'] is False and p['test_used'] is False


def test_codesign_plan_maps_existing_multicomponent_execution_paths():
    from phm_design.representation_registry import codesign_arm_plan
    eps=codesign_arm_plan('eps')
    assert [x.representation_id for x in eps] == ['raw','mechanism.v1']
    assert eps[0].prepare_command is None
    assert eps[0].prepare_mode == 'use_native_dataset_directly'
    assert eps[1].prepare_command == 'prepare-multicomponent --representation-version v1'
    assert all(x.architecture_command == 'design-reference' for x in eps)
    thermal=codesign_arm_plan('thermal')
    assert thermal[-1].representation_id == 'mechanism.v2'
    assert thermal[-1].prepare_command == 'prepare-multicomponent --representation-version v2'


def test_codesign_plan_maps_family_execution_paths_and_prerequisites():
    from phm_design.representation_registry import codesign_arm_plan
    arms=codesign_arm_plan('data',task='family',available_artifacts=['commissioning_profile','hybrid_profile'])
    assert [x.representation_id for x in arms] == [
        'family.context.v1','family.mechanism.v1','family.commissioned.v1','family.hybrid.v1'
    ]
    assert [x.prepare_command for x in arms] == [
        'prepare-family --mode context','prepare-family --mode mechanism','prepare-commissioned-family','prepare-hybrid'
    ]
    assert all(x.architecture_command == 'design-family' for x in arms)
    assert arms[-1].requires == ('commissioning_profile','hybrid_profile')


def test_multidomain_codesign_plan_is_preperformance_only():
    from phm_design.representation_registry import multidomain_codesign_plan
    p=multidomain_codesign_plan()
    assert p['representation_winner_selected'] is False
    assert p['architecture_winner_selected'] is False
    assert p['risk_used'] is False and p['test_used'] is False
    assert len(p['arms']['eps']) == 2
    assert len(p['arms']['thermal']) == 3
    assert len(p['arms']['data']) == 2


def test_cli_plans_executable_codesign_arms(tmp_path,monkeypatch):
    import json,sys
    from phm_design.__main__ import main
    out=tmp_path/'codesign.json'
    monkeypatch.setattr(sys,'argv',['phm_design','plan-codesign-arms','--task','multicomponent','--output',str(out)])
    assert main()==0
    p=json.loads(out.read_text())
    assert p['arms']['eps'][0]['prepare_mode']=='use_native_dataset_directly'
    assert p['arms']['thermal'][-1]['architecture_command']=='design-reference'
    assert p['test_used'] is False
