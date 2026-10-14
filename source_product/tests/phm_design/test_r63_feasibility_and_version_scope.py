"""R63: fail-closed task acceptance and no inherited Stable qualification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phm_design import __version__
from phm_design import platform_limited
from phm_design.platform_features import FIELDS
from phm_design.task_architectures import (
    TaskAcceptancePolicy, TaskDesignRejected, design,
)
from test_task_architectures_08 import fixture_train


def test_current_version_cannot_inherit_frozen_release_certificate():
    c = platform_limited.contract()
    assert __version__.startswith('0.8.0.dev') and __version__ != '0.7.0'
    assert c['active_package_version'] == __version__
    assert c['bundled_legacy_release_version'] == '0.7.0'
    assert c['release_tier'] == 'DEVELOPMENT'
    assert c['qualification_applies_to_current_package'] is False
    assert c['current_package_qualified_cells'] == []
    assert platform_limited._checked_release_certificate()['contract_id'] == c['contract_id']
    allowed = platform_limited.allowed_use_case('EPS:conventional')
    assert allowed['status'] == 'LEGACY_MODEL_DEVELOPMENT_ONLY'
    assert allowed['qualified'] is False


def test_dev_bundle_never_marks_predict_as_qualified():
    fixtures=Path(__file__).parent/'fixtures/limited_opened'
    result=platform_limited.predict_bundle('EPS','EPS:conventional',fixtures/'EPS.csv',healthy_prefix_confirmed=True)
    assert result['scope_status']=='LEGACY_MODEL_DEVELOPMENT_ONLY'
    assert result['qualified'] is False
    assert 'QUALIFIED_PREDICTION' not in result['status']
    assert result['active_package_version']==__version__


def test_default_policy_rejects_bad_oof_even_if_training_can_fit():
    rows,y,g,r,h=fixture_train()
    with pytest.raises(TaskDesignRejected,match='NO_CANDIDATE_MEETS_TASK_ACCEPTANCE_POLICY') as exc:
        design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization'],
               labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h,
               candidate_specs=[('temporal','linear','direct')])
    audit=exc.value.audit
    assert audit['decision']=='REJECT_NO_ELIGIBLE_CANDIDATE'
    assert audit['model_delivered'] is False
    assert audit['selected_train_only'] is None
    assert audit['eligible_candidates']==0
    assert len(audit['candidate_results'])==1
    assert audit['candidate_results'][0]['status']=='COMPLETE'
    assert audit['candidate_results'][0]['acceptance']['failed_gates']
    assert audit['training_only_screen_no_sealed_validation']


def test_looser_explicit_research_contract_allows_model_not_qualification():
    rows,y,g,r,h=fixture_train()
    out=design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization'],
           labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h,
           acceptance_policy=TaskAcceptancePolicy(0,1,0,0,8),
           candidate_specs=[('temporal','linear','direct')])
    assert out.audit['decision']=='RESEARCH_CANDIDATE_ELIGIBLE'
    assert out.audit['model_delivered']
    assert out.audit['not_deployment_qualified']
    assert out.diagnose(rows[0],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)['qualified'] is False


def test_policy_bounds_finite_and_world_count_required():
    for policy in [dict(accuracy_min=float('nan')),dict(worst_fault_recall_min=1.01),
                   dict(healthy_false_alarm_max=-0.01),dict(min_independent_worlds=3)]:
        with pytest.raises(ValueError,match='INVALID_ACCEPTANCE_POLICY'):
            TaskAcceptancePolicy(**policy)
    result=TaskAcceptancePolicy().evaluate({'accuracy':1.,'healthy_false_alarm_rate':0.,
       'worst_fault_recall':1.,'worst_world_accuracy':1.,'independent_worlds':4})
    assert not result['eligible']
    assert result['failed_gates'][0]['metric']=='independent_worlds'


def test_full_historical_r62_8world_train_oof_has_no_eligible_candidate():
    # Original R62 training OOF records are immutable audit fixtures.
    evidence=Path(__file__).parent/'fixtures/r62_train_audits'
    for domain in ('Thermal_localization','EPS_localization'):
        original=json.loads((evidence/f'{domain}_TRAIN_AUDIT.json').read_text())
        results=[TaskAcceptancePolicy().evaluate(r['oof']) for r in original['candidate_results'] if r['status']=='COMPLETE']
        assert results and not any(r['eligible'] for r in results)


def test_default_policy_accepts_clear_group_generalizing_pattern_without_certifying():
    rows,y,g,r,h=fixture_train()
    for i,label in enumerate(y):
        a=rows[i].copy()
        if label in (1,3):a[65:,4]+=(8 if label==1 else 16)
        if label in (2,4):a[65:,5]+=(8 if label==2 else 16)
        rows[i]=a
    result=design(domain='Thermal_localization',observations=rows,
          fields=FIELDS['Thermal_localization'],labels=y,world_ids=g,roles=r,
          healthy_prefix_confirmed=h,candidate_specs=[('physical','extra','direct')])
    assert result.audit['decision']=='RESEARCH_CANDIDATE_ELIGIBLE'
    assert result.audit['selection_oof']['worst_fault_recall'] >= .8
    assert result.audit['selection_oof']['healthy_false_alarm_rate'] <= .1
    assert result.audit['not_deployment_qualified'] is True
    assert result.diagnose(rows[0],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)['qualified'] is False
