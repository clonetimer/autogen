"""R17 offline pairing and information-window audit tests."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paired_observation_audit import _geometry, analyze_pairs


def test_geometry_identical_mechanisms_are_collinear():
    d=_geometry(np.array([[1.,2.],[2.,3.]]),np.array([[1.,2.],[2.,3.]]))
    assert np.isclose(d['cosine'],1.0)
    assert np.isclose(d['relative_separation'],0.0)


def test_geometry_discriminable_direction():
    d=_geometry(np.array([[1.,0.],[2.,0.]]),np.array([[0.,1.],[0.,2.]]))
    assert abs(d['cosine'])<1e-8
    assert d['relative_separation']>1.0


def test_geometry_zero_response_is_unknown():
    d=_geometry(np.zeros((3,2)),np.ones((3,2)))
    assert d['cosine'] is None and d['relative_separation'] is None


def test_real_native_development_data_audited():
    source=Path(__file__).parent/'fixtures/r16_opened'
    v=analyze_pairs(plan_path=source/'FROZEN_PLAN.json', source_root=source)
    assert v['no_model_fit'] and v['no_test_or_holdout']
    assert v['input_world_groups']==18
    assert len(v['world_audit'])==18
    assert all(len(x['case_ids'])==3 for x in v['world_audit'])
    assert v['summary']['48']['two_sensor']['worlds_with_nonzero_both_responses']==18
