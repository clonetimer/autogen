import torch
import pytest
from phm_design.literature_baselines import (
    CATALOG,Grid3DResidualClassifier,TransformerClassifier,MultiScaleConvTransformer,
    DynamicGraphGRUClassifier,class_balanced_focal,parameter_count)


@pytest.mark.parametrize("cls",[Grid3DResidualClassifier,TransformerClassifier,MultiScaleConvTransformer,DynamicGraphGRUClassifier])
def test_literature_baseline_forward_shape(cls):
    m=cls(18,5);x=torch.randn(4,64,18);y=m(x)
    assert y.shape==(4,5)
    assert torch.isfinite(y).all()
    assert 1000<parameter_count(m)<250000


def test_class_balanced_focal_is_finite_and_differentiable():
    logits=torch.randn(8,5,requires_grad=True);target=torch.tensor([0,0,0,0,1,2,3,4])
    loss=class_balanced_focal(logits,target,[4,1,1,1,1])
    assert torch.isfinite(loss);loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_catalog_explicitly_marks_adaptations_not_exact_reproductions():
    assert set(CATALOG)=={"grid3d_res","multiscale_conv_transformer","dynamic_graph_gru","transformer"}
    for rec in CATALOG.values():
        assert "task_difference" in rec and rec["task_difference"]


def test_literature_controls_are_registered_and_under_dev12_budget():
    from phm_design.registry import default_registry
    from phm_design.models import Shape
    r=default_registry();s=Shape(128,19,5)
    specs=[
      {"family":"grid3d_res","parameters":{"width":12}},
      {"family":"transformer_ts","parameters":{"d_model":40,"heads":4,"layers":2}},
      {"family":"multiscale_conv_transformer","parameters":{"d_model":48}},
    ]
    for spec in specs:
        d=r.describe(spec,s)
        assert d["parameter_count"]<=50000
        m=r.build(spec,s,11,50000)
        assert m(torch.zeros(2,128,19)).shape==(2,5)


def test_advanced_dev12_policy_enables_attention_tcn_and_strong_controls():
    import json
    from pathlib import Path
    from phm_design.family_design import make_request
    from phm_design.adapters.sat_multicomponent import binding
    root=Path(__file__).resolve().parents[2]
    p=json.loads((root/"configs/phm/reference_design_advanced_dev12.json").read_text())
    req=make_request(binding("thermal"),p["base_design"])
    assert {"res_tcn","self_attention"}<=set(req["grammar"]["layer_ops"])
    assert {c["family"] for c in req["controls"]}>={"grid3d_res","transformer_ts","multiscale_conv_transformer"}
