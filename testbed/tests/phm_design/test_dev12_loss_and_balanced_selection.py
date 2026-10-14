import copy
import numpy as np
import torch
from phm_design.engine import _training_loss,config_check
from phm_design.registry import default_registry
from phm_design.models import Shape
from phm_design.multi_environment_selection import choose_routes_balanced
from tests.phm_design.test_reference_dev3 import make_policy
from tests.phm_design.test_compositional_design import request


def metric(f1,recalls,errors=0,correct=10,cases=12):
    return {"macro_f1":f1,"recall_by_class":dict(zip(["C0","C1","C2"],recalls)),
            "cases":cases,"accepted_errors":errors,"accepted_correct":correct,
            "accepted_cases":errors+correct}


def test_old_training_schema_still_defaults_to_weighted_cross_entropy():
    training={"epochs":3,"batch_size":4,"learning_rate":1e-3,"weight_decay":0.,"patience":2}
    crit,name=_training_loss(training,np.array([4,2,2]),3)
    assert name=="weighted_cross_entropy"
    logits=torch.tensor([[2.,0.,-1.],[0.,2.,-1.]],requires_grad=True)
    y=torch.tensor([0,1])
    loss=crit(logits,y);assert torch.isfinite(loss);loss.backward()


def test_class_balanced_focal_is_finite_and_changes_loss():
    base={"epochs":3,"batch_size":4,"learning_rate":1e-3,"weight_decay":0.,"patience":2}
    focal={**base,"loss":{"name":"class_balanced_focal","beta":.999,"gamma":2.0}}
    f,name=_training_loss(focal,np.array([20,3,2]),3)
    assert name.startswith("class_balanced_focal")
    logits=torch.tensor([[2.,0.,-1.],[0.,2.,-1.],[1.,0.,.5]],requires_grad=True)
    y=torch.tensor([0,1,2])
    loss=f(logits,y);assert torch.isfinite(loss);loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_config_check_accepts_focal_without_breaking_old_schema():
    cfg={"schema":"phm.search.v1","window":8,
         "candidates":[{"family":"mlp","parameters":{"widths":[8],"representation":"statistics"}}],
         "seeds":[11,23],
         "training":{"epochs":3,"batch_size":4,"learning_rate":1e-3,"weight_decay":0.,"patience":2,
                     "loss":{"name":"class_balanced_focal","beta":.999,"gamma":2.0}},
         "budget":{"max_parameters_per_model":50000,"max_ensemble_parameters":100000,
                   "max_trials":10,"max_trial_seconds":10},
         "selection":"mean_seed_validation_macro_f1","abstention":{"min_uncalibrated_score":.6}}
    rec=config_check(cfg,2,3,default_registry())
    assert rec[0]["feasible"] is True
    old=copy.deepcopy(cfg);old["training"].pop("loss")
    rec2=config_check(old,2,3,default_registry())
    assert rec2[0]["feasible"] is True


def test_balanced_selector_escapes_reference_anchor_lockin():
    # Reference is perfect on one class but collapses another; fixed is balanced in all envs.
    env={
      "development":{
        "reference":metric(.57,[1.0,.75,.125]),
        "fixed":metric(.77,[.75,.875,.625]),
        "raw":metric(.74,[.625,.625,.625])},
      "selection_a":{
        "reference":metric(.72,[1.0,.625,.5]),
        "fixed":metric(.80,[.75,.625,.75]),
        "raw":metric(.79,[.875,.5,.75])},
      "selection_b":{
        "reference":metric(.67,[1.0,.75,.375]),
        "fixed":metric(.72,[.625,.625,.5]),
        "raw":metric(.67,[.625,.5,.625])}}
    chosen,eligible,ranks=choose_routes_balanced(env)
    assert chosen=="fixed"
    assert all(eligible.values())
    assert ranks["fixed"]<ranks["reference"]
