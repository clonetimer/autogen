from phm_design.feature_selection import arm_rank


def m(f1,rec):
    return {"macro_f1":f1,"recall_by_class":{"A":rec,"B":rec+.05},"cases":10,"accepted_cases":10,"accepted_errors":1,"accepted_correct":9}


def s(devf,calf,devr,calr):
    return {"chosen_route":"reference","development_metrics":{"reference":m(devf,devr)},"calibration_metrics":{"reference":m(calf,calr)}}


def test_feature_arm_rank_prioritizes_worst_environment_f1():
    a=s(.9,.7,.6,.6);b=s(.8,.8,.5,.5)
    assert arm_rank(b,1)<arm_rank(a,0)


def test_feature_arm_rank_uses_weakest_class_after_f1():
    a=s(.8,.8,.4,.4);b=s(.8,.8,.6,.6)
    assert arm_rank(b,1)<arm_rank(a,0)


def test_feature_arm_tie_uses_frozen_preference():
    a=s(.8,.8,.6,.6)
    assert arm_rank(a,0)<arm_rank(a,1)
