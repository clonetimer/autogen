from phm_design.reference_design import _feedback_parent_records


def _r(g,mode,f1,i):
    return {"id":i,"kind":"generated","status":"COMPLETED","mode":mode,"graph_sha256":g,
            "mean_validation_macro_f1":f1,"seed_std_macro_f1":0.01,"description":{"parameter_count":100}}

def test_legacy_bounded_parent_behavior_is_preserved():
    records=[_r("a","raw",0.95,"ar"),_r("a","bounded",0.80,"ab"),_r("a","unbounded",0.90,"au")]
    got=_feedback_parent_records(records,"bounded_only")
    assert [x["id"] for x in got]==["ab"]

def test_best_generated_route_selects_best_route_per_topology():
    records=[_r("a","raw",0.95,"ar"),_r("a","bounded",0.80,"ab"),_r("a","unbounded",0.90,"au"),
             _r("b","raw",0.70,"br"),_r("b","bounded",0.91,"bb"),_r("b","unbounded",0.89,"bu")]
    got=_feedback_parent_records(records,"best_generated_route")
    assert {x["graph_sha256"]:x["id"] for x in got}=={"a":"ar","b":"bb"}
