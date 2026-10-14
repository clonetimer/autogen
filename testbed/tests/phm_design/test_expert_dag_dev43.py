import pytest
from phm_design.expert_dag import search_multi_environment


def cfg(**policy):
    return {"schema":"phm.expert-dag-search.v1","data_role":"development","retrospective_replay":False,
            "classes":["N","A"],"experts":["simple","complex"],"structures":["single_expert"],
            "constraints":{},"evidence_policy":{"require_disjoint_groups_and_cases":True,
                "min_environments":2,"min_groups_per_environment":2,
                "min_cases_per_class_per_environment":2,**policy}}


def make_env(prefix):
    rows=[]
    for g in range(2):
        for i,y in enumerate(("N","A")):
            rows.append({"id":f"{prefix}-{g}-{i}","group_id":f"{prefix}-group-{g}",
                         "truth":y,"experts":{"simple":[.9,.1] if y=="N" else [.1,.9],
                                                "complex":[.8,.2] if y=="N" else [.2,.8]}})
    return rows


def test_independent_evidence_passes_when_cases_and_groups_are_disjoint():
    rep=search_multi_environment({"a":make_env("a"),"b":make_env("b")},cfg())
    assert rep["qualification_eligible"] and rep["chosen_worst_macro_f1"]==1.0


def test_duplicates_are_not_counted_as_independent_environments():
    with pytest.raises(ValueError,match="OVERLAP"):
        search_multi_environment({"a":make_env("a"),"b":make_env("a")},cfg())


def test_shared_group_is_rejected_even_when_ids_are_new():
    a,b=make_env("a"),make_env("b")
    for row in b:
        if row["group_id"].endswith("-0"):row["group_id"]="a-group-0"
    with pytest.raises(ValueError,match="GROUP_OVERLAP"):
        search_multi_environment({"a":a,"b":b},cfg())


def test_minimum_per_class_and_group_is_enforced():
    a,b=make_env("a"),make_env("b")
    with pytest.raises(ValueError,match="EVIDENCE_TOO_SMALL"):
        search_multi_environment({"a":a[:-1],"b":b},cfg())


def test_requires_minimum_number_of_independent_environments():
    with pytest.raises(ValueError,match="INSUFFICIENT_ENVIRONMENTS"):
        search_multi_environment({"a":make_env("a"),"b":make_env("b")},cfg(min_environments=3))


def test_release_grade_metadata_required_only_when_policy_is_enabled():
    rows=[{"truth":"N","experts":{"simple":[.9,.1],"complex":[.8,.2]}},
          {"truth":"A","experts":{"simple":[.1,.9],"complex":[.2,.8]}}]
    with pytest.raises(ValueError,match="IDS_REQUIRED"):
        search_multi_environment({"a":rows,"b":rows},cfg())
