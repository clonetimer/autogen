import numpy as np
from phm_design.hierarchical_diagnosis import diagnosis_from_scores,hierarchical_metrics,ontology
from phm_design.family_pipeline import feature_contract


def seq(domain):
    c=feature_contract(domain,"mechanism")
    x=np.zeros((12,len(c["channels"])),dtype=np.float32)
    names=[z["name"] for z in c["channels"]]
    if domain=="eps":
        x[:,names.index("solar_loss_running_mean")]=np.linspace(0,.25,12)
        x[:,names.index("solar_loss_fraction")]=np.linspace(0,.3,12)
        x[:,names.index("load_innovation_w")]=np.linspace(0,5,12)
    elif domain=="thermal":
        x[:,names.index("radiation_loss_slope")]=np.linspace(0,.2,12)
        x[:,names.index("unexplained_heat_intercept_w")]=np.linspace(0,4,12)
        x[:,names.index("energy_residual_running_w")]=np.linspace(0,3,12)
    else:
        x[:,names.index("service_loss_fraction")]=np.linspace(0,.3,12)
        x[:,names.index("contact_innovation_running_bps")]=np.linspace(0,1000,12)
        x[:,names.index("generation_excess_running_bps")]=np.linspace(0,800,12)
    return c,x


def test_eps_fault_report_is_explicitly_functional_not_independent_localization():
    c,x=seq("eps")
    r=diagnosis_from_scores("eps",[.05,.9,.05],x,c,.8)
    assert r["status"]=="PREDICTED"
    assert r["detection"]=="FAULT"
    assert r["functional_location"]=="solar_generation_chain"
    assert r["mechanism"]=="solar_conversion_efficiency_loss"
    assert r["independent_component_localization"] is False
    assert r["causal_root_cause_identified"] is False
    assert {e["feature"] for e in r["evidence"]}=={"solar_loss_running_mean","solar_loss_fraction"}
    assert min(e["tail_direction_consistency"] for e in r["evidence"])==1.0


def test_fail_closed_hides_location_and_root_cause_but_preserves_argmax():
    c,x=seq("thermal")
    r=diagnosis_from_scores("thermal",[.05,.9,.05],x,c,None)
    assert r["status"]=="ABSTAIN_RISK_POLICY"
    assert r["closed_set_argmax"]=="EMISSIVITY_LOSS"
    assert r["detection"]=="UNKNOWN"
    assert r["functional_location"] is None
    assert r["root_cause_hypothesis"] is None


def test_nominal_accepted_report():
    c,x=seq("data")
    r=diagnosis_from_scores("data",[.91,.05,.04],x,c,.9)
    assert r["status"]=="PREDICTED"
    assert r["detection"]=="NOMINAL"
    assert r["functional_location"] is None
    assert r["root_cause_hypothesis"] is None


def test_hierarchical_metrics_binary_detection_and_functional_location():
    truth=["NOMINAL","SOLAR_EFFICIENCY_LOSS","EXTRA_BUS_LOAD","EXTRA_BUS_LOAD"]
    pred =["EXTRA_BUS_LOAD","SOLAR_EFFICIENCY_LOSS","NOMINAL","SOLAR_EFFICIENCY_LOSS"]
    m=hierarchical_metrics("eps",truth,pred)
    assert m["binary_detection_confusion"]=={"tn":0,"fp":1,"fn":1,"tp":2}
    assert m["fault_detection_recall"]==2/3
    assert m["nominal_window_false_alarm_rate"]==1.0
    assert m["functional_localization_accuracy_on_fault_cases"]==1/3
    assert m["mechanism_accuracy_on_fault_cases"]==1/3
    assert m["localization_is_independent_task"] is False


def test_ontology_covers_native_classes_exactly():
    for domain in ("eps","thermal","data"):
        assert list(ontology(domain)["classes"])==feature_contract(domain,"mechanism")["classes"]


def test_multicomponent_task_contract_is_supported_without_relaxing_schema():
    from phm_design.data import validate_contract
    from phm_design.adapters.sat_multicomponent import contract
    for domain in ("eps","thermal","data"):
        c=contract(domain)
        assert c["task"]=="fixed_length_fault_mechanism_and_component_localization"
        validate_contract(c)


def test_multicomponent_metrics_separate_location_from_mechanism():
    from phm_design.hierarchical_diagnosis import multicomponent_metrics
    truth=["NOMINAL","SOLAR_A_EFFICIENCY_LOSS","SOLAR_B_EFFICIENCY_LOSS","LOAD_A_EXTRA","LOAD_B_EXTRA"]
    pred =["NOMINAL","LOAD_A_EXTRA","SOLAR_A_EFFICIENCY_LOSS","LOAD_B_EXTRA","LOAD_B_EXTRA"]
    m=multicomponent_metrics("eps",truth,pred)
    assert m["fault_detection_recall"]==1.0
    assert m["nominal_window_false_alarm_rate"]==0.0
    # A/B correct for cases 1,3,4 -> 3/4, despite mechanism mistakes.
    assert m["component_localization_accuracy_on_fault_cases"]==0.75
    # A/B slot is correct only for cases 1 and 4.
    assert m["component_slot_accuracy_on_fault_cases"]==0.5
    # Exact component instance matches only case 4.
    assert m["exact_component_instance_accuracy_on_fault_cases"]==0.25
    # mechanism type is correct for cases 2,3,4 -> 3/4.
    assert m["mechanism_accuracy_on_fault_cases"]==0.75
    assert m["joint_fault_class_accuracy_on_fault_cases"]==0.25
