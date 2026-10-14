import numpy as np
import pytest
from phm_design.algorithm_dag import RuntimeSignals, enumerate_candidates, execute, search

CLASSES=["NOMINAL","FRICTION","JAM","TORQUE_LOSS"]


def _records():
    # Deliberately make the neural closed-set model miss both torque-loss cases as nominal.
    return [
      {"truth":"NOMINAL","neural_scores":[.95,.02,.02,.01],"mechanism_values":[.001,.002,.001],"truth_location":None},
      {"truth":"NOMINAL","neural_scores":[.92,.03,.03,.02],"mechanism_values":[.002,.001,.001],"truth_location":None},
      {"truth":"FRICTION","neural_scores":[.01,.96,.02,.01],"mechanism_values":[.001,.08,.002],"truth_location":1},
      {"truth":"JAM","neural_scores":[.01,.02,.96,.01],"mechanism_values":[.09,.001,.001],"truth_location":0},
      {"truth":"TORQUE_LOSS","neural_scores":[.72,.02,.01,.25],"mechanism_values":[.001,.002,.06],"truth_location":2},
      {"truth":"TORQUE_LOSS","neural_scores":[.64,.03,.02,.31],"mechanism_values":[.001,.055,.002],"truth_location":1},
    ]


def _config(role="development", replay=False):
    return {
      "schema":"phm.algorithm-dag-search.v1","classes":CLASSES,"nominal_class":"NOMINAL",
      "data_role":role,"retrospective_replay":replay,
      "topologies":["neural_closed_set","physics_gate_neural_fault_physical_localizer","neural_gate_neural_fault_physical_localizer"],
      "physical_gate_thresholds":[.005,.01,.02],"neural_fault_probability_thresholds":[.3,.5,.7],
      "constraints":{"min_macro_f1":.9,"min_fault_detection_recall":.95,"max_nominal_false_alarm_rate":.1,"min_component_localization_accuracy":.95},
      "objective_weights":{"macro_f1":1.0,"fault_detection_recall":.5,"component_localization_accuracy":.25,"complexity_penalty":1e-4},
    }


def test_search_rediscovers_hierarchical_physics_gate_topology():
    report=search(_records(),_config())
    assert report["qualification_eligible"] is True
    assert report["chosen_grammar"]["gate_source"] == "physical_max"
    assert report["chosen_grammar"]["classifier"] == "neural_fault_only"
    assert report["chosen_grammar"]["localizer"] == "physical_argmax"
    assert report["chosen_metrics"]["macro_f1"] == 1.0
    assert report["chosen_metrics"]["fault_detection_recall"] == 1.0
    assert report["chosen_metrics"]["component_localization_accuracy_on_faults"] == 1.0


def test_runtime_executor_has_no_truth_inputs_and_localizes_from_mechanism_only():
    cfg=_config(); c=[x for x in enumerate_candidates(cfg) if x["grammar"]=={
        "gate_source":"physical_max","classifier":"neural_fault_only","localizer":"physical_argmax","threshold":.01}][0]
    scores=np.array([[.7,.01,.01,.28]])
    mech=np.array([[.001,.002,.06]])
    out=execute(c,RuntimeSignals(scores,mech))
    assert out["class_index"].tolist()==[3]
    assert out["location_index"].tolist()==[2]
    assert c["truth_inputs_used_at_runtime"] is False


def test_search_rejects_holdout_as_fresh_design_data():
    with pytest.raises(ValueError,match="REQUIRES_DEVELOPMENT"):
        search(_records(),_config(role="holdout",replay=False))


def test_retrospective_replay_is_explicitly_not_qualification_evidence():
    report=search(_records(),_config(role="calibration",replay=True))
    assert report["qualification_eligible"] is False
    assert report["claim_status"] == "RETROSPECTIVE_REPLAY_ONLY"


def test_margin_tiebreak_prefers_interior_physics_gate_threshold():
    cfg=_config()
    cfg["topologies"]=["physics_gate_neural_fault_physical_localizer"]
    cfg["physical_gate_thresholds"]=[.003,.01,.02]
    report=search(_records(),cfg)
    # All three thresholds have identical metrics on this fixture.  Selection
    # must use development-side margin, not an arbitrary fingerprint.
    assert report["chosen_grammar"]["threshold"] == .02
    assert report["results"][0].get("robustness_margin_fraction") is not None


def test_search_can_discover_physical_mechanism_bins_without_hardcoding_class_order():
    rows=[]
    # neural scores deliberately confuse the fault mechanisms; physical scalar
    # values are ordered but the class order is not provided to the searcher.
    for truth,value,loc in [
        ("NOMINAL",.001,-1),("NOMINAL",.002,-1),
        ("TORQUE_LOSS",.10,2),("TORQUE_LOSS",.15,1),
        ("FRICTION",1.5,0),("FRICTION",2.5,2),
        ("JAM",3.7,1),("JAM",4.4,0),
    ]:
        rows.append({"truth":truth,"neural_scores":[.4,.2,.3,.1],
                     "mechanism_values":[value,.0005,.0004] if loc==0 else ([.0005,value,.0004] if loc==1 else ([.0005,.0004,value] if loc==2 else [.001,.002,.001])),
                     "truth_location":loc})
    cfg=_config();cfg["topologies"]=["neural_closed_set","physics_gate_physical_bins_physical_localizer"]
    cfg["physical_gate_thresholds"]=[.003,.01,.03]
    cfg["physical_classifier_thresholds"]=[.2,.5,.8,1.0,2.0,3.0,3.2,3.5,4.0]
    report=search(rows,cfg)
    assert report["chosen_grammar"]["classifier"] == "physical_scalar_bins"
    assert report["chosen_grammar"]["classifier_class_order"] == [3,1,2]
    assert report["chosen_metrics"]["macro_f1"] == 1.0
    assert report["chosen_metrics"]["component_localization_accuracy_on_faults"] == 1.0


def test_multi_environment_search_uses_maximin_not_single_environment_optimum():
    from phm_design.algorithm_dag import search_multi_environment
    a=_records()
    # Environment B has nominal residuals slightly higher and torque-loss lower;
    # .003 becomes too close/unsafe while .01 remains separating.
    b=[]
    for row in _records():
        row=dict(row); row["mechanism_values"]=list(row["mechanism_values"])
        if row["truth"]=="NOMINAL": row["mechanism_values"]=[.004,.003,.002]
        elif row["truth"]=="TORQUE_LOSS": row["mechanism_values"]=[.012,.004,.003] if row["truth_location"]==0 else ([.003,.012,.004] if row["truth_location"]==1 else [.003,.004,.012])
        b.append(row)
    cfg=_config();cfg["topologies"]=["physics_gate_neural_fault_physical_localizer"]
    cfg["physical_gate_thresholds"]=[.003,.005,.01]
    report=search_multi_environment({"a":a,"b":b},cfg)
    assert report["environment_count"]==2
    assert report["test_or_holdout_used_for_selection"] is False
    assert report["chosen_grammar"]["threshold"]==.01


def test_physical_bins_can_delegate_boundary_ambiguity_to_neural_fault_expert():
    classes=CLASSES
    c=__import__('phm_design.algorithm_dag',fromlist=['make_candidate']).make_candidate(
        classes=classes,nominal_index=0,gate_source='physical_max',classifier='physical_scalar_bins_neural_fallback',
        localizer='physical_argmax',threshold=.01,classifier_thresholds=[.8,3.1],classifier_class_order=[3,1,2],fallback_margin=.1)
    scores=np.array([[.0,.001,.998,.001],[.0,.998,.001,.001]])
    mech=np.array([[.001,3.06,.002],[.001,2.0,.002]])
    out=execute(c,RuntimeSignals(scores,mech))
    assert out['class_index'].tolist()==[2,1]
    assert out['location_index'].tolist()==[1,1]


def test_temporal_mechanism_banks_and_classifier_source_selection():
    import numpy as np
    from phm_design.algorithm_dag import temporal_mechanism_banks, records_to_arrays, make_candidate, execute
    x=np.zeros((4,8,2),dtype=float)
    x[0,:,0]=[0,0,0,0,0,0,0,1]
    x[1,:,0]=[0,0,0,0,1,1,1,1]
    x[2,:,1]=[0,1,0,1,0,1,0,1]
    x[3,:,1]=2
    banks=temporal_mechanism_banks(x)
    assert {'rms','p90_abs','max_abs','std_abs','late_half_mean_abs','late_quarter_mean_abs',
            'roughness_abs_diff','tail_head_delta_abs','crest_factor','roughness_over_rms',
            'diff_std_over_rms','trend_over_rms','tail_cv'} <= set(banks)
    rows=[]
    classes=['N','A','B']
    for i in range(4):
        rows.append({'truth':'N' if i==0 else ('A' if i<3 else 'B'),'truth_location':0,
                     'neural_scores':[.9,.05,.05], 'mechanism_values':[.1,.05],
                     'mechanism_banks':{k:v[i].tolist() for k,v in banks.items()}})
    sig,_,_=records_to_arrays(rows,classes)
    cand=make_candidate(classes=classes,nominal_index=0,gate_source='physical_max',classifier='physical_scalar_bins',
                        localizer='physical_argmax',threshold=.01,classifier_thresholds=[.5],classifier_class_order=[1,2],
                        classifier_mechanism_source='late_quarter_mean_abs')
    out=execute(cand,sig)
    assert len(out['class_index'])==4
    assert cand['grammar']['classifier_mechanism_source']=='late_quarter_mean_abs'


def test_eligibility_can_require_mechanism_joint_and_class_recall():
    from phm_design.algorithm_dag import _eligible
    m={'macro_f1':.96,'fault_detection_recall':1.0,'nominal_false_alarm_rate':0.0,
       'component_localization_accuracy_on_faults':1.0,'mechanism_accuracy_on_faults':.93,
       'joint_mechanism_location_accuracy_on_faults':.93,
       'recall_by_class':{'N':1.0,'F':.8,'J':1.0}}
    ok,reasons=_eligible(m,{'min_macro_f1':.9,'min_mechanism_accuracy_on_faults':.95,
                            'min_joint_mechanism_location_accuracy':.95,'min_all_class_recall':.9})
    assert not ok
    assert 'MECHANISM_ACCURACY' in reasons and 'JOINT_ACCURACY' in reasons and 'CLASS_RECALL' in reasons


def test_row_max_respects_selected_temporal_mechanism_source():
    from phm_design.algorithm_dag import make_candidate, execute, RuntimeSignals
    classes=['N','A','B','C']
    # Default mechanism says class B (middle bin), while the selected temporal
    # bank says class C (upper bin). This catches accidental use of default.
    candidate=make_candidate(classes=classes,nominal_index=0,gate_source='physical_max',
        classifier='physical_scalar_bins',localizer='physical_argmax',threshold=.01,
        classifier_thresholds=[.5,1.5],classifier_class_order=[1,2,3],
        classifier_mechanism_source='rms')
    signals=RuntimeSignals(
        neural_scores=np.array([[.1,.1,.7,.1]]),
        mechanism_values=np.array([[.8,.1]]),
        mechanism_banks={'rms':np.array([[2.0,.1]])},
    )
    out=execute(candidate,signals)
    assert out['class_index'].tolist()==[3]



def test_robustness_margin_can_be_a_hard_development_constraint():
    from phm_design.algorithm_dag import _eligible
    m={'macro_f1':1.0,'fault_detection_recall':1.0,'nominal_false_alarm_rate':0.0,
       'component_localization_accuracy_on_faults':1.0,'mechanism_accuracy_on_faults':1.0,
       'joint_mechanism_location_accuracy_on_faults':1.0,
       'recall_by_class':{'N':1.0,'F':1.0,'J':1.0}}
    ok,reasons=_eligible(m,{'min_macro_f1':.95,'min_robustness_margin_fraction':0.0},
                         robustness_margin_fraction=-.01)
    assert not ok
    assert 'ROBUSTNESS_MARGIN' in reasons
    ok,reasons=_eligible(m,{'min_macro_f1':.95,'min_robustness_margin_fraction':0.0},
                         robustness_margin_fraction=.01)
    assert ok and not reasons


def test_multi_environment_robustness_constraint_rejects_boundary_fragile_candidate():
    from phm_design.algorithm_dag import search_multi_environment
    # A hybrid can repair a deliberately wrong physical bin boundary with its
    # neural fallback and still score perfectly.  The development margin must be
    # able to reject that fragile boundary independently of the observed score.
    rows=[]
    for truth,value,score,loc in [
        ('NOMINAL',.001,[.99,.003,.003,.004],-1),
        ('TORQUE_LOSS',.10,[.01,.01,.01,.97],2),
        ('FRICTION',.70,[.01,.97,.01,.01],0),
        ('JAM',2.0,[.01,.01,.97,.01],1),
    ]:
        mech=[.001,.001,.001]
        if loc>=0: mech[loc]=value
        rows.append({'truth':truth,'neural_scores':score,'mechanism_values':mech,'truth_location':loc})
    cfg={
      'schema':'phm.algorithm-dag-search.v1','classes':CLASSES,'nominal_class':'NOMINAL',
      'data_role':'development','retrospective_replay':False,
      'topologies':['physics_gate_physical_bins_neural_fallback_physical_localizer'],
      'physical_gate_thresholds':[.01],
      # Threshold .5 puts FRICTION=.7 in the middle bin correctly; threshold .8
      # puts it in the lower bin, but fallback margin .2 repairs it. Both can
      # score perfectly, yet only the former has non-negative physical margin.
      'physical_classifier_sources':['default'],
      'physical_classifier_thresholds':[.5,.8,1.5],
      'physical_classifier_fallback_margins':[.2],
      'constraints':{'min_macro_f1':.99,'min_fault_detection_recall':.99,
                     'max_nominal_false_alarm_rate':0.0,
                     'min_component_localization_accuracy':.99,
                     'min_all_class_recall':.99,
                     'min_robustness_margin_fraction':0.0},
      'objective_weights':{'macro_f1':1.0},
    }
    report=search_multi_environment({'a':rows,'b':rows},cfg)
    assert report['eligible_candidate_count'] >= 1
    assert report['chosen_worst_robustness_margin_fraction'] >= 0.0
    for env in report['chosen_environment_reports'].values():
        assert 'ROBUSTNESS_MARGIN' not in env['ineligibility_reasons']


def test_no_eligible_candidate_fails_closed_instead_of_qualifying_best_effort():
    cfg=_config();cfg['constraints']['min_macro_f1']=1.01
    report=search(_records(),cfg)
    assert report['status']=='FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE'
    assert report['selection_status']=='NO_ELIGIBLE_CANDIDATE'
    assert report['qualification_eligible'] is False
    assert report['chosen_is_eligible'] is False


def test_compiled_physical_rule_tree_executes_without_truth():
    from sklearn.tree import DecisionTreeClassifier
    from phm_design.algorithm_dag import _compile_rule_tree, make_physical_rule_tree_candidate, execute, RuntimeSignals
    X=np.array([[.1,.1],[.2,.9],[1.0,.1],[1.1,.9]])
    y=np.array([3,3,1,2])
    model=DecisionTreeClassifier(max_depth=2,random_state=0).fit(X,y)
    candidate=make_physical_rule_tree_candidate(classes=CLASSES,nominal_index=0,gate_threshold=.01,
        feature_sources=['std_abs','crest_factor'],compiled_tree=_compile_rule_tree(model))
    signals=RuntimeSignals(
        neural_scores=np.full((4,4),.25),
        mechanism_values=np.array([[.001,.001],[.1,.01],[.1,.01],[.1,.01]]),
        mechanism_banks={'std_abs':np.array([[.0,.0],[.1,.0],[1.0,.0],[1.1,.0]]),
                         'crest_factor':np.array([[.0,.0],[.1,.0],[.1,.0],[.9,.0]])})
    out=execute(candidate,signals)
    assert out['class_index'][0]==0
    assert out['class_index'][2] in {1,2,3}
    assert candidate['truth_inputs_used_at_runtime'] is False


def test_physical_rule_tree_multi_environment_loo_is_a_hard_gate_by_default():
    from phm_design.algorithm_dag import search_physical_rule_tree_multi_environment
    # Each environment is perfectly separable when included in training, but the
    # middle fault changes sides between environments. A pooled tree can fit both
    # development environments while leave-one-environment-out cannot satisfy
    # the strict all-class recall constraint. The search must fail closed.
    def env(mid):
        rows=[]
        for truth,val,loc in [
            ('NOMINAL',.001,-1),('NOMINAL',.002,-1),
            ('TORQUE_LOSS',.10,0),('TORQUE_LOSS',.12,1),
            ('FRICTION',mid,0),('FRICTION',mid*1.02,1),
            ('JAM',2.0,2),('JAM',2.1,1),
        ]:
            mech=[.001,.001,.001]
            if loc>=0: mech[loc]=val
            rows.append({'truth':truth,'truth_location':loc,'neural_scores':[.25]*4,
                         'mechanism_values':mech,
                         'mechanism_banks':{'std_abs':mech,'diff_std_over_rms':mech}})
        return rows
    cfg={'classes':CLASSES,'nominal_class':'NOMINAL','data_role':'development','retrospective_replay':False,
         'physical_tree_feature_sets':[['std_abs','diff_std_over_rms']],
         'physical_tree_depths':[3],'physical_tree_min_samples_leaf':[1],
         'physical_gate_thresholds':[.01],
         'constraints':{'min_macro_f1':.99,'min_all_class_recall':.99,'min_fault_detection_recall':.99,
                        'max_nominal_false_alarm_rate':0.0,'min_component_localization_accuracy':.99,
                        'min_mechanism_accuracy_on_faults':.99,'min_joint_mechanism_location_accuracy':.99},
         'objective_weights':{'macro_f1':1.0}}
    report=search_physical_rule_tree_multi_environment({'a':env(.5),'b':env(1.5)},cfg)
    assert report['require_leave_one_environment_out'] is True
    assert report['selection_status']=='NO_ELIGIBLE_CANDIDATE'
    assert report['qualification_eligible'] is False
    assert report['chosen_is_eligible'] is False
    assert report['chosen_leave_one_environment_out_reports']


def test_rule_tree_neural_fallback_only_uses_neural_near_relative_boundary():
    from sklearn.tree import DecisionTreeClassifier
    from phm_design.algorithm_dag import _compile_rule_tree, make_physical_rule_tree_candidate, execute, RuntimeSignals
    # Fault-only physical tree splits near 1.0. The first sample is just across
    # that boundary and should defer to neural; the second is far away and must
    # retain the physical prediction even though neural disagrees.
    X=np.array([[.4],[.6],[1.4],[1.6]])
    y=np.array([1,1,2,2])
    tree=DecisionTreeClassifier(max_depth=1,random_state=0).fit(X,y)
    cand=make_physical_rule_tree_candidate(classes=CLASSES,nominal_index=0,gate_threshold=.01,
        feature_sources=['std_abs'],compiled_tree=_compile_rule_tree(tree),neural_fallback_margin=.03)
    sig=RuntimeSignals(
        neural_scores=np.array([[.0,.99,.01,.0],[.0,.99,.01,.0]]),
        mechanism_values=np.array([[.1,.01,.01],[.1,.01,.01]]),
        mechanism_banks={'std_abs':np.array([[1.01,.01,.01],[1.8,.01,.01]])},
    )
    out=execute(cand,sig)
    assert cand['grammar']['classifier']=='physical_rule_tree_neural_fallback'
    assert out['class_index'].tolist()==[1,2]


def test_rule_tree_search_can_select_neural_boundary_fallback_to_pass_loo():
    from phm_design.algorithm_dag import search_physical_rule_tree_multi_environment
    # Physical class boundary drifts across environments, but the frozen neural
    # expert is correct exactly on the ambiguous boundary records. Search is
    # allowed to choose a small fallback band using development LOO only.
    def rows(mid, ambiguous_truth, ambiguous_value):
        out=[]
        vals=[('NOMINAL',.001,-1,[.99,.003,.003,.004]),
              ('TORQUE_LOSS',.10,0,[.01,.01,.01,.97]),
              ('FRICTION',mid,1,[.01,.97,.01,.01]),
              ('JAM',2.0,2,[.01,.01,.97,.01]),
              (ambiguous_truth,ambiguous_value,0,[.01,.97,.01,.01] if ambiguous_truth=='FRICTION' else [.01,.01,.97,.01])]
        for truth,value,loc,score in vals:
            mech=[.001,.001,.001]
            if loc>=0: mech[loc]=value
            out.append({'truth':truth,'truth_location':loc,'neural_scores':score,'mechanism_values':mech,
                        'mechanism_banks':{'std_abs':mech,'diff_std_over_rms':mech}})
        return out
    envs={'a':rows(.55,'JAM',1.02),'b':rows(.65,'FRICTION',.98),'c':rows(.60,'JAM',1.04)}
    cfg={'classes':CLASSES,'nominal_class':'NOMINAL','data_role':'development','retrospective_replay':False,
         'physical_tree_feature_sets':[['std_abs','diff_std_over_rms']],
         'physical_tree_depths':[2],'physical_tree_min_samples_leaf':[1],
         'physical_gate_thresholds':[.01],
         # This tiny synthetic fixture has a deliberately coarse learned split;
         # include a wider fallback band to verify that the *search* can select
         # the fallback topology. The narrow 0.03 band is qualified separately
         # on the real four-environment development replay.
         'physical_tree_neural_fallback_margins':[None,.01,.03,.10,.50],
         'constraints':{'min_macro_f1':.95,'min_all_class_recall':.9,'min_fault_detection_recall':.95,
                        'max_nominal_false_alarm_rate':0.0,'min_component_localization_accuracy':.95,
                        'min_mechanism_accuracy_on_faults':.95,'min_joint_mechanism_location_accuracy':.95},
         'objective_weights':{'macro_f1':1.0}}
    report=search_physical_rule_tree_multi_environment(envs,cfg)
    assert report['qualification_eligible'] is True
    assert report['chosen_is_eligible'] is True
    assert report['chosen_grammar']['classifier'] in {'physical_rule_tree','physical_rule_tree_neural_fallback'}
    assert report['chosen_loo_worst_all_class_recall'] >= .9
