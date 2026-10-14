from research.explain_r65_thermal_timescales import explain

def test_timescale_formula_and_development_role():
    p={'w':{
      'thermal_a_mass_kg':0.5,'thermal_a_emissivity':0.8,
      'thermal_a_area_m2':0.02,'thermal_a_initial_k':300.,
      'thermal_b_mass_kg':0.5,'thermal_b_emissivity':0.8,
      'thermal_b_area_m2':0.02,'thermal_b_initial_k':300.,
    }}
    r=explain(p)
    assert r['tau_min_s']==r['tau_max_s']
    assert r['tau_median_s']>220
    assert r['diagnostic_qualification']=='NOT_EVALUATED'
    assert r['new_blind_worlds']==0
