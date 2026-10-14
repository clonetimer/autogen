import copy
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from phm_design.thermal_family_v2_model import (
    summarize, to_json_forest, predict_proba_from_features,
)
from phm_design.thermal_family_v2 import feature_contract


def _fixture():
    rng=np.random.RandomState(9)
    n=len(feature_contract()['channels'])
    x=rng.normal(0,1,size=(45,128,n)).astype(np.float32)
    X=summarize(x)
    y=np.repeat([0,1,2],15)
    model=RandomForestClassifier(n_estimators=11,max_depth=3,min_samples_leaf=1,
        class_weight='balanced',random_state=19,n_jobs=1).fit(X,y)
    package=to_json_forest(model,evidence={'role':'train-only'})
    return x,model,package


def test_portable_json_predictions_match_sklearn():
    x,model,package=_fixture()
    expected=model.predict_proba(summarize(x))
    actual=predict_proba_from_features(x,package)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))


def test_json_runtime_ignores_truth_and_contract_excludes_it():
    x,_,package=_fixture()
    assert 'truth' not in package
    assert not any('truth' in item for item in package['feature_names'])
    y=predict_proba_from_features(x,package)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y.sum(1),1)


def test_changed_feature_schema_fails_closed():
    x,_,package=_fixture()
    broken=copy.deepcopy(package)
    broken['feature_names'][0]='ground_truth'
    with pytest.raises(ValueError,match='CHANNELS_CHANGED'):
        predict_proba_from_features(x,broken)


def test_corrupt_child_rejected_and_bad_shape_rejected():
    x,_,package=_fixture()
    broken=copy.deepcopy(package)
    tree=broken['model']['trees'][0]
    branch=next(i for i,j in enumerate(tree['feature']) if j >= 0)
    tree['children_left'][branch]=999999
    with pytest.raises(ValueError,match='BAD_CHILD'):
        predict_proba_from_features(x,broken)
    with pytest.raises(ValueError,match='SUMMARY_SHAPE'):
        predict_proba_from_features(x[:,:,:-1],package)
