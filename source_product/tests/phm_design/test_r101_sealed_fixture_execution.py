"""R101 pilot protocol tests: raw BYTE fixtures, not real diagnostics."""
from copy import deepcopy
import hashlib
import pytest

from phm_design.industrial_contracts.sealed_fixture_execution import (
    SealedExecutionRejected, seal_fixture_model, execute_sealed_fixture,
    SCHEMA)

def setup(tmp_path):
    model={"schema":SCHEMA,"algorithm":"linear_binary_logit",
           "transform":"raw_byte_mean_div_255","classes":["healthy","fault"],
           "coefficients":[8.0],"intercept":-4.0,
           "source_plan_sha256":"a"*64,"trained_on_source_only":True}
    paths={}
    digests={}
    for rid,b in (("train_a",bytes([10])*1024),
                  ("held_a",bytes([240])*1024)):
        p=tmp_path/(rid+".bin")
        p.write_bytes(b)
        paths[rid]=str(p)
        digests[rid]=hashlib.sha256(b).hexdigest()
    return model,paths,digests

def test_r101_run_original_raw_bytes_yields_bound_prediction_receipt(tmp_path):
    model,paths,digests=setup(tmp_path)
    seal=seal_fixture_model(model,digests)
    result=execute_sealed_fixture(seal,model,paths)
    assert result["predictions"]["train_a"]["predicted_label"]=="healthy"
    assert result["predictions"]["held_a"]["predicted_label"]=="fault"
    assert result["model_sha256"]==seal["model_sha256"]
    assert all(result["predictions"][i]["raw_sha256"]==digests[i] for i in paths)
    assert not result["industrial_feature_graph_executed"]
    assert not result["native_torch_like_provider_executed"]
    assert not result["independent_real_equipment_superiority_proven"]
    assert not result["eligible_for_stable"]
    assert len(result["receipt_sha256"])==64

@pytest.mark.parametrize("name",["mutate_model","alter_seal","alter_source","extra_file","missing_file"])
def test_r101_refuses_identity_or_bytes_tampering(tmp_path,name):
    model,paths,digests=setup(tmp_path)
    seal=seal_fixture_model(model,digests)
    if name=="mutate_model":
        model["intercept"]=2.
    elif name=="alter_seal":
        seal["predictor"]="arbitrary_code"
    elif name=="alter_source":
        from pathlib import Path
        Path(paths["held_a"]).write_bytes(bytes([1])*1024)
    elif name=="extra_file":
        paths["new"]="a.bin"
    else:
        del paths["held_a"]
    with pytest.raises(SealedExecutionRejected):
        execute_sealed_fixture(seal,model,paths)

@pytest.mark.parametrize("name",["unknown_provider","bad_transform","fake_source","object_code","bad_numeric"])
def test_r101_only_allowlisted_safe_numeric_fixture_models(tmp_path,name):
    model,paths,digests=setup(tmp_path)
    if name=="unknown_provider":model["algorithm"]="torch_like_pickle"
    elif name=="bad_transform":model["transform"]="eval_python"
    elif name=="fake_source":model["trained_on_source_only"]=False
    elif name=="object_code":model["payload"]="exec('malware')"
    else:model["coefficients"]=[float("nan")]
    with pytest.raises(SealedExecutionRejected):
        seal_fixture_model(model,digests)

def test_r101_reject_symlink_and_bad_record_input(tmp_path):
    model,paths,digests=setup(tmp_path)
    seal=seal_fixture_model(model,digests)
    from pathlib import Path
    link=tmp_path/"alias.bin"
    try:link.symlink_to(Path(paths["held_a"]))
    except (OSError,NotImplementedError):pytest.skip("symlink not available")
    paths["held_a"]=str(link)
    with pytest.raises(SealedExecutionRejected,match="R101_UNSAFE_OR_MISSING_RAW_FILE"):
        execute_sealed_fixture(seal,model,paths)
