"""Use the existing compositional designer with a device binding, not wheel IDs.

This entry accepts development data only. It does not synthesize physical laws,
perform cross-device weight transfer, or silently enable RW-specific robustness.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .io import read_json, write_json, digest
from .data import Dataset, validate_contract
from .synthesis import design, validate_request


def make_request(binding:dict,policy:dict)->dict:
    if binding.get("schema")!="phm.device-binding.v1":raise ValueError("DEVICE_BINDING_SCHEMA")
    validate_contract(binding["contract"])
    if binding.get("quality_mode")!="complete_observation_only_missing_values_rejected":
        raise ValueError("UNSUPPORTED_FAMILY_QUALITY_CONTRACT")
    fields={"schema","grammar","evolution","controls","seeds","training","budget","selection","abstention"}
    if set(policy)!=fields or policy["schema"]!="phm.family-design-policy.v1":raise ValueError("FAMILY_POLICY_SCHEMA")
    request={**copy.deepcopy(policy),"schema":"phm.compositional-design.v1",
        "window":binding["window_samples"],"groups":copy.deepcopy(binding["groups"]),
        "structure_reference":"device-binding sha256:"+digest(binding)+"; component grouping, not an approved causal graph"}
    # Semantics of heterogeneous components cannot be inferred from equal units.
    # Disable automatic sharing across groups in this first multi-family entry.
    if request["grammar"]["allow_sharing"]:raise ValueError("HETEROGENEOUS_SHARING_NOT_REVIEWED")
    validate_request(request,binding["contract"])
    return request


def design_family(dataset_path:str|Path,binding_path:str|Path,policy_path:str|Path,output:str|Path,
                  *,resume:bool=False,on_trial_complete=None)->dict:
    dataset=Dataset(dataset_path,allowed_splits={"train","validation"})
    bound=read_json(binding_path)
    if dataset.contract!=bound.get("contract"):raise ValueError("FAMILY_INPUT_BINDING_CHANGED")
    request=make_request(bound,read_json(policy_path));output=Path(output)
    request_path=output.parent/(output.name+".family-request.json")
    if request_path.exists():
        if read_json(request_path)!=request:raise ValueError("FROZEN_FAMILY_REQUEST_CHANGED")
    else:write_json(request_path,request)
    return design(dataset_path,request_path,output,resume=resume,on_trial_complete=on_trial_complete)
