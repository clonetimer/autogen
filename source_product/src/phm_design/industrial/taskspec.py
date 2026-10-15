"""R80 industrial TaskSpec compiler: a device-agnostic, pre-fit design contract.

All outputs are declarative PLAN_ONLY artifacts. This module NEVER reads FINAL
labels, trains a model, makes a diagnosis, or certifies physical observability.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

SCHEMA = "phm.industrial.task.v0"
KINDS = frozenset({"fault_detection", "fault_classification",
                   "component_localization", "mechanism_classification", "joint_diagnosis"})
MODES = frozenset({"offline_trajectory", "causal_window"})
BACKBONES = frozenset({"linear", "tree", "cnn1d", "tcn", "gru", "transformer"})
OBJECTIVES = frozenset({"ERM", "GroupDRO", "V-REx"})
NEURAL = frozenset({"cnn1d", "tcn", "gru", "transformer"})
INFERENCE_ROLES = frozenset({"sensor", "command"})
PROTECTED_ROLES = frozenset({"FINAL_ID", "FINAL_OOD"})
_PROTECTED_NAMES = frozenset({
    "label", "labels", "target", "target_label", "ground_truth", "truth",
    "fault_label", "fault_class", "fault_injection", "fault_onset",
    "fault_onset_s", "failure_time", "rul", "world_id", "group_id",
    "generation_family_id", "base_physics_id", "bearing_instance_id",
    "environment", "environment_id", "environment_family", "split",
    "split_name", "role", "data_role", "fault_stratum", "fault_severity",
    "file_path", "source_path", "test_result", "final_metric",
})
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")


class TaskSpecRejected(ValueError):
    """Structured, fail-closed pre-fit validation failure."""
    def __init__(self, code: str, detail: str = ""):
        self.code, self.detail = code, detail
        super().__init__(code + (":" + detail if detail else ""))


def _reject(code: str, detail: str = ""):
    raise TaskSpecRejected(code, detail)


def _mapping(value, context, required=(), allowed=()):
    if not isinstance(value, Mapping):
        _reject("INVALID_TASKSPEC_MAPPING", context)
    if any(not isinstance(key, str) for key in value):
        _reject("INVALID_TASKSPEC_KEY", context)
    absent = set(required) - set(value)
    unknown = set(value) - set(allowed)
    if absent:
        _reject("MISSING_TASKSPEC_FIELDS", context + ":" + ",".join(sorted(absent)))
    if unknown:
        _reject("UNKNOWN_TASKSPEC_FIELDS", context + ":" + ",".join(sorted(unknown)))
    return value


def _identity(value, context):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _reject("INVALID_IDENTITY", context)
    return value


def _enum(value, allowed, context):
    if not isinstance(value, str) or value not in allowed:
        _reject("UNSUPPORTED_TASKSPEC_VALUE", context)
    return value


def _bool(value, context):
    if type(value) is not bool:
        _reject("INVALID_BOOLEAN", context)
    return value


def _positive_int(value, context, *, minimum=1, maximum=100000):
    if type(value) is not int or not minimum <= value <= maximum:
        _reject("INVALID_INTEGER", context)
    return value


def _nonempty_strings(value, context, *, require_ids=True):
    if (not isinstance(value, (list, tuple)) or not value or
            any(not isinstance(item, str) or not item.strip() for item in value) or
            len(value) != len(set(value))):
        _reject("INVALID_OR_DUPLICATE_ITEMS", context)
    if require_ids:
        for item in value:
            _identity(item, context)
    return list(value)


def _json_digest(value) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TaskSpecRejected("NON_SERIALIZABLE_TASKSPEC") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _channel_name(name, context):
    name = _identity(name, context)
    lower = name.lower()
    if (lower in _PROTECTED_NAMES or lower.startswith(("truth_", "target_", "label_",
         "final_", "injected_", "injection_", "split_", "role_"))):
        _reject("PROTECTED_FIELD_CANNOT_BE_INFERENCE_INPUT", name)
    return name


@dataclass(frozen=True)
class TaskSpec:
    """Canonical, hash-identifiable task specification; no mutable source dict."""
    canonical_json: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping) -> "TaskSpec":
        _mapping(raw, "root", required=(
            "schema", "id", "equipment", "task", "observations", "data",
            "design", "deployment"), allowed=(
            "schema", "id", "equipment", "task", "observations", "data",
            "design", "deployment"))
        if raw["schema"] != SCHEMA:
            _reject("UNKNOWN_TASKSPEC_SCHEMA")
        _identity(raw["id"], "id")
        # Coercion by JSON guarantees no Python classes, executable callables,
        # NaNs, or object identity can silently enter the plan.
        digest = _json_digest(raw)
        return cls(json.dumps(raw, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False), digest)

    def to_dict(self) -> dict:
        return json.loads(self.canonical_json)


def compile_task_spec(source: Mapping | TaskSpec, *, train_manifest: Sequence[Mapping] | None = None) -> dict:
    """Compile an industrial diagnosis plan *without fitting anything*.

    A missing TRAIN manifest produces a plan needing a subsequent data audit.
    A supplied manifest is TRAIN-only; FINAL metadata, labels, or statistics
    never enter a design/selection worker.
    """
    task_spec = source if isinstance(source, TaskSpec) else TaskSpec.from_dict(source)
    d = task_spec.to_dict()
    equipment = _mapping(d["equipment"], "equipment",
                         required=("family", "adapter"),
                         allowed=("family", "adapter", "version"))
    _identity(equipment["family"], "equipment.family")
    _identity(equipment["adapter"], "equipment.adapter")
    if "version" in equipment and (
        not isinstance(equipment["version"], str) or not equipment["version"].strip()):
        _reject("INVALID_ADAPTER_VERSION")

    task = _mapping(d["task"], "task",
                    required=("kind", "decision_mode", "labels", "healthy_label",
                              "allow_unknown", "multifault", "structure"),
                    allowed=("kind", "decision_mode", "labels", "healthy_label",
                             "allow_unknown", "multifault", "structure"))
    kind = _enum(task["kind"], KINDS, "task.kind")
    mode = _enum(task["decision_mode"], MODES, "task.decision_mode")
    labels = _nonempty_strings(task["labels"], "task.labels")
    if len(labels) < 2 or len(labels) > 256 or task["healthy_label"] not in labels:
        _reject("INVALID_TASK_LABEL_ONTOLOGY")
    _bool(task["allow_unknown"], "task.allow_unknown")
    if _bool(task["multifault"], "task.multifault"):
        _reject("UNSUPPORTED_MULTIFAULT_TASK")
    structure = _mapping(task["structure"], "task.structure",
                         required=("components", "mechanisms", "class_attributes"),
                         allowed=("components", "mechanisms", "class_attributes"))
    components = structure["components"]
    mechanisms = structure["mechanisms"]
    if not isinstance(components, list) or not isinstance(mechanisms, list):
        _reject("INVALID_TASK_ONTOLOGY_STRUCTURE")
    if components:
        _nonempty_strings(components, "structure.components")
    if mechanisms:
        _nonempty_strings(mechanisms, "structure.mechanisms")
    attrs = structure["class_attributes"]
    if not isinstance(attrs, Mapping) or any(not isinstance(k, str) for k in attrs):
        _reject("INVALID_CLASS_ATTRIBUTES")
    if set(attrs) != set(labels) - {task["healthy_label"]}:
        _reject("INCOMPLETE_OR_EXTRANEOUS_CLASS_ATTRIBUTES")
    needs_component = kind in ("component_localization", "joint_diagnosis")
    needs_mechanism = kind in ("mechanism_classification", "joint_diagnosis")
    if needs_component and not components:
        _reject("COMPONENT_ONTOLOGY_REQUIRED")
    if needs_mechanism and not mechanisms:
        _reject("MECHANISM_ONTOLOGY_REQUIRED")
    for label, desc in attrs.items():
        desc = _mapping(desc, "structure.class_attributes." + label,
                        allowed=("component", "mechanism"))
        component, mechanism = desc.get("component"), desc.get("mechanism")
        if component is not None and component not in components:
            _reject("ILLEGAL_CLASS_COMPONENT", label)
        if mechanism is not None and mechanism not in mechanisms:
            _reject("ILLEGAL_CLASS_MECHANISM", label)
        if needs_component and not component:
            _reject("MISSING_CLASS_COMPONENT", label)
        if needs_mechanism and not mechanism:
            _reject("MISSING_CLASS_MECHANISM", label)

    obs = _mapping(d["observations"], "observations",
                   required=("time", "healthy_prefix_policy", "channels"),
                   allowed=("time", "healthy_prefix_policy", "channels"))
    timing = _mapping(obs["time"], "observations.time",
                      required=("field", "sampling"),
                      allowed=("field", "sampling", "window_samples", "horizon_samples"))
    time_field = _channel_name(timing["field"], "time.field")
    _enum(timing["sampling"], {"fixed", "irregular"}, "time.sampling")
    health = _enum(obs["healthy_prefix_policy"],
                   {"not_assumed", "verified_if_required", "required"},
                   "healthy_prefix_policy")
    channels = obs["channels"]
    if not isinstance(channels, list) or not channels:
        _reject("NO_DEPLOYABLE_CHANNELS")
    names = []
    for i, channel in enumerate(channels):
        c = _mapping(channel, "observations.channels." + str(i),
                     required=("name", "role", "unit", "available_at_decision"),
                     allowed=("name", "role", "unit", "available_at_decision"))
        name = _channel_name(c["name"], "channel.name")
        _enum(c["role"], INFERENCE_ROLES, "channel.role")
        if not isinstance(c["unit"], str) or not c["unit"].strip():
            _reject("MISSING_SENSOR_UNIT", name)
        if not _bool(c["available_at_decision"], "available_at_decision"):
            _reject("POST_DECISION_CHANNEL", name)
        names.append(name)
    if len(names) != len(set(names)) or time_field in names:
        _reject("DUPLICATE_OR_TIME_CHANNEL")
    if mode == "causal_window":
        _positive_int(timing.get("window_samples"), "time.window_samples", minimum=2)
        _positive_int(timing.get("horizon_samples", 0), "time.horizon_samples",
                      minimum=0)
        if timing.get("horizon_samples", 0) > 0:
            _reject("FUTURE_SAMPLES_IN_CAUSAL_INPUT")
    elif "window_samples" in timing or "horizon_samples" in timing:
        _reject("OFFLINE_TIMING_CONTRACT_MIXED_WITH_CAUSAL")

    data = _mapping(d["data"], "data",
                    required=("role", "group_keys", "environment_axes"),
                    allowed=("role", "group_keys", "environment_axes"))
    if data["role"] != "TRAIN":
        _reject("DESIGN_WORKER_PROTECTED_ROLE")
    groups = _nonempty_strings(data["group_keys"], "data.group_keys")
    environments = _nonempty_strings(data["environment_axes"],
                                     "data.environment_axes")
    if set(groups).intersection(environments):
        _reject("GROUP_ENVIRONMENT_KEYS_OVERLAP")
    if ({time_field, *names} & (set(groups) | set(environments))):
        _reject("PROVENANCE_METADATA_CANNOT_BE_DEPLOYMENT_FIELDS")

    design = _mapping(d["design"], "design",
                      required=("candidate_max", "compare", "objectives",
                                "allow_refusal", "budget"),
                      allowed=("candidate_max", "compare", "objectives",
                               "allow_refusal", "budget", "representations", "topologies",
                               "calibration", "feature_programs",
                               "adaptive_feature_search"))
    max_trials = _positive_int(design["candidate_max"], "design.candidate_max",
                               maximum=128)
    backbones = _nonempty_strings(design["compare"], "design.compare")
    objectives = _nonempty_strings(design["objectives"], "design.objectives")
    if not set(backbones).issubset(BACKBONES) or not set(objectives).issubset(OBJECTIVES):
        _reject("UNREGISTERED_ARCHITECTURE_OR_TRAINING_OBJECTIVE")
    # Explicit bounded programs share open_design's registered feature atoms.
    # The predeclared set is fixed before any source OOF feedback is seen.
    from .feature_program import ALLOWED_ATOMS, encode_program
    adaptive = None
    if "adaptive_feature_search" in design:
        from itertools import combinations
        config = _mapping(
            design["adaptive_feature_search"], "design.adaptive_feature_search",
            required=("atoms", "max_programs"),
            allowed=("atoms", "max_programs"))
        atoms = _nonempty_strings(config["atoms"], "adaptive_feature_search.atoms")
        if not set(atoms).issubset(ALLOWED_ATOMS) or len(atoms) < 3:
            _reject("INVALID_ADAPTIVE_FEATURE_ATOMS")
        max_programs = _positive_int(
            config["max_programs"], "adaptive_feature_search.max_programs",
            minimum=3, maximum=128)
        if ("feature_programs" in design or "representations" in design or
                len(backbones) != 1 or backbones[0] not in ("linear", "tree") or
                objectives != ["ERM"] or
                design.get("topologies", ["direct"]) not in (["direct"], ["factorized"])):
            _reject("ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE")
        pool_atoms = sorted(atoms)
        pool = [list(combo) for count in range(1, min(3, len(atoms)) + 1)
                for combo in combinations(pool_atoms, count)]
        if max_programs > len(pool) or max_trials != max_programs:
            _reject("ADAPTIVE_SEARCH_TRIAL_BUDGET_MISMATCH")
        adaptive = {
            "mode": "bounded_feedback",
            "atoms": pool_atoms,
            "candidate_pool_size": len(pool),
            "max_programs": max_programs,
            "seed_programs": [encode_program([atom]) for atom in atoms[:2]],
            "loss": "SOURCE_TRAIN_OOF_MEAN_TAIL_CLASS_ERROR",
        }
    programs = []
    if "feature_programs" in design:
        raw_programs = design["feature_programs"]
        if not isinstance(raw_programs, list) or not raw_programs:
            _reject("EMPTY_INDUSTRIAL_FEATURE_PROGRAMS")
        for index, entry in enumerate(raw_programs):
            cfg = _mapping(entry, "design.feature_programs." + str(index),
                           required=("atoms", "composition"),
                           allowed=("atoms", "composition", "bindings"))
            composition = _enum(cfg["composition"],
                                {"concat", "available_stack", "available_average"},
                                "feature_programs.composition")
            atoms = _nonempty_strings(cfg["atoms"], "design.feature_programs.atoms")
            if len(atoms) > 3 or not set(atoms).issubset(ALLOWED_ATOMS):
                _reject("UNREGISTERED_FEATURE_PROGRAM")
            ordered = sorted(atoms)
            bindings = []
            if composition != "concat":
                if not task["allow_unknown"]:
                    _reject("AVAILABILITY_REQUIRES_UNKNOWN_TASK_CONTRACT")
                if len(atoms) < 2 or "calibration" in design:
                    _reject("UNSUPPORTED_AVAILABILITY_CALIBRATION_OR_BRANCH_COUNT")
                raw_bindings = _mapping(cfg.get("bindings"), "feature_programs.bindings",
                                        required=ordered, allowed=ordered)
                used = set()
                for atom in ordered:
                    binding = _mapping(raw_bindings[atom],
                                       "feature_programs.bindings." + atom,
                                       required=("signal", "validity_flag"),
                                       allowed=("signal", "validity_flag"))
                    signal, valid = binding["signal"], binding["validity_flag"]
                    if (signal not in names or valid not in names or
                            signal == valid or signal in used or valid in used):
                        _reject("INVALID_AVAILABILITY_CHANNEL_BINDING")
                    flag = channels[names.index(valid)]
                    if (flag["role"] != "sensor" or flag["unit"] != "1"):
                        _reject("INVALID_AVAILABILITY_FLAG_CHANNEL_CONTRACT")
                    used.update((signal, valid))
                    bindings.append({"atom": atom, "signal": signal,
                                     "validity_flag": valid,
                                     "sensor_index": 1 + names.index(signal),
                                     "validity_index": 1 + names.index(valid)})
            elif "bindings" in cfg:
                _reject("CONCAT_MUST_NOT_DECLARE_AVAILABILITY_BINDINGS")
            item = {"atoms": ordered, "composition": composition,
                    "identity": encode_program(atoms)}
            if composition != "concat":
                item["bindings"] = bindings
            programs.append(item)
        identifiers = [(x["identity"], x["composition"]) for x in programs]
        if len(set(identifiers)) != len(identifiers):
            _reject("DUPLICATE_INDUSTRIAL_FEATURE_PROGRAM")
        if any(x["composition"] != "concat" for x in programs) and (
                len(backbones) != 1 or backbones[0] != "linear" or
                objectives != ["ERM"] or
                design.get("topologies", ["direct"]) != ["direct"]):
            _reject("AVAILABILITY_REQUIRES_LINEAR_DIRECT_ERM")
    if "representations" in design:
        representations = _nonempty_strings(
            design["representations"], "design.representations")
    else:
        representations = ([] if programs or adaptive else ["temporal_bins"])
    if not set(representations).issubset(
            {"temporal_bins", "random_kernels", "difference_kernels"}):
        _reject("UNREGISTERED_FEATURE_PROGRAM")
    topologies = _nonempty_strings(
        design.get("topologies", ["direct"]), "design.topologies")
    if not set(topologies).issubset({"direct", "factorized"}):
        _reject("UNREGISTERED_DIAGNOSIS_TOPOLOGY")
    # Opt-in, predeclared before any fitted candidate; old TasksSpecs are unchanged.
    calibration_policy = None
    if "calibration" in design:
        config = _mapping(design["calibration"], "design.calibration",
                          required=("method", "max_selective_error",
                                    "minimum_coverage", "min_calibration_samples"),
                          allowed=("method", "max_selective_error",
                                   "minimum_coverage", "min_calibration_samples"))
        _enum(config["method"], {"temperature_selective"}, "calibration.method")
        if not task["allow_unknown"]:
            _reject("CALIBRATION_REQUIRES_UNKNOWN_TASK_CONTRACT")
        for name, minimum, maximum in (
                ("max_selective_error", 0.01, 0.5),
                ("minimum_coverage", 0.05, 1.0)):
            value = config[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not minimum <= value <= maximum):
                _reject("INVALID_CALIBRATION_POLICY", name)
        _positive_int(config["min_calibration_samples"],
                      "calibration.min_calibration_samples", minimum=12)
        calibration_policy = dict(config)
    if "factorized" in topologies and kind != "joint_diagnosis":
        _reject("FACTORIZED_REQUIRES_JOINT_DIAGNOSIS")
    if "factorized" in topologies:
        pairs = [(attrs[label]["component"], attrs[label]["mechanism"])
                 for label in labels if label != task["healthy_label"]]
        if len(pairs) != len(set(pairs)):
            _reject("FACTORIZED_DUPLICATE_COMPONENT_MECHANISM_PAIR")
    if not _bool(design["allow_refusal"], "design.allow_refusal"):
        _reject("DESIGN_REFUSAL_MUST_BE_AVAILABLE")
    budget = _mapping(design["budget"], "design.budget",
                      required=("max_fits", "max_cpu_seconds", "seed"),
                      allowed=("max_fits", "max_cpu_seconds", "seed",
                               "max_wall_seconds"))
    max_fits = _positive_int(budget["max_fits"], "budget.max_fits", maximum=10000)
    seconds = _positive_int(budget["max_cpu_seconds"],
                            "budget.max_cpu_seconds", maximum=604800)
    wall_seconds = (_positive_int(budget["max_wall_seconds"],
                                  "budget.max_wall_seconds", maximum=604800)
                    if "max_wall_seconds" in budget else None)
    seed = _positive_int(budget["seed"], "budget.seed", minimum=0,
                         maximum=2**32 - 1)
    grid = []
    for backbone in backbones:
        for objective in objectives:
            if objective != "ERM" and backbone not in NEURAL:
                continue  # Tree and linear models cannot use backprop DG.
            choices = ([{"representation": "learned_sequence"}]
                       if backbone in NEURAL else
                       [{"representation": item} for item in representations] +
                       [{"representation": "feature_program",
                         "feature_program": entry["identity"],
                         **({"composition": entry["composition"],
                             "availability_bindings": entry["bindings"]}
                            if entry["composition"] != "concat" else {})}
                        for entry in programs])
            for choice in choices:
                for topology in topologies:
                    grid.append({**choice, "backbone": backbone,
                                 "objective": objective, "topology": topology})
    if adaptive is not None:
        # An immutable candidate pool, not an exhaustively evaluated portfolio.
        # Only max_programs proposals may be trained in one study.
        grid = [
            {"representation": "feature_program",
             "feature_program": encode_program(atoms),
             "backbone": backbones[0], "objective": "ERM",
             "topology": topologies[0]}
            for atoms in pool
        ]
        programs = [
            {"atoms": atoms, "composition": "concat",
             "identity": encode_program(atoms)} for atoms in pool
        ]
    if not grid:
        _reject("AUTO_NO_VALID_PLAN")
    if max_fits < max_trials:
        _reject("SEARCH_BUDGET_FITS_BELOW_TRIALS")
    if len(grid) > max_trials and adaptive is None:
        _reject("SEARCH_SPACE_EXCEEDS_PREDECLARED_TRIAL_BUDGET")

    deployment = _mapping(d["deployment"], "deployment",
                          required=("causal_online", "prototype_only"),
                          allowed=("causal_online", "prototype_only"))
    if _bool(deployment["causal_online"], "deployment.causal_online") != (
            mode == "causal_window"):
        _reject("DECISION_MODE_DEPLOYMENT_MISMATCH")
    if not _bool(deployment["prototype_only"], "deployment.prototype_only"):
        _reject("INDUSTRIAL_FIELD_QUALIFICATION_UNSUPPORTED")

    nodes = {
        "fault_detection": ["detection"],
        "fault_classification": ["detection", "fault_classification"],
        "component_localization": ["detection", "localization"],
        "mechanism_classification": ["detection", "mechanism"],
        "joint_diagnosis": ["detection", "localization", "mechanism", "decision_gate"],
    }[kind]
    manifest = (audit_train_manifest(d, train_manifest)
                if train_manifest is not None else None)
    if manifest is not None and not manifest["eligible"]:
        _reject("UNSUPPORTED_TRAINING_CLASS_OR_GROUP_SUPPORT")
    if manifest is not None and any(
            obj != "ERM" for obj in objectives) and not manifest["dg_supported"]:
        _reject("UNSUPPORTED_ENVIRONMENT_CLASS_SUPPORT_FOR_DG")
    return {
        "schema": "phm.industrial.autodesign-plan.v0",
        "plan_id": "plan-" + task_spec.sha256[:16],
        "task_id": d["id"],
        "task_spec_sha256": task_spec.sha256,
        "equipment_adapter": equipment["adapter"],
        "task_kind": kind,
        "decision_mode": mode,
        "eligible_task_nodes": nodes,
        "ontology": {"labels": labels, "healthy_label": task["healthy_label"],
                     "class_attributes": attrs,
                     "unknown_label": "__UNKNOWN__" if task["allow_unknown"] else None},
        "feature_contract": {"time_field": time_field, "ordered_channel_names": names,
                             "healthy_prefix_policy": health,
                             "forbid_truth_or_environment_features": True,
                             "causal_past_only": mode == "causal_window"},
        "train_data_contract": {"role_allowlist": ["TRAIN"], "group_keys": groups,
                                "environment_axes": environments,
                                "partition_before_windowing": True},
        "fold_strategy": "leave_physical_family_out",
        "source_only_nested_selection": True,
        "candidate_grid": grid,
        "registered_feature_programs": programs,
        "adaptive_feature_search": adaptive,
        "calibration_policy": calibration_policy,
        "budget": {"max_trials": max_trials, "max_fits": max_fits,
                   "max_cpu_seconds": seconds, "seed": seed,
                   **({"max_wall_seconds": wall_seconds}
                      if wall_seconds is not None else {})},
        "fold_preprocessing": "fit_on_training_partition_only",
        "predeclared_refusal_modes": [
            "AUTO_NO_VALID_PLAN", "NOT_OBSERVABLE_WITH_DECLARED_SENSORS",
            "UNSUPPORTED_TRAINING_CLASS_OR_GROUP_SUPPORT",
            "UNSUPPORTED_ENVIRONMENT_CLASS_SUPPORT_FOR_DG",
            "NO_ACCEPTABLE_MODEL", "UNKNOWN"],
        "manifest_audit": manifest,
        "status": ("TRAIN_PREFIT_READY_NOT_TRAINED" if manifest is not None
                   else "PLAN_ONLY_SOURCE_DATA_NOT_AUDITED"),
        "qualified": False, "independent_qualification": False,
        "target_data_used": False,
        "prohibited_target_actions": ["fit", "normalize", "select", "calibrate",
                                      "threshold_tune", "architecture_feedback"],
    }


def audit_train_manifest(task_spec: Mapping, records: Sequence[Mapping]) -> dict:
    """TRAIN metadata audit; not a steward for protected FINAL worlds.

    Labels and operating regime metadata remain outside inference channels.
    Physical group identity, not a window counter, is the unit of separation.
    """
    if (not isinstance(records, (list, tuple)) or not records):
        _reject("EMPTY_TRAIN_MANIFEST")
    labels = task_spec["task"]["labels"]
    group_keys = task_spec["data"]["group_keys"]
    env_keys = task_spec["data"]["environment_axes"]
    group_env = {}
    counts = {label: set() for label in labels}
    environment_counts = {}
    for i, row in enumerate(records):
        required = {"role", "label", *group_keys, *env_keys}
        r = _mapping(row, "manifest." + str(i), required=required, allowed=required)
        if r["role"] != "TRAIN":
            _reject("DESIGN_WORKER_PROTECTED_ROLE", str(i))
        if r["label"] not in labels:
            _reject("UNKNOWN_TRAINING_LABEL", str(i))
        ids = tuple(_identity(r[key], "group." + key) for key in group_keys)
        env = tuple(_identity(r[key], "environment." + key) for key in env_keys)
        if ids in group_env and group_env[ids] != env:
            _reject("PHYSICAL_GROUP_CROSSES_ENVIRONMENTS", str(ids))
        group_env[ids] = env
        counts[r["label"]].add(ids)
        environment_counts.setdefault(env, {label: set() for label in labels})[r["label"]].add(ids)
    eligible = len(group_env) >= 3 and all(len(groups) >= 2 for groups in counts.values())
    dg_supported = (len(environment_counts) >= 3 and
                    all(all(len(groups) > 0 for groups in v.values())
                        for v in environment_counts.values()))
    return {
        "eligible": eligible, "dg_supported": dg_supported,
        "physical_groups": len(group_env),
        "environment_families": len(environment_counts),
        "per_class_physical_group_support": {k: len(v) for k, v in counts.items()},
        "per_environment_support": {
            "|".join(key): {label: len(groups) for label, groups in support.items()}
            for key, support in sorted(environment_counts.items())},
        "provenance_self_reported_not_independently_verified": True,
        "contains_final_data": False,
    }


def validate_observations(spec: Mapping | TaskSpec, rows: Sequence[Mapping],
                          *, decision_timestamp: float | None = None,
                          healthy_prefix_verified: bool = False) -> dict:
    """Validate deployment inputs without reading labels/simulator truth."""
    # Observation checking must not be weaker than plan compilation.
    compiled = compile_task_spec(spec)
    d = spec.to_dict() if isinstance(spec, TaskSpec) else TaskSpec.from_dict(spec).to_dict()
    obs = d["observations"]
    time = obs["time"]
    time_field = time["field"]
    allowed = {time_field, *(x["name"] for x in obs["channels"])}
    if not isinstance(rows, (list, tuple)) or not rows:
        _reject("EMPTY_OBSERVATION_BATCH")
    previous = None
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _reject("INVALID_OBSERVATION_ROW", str(i))
        if any(name not in allowed for name in row):
            _reject("UNDECLARED_OR_PROTECTED_OBSERVATION", str(i))
        if any(name not in row for name in allowed):
            _reject("MISSING_DEPLOYMENT_CHANNEL", str(i))
        for name in allowed:
            v = row[name]
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                _reject("NONFINITE_OR_INVALID_DEPLOYMENT_VALUE", name)
        timestamp = row[time_field]
        if previous is not None and timestamp <= previous:
            _reject("NONMONOTONE_TIMESTAMPS")
        previous = timestamp
        if d["task"]["decision_mode"] == "causal_window" and (
                decision_timestamp is None or timestamp > decision_timestamp):
            _reject("FUTURE_OR_UNBOUNDED_CAUSAL_OBSERVATIONS")
    if d["task"]["decision_mode"] == "causal_window" and (
            len(rows) > time["window_samples"] or
            isinstance(decision_timestamp, bool) or not isinstance(decision_timestamp, (int, float))
            or not math.isfinite(decision_timestamp)):
        _reject("INVALID_CAUSAL_DECISION_WINDOW")
    if obs["healthy_prefix_policy"] == "required" and healthy_prefix_verified is not True:
        _reject("HEALTHY_PREFIX_NOT_VERIFIED")
    return {"rows": len(rows), "field_order": [time_field, *[c["name"] for c in obs["channels"]]],
            "causal": d["task"]["decision_mode"] == "causal_window",
            "ground_truth_consumed": False, "qualified": False}
