"""CPU search; separate fit and holdout commands. See docs for evidence limits.

Resume commits completed trials. Incomplete trials restart from their seed.
Workflow separation is NOT an operating-system data access security boundary.
"""
from __future__ import annotations
import contextlib
import os
import platform
import time
from pathlib import Path
from typing import Callable
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn
from . import __version__
from .data import (Dataset, fit_scaler, transform, statistics, reject_overlap,
                   reject_numeric_overlap, sequence_from_csv)
from .io import (digest, read_json, write_json, sha_file, seal, verify, run_lock, safe_child)
from .models import Shape, Registry, integer
from .registry import default_registry


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": str(torch.__version__),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "device": "cpu",
            "threads": 1, "phm_design": __version__, "cuda_used": False}


MODEL_EXECUTION_FILES = (
    "data.py", "engine.py", "graph.py", "io.py", "models.py", "reference.py", "registry.py"
)

def implementation_hash(files: tuple[str, ...] | None = None) -> str:
    """Hash a declared implementation scope.

    The default remains the complete package hash for audit/reporting. Training
    fingerprints use a narrower explicit scope so unrelated evidence/CLI changes
    do not force model retraining.
    """
    root = Path(__file__).parent
    if files is None:
        paths = sorted(root.rglob("*.py"))
    else:
        if not files or len(set(files)) != len(files):
            raise ValueError("IMPLEMENTATION_HASH_SCOPE")
        paths = [root / name for name in files]
        if any(not p.is_file() for p in paths):
            raise ValueError("IMPLEMENTATION_HASH_SCOPE_MISSING_FILE")
    return digest({str(p.relative_to(root)): sha_file(p) for p in paths})

def training_implementation_hash() -> str:
    return implementation_hash(MODEL_EXECUTION_FILES)


def config_check(config: dict, channels: int, classes: int, registry: Registry) -> list[dict]:
    if set(config) != {"schema", "window", "candidates", "seeds", "training", "budget", "selection", "abstention"} or config["schema"] != "phm.search.v1":
        raise ValueError("SEARCH_CONFIG_SCHEMA")
    shape = Shape(config["window"], channels, classes)
    if config["selection"] != "mean_seed_validation_macro_f1":
        raise ValueError("UNSUPPORTED_SELECTION")
    seeds = config["seeds"]
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("TWO_OR_MORE_DISTINCT_SEEDS_REQUIRED")
    for s in seeds:
        integer(s, "seed", 0, 2**32-1)
    budget = config["budget"]
    base_budget = {"max_parameters_per_model", "max_ensemble_parameters", "max_trials", "max_trial_seconds"}
    if set(budget) not in (base_budget, base_budget | {"max_multiply_accumulate_per_case"}):
        raise ValueError("BUDGET_SCHEMA")
    for field in ("max_parameters_per_model", "max_ensemble_parameters", "max_trials"):
        integer(budget[field], field, 1, 100_000_000)
    if "max_multiply_accumulate_per_case" in budget:
        integer(budget["max_multiply_accumulate_per_case"], "max_multiply_accumulate_per_case", 1, 10_000_000_000)
    if not np.isfinite(budget["max_trial_seconds"]) or budget["max_trial_seconds"] <= 0:
        raise ValueError("INVALID_TIME_BUDGET")
    training = config["training"]
    base_training = {"epochs", "batch_size", "learning_rate", "weight_decay", "patience"}
    if set(training) not in (base_training, base_training|{"loss"}):
        raise ValueError("TRAINING_SCHEMA")
    if "loss" in training:
        loss=training["loss"]
        if not isinstance(loss,dict) or "name" not in loss:
            raise ValueError("TRAINING_LOSS_SCHEMA")
        if loss["name"]=="weighted_cross_entropy":
            if set(loss)!={"name"}:raise ValueError("TRAINING_LOSS_SCHEMA")
        elif loss["name"]=="class_balanced_focal":
            if set(loss)!={"name","beta","gamma"}:raise ValueError("TRAINING_LOSS_SCHEMA")
            if type(loss["beta"]) not in (int,float) or not 0<=float(loss["beta"])<1:
                raise ValueError("INVALID_FOCAL_BETA")
            if type(loss["gamma"]) not in (int,float) or not 0<=float(loss["gamma"])<=10:
                raise ValueError("INVALID_FOCAL_GAMMA")
        else:
            raise ValueError("UNSUPPORTED_TRAINING_LOSS")
    for field in ("epochs", "batch_size", "patience"):
        integer(training[field], field, 1, 100_000)
    if not np.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("INVALID_LEARNING_RATE")
    if not np.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("INVALID_WEIGHT_DECAY")
    if set(config["abstention"]) != {"min_uncalibrated_score"}:
        raise ValueError("ABSTENTION_SCHEMA")
    q = config["abstention"]["min_uncalibrated_score"]
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("INVALID_ABSTENTION_THRESHOLD")
    if not config["candidates"]:
        raise ValueError("EMPTY_CANDIDATE_SPACE")
    records, ids = [], set()
    for spec in config["candidates"]:
        desc = registry.describe(spec, shape)
        key = digest(desc)[:16]
        if key in ids:
            raise ValueError("DUPLICATE_CANDIDATE")
        ids.add(key)
        feasible = (desc["parameter_count"] <= budget["max_parameters_per_model"] and
                    desc["parameter_count"] * len(seeds) <= budget["max_ensemble_parameters"])
        records.append({"id": key, "spec": spec, "description": desc,
                        "feasible": feasible, "reason": None if feasible else "PARAMETER_BUDGET_EXCEEDED"})
    if sum(r["feasible"] for r in records) * len(seeds) > budget["max_trials"]:
        raise ValueError("TRIAL_BUDGET_EXCEEDED_NO_SILENT_TRUNCATION")
    return records


def metrics(y: np.ndarray, predictions: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    matrix = confusion_matrix(y, predictions, labels=labels)
    recalls = np.divide(np.diag(matrix), matrix.sum(1), out=np.zeros(len(classes), dtype=float), where=matrix.sum(1) > 0)
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(recalls.mean()), "confusion_matrix": matrix.tolist(),
            "recall_by_class": dict(zip(classes, recalls.tolist())), "cases": int(len(y))}


def softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.exp(logits - logits.max(1, keepdims=True))
    return scores / scores.sum(1, keepdims=True)


@contextlib.contextmanager
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)


def _save_weights(model: nn.Module, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def _load_weights(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("NONFINITE_WEIGHTS")
    model.load_state_dict(state, strict=True)
    model.eval()



def _training_loss(training: dict, counts: np.ndarray, classes: int):
    if np.any(counts<=0):
        raise ValueError("TRAINING_CLASS_MISSING")
    spec=training.get("loss",{"name":"weighted_cross_entropy"})
    if spec["name"]=="weighted_cross_entropy":
        weights=torch.tensor(counts.sum()/(classes*counts),dtype=torch.float32)
        criterion=nn.CrossEntropyLoss(weight=weights)
        return criterion, "weighted_cross_entropy"
    if spec["name"]=="class_balanced_focal":
        beta=float(spec["beta"]);gamma=float(spec["gamma"])
        c=torch.tensor(counts,dtype=torch.float32)
        effective=1-torch.pow(torch.tensor(beta,dtype=torch.float32),c)
        weights=(1-beta)/torch.clamp(effective,min=1e-8)
        weights=weights/weights.sum()*classes
        def criterion(logits,target):
            plain=F.cross_entropy(logits,target,reduction="none")
            weighted=F.cross_entropy(logits,target,reduction="none",weight=weights.to(logits.device))
            pt=torch.exp(-plain)
            return (((1-pt)**gamma)*weighted).mean()
        return criterion, f"class_balanced_focal(beta={beta},gamma={gamma})"
    raise ValueError("UNSUPPORTED_TRAINING_LOSS")

def train_trial(registry: Registry, spec: dict, shape: Shape, seed: int,
                config: dict, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                destination: Path, classes: list[str]) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = registry.build(spec, shape, seed, config["budget"]["max_parameters_per_model"])
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    counts = np.bincount(y, minlength=shape.classes)
    criterion, loss_name = _training_loss(training, counts, shape.classes)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    tvx, tvy = torch.from_numpy(vx), torch.from_numpy(vy)
    rng = np.random.default_rng(seed)
    history, best_loss, best_epoch, stale, best = [], float("inf"), 0, 0, None
    # Fork also makes stochastic third-party CPU plugins replayable without
    # perturbing their caller's RNG. CUDA training is deliberately unsupported.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        for epoch in range(1, training["epochs"]+1):
            model.train()
            order = rng.permutation(len(y))
            losses = []
            for offset in range(0, len(y), training["batch_size"]):
                if time.perf_counter()-start > config["budget"]["max_trial_seconds"]:
                    raise TimeoutError("TRIAL_TIME_BUDGET_EXCEEDED")
                batch = order[offset:offset+training["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tx[batch]), ty[batch])
                if not torch.isfinite(loss):
                    raise ValueError("NONFINITE_TRAINING_LOSS")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                logits = model(tvx)
                validation_loss = float(criterion(logits, tvy))
            if not np.isfinite(validation_loss):
                raise ValueError("NONFINITE_VALIDATION_LOSS")
            history.append({"epoch": epoch, "train_loss_batch_mean": float(np.mean(losses)), "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-9:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best = {k: v.detach().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= training["patience"]:
                break
    if best is None:
        raise ValueError("NO_FINITE_CHECKPOINT")
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        scores = model(tvx).softmax(1).numpy()
    _save_weights(model, destination / "model.pt")
    result = {"status": "COMPLETED", "seed": seed, "epochs_executed": len(history), "best_epoch": best_epoch,
              "selection_within_trial": "minimum_development_validation_training_loss",
              "training_loss": loss_name,
              "validation": metrics(vy, scores.argmax(1), classes), "validation_scores": scores.tolist(),
              "history": history, "elapsed_seconds": time.perf_counter()-start,
              "weights_sha256": sha_file(destination / "model.pt")}
    write_json(destination / "trial.json", result)
    seal(destination, ["model.pt", "trial.json"])
    return result


def fit_baselines(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, classes: list[str]) -> list[dict]:
    fx, fvx = statistics(x), statistics(vx)
    mean, scale = fx.mean(0), np.maximum(fx.std(0), 1e-8)
    counts = np.bincount(y, minlength=len(classes))
    prior = counts/counts.sum()
    majority = {"kind": "majority", "scores": prior.tolist(),
                "validation": metrics(vy, np.full(len(vy), prior.argmax()), classes)}
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    classifier.fit((fx-mean)/scale, y)
    logistic = {"kind": "logistic_regression", "mean": mean.tolist(), "scale": scale.tolist(),
                "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
                "class_indices": classifier.classes_.tolist(), "feature_view": "seven_statistics_per_channel",
                "preprocessing_fit_role": "train_only"}
    logistic["validation"] = metrics(vy, baseline_scores(logistic, vx).argmax(1), classes)
    return [majority, logistic]


def baseline_scores(baseline: dict, x: np.ndarray) -> np.ndarray:
    if baseline["kind"] == "majority":
        return np.repeat(np.asarray(baseline["scores"])[None, :], len(x), axis=0)
    features = (statistics(x)-np.asarray(baseline["mean"])) / np.asarray(baseline["scale"])
    logits = features@np.asarray(baseline["coef"]).T + np.asarray(baseline["intercept"])
    if logits.shape[1] == 1:
        logits = np.column_stack((np.zeros(len(x)), logits[:, 0]))
    return softmax(logits)


def fit(dataset_path: str | Path, config_path: str | Path, output: str | Path, *, resume: bool = False,
        registry: Registry | None = None, on_trial_complete: Callable[[dict], None] | None = None) -> dict:
    registry = registry or default_registry()
    dataset = Dataset(dataset_path, allowed_splits={"train", "validation"})
    # Verify only the development files. No test manifest or telemetry is read.
    dataset.verify_files()
    config = read_json(config_path)
    classes = dataset.contract["classes"]
    shape = Shape(config["window"], len(dataset.contract["channels"]), len(classes))
    candidates = config_check(config, shape.channels, shape.classes, registry)
    train, val = dataset.load("train", shape.window), dataset.load("validation", shape.window)
    reject_numeric_overlap(train, val)
    scaler = fit_scaler(train.x)
    for spec in config["candidates"]:
        if spec["family"] == "mechanism_dag":
            from .physics import fit_feature_scaler
            p = spec["parameters"]
            if p["input_contract"] != dataset.contract or p["input_scaler"] != scaler:
                raise ValueError("PHYSICAL_BINDING_OR_TRAIN_SCALER_CHANGED")
            expected = fit_feature_scaler(p["program"],dataset.contract,train.x,p["feature_mode"],p["control_indices"])
            if expected != p["feature_scaler"]:
                raise ValueError("PHYSICAL_FEATURE_SCALER_NOT_TRAIN_ONLY")
    x, vx = transform(train.x, scaler), transform(val.x, scaler)
    from .reference import verify_training_binding
    verify_training_binding(config["candidates"],shape,x,train.y,dataset.contract,train.samples)
    fingerprint = digest({"config": config, "dataset": dataset.hash, "candidates": candidates,
                          "code": training_implementation_hash(), "environment": environment()})
    output = Path(output)
    with run_lock(output), deterministic_cpu():
        plan_path = output / "plan.json"
        if plan_path.exists():
            old = read_json(plan_path)
            if old["fingerprint"] != fingerprint:
                raise ValueError("RESUME_INPUT_OR_IMPLEMENTATION_CHANGED")
            if not resume:
                raise FileExistsError("RUN_EXISTS_USE_RESUME")
            if (output / "manifest.json").exists():
                verify(output)
                return read_json(output / "search_report.json")
        else:
            write_json(plan_path, {"schema": "phm.frozen-plan.v1", "fingerprint": fingerprint,
                       "config": config, "candidates": candidates, "environment": environment(),
                       "dataset_hash": dataset.hash, "implementation_sha256": training_implementation_hash(),
                       "holdout_access_in_fit": False})
        write_json(output / "development_snapshot.json", dataset.raw)
        write_json(output / "preprocessing.json", scaler)
        write_json(output / "data_counts.json", dataset.counts())
        write_json(output / "observation_hashes.json", [digest(a.tolist()) for a in np.concatenate((train.x, val.x))])
        results = []
        for candidate in candidates:
            if not candidate["feasible"]:
                results.append({**candidate, "status": "BUDGET_REJECTED", "trials": []})
                continue
            trials = []
            for seed in config["seeds"]:
                folder = output / "trials" / candidate["id"] / f"seed_{seed}"
                if (folder / "manifest.json").exists():
                    included = verify(folder)
                    has_trial, has_failure = "trial.json" in included, "failure.json" in included
                    if has_trial == has_failure:
                        raise ValueError("TRIAL_TERMINAL_ARTIFACT_SCHEMA")
                    record = read_json(folder / ("trial.json" if has_trial else "failure.json"))
                else:
                    try:
                        record = train_trial(registry, candidate["spec"], shape, seed, config,
                                             x, train.y, vx, val.y, folder, classes)
                    except Exception as exc:
                        record = {"status": "FAILED", "seed": seed, "exception": type(exc).__name__, "reason": str(exc)}
                        write_json(folder / "failure.json", record)
                        seal(folder, ["failure.json"])
                trials.append(record)
                if on_trial_complete:
                    # Exceptions from this callback are deliberate interruption,
                    # not failed model training. Completed trials stay committed.
                    on_trial_complete({"candidate": candidate["id"], **record})
            complete = all(t["status"] == "COMPLETED" for t in trials)
            item = {**candidate, "status": "COMPLETED" if complete else "INCOMPLETE_SEEDS", "trials": trials}
            if complete:
                fs = [t["validation"]["macro_f1"] for t in trials]
                item["mean_validation_macro_f1"] = float(np.mean(fs))
                item["seed_std_macro_f1"] = float(np.std(fs))
                ensemble_scores = np.asarray([t["validation_scores"] for t in trials]).mean(0)
                item["ensemble_validation"] = metrics(val.y, ensemble_scores.argmax(1), classes)
            results.append(item)
            write_json(output / "progress.json", {"fingerprint": fingerprint, "candidates": results})
        eligible = [r for r in results if r["status"] == "COMPLETED"]
        # F1 first, seed stability second, total parameters third; all frozen.
        eligible.sort(key=lambda r: (-r["mean_validation_macro_f1"], r["seed_std_macro_f1"], r["description"]["parameter_count"], r["id"]))
        baselines = fit_baselines(x, train.y, vx, val.y, classes)
        best_baseline = sorted(baselines, key=lambda r: (-r["validation"]["macro_f1"], r["kind"]))[0]
        winner = eligible[0] if eligible else None
        baseline_path = output / "baseline.json"
        write_json(baseline_path, best_baseline)
        report = {"schema": "phm.search-report.v1", "status": "COMPLETED" if winner else "NO_FEASIBLE_NEURAL_CANDIDATE",
                  "fingerprint": fingerprint, "counts": dataset.counts(), "candidate_results": results,
                  "best_neural_candidate": winner["id"] if winner else None,
                  "best_non_neural_baseline": best_baseline["kind"], "baseline_results": baselines,
                  "neural_output_replaced_by_baseline": False, "test_used": False,
                  "selection_scope": "best observed candidate in the frozen finite search; no global or statistical optimality guarantee",
                  "claim_status": "RESEARCH_ONLY", "simulation_performance_qualified": False,
                  "real_device_qualified": False, "production_release_qualified": False,
                  "calibration_status": "NOT_CALIBRATED", "optimizer_state_resume": False,
                  "resume_granularity": "completed_trial", "implementation_sha256": training_implementation_hash()}
        if winner:
            weight_paths = [f"trials/{winner['id']}/seed_{s}/model.pt" for s in config["seeds"]]
            bundle = {"schema": "phm.neural-ensemble.v1", "spec": winner["spec"],
                      "architecture_description": winner["description"], "shape": shape.__dict__,
                      "contract": dataset.contract, "scaler": scaler, "weights": weight_paths,
                      "seeds": config["seeds"], "max_parameters": config["budget"]["max_parameters_per_model"],
                      "abstention": config["abstention"], "score_semantics": "uncalibrated_known_class_softmax",
                      "fingerprint": fingerprint, "torch_version": str(torch.__version__),
                      "training_roles": ["train"], "validation_used_for_early_stopping": True,
                      "refit_on_validation": False, "claim_status": "RESEARCH_ONLY"}
            write_json(output / "model_bundle.json", bundle)
        write_json(output / "search_report.json", report)
        files = [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
                 and p.name not in {".lock", "manifest.json"} and not p.name.endswith(".tmp")]
        seal(output, files)
        return report


def load_ensemble(root: str | Path, registry: Registry | None = None) -> tuple[dict, list[nn.Module]]:
    registry = registry or default_registry()
    root = Path(root)
    files = verify(root)
    if "model_bundle.json" not in files:
        raise ValueError("NO_FROZEN_NEURAL_MODEL")
    bundle = read_json(root / "model_bundle.json")
    if bundle["schema"] != "phm.neural-ensemble.v1" or bundle["torch_version"] != str(torch.__version__):
        raise ValueError("MODEL_SCHEMA_OR_TORCH_VERSION_MISMATCH")
    shape = Shape(**bundle["shape"])
    if registry.describe(bundle["spec"], shape) != bundle["architecture_description"]:
        raise ValueError("ARCHITECTURE_IMPLEMENTATION_CHANGED")
    models = []
    for path, seed in zip(bundle["weights"], bundle["seeds"], strict=True):
        if path not in files:
            raise ValueError("UNSEALED_WEIGHT_PATH")
        model = registry.build(bundle["spec"], shape, seed, bundle["max_parameters"])
        _load_weights(model, safe_child(root, path))
        models.append(model)
    if not models:
        raise ValueError("EMPTY_ENSEMBLE")
    return bundle, models


def ensemble_scores(models: list[nn.Module], x: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        result = np.asarray([m(torch.from_numpy(x)).softmax(1).numpy() for m in models]).mean(0)
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_PREDICTION")
    return result


def evaluate(run: str | Path, holdout: str | Path, output: str | Path, registry: Registry | None = None) -> dict:
    run, output = Path(run), Path(output)
    with run_lock(run), deterministic_cpu():
        bundle, models = load_ensemble(run, registry)
        dataset = Dataset(holdout, allowed_splits={"test"})
        if dataset.contract != bundle["contract"]:
            raise ValueError("HOLDOUT_CONTRACT_MISMATCH")
        development = read_json(run / "development_snapshot.json")
        reject_overlap(development["samples"], dataset.samples)
        ledger = run / "evaluation_ledger.json"
        key = digest({"run": bundle["fingerprint"], "holdout": dataset.hash})
        if ledger.exists():
            old = read_json(ledger)
            if old.get("key") != key:
                raise ValueError("HOLDOUT_ALREADY_REGISTERED_DIFFERENT_DATASET")
            if old["status"] == "COMPLETE":
                # Return the prior result, never silently claim fresh evidence.
                return old["report"]
            raise ValueError("HOLDOUT_ALREADY_OPENED_INCOMPLETE_REVIEW_REQUIRED")
        dataset.verify_files()
        # Mark opening BEFORE reading values. Failure remains visible and does
        # not permit trying another test set until reviewed outside this tool.
        write_json(ledger, {"key": key, "status": "OPENED", "holdout_hash": dataset.hash})
        test = dataset.load("test", bundle["shape"]["window"])
        prior_hashes = set(read_json(run / "observation_hashes.json"))
        if prior_hashes & {digest(x.tolist()) for x in test.x}:
            raise ValueError("IDENTICAL_OBSERVATIONS_ACROSS_ROLES")
        x = transform(test.x, bundle["scaler"])
        scores = ensemble_scores(models, x)
        predictions = scores.argmax(1)
        baseline = read_json(run / "baseline.json")
        base_predictions = baseline_scores(baseline, x).argmax(1)
        classes = bundle["contract"]["classes"]
        threshold = bundle["abstention"]["min_uncalibrated_score"]
        accepted = scores.max(1) >= threshold
        # Warm single-case host latency; not a deployment-target real-time gate.
        for _ in range(3):
            ensemble_scores(models, x[:1])
        start = time.perf_counter()
        for _ in range(20):
            ensemble_scores(models, x[:1])
        milliseconds = (time.perf_counter()-start)*1000/20
        result_rows = [{"id": s["id"], "group_id": s["group_id"], "truth": classes[int(y)],
                        "prediction": classes[int(p)], "accepted": bool(a), "class_scores": sc.tolist()}
                       for s,y,p,a,sc in zip(test.samples, test.y, predictions, accepted, scores)]
        report = {"schema": "phm.holdout-report.v1", "status": "EVALUATED", "fingerprint": bundle["fingerprint"],
                  "holdout_hash": dataset.hash, "counts": dataset.counts(), "neural": metrics(test.y, predictions, classes),
                  "baseline": {"kind": baseline["kind"], **metrics(test.y, base_predictions, classes)},
                  "selective_prediction": {"fixed_score_threshold": threshold, "coverage": float(accepted.mean()),
                     "accepted_accuracy": float((predictions[accepted] == test.y[accepted]).mean()) if accepted.any() else None,
                     "correct_and_accepted_fraction_all_cases": float(((predictions == test.y) & accepted).mean()),
                     "unknown_fault_detection_validated": False, "probability_calibrated": False},
                  "latency": {"host_cpu_mean_ms_single_case_ensemble": milliseconds, "target_hardware_qualified": False},
                  "predictions": result_rows, "test_used_for_selection": False, "open_count_in_this_run": 1,
                  "claim_status": "SIMULATION_RESEARCH_EVIDENCE" if dataset.contract["evidence_type"] == "simulation" else "SOFTWARE_OR_UNREVIEWED_EVIDENCE",
                  "real_device_qualified": False, "production_release_qualified": False,
                  "independence_scope": "held-out simulated parameter lineages; shared simulator, no real-world independence claim"}
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "evaluation_report.json", report)
        seal(output, ["evaluation_report.json"])
        write_json(ledger, {"key": key, "status": "COMPLETE", "holdout_hash": dataset.hash, "report": report})
        return report


def predict(run: str | Path, csv_path: str | Path, units: dict, registry: Registry | None = None) -> dict:
    with deterministic_cpu():
        bundle, models = load_ensemble(run, registry)
        expected_units = {c["name"]: c["unit"] for c in bundle["contract"]["channels"]}
        if units != expected_units:
            raise ValueError("INPUT_UNITS_MISMATCH")
        x = sequence_from_csv(Path(csv_path), bundle["contract"], bundle["shape"]["window"])[None]
        scores = ensemble_scores(models, transform(x, bundle["scaler"]))[0]
        best = int(scores.argmax())
        accepted = bool(scores[best] >= bundle["abstention"]["min_uncalibrated_score"])
        return {"status": "PREDICTED" if accepted else "ABSTAIN_LOW_SCORE", "prediction": bundle["contract"]["classes"][best] if accepted else None,
                "closed_set_argmax": bundle["contract"]["classes"][best],
                "scores": dict(zip(bundle["contract"]["classes"], scores.tolist())),
                "probability_calibrated": False, "unknown_fault_detection_validated": False,
                "claim_status": "RESEARCH_ONLY", "model_fingerprint": bundle["fingerprint"]}
