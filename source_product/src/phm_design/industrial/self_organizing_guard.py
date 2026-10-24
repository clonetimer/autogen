"""R83b opt-in process-isolated deadline for self-organizing neural research.

Runs on Windows/POSIX via multiprocessing.spawn. The parent terminates its
single training worker at the predeclared wall deadline, including startup,
search, refit, model return and serialization. This is NOT a container sandbox,
CPU/GPU quota, filesystem isolation, or independent qualification.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile

from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec
from .design_engine import DesignAcceptance, IndustrialDesignRejected
from .self_organizing import (
    EvolutionPolicy, SelfOrganizingArtifact, run_self_organizing_design,
)
from .resource_guard import GuardedDesignTimeout, _invoke_worker


def _self_organizing_worker(conn, raw, records, policy, acceptance):
    """Spawn-safe top-level training worker, no persistent background process."""
    try:
        result = run_self_organizing_design(
            raw, records, policy=policy, acceptance=acceptance,
            _guarded_execution=True,
        )
        # Do not Pipe.send() live Torch tensors: PyTorch shares their storage
        # through child-owned file descriptors that can disappear at worker exit.
        # Transfer only bounded, checksum-verified numeric NPZ and JSON bytes.
        with tempfile.TemporaryDirectory(prefix="r83b-worker-") as temp:
            folder = Path(temp) / "model"
            result.export(folder)
            manifest = (folder / "manifest.json").read_bytes()
            weights = (folder / "weights.npz").read_bytes()
            if len(manifest) > 1_000_000 or len(weights) > 12_000_000:
                raise ValueError("SELF_ORGANIZING_WORKER_EXPORT_TOO_LARGE")
            conn.send(("delivered", (manifest, weights)))
    except IndustrialDesignRejected as exc:
        conn.send(("refused", exc.audit))
    except BaseException as exc:
        conn.send(("failed", (type(exc).__name__, str(exc))))
    finally:
        conn.close()


def run_self_organizing_design_guarded(
    task_spec: Mapping | TaskSpec,
    records: Sequence[Mapping],
    *,
    policy: EvolutionPolicy = EvolutionPolicy(),
    acceptance: DesignAcceptance = DesignAcceptance(),
) -> SelfOrganizingArtifact:
    """Only documented hard-wall execution entry for R83 self-organization.

    The source TaskSpec requires design.budget.max_wall_seconds and declares
    existing group, environment, signal, budget, and TRAIN-only constraints.
    Failure returns no trained artifact. Source TRAIN folds remain selection
    feedback rather than an independent OOD test.
    """
    if not isinstance(policy, EvolutionPolicy):
        raise ValueError("INVALID_EVOLUTION_POLICY_TYPE")
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_DESIGN_ACCEPTANCE_TYPE")
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    plan = compile_task_spec(raw)
    limit = plan["budget"].get("max_wall_seconds")
    if limit is None:
        raise TaskSpecRejected("SELF_ORGANIZING_HARD_WALL_BUDGET_REQUIRED")
    if policy.max_candidates > plan["budget"]["max_trials"]:
        raise TaskSpecRejected("SELF_ORGANIZING_TRIAL_BUDGET_EXCEEDED")

    status, result = _invoke_worker(
        _self_organizing_worker, (raw, records, policy, acceptance), limit
    )
    if status == "refused":
        raise IndustrialDesignRejected(result)
    if status == "failed":
        raise RuntimeError("GUARDED_SELF_ORGANIZING_WORKER_FAILED:" + str(result))
    if status != "delivered":
        raise RuntimeError("UNKNOWN_SELF_ORGANIZING_WORKER_RESULT")
    if (not isinstance(result, tuple) or len(result) != 2 or
            not all(isinstance(blob, bytes) for blob in result)):
        raise RuntimeError("INVALID_SELF_ORGANIZING_WORKER_ARTIFACT")
    manifest, weights = result
    if len(manifest) > 1_000_000 or len(weights) > 12_000_000:
        raise RuntimeError("OVERSIZED_SELF_ORGANIZING_WORKER_ARTIFACT")
    with tempfile.TemporaryDirectory(prefix="r83b-parent-") as temp:
        folder = Path(temp) / "model"
        folder.mkdir()
        (folder / "manifest.json").write_bytes(manifest)
        (folder / "weights.npz").write_bytes(weights)
        artifact = SelfOrganizingArtifact.load(folder)
    if (artifact.audit.get("independent_qualification") is not False or
            artifact.audit.get("wall_guard") != "PARENT_SPAWN_TERMINATION"):
        raise RuntimeError("INVALID_SELF_ORGANIZING_WORKER_AUDIT")
    return artifact


__all__ = [
    "GuardedDesignTimeout",
    "run_self_organizing_design_guarded",
]
