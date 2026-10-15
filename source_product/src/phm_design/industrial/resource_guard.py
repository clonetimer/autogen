"""Spawn-isolated, predeclared wall-clock deadline for research AutoDesign.

Unlike process_time checks inside a fit, the parent can terminate an actively
running Python model-fit child. Supports Windows spawn and POSIX spawn.
Does not enforce GPU memory, disk quotas, CPU-time preemption, or child trees.
No qualified industrial deployment is implied.
"""
from __future__ import annotations

import multiprocessing as mp
import time

from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec


class GuardedDesignTimeout(RuntimeError):
    """Wall-clock hard deadline expired: no trained model is delivered."""

    def __init__(self, wall_seconds: int):
        self.audit = {
            "schema": "phm.industrial.resource-guard.v1",
            "decision": "HARD_WALL_TIMEOUT",
            "max_wall_seconds": wall_seconds,
            "model_delivered": False,
            "independent_qualification": False,
        }
        super().__init__("HARD_WALL_TIMEOUT")


def _worker(conn, raw, records, acceptance):
    # This callable is top-level for Windows and always uses the spawn context.
    from .design_engine import IndustrialDesignRejected, run_design

    try:
        model = run_design(raw, records, acceptance=acceptance,
                           _guarded_execution=True)
        conn.send(("delivered", model))
    except IndustrialDesignRejected as exc:
        conn.send(("refused", exc.audit))
    except BaseException as exc:
        conn.send(("failed", (type(exc).__name__, str(exc))))
    finally:
        conn.close()


def _terminate(process):
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def run_design_guarded(task_spec, records, *, acceptance=None):
    """Train in a separate process under an opt-in HARD wall time bound.

    The TaskSpec MUST declare design.budget.max_wall_seconds. This is a wall
    limit measured by the parent from before spawning until the worker emits a
    complete result; the parent terminates the worker when it runs out.
    """
    from .design_engine import DesignAcceptance, IndustrialDesignRejected

    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    plan = compile_task_spec(raw)
    limit = plan["budget"].get("max_wall_seconds")
    if limit is None:
        raise TaskSpecRejected("HARD_WALL_BUDGET_NOT_DECLARED")
    if acceptance is None:
        acceptance = DesignAcceptance()
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_DESIGN_ACCEPTANCE_TYPE")

    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_worker, args=(writer, raw, records, acceptance))
    started = time.monotonic()
    try:
        child.start()
        writer.close()
        remaining = max(0.0, limit - (time.monotonic() - started))
        if not reader.poll(remaining):
            _terminate(child)
            raise GuardedDesignTimeout(limit)
        try:
            status, result = reader.recv()
        except EOFError as exc:
            raise RuntimeError("GUARDED_WORKER_EXITED_WITHOUT_AUDIT") from exc
        if time.monotonic() - started > limit:
            _terminate(child)
            raise GuardedDesignTimeout(limit)
        if status == "refused":
            raise IndustrialDesignRejected(result)
        if status == "failed":
            raise RuntimeError("GUARDED_WORKER_FAILED:" + str(result))
        if status != "delivered":
            raise RuntimeError("UNKNOWN_GUARDED_WORKER_RESULT")
        return result
    finally:
        reader.close()
        writer.close()
        if child.pid is not None:
            _terminate(child)
            child.join(timeout=1)
