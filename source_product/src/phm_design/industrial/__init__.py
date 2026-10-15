"""Equipment-agnostic industrial diagnostic TaskSpec and plan contracts (R80)."""
from .taskspec import (
    TaskSpec, TaskSpecRejected, audit_train_manifest,
    compile_task_spec, validate_observations,
)

__all__ = [
    "TaskSpec", "TaskSpecRejected", "audit_train_manifest",
    "compile_task_spec", "validate_observations",
]
