"""Cross-equipment design contracts; research-only P1 API, no model training."""
from .contracts import (
    ContractRejected, validate_equipment, validate_trial,
    validate_dataset, compile_source_plan,
)
