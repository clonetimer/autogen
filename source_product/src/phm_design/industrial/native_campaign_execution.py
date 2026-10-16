"""R82.3 trusted native Basilisk campaign runner (NOT an AutoDesign worker).

Runs the real SatAgentSimulation RW factory only with an importable approved
Basilisk runtime and an unmodified Git checkout. Never installs dependencies,
falls back to proxy, or promotes self-reported physics into qualification.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from ..io import read_json, sha_file, write_json
from .native_campaign_gate import review_native_rw_train_campaign

_ACCEPTED_BSK = {"2.11.0", "2.11.0+satfix1"}
_CAPABILITY = "subsystem.adcs_unified_native.v1"
_SCRIPT = Path("scripts/generate_astrograph_rw_fault_dataset.py")
_CONFIG_KEYS = frozenset({
    "schema_version", "campaign_id", "base_seed", "counts", "fault_ids",
    "duration_s", "sample_s", "solver_step_s", "capability_id",
    "simulation_backend", "allow_test_proxy", "output_root",
})


def inspect_native_campaign_prerequisites(source_checkout, config):
    """Check exact source and installed runtime before making any output."""
    source_checkout = Path(source_checkout).resolve()
    config = Path(config).resolve()
    if (not (source_checkout / _SCRIPT).is_file() or
            not (source_checkout / "src/sat_sim/bsk_engine/unified_native.py").is_file() or
            not (source_checkout / "src/sat_sim/bsk_engine/project_native_modules.py").is_file()):
        raise RuntimeError("R82_NATIVE_PRODUCER_CHECKOUT_INCOMPLETE")
    settings = read_json(config)
    if (not isinstance(settings, dict) or
            set(settings) != _CONFIG_KEYS or
            settings.get("schema_version") !=
            "sat-sim.astrograph-fault-dataset-factory.v3" or
            settings.get("simulation_backend") != "basilisk" or
            settings.get("capability_id") != _CAPABILITY or
            settings.get("allow_test_proxy") is not False):
        raise ValueError("R82_NATIVE_ONLY_FACTORY_CONFIG_REQUIRED")
    counts = settings["counts"]
    classes = settings["fault_ids"]
    if (not isinstance(counts, dict) or
            set(counts) != {"train", "validation", "test"} or
            not all(type(v) is int and v >= 1 for v in counts.values()) or
            set(classes) != {
                "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM",
                "ADCS_RW_TORQUE_AUTHORITY_LOSS"} or len(classes) != 3):
        raise ValueError("R82_NATIVE_FACTORY_SPLIT_OR_ONTOLOGY_INVALID")
    try:
        from Basilisk.utilities import simIncludeRW
        installed = importlib.metadata.version("bsk")
    except (ModuleNotFoundError, ImportError,
            importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("R82_BASILISK_RUNTIME_NOT_INSTALLED") from exc
    if installed not in _ACCEPTED_BSK:
        raise RuntimeError("R82_BASILISK_RUNTIME_VERSION_NOT_APPROVED:" + installed)
    wheel = Path(simIncludeRW.__file__).resolve()
    if wheel.name != "simIncludeRW.py" or not wheel.is_file():
        raise RuntimeError("R82_BASILISK_RUNTIME_SOURCE_UNAVAILABLE")
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_checkout,
            text=True, capture_output=True, check=True, timeout=15).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source_checkout, text=True, capture_output=True,
            check=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("R82_NATIVE_PRODUCER_GIT_REVISION_REQUIRED") from exc
    if (len(git_sha) != 40 or
            any(ch not in "0123456789abcdef" for ch in git_sha) or dirty):
        raise RuntimeError("R82_NATIVE_PRODUCER_CHECKOUT_NOT_CLEAN")
    return {
        "schema": "phm.industrial.native-execution-prerequisites.v1",
        "source_revision": git_sha,
        "producer_script_sha256": sha_file(source_checkout / _SCRIPT),
        "native_runtime_sha256": sha_file(
            source_checkout / "src/sat_sim/bsk_engine/unified_native.py"),
        "native_fault_effect_sha256": sha_file(
            source_checkout / "src/sat_sim/bsk_engine/project_native_modules.py"),
        "bsk_version": installed,
        "installed_wheel_factory_path": str(wheel),
        "installed_wheel_factory_sha256": sha_file(wheel),
        "configuration_sha256": sha_file(config),
        "proxy_fallback_allowed": False,
        "native_execution_performed": False,
        "independent_qualification": False,
    }


def execute_native_rw_campaign(*, source_checkout, config, output,
                               evidence_dir):
    """Generate real producer output, review TRAIN candidate evidence.

    Runs in an operator-controlled native environment. An error, incompatible
    module, or failed candidate review produces an explicit refusal receipt.
    Never accesses FINAL/TEST trace bytes during the post-run review.
    """
    source_checkout = Path(source_checkout).resolve()
    config = Path(config).resolve()
    output = Path(output).resolve()
    evidence_dir = Path(evidence_dir).resolve()
    prereq = inspect_native_campaign_prerequisites(source_checkout, config)
    if output.exists() or evidence_dir.exists():
        raise FileExistsError("R82_NATIVE_EXECUTION_OUTPUT_EXISTS")
    if (output.is_relative_to(source_checkout) or
            evidence_dir.is_relative_to(source_checkout) or
            output.is_relative_to(evidence_dir) or
            evidence_dir.is_relative_to(output)):
        raise ValueError("R82_NATIVE_EXECUTION_OUTPUT_PATH_ISOLATION")
    evidence_dir.mkdir(parents=True)
    cmd = [
        sys.executable, str(source_checkout / _SCRIPT),
        "--config", str(config), "--output", str(output), "--fail-fast",
    ]
    receipt = {
        **prereq, "schema": "phm.industrial.r82-native-execution.v1",
        "native_execution_attempted": True,
        "native_execution_performed": False,
        "source_verified_before_launch": True,
        "status": "STARTED", "model_trained": False,
        "physical_execution_independently_attested": False,
        "independent_qualification": False,
        "argv": cmd,
    }
    logs = evidence_dir / "producer.log"
    try:
        # The producer uses the native environment and its own strict
        # source/physics validation chain. This runner does not install bsk.
        with logs.open("w", encoding="utf-8") as out:
            completed = subprocess.run(
                cmd, cwd=source_checkout, stdout=out,
                stderr=subprocess.STDOUT, check=False)
        receipt["producer_exit_code"] = int(completed.returncode)
        receipt["producer_log_sha256"] = sha_file(logs)
        if completed.returncode != 0:
            raise RuntimeError("R82_NATIVE_CAMPAIGN_PRODUCER_FAILED")
        index = output / "astrograph_dataset_index.json"
        if not index.is_file():
            raise RuntimeError("R82_NATIVE_CAMPAIGN_INDEX_MISSING")
        index_data = read_json(index)
        if index_data.get("dry_run") is not False:
            raise RuntimeError("R82_NATIVE_CAMPAIGN_DRY_RUN_ONLY")
        receipt["producer_index_sha256"] = sha_file(index)
        source_review = review_native_rw_train_campaign(output)
        expected_pairs = (
            read_json(config)["counts"]["train"] *
            len(read_json(config)["fault_ids"])
        )
        if source_review["train_pairs"] != expected_pairs:
            raise RuntimeError("R82_NATIVE_TRAIN_PAIR_COUNT_MISMATCH")
        if source_review["source_bsk_version_claimed"] != prereq["bsk_version"]:
            raise RuntimeError("R82_NATIVE_CAMPAIGN_RUNTIME_VERSION_MISMATCH")
        receipt["expected_train_pairs"] = expected_pairs
        receipt["train_candidate_review"] = source_review
        receipt["status"] = "NATIVE_TRAIN_CANDIDATE_REVIEWED"
        receipt["native_execution_performed"] = True
        receipt["independent_qualification"] = False
        write_json(evidence_dir / "execution_receipt.json", receipt)
        return receipt
    except Exception as exc:
        receipt["status"] = "REFUSED"
        receipt["refusal_code"] = str(exc)
        receipt["native_execution_performed"] = False
        if logs.is_file():
            receipt["producer_log_sha256"] = sha_file(logs)
        write_json(evidence_dir / "execution_receipt.json", receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    execute_native_rw_campaign(
        source_checkout=args.source_checkout, config=args.config,
        output=args.output, evidence_dir=args.evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
