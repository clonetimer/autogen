"""Clean-install 0.8.0rc1 smoke for a frozen release candidate.

Run this from source_product/scripts/ *after installing the BUILT wheel* in
a fresh Python environment containing the pinned real torch_like provider.
The CLI subprocesses run outside the working tree with PYTHONPATH unset.

This proves package distribution, genuine provider fit/predict and source-only
design CLI. Synthetic example metadata MUST NOT be called real factory data.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

RC_VERSION = "0.8.0rc1"


def _run(argv, cwd):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, *argv], cwd=cwd, env=env,
        text=True, capture_output=True, timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"RELEASE_SMOKE_FAILURE: {argv}\n"
            f"stdout={result.stdout[-4000:]}\n"
            f"stderr={result.stderr[-4000:]}"
        )
    return result.stdout


def main():
    import phm_design
    import torch_like
    from phm_design.industrial_contracts.torch_like_bridge import available_adapters

    project = Path(__file__).resolve().parents[1]
    examples = project / "examples" / "industrial_contracts"
    assert phm_design.__version__ == RC_VERSION
    assert importlib.metadata.version("phm-design") == RC_VERSION
    assert Path(phm_design.__file__).is_file()
    package_file = Path(phm_design.__file__).resolve()
    assert "site-packages" in str(package_file).lower(), (
        "candidate must be imported from an INSTALLED wheel, not the source tree",
        str(package_file),
    )
    assert torch_like.__version__ == "0.9.0", (
        "external provider revision changed unexpectedly",
        torch_like.__version__,
    )
    assert len(available_adapters()) == 4
    with tempfile.TemporaryDirectory(prefix="autogen-rc-smoke-") as temp:
        work = Path(temp)
        assert "usage:" in _run(["-m", "phm_design", "--help"], work).lower()
        assert "usage:" in _run([
            "-m", "phm_design.industrial_contracts.design_cli", "--help"
        ], work).lower()
        manifest = [
            "--equipment", str(examples/"bearing_equipment.json"),
            "--train-manifest", str(examples/"bearing_train_only.json"),
            "--trial", str(examples/"bearing_trial.json"),
        ]
        design = work/"frozen-source-study.json"
        _run([
            "-m", "phm_design.industrial_contracts.design_cli",
            "propose", *manifest, "--output", str(design),
        ], work)
        proposal = json.loads(design.read_text(encoding="utf-8"))
        assert proposal["training_already_run"] is False
        assert proposal["independent_validation_or_final_accessed"] is False
        assert len(proposal["study_sha256"]) == 64
        output = work/"source-only-scores.json"
        _run([
            "-m", "phm_design.industrial_contracts.design_cli",
            "source-study", *manifest,
            "--source-features", str(examples/"bearing_train_features.json"),
            "--frozen-study", str(design),
            "--expected-study-sha256", proposal["study_sha256"],
            "--output", str(output),
        ], work)
        scores = json.loads(output.read_text(encoding="utf-8"))
        assert scores["status"] == "SOURCE_TRAIN_OOF_DESCRIPTIVE_SHORTLIST_ONLY"
        assert scores["source_fit_calls_actual"] > 0
        assert scores["algorithm_generalization_proven"] is False
        assert scores["production_qualified"] is False

        refusal = work/"refused.json"
        bad = subprocess.run(
            [sys.executable,
             "-m", "phm_design.industrial_contracts.design_cli", "propose",
             "--equipment", str(examples/"bearing_equipment.json"),
             "--train-manifest", str(examples/"bearing_trusted_audit.json"),
             "--trial", str(examples/"bearing_trial.json"),
             "--output", str(refusal)],
            cwd=work, capture_output=True, text=True, timeout=60,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
        assert bad.returncode != 0
        assert "R93_PROTECTED_ROLE_IN_DESIGN_WORKER" in bad.stderr
        assert not refusal.exists()

    print(json.dumps({
        "scope": "CLEAN_WHEEL_AND_GENUINE_PROVIDER_SMOKE",
        "wheel_version": RC_VERSION,
        "external_torch_like_version": torch_like.__version__,
        "native_provider_candidates": len(available_adapters()),
        "source_only_fit_calls": scores["source_fit_calls_actual"],
        "source_fit_validation_or_final_accessed": False,
        "independent_real_equipment_validated": False,
        "general_industrial_deployment_qualified": False,
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
