"""Check distributable research-contract CLI from BUILT wheel without torch_like.

This is deliberately NOT a replacement for the RC blocker: actual private
torch_like native fit/predict/save/load must pass separately. This smoke only
proves wheel portability and a non-mutating TRAIN-only design proposal.
"""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import phm_design
    assert phm_design.__version__ == "0.8.0rc1"
    assert metadata.version("phm-design") == "0.8.0rc1"
    source = Path(phm_design.__file__).resolve()
    assert "site-packages" in str(source).lower(), ("wrong import source", source)
    examples = Path(__file__).resolve().parents[1] / "examples" / "industrial_contracts"
    command = [
        sys.executable, "-m", "phm_design.industrial_contracts.design_cli",
        "propose",
        "--equipment", str(examples / "bearing_equipment.json"),
        "--train-manifest", str(examples / "bearing_train_only.json"),
        "--trial", str(examples / "bearing_trial.json"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory(prefix="autogen-core-rc-") as dirname:
        out = Path(dirname) / "source-proposal.json"
        result = subprocess.run(command + ["--output", str(out)],
                                env=env, cwd=dirname, capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["source_data_origin_self_reported"] == "synthetic"
        assert payload["independent_validation_or_final_accessed"] is False
        assert payload["candidate_count"] >= 2
        assert len(payload["study_sha256"]) == 64
    print(json.dumps({
        "status": "PASS", "installed_version": phm_design.__version__,
        "candidate_count": payload["candidate_count"],
        "artifact_origin": "BUILT_WHEEL",
        "genuine_torch_like_fit_verified_by_this_test": False,
        "industrial_accuracy_proven": False,
        "needs_real_provider_gate": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
