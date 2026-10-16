"""Collect an identity-only source binding from the ACTUAL installed Basilisk.

Run in the same trusted environment as the native campaign. The emitted JSON
is compatible with R82 intake, but is NOT a signed execution receipt, a
physics-model review or an independent instrument verification.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
from pathlib import Path

from ..io import safe_child, sha_file, write_json


def collect_native_source_binding(native_source_checkout, destination):
    """Fail closed unless an actual installed Basilisk wheel factory is present.

    Do not invent a Basilisk version, wheel source location, or SHA. The
    simulation's deployed Python environment must import the same module.
    """
    root = Path(native_source_checkout)
    paths = {
        "runtime_source": ("src/sat_sim/bsk_engine/unified_native.py",
                           "src/sat_sim/bsk_engine/unified_native.py"),
        "fault_effect_source": ("src/sat_sim/bsk_engine/project_native_modules.py",
                                "src/sat_sim/bsk_engine/project_native_modules.py"),
    }
    result = {
        "scope": "simulation_source_binding_only",
        "physical_source_independently_attested": False,
        "basilisk_physics_executed_by_collector": False,
    }
    for key, (reference, relative) in paths.items():
        f = safe_child(root, relative)
        if not f.is_file():
            raise FileNotFoundError("R82_NATIVE_SOURCE_NOT_FOUND:" + key)
        result[key] = {"reference": reference, "sha256": sha_file(f)}
    try:
        from Basilisk.utilities import simIncludeRW
        installed = md.version("bsk")
    except (ImportError, ModuleNotFoundError, md.PackageNotFoundError) as exc:
        raise RuntimeError("R82_INSTALLED_BASILISK_REQUIRED") from exc
    wheel = Path(simIncludeRW.__file__).resolve()
    if wheel.name != "simIncludeRW.py" or not wheel.is_file():
        raise RuntimeError("R82_BASILISK_WHEEL_FACTORY_LOCATION_UNKNOWN")
    result["wheel_factory_source"] = {
        "reference": "Basilisk/utilities/simIncludeRW.py",
        "sha256": sha_file(wheel),
    }
    result["installed_bsk_version"] = installed
    result["installed_wheel_source_path"] = str(wheel)
    out = Path(destination)
    if out.exists():
        raise FileExistsError("R82_NATIVE_BINDING_ALREADY_EXISTS")
    write_json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source-checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    collect_native_source_binding(args.native_source_checkout, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
