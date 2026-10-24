"""Standalone PHYSICAL Basilisk 2.11 reaction-wheel dynamic integration smoke.

This intentionally does NOT call SatAgentSimulation's campaign runner.
No Python approximation substitutes for wheel dynamics. Only Basilisk
spacecraft+state effector and its recorded wheel-speed message are used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import reactionWheelStateEffector, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simHelpers, simIncludeRW


def simulate(output: Path, *, torque_x: float = 0.02, seconds: float = 12.0):
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("bsk_process")
    proc.addTask(sim.CreateNewTask("bsk_task", macros.sec2nano(0.1)))

    body = spacecraft.Spacecraft()
    body.ModelTag = "nativeSpacecraft"
    body.hub.mHub = 250.0
    body.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
        np.diag([120.0, 115.0, 98.0]).reshape(-1)
    )
    wheel_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
    wheel_factory = simIncludeRW.rwFactory()
    for axis, rpm in zip(
        ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]), (250., 300., 225.)
    ):
        wheel_factory.create(
            "Honeywell_HR16", axis, Omega=float(rpm), maxMomentum=50.0,
            u_max=0.2, useMaxTorque=True
        )
    wheel_factory.addToSpacecraft("rws", wheel_effector, body)
    if len(wheel_factory.rwList) != 3:
        raise RuntimeError("RW_FACTORY_ATTACHMENT_FAILED")
    payload = messaging.ArrayMotorTorqueMsgPayload()
    payload.motorTorque = [float(torque_x), 0., 0.]
    torque_msg = messaging.ArrayMotorTorqueMsg().write(payload)
    wheel_effector.rwMotorCmdInMsg.subscribeTo(torque_msg)

    # Both links are required: dynamic body state effect AND periodic
    # effector module UpdateState() for motor-command messages.
    sim.AddModelToTask("bsk_task", body)
    sim.AddModelToTask("bsk_task", wheel_effector)
    rec = wheel_effector.rwSpeedOutMsg.recorder()
    sim.AddModelToTask("bsk_task", rec)
    sim.InitializeSimulation()
    if wheel_effector.numRW != 3 or len(wheel_effector.rwOutMsgs) != 3:
        raise RuntimeError("RW_RUNTIME_ATTACHMENT_FAILED")
    sim.ConfigureStopTime(macros.sec2nano(seconds))
    sim.ExecuteSimulation()

    time_s = np.asarray(rec.times()) * macros.NANO2SEC
    # Basilisk message carries a fixed-width array (up to 36), but exactly
    # three wheels were physically instantiated.
    rates = np.asarray(rec.wheelSpeeds, dtype=float)[:, :3]
    if (len(time_s) < 10 or rates.shape != (len(time_s), 3)
            or not np.isfinite(rates).all()):
        raise RuntimeError("RW_NATIVE_OUTPUT_INVALID")
    delta = float(rates[-1, 0] - rates[0, 0])
    if abs(torque_x) > 1e-12 and delta * torque_x <= 0.01 * abs(torque_x):
        raise RuntimeError("RW_MOTOR_TORQUE_CAUSAL_RESPONSE_MISSING")
    if abs(torque_x) < 1e-12 and abs(delta) > 1e-6:
        raise RuntimeError("RW_ZERO_MOTOR_RESPONSE_INVALID")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["time_s", *[f"adcs.rw.speed_rad_s_{i}" for i in range(3)],
               *[f"adcs.rw.command_torque_nm_{i}" for i in range(3)]]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for t, speed in zip(time_s, rates):
            writer.writerow([float(t), *map(float, speed),
                             float(torque_x), 0., 0.])
    return {"numRW": 3, "rows": len(time_s), "time_s": float(time_s[-1]),
            "wheel0_speed_delta_rad_s": delta,
            "motor_command_nm": float(torque_x),
            "native_module": "Basilisk.simulation.reactionWheelStateEffector",
            "upstream_campaign_executed": False,
            "independent_physical_qualification": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torque-x", type=float, default=0.02)
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args(argv)
    print(json.dumps(simulate(args.output, torque_x=args.torque_x,
                              seconds=args.seconds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
