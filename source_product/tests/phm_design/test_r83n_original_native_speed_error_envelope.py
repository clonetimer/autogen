"""R83n tests use SYNTHETIC contracts, never count as native experiments."""
from __future__ import annotations

import pytest
from phm_design.industrial import r83n_original_native_speed_error_envelope as env
from test_r83m_original_native_guidance_threshold_gate import fixture_source


def test_bounded_error_oracle_is_not_deployable_detection(tmp_path,monkeypatch):
    # This is a hand-written 54-case PHYSICS CONTRACT fixture only.
    path,_=fixture_source(tmp_path,monkeypatch)
    report=env.evaluate_original_native_encoder_error_envelope(path)
    assert report["real_native_original_runs"]==54
    assert report["physical_wheel_initial_groups"]==9
    assert report["counterfactual_pair_cases"]==27
    assert report["native_sampling_interval_s"]==.2
    m=report["error_envelope_physics_oracle_matrix"]
    assert m["zero"]["0.0000"]["paired_true_speed_contrasts_above_error_difference_bound"]==0
    assert m["pulse05"]["0.0000"]["paired_true_speed_contrasts_above_error_difference_bound"]==0
    assert m["pulse09"]["0.0000"]["paired_true_speed_contrasts_above_error_difference_bound"]==9
    assert m["pulse09"]["0.0010"]["paired_true_speed_contrasts_above_error_difference_bound"]==9
    assert m["pulse09"]["0.0100"]["paired_true_speed_contrasts_above_error_difference_bound"]==0
    assert report["actual_basilisk_sensor_emulator_used"] is False
    assert report["actual_hardware_encoder_precision_known"] is False
    assert report["original_counterfactual_healthy_twin_available_on_orbit"] is False
    assert report["source_fault_target_or_applied_torque_used_as_model_input"] is False
    assert report["false_alarm_rate_or_detection_delay_measured"] is False
    assert report["diagnostic_model_trained"] is False
    assert report["stable_release_eligible"] is False


def test_robust_oracle_counts_never_increase_with_more_uncertainty(tmp_path,monkeypatch):
    path,_=fixture_source(tmp_path,monkeypatch)
    m=env.evaluate_original_native_encoder_error_envelope(path)[
        "error_envelope_physics_oracle_matrix"]
    for ref in env.REFERENCES:
        counts=[m[ref][f"{epsilon:.4f}"][
            "paired_true_speed_contrasts_above_error_difference_bound"]
                for epsilon in env.SINGLE_SAMPLE_ABS_ERROR_LEVELS_RAD_S]
        assert counts==sorted(counts,reverse=True)
        assert all(0<=x<=9 for x in counts)


def test_never_export_oracle_counterfactual_rows_as_deployable_sensor_samples(tmp_path,monkeypatch):
    path,_=fixture_source(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="PROHIBITED_EXPORT"):
        env.evaluate_original_native_encoder_error_envelope(
            path,write_source_samples=True)


def test_refuse_tampered_original_r83m_source_bytes(tmp_path,monkeypatch):
    path,_=fixture_source(tmp_path,monkeypatch)
    path.write_bytes(path.read_bytes()+b" MALFORMED_ORIGINAL")
    with pytest.raises(ValueError,match="SHA256_MISMATCH"):
        env.evaluate_original_native_encoder_error_envelope(path)
