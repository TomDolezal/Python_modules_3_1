"""
test_profile_generation.py
============================

Simple, dependency-light sanity tests for profile_generation.py and its
_pg_helpers.py building blocks. Not a pytest suite -- plain functions
with assertions, runnable directly:

    python test_profile_generation.py

Each test prints PASS/FAIL so you can see at a glance what works.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _pg_helpers import (
    make_clamped_interp1d,
    make_clamped_interp2d,
    current_comp_safe,
    detect_battery_mode,
    apply_default_options,
    apply_default_datasheet,
    make_sim_cell_model,
    get_cell_params_fast,
)
from profile_generation import run_battery_model, tune_battery_model, profile_generation


# ============================================================
# Shared fixtures
# ============================================================

def _flat_cell_model():
    """Constant OCV=3.7V, R0=0.1 ohm, Q=5Ah, eta=0.99 -- easy to hand-check."""
    return {
        "SOC": np.array([0.0, 0.5, 1.0]),
        "TEMP": np.array([0.0, 25.0]),
        "Q": np.array([5.0, 5.0]),
        "OCV0": np.array([3.7, 3.7, 3.7]),
        "OCVrel": np.array([0.0, 0.0, 0.0]),
        "R0": np.full((3, 2), 0.1),
        "eta": np.array([0.99, 0.99]),
    }


def _realistic_cell_model():
    """A slightly more realistic (still synthetic) cell model with actual
    SOC/TEMP-dependent curves, for the tuning-recovery integration test."""
    return {
        "SOC": np.array([0.0, 20.0, 50.0, 80.0, 100.0]) / 100.0,
        "TEMP": np.array([10.0, 25.0, 40.0]),
        "Q": np.array([4.9, 5.0, 4.95]),
        "OCV0": np.array([3.0, 3.4, 3.7, 3.95, 4.2]),
        "OCVrel": np.array([0.002] * 5),
        "R0": np.array([
            [0.09, 0.08, 0.085],
            [0.07, 0.06, 0.065],
            [0.06, 0.05, 0.055],
            [0.065, 0.055, 0.06],
            [0.08, 0.07, 0.075],
        ]),
        "eta": np.array([0.985, 0.99, 0.988]),
    }


# ============================================================
# _pg_helpers tests
# ============================================================

def test_interp1d_clamping():
    x = np.array([0, 25, 50, 75, 100])
    y = np.array([3.0, 3.5, 3.7, 3.9, 4.2])
    f = make_clamped_interp1d(x, y)
    assert np.isclose(f(50), 3.7)
    assert np.isclose(f(12.5), 3.25)
    assert np.isclose(f(-20), 3.0), "should clamp below range, not extrapolate"
    assert np.isclose(f(150), 4.2), "should clamp above range, not extrapolate"
    print("PASS: 1D clamped interpolant")


def test_interp2d_orientation_and_clamping():
    soc = np.array([0, 50, 100])
    temp = np.array([0, 25])
    R0 = np.array([[s * 1.0 + t * 0.1 for t in temp] for s in soc])
    f = make_clamped_interp2d(soc, temp, R0)
    assert np.isclose(f(50, 25), 52.5)
    assert np.isclose(f(25, 12.5), 26.25)
    assert np.isclose(f(-10, -10), 0.0)
    assert np.isclose(f(200, 100), 102.5)
    print("PASS: 2D clamped interpolant orientation + extrapolation")


def test_current_comp_safe():
    I, ok = current_comp_safe(3.7, 0.05, 10.0)
    assert ok and np.isclose(I, 2.8093580054424816)
    I0, ok0 = current_comp_safe(3.7, 0.05, 0.0)
    assert ok0 and np.isclose(I0, 0.0)
    I_bad, ok_bad = current_comp_safe(3.7, 0.05, 1e6)
    assert not ok_bad, "should flag infeasible power instead of raising"
    print("PASS: current_comp_safe quadratic solver")


def test_detect_battery_mode():
    assert detect_battery_mode({"power": np.zeros(3), "soc": np.zeros(3)}) == "power_soc"
    assert detect_battery_mode({"power": np.zeros(3)}) == "power_only"
    assert detect_battery_mode({"power": np.zeros(3), "voltage": np.zeros(3)}) == "power_voltage"
    assert detect_battery_mode({"voltage": np.zeros(3), "current": np.zeros(3)}) == "voltage_current"
    try:
        detect_battery_mode({})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("PASS: detect_battery_mode")


def test_apply_default_options_validation():
    o = apply_default_options({})
    assert o["scale_divisor"] == 100 and o["dt"] == 1
    orig = {"scale_divisor": 5000}
    apply_default_options(orig)
    assert "dt" not in orig, "apply_default_options must not mutate the caller's dict"
    for bad in [{"scale_divisor": -1}, {"dt": 0}, {"eta_coulomb_ef": 1.5}, {"drift_correction": -1}]:
        try:
            apply_default_options(bad)
            raise AssertionError(f"should have raised for {bad}")
        except ValueError:
            pass
    print("PASS: apply_default_options validation + non-mutation")


def test_get_cell_params_fast_and_ocv_quirk():
    cell_model = _flat_cell_model()
    cell_model["OCVrel"] = np.array([0.001, 0.001, 0.001])
    sim_model = make_sim_cell_model(cell_model)
    opts = {"eta_coulomb_ef": 0.5}

    params = get_cell_params_fast(sim_model, 50.0, 25.0, opts)
    assert np.isclose(params["OCV"], 3.7 + 25 * 0.001)
    assert np.isclose(params["eta"], 0.99), "should use cell model eta, not opts fallback"

    # Out-of-range T_degC: OCV formula uses the RAW t_degC (matches MATLAB
    # behaviour exactly -- only the interpolant lookups themselves clamp).
    params2 = get_cell_params_fast(sim_model, 50.0, 100.0, opts)
    assert np.isclose(params2["OCV"], 3.7 + 100 * 0.001), \
        "OCV formula should use raw (unclamped) T_degC, matching the MATLAB source"
    print("PASS: get_cell_params_fast, including the raw-T_degC OCV quirk")


# ============================================================
# run_battery_model tests
# ============================================================

def test_run_battery_model_matches_hand_calc():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})
    opts = apply_default_options({
        "scale_divisor": 1.0, "dt": 1.0, "T_cell_degC": 25, "charging_positive": False,
    })
    profile_data = {"power": np.array([5.0, 5.0, 5.0]), "soc_init": 50.0}

    res = run_battery_model(profile_data, "power_only", sim_model, datasheet, opts)
    assert res.is_valid

    disc = 3.7 ** 2 - 4 * 0.1 * 5
    I_expected = (3.7 - np.sqrt(disc)) / (2 * 0.1)
    V_expected = 3.7 - I_expected * 0.1
    soc1_expected = 50.0 - (I_expected / (5 * 3600)) * 100 * 1.0

    assert np.isclose(res.current[0], I_expected)
    assert np.isclose(res.voltage[0], V_expected)
    assert np.isclose(res.soc[1], soc1_expected)
    assert np.isclose(res.current[-1], res.current[-2]), "last sample should repeat second-to-last"
    print("PASS: run_battery_model matches hand-calculated discharge case")


def test_run_battery_model_charging_and_drift_correction():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})
    opts = apply_default_options({
        "scale_divisor": 1.0, "dt": 1.0, "T_cell_degC": 25,
        "charging_positive": False, "drift_correction": 0.9,
    })
    profile_data = {"power": np.array([-5.0, -5.0, -5.0]), "soc_init": 50.0}
    res = run_battery_model(profile_data, "power_only", sim_model, datasheet, opts)
    assert res.is_valid

    disc = 3.7 ** 2 - 4 * 0.1 * (-5)
    I1, I2 = (3.7 + np.sqrt(disc)) / 0.2, (3.7 - np.sqrt(disc)) / 0.2
    I_raw_expected = min(I1, I2)
    assert I_raw_expected < 0
    I_expected = I_raw_expected * 0.9
    assert np.isclose(res.current[0], I_expected), \
        "drift_correction should apply only to the charging branch"
    print("PASS: charging + drift_correction two-pass sign logic")


def test_run_battery_model_current_limit_violation():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({"max_current_charge_A": 0.5, "max_current_discharge_A": 0.5})
    opts = apply_default_options({"scale_divisor": 1.0, "dt": 1.0, "T_cell_degC": 25, "use_current_limit": True})
    profile_data = {"power": np.array([5.0, 5.0, 5.0]), "soc_init": 50.0}
    res = run_battery_model(profile_data, "power_only", sim_model, datasheet, opts)
    assert not res.is_valid
    assert "limit exceeded" in res.message
    print("PASS: current-limit violation invalidates run with a clear message")


def test_run_battery_model_infeasible_power():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})
    opts = apply_default_options({"scale_divisor": 1.0, "dt": 1.0, "T_cell_degC": 25})
    profile_data = {"power": np.array([1e6, 1e6, 1e6]), "soc_init": 50.0}
    res = run_battery_model(profile_data, "power_only", sim_model, datasheet, opts)
    assert not res.is_valid
    assert "invalid result" in res.message
    print("PASS: infeasible power (negative discriminant) fails cleanly, no crash")


def test_run_battery_model_length_mismatch():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})
    opts = apply_default_options({"scale_divisor": 1.0, "dt": 1.0})
    profile_data = {"power": np.array([1.0, 2.0, 3.0]), "soc": np.array([50.0, 49.0])}
    res = run_battery_model(profile_data, "power_soc", sim_model, datasheet, opts)
    assert not res.is_valid
    print("PASS: power/soc length mismatch in power_soc mode fails cleanly")


# ============================================================
# Integration test: tuning recovers known parameters
# ============================================================

def test_tuning_recovers_known_parameters():
    cell_model = _realistic_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})

    rng = np.random.default_rng(1)
    n = 200
    t = np.arange(n)
    bess_power = 800 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 80, n)

    TRUE_SCALE, TRUE_DRIFT = 200.0, 0.95
    opts_fwd = apply_default_options({
        "scale_divisor": TRUE_SCALE, "dt": 1.0, "T_cell_degC": 25,
        "charging_positive": False, "drift_correction": TRUE_DRIFT,
    })
    fwd_res = run_battery_model({"power": bess_power, "soc_init": 55.0}, "power_only",
                                 sim_model, datasheet, opts_fwd)
    assert fwd_res.is_valid

    tune_profile = {"power": bess_power, "soc": fwd_res.soc}
    opts_tune = {
        "dt": 1.0, "T_cell_degC": 25, "charging_positive": False,
        "tune_scale_divisor": True, "scale_divisor_values": np.arange(150, 251, 10),
        "tune_drift_correction": True, "drift_correction_values": np.arange(0.90, 1.001, 0.01),
    }
    out = profile_generation(cell_model, tune_profile, datasheet, opts_tune)

    assert out.success
    assert np.isclose(out.best_scale_divisor, TRUE_SCALE)
    assert np.isclose(out.best_drift_correction, TRUE_DRIFT)
    assert out.tuning.min_rmse < 1e-6
    print("PASS: tuning grid search recovers known scale_divisor/drift_correction")


def test_power_only_mode_missing_soc_init():
    cell_model = _flat_cell_model()
    sim_model = make_sim_cell_model(cell_model)
    datasheet = apply_default_datasheet({})
    opts = apply_default_options({})
    res = run_battery_model({"power": np.array([1.0, 2.0])}, "power_only", sim_model, datasheet, opts)
    assert not res.is_valid
    assert "soc_init" in res.message
    print("PASS: power_only mode without soc_init fails cleanly")


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_interp1d_clamping,
        test_interp2d_orientation_and_clamping,
        test_current_comp_safe,
        test_detect_battery_mode,
        test_apply_default_options_validation,
        test_get_cell_params_fast_and_ocv_quirk,
        test_run_battery_model_matches_hand_calc,
        test_run_battery_model_charging_and_drift_correction,
        test_run_battery_model_current_limit_violation,
        test_run_battery_model_infeasible_power,
        test_run_battery_model_length_mismatch,
        test_power_only_mode_missing_soc_init,
        test_tuning_recovers_known_parameters,
    ]

    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"FAIL: {t.__name__} -> {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    run_all()
