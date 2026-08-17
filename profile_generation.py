"""
profile_generation
====================

Python port of ``profile_generation.m``.

Scales a BESS-level power profile down to single-cell level and
simulates the electrical response of one cell (simple OCV + R0
equivalent-circuit model), to check feasibility against cell limits
before running an actual laboratory test.

Supported modes (auto-detected from profile_data, same as MATLAB):
  1) power + soc  -> validation / tuning of scale_divisor & drift_correction
  2) power only   -> forward simulation from a given initial SOC

See _pg_helpers.py for the interpolation, quadratic-solve, and
defaulting building blocks this module is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from _pg_helpers import (
    detect_battery_mode,
    apply_default_options,
    apply_default_datasheet,
    make_sim_cell_model,
    get_cell_params_fast,
    current_comp_safe,
)


# ============================================================
# RESULT CONTAINERS
# ============================================================

@dataclass
class RunResult:
    mode: str
    is_valid: bool = True
    message: str = ""
    rmse_soc: float = np.nan
    soc: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    voltage: Optional[np.ndarray] = None
    power_cell: Optional[np.ndarray] = None


@dataclass
class TuningResult:
    rmse_values: np.ndarray            # shape (len(scale_values), len(drift_values))
    results_grid: list                 # same shape, holds RunResult or None
    scale_values: np.ndarray
    drift_values: np.ndarray
    best_scale_divisor: float
    best_drift_correction: float
    best_results: Optional[RunResult]
    min_rmse: float


@dataclass
class ProfileGenerationOutput:
    mode: str
    success: bool = True
    message: str = ""
    tuning: Optional[TuningResult] = None
    results: Optional[RunResult] = None
    best_results: Optional[RunResult] = None
    best_scale_divisor: Optional[float] = None
    best_drift_correction: Optional[float] = None


# ============================================================
# CORE SIMULATION (single scale_divisor / drift_correction pair)
# ============================================================

def run_battery_model(profile_data: dict, mode: str, sim_model: dict,
                       datasheet: dict, opts: dict) -> RunResult:
    """Runs the step-by-step OCV+R0 simulation for one fixed
    (scale_divisor, drift_correction) combination. Mirrors
    runBatteryModel in the MATLAB code, including its 1-based step
    numbering in error messages (for easy cross-referencing).
    """
    result = RunResult(mode=mode)

    scale_divisor = opts["scale_divisor"]
    dt = opts["dt"]
    T_cell_degC = opts["T_cell_degC"]
    charging_positive = opts["charging_positive"]
    drift_correction = opts["drift_correction"]
    use_current_limit = opts["use_current_limit"]

    if use_current_limit:
        max_current_charge_A = datasheet["max_current_charge_A"]
        max_current_discharge_A = datasheet["max_current_discharge_A"]
    else:
        max_current_charge_A = np.inf
        max_current_discharge_A = np.inf

    if mode == "power_soc":
        power_bess = np.asarray(profile_data["power"], dtype=float).reshape(-1)
        soc_ref = np.asarray(profile_data["soc"], dtype=float).reshape(-1)
        if power_bess.size != soc_ref.size:
            result.is_valid = False
            result.message = "power and soc must have the same length in power_soc mode."
            return result
        soc0 = soc_ref[0]

    elif mode == "power_only":
        power_bess = np.asarray(profile_data["power"], dtype=float).reshape(-1)
        if profile_data.get("soc_init") is None:
            result.is_valid = False
            result.message = "power_only mode requires profile_data['soc_init']."
            return result
        soc_ref = None
        soc0 = profile_data["soc_init"]

    else:
        result.is_valid = False
        result.message = f'Mode "{mode}" not implemented yet.'
        return result

    n = power_bess.size
    power_cell = power_bess / scale_divisor

    soc = np.zeros(n)
    current = np.zeros(n)
    voltage = np.zeros(n)
    soc[0] = soc0

    for k in range(n - 1):
        step_1based = k + 1  # for messages, matches MATLAB's loop index
        soc_now = soc[k]

        params = get_cell_params_fast(sim_model, soc_now, T_cell_degC, opts)
        OCV_now = params["OCV"]
        R0_now = params["R0"]
        capacity_nom_As = params["capacity_nom_As"]
        eta_coulomb = params["eta"]

        if not (np.isfinite(OCV_now) and np.isfinite(R0_now) and np.isfinite(capacity_nom_As)) \
                or R0_now <= 0 or capacity_nom_As <= 0:
            result.is_valid = False
            result.message = f"Step {step_1based}: invalid interpolated parameters."
            return result

        power_now = power_cell[k]

        # Raw current from the quadratic OCV/R0 solve (before drift correction)
        I_raw, ok = current_comp_safe(OCV_now, R0_now, power_now)
        if not ok or not np.isfinite(I_raw):
            result.is_valid = False
            result.message = (
                f"Step {step_1based}: complex/invalid result "
                f"(OCV={OCV_now:.3f}, R0={R0_now:.6f}, P={power_now:.3f})"
            )
            return result

        # Determine charge/discharge from the RAW current
        if charging_positive:
            is_charge = I_raw > 0
        else:
            is_charge = I_raw < 0

        # Apply drift correction to charging current only
        I = I_raw
        if is_charge:
            I = I_raw * drift_correction

        # Re-evaluate charge/discharge AFTER correction (two-pass, matches MATLAB)
        if charging_positive:
            is_charge = I > 0
            is_discharge = I < 0
        else:
            is_charge = I < 0
            is_discharge = I > 0

        if use_current_limit:
            if is_charge and abs(I) > max_current_charge_A:
                result.is_valid = False
                result.message = (
                    f"Step {step_1based}: charging current limit exceeded "
                    f"(I={I:.3f} A, limit={max_current_charge_A:.3f} A)"
                )
                return result
            if is_discharge and abs(I) > max_current_discharge_A:
                result.is_valid = False
                result.message = (
                    f"Step {step_1based}: discharging current limit exceeded "
                    f"(I={I:.3f} A, limit={max_current_discharge_A:.3f} A)"
                )
                return result

        V = OCV_now - I * R0_now

        current[k] = I
        voltage[k] = V

        # SOC update via coulomb counting; coulombic efficiency applied
        # during charging only, matching the MATLAB sign-convention logic.
        if charging_positive:
            if I > 0:
                soc[k + 1] = soc[k] + eta_coulomb * (I / capacity_nom_As) * 100 * dt
            else:
                soc[k + 1] = soc[k] + (I / capacity_nom_As) * 100 * dt
        else:
            if I < 0:
                soc[k + 1] = soc[k] - eta_coulomb * (I / capacity_nom_As) * 100 * dt
            else:
                soc[k + 1] = soc[k] - (I / capacity_nom_As) * 100 * dt

        if not np.isfinite(soc[k + 1]):
            result.is_valid = False
            result.message = f"Step {k + 2}: SOC became invalid."
            return result

    if n > 1:
        current[-1] = current[-2]
        voltage[-1] = voltage[-2]

    result.soc = soc
    result.current = current
    result.voltage = voltage
    result.power_cell = power_cell

    if mode == "power_soc":
        if np.all(np.isfinite(soc)) and np.all(np.isfinite(soc_ref)):
            result.rmse_soc = float(np.sqrt(np.mean((soc - soc_ref) ** 2)))
        else:
            result.is_valid = False
            result.message = "RMSE could not be computed due to invalid SOC values."

    return result


# ============================================================
# TUNING WRAPPER (grid search over scale_divisor x drift_correction)
# ============================================================

def tune_battery_model(profile_data: dict, sim_model: dict, datasheet: dict, opts: dict,
                        scale_values: Sequence[float], drift_values: Sequence[float],
                        verbose: bool = False) -> TuningResult:
    """Grid search over scale_divisor x drift_correction, selecting the
    combination that minimizes SOC RMSE. Mirrors tuneBatteryModel.
    """
    scale_values = np.asarray(scale_values, dtype=float).reshape(-1)
    drift_values = np.asarray(drift_values, dtype=float).reshape(-1)

    rmse_values = np.full((scale_values.size, drift_values.size), np.inf)
    results_grid = [[None] * drift_values.size for _ in range(scale_values.size)]

    for i, sd in enumerate(scale_values):
        for j, dc in enumerate(drift_values):
            opts_local = dict(opts)
            opts_local["scale_divisor"] = sd
            opts_local["drift_correction"] = dc

            res = run_battery_model(profile_data, "power_soc", sim_model, datasheet, opts_local)

            if res.is_valid and np.all(np.isfinite(res.soc)):
                rmse_values[i, j] = res.rmse_soc
                results_grid[i][j] = res
                if verbose:
                    print(f"scale_divisor={sd:.6g}, drift_correction={dc:.5f} "
                          f"-> RMSE = {res.rmse_soc:.3f} %")
            elif verbose:
                print(f"scale_divisor={sd:.6g}, drift_correction={dc:.5f} "
                      f"-> invalid ({res.message})")

    if np.any(np.isfinite(rmse_values)):
        flat_idx = int(np.nanargmin(rmse_values))
        best_i, best_j = np.unravel_index(flat_idx, rmse_values.shape)
        best_results = results_grid[best_i][best_j]
        best_scale_divisor = float(scale_values[best_i])
        best_drift_correction = float(drift_values[best_j])
        min_rmse = float(rmse_values[best_i, best_j])
    else:
        best_results = None
        best_scale_divisor = np.nan
        best_drift_correction = np.nan
        min_rmse = np.inf

    return TuningResult(
        rmse_values=rmse_values,
        results_grid=results_grid,
        scale_values=scale_values,
        drift_values=drift_values,
        best_scale_divisor=best_scale_divisor,
        best_drift_correction=best_drift_correction,
        best_results=best_results,
        min_rmse=min_rmse,
    )


# ============================================================
# TOP-LEVEL ENTRY POINT
# ============================================================

def profile_generation(cell_model: dict, profile_data: dict, datasheet: dict,
                        opts: dict, verbose: bool = False) -> ProfileGenerationOutput:
    """Top-level entry point, mirrors the MATLAB profile_generation function.

    Parameters
    ----------
    cell_model : dict with SOC, TEMP, Q, OCV0, OCVrel, R0, and optionally eta
    profile_data : dict with power, and either soc (power_soc mode) or
        soc_init (power_only mode)
    datasheet : dict with max_current_charge_A / max_current_discharge_A
    opts : dict of simulation/tuning settings (see _pg_helpers.apply_default_options)
    verbose : if True, prints progress during the tuning grid search
    """
    mode = detect_battery_mode(profile_data)
    opts = apply_default_options(opts)
    datasheet = apply_default_datasheet(datasheet)
    sim_model = make_sim_cell_model(cell_model)

    output = ProfileGenerationOutput(mode=mode)

    if mode == "power_soc":
        scale_values = opts["scale_divisor_values"] if opts["tune_scale_divisor"] \
            else [opts["scale_divisor"]]
        drift_values = opts["drift_correction_values"] if opts["tune_drift_correction"] \
            else [opts["drift_correction"]]

        tuning = tune_battery_model(profile_data, sim_model, datasheet, opts,
                                     scale_values, drift_values, verbose=verbose)

        output.tuning = tuning
        output.best_results = tuning.best_results
        output.best_scale_divisor = tuning.best_scale_divisor
        output.best_drift_correction = tuning.best_drift_correction

        if tuning.best_results is None or not tuning.best_results.is_valid:
            output.success = False
            output.message = "No valid tuning result found."

    elif mode == "power_only":
        results = run_battery_model(profile_data, mode, sim_model, datasheet, opts)

        output.results = results
        output.best_results = results
        output.best_scale_divisor = opts["scale_divisor"]
        output.best_drift_correction = opts["drift_correction"]

        if not results.is_valid:
            output.success = False
            output.message = results.message

    else:
        output.success = False
        output.message = f'Mode "{mode}" not implemented yet.'

    return output
