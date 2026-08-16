"""
_pg_helpers.py
===============

Foundational helpers for profile_generation.py:
  - clamped 1D / 2D interpolants that replicate MATLAB's
    griddedInterpolant(..., 'linear', 'nearest') behaviour
  - the quadratic cell-current solve (currentCompSafe equivalent)

Kept in a separate module so we can test them in isolation before
wiring them into the full recursive simulation loop.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator


def make_clamped_interp1d(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """1D linear interpolant with MATLAB-style 'nearest' extrapolation.

    Equivalent to griddedInterpolant(x, y, 'linear', 'nearest'): linear
    inside the grid, clamped to the boundary value outside it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_min, x_max = x.min(), x.max()
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[np.argmin(x)], y[np.argmax(x)]))

    def interpolant(query):
        q = np.clip(query, x_min, x_max)
        return f(q)

    return interpolant


def make_clamped_interp2d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Callable[[float, float], float]:
    """2D linear interpolant with MATLAB-style 'nearest' extrapolation.

    Mirrors griddedInterpolant(ndgrid(x, y), z, 'linear', 'nearest').
    ``z`` must be shaped (len(x), len(y)), i.e. z[i, j] corresponds to
    x[i], y[j] -- the same convention MATLAB's ndgrid(x, y) produces.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if z.shape != (x.size, y.size):
        raise ValueError(
            f"z must have shape ({x.size}, {y.size}) to match (x, y) grid, "
            f"got {z.shape}."
        )

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    rgi = RegularGridInterpolator((x, y), z, method="linear",
                                   bounds_error=False, fill_value=None)

    def interpolant(qx, qy):
        qx_c = np.clip(qx, x_min, x_max)
        qy_c = np.clip(qy, y_min, y_max)
        pt = np.array([qx_c, qy_c]).reshape(1, 2) if np.isscalar(qx) or np.ndim(qx) == 0 \
            else np.column_stack([qx_c, qy_c])
        result = rgi(pt)
        return float(result[0]) if result.size == 1 else result

    return interpolant


def detect_battery_mode(profile_data: dict) -> str:
    """Replicates detectBatteryMode: infers the simulation mode from which
    fields are present (and non-empty) in profile_data.
    """
    def present(key):
        return key in profile_data and profile_data[key] is not None and \
            (not hasattr(profile_data[key], "__len__") or len(profile_data[key]) > 0)

    has_power = present("power")
    has_soc = present("soc")
    has_voltage = present("voltage")
    has_current = present("current")

    if has_power and has_soc:
        return "power_soc"
    elif has_power and not has_voltage and not has_current:
        return "power_only"
    elif has_power and has_voltage:
        return "power_voltage"
    elif has_voltage and has_current:
        return "voltage_current"
    else:
        raise ValueError("Unsupported input combination.")


def apply_default_options(opts: dict) -> dict:
    """Replicates applyDefaultOptions: fills missing fields with defaults
    and validates them. Returns a new dict (does not mutate the input).
    """
    opts = dict(opts)  # shallow copy, don't mutate caller's dict

    opts.setdefault("scale_divisor", 100)
    opts.setdefault("dt", 1)
    opts.setdefault("eta_coulomb_ef", 0.99)
    opts.setdefault("T_cell_degC", 25)
    opts.setdefault("charging_positive", False)
    opts.setdefault("drift_correction", 1.0)
    opts.setdefault("use_current_limit", False)

    opts.setdefault("tune_scale_divisor", True)
    opts.setdefault("scale_divisor_values", np.round(np.logspace(1, 4, 50)))

    opts.setdefault("tune_drift_correction", False)
    opts.setdefault("drift_correction_values", np.arange(0.97, 1.03 + 1e-9, 0.005))

    if opts["scale_divisor"] <= 0:
        raise ValueError("opts['scale_divisor'] must be positive.")
    if opts["dt"] <= 0:
        raise ValueError("opts['dt'] must be positive.")
    if not (0 < opts["eta_coulomb_ef"] <= 1):
        raise ValueError("opts['eta_coulomb_ef'] must be in the interval (0, 1].")
    if opts["drift_correction"] <= 0:
        raise ValueError("opts['drift_correction'] must be positive.")
    if np.any(np.asarray(opts["scale_divisor_values"]) <= 0):
        raise ValueError("All opts['scale_divisor_values'] must be positive.")
    if np.any(np.asarray(opts["drift_correction_values"]) <= 0):
        raise ValueError("All opts['drift_correction_values'] must be positive.")

    return opts


def apply_default_datasheet(datasheet: dict) -> dict:
    """Replicates applyDefaultDatasheet: missing current limits default to
    infinity (i.e. unconstrained).
    """
    datasheet = dict(datasheet)
    datasheet.setdefault("max_current_charge_A", np.inf)
    datasheet.setdefault("max_current_discharge_A", np.inf)
    return datasheet


def make_sim_cell_model(cell_model: dict) -> dict:
    """Replicates makeSimCellModel: builds the fast clamped interpolants
    for a cell model once, so they can be reused across every simulation
    time step (and across every tuning-grid combination).

    cell_model expects (arrays):
        SOC     - SOC breakpoints, normalized 0..1
        TEMP    - temperature breakpoints [degC]
        Q       - capacity vs TEMP [Ah]
        OCV0    - base OCV vs SOC [V]
        OCVrel  - temperature-dependent OCV correction vs SOC [V/degC]
        R0      - internal resistance map, shape (len(SOC), len(TEMP)) [ohm]
        eta     - optional, coulombic efficiency vs TEMP [-]

    Returns a dict of interpolant callables plus the raw grids.
    """
    soc_percent = np.asarray(cell_model["SOC"], dtype=float) * 100.0
    temp_degC = np.asarray(cell_model["TEMP"], dtype=float)

    ocv0_interp = make_clamped_interp1d(soc_percent, np.asarray(cell_model["OCV0"], dtype=float))
    ocvrel_interp = make_clamped_interp1d(soc_percent, np.asarray(cell_model["OCVrel"], dtype=float))
    q_interp = make_clamped_interp1d(temp_degC, np.asarray(cell_model["Q"], dtype=float))

    r0_map = np.asarray(cell_model["R0"], dtype=float)
    expected_shape = (soc_percent.size, temp_degC.size)
    if r0_map.shape != expected_shape:
        if r0_map.shape == expected_shape[::-1]:
            r0_map = r0_map.T
        else:
            raise ValueError(
                f"R0 dimensions {r0_map.shape} do not match SOC/TEMP grid "
                f"{expected_shape}."
            )
    r0_interp = make_clamped_interp2d(soc_percent, temp_degC, r0_map)

    eta_interp = None
    if cell_model.get("eta") is not None and len(cell_model["eta"]) > 0:
        eta_interp = make_clamped_interp1d(temp_degC, np.asarray(cell_model["eta"], dtype=float))

    return {
        "soc_grid_percent": soc_percent,
        "temp_grid_degC": temp_degC,
        "OCV0_interpolant": ocv0_interp,
        "OCVrel_interpolant": ocvrel_interp,
        "Q_interpolant": q_interp,
        "R0_interpolant": r0_interp,
        "eta_interpolant": eta_interp,
    }


def get_cell_params_fast(sim_model: dict, soc_percent: float, t_degC: float, opts: dict) -> dict:
    """Replicates getCellParamsFast: evaluates OCV, R0, nominal capacity
    [A*s], and coulombic efficiency at a given SOC/temperature.
    """
    ocv0_now = float(sim_model["OCV0_interpolant"](soc_percent))
    ocvrel_now = float(sim_model["OCVrel_interpolant"](soc_percent))
    q_ah = float(sim_model["Q_interpolant"](t_degC))
    r0_now = float(sim_model["R0_interpolant"](soc_percent, t_degC))

    if sim_model["eta_interpolant"] is not None:
        eta = float(sim_model["eta_interpolant"](t_degC))
    else:
        eta = opts["eta_coulomb_ef"]

    return {
        "capacity_nom_As": q_ah * 3600.0,
        "OCV": ocv0_now + t_degC * ocvrel_now,
        "R0": r0_now,
        "eta": eta,
    }


def current_comp_safe(ocv: float, r0: float, power: float) -> Tuple[float, bool]:
    """Solve R0*I^2 - OCV*I + P = 0 for the cell current, keeping the
    smaller of the two roots (matches MATLAB currentCompSafe).

    Returns (current, ok). ok=False means the discriminant was negative
    (no real solution for the requested power at this OCV/R0).
    """
    a = r0
    b = -ocv
    c = power

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return float("nan"), False

    sqrt_disc = np.sqrt(discriminant)
    i1 = (-b + sqrt_disc) / (2 * a)
    i2 = (-b - sqrt_disc) / (2 * a)

    return float(min(i1, i2)), True
