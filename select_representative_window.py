"""
select_representative_window
=============================

Python port of ``select_representative_window.m``.

Selects the most representative fixed-length window from a power time
series (optionally including an aligned SOC series), by dividing the
series into consecutive non-overlapping windows and picking the window
whose metric vector is closest (lowest relative-RMSE score) to the
median metric vector across all windows.

Notes on the MATLAB -> Python port
-----------------------------------
- MATLAB is 1-indexed and reshapes column-major; here everything is
  0-indexed and we reshape with ``order='F'`` so that column *i* of the
  resulting (samples_per_window, num_windows) array corresponds exactly
  to window *i* in the original MATLAB code.
- ``out.idxBest`` in MATLAB is a 1-based index. The Python output keeps
  both: ``idx_best`` (0-based, Pythonic) and ``idx_best_matlab``
  (1-based, for readers cross-checking against MATLAB).
- The metrics table is returned as a ``pandas.DataFrame``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import rainflow as _rainflow
except ImportError:  # pragma: no cover
    _rainflow = None

# --- internal constants (kept identical to the MATLAB implementation) ---
REST_THRESHOLD_W = 0.01
ZERO_IS_REST = True

DEFAULT_METRICS = ("PosWh", "RT", "Nswap", "Eswap")

SUPPORTED_METRICS = {
    "POSWH", "RT", "NSWAP", "ESWAP", "RAINFLOWDOD",
    "PMAXCHARGE", "PMAXDISCHARGE", "PRMS",
    "SOCRANGE", "SOCMEAN", "SOCMIN", "SOCMAX",
}


@dataclass
class WindowSelectionResult:
    idx_best: int                      # 0-based index into windows
    idx_best_matlab: int                # 1-based index, for MATLAB cross-checks
    best_window_power: np.ndarray
    best_window_soc: Optional[np.ndarray]
    metrics_table: pd.DataFrame
    score_per_window: np.ndarray
    options: dict = field(default_factory=dict)


def select_representative_window(
    power: Sequence[float],
    dt: float = 1.0,
    window_sec: float = 7 * 24 * 3600,
    soc: Optional[Sequence[float]] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> WindowSelectionResult:
    """Select the most representative fixed-length window from ``power``.

    Parameters
    ----------
    power : array-like [W]
        Power time series sampled at constant interval ``dt``.
    dt : float, default 1
        Sampling time [s].
    window_sec : float, default 7*24*3600
        Window length [s].
    soc : array-like, optional
        SOC values aligned sample-by-sample with ``power``. Must have the
        same length as ``power`` if provided.
    metrics : sequence of str
        Metrics used for representativeness evaluation. Supported:
        PosWh, RT, Nswap, Eswap, PmaxCharge, PmaxDischarge, Prms,
        SOCrange, SOCmean, SOCmin, SOCmax, RainflowDOD (requires ``soc``).

    Returns
    -------
    WindowSelectionResult
    """
    power = np.asarray(power, dtype=float).reshape(-1)
    n = power.size

    has_soc = soc is not None
    if has_soc:
        soc = np.asarray(soc, dtype=float).reshape(-1)
        if soc.size != n:
            raise ValueError(
                f"SOC and POWER must have the same length. "
                f"Got len(power)={n} and len(soc)={soc.size}."
            )
    else:
        soc = None

    metrics = [m for m in metrics]
    for m in metrics:
        if m.upper() not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric: {m}")

    samples_per_window = round(window_sec / dt)
    num_full = n // samples_per_window
    if num_full < 1:
        raise ValueError(
            f"Data too short for one window (need at least "
            f"{samples_per_window} samples)."
        )

    use_n = num_full * samples_per_window
    # order='F' reproduces MATLAB's column-major reshape: each column is
    # one window, matching P = reshape(power(1:useN), samplesPerWindow, numFull)
    P = power[:use_n].reshape(samples_per_window, num_full, order="F")
    S = soc[:use_n].reshape(samples_per_window, num_full, order="F") if has_soc else None

    # ---- compute requested metrics ----
    computed = {}
    for name in metrics:
        key = name.upper()
        if key == "POSWH":
            computed["PosWh"] = _metric_pos_wh(P, dt)
        elif key == "RT":
            computed["RT_h"] = _metric_rest_hours(P, REST_THRESHOLD_W, dt)
        elif key == "NSWAP":
            computed["Nswap"] = _metric_sign_changes(P, ZERO_IS_REST)
        elif key == "ESWAP":
            computed["Eswap_Wh"] = _metric_energy_between_swaps(P, dt, ZERO_IS_REST)
        elif key == "RAINFLOWDOD":
            if not has_soc:
                raise ValueError("SOC must be provided for RainflowDOD metric.")
            computed["RainflowDOD"] = _metric_rainflow_dod(S)
        elif key == "PMAXCHARGE":
            computed["PmaxCharge_W"] = _metric_pmax_charge(P)
        elif key == "PMAXDISCHARGE":
            computed["PmaxDischarge_W"] = _metric_pmax_discharge(P)
        elif key == "PRMS":
            computed["Prms_W"] = _metric_prms(P)
        elif key == "SOCRANGE":
            if not has_soc:
                raise ValueError("SOC must be provided for SOCrange metric.")
            computed["SOCrange"] = _metric_soc_range(S)
        elif key == "SOCMEAN":
            if not has_soc:
                raise ValueError("SOC must be provided for SOCmean metric.")
            computed["SOCmean"] = _metric_soc_mean(S)
        elif key == "SOCMIN":
            if not has_soc:
                raise ValueError("SOC must be provided for SOCmin metric.")
            computed["SOCmin"] = _metric_soc_min(S)
        elif key == "SOCMAX":
            if not has_soc:
                raise ValueError("SOC must be provided for SOCmax metric.")
            computed["SOCmax"] = _metric_soc_max(S)

    metric_mat, metric_names = _pack_metrics(computed, metrics)

    # ---- center vector (median across windows) & relative-RMSE score ----
    center_vec = np.nanmedian(metric_mat, axis=0)
    denom = np.abs(center_vec)
    denom[denom < 1e-12] = 1.0  # avoid blow-ups if a typical value is ~0
    rel = (metric_mat - center_vec) / denom
    score = np.sqrt(np.nanmean(rel ** 2, axis=1))

    idx_best = int(np.argmin(score))

    metrics_table = pd.DataFrame(metric_mat, columns=metric_names)
    metrics_table.insert(0, "WindowIdx", np.arange(1, num_full + 1))  # 1-based, matches MATLAB
    metrics_table["Score"] = score

    return WindowSelectionResult(
        idx_best=idx_best,
        idx_best_matlab=idx_best + 1,
        best_window_power=P[:, idx_best].copy(),
        best_window_soc=(S[:, idx_best].copy() if has_soc else None),
        metrics_table=metrics_table,
        score_per_window=score,
        options={
            "dt": dt,
            "windowSec": window_sec,
            "metrics": list(metrics),
            "hasSOC": has_soc,
        },
    )


# ----------------------- metric helpers -----------------------
# All operate on P (or S) shaped (samples_per_window, num_windows) and
# reduce along axis 0 (i.e. "per column" == "per window"), exactly like
# the MATLAB functions that reduce along dim 1.

def _metric_pos_wh(P: np.ndarray, dt: float) -> np.ndarray:
    pos_j = np.sum(np.maximum(P, 0), axis=0) * dt  # J = W*s
    return pos_j / 3600.0


def _metric_rest_hours(P: np.ndarray, thr: float, dt: float) -> np.ndarray:
    rt_s = np.sum(np.abs(P) < thr, axis=0) * dt
    return rt_s / 3600.0


def _metric_sign_changes(P: np.ndarray, zero_is_rest: bool) -> np.ndarray:
    n_windows = P.shape[1]
    out = np.zeros(n_windows)
    for i in range(n_windows):
        s = np.sign(P[:, i])
        if not zero_is_rest:
            s = s.copy()
            for k in range(1, len(s)):
                if s[k] == 0:
                    s[k] = s[k - 1]
        out[i] = np.sum(np.diff(s) != 0)
    return out


def _metric_energy_between_swaps(P: np.ndarray, dt: float, zero_is_rest: bool) -> np.ndarray:
    n_windows = P.shape[1]
    out = np.full(n_windows, np.nan)
    for i in range(n_windows):
        x = P[:, i]
        s = np.sign(x)
        if not zero_is_rest:
            s = s.copy()
            for k in range(1, len(s)):
                if s[k] == 0:
                    s[k] = s[k - 1]
        idx = np.flatnonzero(np.diff(s) != 0)
        edges = np.concatenate(([0], idx + 1, [len(x)]))
        seg_e = np.array([
            np.sum(np.abs(x[edges[j]:edges[j + 1]])) * dt
            for j in range(len(edges) - 1)
        ])
        if seg_e.size:
            out[i] = np.mean(seg_e) / 3600.0
    return out


def _metric_rainflow_dod(S: np.ndarray) -> np.ndarray:
    if _rainflow is None:
        raise ImportError(
            "The 'rainflow' package is required for the RainflowDOD metric. "
            "Install it with: pip install rainflow"
        )
    n_windows = S.shape[1]
    out = np.full(n_windows, np.nan)
    for i in range(n_windows):
        soc_col = S[:, i]
        cycles = list(_rainflow.extract_cycles(soc_col))  # (range, mean, count, i_start, i_end)
        if not cycles:
            continue
        ranges = np.array([c[0] for c in cycles], dtype=float)
        out[i] = np.nanmean(ranges)
    return out


def _metric_pmax_charge(P: np.ndarray) -> np.ndarray:
    return np.abs(np.nanmin(np.minimum(P, 0), axis=0))


def _metric_pmax_discharge(P: np.ndarray) -> np.ndarray:
    return np.nanmax(np.maximum(P, 0), axis=0)


def _metric_prms(P: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(P ** 2, axis=0))


def _metric_soc_range(S: np.ndarray) -> np.ndarray:
    return np.nanmax(S, axis=0) - np.nanmin(S, axis=0)


def _metric_soc_mean(S: np.ndarray) -> np.ndarray:
    return np.nanmean(S, axis=0)


def _metric_soc_min(S: np.ndarray) -> np.ndarray:
    return np.nanmin(S, axis=0)


def _metric_soc_max(S: np.ndarray) -> np.ndarray:
    return np.nanmax(S, axis=0)


def _pack_metrics(computed: dict, requested: Sequence[str]):
    name_map = {
        "POSWH": "PosWh", "RT": "RT_h", "NSWAP": "Nswap", "ESWAP": "Eswap_Wh",
        "RAINFLOWDOD": "RainflowDOD", "PMAXCHARGE": "PmaxCharge_W",
        "PMAXDISCHARGE": "PmaxDischarge_W", "PRMS": "Prms_W",
        "SOCRANGE": "SOCrange", "SOCMEAN": "SOCmean",
        "SOCMIN": "SOCmin", "SOCMAX": "SOCmax",
    }
    cols = []
    names = []
    for r in requested:
        key = r.upper()
        nm = name_map[key]
        cols.append(computed[nm])
        names.append(nm)
    mat = np.column_stack(cols) if cols else np.empty((0, 0))
    return mat, names
