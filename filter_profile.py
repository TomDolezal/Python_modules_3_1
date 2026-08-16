"""
filter_profile
===============

Python port of ``filter_profile.m``.

Removes long resting plateaus from a power time series while preserving
short rests and dynamic behaviour, to shorten the profile for laboratory
testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class FilterResult:
    power_f: np.ndarray
    soc_f: Optional[np.ndarray]
    keep_mask: np.ndarray
    n_removed: int
    n_kept: int
    removed_fraction: float
    filter_rest_threshold: float
    dt: float
    keep_rest_sec: float


def filter_profile(
    power: Sequence[float],
    soc: Optional[Sequence[float]] = None,
    filter_rest_threshold: float = 0.0,
    dt: float = 1.0,
    keep_rest_sec: float = 300.0,
) -> FilterResult:
    """Remove long resting plateaus while preserving dynamics.

    Parameters
    ----------
    power : array-like [W]
    soc : array-like, optional
        Must have the same length as ``power`` if provided.
    filter_rest_threshold : float, default 0
        Rest if abs(power) <= threshold [W].
    dt : float, default 1
        Sampling time [s].
    keep_rest_sec : float, default 300
        For each continuous rest segment, keep only the first
        ``keep_rest_sec`` seconds; remove the rest of that segment.
        Shorter rest segments are kept in full.

    Returns
    -------
    FilterResult
    """
    if filter_rest_threshold < 0:
        raise ValueError("filter_rest_threshold must be >= 0.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if keep_rest_sec < 0:
        raise ValueError("keep_rest_sec must be >= 0.")

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

    thr = filter_rest_threshold
    rest_mask = np.abs(power) <= thr  # True where resting

    keep_rest_samples = max(int(np.ceil(keep_rest_sec / dt)), 0)

    keep_mask = np.ones(n, dtype=bool)

    if rest_mask.any():
        # Find contiguous rest runs via edge detection, mirroring the
        # MATLAB diff([false; restMask; false]) approach.
        padded = np.concatenate(([False], rest_mask, [False]))
        d = np.diff(padded.astype(int))
        starts = np.flatnonzero(d == 1)          # 0-based start indices
        ends = np.flatnonzero(d == -1) - 1        # 0-based inclusive end indices

        for s, e in zip(starts, ends):
            length = e - s + 1
            if length > keep_rest_samples:
                cut_start = s + keep_rest_samples  # first index to remove
                keep_mask[cut_start:e + 1] = False

    power_f = power[keep_mask]
    soc_f = soc[keep_mask] if has_soc else None

    n_removed = int(np.sum(~keep_mask))
    n_kept = int(np.sum(keep_mask))

    return FilterResult(
        power_f=power_f,
        soc_f=soc_f,
        keep_mask=keep_mask,
        n_removed=n_removed,
        n_kept=n_kept,
        removed_fraction=n_removed / n,
        filter_rest_threshold=thr,
        dt=dt,
        keep_rest_sec=keep_rest_sec,
    )
