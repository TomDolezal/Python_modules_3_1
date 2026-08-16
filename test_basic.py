"""
test_basic.py
==============

Simple, dependency-light sanity tests for select_representative_window
and filter_profile. Not a pytest suite -- just plain functions with
assertions, runnable directly:

    python test_basic.py

Each test prints PASS/FAIL so you can see at a glance what works.
"""

import sys
import os
import numpy as np

# Make sure the package modules are importable regardless of where this
# script is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from select_representative_window import select_representative_window
from filter_profile import filter_profile


def test_select_representative_window_basic():
    """A simple synthetic signal where we know which window should win."""
    dt = 1.0
    window_sec = 10.0  # 10-sample windows
    n_windows = 5

    # Build 5 windows of 10 samples each. Windows 0,1,3,4 are "similar"
    # (small oscillation around 0), window 2 is a clear outlier (huge swing).
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (10, n_windows))
    base[:, 2] += 100  # make window index 2 the outlier

    power = base.flatten(order="F")  # column-major flatten, matches reshape(order='F')

    result = select_representative_window(power, dt=dt, window_sec=window_sec,
                                           metrics=["Prms"])
    assert result.idx_best != 2, "Outlier window should NOT be selected as representative"
    assert len(result.best_window_power) == 10
    assert result.metrics_table.shape[0] == n_windows
    print("PASS: select_representative_window picks a non-outlier window")


def test_select_representative_window_soc_required():
    """Requesting an SOC-based metric without providing SOC should raise."""
    power = np.zeros(20)
    try:
        select_representative_window(power, dt=1, window_sec=10, metrics=["SOCrange"])
    except ValueError as e:
        print(f"PASS: correctly raised ValueError ({e})")
        return
    raise AssertionError("Expected ValueError when SOC metric requested without soc data")


def test_select_representative_window_length_mismatch():
    power = np.zeros(20)
    soc = np.zeros(15)  # wrong length on purpose
    try:
        select_representative_window(power, dt=1, window_sec=10, soc=soc)
    except ValueError as e:
        print(f"PASS: correctly raised ValueError on length mismatch ({e})")
        return
    raise AssertionError("Expected ValueError on power/soc length mismatch")


def test_filter_profile_removes_long_rest():
    """A single long rest segment longer than keepRestSec should be trimmed."""
    dt = 1.0
    # 5 s active, 100 s rest (abs(power) <= 0), 5 s active
    power = np.concatenate([np.ones(5) * 50, np.zeros(100), np.ones(5) * 50])
    result = filter_profile(power, filter_rest_threshold=0.0, dt=dt, keep_rest_sec=10)

    # Expect: 5 active + 10 kept rest + 5 active = 20 samples kept
    assert result.n_kept == 20, f"Expected 20 kept samples, got {result.n_kept}"
    assert result.n_removed == 90, f"Expected 90 removed samples, got {result.n_removed}"
    assert len(result.power_f) == 20
    print("PASS: filter_profile trims a long rest segment to keepRestSec")


def test_filter_profile_keeps_short_rest():
    """A rest segment shorter than keepRestSec should be kept in full."""
    dt = 1.0
    power = np.concatenate([np.ones(5) * 50, np.zeros(3), np.ones(5) * 50])
    result = filter_profile(power, filter_rest_threshold=0.0, dt=dt, keep_rest_sec=10)

    assert result.n_removed == 0, "Short rest segment should not be trimmed"
    assert result.n_kept == len(power)
    print("PASS: filter_profile keeps rest segments shorter than keepRestSec in full")


def test_filter_profile_soc_alignment():
    """soc_f should stay aligned with power_f after filtering."""
    dt = 1.0
    power = np.concatenate([np.ones(5) * 50, np.zeros(100), np.ones(5) * 50])
    soc = np.arange(len(power), dtype=float)  # unique values so we can check alignment

    result = filter_profile(power, soc=soc, filter_rest_threshold=0.0, dt=dt, keep_rest_sec=10)

    assert result.soc_f is not None
    assert len(result.soc_f) == len(result.power_f)
    # The kept SOC values should exactly match soc[keep_mask]
    assert np.array_equal(result.soc_f, soc[result.keep_mask])
    print("PASS: filter_profile keeps soc_f aligned with power_f")


def test_filter_profile_length_mismatch():
    power = np.zeros(20)
    soc = np.zeros(15)
    try:
        filter_profile(power, soc=soc)
    except ValueError as e:
        print(f"PASS: correctly raised ValueError on length mismatch ({e})")
        return
    raise AssertionError("Expected ValueError on power/soc length mismatch")


def run_all():
    tests = [
        test_select_representative_window_basic,
        test_select_representative_window_soc_required,
        test_select_representative_window_length_mismatch,
        test_filter_profile_removes_long_rest,
        test_filter_profile_keeps_short_rest,
        test_filter_profile_soc_alignment,
        test_filter_profile_length_mismatch,
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
