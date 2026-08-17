"""
generate_sample_data.py
========================

Creates a small synthetic BESS power + SOC dataset for testing the
Python port, loosely modeled on the household-BESS-with-PV scenario
from the poster/paper (charging from solar during the day, discharging
to cover household load in the evening, occasional long overnight rests).

This is NOT real measured data -- just enough structure (daily cycles,
rest periods, sign changes, SOC excursions) to exercise every code path
in select_representative_window / filter_profile / profile_generation.

Run directly to write sample_bess_data.csv next to this script:
    python generate_sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_data(
    days: int = 14,
    dt: float = 60.0,          # sampling interval [s] -> 1-minute data
    capacity_wh: float = 7000.0,  # nominal BESS energy capacity [Wh]
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n = int(days * 24 * 3600 / dt)
    t = np.arange(n) * dt
    hours = (t / 3600.0) % 24

    # --- PV-like charging (negative = charging convention: charge<0, discharge>0) ---
    # Bell-shaped solar profile centered at 13:00, present ~7:00-19:00
    solar = np.clip(np.sin((hours - 7) / 12 * np.pi), 0, None) ** 1.5
    solar *= 2400  # peak ~2.4 kW
    day_factor = 0.55 + 0.5 * rng.random(days)  # cloudy/sunny day variability
    day_factor_full = np.repeat(day_factor, int(24 * 3600 / dt))[:n]
    charge_power = solar * day_factor_full  # W available to charge (positive magnitude)

    # --- household load-like discharging, higher morning/evening peaks ---
    morning = np.exp(-0.5 * ((hours - 7.5) / 1.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 19.5) / 1.5) ** 2)
    base_load = 250 + 350 * (morning + evening)
    noise = rng.normal(0, 60, n)
    load_power = np.clip(base_load + noise, 0, None)

    # Net BESS power: charging (from PV surplus) is negative, discharging
    # (covering load) is positive -- i.e. discharge-positive convention,
    # matching opts.charging_positive = false (the profile_generation default).
    power = load_power - charge_power

    # Inject some clean rest periods (BESS idle / fully charged / no load)
    # a few times per week, each lasting a few hours, to exercise filter_profile.
    rest_len = int(3 * 3600 / dt)  # 3 hours
    for day in range(0, days, 3):
        start = int(day * 24 * 3600 / dt) + int(2 * 3600 / dt)  # 02:00
        end = min(start + rest_len, n)
        power[start:end] = rng.normal(0, 0.005, end - start)  # near-zero noise

    # Occasional short high-power transients (sign swaps), e.g. EV-charger-like spikes
    n_spikes = days * 2
    spike_idx = rng.integers(0, n, n_spikes)
    spike_sign = rng.choice([-1, 1], n_spikes)
    spike_mag = rng.uniform(1500, 4000, n_spikes)
    spike_dur = (rng.uniform(5, 20, n_spikes) * 60 / dt).astype(int)
    for idx, sign, mag, dur in zip(spike_idx, spike_sign, spike_mag, spike_dur):
        end = min(idx + dur, n)
        power[idx:end] += sign * mag

    # --- crude coulomb-counted SOC trace for internal consistency ---
    # discharge-positive convention: SOC decreases when power > 0
    soc = np.zeros(n)
    soc[0] = 60.0  # start at 60%
    capacity_j = capacity_wh * 3600.0
    for k in range(n - 1):
        d_soc = -(power[k] / capacity_j) * 100.0 * dt
        soc[k + 1] = np.clip(soc[k] + d_soc, 2.0, 98.0)

    df = pd.DataFrame({
        "time_s": t,
        "power_W": power,
        "soc_percent": soc,
    })
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    out_path = "sample_bess_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.describe())
