"""Weak labels of surface water from cloud-free Sentinel-2 pixels (MNDWI).

Idea: optical water indices are reliable when the sky is clear, but the Mekong is cloudy
much of the year. We use clear-sky Sentinel-2 to label Sentinel-1 dates acquired at almost
the same time, then train a SAR model that also works under clouds.

Known label noise (see README, "Known limitations"):
  * MNDWI misses water under a dense rice canopy (late season) -> false "dry" labels.
  * Mixed pixels at field borders.
"""
import numpy as np
import pandas as pd
import xarray as xr

# SCL classes kept as "clear": 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
CLEAR_SCL = [4, 5, 6, 7]
WATER, DRY, UNLABELED = 1, 0, -1


def mndwi(s2: xr.Dataset) -> xr.DataArray:
    m = (s2["green"] - s2["swir1"]) / (s2["green"] + s2["swir1"])
    return m.where(s2["scl"].isin(CLEAR_SCL))


def water_labels(s2: xr.Dataset, water_thr: float = 0.0, dry_thr: float = -0.2) -> xr.DataArray:
    """1 = water, 0 = dry, -1 = cloudy or ambiguous."""
    m = mndwi(s2)
    lab = xr.full_like(m, UNLABELED, dtype="int8")
    lab = lab.where(~(m > water_thr), WATER)
    lab = lab.where(~(m < dry_thr), DRY)
    return lab.rename("label")


def match_dates(s1_times, s2_times, max_days: float = 1.0) -> dict:
    """Map each Sentinel-1 date to the closest Sentinel-2 date within max_days."""
    s1_times, s2_times = pd.DatetimeIndex(s1_times), pd.DatetimeIndex(s2_times)
    pairs = {}
    for t in s1_times:
        dt = np.abs((s2_times - t) / pd.Timedelta("1D"))
        i = int(np.argmin(dt))
        if dt[i] <= max_days:
            pairs[t] = s2_times[i]
    return pairs


def labels_on_s1_dates(s1: xr.Dataset, labels: xr.DataArray, max_days: float = 1.0) -> xr.DataArray:
    """Label cube on the Sentinel-1 time axis (-1 where no matching clear S2 pixel)."""
    pairs = match_dates(s1.time.values, labels.time.values, max_days)
    out = xr.full_like(s1["vh_db"], UNLABELED, dtype="int8").rename("label")
    for t1, t2 in pairs.items():
        out.loc[dict(time=t1)] = labels.sel(time=t2).values
    return out
