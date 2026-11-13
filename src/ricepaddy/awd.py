"""From per-date water status to field-level AWD indicators."""
import numpy as np
import xarray as xr


def drying_events(water: xr.DataArray, min_dry_dates: int = 1) -> xr.DataArray:
    """Count flooded -> dry transitions after the first flooding (one per drying episode).
    `water` is a (time, y, x) array of 0/1 (NaN = unknown, carried forward)."""
    w = water.ffill("time")
    prev = w.shift(time=1)
    starts = (prev == 1) & (w == 0)
    if min_dry_dates > 1:
        # require the dry state to last at least min_dry_dates acquisitions
        stay = w.rolling(time=min_dry_dates).sum().shift(time=-(min_dry_dates - 1)) == 0
        starts &= stay
    return starts.sum("time").rename("drying_events")


def flooded_fraction(water: xr.DataArray) -> xr.DataArray:
    return water.mean("time").rename("flooded_fraction")


def intermediate_dry_spells(water: xr.DataArray, start=None, end=None, min_dry_dates: int = 1) -> xr.DataArray:
    """Count dry spells that are *followed by re-flooding* within the crop window.

    Within one crop, continuous flooding looks like  0 0 1 1 1 1 1 1 0 0  (flood at sowing,
    final drainage before harvest) and has no intermediate dry spell. AWD looks like
    0 0 1 1 0 1 1 0 1 1 0  : the field dries and is flooded again, several times.
    So the AWD signal is the number of 1 -> 0 -> 1 patterns, not the number of 1 -> 0.
    Restrict to one crop with `start`/`end` so the re-flooding for the next crop is not
    counted. NaN dates are skipped (carried forward). `min_dry_dates` = 2 ignores dry spells
    seen on a single acquisition, which are often classifier errors.
    """
    w = water.sel(time=slice(start, end)).ffill("time")
    v = w.transpose("time", ...).values
    out = np.zeros(v.shape[1:], dtype="float32")
    seen_flood = np.zeros(v.shape[1:], dtype=bool)
    dry_run = np.zeros(v.shape[1:], dtype=int)    # length of the current dry spell after a flood
    for t in range(v.shape[0]):
        x = v[t]
        wet, dry = x == 1, x == 0
        out += wet & (dry_run >= min_dry_dates)     # re-flooding closes a long-enough dry spell
        dry_run = np.where(wet, 0, np.where(seen_flood & dry, dry_run + 1, dry_run))
        seen_flood |= wet
    valid = np.isfinite(v).any(axis=0)
    out[~valid] = np.nan
    return xr.DataArray(out, coords={k: w.coords[k] for k in w.dims if k != "time"},
                        dims=[d for d in w.dims if d != "time"], name="intermediate_dry_spells")


PRACTICES = {0: "continuous flooding", 1: "single mid-season drainage", 2: "AWD-like (2+ re-floodings)"}


def practice_class(spells: xr.DataArray) -> xr.DataArray:
    """0 = continuous flooding, 1 = one mid-season drainage, 2 = AWD-like (2 or more)."""
    return spells.clip(max=2).rename("practice")


def majority_filter(water: xr.DataArray, size: int = 3) -> xr.DataArray:
    """Spatial majority vote of the 0/1 water map on each date (size x size window).
    Neighbouring pixels of a field share one water state, so isolated flips are mostly
    classifier noise. NaN (non-rice) pixels are ignored and kept as NaN."""
    from scipy.ndimage import uniform_filter
    if size <= 1:
        return water
    v = water.transpose("time", ...).values
    valid = np.isfinite(v)
    num = uniform_filter(np.where(valid, v, 0).astype("float32"), size=(1, size, size), mode="nearest")
    den = uniform_filter(valid.astype("float32"), size=(1, size, size), mode="nearest")
    out = np.where(den > 0, (num / np.maximum(den, 1e-6) >= 0.5).astype("float32"), np.nan)
    out[~valid] = np.nan
    return water.transpose("time", ...).copy(data=out)
