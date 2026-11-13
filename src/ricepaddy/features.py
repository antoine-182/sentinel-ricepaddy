"""Per-pixel, per-date features for the baseline."""
import numpy as np
import pandas as pd
import xarray as xr

FEATURES = ["vv_db", "vh_db", "ratio_db", "vh_prev", "vh_next", "dvh_prev", "dvh_next", "vh_pair", "orbit_code"]


def _shift_within_orbit(vh: xr.DataArray, step: int) -> xr.DataArray:
    """Previous (step=1) or next (step=-1) acquisition *of the same relative orbit*.
    With two orbits, consecutive images alternate between geometries and are sometimes only
    one day apart, so the same-orbit neighbour is the meaningful temporal context."""
    if "orbit" not in vh.coords:
        return vh.shift(time=step)
    parts = [vh.sel(time=vh.orbit == o).shift(time=step) for o in np.unique(vh.orbit.values)]
    return xr.concat(parts, dim="time").sortby("time")


def _pair(vh: xr.DataArray, max_days: float = 2.0) -> xr.DataArray:
    """VH of the other-orbit image acquired within `max_days` (a second look at almost the
    same moment); falls back to the image itself when there is none."""
    if "orbit" not in vh.coords:
        return vh.copy()
    t = pd.DatetimeIndex(vh.time.values)
    orb = vh.orbit.values
    out = vh.copy()
    for i in range(len(t)):
        other = np.nonzero(orb != orb[i])[0]
        if len(other):
            dt = np.abs((t[other] - t[i]) / pd.Timedelta("1D"))
            j = other[np.argmin(dt)]
            if dt.min() <= max_days:
                out[i] = vh[j].values
    return out


def build_features(s1: xr.Dataset) -> xr.Dataset:
    vh, vv = s1["vh_db"], s1["vv_db"]
    f = xr.Dataset({"vv_db": vv, "vh_db": vh, "ratio_db": vh - vv})
    # Temporal context: the previous and next acquisitions of the same orbit. Flooding is a
    # drop in backscatter relative to neighbours, so these help a lot.
    f["vh_prev"] = _shift_within_orbit(vh, 1)
    f["vh_next"] = _shift_within_orbit(vh, -1)
    f["dvh_prev"] = vh - f["vh_prev"]
    f["dvh_next"] = vh - f["vh_next"]
    f["vh_pair"] = _pair(vh)
    # Which relative orbit the image comes from (0 with a single orbit): lets the model
    # account for remaining geometry differences between passes.
    if "orbit" in s1.coords:
        codes = {o: i for i, o in enumerate(sorted(set(s1.orbit.values.tolist())))}
        code = np.array([codes[o] for o in s1.orbit.values], dtype="float32")
    else:
        code = np.zeros(s1.sizes["time"], dtype="float32")
    f["orbit_code"] = xr.DataArray(code, dims="time", coords={"time": s1.time}).broadcast_like(vh)
    return f


def rice_mask_heuristic(s1: xr.Dataset, flood_db: float = -20.0, growth_db: float = 5.0) -> xr.DataArray:
    """Rough rice mask: flooded at least once (very low VH) AND strong VH rise as the crop grows.
    A heuristic to check visually in the notebook, not ground truth."""
    vh = s1["vh_db"]
    return ((vh.min("time") < flood_db) & ((vh.max("time") - vh.min("time")) > growth_db)).rename("rice")


def to_table(feat: xr.Dataset, labels: xr.DataArray, block_size: int, mask: xr.DataArray | None = None) -> pd.DataFrame:
    """Flatten to one row per labelled (pixel, date), with spatial block ids for cross-validation."""
    ny, nx = feat.sizes["y"], feat.sizes["x"]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    nbx = int(np.ceil(nx / block_size))
    block = (yy // block_size) * nbx + (xx // block_size)
    arrs = {k: feat[k].transpose("time", "y", "x").values for k in feat.data_vars}
    lab = labels.transpose("time", "y", "x").values
    keep = lab >= 0
    if mask is not None:
        keep &= mask.transpose("y", "x").values[None]
    keep &= np.all([np.isfinite(a) for a in arrs.values()], axis=0)
    t_idx, y_idx, x_idx = np.nonzero(keep)
    df = pd.DataFrame({k: a[keep] for k, a in arrs.items()})
    df["label"] = lab[keep]
    df["time"] = feat.time.values[t_idx]
    df["y"], df["x"] = y_idx, x_idx
    df["block"] = block[y_idx, x_idx]
    return df
