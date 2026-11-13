import numpy as np
import xarray as xr

from ricepaddy.awd import intermediate_dry_spells, practice_class
from ricepaddy.data import normalize_per_orbit
from ricepaddy.synthetic import make_synthetic


def _w(seq):
    return xr.DataArray(np.array(seq, dtype=float)[:, None, None], dims=("time", "y", "x"))


def test_intermediate_spells():
    assert intermediate_dry_spells(_w([0, 0, 1, 1, 1, 1, 0, 0])).item() == 0      # continuous
    assert intermediate_dry_spells(_w([0, 1, 1, 0, 1, 1, 0])).item() == 1         # mid-season drainage
    assert intermediate_dry_spells(_w([0, 1, 0, 1, 0, 0, 1, 1, 0])).item() == 2   # AWD
    assert intermediate_dry_spells(_w([0, 1, np.nan, 0, 1, 0])).item() == 1       # NaN carried forward
    assert intermediate_dry_spells(_w([0, 1, 0, 1, 0, 0, 1, 1, 0]), min_dry_dates=2).item() == 1
    assert practice_class(intermediate_dry_spells(_w([0, 1, 0, 1, 0, 1, 0, 1]))).item() == 2


def test_normalize_per_orbit():
    s1, _, _ = make_synthetic(ny=32, nx=32, n_dates=12)
    orb = np.array(["asc_1", "des_2"] * 6, dtype="U16")
    s1 = s1.assign_coords(orbit=("time", orb))
    s1["vh_db"] = s1.vh_db + xr.DataArray(np.where(orb == "des_2", 3.0, 0.0), dims="time")  # geometry offset
    n = normalize_per_orbit(s1)
    m = n.vh_db.groupby("orbit").mean(...).values
    sd = n.vh_db.groupby("orbit").std(...).values
    assert abs(m[0] - m[1]) < 1e-3 and abs(sd[0] - sd[1]) < 1e-3


def test_same_orbit_features():
    from ricepaddy.features import build_features
    s1, _, _ = make_synthetic(ny=8, nx=8, n_dates=6)
    orb = np.array(["a", "b"] * 3, dtype="U4")
    s1 = s1.assign_coords(orbit=("time", orb))
    f = build_features(s1)
    # previous same-orbit image of date 2 (orbit a) is date 0
    assert np.allclose(f.vh_prev[2].values, s1.vh_db[0].values)
    assert np.isfinite(f.vh_pair).all()


def test_majority_filter():
    from ricepaddy.awd import majority_filter
    v = np.ones((1, 5, 5)); v[0, 2, 2] = 0; v[0, 0, 0] = np.nan
    w = xr.DataArray(v, dims=("time", "y", "x"))
    out = majority_filter(w, 3).values
    assert out[0, 2, 2] == 1 and np.isnan(out[0, 0, 0])
