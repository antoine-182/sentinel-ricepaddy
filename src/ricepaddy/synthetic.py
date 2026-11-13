"""Small synthetic Sentinel-1 / label cubes, to test the pipeline without downloading data.
Fields are rectangles; each field follows a rice calendar with continuous flooding or AWD."""
import numpy as np
import pandas as pd
import xarray as xr


def make_synthetic(ny=128, nx=128, n_dates=25, field=16, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-11-01", periods=n_dates, freq="6D")
    water = np.zeros((n_dates, ny, nx), dtype="int8")
    canopy = np.zeros((n_dates, ny, nx), dtype="float32")
    for y0 in range(0, ny, field):
        for x0 in range(0, nx, field):
            sow = rng.integers(0, 4)
            awd = rng.random() < 0.5
            for t in range(n_dates):
                age = t - sow
                flooded = 0 <= age < 16
                if awd and 4 <= age < 14 and age % 4 >= 2:
                    flooded = False                  # periodic drying
                water[t, y0:y0 + field, x0:x0 + field] = flooded
                canopy[t, y0:y0 + field, x0:x0 + field] = np.clip(age / 12, 0, 1) if age >= 0 else 0
    noise = lambda: rng.normal(0, 1.5, water.shape)  # speckle-like noise (dB)
    vh = -16 + 6 * canopy - 9 * water * (1 - 0.6 * canopy) + noise()
    vv = -10 + 3 * canopy - 8 * water * (1 - 0.6 * canopy) + noise()
    coords = dict(time=times, y=np.arange(ny), x=np.arange(nx))
    s1 = xr.Dataset({"vv_db": (("time", "y", "x"), vv.astype("float32")),
                     "vh_db": (("time", "y", "x"), vh.astype("float32"))}, coords=coords)
    lab = water.copy()
    lab[rng.random(water.shape) < 0.5] = -1         # clouds: half of the labels missing
    labels = xr.DataArray(lab, coords=coords, dims=("time", "y", "x"), name="label")
    truth = xr.DataArray(water, coords=coords, dims=("time", "y", "x"), name="water")
    return s1, labels, truth
