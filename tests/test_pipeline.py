import numpy as np
import pandas as pd
import xarray as xr

from ricepaddy.awd import drying_events
from ricepaddy.baseline import evaluate
from ricepaddy.crossval import spatial_block_splits, temporal_splits
from ricepaddy.features import build_features, to_table
from ricepaddy.labels import match_dates, water_labels
from ricepaddy.synthetic import make_synthetic


def _table():
    s1, labels, _ = make_synthetic(ny=64, nx=64, n_dates=20)
    return to_table(build_features(s1), labels, block_size=16)


def test_spatial_blocks_do_not_leak():
    df = _table()
    for tr, te in spatial_block_splits(df, 4):
        assert set(df.iloc[tr]["block"]).isdisjoint(df.iloc[te]["block"])


def test_temporal_splits_train_before_test():
    df = _table()
    for tr, te in temporal_splits(df, 3):
        assert df.iloc[tr]["time"].max() < df.iloc[te]["time"].min()


def test_baseline_learns_water():
    res = evaluate(_table(), n_splits=3)
    rf = res[(res.model == "random_forest") & (res.scheme == "spatial_blocks")]
    assert rf["balanced_accuracy"].mean() > 0.8


def test_match_dates():
    s1 = pd.to_datetime(["2025-01-01", "2025-01-07"])
    s2 = pd.to_datetime(["2025-01-02", "2025-01-20"])
    assert match_dates(s1, s2, 1) == {s1[0]: s2[0]}


def test_water_labels_thresholds():
    s2 = xr.Dataset({"green": ("t", [0.10, 0.05, 0.08, 0.10]),
                     "swir1": ("t", [0.02, 0.15, 0.09, 0.02]),
                     "scl": ("t", [6, 4, 5, 9])})
    assert water_labels(s2).values.tolist() == [1, 0, -1, -1]   # water, dry, ambiguous, cloud


def test_drying_events():
    w = xr.DataArray(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=float)[:, None, None], dims=("time", "y", "x"))
    assert int(drying_events(w).item()) == 2
