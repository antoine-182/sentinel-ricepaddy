"""Cross-validation schemes: random (leaky) vs spatial blocks vs temporal.

Neighbouring pixels in the same field are almost identical. A random split puts some of
them in train and others in test, so the score measures memorisation of fields rather
than generalisation to new fields. Comparing the three schemes quantifies that leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def random_splits(df: pd.DataFrame, n_splits: int, seed: int = 0):
    yield from KFold(n_splits, shuffle=True, random_state=seed).split(df)


def spatial_block_splits(df: pd.DataFrame, n_splits: int, seed: int = 0):
    # Shuffle block ids so folds are not spatial stripes.
    rng = np.random.default_rng(seed)
    blocks = df["block"].unique()
    perm = dict(zip(blocks, rng.permutation(len(blocks))))
    yield from GroupKFold(n_splits).split(df, groups=df["block"].map(perm))


def temporal_splits(df: pd.DataFrame, n_splits: int, seed: int = 0):
    """Forward chaining: train on earlier dates, test on the next chunk of dates."""
    dates = np.sort(df["time"].unique())
    n = min(n_splits, len(dates) - 1)          # with few labelled dates, fewer folds
    if n < 1:
        return
    chunks = np.array_split(dates, n + 1)
    for i in range(1, n + 1):
        train = df["time"].isin(np.concatenate(chunks[:i])).to_numpy()
        test = df["time"].isin(chunks[i]).to_numpy()
        if test.any() and train.any():
            yield np.nonzero(train)[0], np.nonzero(test)[0]


SCHEMES = {"random": random_splits, "spatial_blocks": spatial_block_splits, "temporal": temporal_splits}
