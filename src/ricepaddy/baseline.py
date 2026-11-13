"""Two baselines: a fixed VH threshold and a random forest, evaluated with three cross-validation schemes."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from .crossval import SCHEMES
from .features import FEATURES


def threshold_predict(df: pd.DataFrame, thr_db: float = -20.0) -> np.ndarray:
    """Open water is a mirror for radar: little energy comes back, so VH is very low."""
    return (df["vh_db"] < thr_db).astype(int).to_numpy()


def fit_threshold(df: pd.DataFrame, grid=np.arange(-28, -12, 0.5)) -> float:
    scores = [balanced_accuracy_score(df["label"], threshold_predict(df, t)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def make_rf(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.3, n_jobs=-1,
                                  class_weight="balanced", random_state=seed)


def evaluate(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    rows = []
    for scheme, splitter in SCHEMES.items():
        for fold, (tr, te) in enumerate(splitter(df, n_splits, seed)):
            train, test = df.iloc[tr], df.iloc[te]
            y = test["label"].to_numpy()
            thr = fit_threshold(train)
            preds = {"vh_threshold": threshold_predict(test, thr)}
            rf = make_rf(seed).fit(train[FEATURES], train["label"])
            preds["random_forest"] = rf.predict(test[FEATURES])
            for model, p in preds.items():
                rows.append(dict(scheme=scheme, fold=fold, model=model,
                                 balanced_accuracy=balanced_accuracy_score(y, p),
                                 f1_water=f1_score(y, p, zero_division=0),
                                 n_test=len(y)))
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    return (res.groupby(["model", "scheme"])[["balanced_accuracy", "f1_water"]]
               .agg(["mean", "std"]).round(3))
