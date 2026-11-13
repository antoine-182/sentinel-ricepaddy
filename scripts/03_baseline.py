"""Train and evaluate the baselines with random / spatial-block / temporal cross-validation,
then map water status and AWD drying events over the whole AOI.

    python scripts/03_baseline.py              # real data (run 01 and 02 first)
    python scripts/03_baseline.py --synthetic  # quick check without downloading anything
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ricepaddy.awd import PRACTICES, drying_events, flooded_fraction, intermediate_dry_spells, majority_filter, practice_class
from ricepaddy.baseline import evaluate, make_rf, summarize
from ricepaddy.config import load_config
from ricepaddy.data import open_s1
from ricepaddy.features import FEATURES, build_features, rice_mask_heuristic, to_table


warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-mask", action="store_true", help="use all pixels, not only the rice heuristic mask")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["outputs"]) / ("synthetic" if args.synthetic else cfg.get("run_name", "real"))
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from ricepaddy.synthetic import make_synthetic
        s1, labels, _ = make_synthetic()
        block = 16
    else:
        s1 = open_s1(cfg)
        if "orbit" in s1.coords:
            print("orbits:", dict(zip(*np.unique(s1.orbit.values, return_counts=True))))
        labels = xr.open_zarr(cfg["paths"]["labels"])["label"].load()
        block = cfg["crossval"]["block_size_px"]

    mask = None if (args.no_mask or args.synthetic) else rice_mask_heuristic(s1)
    feat = build_features(s1)
    df = to_table(feat, labels, block, mask)
    n_max = cfg["crossval"]["max_train_samples"]
    if len(df) > n_max:
        df = df.sample(n_max, random_state=cfg["crossval"]["seed"])
    print(f"{len(df):,} labelled samples, water share = {df['label'].mean():.2f}, "
          f"{df['block'].nunique()} spatial blocks")

    res = evaluate(df, cfg["crossval"]["n_splits"], cfg["crossval"]["seed"])
    res.to_csv(out / "cv_results.csv", index=False)
    summary = summarize(res)
    print(summary.to_string())
    summary.to_csv(out / "cv_summary.csv")

    # Fit on everything, predict every pixel and date, derive AWD indicators.
    rf = make_rf(cfg["crossval"]["seed"]).fit(df[FEATURES], df["label"])
    import pandas as pd
    X = pd.DataFrame({k: feat[k].values.ravel() for k in FEATURES})
    ok = np.isfinite(X.to_numpy()).all(axis=1)
    pred = np.full(len(X), np.nan, dtype="float32")
    pred[ok] = rf.predict(X[ok])
    water = xr.DataArray(pred.reshape(feat["vh_db"].shape), coords=feat["vh_db"].coords, name="water")
    if mask is not None:
        water = water.where(mask)
    awd_cfg = cfg.get("awd", {})
    crop = (None, None) if args.synthetic else (awd_cfg.get("crop_start"), awd_cfg.get("crop_end"))
    smoothed = majority_filter(water, awd_cfg.get("majority_px", 1))
    spells = intermediate_dry_spells(smoothed, *crop, min_dry_dates=awd_cfg.get("min_dry_dates", 1))
    if mask is not None:
        spells = spells.where(mask)
    practice = practice_class(spells)
    xr.Dataset({"water": water, "drying_events": drying_events(water),
                "flooded_fraction": flooded_fraction(water),
                "intermediate_dry_spells": spells, "practice": practice}).to_zarr(out / "predictions.zarr", mode="w")
    shares = practice.to_series().value_counts(normalize=True).sort_index()
    print("irrigation practice over the crop window (share of rice pixels):")
    for k, v in shares.items():
        print(f"  {PRACTICES[int(k)]:<30s} {v:.1%}")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    drying_events(water).where(np.isfinite(water).any("time")).plot(ax=axs[0], cmap="viridis")
    axs[0].set_title("Drying events per pixel (AWD proxy)")
    flooded_fraction(water).plot(ax=axs[1], cmap="Blues", vmin=0, vmax=1)
    axs[1].set_title("Fraction of dates flooded")
    from matplotlib.colors import ListedColormap
    im = axs[2].imshow(practice.values, cmap=ListedColormap(["#9ecae1", "#fdae6b", "#31a354"]), vmin=-0.5, vmax=2.5,
                       interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[2], ticks=[0, 1, 2]); cb.ax.set_yticklabels(["continuous", "1 drainage", "AWD-like"])
    axs[2].set_title("Irrigation practice (crop window)")
    for a in axs:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "awd_maps.png", dpi=120)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
