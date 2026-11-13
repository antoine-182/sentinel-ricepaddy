"""Make the figures of the project article (docs/index.md) from the real data and results.

    python scripts/04_figures.py                          # chapter 1 -> docs/assets/
    python scripts/04_figures.py --config config_ch2.yaml # chapter 2 -> docs/assets/both_orbits/

Run after 01, 02 and 03.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

from ricepaddy.awd import PRACTICES
from ricepaddy.config import ROOT, load_config
from ricepaddy.data import open_s1
from ricepaddy.features import rice_mask_heuristic
from ricepaddy.labels import CLEAR_SCL, match_dates, mndwi

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
LAB_CMAP = ListedColormap(["#f2f2f2", "#b5835a", "#2b7bba"])   # unlabelled, dry, water
OUT = ROOT / "docs" / "assets"
PRACTICE_COLORS = ["#9ecae1", "#fdae6b", "#31a354"]


def extent(da):
    x, y = da.x.values / 1000, da.y.values / 1000
    return [x.min(), x.max(), y.min(), y.max()]


def km_axes(ax):
    ax.set_xlabel("km (UTM 48N)"); ax.set_ylabel("km")


def fig_timeline(s1, s2, pairs, clear):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    cs = clear.mean(["y", "x"]).to_series()
    ax.bar(cs.index, cs.values, width=2.5, color="#f0a35e", label="Sentinel-2 image (height = cloud-free share)")
    for t in s1.time.values:
        ax.axvline(t, ymin=0, ymax=1, color="#2b7bba" if pd.Timestamp(t) in pairs else "#c8c8c8", lw=2)
    ax.plot([], [], color="#2b7bba", lw=2, label="Sentinel-1 date with a label")
    ax.plot([], [], color="#c8c8c8", lw=2, label="Sentinel-1 date without label")
    ax.set_ylim(0, 1.05); ax.set_ylabel("clear share")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.set_title("Two satellites, two calendars")
    fig.savefig(OUT / "fig1_timeline.png", bbox_inches="tight"); plt.close(fig)


def fig_s1_s2(s1, s2, m, labels, pairs):
    # pick the matched date with most labelled pixels
    t1 = max(pairs, key=lambda t: int((labels.sel(time=t) >= 0).sum()))
    t2 = pairs[t1]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    ext = extent(s1)
    axs[0].imshow(s1.vh_db.sel(time=t1), cmap="gray", vmin=-28, vmax=-8, extent=ext)
    axs[0].set_title(f"Sentinel-1 VH, {str(t1)[:10]}\n(dark = smooth water)")
    im = axs[1].imshow(m.sel(time=t2), cmap="RdBu", vmin=-0.6, vmax=0.6, extent=ext)
    axs[1].set_title(f"Sentinel-2 MNDWI, {str(t2)[:10]}\n(blue = water, grey = cloud)")
    axs[1].set_facecolor("#bdbdbd")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    axs[2].imshow(labels.sel(time=t1) + 1, cmap=LAB_CMAP, vmin=0, vmax=2, interpolation="nearest", extent=ext)
    axs[2].set_title("Weak label for the radar date\n(blue water, brown dry, light = none)")
    for a in axs: km_axes(a)
    fig.tight_layout(); fig.savefig(OUT / "fig2_s1_s2.png", bbox_inches="tight"); plt.close(fig)


def fig_timeseries(s1, s2, m, labels, mask):
    # pick a rice pixel with many labels and both states
    lab = labels.values
    score = ((lab == 1).sum(0) > 0) & ((lab == 0).sum(0) > 0) & mask.values
    n = np.where(score, (lab >= 0).sum(0), -1)
    iy, ix = np.unravel_index(np.argmax(n), n.shape)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(s1.time, s1.vh_db[:, iy, ix], "o-", color="#2b2b2b", label="Sentinel-1 VH (dB)")
    ax.plot(s1.time, s1.vv_db[:, iy, ix], "s-", color="#8c8c8c", alpha=.7, label="Sentinel-1 VV (dB)")
    for t, l in zip(s1.time.values, lab[:, iy, ix]):
        if l >= 0:
            ax.axvspan(t - np.timedelta64(2, "D"), t + np.timedelta64(2, "D"),
                       color="#2b7bba" if l == 1 else "#b5835a", alpha=.25, lw=0)
    ax2 = ax.twinx()
    ax2.plot(s2.time, m[:, iy, ix], "D", color="#2b7bba", ms=5, label="Sentinel-2 MNDWI (clear days)")
    ax2.axhline(0, ls=":", color="#2b7bba", lw=1); ax2.set_ylim(-0.8, 0.8); ax2.set_ylabel("MNDWI")
    ax2.spines["right"].set_visible(True)
    ax.set_ylabel("backscatter (dB)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title("One rice pixel through the season (shading = weak label: blue water, brown dry)")
    fig.savefig(OUT / "fig3_timeseries.png", bbox_inches="tight"); plt.close(fig)


def fig_cv(res):
    order = ["random", "spatial_blocks", "temporal"]
    names = {"random": "random", "spatial_blocks": "spatial blocks", "temporal": "temporal"}
    models = {"vh_threshold": ("VH threshold", "#c8c8c8"), "random_forest": ("random forest", "#2b7bba")}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.36
    for k, (mod, (lab, col)) in enumerate(models.items()):
        for i, s in enumerate(order):
            v = res[(res.model == mod) & (res.scheme == s)]["balanced_accuracy"].to_numpy()
            x = i + (k - 0.5) * w
            ax.bar(x, v.mean(), w * 0.95, color=col, label=lab if i == 0 else None)
            ax.plot(np.full(len(v), x), v, "o", color="k", ms=3, alpha=.6)
            ax.text(x, v.mean() + 0.01, f"{v.mean():.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([names[s] for s in order])
    ax.set_ylim(0.4, 1); ax.set_ylabel("balanced accuracy")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("Same model, three ways to split the data (dots = folds)")
    fig.tight_layout(); fig.savefig(OUT / "fig4_cv.png", bbox_inches="tight"); plt.close(fig)


def fig_awd(pred):
    de = pred.drying_events.where(np.isfinite(pred.water).any("time"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    im = axs[0].imshow(de, cmap="viridis", vmin=0, vmax=5, extent=extent(pred), interpolation="nearest")
    fig.colorbar(im, ax=axs[0], fraction=0.046, label="drying events")
    axs[0].set_title("Predicted drying events per pixel (white = not rice)"); km_axes(axs[0])
    v = de.values.ravel(); v = v[np.isfinite(v)]
    axs[1].hist(v, bins=np.arange(-0.5, 8.5, 1), color="#2b7bba")
    axs[1].set_xlabel("drying events over the season"); axs[1].set_ylabel("rice pixels")
    axs[1].set_title("Distribution")
    fig.tight_layout(); fig.savefig(OUT / "fig5_awd.png", bbox_inches="tight"); plt.close(fig)


def fig_practice(pred, s1, mask):
    pr = pred.practice
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    im = axs[0].imshow(pr, cmap=ListedColormap(PRACTICE_COLORS), vmin=-0.5, vmax=2.5,
                       extent=extent(pred), interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["continuous\nflooding", "1 mid-season\ndrainage", "AWD-like\n(2+ re-floodings)"])
    axs[0].set_title("Irrigation practice over the crop (white = not rice)"); km_axes(axs[0])
    # median VH time series per class: what each practice looks like to the radar
    for k, col in enumerate(PRACTICE_COLORS):
        sel = (pr == k).values
        if sel.sum() == 0:
            continue
        ts = np.nanmedian(s1.vh_db.values[:, sel], axis=1)
        axs[1].plot(s1.time, ts, "o-", color=col, label=f"{PRACTICES[k]} ({sel.mean() / mask.values.mean():.0%})")
    axs[1].set_ylabel("median VH (dB)"); axs[1].legend(fontsize=7, frameon=False)
    axs[1].set_title("Median radar signal per class"); axs[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "fig6_practice.png", bbox_inches="tight"); plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    run = cfg.get("run_name", "real")
    if run != "real":
        OUT = OUT / run
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = open_s1(cfg)
    s2 = xr.open_zarr(cfg["paths"]["s2"]).load()
    labels = xr.open_zarr(cfg["paths"]["labels"])["label"].load()
    res_dir = Path(cfg["paths"]["outputs"]) / run
    m, clear = mndwi(s2), s2["scl"].isin(CLEAR_SCL)
    pairs = match_dates(s1.time.values, s2.time.values, cfg["labels"]["max_days_apart"])
    fig_timeline(s1, s2, pairs, clear)
    fig_s1_s2(s1, s2, m, labels, pairs)
    mask = rice_mask_heuristic(s1)
    fig_timeseries(s1, s2, m, labels, mask)
    fig_cv(pd.read_csv(res_dir / "cv_results.csv"))
    pred = xr.open_zarr(res_dir / "predictions.zarr").load()
    fig_awd(pred)
    if "practice" in pred:
        fig_practice(pred, s1, mask)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
