"""Build weak water labels on the Sentinel-1 dates from clear-sky Sentinel-2 MNDWI."""
import argparse

import xarray as xr

from ricepaddy.config import load_config
from ricepaddy.data import open_s1
from ricepaddy.labels import labels_on_s1_dates, match_dates, water_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    s1 = open_s1(cfg)
    s2 = xr.open_zarr(cfg["paths"]["s2"]).load()
    lc = cfg["labels"]
    lab_s2 = water_labels(s2, lc["water_threshold"], lc["dry_threshold"])
    pairs = match_dates(s1.time.values, s2.time.values, lc["max_days_apart"])
    print(f"{len(pairs)}/{s1.sizes['time']} Sentinel-1 dates have a Sentinel-2 image within "
          f"{lc['max_days_apart']} day(s)")
    lab = labels_on_s1_dates(s1, lab_s2, lc["max_days_apart"])
    n_w, n_d = int((lab == 1).sum()), int((lab == 0).sum())
    print(f"labelled pixel-dates: water={n_w:,}  dry={n_d:,}")
    lab.to_dataset().to_zarr(cfg["paths"]["labels"], mode="w", consolidated=True)


if __name__ == "__main__":
    main()
