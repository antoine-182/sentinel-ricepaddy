"""Download Sentinel-1 RTC and Sentinel-2 L2A cubes for the AOI and save them as Zarr.

    python scripts/01_download.py            # both sensors
    python scripts/01_download.py --only s1
"""
import argparse
import time

import numpy as np

from ricepaddy.config import load_config
from ricepaddy.data import check_dates, load_s1, load_s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", choices=["s1", "s2"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name, loader in [("s1", load_s1), ("s2", load_s2)]:
        if args.only and args.only != name:
            continue
        t0 = time.time()
        ds = loader(cfg)
        ds = check_dates(ds, "vh_db" if name == "s1" else "green", min_valid=0.5 if name == "s1" else 0.0)
        if name == "s1" and "orbit" in ds.coords:
            print("  dates per orbit:", dict(zip(*np.unique(ds.orbit.values, return_counts=True))))
        print(f"{name}: {ds.sizes['time']} dates, {ds.sizes['y']}x{ds.sizes['x']} px -> {cfg['paths'][name]}")
        ds.load().to_zarr(cfg["paths"][name], mode="w", consolidated=True)
        print(f"{name}: saved in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
