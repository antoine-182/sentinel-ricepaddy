"""Download Sentinel-1 RTC and Sentinel-2 L2A cubes from Microsoft Planetary Computer.

Sentinel-1 RTC = radiometrically terrain corrected backscatter (gamma0, linear power),
already analysis-ready, so no SNAP/ESA toolbox preprocessing is needed.
"""
import numpy as np
import xarray as xr

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def load_s1(cfg: dict) -> xr.Dataset:
    """Sentinel-1 VV/VH backscatter in dB, dims (time, y, x), with an `orbit` coordinate.

    config `sentinel1.orbits`:
      "single" (default): keep the most frequent orbit direction only (consistent geometry).
      "both":  keep ascending and descending passes (about twice as many dates). Each
               relative orbit has its own viewing geometry, so use `normalize_per_orbit`.
    """
    from odc.stac import load
    aoi, per = cfg["aoi"], cfg["period"]
    mode = cfg.get("sentinel1", {}).get("orbits", "single")
    items = _catalog().search(
        collections=["sentinel-1-rtc"],
        bbox=aoi["bbox"],
        datetime=f"{per['start']}/{per['end']}",
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError("No Sentinel-1 RTC items found for this AOI/period.")
    key = lambda it: f"{it.properties.get('sat:orbit_state', '?')[:3]}_{it.properties.get('sat:relative_orbit', '?')}"
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    if mode == "single":
        state = lambda k: k.split("_")[0]
        counts = {}
        for k, its in groups.items():
            counts[state(k)] = counts.get(state(k), 0) + len(its)
        main = max(counts, key=counts.get)
        groups = {k: v for k, v in groups.items() if state(k) == main}
    print("Sentinel-1 orbits kept:", {k: len(v) for k, v in groups.items()})
    parts = []
    for k, its in groups.items():
        # load each relative orbit separately: groupby="solar_day" would otherwise merge
        # a morning (descending) and an evening (ascending) pass of the same day
        ds = load(its, bands=["vv", "vh"], bbox=aoi["bbox"], crs=aoi["crs"],
                  resolution=aoi["resolution"], groupby="solar_day", chunks={})
        parts.append(ds.assign_coords(orbit=("time", np.array([k] * ds.sizes["time"], dtype="U16"))))
    ds = xr.concat(parts, dim="time").sortby("time")
    ds = ds.where(ds > 0)                          # 0 = nodata
    ds = 10 * np.log10(ds)                         # linear power -> dB
    return ds.rename({"vv": "vv_db", "vh": "vh_db"})


def open_s1(cfg: dict) -> xr.Dataset:
    """Open the saved Sentinel-1 cube, normalised per orbit if the config asks for it."""
    ds = xr.open_zarr(cfg["paths"]["s1"]).load()
    if cfg.get("sentinel1", {}).get("normalize") == "per_orbit":
        ds = normalize_per_orbit(ds)
    return ds


def normalize_per_orbit(s1: xr.Dataset, variables=("vv_db", "vh_db")) -> xr.Dataset:
    """Remove the geometry offset between relative orbits.

    For each orbit, standardise with that orbit's scene-wide mean and std (all pixels,
    all its dates), then map back to the mean and std of the reference orbit (the one
    with most dates). Values stay in dB-like units, so thresholds remain meaningful,
    while each orbit gets the same overall level and spread. Pixel-to-pixel and
    date-to-date contrasts within an orbit are preserved.
    """
    if "orbit" not in s1.coords or len(np.unique(s1.orbit.values)) < 2:
        return s1
    out = s1.copy()
    orbits, counts = np.unique(s1.orbit.values, return_counts=True)
    ref = orbits[np.argmax(counts)]
    for v in variables:
        da = s1[v]
        r = da.sel(time=da.orbit == ref)
        mu_ref, sd_ref = float(r.mean()), float(r.std())
        pieces = []
        for o in orbits:
            d = da.sel(time=da.orbit == o)
            pieces.append((d - float(d.mean())) / float(d.std()) * sd_ref + mu_ref)
        out[v] = xr.concat(pieces, dim="time").sortby("time")
    out.attrs["normalized_to_orbit"] = str(ref)
    return out


def check_dates(ds: xr.Dataset, var: str, min_valid: float = 0.5) -> xr.Dataset:
    """Print the valid-pixel share and median of `var` per date, and drop dates where
    less than `min_valid` of the AOI has data (partial swaths, failed reads)."""
    da = ds[var].load()
    valid = np.isfinite(da).mean(["y", "x"]).values
    med = da.median(["y", "x"]).values
    for t, v, m in zip(da.time.values, valid, med):
        flag = "" if v >= min_valid else "  <- dropped"
        print(f"  {str(t)[:10]}  valid={v:4.2f}  median={m:6.2f}{flag}")
    return ds.isel(time=np.nonzero(valid >= min_valid)[0])


def load_s2(cfg: dict) -> xr.Dataset:
    """Sentinel-2 green (B03), SWIR1 (B11) reflectance and scene classification (SCL)."""
    from odc.stac import load
    aoi, per = cfg["aoi"], cfg["period"]
    items = _catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=aoi["bbox"],
        datetime=f"{per['start']}/{per['end']}",
        query={"eo:cloud_cover": {"lt": cfg["sentinel2"]["max_cloud_cover"]}},
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError("No Sentinel-2 items found for this AOI/period.")
    ds = load(items, bands=["B03", "B11", "SCL"], bbox=aoi["bbox"], crs=aoi["crs"],
              resolution=aoi["resolution"], groupby="solar_day", chunks={})
    # Processing baseline >= 04.00 (all data since 25 Jan 2022) adds a +1000 offset to DNs.
    refl = lambda b: ((ds[b].astype("float32") - 1000) / 10000).where(ds[b] > 0)
    return xr.Dataset({"green": refl("B03"), "swir1": refl("B11"), "scl": ds["SCL"]})
