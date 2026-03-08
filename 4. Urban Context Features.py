"""
WalkGeoAI Feature Extraction Pipeline
-------------------------------------
Complete automated pipeline for extracting multi-scale built environment
features for street-level pedestrian density estimation.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.features import geometry_mask
from affine import Affine
import fiona
from pyproj import CRS
from scipy.ndimage import (
    gaussian_filter, maximum_filter,
)

warnings.filterwarnings("ignore")


def get_paths(base_dir, city_name):
    """Generate standard directory paths for a city."""
    shp_dir = os.path.join(base_dir, city_name, "shp file")
    data_dir = os.path.join(base_dir, city_name, "data file")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return shp_dir, data_dir

# =====================================================================
# Urban Context Features
# =====================================================================
def urban_context_features(base_dir, city_name, gub_path):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    REPROJECTED_RASTER = os.path.join(shp_dir, "pop_clipped_utm.tif")
    BORO_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")
    OUTPUT_PATH = os.path.join(data_dir, "urban_context_features.csv")

    CORE_Q = 0.90
    SUBCENTER_K = 8
    SUBCENTER_MIN_DIST_M = 2000
    SUBCENTER_SMOOTH_M = 800
    SUBCENTER_PEAK_Q = 0.995
    DOWNSAMPLE = 2
    CHUNK = 200_000

    t0 = time.time()

    print("Reading population raster (UTM)...")
    with rasterio.open(REPROJECTED_RASTER) as src:
        pop = src.read(1).astype(np.float32)
        aff = src.transform
        pop_crs = src.crs
        nodata = src.nodata
    H, W = pop.shape
    xres = float(aff.a)
    yres = float(-aff.e)

    if pop_crs is None: raise ValueError("Population raster has no CRS. Must be projected.")
    if nodata is not None: pop = np.where(pop == nodata, 0.0, pop)
    pop = np.where(np.isfinite(pop), pop, 0.0)
    pop = np.where(pop < 0, 0.0, pop)

    print(f"  raster size: {W} x {H}, res≈{xres:.1f}m")

    print("Loading roads...")
    roads = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    if roads.crs is None: raise ValueError("roads CRS is None.")
    roads = roads.to_crs(pop_crs)

    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    N = len(roads)
    print(f"  {N:,} segments")

    mid = roads.geometry.interpolate(0.5, normalized=True)
    xm = mid.x.values.astype(np.float64)
    ym = mid.y.values.astype(np.float64)

    print("Loading administrative boundary...")
    boro = gpd.read_file(BORO_PATH)
    if boro.crs is None: boro = boro.set_crs("EPSG:4326")
    boro_utm = boro.to_crs(pop_crs)
    boro_poly = boro_utm.union_all()

    print("Selecting GUB city polygon (spatial match)...")
    gub_poly = None
    try:
        with fiona.open(gub_path) as fsrc:
            gub_crs = CRS.from_user_input(fsrc.crs_wkt or fsrc.crs)

        boro_wgs = gpd.GeoSeries([boro_poly], crs=pop_crs).to_crs(gub_crs).iloc[0]
        minx, miny, maxx, maxy = boro_wgs.bounds
        gub_sub = gpd.read_file(gub_path, bbox=(minx, miny, maxx, maxy))

        if len(gub_sub) > 0:
            gub_sub = gub_sub.set_crs(gub_crs, allow_override=True).to_crs(pop_crs)
            inter_area = gub_sub.geometry.intersection(boro_poly).area.values
            j = int(np.argmax(inter_area))
            if inter_area[j] > 0:
                gub_poly = gub_sub.geometry.iloc[j]
    except Exception:
        gub_poly = None

    city_poly = gub_poly if gub_poly is not None else boro_poly
    city_boundary_line = city_poly.boundary

    print(f"  boundary source: {'GUB' if gub_poly is not None else 'Administrative boundary'}")

    print("Masking population raster to city polygon...")
    mask_outside = geometry_mask([mapping(city_poly)], out_shape=(H, W), transform=aff, invert=False)
    inside = ~mask_outside
    pop_city = pop.copy()
    pop_city[~inside] = 0.0

    tot_pop = float(pop_city.sum())
    if tot_pop <= 0:
        raise ValueError("Total population inside city polygon is 0. Check CRS alignment.")

    print("Computing population-weighted center (PWC)...")
    col_sums = pop_city.sum(axis=0).astype(np.float64)
    row_sums = pop_city.sum(axis=1).astype(np.float64)

    x_centers = aff.c + (np.arange(W) + 0.5) * aff.a
    y_centers = aff.f + (np.arange(H) + 0.5) * aff.e

    pwc_x = float((col_sums * x_centers).sum() / tot_pop)
    pwc_y = float((row_sums * y_centers).sum() / tot_pop)

    dx = xm - pwc_x
    dy = ym - pwc_y
    dist_pwc_km = np.sqrt(dx * dx + dy * dy) / 1000.0

    print("Computing distance to city edge (chunked)...")
    edge_dist_km = np.zeros(N, dtype=np.float32)
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        pts = gpd.GeoSeries(gpd.points_from_xy(xm[s:e], ym[s:e]), crs=pop_crs)
        edge_dist_km[s:e] = (pts.distance(city_boundary_line).values / 1000.0).astype(np.float32)

    print("Computing core indicator (cell-based, quantile threshold)...")
    pos = pop_city[pop_city > 0]
    if len(pos) == 0:
        core_thr = 0.0
    else:
        core_thr = float(np.quantile(pos, CORE_Q))

    cols = np.floor((xm - aff.c) / aff.a).astype(np.int32)
    rows = np.floor((ym - aff.f) / aff.e).astype(np.int32)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    mid_pop = np.zeros(N, dtype=np.float32)
    mid_pop[valid] = pop_city[rows[valid], cols[valid]]
    core_bin = (mid_pop >= core_thr).astype(np.int8)

    print("Detecting subcenters from population raster (downsample + smooth + peaks)...")
    f = int(max(1, DOWNSAMPLE))
    H2 = (H // f) * f
    W2 = (W // f) * f
    pop_ds = pop_city[:H2, :W2].reshape(H2 // f, f, W2 // f, f).sum(axis=(1, 3)).astype(np.float32)

    aff_ds = Affine(aff.a * f, aff.b, aff.c, aff.d, aff.e * f, aff.f)
    xres_ds = xres * f

    sigma = max(0.5, SUBCENTER_SMOOTH_M / xres_ds)
    smooth = gaussian_filter(pop_ds, sigma=sigma, mode="constant").astype(np.float32)

    win = int(max(1, np.ceil(SUBCENTER_MIN_DIST_M / xres_ds)))
    win = win * 2 + 1
    mx = maximum_filter(smooth, size=win, mode="constant")

    cand = (smooth == mx) & (smooth > 0)
    sp = smooth[smooth > 0]
    if sp.size > 0:
        thr = float(np.quantile(sp, SUBCENTER_PEAK_Q))
        cand &= (smooth >= thr)

    r_idx, c_idx = np.where(cand)
    if r_idx.size == 0:
        centers_xy = np.array([[pwc_x, pwc_y]], dtype=np.float64)
        center_rank = np.array([1], dtype=np.int32)
    else:
        vals = smooth[r_idx, c_idx]
        order = np.argsort(vals)[::-1]
        r_idx = r_idx[order]
        c_idx = c_idx[order]

        k = min(SUBCENTER_K, r_idx.size)
        r_idx = r_idx[:k]
        c_idx = c_idx[:k]

        cx = aff_ds.c + (c_idx + 0.5) * aff_ds.a
        cy = aff_ds.f + (r_idx + 0.5) * aff_ds.e
        centers_xy = np.column_stack([cx, cy]).astype(np.float64)
        center_rank = np.arange(1, centers_xy.shape[0] + 1, dtype=np.int32)

    K = centers_xy.shape[0]
    print(f"  subcenters used: {K}")

    dx = xm[:, None] - centers_xy[None, :, 0]
    dy = ym[:, None] - centers_xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy).astype(np.float64) / 1000.0

    idx1 = np.argmin(d, axis=1)
    d1 = d[np.arange(N), idx1].astype(np.float32)

    if K >= 2:
        d2 = np.partition(d, 1, axis=1)[:, 1].astype(np.float32)
    else:
        d2 = np.full(N, np.nan, dtype=np.float32)

    rank_nearest = center_rank[idx1].astype(np.float32)

    df = pd.DataFrame({
        "id": roads["id"].values,
        "urban_dist_pwc_seg_val": dist_pwc_km.astype(np.float32),
        "urban_dist_edge_seg_val": edge_dist_km.astype(np.float32),
        "urban_core_indicator_seg_bin": core_bin,
        "urban_dist_subcenter1_seg_val": d1,
        "urban_dist_subcenter2_seg_val": d2,
        "urban_subcenter_rank_seg_val": rank_nearest
    })

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {N:,} segments × {len(df.columns)} columns")
    print(f"  Total time: {time.time() - t0:.1f}s")
