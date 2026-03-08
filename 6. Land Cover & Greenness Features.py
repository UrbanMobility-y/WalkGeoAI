"""
WalkGeoAI Feature Extraction Pipeline
-------------------------------------
Complete automated pipeline for extracting multi-scale built environment
features for street-level pedestrian density estimation.
"""

import os
import glob
import time
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.merge import merge
from scipy.ndimage import (
    uniform_filter, percentile_filter, distance_transform_edt
)

def get_paths(base_dir, city_name):
    """Generate standard directory paths for a city."""
    shp_dir = os.path.join(base_dir, city_name, "shp file")
    data_dir = os.path.join(base_dir, city_name, "data file")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return shp_dir, data_dir


# =====================================================================
# Land Cover & Greenness Features
# =====================================================================
def landcover_greenness_features(base_dir, city_name, ndvi_raster_path):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    BOUNDARY_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")
    OUTPUT_PATH = os.path.join(data_dir, "landcover_greenness_features.csv")

    BUFFER_RADII = [50, 200, 500]

    wc_search_pattern = os.path.join(shp_dir, "*WorldCover*.tif")
    wc_files = glob.glob(wc_search_pattern)

    if len(wc_files) == 0:
        raise FileNotFoundError(f"No WorldCover tif files found: {wc_search_pattern}")
    elif len(wc_files) == 1:
        WC_RASTER_IN = wc_files[0]
        print(f"Found 1 WorldCover file: {WC_RASTER_IN}")
    else:
        print(f"Found {len(wc_files)} WorldCover files, performing automatic Mosaic...")
        src_files_to_mosaic = [rasterio.open(fp) for fp in wc_files]

        mosaic, out_trans = merge(src_files_to_mosaic)

        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2],
            "transform": out_trans
        })

        WC_RASTER_IN = os.path.join(tempfile.gettempdir(), f"{city_name.replace(' ', '_')}_WorldCover_merged_temp.tif")
        with rasterio.open(WC_RASTER_IN, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files_to_mosaic:
            src.close()
        print(f"Mosaic complete! Temporary file used as input: {WC_RASTER_IN}")

    RES_M = 30.0
    CLIP_BUFFER = 3000.0

    WC_CLASSES = {
        10: "tree", 20: "shrubland", 30: "grassland", 40: "cropland",
        50: "builtup", 60: "bare", 70: "snowice", 80: "water",
        90: "wetland", 95: "mangroves", 100: "mosslichen"
    }
    GREEN_CODES = {10, 20, 30, 90, 95, 100}
    BLUE_CODES = {80}
    IMPERV_CODES = {50}
    NDVI_NODATA_DEFAULT = -9999

    def crop_bbox_to_temp(raster_in, bbox_geom_proj, crs_proj, buffer_m, tag):
        with rasterio.open(raster_in) as src:
            raster_crs = src.crs
            gdf = gpd.GeoDataFrame(geometry=[bbox_geom_proj.buffer(buffer_m)], crs=crs_proj).to_crs(raster_crs)
            geom_r = gdf.geometry.iloc[0]
            minx, miny, maxx, maxy = geom_r.bounds

            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            win = win.round_offsets().round_lengths()

            arr = src.read(1, window=win)
            out_transform = src.window_transform(win)

            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                "transform": out_transform, "count": 1, "compress": "lzw"
            })

            out_path = os.path.join(tempfile.gettempdir(), f"{city_name.replace(' ', '_')}_{tag}_bboxcrop.tif")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(arr, 1)
        return out_path

    def reproject_to_utm_res(raster_in, dst_crs, res_m, tag, is_categorical, force_float32=False):
        out_path = os.path.join(tempfile.gettempdir(), f"{city_name.replace(' ', '_')}_{tag}_utm_{int(res_m)}m.tif")

        with rasterio.open(raster_in) as src:
            src_dtype = src.dtypes[0]
            out_dtype = "float32" if force_float32 else src_dtype

            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=res_m
            )

            meta = src.meta.copy()
            meta.update({
                "crs": dst_crs, "transform": transform, "width": width, "height": height,
                "dtype": out_dtype, "count": 1, "compress": "lzw"
            })

            resampling = Resampling.nearest if is_categorical else Resampling.bilinear

            with rasterio.open(out_path, "w", **meta) as dst:
                dest = np.zeros((height, width), dtype=np.float32 if out_dtype == "float32" else np.dtype(out_dtype))
                reproject(
                    source=rasterio.band(src, 1), destination=dest,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs,
                    resampling=resampling, src_nodata=src.nodata, dst_nodata=None,
                )
                dst.write(dest.astype(out_dtype), 1)
        return out_path

    def sample3(arr2d, r0, c0, rm, cm, r1, c1):
        return (arr2d[r0, c0] + arr2d[rm, cm] + arr2d[r1, c1]) / 3.0

    def shannon_entropy_from_props(P):
        try:
            P = np.clip(P, 0.0, 1.0)
            mask = P > 0
            out = np.zeros(P.shape[0], dtype=np.float32)
            out[mask.any(axis=1)] = (-np.sum(np.where(mask, P * np.log(P), 0.0), axis=1)).astype(np.float32)
        except:
            print('shannon_entropy_from_props errors !')
            return 0
        return out

    t0 = time.time()
    print("Loading boundary & roads...")
    boundary = gpd.read_file(BOUNDARY_PATH)
    if boundary.crs is None: boundary = boundary.set_crs("EPSG:4326")
    CRS_PROJ = boundary.estimate_utm_crs()
    boundary = boundary.to_crs(CRS_PROJ)
    city_poly = boundary.union_all()
    try: city_poly = shapely.make_valid(city_poly)
    except Exception: pass

    roads = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    if roads.crs is None: raise ValueError("road_all.geojson has no CRS.")
    roads = roads.to_crs(CRS_PROJ)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    N = len(roads)
    print(f"  segments: {N:,}  ({time.time() - t0:.1f}s)")

    gvals = roads.geometry.values
    c0 = np.array([list(g.coords)[0] for g in gvals], dtype=np.float64)
    c1 = np.array([list(g.coords)[-1] for g in gvals], dtype=np.float64)
    mid = roads.geometry.interpolate(0.5, normalized=True)
    xm = mid.x.values.astype(np.float64)
    ym = mid.y.values.astype(np.float64)
    x0, y0 = c0[:, 0], c0[:, 1]
    x1, y1 = c1[:, 0], c1[:, 1]

    df = pd.DataFrame({"id": roads["id"].values})

    print("\nCropping rasters by bbox and reprojecting to UTM...")
    t1 = time.time()

    wc_crop = crop_bbox_to_temp(WC_RASTER_IN, city_poly, CRS_PROJ, CLIP_BUFFER, "wc")
    nd_crop = crop_bbox_to_temp(ndvi_raster_path, city_poly, CRS_PROJ, CLIP_BUFFER, "ndvi")

    wc_utm = reproject_to_utm_res(wc_crop, CRS_PROJ, RES_M, "wc", is_categorical=True, force_float32=False)
    nd_utm = reproject_to_utm_res(nd_crop, CRS_PROJ, RES_M, "ndvi", is_categorical=False, force_float32=True)

    print(f"  done ({time.time() - t1:.1f}s)")

    print("\nReading UTM rasters into memory...")
    t1 = time.time()
    with rasterio.open(wc_utm) as src:
        wc = src.read(1)
        t_wc = src.transform
        h, w = wc.shape

    with rasterio.open(nd_utm) as src:
        ndvi = src.read(1).astype(np.float32)
        t_nd = src.transform
        nd_nodata = src.nodata if src.nodata is not None else NDVI_NODATA_DEFAULT

    def to_rc(transform, xs, ys, H, W):
        rr, cc = rasterio.transform.rowcol(transform, xs, ys)
        rr = np.asarray(rr, dtype=np.int32)
        cc = np.asarray(cc, dtype=np.int32)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        return rr, cc

    r0, c0i = to_rc(t_wc, x0, y0, h, w)
    rm, cmi = to_rc(t_wc, xm, ym, h, w)
    r1, c1i = to_rc(t_wc, x1, y1, h, w)

    print("Filling NDVI nodata (nearest valid)...")
    valid = np.isfinite(ndvi) & (ndvi != nd_nodata)
    if (~valid).any():
        invalid = ~valid
        _, inds = distance_transform_edt(invalid, return_indices=True)
        ndvi = ndvi[tuple(inds)]

    print("\nComputing WorldCover proportions (A1/A2) + entropy (A3) ...")
    class_codes = list(WC_CLASSES.keys())
    for radius in BUFFER_RADII:
        bk = f"buf{radius}"
        rad_px = int(np.ceil(radius / RES_M))
        win = 2 * rad_px + 1

        P = np.zeros((N, len(class_codes)), dtype=np.float32)

        for j, code in enumerate(class_codes):
            m = (wc == code).astype(np.float32)
            p = uniform_filter(m, size=win, mode="nearest")
            P[:, j] = sample3(p, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)
            df[f"lc_wc_{WC_CLASSES[code]}_{bk}_ratio"] = P[:, j]
            del m, p

        green_idx = [i for i, c in enumerate(class_codes) if c in GREEN_CODES]
        blue_idx = [i for i, c in enumerate(class_codes) if c in BLUE_CODES]
        imperv_idx = [i for i, c in enumerate(class_codes) if c in IMPERV_CODES]

        df[f"lc_green_share_{bk}_ratio"] = P[:, green_idx].sum(axis=1) if green_idx else 0.0
        df[f"lc_blue_share_{bk}_ratio"] = P[:, blue_idx].sum(axis=1) if blue_idx else 0.0
        df[f"lc_impervious_share_{bk}_ratio"] = P[:, imperv_idx].sum(axis=1) if imperv_idx else 0.0

        df[f"lc_wc_entropy_{bk}_val"] = shannon_entropy_from_props(P)
        del P

    print("\nComputing NDVI stats (B1) ...")
    for radius in BUFFER_RADII:
        bk = f"buf{radius}"
        rad_px = int(np.ceil(radius / RES_M))
        win = 2 * rad_px + 1

        nd_mean = uniform_filter(ndvi, size=win, mode="nearest")
        nd2 = ndvi * ndvi
        nd_m2 = uniform_filter(nd2, size=win, mode="nearest")
        nd_std = np.sqrt(np.maximum(nd_m2 - nd_mean * nd_mean, 0.0)).astype(np.float32)

        nd_p10 = percentile_filter(ndvi, percentile=10, size=win, mode="nearest").astype(np.float32)
        nd_med = percentile_filter(ndvi, percentile=50, size=win, mode="nearest").astype(np.float32)
        nd_p90 = percentile_filter(ndvi, percentile=90, size=win, mode="nearest").astype(np.float32)

        df[f"green_ndvi_{bk}_mean"] = sample3(nd_mean, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)
        df[f"green_ndvi_{bk}_median"] = sample3(nd_med, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)
        df[f"green_ndvi_{bk}_std"] = sample3(nd_std, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)
        df[f"green_ndvi_{bk}_p10"] = sample3(nd_p10, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)
        df[f"green_ndvi_{bk}_p90"] = sample3(nd_p90, r0, c0i, rm, cmi, r1, c1i).astype(np.float32)

        del nd_mean, nd2, nd_m2, nd_std, nd_p10, nd_med, nd_p90

    df["green_ndvi_gradient_multi_val"] = (df["green_ndvi_buf200_mean"] - df["green_ndvi_buf500_mean"])

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')
    print(f"\nSaved: {OUTPUT_PATH}")
