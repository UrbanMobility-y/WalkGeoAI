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
from shapely.geometry import box, mapping
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import convolve


warnings.filterwarnings("ignore")


def get_paths(base_dir, city_name):
    """Generate standard directory paths for a city."""
    shp_dir = os.path.join(base_dir, city_name, "shp file")
    data_dir = os.path.join(base_dir, city_name, "data file")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return shp_dir, data_dir

# =====================================================================
# Terrain Topography Features
# =====================================================================
def terrain_topography_features(base_dir, city_name, dem_raster_path):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    CLIPPED_DEM = os.path.join(shp_dir, "NASADEM_clipped.tif")
    DEM_UTM = os.path.join(shp_dir, "NASADEM_clipped_utm.tif")
    OUTPUT_PATH = os.path.join(data_dir, "terrain_topography_features.csv")

    CLIP_MARGIN_M = 2000
    DST_RES_M = None
    RADII = [200, 500]
    NODATA_FALLBACK = -99999.0

    t0 = time.time()
    print("Loading road network...")
    gdf = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    CRS_PROJ = gdf.estimate_utm_crs()
    if gdf.crs is None: raise ValueError("Road GeoJSON has no CRS.")
    gdf = gdf.to_crs(CRS_PROJ)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["length_m"] = gdf.geometry.length.astype(np.float64)
    N = len(gdf)
    print(f"  {N:,} segments ({time.time() - t0:.1f}s)")

    coords0 = np.array([list(geom.coords)[0] for geom in gdf.geometry.values], dtype=np.float64)
    coords1 = np.array([list(geom.coords)[-1] for geom in gdf.geometry.values], dtype=np.float64)
    x0, y0 = coords0[:, 0], coords0[:, 1]
    x1, y1 = coords1[:, 0], coords1[:, 1]

    mid = gdf.geometry.interpolate(0.5, normalized=True)
    xm = mid.x.values.astype(np.float64)
    ym = mid.y.values.astype(np.float64)

    lens = gdf["length_m"].values
    lens_safe = np.where(lens > 1e-6, lens, np.nan)

    print("Clipping DEM to city extent (road bbox + margin)...")
    t1 = time.time()

    minx, miny, maxx, maxy = gdf.total_bounds
    clip_poly_utm = box(minx - CLIP_MARGIN_M, miny - CLIP_MARGIN_M, maxx + CLIP_MARGIN_M, maxy + CLIP_MARGIN_M)

    with rasterio.open(dem_raster_path) as src:
        dem_crs = src.crs
        dem_nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK

    clip_poly_dem = gpd.GeoSeries([clip_poly_utm], crs=CRS_PROJ).to_crs(dem_crs).iloc[0]

    with rasterio.open(dem_raster_path) as src:
        out_img, out_tr = rio_mask(src, [mapping(clip_poly_dem)], crop=True, nodata=dem_nodata, filled=True)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2],
            "transform": out_tr, "nodata": dem_nodata, "count": 1
        })
        with rasterio.open(CLIPPED_DEM, "w", **meta) as dst:
            dst.write(out_img[0], 1)

    print(f"  clipped DEM saved ({time.time() - t1:.1f}s)")

    print("Reprojecting DEM to UTM...")
    t1 = time.time()
    with rasterio.open(CLIPPED_DEM) as src:
        if DST_RES_M is None:
            transform, width, height = calculate_default_transform(src.crs, CRS_PROJ, src.width, src.height, *src.bounds)
        else:
            transform, width, height = calculate_default_transform(src.crs, CRS_PROJ, src.width, src.height, *src.bounds, resolution=DST_RES_M)

        meta = src.meta.copy()
        meta.update({
            "crs": CRS_PROJ, "transform": transform, "width": width, "height": height,
            "nodata": dem_nodata, "count": 1
        })

        with rasterio.open(DEM_UTM, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=CRS_PROJ,
                resampling=Resampling.bilinear, src_nodata=dem_nodata, dst_nodata=dem_nodata, init_dest_nodata=True
            )

    print(f"  reprojected DEM saved ({time.time() - t1:.1f}s)")

    print("Computing slope raster (%) ...")
    t1 = time.time()

    with rasterio.open(DEM_UTM) as src:
        dem = src.read(1).astype(np.float32)
        aff = src.transform
        nodata = src.nodata if src.nodata is not None else dem_nodata

    H, W = dem.shape
    xres = float(aff.a)
    yres = float(-aff.e)

    valid_dem = (dem != nodata) & np.isfinite(dem)
    fill_val = float(np.nanmedian(dem[valid_dem])) if valid_dem.any() else 0.0
    dem_f = dem.copy()
    dem_f[~valid_dem] = fill_val

    dz_dy, dz_dx = np.gradient(dem_f, yres, xres)
    slope = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy) * 100.0
    slope = slope.astype(np.float32)
    mask = valid_dem.astype(np.float32)

    print(f"  slope computed ({time.time() - t1:.1f}s), raster {W}x{H}, res≈{xres:.1f}m")

    print("Computing neighborhood stats (disk kernels)...")
    t1 = time.time()

    def disk_kernel(radius_m):
        rcx = int(np.ceil(radius_m / xres))
        rcy = int(np.ceil(radius_m / yres))
        yy, xx = np.ogrid[-rcy:rcy + 1, -rcx:rcx + 1]
        k = (((xx * xres) ** 2 + (yy * yres) ** 2) <= (radius_m ** 2)).astype(np.float32)
        return k

    k500 = disk_kernel(500)
    w500 = convolve(mask, k500, mode="constant", cval=0.0)
    sum_dem_500 = convolve(dem * mask, k500, mode="constant", cval=0.0)
    mean_dem_500 = np.zeros_like(dem, dtype=np.float32)
    m500 = w500 > 0
    mean_dem_500[m500] = (sum_dem_500[m500] / w500[m500]).astype(np.float32)

    def slope_mean_std(radius_m):
        k = disk_kernel(radius_m)
        w = convolve(mask, k, mode="constant", cval=0.0)
        s1 = convolve(slope * mask, k, mode="constant", cval=0.0)
        s2 = convolve((slope * slope) * mask, k, mode="constant", cval=0.0)

        mean = np.zeros_like(slope, dtype=np.float32)
        std = np.zeros_like(slope, dtype=np.float32)
        m = w > 0
        mean[m] = (s1[m] / w[m]).astype(np.float32)
        var = np.zeros_like(slope, dtype=np.float32)
        var[m] = (s2[m] / w[m] - mean[m] * mean[m]).astype(np.float32)
        var = np.maximum(var, 0.0)
        std[m] = np.sqrt(var[m]).astype(np.float32)
        return mean, std

    slope_mean_200, slope_std_200 = slope_mean_std(200)
    slope_mean_500, slope_std_500 = slope_mean_std(500)

    print(f"  neighborhood stats done ({time.time() - t1:.1f}s)")

    def xy_to_rc(x, y):
        c = np.floor((x - aff.c) / aff.a).astype(np.int32)
        r = np.floor((y - aff.f) / aff.e).astype(np.int32)
        return r, c

    r0, c0 = xy_to_rc(x0, y0)
    r1, c1 = xy_to_rc(x1, y1)
    rm, cm = xy_to_rc(xm, ym)

    def sample_array(arr, r, c):
        out = np.full(len(r), np.nan, dtype=np.float32)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out[ok] = arr[r[ok], c[ok]]
        return out

    z0 = sample_array(dem, r0, c0)
    z1 = sample_array(dem, r1, c1)
    zm = sample_array(dem, rm, cm)

    mean_z500_m = sample_array(mean_dem_500, rm, cm)
    smean200_m = sample_array(slope_mean_200, rm, cm)
    sstd200_m = sample_array(slope_std_200, rm, cm)
    smean500_m = sample_array(slope_mean_500, rm, cm)
    sstd500_m = sample_array(slope_std_500, rm, cm)

    df = pd.DataFrame({"id": gdf["id"].values})

    seg_slope = np.abs(z1 - z0) / lens_safe * 100.0
    seg_slope = np.where(np.isfinite(seg_slope), seg_slope, 0.0).astype(np.float32)

    df["terrain_slope_seg_val"] = seg_slope
    df["terrain_elevation_seg_val"] = np.where(np.isfinite(zm), zm, 0.0).astype(np.float32)
    df["terrain_rel_elevation_seg_val"] = np.where(
        np.isfinite(zm) & np.isfinite(mean_z500_m),
        (zm - mean_z500_m),
        0.0
    ).astype(np.float32)

    df["terrain_slope_buf200_mean"] = np.where(np.isfinite(smean200_m), smean200_m, 0.0).astype(np.float32)
    df["terrain_slope_buf200_std"] = np.where(np.isfinite(sstd200_m), sstd200_m, 0.0).astype(np.float32)
    df["terrain_slope_buf500_mean"] = np.where(np.isfinite(smean500_m), smean500_m, 0.0).astype(np.float32)
    df["terrain_slope_buf500_std"] = np.where(np.isfinite(sstd500_m), sstd500_m, 0.0).astype(np.float32)

    col_order = [
        "id", "terrain_slope_seg_val", "terrain_elevation_seg_val", "terrain_rel_elevation_seg_val",
        "terrain_slope_buf200_mean", "terrain_slope_buf200_std", "terrain_slope_buf500_mean", "terrain_slope_buf500_std",
    ]
    df = df[col_order]
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {N:,} segments × {len(df.columns)} columns")
    print(f"  Total time: {time.time() - t0:.1f}s")
