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
# Population Demand Features
# =====================================================================
def population_demand_features(base_dir, city_name, pop_raster_path):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    CLIPPED_RASTER = os.path.join(shp_dir, "pop_clipped.tif")
    REPROJECTED_RASTER = os.path.join(shp_dir, "pop_clipped_utm.tif")
    BOUNDARY_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")
    OUTPUT_PATH = os.path.join(data_dir, "pop_demand_features.csv")

    BUFFER_RADII = [50, 200, 500]
    CLIP_BUFFER = 2000
    DST_RES = 100
    NODATA_VAL = -99999.0

    t0 = time.time()
    print("Loading road network...")
    gdf = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    if gdf.crs is None:
        raise ValueError("Road GeoJSON has no CRS.")
    CRS_PROJ = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(CRS_PROJ)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    N = len(gdf)
    print(f"  {N:,} segments ({time.time() - t0:.1f}s)")

    mid = gdf.geometry.interpolate(0.5, normalized=True)
    mid_x = mid.x.values.astype(np.float64)
    mid_y = mid.y.values.astype(np.float64)

    print("Loading and clipping WorldPop raster to city extent...")
    t1 = time.time()

    boundary = gpd.read_file(BOUNDARY_PATH)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    CRS_PROJ = boundary.estimate_utm_crs()
    boundary = boundary.to_crs(CRS_PROJ)

    city_extent = boundary.union_all().buffer(CLIP_BUFFER)

    with rasterio.open(pop_raster_path) as src:
        raster_crs = src.crs
        if raster_crs is None: raise ValueError("Population raster has no CRS.")
        src_nodata = src.nodata if src.nodata is not None else NODATA_VAL

    city_extent_raster_crs = (
        gpd.GeoDataFrame(geometry=[city_extent], crs=CRS_PROJ)
        .to_crs(raster_crs)
        .geometry.iloc[0]
    )

    with rasterio.open(pop_raster_path) as src:
        out_image, out_transform = rio_mask(
            src, [mapping(city_extent_raster_crs)], crop=True, nodata=src_nodata, all_touched=True, filled=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "nodata": src_nodata, "count": 1
        })
        with rasterio.open(CLIPPED_RASTER, "w", **out_meta) as dst:
            dst.write(out_image[0], 1)

    print(f"  clipped raster saved: {CLIPPED_RASTER} ({time.time() - t1:.1f}s)")

    print("Reprojecting clipped raster to UTM (count-conserving)...")
    t1 = time.time()
    with rasterio.open(CLIPPED_RASTER) as src:
        transform, width, height = calculate_default_transform(
            src.crs, CRS_PROJ, src.width, src.height, *src.bounds, resolution=DST_RES
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": CRS_PROJ, "transform": transform, "width": width, "height": height,
            "nodata": src_nodata, "count": 1
        })

        with rasterio.open(REPROJECTED_RASTER, "w", **out_meta) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=CRS_PROJ,
                resampling=Resampling.sum, src_nodata=src_nodata, dst_nodata=src_nodata, init_dest_nodata=True
            )

    print(f"  reprojected raster saved: {REPROJECTED_RASTER} ({time.time() - t1:.1f}s)")

    print("Reading UTM population raster...")
    t1 = time.time()
    with rasterio.open(REPROJECTED_RASTER) as src:
        pop = src.read(1).astype(np.float32)
        aff = src.transform
        nodata = src.nodata if src.nodata is not None else src_nodata

    pop = np.where(pop == nodata, 0.0, pop)
    pop = np.where(np.isfinite(pop), pop, 0.0)
    pop = np.where(pop < 0, 0.0, pop)

    H, W = pop.shape
    xres = float(aff.a)
    yres = float(-aff.e)
    print(f"  raster size: {W} x {H}, res≈{xres:.1f}m ({time.time() - t1:.1f}s)")

    cols = np.floor((mid_x - aff.c) / aff.a).astype(np.int32)
    rows = np.floor((mid_y - aff.f) / aff.e).astype(np.int32)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    print("Computing neighborhood population sums (disk kernels)...")
    t1 = time.time()

    df = pd.DataFrame({"id": gdf["id"].values})

    for radius in BUFFER_RADII:
        rc = max(1, int(np.ceil(radius / xres)))
        yy, xx = np.ogrid[-rc:rc + 1, -rc:rc + 1]
        kernel = ((xx * xx + yy * yy) <= (rc * rc)).astype(np.float32)

        pop_sum = convolve(pop, kernel, mode="constant", cval=0.0)

        dens = np.zeros(N, dtype=np.float32)
        if valid.any():
            circle_area_ha = (np.pi * (radius ** 2)) / 1e4
            dens[valid] = (pop_sum[rows[valid], cols[valid]] / circle_area_ha).astype(np.float32)

        df[f"pop_pop_density_buf{radius}_density"] = dens
        del pop_sum

    print(f"  buffers done ({time.time() - t1:.1f}s)")

    df["pop_pop_gradient_multi_val"] = (
        df["pop_pop_density_buf200_density"] - df["pop_pop_density_buf500_density"]
    ).astype(np.float32)

    col_order = ["id"] + [f"pop_pop_density_buf{r}_density" for r in BUFFER_RADII] + ["pop_pop_gradient_multi_val"]
    df = df[col_order]
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')

    print(f"\nSaved: {OUTPUT_PATH}")
