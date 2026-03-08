"""
WalkGeoAI Feature Extraction Pipeline
-------------------------------------
Complete automated pipeline for extracting multi-scale built environment
features for street-level pedestrian density estimation.
"""

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import STRtree

warnings.filterwarnings("ignore")


def get_paths(base_dir, city_name):
    """Generate standard directory paths for a city."""
    shp_dir = os.path.join(base_dir, city_name, "shp file")
    data_dir = os.path.join(base_dir, city_name, "data file")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return shp_dir, data_dir


# =====================================================================
# Building Morphology Features
# =====================================================================
def bldg_morphology_features(base_dir, city_name):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    BLDG_DIR = os.path.join(shp_dir, "3D-GloBFP")
    BORO_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")
    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    WRITE_BLDG_GEOJSON = True
    BLDG_OUT = os.path.join(shp_dir, "3DGloBFP.geojson")
    OUTPUT_PATH = os.path.join(data_dir, "bldg_morphology_features.csv")

    BUFFER_RADII = [50, 200, 500]
    FLOOR_HEIGHT = 3.0
    WALL_DIST = 15.0
    CHUNK = 80_000

    def gini_np(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x) & (x > 0)]
        n = x.size
        if n < 2: return 0.0
        x.sort()
        s = x.sum()
        if s <= 0: return 0.0
        i = np.arange(1, n + 1, dtype=np.float64)
        return (2.0 * (i * x).sum() / (n * s)) - (n + 1.0) / n

    t0 = time.time()

    print("Loading borough boundary...")
    boro = gpd.read_file(BORO_PATH)
    if boro.crs is None: boro = boro.set_crs("EPSG:4326")
    CRS_PROJ = boro.estimate_utm_crs()
    boro = boro.to_crs(CRS_PROJ)
    mask = boro.union_all()
    try:
        mask = shapely.make_valid(mask)
    except Exception:
        pass

    minx, miny, maxx, maxy = mask.bounds
    print("  boundary ready")

    print(f"\nScanning building folder:\n  {BLDG_DIR}")
    shp_files = sorted(glob.glob(os.path.join(BLDG_DIR, "*.shp")))
    if len(shp_files) == 0:
        raise FileNotFoundError(f"No .shp files found in: {BLDG_DIR}")
    print(f"  found {len(shp_files)} shapefiles")

    bldg_parts = []
    for fp in shp_files:
        try:
            g = gpd.read_file(fp)
            if len(g) == 0: continue
            if g.crs is None: g = g.set_crs("EPSG:4326")
            g = g.to_crs(CRS_PROJ)
            g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()

            try:
                g["geometry"] = g.geometry.buffer(0)
                g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
            except Exception:
                pass

            idx_bbox = list(g.sindex.intersection((minx, miny, maxx, maxy)))
            if len(idx_bbox) == 0: continue
            g2 = g.iloc[idx_bbox].copy()

            try:
                idx_hit = g2.sindex.query(mask, predicate="intersects")
                g2 = g2.iloc[idx_hit].copy()
            except Exception:
                g2 = g2[g2.intersects(mask)].copy()

            if len(g2) > 0:
                bldg_parts.append(g2)
                print(f"    kept: {os.path.basename(fp)}  -> {len(g2):,}")

        except Exception as e:
            print(f"    skipped: {os.path.basename(fp)}  ({e})")

    if len(bldg_parts) == 0:
        raise RuntimeError("No buildings intersect the boundary after filtering.")

    print("Merging filtered building grids...")
    bldg = pd.concat(bldg_parts, ignore_index=True)
    bldg = gpd.GeoDataFrame(bldg, geometry="geometry", crs=CRS_PROJ)
    bldg = bldg[bldg.geometry.notna() & ~bldg.geometry.is_empty].copy()
    print(f"  buildings kept: {len(bldg):,}")

    if WRITE_BLDG_GEOJSON:
        print(f"Writing filtered buildings -> {BLDG_OUT}")
        bldg.to_crs(epsg=4326).to_file(BLDG_OUT, driver="GeoJSON")
        print("  building geojson saved")

    print("\nPreparing building attributes & spatial indexes...")
    height_candidates = [
        "height", "Height", "HEIGHT", "bldg_h", "bldg_height", "H_m", "h_m", "h", "HGT",
        "max_h", "mean_h", "levels", "Levels", "floors", "Floors"
    ]
    hcol = None
    for c in height_candidates:
        if c in bldg.columns:
            hcol = c
            break

    if hcol is None:
        print("  WARNING: no height column found -> heights=0, FAR=0")
        h = np.zeros(len(bldg), dtype=np.float32)
    else:
        hv = pd.to_numeric(bldg[hcol], errors="coerce").astype(np.float32).values
        if ("level" in hcol.lower()) or ("floor" in hcol.lower()):
            hv = hv * 3.0
        hv = np.where(np.isfinite(hv) & (hv > 0), hv, np.nan)
        med_h = float(np.nanmedian(hv)) if np.isfinite(hv).any() else 0.0
        hv = np.where(np.isfinite(hv), hv, med_h).astype(np.float32)

        p99 = float(np.quantile(hv, 0.99)) if hv.size > 0 else 0.0
        if p99 > 0:
            hv = np.clip(hv, 0, p99 * 2).astype(np.float32)
        h = hv

    geom = bldg.geometry.values
    a = bldg.geometry.area.astype(np.float64).values
    p = bldg.geometry.length.astype(np.float64).values

    compact = np.zeros_like(a, dtype=np.float64)
    m = p > 1e-6
    compact[m] = (4.0 * np.pi * a[m]) / (p[m] ** 2)
    compact = np.clip(compact, 0, 1).astype(np.float32)

    cent = shapely.centroid(geom)
    cent_tree = STRtree(list(cent))
    poly_tree = STRtree(list(geom))
    print("  done")

    print("\nLoading roads...")
    roads = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    if roads.crs is None: raise ValueError("road_all.geojson has no CRS.")
    roads = roads.to_crs(CRS_PROJ)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()

    N = len(roads)
    roads["length_m"] = roads.geometry.length.astype(np.float64)
    lens = roads["length_m"].values
    print(f"  segments: {N:,}")

    gvals = roads.geometry.values
    c0 = np.array([list(g.coords)[0] for g in gvals], dtype=np.float64)
    c1 = np.array([list(g.coords)[-1] for g in gvals], dtype=np.float64)
    x0, y0 = c0[:, 0], c0[:, 1]
    x1, y1 = c1[:, 0], c1[:, 1]

    mid = roads.geometry.interpolate(0.5, normalized=True)
    xm = mid.x.values.astype(np.float64)
    ym = mid.y.values.astype(np.float64)

    print("\nComputing segment-scale setback & street wall (fast)...")
    setback_mean = np.zeros(N, dtype=np.float32)
    wall_ratio = np.zeros(N, dtype=np.float32)

    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        nseg = e - s

        pts = []
        pts.extend(gpd.points_from_xy(x0[s:e], y0[s:e]))
        pts.extend(gpd.points_from_xy(xm[s:e], ym[s:e]))
        pts.extend(gpd.points_from_xy(x1[s:e], y1[s:e]))
        pts = np.array(pts, dtype=object)

        try:
            qi, bi = poly_tree.query_nearest(pts)
            dist = np.full(len(pts), np.inf, dtype=np.float64)
            dist[qi] = np.array([pts[i].distance(geom[j]) for i, j in zip(qi, bi)], dtype=np.float64)
        except Exception:
            dist = np.array([pt.distance(geom[poly_tree.nearest(pt)]) for pt in pts], dtype=np.float64)

        d0 = dist[0:nseg]
        dm = dist[nseg:2 * nseg]
        d1_ = dist[2 * nseg:3 * nseg]

        dmean = (d0 + dm + d1_) / 3.0
        setback_mean[s:e] = np.clip(dmean, 0, 1e6).astype(np.float32)
        wall_ratio[s:e] = ((d0 <= WALL_DIST).astype(np.float32) +
                           (dm <= WALL_DIST).astype(np.float32) +
                           (d1_ <= WALL_DIST).astype(np.float32)) / 3.0

    print("  segment-scale done")

    df = pd.DataFrame({"id": roads["id"].values})
    df["bldg_street_wall_seg_ratio"] = wall_ratio
    df["bldg_setback_seg_mean"] = setback_mean

    print("\nComputing buffer-scale building morphology (fast)...")

    for radius in BUFFER_RADII:
        t1 = time.time()
        bk = f"buf{radius}"
        print(f"  {bk} ...")

        dens = np.zeros(N, dtype=np.float32)
        covr = np.zeros(N, dtype=np.float32)
        per_dens = np.zeros(N, dtype=np.float32)
        h_mean = np.zeros(N, dtype=np.float32)
        h_std = np.zeros(N, dtype=np.float32)
        h_p90 = np.zeros(N, dtype=np.float32)
        a_med = np.zeros(N, dtype=np.float32)
        far = np.zeros(N, dtype=np.float32)
        comp_mean = np.zeros(N, dtype=np.float32)
        a_gini = np.zeros(N, dtype=np.float32)

        for s in range(0, N, CHUNK):
            e = min(N, s + CHUNK)
            chunkN = e - s

            buf = gpd.GeoSeries(gvals[s:e], crs=CRS_PROJ).buffer(radius, cap_style=1).values
            si, bi = cent_tree.query(buf, predicate="intersects")
            if len(si) == 0:
                continue

            L = lens[s:e].astype(np.float64)
            buf_area_m2 = (np.pi * radius * radius + 2.0 * radius * L)
            buf_area_ha = buf_area_m2 / 1e4

            cnt = np.bincount(si, minlength=chunkN).astype(np.float64)
            ok = cnt > 0

            area_sum = np.bincount(si, weights=a[bi], minlength=chunkN).astype(np.float64)
            per_sum = np.bincount(si, weights=p[bi], minlength=chunkN).astype(np.float64)

            hs = h[bi].astype(np.float64)
            hsum = np.bincount(si, weights=hs, minlength=chunkN).astype(np.float64)
            hsum2 = np.bincount(si, weights=hs * hs, minlength=chunkN).astype(np.float64)

            csum = np.bincount(si, weights=compact[bi].astype(np.float64), minlength=chunkN).astype(np.float64)

            dens[s:e][ok] = (cnt[ok] / buf_area_ha[ok]).astype(np.float32)
            covr[s:e][ok] = (area_sum[ok] / buf_area_m2[ok]).astype(np.float32)
            per_dens[s:e][ok] = (per_sum[ok] / buf_area_ha[ok]).astype(np.float32)

            hm = np.zeros(chunkN, dtype=np.float64)
            hm[ok] = hsum[ok] / cnt[ok]
            var = np.zeros(chunkN, dtype=np.float64)
            var[ok] = hsum2[ok] / cnt[ok] - hm[ok] * hm[ok]
            var = np.maximum(var, 0.0)

            h_mean[s:e] = hm.astype(np.float32)
            h_std[s:e] = np.sqrt(var).astype(np.float32)

            far_num = np.bincount(si, weights=(a[bi] * hs / FLOOR_HEIGHT), minlength=chunkN).astype(np.float64)
            far[s:e][ok] = (far_num[ok] / buf_area_m2[ok]).astype(np.float32)

            cm = np.zeros(chunkN, dtype=np.float64)
            cm[ok] = csum[ok] / cnt[ok]
            comp_mean[s:e] = cm.astype(np.float32)

            tmp = pd.DataFrame({"si": si, "h": hs.astype(np.float32), "a": a[bi].astype(np.float32)})

            q = tmp.groupby("si")["h"].quantile(0.90)
            h_p90[s:e][q.index.values] = q.values.astype(np.float32)

            med = tmp.groupby("si")["a"].median()
            a_med[s:e][med.index.values] = med.values.astype(np.float32)

            g = tmp.groupby("si")["a"].apply(lambda x: gini_np(x.values)).astype(np.float32)
            a_gini[s:e][g.index.values] = g.values

        df[f"bldg_density_{bk}_density"] = dens
        df[f"bldg_coverage_{bk}_ratio"] = covr
        df[f"bldg_perimeter_density_{bk}_density"] = per_dens
        df[f"bldg_height_{bk}_mean"] = h_mean
        df[f"bldg_height_{bk}_std"] = h_std
        df[f"bldg_height_{bk}_p90"] = h_p90
        df[f"bldg_footprint_{bk}_median"] = a_med
        df[f"bldg_far_{bk}_val"] = far
        df[f"bldg_compactness_{bk}_mean"] = comp_mean
        df[f"bldg_area_gini_{bk}_val"] = a_gini

        print(f"    done ({time.time() - t1:.1f}s)")

    col_order = ["id"]
    for r in BUFFER_RADII:
        bk = f"buf{r}"
        col_order += [
            f"bldg_density_{bk}_density", f"bldg_coverage_{bk}_ratio", f"bldg_perimeter_density_{bk}_density",
            f"bldg_height_{bk}_mean", f"bldg_height_{bk}_std", f"bldg_height_{bk}_p90",
            f"bldg_footprint_{bk}_median", f"bldg_far_{bk}_val", f"bldg_compactness_{bk}_mean", f"bldg_area_gini_{bk}_val",
        ]
    col_order += ["bldg_street_wall_seg_ratio", "bldg_setback_seg_mean"]

    df = df[col_order]
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')
    print(f"\nSaved: {OUTPUT_PATH}")
