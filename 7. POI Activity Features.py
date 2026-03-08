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
# POI Activity Features
# =====================================================================
def download_overture_pois(base_dir, city_name):
    """Download Overture Maps POIs via DuckDB."""
    import duckdb

    shp_dir, _ = get_paths(base_dir, city_name)
    out_csv = os.path.join(shp_dir, "POI.csv")
    BORO_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")

    boro = gpd.read_file(BORO_PATH)
    west, south, east, north = boro.total_bounds

    release = "2026-01-21.0"
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    out_csv_sql = out_csv.replace("\\", "/")

    sql = f"""
    INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';
    COPY (
      SELECT id, COALESCE(names.primary, '') AS name, COALESCE(categories.primary, '') AS category_primary,
        COALESCE(basic_category, '') AS basic_category, list_extract(taxonomy.hierarchy, 1) AS cat_l1,
        list_extract(taxonomy.hierarchy, 2) AS cat_l2, taxonomy.primary AS cat_leaf,
        to_json(taxonomy.hierarchy) AS taxonomy_path_json, confidence, ST_Y(geometry) AS lat, ST_X(geometry) AS lon
      FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {west} AND {east} AND bbox.ymin BETWEEN {south} AND {north}
    ) TO '{out_csv_sql}' (HEADER, DELIMITER ',');
    """
    duckdb.connect().execute(sql)
    POI_df = pd.read_csv(out_csv)
    POI_df = POI_df[['name', 'cat_l1', 'cat_l2', 'lat', 'lon']].copy()
    POI_df.to_csv(out_csv, index=False, mode='w+')
    print("Saved to:", out_csv)

def poi_activity_features(base_dir, city_name):
    shp_dir, data_dir = get_paths(base_dir, city_name)

    ROAD_PATH = os.path.join(shp_dir, "road_all.geojson")
    POI_PATH = os.path.join(shp_dir, "POI.csv")
    BOUNDARY_PATH = os.path.join(shp_dir, "Administrative_buildup.geojson")
    OUTPUT_PATH = os.path.join(data_dir, "poi_density_buf200.csv")

    RADIUS = 200.0
    CHUNK = 80_000

    L1_LIST = [
        "services_and_business", "food_and_drink", "shopping", "lifestyle_services", "health_care",
        "travel_and_transportation", "cultural_and_historic", "education", "sports_and_recreation",
        "community_and_government", "arts_and_entertainment", "lodging", "geographic_entities"
    ]
    L2_LIST = [
        "professional_service", "restaurant", "specialty_store", "financial_service", "beauty_service", "doctor",
        "home_service", "automotive_and_ground_transport", "fashion_and_apparel_store", "casual_eatery",
        "real_estate", "religious_organization", "food_and_beverage_store", "business_to_business", "school",
        "dentist", "sports_and_recreation_venue", "community_service", "bar", "beverage_shop", "organization",
        "architectural_landmark", "park", "media_and_news", "sports_and_fitness_instruction", "travel", "spa",
        "private_establishments_and_corporates", "business", "arts_and_crafts_space", "specialty_school",
        "counseling_and_mental_health", "vehicle_dealer", "performing_arts_venue", "medical_center", "pets",
        "physical_therapy", "sports_club_and_league", "hotel", "convenience_store", "hospital",
        "college_university", "chiropractor", "post_office", "body_modification", "optometrist",
        "second_hand_store", "personal_care_service", "massage_therapy", "lounge"
    ]

    USE_OTHER_BIN = True
    OTHER = "other"

    t0 = time.time()
    print("Loading roads...")
    roads = gpd.read_file(ROAD_PATH, driver="GeoJSON")
    if roads.crs is None: raise ValueError("Road file has no CRS.")
    CRS_PROJ = roads.estimate_utm_crs()
    roads = roads.to_crs(CRS_PROJ)

    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads["length_m"] = roads.geometry.length.astype(np.float64)
    N = len(roads)
    lens = roads["length_m"].values
    print(f"  segments: {N:,}  ({time.time() - t0:.1f}s)")

    boro_poly = None
    if os.path.exists(BOUNDARY_PATH):
        boro = gpd.read_file(BOUNDARY_PATH)
        if boro.crs is None: boro = boro.set_crs("EPSG:4326")
        boro = boro.to_crs(CRS_PROJ)
        boro_poly = boro.union_all()
        try: boro_poly = shapely.make_valid(boro_poly)
        except Exception: pass

    print("Loading POIs...")
    poi = pd.read_csv(POI_PATH)

    poi = poi[["cat_l1", "cat_l2", "lat", "lon"]].copy()
    poi = poi.dropna(subset=["lat", "lon"])
    poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
    poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
    poi = poi[np.isfinite(poi["lat"]) & np.isfinite(poi["lon"])].copy()

    poi_gdf = gpd.GeoDataFrame(
        poi, geometry=gpd.points_from_xy(poi["lon"].values, poi["lat"].values), crs="EPSG:4326"
    ).to_crs(CRS_PROJ)

    if boro_poly is not None:
        minx, miny, maxx, maxy = boro_poly.bounds
        idx_bbox = list(poi_gdf.sindex.intersection((minx, miny, maxx, maxy)))
        poi_gdf = poi_gdf.iloc[idx_bbox].copy()
        try:
            hit = poi_gdf.sindex.query(boro_poly, predicate="intersects")
            poi_gdf = poi_gdf.iloc[hit].copy()
        except Exception:
            poi_gdf = poi_gdf[poi_gdf.intersects(boro_poly)].copy()

    poi_gdf = poi_gdf[poi_gdf.geometry.notna() & ~poi_gdf.geometry.is_empty].copy()
    M = len(poi_gdf)
    print(f"  POIs kept: {M:,}")

    l1_cats = L1_LIST + ([OTHER] if USE_OTHER_BIN else [])
    l2_cats = L2_LIST + ([OTHER] if USE_OTHER_BIN else [])

    l1_map = {k: i for i, k in enumerate(l1_cats)}
    l2_map = {k: i for i, k in enumerate(l2_cats)}
    K1, K2 = len(l1_cats), len(l2_cats)

    l1 = poi_gdf["cat_l1"].astype(str).values
    l2 = poi_gdf["cat_l2"].astype(str).values

    l1_code = np.array([l1_map.get(x, l1_map[OTHER] if USE_OTHER_BIN else -1) for x in l1], dtype=np.int32)
    l2_code = np.array([l2_map.get(x, l2_map[OTHER] if USE_OTHER_BIN else -1) for x in l2], dtype=np.int32)

    if not USE_OTHER_BIN:
        keep = (l1_code >= 0) & (l2_code >= 0)
        poi_gdf = poi_gdf.iloc[np.where(keep)[0]].copy()
        l1_code = l1_code[keep]
        l2_code = l2_code[keep]

    print("Building STRtree for POIs...")
    poi_pts = list(poi_gdf.geometry.values)
    tree = STRtree(poi_pts)

    out_l1 = np.zeros((N, K1), dtype=np.float32)
    out_l2 = np.zeros((N, K2), dtype=np.float32)

    print("\nComputing POI densities in 200m buffers (chunked)...")
    gvals = roads.geometry.values

    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        nseg = e - s
        t1 = time.time()

        buf = gpd.GeoSeries(gvals[s:e], crs=CRS_PROJ).buffer(RADIUS, cap_style=1).values

        try:
            si, pi = tree.query(buf, predicate="intersects")
        except Exception:
            pairs = []
            for i, b in enumerate(buf):
                hits = tree.query(b)
                if len(hits):
                    pairs.append((np.full(len(hits), i, dtype=np.int32), np.array(hits, dtype=np.int32)))
            if len(pairs) == 0: continue
            si = np.concatenate([p[0] for p in pairs])
            pi = np.concatenate([p[1] for p in pairs])

        if len(si) == 0: continue

        L = lens[s:e].astype(np.float64)
        area_ha = (np.pi * RADIUS * RADIUS + 2.0 * RADIUS * L) / 1e4
        area_ha = np.maximum(area_ha, 1e-6)

        l1c = l1_code[pi]
        keys1 = si.astype(np.int64) * K1 + l1c.astype(np.int64)
        cnt1 = np.bincount(keys1, minlength=nseg * K1).reshape(nseg, K1).astype(np.float64)
        out_l1[s:e, :] = (cnt1 / area_ha[:, None]).astype(np.float32)

        l2c = l2_code[pi]
        keys2 = si.astype(np.int64) * K2 + l2c.astype(np.int64)
        cnt2 = np.bincount(keys2, minlength=nseg * K2).reshape(nseg, K2).astype(np.float64)
        out_l2[s:e, :] = (cnt2 / area_ha[:, None]).astype(np.float32)

        print(f"  chunk {s:,}-{e - 1:,}: pairs={len(si):,}  ({time.time() - t1:.1f}s)")

    print("\nBuilding output table...")
    df = pd.DataFrame({"id": roads["id"].values})

    for k, cat in enumerate(l1_cats):
        df[f"poi_l1_{cat}_buf200_density"] = out_l1[:, k]

    for k, cat in enumerate(l2_cats):
        df[f"poi_l2_{cat}_buf200_density"] = out_l2[:, k]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format='%.6f')
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {len(df):,} segments × {df.shape[1]} columns")
    print(f"  Total time: {time.time() - t0:.1f}s")