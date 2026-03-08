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
import networkx as nx
from shapely import STRtree
from shapely.geometry import Point

warnings.filterwarnings("ignore")


def get_paths(base_dir, city_name):
    """Generate standard directory paths for a city."""
    shp_dir = os.path.join(base_dir, city_name, "shp file")
    data_dir = os.path.join(base_dir, city_name, "data file")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return shp_dir, data_dir


# =====================================================================
# 1. Road Network Download and Standardization
# =====================================================================
def osmnx_roads(base_dir, city_name, osm_query):
    import osmnx as ox

    shp_dir, _ = get_paths(base_dir, city_name)

    print(f'{city_name} downloading, processing ... ...')
    G = ox.graph_from_place(osm_query, network_type='all')

    boundary_path = os.path.join(shp_dir, "Administrative_buildup.geojson")
    boundary = gpd.read_file(boundary_path, encoding='utf-8')

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_edges = gdf_edges[['highway', 'geometry']].copy()
    gdf_edges['highway'] = gdf_edges['highway'].astype(str)
    gdf_edges.index = np.arange(len(gdf_edges))

    # STEP 1: Remove duplicate edges and assign IDs
    def get_line_endpoints(line):
        coords = list(line.coords)
        start = coords[0]  # Start point
        end = coords[-1]   # End point
        # Retain 6 decimal places
        return {
            "from_x": round(start[0], 6),
            "from_y": round(start[1], 6),
            "to_x": round(end[0], 6),
            "to_y": round(end[1], 6)
        }

    endpoints = gdf_edges.geometry.apply(get_line_endpoints)
    endpoints_df = gpd.GeoDataFrame(endpoints.tolist(), index=gdf_edges.index)
    gdf_edges = gdf_edges.join(endpoints_df)
    edges = gdf_edges.copy()

    # Round endpoint coordinates to avoid uniqueness failures due to floating-point errors
    PREC = 7  # Can be changed to 6 (~0.1m level) or 5 (~1m level)
    fx = edges["from_x"].round(PREC)
    fy = edges["from_y"].round(PREC)
    tx = edges["to_x"].round(PREC)
    ty = edges["to_y"].round(PREC)

    # Construct "endpoint keys" (hashable) for uniqueness
    from_key = list(zip(fx, fy))
    to_key = list(zip(tx, ty))

    # Merge all endpoints, generate unique node table
    all_keys = pd.Index(from_key + to_key)
    uniq_keys = all_keys.unique()  # Keep order of unique values
    node_id = pd.Series(np.arange(1, len(uniq_keys) + 1), index=uniq_keys)

    # Map back to edge table to get from/to node IDs
    edges["from"] = pd.Index(from_key).map(node_id).astype("int64")
    edges["to"] = pd.Index(to_key).map(node_id).astype("int64")
    gdf_edges[["from", "to"]] = edges[["from", "to"]]

    # Remove duplicate bidirectional roads
    # Create unique identifier for undirected edges (e.g., small -> large)
    gdf_edges['edge_key'] = gdf_edges.apply(
        lambda row: tuple(sorted([row['from'], row['to']])), axis=1)

    # Deduplicate by undirected edge (keep first in each group)
    gdf_edges = gdf_edges.drop_duplicates(subset='edge_key').drop(columns='edge_key')
    gdf_edges.index = np.arange(len(gdf_edges))
    gdf_edges['id'] = np.arange(len(gdf_edges))
    gdf_edges = gdf_edges[['id', 'from', 'to', 'highway', 'geometry']].copy()

    # STEP 2: Road type standardization
    road_type = ["motorway", "trunk", "primary", "secondary", "tertiary", "pedestrian", "residential",
                 "living_street", 'service', 'bridleway', 'footway', 'path', 'steps', 'track',
                 'corridor', 'elevator', 'cycleway']
    for road in road_type:
        mask = gdf_edges["highway"].astype("string").str.contains(road, case=False, na=False)
        gdf_edges.loc[mask, "highway"] = road

    gdf_edges = gdf_edges[gdf_edges['highway'].isin(road_type)].copy()

    # Ensure gdf_edges and boundary use the same CRS
    if gdf_edges.crs != boundary.crs:
        boundary = boundary.to_crs(gdf_edges.crs)

    # Filter lines in gdf_edges that are within the boundary
    gdf_edges_within_boundary = gdf_edges[gdf_edges.geometry.within(boundary.unary_union)].copy()
    gdf_edges_within_boundary['id'] = np.arange(len(gdf_edges_within_boundary))

    gdf_edges_within_boundary.to_file(os.path.join(shp_dir, 'road_all.geojson'), driver="GeoJSON")
    gdf_edges_within_boundary.to_file(os.path.join(shp_dir, 'road_flows.geojson'), driver="GeoJSON")


# =====================================================================
# 2. Network Topology Features
# =====================================================================
def network_topology_features(base_dir, city_name):
    import networkit as nk

    shp_dir, data_dir = get_paths(base_dir, city_name)
    input_path = os.path.join(shp_dir, "road_all.geojson")
    output_path = os.path.join(data_dir, "net_topology_features.csv")

    BUFFER_RADII = [50, 200, 500]  # meters
    ORIENT_BINS = 36  # 0–180° bins (e.g., 36 => 5 degrees each)

    CHUNK = 30000  # 10k~30k is reasonable; increase to 50k if memory is larger
    BW_SOURCES = 3000  # edge betweenness subset sources (smaller is faster)
    HC_SOURCES = 400   # closeness sources (smaller is faster)

    RT_LIST = [
        "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
        "living_street", "pedestrian", "cycleway", "footway", "path", "service"
    ]

    t0 = time.time()
    print(f"Loading {city_name}...")
    gdf = gpd.read_file(input_path, driver="GeoJSON")
    CRS_PROJ = gdf.estimate_utm_crs()
    if gdf.crs is None:
        raise ValueError("gdf.crs is None, please ensure road_all.geojson has CRS (usually EPSG:4326)")
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(CRS_PROJ)

    gdf["length_m"] = gdf.geometry.length.astype(np.float64)
    N = len(gdf)
    print(f"  {N} segments loaded ({time.time() - t0:.1f}s)")

    # Endpoints / straightness / bearing
    coords0 = np.array([list(geom.coords)[0] for geom in gdf.geometry.values], dtype=np.float64)
    coords1 = np.array([list(geom.coords)[-1] for geom in gdf.geometry.values], dtype=np.float64)
    x0, y0 = coords0[:, 0], coords0[:, 1]
    x1, y1 = coords1[:, 0], coords1[:, 1]

    dx = x1 - x0
    dy = y1 - y0
    eucl = np.sqrt(dx * dx + dy * dy)
    lens = gdf["length_m"].values

    straight = np.zeros(N, dtype=np.float64)
    m = lens > 1e-6
    straight[m] = np.clip(eucl[m] / lens[m], 0, 1)

    seg_circuity = np.ones(N, dtype=np.float64)
    m2 = eucl > 1e-6
    seg_circuity[m2] = lens[m2] / eucl[m2]

    # Bearing uses atan2(dy, dx)
    bearings = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 180.0
    bin_size = 180.0 / ORIENT_BINS
    bearing_bins = np.clip((bearings // bin_size).astype(np.int16), 0, ORIENT_BINS - 1)

    print("Building graph...")
    G = nx.Graph()
    for f, t, l in zip(gdf["from"].values, gdf["to"].values, lens):
        f = int(f); t = int(t)
        if G.has_edge(f, t):
            if l < G[f][t]["weight"]:
                G[f][t]["weight"] = float(l)
        else:
            G.add_edge(f, t, weight=float(l))

    deg = dict(G.degree())
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Vectorized node coordinates (much faster than iterrows)
    nodes_from = pd.DataFrame({"node": gdf["from"].astype(int).values, "x": x0, "y": y0})
    nodes_to = pd.DataFrame({"node": gdf["to"].astype(int).values, "x": x1, "y": y1})
    nodes_df = pd.concat([nodes_from, nodes_to], ignore_index=True).drop_duplicates("node", keep="first")

    nids = nodes_df["node"].values.astype(np.int64)
    nxy = nodes_df[["x", "y"]].values.astype(np.float64)
    ndeg = np.array([deg.get(int(n), 0) for n in nids], dtype=np.int16)

    node_pts = [Point(xy) for xy in nxy]
    node_tree = STRtree(node_pts)

    # Midpoints (for STRtree)
    mid_geoms = gdf.geometry.interpolate(0.5, normalized=True)
    mids = np.column_stack([mid_geoms.x.values, mid_geoms.y.values]).astype(np.float64)
    mid_pts = [Point(xy) for xy in mids]
    mid_tree = STRtree(mid_pts)

    df = pd.DataFrame({"id": gdf["id"].values})

    # Segment-level features
    df["net_betweenness_seg_val"] = 0.0
    df["net_closeness_seg_val"] = np.nan
    df["net_straightness_seg_val"] = straight
    df["net_degree_seg_val"] = np.array(
        [(deg.get(int(f), 0) + deg.get(int(t), 0)) / 2.0 for f, t in zip(gdf["from"].values, gdf["to"].values)],
        dtype=np.float64
    )
    df["net_length_seg_val"] = lens

    # Road type one-hot
    hw = gdf["highway"].astype(str).values
    for rt in RT_LIST:
        df[f"net_rt_{rt}_seg_bin"] = (hw == rt).astype(np.int8)
    df["net_rt_other_seg_bin"] = (~np.isin(hw, RT_LIST)).astype(np.int8)

    # Convert NetworkX -> NetworKit (fast build via COO)
    print("Converting graph to NetworKit...")
    t1 = time.time()

    nx_nodes = list(G.nodes())
    nx2nk = {n: i for i, n in enumerate(nx_nodes)}
    n_nodes = len(nx_nodes)
    m_edges = G.number_of_edges()

    row = np.empty(m_edges, dtype=np.int64)
    col = np.empty(m_edges, dtype=np.int64)
    wgt = np.empty(m_edges, dtype=np.float64)

    for i, (u, v, d) in enumerate(G.edges(data=True)):
        row[i] = nx2nk[u]
        col[i] = nx2nk[v]
        w = d.get("weight", 1.0)
        if w <= 0: w = 1e-6
        wgt[i] = w

    nk_G = nk.GraphFromCoo((wgt, (row, col)), n=n_nodes, directed=False, weighted=True)
    try:
        nk.setNumberOfThreads(max(1, (os.cpu_count() or 8) - 1))
    except Exception:
        pass

    print(f"  converted: {nk_G.numberOfNodes()} nodes, {nk_G.numberOfEdges()} edges ({time.time() - t1:.1f}s)")

    from_arr = gdf["from"].astype(int).values
    to_arr = gdf["to"].astype(int).values

    map_ser = pd.Series(nx2nk)
    u_idx = map_ser.reindex(from_arr).to_numpy()
    v_idx = map_ser.reindex(to_arr).to_numpy()
    mask_uv = (~pd.isna(u_idx)) & (~pd.isna(v_idx))
    u_idx2 = u_idx[mask_uv].astype(np.int64)
    v_idx2 = v_idx[mask_uv].astype(np.int64)

    # Betweenness (approximate node betweenness as proxy for segments)
    print(f"Computing approximate betweenness (EstimateBetweenness nSamples={BW_SOURCES})...")
    t1 = time.time()

    k = min(BW_SOURCES, nk_G.numberOfNodes())
    eb = nk.centrality.EstimateBetweenness(nk_G, nSamples=k, normalized=True, parallel=True)
    eb.run()
    node_bw = np.asarray(eb.scores(), dtype=np.float64)

    bw_vals = np.zeros(len(from_arr), dtype=np.float64)
    bw_vals[mask_uv] = (node_bw[u_idx2] + node_bw[v_idx2]) * 0.5
    vmax = bw_vals.max()
    if vmax > 0:
        bw_vals /= vmax
    df["net_betweenness_seg_val"] = bw_vals
    print(f"  betweenness done ({time.time() - t1:.1f}s)")

    # Closeness
    print(f"Computing approx closeness (ApproxCloseness nSamples={min(HC_SOURCES, nk_G.numberOfNodes())})...")
    t1 = time.time()

    comp = nk.components.ConnectedComponents(nk_G)
    comp.run()
    try:
        comp_vec = np.asarray(comp.getComponentsVector(), dtype=np.int64)
    except Exception:
        comp_vec = np.array([comp.componentOfNode(i) for i in range(nk_G.numberOfNodes())], dtype=np.int64)

    counts = np.bincount(comp_vec)
    largest = int(np.argmax(counts))
    in_lcc = (comp_vec == largest)

    old_to_new = -np.ones(n_nodes, dtype=np.int64)
    old_to_new[in_lcc] = np.arange(in_lcc.sum(), dtype=np.int64)

    keep = in_lcc[row] & in_lcc[col]
    row2 = old_to_new[row[keep]]
    col2 = old_to_new[col[keep]]
    wgt2 = wgt[keep]

    nk_G_lcc = nk.GraphFromCoo((wgt2, (row2, col2)), n=int(in_lcc.sum()), directed=False, weighted=True)

    cc = nk.centrality.ApproxCloseness(
        nk_G_lcc,
        nSamples=min(HC_SOURCES, nk_G_lcc.numberOfNodes()),
        epsilon=0.1,
        normalized=True
    )
    cc.run()
    node_cc_lcc = cc.scores()

    node_cc = np.zeros(n_nodes, dtype=np.float64)
    node_cc[in_lcc] = node_cc_lcc

    cc_vals = np.zeros(len(from_arr), dtype=np.float64)
    cc_vals[mask_uv] = (node_cc[u_idx2] + node_cc[v_idx2]) * 0.5
    vmax = cc_vals.max()
    if vmax > 0:
        cc_vals /= vmax
    df["net_closeness_seg_val"] = cc_vals
    print(f"  closeness done ({time.time() - t1:.1f}s)")

    # Buffer Features (chunked)
    intx50, dead50, rd50 = np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)
    intx200, dead200, rd200 = np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)
    intx500, dead500, rd500 = np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)
    cir500 = np.ones(N, dtype=np.float64)
    ent500 = np.zeros(N, dtype=np.float64)

    print(f"Computing buffer features with CHUNK={CHUNK} ...")
    for radius in BUFFER_RADII:
        t1 = time.time()
        print(f"  Buffer {radius}m ...")
        for start in range(0, N, CHUNK):
            end = min(N, start + CHUNK)
            geom_chunk = gdf.geometry.iloc[start:end]
            buffered = geom_chunk.buffer(radius).values

            lens_chunk = lens[start:end]
            area_ha = (np.pi * radius ** 2 + 2.0 * radius * lens_chunk) / 1e4
            area_km2 = area_ha / 100.0

            # Nodes in buffer
            si_local, ni = node_tree.query(buffered, predicate="intersects")
            if len(si_local) > 0:
                deg_sel = ndeg[ni].astype(np.int16)
                chunk_n = end - start

                total = np.bincount(si_local, minlength=chunk_n).astype(np.float64)
                n_intx = np.bincount(si_local, weights=(deg_sel >= 3).astype(np.int16), minlength=chunk_n).astype(np.float64)
                n_dead = np.bincount(si_local, weights=(deg_sel == 1).astype(np.int16), minlength=chunk_n).astype(np.float64)

                intx_den = np.zeros(chunk_n, dtype=np.float64)
                mA = area_ha > 0
                intx_den[mA] = n_intx[mA] / area_ha[mA]

                dead_ratio = np.zeros(chunk_n, dtype=np.float64)
                mt = total > 0
                dead_ratio[mt] = n_dead[mt] / total[mt]

                if radius == 50:
                    intx50[start:end] = intx_den
                    dead50[start:end] = dead_ratio
                elif radius == 200:
                    intx200[start:end] = intx_den
                    dead200[start:end] = dead_ratio
                else:
                    intx500[start:end] = intx_den
                    dead500[start:end] = dead_ratio

            # Road density + (buf500 extras)
            si_local2, mid_j = mid_tree.query(buffered, predicate="intersects")
            if len(si_local2) > 0:
                chunk_n = end - start
                sumlen_m = np.bincount(si_local2, weights=lens[mid_j], minlength=chunk_n).astype(np.float64)
                rd = np.zeros(chunk_n, dtype=np.float64)
                mB = area_km2 > 0
                rd[mB] = (sumlen_m[mB] / 1000.0) / area_km2[mB]

                if radius == 50:
                    rd50[start:end] = rd
                elif radius == 200:
                    rd200[start:end] = rd
                else:
                    rd500[start:end] = rd

                if radius == 500:
                    cnt = np.bincount(si_local2, minlength=chunk_n).astype(np.float64)
                    sum_c = np.bincount(si_local2, weights=seg_circuity[mid_j], minlength=chunk_n).astype(np.float64)
                    mC = cnt > 0
                    cir = np.ones(chunk_n, dtype=np.float64)
                    cir[mC] = sum_c[mC] / cnt[mC]
                    cir500[start:end] = cir

                    b = bearing_bins[mid_j].astype(np.int16)
                    key = si_local2.astype(np.int64) * ORIENT_BINS + b.astype(np.int64)
                    cnt2 = np.bincount(key, minlength=chunk_n * ORIENT_BINS).reshape(chunk_n, ORIENT_BINS).astype(np.float64)
                    row_sum = cnt2.sum(axis=1)
                    ent = np.zeros(chunk_n, dtype=np.float64)
                    mD = row_sum > 0
                    p = np.zeros_like(cnt2)
                    p[mD] = cnt2[mD] / row_sum[mD, None]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ent[mD] = -np.nansum(np.where(p[mD] > 0, p[mD] * np.log(p[mD]), 0.0), axis=1)
                    ent500[start:end] = ent

        print(f"    done ({time.time() - t1:.1f}s)")

    df["net_intx_density_buf50_density"] = intx50
    df["net_deadend_ratio_buf50_ratio"] = dead50
    df["net_road_density_buf50_density"] = rd50
    df["net_intx_density_buf200_density"] = intx200
    df["net_deadend_ratio_buf200_ratio"] = dead200
    df["net_road_density_buf200_density"] = rd200
    df["net_intx_density_buf500_density"] = intx500
    df["net_deadend_ratio_buf500_ratio"] = dead500
    df["net_road_density_buf500_density"] = rd500
    df["net_circuity_buf500_mean"] = cir500
    df["net_orient_entropy_buf500_val"] = ent500

    cols = [
        "id", "net_betweenness_seg_val", "net_closeness_seg_val", "net_straightness_seg_val",
        "net_degree_seg_val", "net_length_seg_val",
    ] + [f"net_rt_{rt}_seg_bin" for rt in RT_LIST] + [
        "net_rt_other_seg_bin", "net_intx_density_buf50_density", "net_deadend_ratio_buf50_ratio",
        "net_road_density_buf50_density", "net_intx_density_buf200_density", "net_deadend_ratio_buf200_ratio",
        "net_road_density_buf200_density", "net_intx_density_buf500_density", "net_deadend_ratio_buf500_ratio",
        "net_road_density_buf500_density", "net_circuity_buf500_mean", "net_orient_entropy_buf500_val",
    ]

    df = df[cols].copy()
    df.to_csv(output_path, index=False, encoding="utf-8-sig", float_format='%.6f')

    print(f"\nSaved: {output_path}")
    print(f"  {N} segments × {len(df.columns)} columns")
    print(f"  Total time: {time.time() - t0:.1f}s")
