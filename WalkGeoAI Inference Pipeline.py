"""
WalkGeoAI Inference Pipeline
----------------------------
Loads a pretrained pooled GeoAI model (e.g., trained on NYC + Beijing)
and performs zero-shot spatial transfer inference on a new target city.
Estimates relative pedestrian density and model uncertainty.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import AddRandomWalkPE
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Model Architecture Definitions
# =============================================================================
class TabularEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedGCNLayer(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        row, col = edge_index
        g = torch.sigmoid(self.gate(torch.cat([x[row], x[col]], dim=1)))
        m = g * self.W(x[col])
        out = torch.zeros_like(x)
        out.index_add_(0, row, m)
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1.0)
        out = out / deg.unsqueeze(1)
        return F.relu(self.norm(x + self.dropout(out)))


class GatedGCNModel(nn.Module):
    def __init__(self, in_dim, enc_hidden, hidden, num_layers, dropout):
        super().__init__()
        self.encoder = TabularEncoder(in_dim, enc_hidden, hidden, dropout)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)


# =============================================================================
# 2. Helper Functions
# =============================================================================
def read_and_merge_feature_csvs(feature_dir):
    """Reads and merges all CSV files in the directory based on the 'id' column."""
    csvs = sorted(glob.glob(os.path.join(feature_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV feature files found in {feature_dir}!")

    dfs = []
    for fp in csvs:
        df = pd.read_csv(fp)
        if "id" in df.columns:
            dfs.append(df.loc[:, ~df.columns.duplicated()])

    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="id", how="left")
    return out


# =============================================================================
# 3. Main Inference Function
# =============================================================================
def estimate_new_city_using_pooled_model(base_dir, city_name, ckpt_filename="best_Pooled_Model_active_density.pt"):
    """
    Executes the zero-shot spatial transfer inference on a new city.

    Args:
        base_dir (str): The root directory containing city data and models.
        city_name (str): The name of the target city (e.g., "Philadelphia").
        ckpt_filename (str): The filename of the trained PyTorch model checkpoint.
    """
    print(f"\n{'=' * 70}")
    print(f"[ZERO-SHOT INFERENCE] PREDICTING {city_name.upper()}")
    print(f"{'=' * 70}")

    # 1. Prepare and validate paths
    ckpt_path = os.path.join(base_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}\n"
            f"Please ensure the Pooled model is trained and saved in {base_dir}."
        )

    data_dir = os.path.join(base_dir, city_name, "data file")
    road_path = os.path.join(base_dir, city_name, "shp file", "road_flows.geojson")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4096
    NEIGHBOR_SIZES = [15, 10]

    # 2. Load trained model checkpoint and metadata
    print("Loading Pooled model weights and configuration...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt["config"]
    saved_features = ckpt["feature_cols"]  # Features used during training
    scaler = ckpt["scaler"]  # QuantileTransformer fitted on training data

    # 3. Load target city data
    print(f"Loading road network and features for {city_name}...")
    roads = gpd.read_file(road_path, driver="GeoJSON").drop_duplicates(subset=["id"])

    feat = read_and_merge_feature_csvs(data_dir)
    df = roads[["id", "from", "to"]].merge(feat, on="id", how="left")

    # 4. Feature alignment and normalization
    missing_feats = set(saved_features) - set(df.columns)
    if missing_feats:
        print(f"  -> Warning: {city_name} is missing {len(missing_feats)} features. Padding with 0.")

    for c in missing_feats:
        df[c] = 0

    X = df[saved_features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    # Apply the pre-fitted scaler directly
    X_scaled = np.nan_to_num(scaler.transform(X))

    # 5. Build Graph Topology (Edge Index)
    print("Constructing graph topology (Edge Index)...")
    N = len(df)
    node2segs = {}
    for n, si in zip(np.concatenate([df["from"].astype(int), df["to"].astype(int)]),
                     np.concatenate([np.arange(N), np.arange(N)])):
        node2segs.setdefault(int(n), []).append(int(si))

    edges_u, edges_v = [], []
    for lst in node2segs.values():
        if len(lst) >= 2:
            max_l = min(len(lst), 20)
            for i in range(max_l):
                for j in range(i + 1, max_l):
                    edges_u.extend([lst[i], lst[j]])
                    edges_v.extend([lst[j], lst[i]])

    # 6. PyG Data Assembly and Random Walk PE (RWSE)
    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=torch.tensor([edges_u, edges_v], dtype=torch.long)
    )
    data.num_nodes = N

    print("Computing Random Walk Spatial PE (RWSE)...")
    data = AddRandomWalkPE(walk_length=cfg['rwse_dim'], attr_name='rwse')(data)
    data.x = torch.cat([data.x, data.rwse], dim=1)

    # 7. Model Initialization
    model = GatedGCNModel(
        in_dim=data.x.shape[1],
        enc_hidden=cfg['enc_hidden'],
        hidden=cfg['hidden'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout']
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])

    loader = NeighborLoader(
        data,
        input_nodes=torch.arange(N),
        num_neighbors=NEIGHBOR_SIZES,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 8. Standard Inference
    print("Executing GNN spatial prediction...")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            preds.extend(model(batch.x, batch.edge_index)[:batch.batch_size].cpu().numpy().flatten())

    # Map predictions to a 0-1 uniform distribution
    qt = QuantileTransformer(n_quantiles=min(10000, len(preds)), output_distribution="uniform", random_state=0)
    preds_norm = qt.fit_transform(np.array(preds).reshape(-1, 1)).flatten()

    roads["pred_density"] = preds_norm

    # 9. Monte Carlo (MC) Dropout for Uncertainty Estimation
    print("Executing MC Dropout for uncertainty estimation (30 samples)...")

    def apply_dropout(m):
        """Forces dropout layers to remain active during evaluation for MC sampling."""
        if type(m) == nn.Dropout:
            m.train()

    mc_preds = []
    with torch.no_grad():
        for s in range(30):
            model.eval()
            model.apply(apply_dropout)  # Force dropout on
            pass_p = []
            for batch in loader:
                batch = batch.to(DEVICE)
                pass_p.extend(model(batch.x, batch.edge_index)[:batch.batch_size].cpu().numpy().flatten())
            mc_preds.append(pass_p)
            if (s + 1) % 10 == 0:
                print(f"  -> MC sampling progress: {s + 1}/30 completed")

    # Standard deviation over 30 runs acts as our spatial uncertainty metric
    roads["pred_uncertainty"] = np.array(mc_preds).std(axis=0)

    roads = roads[["id", "pred_density", "pred_uncertainty", "geometry"]].copy()

    # 10. Save Results to GeoJSON
    print("Writing results back to GeoJSON file...")
    roads.to_file(road_path, driver="GeoJSON")
    print(f"\nZero-shot inference for {city_name} completed successfully!")
    print(f"Data updated and saved to: {road_path}")