import pandas as pd
import torch
from torch_geometric.data import Data

def load_data():
    print("Loading dataset...")

    # Load files
    features = pd.read_csv("data/raw/elliptic_txs_features.csv", header=None)
    edges = pd.read_csv("data/raw/elliptic_txs_edgelist.csv")
    labels = pd.read_csv("data/raw/elliptic_txs_classes.csv")

    print("Loaded CSV files")

    # First column = transaction ID
    tx_ids = features[0].values

    # Create mapping: tx_id → index
    id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    # Remove tx_id column from features
    features = features.drop(columns=[0])

    # Convert features
    x = torch.tensor(features.values, dtype=torch.float)

    # Map edges using id_map
    edges = edges.replace(id_map)

    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # Map labels
    label_map = {"unknown": -1, "1": 1, "2": 0}
    y = labels["class"].map(label_map)
    y = torch.tensor(y.values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    print("Graph created successfully!")
    return data