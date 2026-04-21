import torch
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    print("Preprocessing data...")

    # 1. Mask for valid labels (remove unknown = -1)
    mask = data.y != -1

    x = data.x
    y = data.y

    # 2. Normalize features (important)
    scaler = StandardScaler()
    x = scaler.fit_transform(x.numpy())
    x = torch.tensor(x, dtype=torch.float)

    # 3. Create train/test masks ONLY on known labels
    num_nodes = x.shape[0]

    known_idx = mask.nonzero(as_tuple=True)[0]

    perm = torch.randperm(len(known_idx))

    train_size = int(0.8 * len(known_idx))

    train_idx = known_idx[perm[:train_size]]
    test_idx = known_idx[perm[train_size:]]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print("Preprocessing done!")
    print("Train samples:", train_mask.sum().item())
    print("Test samples:", test_mask.sum().item())

    return x, y, train_mask, test_mask