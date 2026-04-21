import torch
import torch.nn.functional as F
from src.model import GAT

def train_model(data, x, y, train_mask):
    print("Starting training...")

    model = GAT(in_channels=x.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Compute class weights (important for imbalance)
    class_counts = torch.bincount(y[train_mask])
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()

    print("Class weights:", weights)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # Training loop
    for epoch in range(30):
        model.train()

        optimizer.zero_grad()

        out = model(x, data.edge_index)

        loss = criterion(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training complete!")
    return model