import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # First GAT layer
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=32,
            heads=4,
            dropout=0.6
        )

        # Second GAT layer (output)
        self.gat2 = GATConv(
            in_channels=32 * 4,  # heads * out_channels
            out_channels=2,
            heads=1,
            concat=False,
            dropout=0.6
        )

    def forward(self, x, edge_index):
        # Layer 1
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # Layer 2
        x = self.gat2(x, edge_index)

        return x