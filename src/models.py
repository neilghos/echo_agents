from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import GraphData


class GatedFeatureBlock(nn.Module):
    """
    Feature-Gating Block:
    Applies a learned gate g in [0, 1] to filter node feature signals based on
    ideological alignment and neighborhood cohesion.
    """

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.fc_transform = nn.Linear(in_features, hidden_dim)
        self.fc_gate = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.fc_transform(x))
        g = torch.sigmoid(self.fc_gate(x))
        return self.norm(z * g)


class FeatureGatedMLP(nn.Module):
    """
    Fast, lightweight Local Node-Level Feature-Gated Network for Echo Chamber Detection.

    Input features (7 dimensions):
    0: Raw opinion o_v in [-1, 1]
    1: Extremism distance |o_v - global_mean|
    2: In-degree log1p(d_in)
    3: Out-degree log1p(d_out)
    4: Local neighbor mean opinion
    5: Local neighbor opinion std
    6: Local neighbor agreement ratio (fraction of out-neighbors with same opinion sign)

    Output:
    p in [0, 1]^N: Soft assignment probability that each node belongs to the echo chamber.
    """

    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Gated feature projection
        self.input_gate = GatedFeatureBlock(in_features, hidden_dim)

        # Residual hidden layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output projection head (outputs scalar logit per node)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data_or_x: Union[GraphData, torch.Tensor]) -> torch.Tensor:
        """
        Accepts either a GraphData object or raw node feature tensor x of shape (N, 7).
        Returns soft membership probabilities p of shape (N,) in [0, 1].
        """
        if isinstance(data_or_x, GraphData):
            x = data_or_x.x
        elif hasattr(data_or_x, "x"):
            x = data_or_x.x
        else:
            x = data_or_x

        # 1. Gated feature processing
        h = self.input_gate(x)

        # 2. Residual blocks
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))

        # 3. Prediction head
        logits = self.head(h).squeeze(-1)

        # 4. Temperature-scaled sigmoid
        p = torch.sigmoid(logits / self.temperature)
        return p
