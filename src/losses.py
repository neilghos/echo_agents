from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    loss_isolation: torch.Tensor
    loss_homogeneity: torch.Tensor
    loss_extremism: torch.Tensor
    loss_size: torch.Tensor
    loss_binarization: torch.Tensor
    E_in: float
    E_cut: float
    var_S: float
    mean_S: float
    effective_size: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_loss": float(self.total_loss.item()),
            "loss_isolation": float(self.loss_isolation.item()),
            "loss_homogeneity": float(self.loss_homogeneity.item()),
            "loss_extremism": float(self.loss_extremism.item()),
            "loss_size": float(self.loss_size.item()),
            "loss_binarization": float(self.loss_binarization.item()),
            "E_in": self.E_in,
            "E_cut": self.E_cut,
            "var_S": self.var_S,
            "mean_S": self.mean_S,
            "effective_size": self.effective_size,
        }


class EchoChamberLoss(nn.Module):
    """
    Continuous, differentiable relaxation of the Echo Chamber Detection optimization problem.

    Components:
    1. Structural Isolation (R_soft): E_cut / (E_in + eps)
    2. Opinion Homogeneity: Hinge penalty when opinion variance > theta_V
    3. Opinion Extremism: Hinge penalty when |mean(S) - global_mean| < theta_E
    4. Size Regularization: Floor penalty below min_size (10) and ceiling penalty above max_size
    5. Binarization Regularization: Push soft probabilities p toward crisp {0, 1} decisions
    """

    def __init__(
        self,
        lambda_hom: float = 25.0,
        lambda_ext: float = 15.0,
        lambda_size: float = 2.0,
        lambda_bin: float = 0.5,
        min_size: float = 10.0,
        max_size: float = 250.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_hom = lambda_hom
        self.lambda_ext = lambda_ext
        self.lambda_size = lambda_size
        self.lambda_bin = lambda_bin
        self.min_size = min_size
        self.max_size = max_size
        self.eps = eps

    def forward(
        self,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        d_out: torch.Tensor,
        opinions: torch.Tensor,
        global_mean: float,
        theta_V: float,
        theta_E: float,
    ) -> LossOutput:
        """
        Args:
            p: (N,) soft membership vector in [0, 1]
            edge_index: (2, E) directed edge tensor (source -> target)
            d_out: (N,) out-degree vector
            opinions: (N,) continuous opinion vector in [-1, 1]
            global_mean: float, overall population mean opinion
            theta_V: float, maximum allowable internal opinion variance
            theta_E: float, minimum required opinion deviation from global mean
        """
        # Ensure p is clamped to valid range
        p = torch.clamp(p, min=0.0, max=1.0)
        effective_size = p.sum() + self.eps

        # -------------------------------------------------------------
        # 1. Structural Isolation: R = Cut_Edges / Internal_Edges
        # -------------------------------------------------------------
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # Internal edges: sum_{(u, v)} p_u * p_v
            E_in = torch.sum(p[src] * p[dst])
        else:
            E_in = torch.tensor(0.0, device=p.device)

        # Total outgoing edge mass from S: sum_u (p_u * d_out[u])
        total_out = torch.sum(p * d_out)
        # Outgoing cut edges: total_out - E_in
        E_cut = F.relu(total_out - E_in)

        # Isolation loss ratio
        loss_isolation = E_cut / (E_in + self.eps)

        # -------------------------------------------------------------
        # 2. Opinion Homogeneity Constraint (Variance <= theta_V)
        # -------------------------------------------------------------
        # Weighted mean opinion of group S
        mean_S = torch.sum(p * opinions) / effective_size
        # Weighted variance of group S
        var_S = torch.sum(p * ((opinions - mean_S) ** 2)) / effective_size
        loss_homogeneity = F.relu(var_S - theta_V)

        # -------------------------------------------------------------
        # 3. Opinion Extremism Constraint (|mean_S - global_mean| >= theta_E)
        # -------------------------------------------------------------
        dev = torch.abs(mean_S - global_mean)
        loss_extremism = F.relu(theta_E - dev)

        # -------------------------------------------------------------
        # 4. Group Size Regularization (min_size <= Size <= max_size)
        # -------------------------------------------------------------
        # Minimum size constraint (strictly prevents p -> 0 collapse)
        loss_min_size = F.relu(self.min_size - effective_size)
        # Maximum size constraint (prevents swallowing whole graph)
        loss_max_size = F.relu(effective_size - self.max_size)
        loss_size = loss_min_size + 0.1 * loss_max_size

        # -------------------------------------------------------------
        # 5. Binarization Regularization (p -> 0 or 1)
        # -------------------------------------------------------------
        loss_binarization = torch.mean(p * (1.0 - p))

        # -------------------------------------------------------------
        # Total Weighted Loss
        # -------------------------------------------------------------
        total_loss = (
            loss_isolation
            + self.lambda_hom * loss_homogeneity
            + self.lambda_ext * loss_extremism
            + self.lambda_size * loss_size
            + self.lambda_bin * loss_binarization
        )

        return LossOutput(
            total_loss=total_loss,
            loss_isolation=loss_isolation,
            loss_homogeneity=loss_homogeneity,
            loss_extremism=loss_extremism,
            loss_size=loss_size,
            loss_binarization=loss_binarization,
            E_in=float(E_in.item()),
            E_cut=float(E_cut.item()),
            var_S=float(var_S.item()),
            mean_S=float(mean_S.item()),
            effective_size=float(effective_size.item()),
        )
