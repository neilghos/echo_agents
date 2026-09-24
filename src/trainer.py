import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import GraphData
from src.losses import EchoChamberLoss, LossOutput


def train_echo_chamber_model(
    model: nn.Module,
    data: GraphData,
    loss_fn: Optional[EchoChamberLoss] = None,
    num_epochs: int = 200,
    lr: float = 0.015,
    weight_decay: float = 1e-4,
    theta_V: float = 0.05,
    theta_E: float = 0.3,
    verbose: bool = False,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Self-supervised training loop for the Echo Chamber Detection neural model.
    Minimizes the continuous relaxation of structural isolation R subject to
    homogeneity, extremism, and size constraints.
    """
    if loss_fn is None:
        loss_fn = EchoChamberLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)

    best_R = float("inf")
    best_loss = float("inf")
    best_weights = None
    history = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass: predict soft membership p in [0, 1]
        p = model(data)

        # Compute unsupervised loss
        out = loss_fn(
            p=p,
            edge_index=data.edge_index,
            d_out=data.d_out,
            opinions=data.opinions,
            global_mean=data.global_mean,
            theta_V=theta_V,
            theta_E=theta_E,
        )

        out.total_loss.backward()
        optimizer.step()
        scheduler.step()

        metrics = out.to_dict()
        metrics["epoch"] = epoch
        history.append(metrics)

        # Track weights that best minimize structural isolation while feasible
        is_feasible = (out.var_S <= theta_V * 1.15) and (out.effective_size >= 9.0)
        if is_feasible and (out.loss_isolation.item() < best_R):
            best_R = out.loss_isolation.item()
            best_weights = copy.deepcopy(model.state_dict())
        elif best_weights is None and (out.total_loss.item() < best_loss):
            best_loss = out.total_loss.item()
            best_weights = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  [Autograd Ep {epoch+1:02d}] Loss: {metrics['total_loss']:.3f} | "
                f"R: {metrics['loss_isolation']:.3f} | Var: {metrics['var_S']:.4f} | "
                f"Size: {metrics['effective_size']:.1f}"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    return model, history
