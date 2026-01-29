"""
BSH_DRIO (Balanced Sinkhorn DRIO) - Ablation Study Module

This module implements DRIO with BALANCED Sinkhorn divergence (no marginal relaxation).
It is completely self-contained and does not modify any existing DRIO code.

Key difference from standard DRIO:
- Uses balanced Sinkhorn (reach=None in geomloss) instead of unbalanced (reach=tau^0.5)
- No tau parameter needed

Based on Algorithm 1 from the paper, but with balanced OT:
    min_θ α*R_θ + (1-α) * sup_Z (S_ε(Q_Z, P_θ) - γ*C_Z)

where S_ε is the BALANCED Sinkhorn divergence (strict mass conservation).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

# Use geomloss for efficient Sinkhorn computation
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("Warning: geomloss not installed. Install with: pip install geomloss")


# =============================================================================
# BALANCED SINKHORN LOSS FUNCTIONS
# =============================================================================

def pick_epsilon(
    x: torch.Tensor,
    y: torch.Tensor,
    quant: float = 0.5,
    mult: float = 0.05,
    max_points: int = 500
) -> float:
    """
    Select epsilon adaptively based on pairwise distances in the batch.

    Same heuristic as in loss.py (from MissingDataOT).
    """
    combined = torch.cat([x, y], dim=0)
    n = combined.shape[0]
    if n > max_points:
        idx = torch.randperm(n, device=combined.device)[:max_points]
        combined = combined[idx]

    dists = ((combined[:, None] - combined) ** 2).sum(dim=2) / 2.0
    dists = dists.flatten()
    dists = dists[dists > 0]

    if len(dists) > 0:
        epsilon = torch.quantile(dists, quant).item() * mult
    else:
        epsilon = 0.1

    return max(epsilon, 1e-4)


def create_balanced_sinkhorn_loss(epsilon: float = 0.1, p: int = 2) -> nn.Module:
    """
    Create a BALANCED Sinkhorn divergence loss using geomloss.

    Key difference: reach=None (balanced OT, strict mass conservation)
    vs reach=tau^0.5 (unbalanced OT) in the standard DRIO.
    """
    if not GEOMLOSS_AVAILABLE:
        raise ImportError("geomloss is required. Install with: pip install geomloss")

    blur = epsilon ** (1.0 / p)

    return SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        reach=None,         # BALANCED OT (no marginal relaxation)
        debias=True,
        scaling=0.9,
        backend="tensorized"
    )


def reconstruction_loss(
    X_true: torch.Tensor,
    X_imputed: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Compute normalized reconstruction loss on observed entries."""
    diff = mask * (X_true - X_imputed)
    squared_error = (diff ** 2).sum()
    num_observed = mask.sum()

    if num_observed > 0:
        loss = squared_error / num_observed
    else:
        loss = squared_error

    return loss


def ground_cost(
    X_mean: torch.Tensor,
    Z_adversary: torch.Tensor
) -> torch.Tensor:
    """Compute the ground cost (transport cost) between mean-imputed data and adversary."""
    diff = X_mean - Z_adversary
    cost_per_sample = (diff ** 2).sum(dim=(1, 2))
    cost = cost_per_sample.mean()
    return cost


# =============================================================================
# BSH_DRIO LOSS CLASS
# =============================================================================

class BSH_DRIOLoss(nn.Module):
    """
    Balanced Sinkhorn DRIO Loss.

    Same as DRIOLoss but uses BALANCED Sinkhorn (reach=None).
    No tau parameter.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        adaptive_epsilon: bool = True,
        epsilon_quant: float = 0.5,
        epsilon_mult: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon
        self.epsilon_quant = epsilon_quant
        self.epsilon_mult = epsilon_mult

        if GEOMLOSS_AVAILABLE and not adaptive_epsilon:
            self.sinkhorn_loss = create_balanced_sinkhorn_loss(epsilon=epsilon)
        else:
            self.sinkhorn_loss = None

    def _compute_sinkhorn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute BALANCED Sinkhorn divergence."""
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y.dim() > 2:
            y = y.reshape(y.size(0), -1)

        if self.adaptive_epsilon:
            epsilon = pick_epsilon(x, y, self.epsilon_quant, self.epsilon_mult)
            blur = epsilon ** 0.5

            if GEOMLOSS_AVAILABLE:
                loss_fn = SamplesLoss(
                    loss="sinkhorn", p=2, blur=blur, reach=None,  # BALANCED
                    debias=True, scaling=0.9, backend="tensorized"
                )
                return loss_fn(x, y)
            else:
                raise ImportError("geomloss required for balanced Sinkhorn")
        else:
            if self.sinkhorn_loss is not None:
                return self.sinkhorn_loss(x, y)
            else:
                raise ImportError("geomloss required for balanced Sinkhorn")

    def compute_inner_objective(
        self,
        Z_adversary: torch.Tensor,
        X_imputed: torch.Tensor,
        X_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inner maximization objective for adversary update.
        J(Z) = S_ε(Q_Z, P_θ) - γ * C_Z
        """
        sinkhorn_div = self._compute_sinkhorn(Z_adversary, X_imputed)
        transport_cost = ground_cost(X_mean, Z_adversary)
        objective = sinkhorn_div - self.gamma * transport_cost
        return objective

    def forward(
        self,
        X_true: torch.Tensor,
        X_pred: torch.Tensor,
        X_imputed: torch.Tensor,
        Z_adversary: torch.Tensor,
        mse_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the full BSH_DRIO loss for imputer update.
        L(theta) = alpha * R_theta + (1 - alpha) * S_ε(Q_Z*, P_theta)
        """
        recon_loss = reconstruction_loss(X_true, X_pred, mse_mask)
        sinkhorn_div = self._compute_sinkhorn(Z_adversary, X_imputed)
        total_loss = self.alpha * recon_loss + (1 - self.alpha) * sinkhorn_div

        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'sinkhorn_divergence': sinkhorn_div.item(),
            'alpha': self.alpha,
            'gamma': self.gamma
        }

        return total_loss, loss_dict


# =============================================================================
# BSH_DRIO TRAINER CLASS
# =============================================================================

class BSH_DRIOTrainer:
    """
    Balanced Sinkhorn DRIO Trainer.

    Same as DRIOTrainer but uses balanced Sinkhorn (no tau parameter).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        adaptive_epsilon: bool = True,
        epsilon_quant: float = 0.5,
        epsilon_mult: float = 0.05,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.adaptive_epsilon = adaptive_epsilon
        self.epsilon_quant = epsilon_quant
        self.epsilon_mult = epsilon_mult

        self.bsh_drio_loss = BSH_DRIOLoss(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            adaptive_epsilon=adaptive_epsilon,
            epsilon_quant=epsilon_quant,
            epsilon_mult=epsilon_mult
        )

    def compute_batch_mean(
        self,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature-wise mean from batch for adversary initialization."""
        obs_sum = (observed_data * observed_mask).sum(dim=0)
        obs_count = observed_mask.sum(dim=0)
        batch_mean = obs_sum / (obs_count + 1e-10)
        return batch_mean

    def initialize_adversary(
        self,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Initialize adversarial trajectories at batch mean."""
        B, D, T = observed_data.shape
        batch_mean = self.compute_batch_mean(observed_data, observed_mask)
        Z = batch_mean.unsqueeze(0).expand(B, -1, -1).clone()
        Z.requires_grad_(True)
        return Z

    def inner_maximization(
        self,
        Z: torch.Tensor,
        X_imputed_detached: torch.Tensor,
        X_mean: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Inner maximization: Find worst-case adversary.
        Same as DRIOTrainer but uses balanced Sinkhorn.
        """
        Z_current = Z.clone().detach().requires_grad_(True)

        # Compute per (d,t) min/max bounds from batch, scaled by 1.05
        batch_min = X_imputed_detached.min(dim=0).values
        batch_max = X_imputed_detached.max(dim=0).values
        center = (batch_min + batch_max) / 2
        half_range = (batch_max - batch_min) / 2
        scaled_half_range = half_range * 1.05
        z_min = center - scaled_half_range
        z_max = center + scaled_half_range

        inner_obj_history = []

        for k in range(self.inner_steps):
            inner_obj = self.bsh_drio_loss.compute_inner_objective(
                Z_adversary=Z_current,
                X_imputed=X_imputed_detached,
                X_mean=X_mean
            )

            inner_obj_history.append(inner_obj.item())
            grad_Z = torch.autograd.grad(inner_obj, Z_current, create_graph=False)[0]

            if torch.isnan(grad_Z).any() or torch.isinf(grad_Z).any():
                break

            with torch.no_grad():
                grad_Z = torch.clamp(grad_Z, -1.0, 1.0)
                Z_current = Z_current + self.inner_lr * grad_Z
                Z_current = torch.max(Z_current, z_min.unsqueeze(0))
                Z_current = torch.min(Z_current, z_max.unsqueeze(0))

            Z_current = Z_current.clone().requires_grad_(True)

        Z_star = Z_current.detach()

        info = {
            'inner_obj_init': inner_obj_history[0] if inner_obj_history else 0.0,
            'inner_obj_final': inner_obj_history[-1] if inner_obj_history else 0.0,
            'inner_steps': self.inner_steps
        }

        return Z_star, info

    def compute_bsh_drio_loss(
        self,
        X_true: torch.Tensor,
        X_pred: torch.Tensor,
        X_imputed: torch.Tensor,
        Z_star: torch.Tensor,
        mse_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute BSH_DRIO loss for imputer update."""
        return self.bsh_drio_loss(X_true, X_pred, X_imputed, Z_star, mse_mask)

    def train_step(
        self,
        model: nn.Module,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Complete BSH_DRIO training step (Algorithm 1, one iteration).
        Same structure as DRIOTrainer but uses balanced Sinkhorn.
        """
        model.train()

        # Step 1: Generate detached imputation for inner maximization
        with torch.no_grad():
            predictions_detached = model(observed_data, gt_mask)
            X_imputed_detached = gt_mask * observed_data + (1 - gt_mask) * predictions_detached

        # Compute mean-imputed data (center of ambiguity set)
        X_mean = self._mean_impute(observed_data, gt_mask)

        # Step 2: Initialize adversary
        Z = self.initialize_adversary(observed_data, observed_mask)

        # Step 3: Inner maximization - find worst-case adversary
        Z_star, inner_info = self.inner_maximization(
            Z=Z,
            X_imputed_detached=X_imputed_detached,
            X_mean=X_mean
        )

        # Step 4: Outer minimization - update imputer
        optimizer.zero_grad()
        predictions = model(observed_data, gt_mask)
        X_imputed = gt_mask * observed_data + (1 - gt_mask) * predictions

        loss, loss_dict = self.compute_bsh_drio_loss(
            X_true=observed_data,
            X_pred=predictions,
            X_imputed=X_imputed,
            Z_star=Z_star,
            mse_mask=gt_mask
        )

        loss.backward()
        optimizer.step()

        metrics = {
            'total_loss': loss_dict['total_loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'sinkhorn_divergence': loss_dict['sinkhorn_divergence'],
            'inner_obj_init': inner_info['inner_obj_init'],
            'inner_obj_final': inner_info['inner_obj_final'],
        }

        return metrics

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace missing values with mean per (d,t) position across samples."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)
        obs_count = mask.sum(dim=0, keepdim=True)
        position_mean = obs_sum / (obs_count + 1e-10)
        X_mean = X_obs * mask + (1 - mask) * position_mean
        return X_mean


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_bsh_drio_trainer(
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    adaptive_epsilon: bool = True,
    epsilon_quant: float = 0.5,
    epsilon_mult: float = 0.05,
) -> BSH_DRIOTrainer:
    """
    Factory function to create a Balanced Sinkhorn DRIO trainer.

    Note: No tau parameter (balanced OT uses reach=None).
    """
    return BSH_DRIOTrainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adaptive_epsilon=adaptive_epsilon,
        epsilon_quant=epsilon_quant,
        epsilon_mult=epsilon_mult
    )
