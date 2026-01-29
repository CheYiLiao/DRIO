"""
DRIO v2 (Distributionally Robust Imputer Objective) - With CSDI-style Internal Masking

This module implements DRIO v2 with:
1. CSDI-style internal masking during training (self-supervised learning)
2. UNBALANCED Sinkhorn divergence (marginal relaxation)

Key difference from DRIO v1:
- v1: MSE on all observed entries (gt_mask), model sees all observed data
- v2: MSE on artificially masked entries (target_mask), model only sees cond_mask
      This simulates test conditions during training (like CSDI)

Based on Algorithm 1 from the paper with modification:
    min_θ α*R_θ(target_mask) + (1-α) * sup_Z (S_{ε,τ}(Q_Z, P_θ) - γ*C_Z)

where:
- R_θ: reconstruction error on ARTIFICIALLY MASKED entries (not all observed)
- Q_Z: empirical adversary distribution
- P_θ: empirical imputer distribution (complete trajectories)
- C_Z: transport cost penalty
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

# Use geomloss for efficient Sinkhorn computation
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("Warning: geomloss not installed. Install with: pip install geomloss")


# =============================================================================
# INTERNAL MASKING (CSDI-style)
# =============================================================================

def create_random_mask(observed_mask: torch.Tensor, mask_ratio_range: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """
    Create random conditioning mask from observed entries (CSDI-style).

    For each sample in the batch, randomly select a portion of observed entries
    to hide. The model will only see the remaining entries (cond_mask) and
    must predict the hidden entries (target_mask).

    Args:
        observed_mask: Binary mask (B, D, T) where 1=observed, 0=missing
        mask_ratio_range: Tuple (min, max) for uniform sampling of mask ratio

    Returns:
        cond_mask: Binary mask (B, D, T) - what model sees (subset of observed)
    """
    B, D, T = observed_mask.shape
    device = observed_mask.device

    # Random values for masking, scaled by observed_mask
    rand_for_mask = torch.rand_like(observed_mask.float()) * observed_mask.float()
    rand_for_mask = rand_for_mask.reshape(B, -1)  # (B, D*T)

    cond_mask = observed_mask.clone().float()

    for i in range(B):
        # Sample mask ratio uniformly from range (like CSDI samples from [0, 1])
        mask_ratio = np.random.uniform(mask_ratio_range[0], mask_ratio_range[1])

        num_observed = observed_mask[i].sum().item()
        num_to_mask = int(round(num_observed * mask_ratio))

        if num_to_mask > 0 and num_observed > 0:
            # Find indices of top num_to_mask values (random selection among observed)
            sample_rand = rand_for_mask[i]  # (D*T,)
            _, top_indices = sample_rand.topk(num_to_mask)

            # Create flat mask and set selected indices to 0
            flat_cond = cond_mask[i].reshape(-1)
            flat_cond[top_indices] = 0
            cond_mask[i] = flat_cond.reshape(D, T)

    return cond_mask


# =============================================================================
# UNBALANCED SINKHORN LOSS FUNCTIONS
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
    Heuristic from MissingDataOT (Muzellec et al.).
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


def create_unbalanced_sinkhorn_loss(epsilon: float = 0.1, tau: float = 10.0, p: int = 2) -> nn.Module:
    """
    Create an UNBALANCED Sinkhorn divergence loss using geomloss.
    Key: reach=sqrt(tau) enables marginal relaxation (unbalanced OT).
    """
    if not GEOMLOSS_AVAILABLE:
        raise ImportError("geomloss is required. Install with: pip install geomloss")

    blur = epsilon ** (1.0 / p)
    reach = tau ** (1.0 / p)

    return SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        reach=reach,
        debias=True,
        scaling=0.9,
        backend="tensorized"
    )


def reconstruction_loss(
    X_true: torch.Tensor,
    X_pred: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Compute normalized reconstruction loss on masked entries."""
    diff = mask * (X_true - X_pred)
    squared_error = (diff ** 2).sum()
    num_masked = mask.sum()

    if num_masked > 0:
        loss = squared_error / num_masked
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
# DRIO V2 LOSS CLASS
# =============================================================================

class DRIOv2Loss(nn.Module):
    """
    DRIO v2 Loss with UNBALANCED Sinkhorn divergence and internal masking support.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        tau: float = 10.0,
        adaptive_epsilon: bool = True,
        epsilon_quant: float = 0.5,
        epsilon_mult: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.adaptive_epsilon = adaptive_epsilon
        self.epsilon_quant = epsilon_quant
        self.epsilon_mult = epsilon_mult

        if GEOMLOSS_AVAILABLE and not adaptive_epsilon:
            self.sinkhorn_loss = create_unbalanced_sinkhorn_loss(epsilon=epsilon, tau=tau)
        else:
            self.sinkhorn_loss = None

    def _compute_sinkhorn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute UNBALANCED Sinkhorn divergence."""
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y.dim() > 2:
            y = y.reshape(y.size(0), -1)

        if self.adaptive_epsilon:
            epsilon = pick_epsilon(x, y, self.epsilon_quant, self.epsilon_mult)
            blur = epsilon ** 0.5
            reach = self.tau ** 0.5

            if GEOMLOSS_AVAILABLE:
                loss_fn = SamplesLoss(
                    loss="sinkhorn", p=2, blur=blur, reach=reach,
                    debias=True, scaling=0.9, backend="tensorized"
                )
                return loss_fn(x, y)
            else:
                raise ImportError("geomloss required for unbalanced Sinkhorn")
        else:
            if self.sinkhorn_loss is not None:
                return self.sinkhorn_loss(x, y)
            else:
                raise ImportError("geomloss required for unbalanced Sinkhorn")

    def compute_inner_objective(
        self,
        Z_adversary: torch.Tensor,
        X_imputed: torch.Tensor,
        X_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inner maximization objective for adversary update.
        J(Z) = S_{ε,τ}(Q_Z, P_θ) - γ * C_Z
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
        Compute the full DRIO v2 loss for imputer update.

        L(theta) = alpha * R_theta(target_mask) + (1 - alpha) * S_{ε,τ}(Q_Z*, P_theta)

        Key difference from v1: mse_mask is now target_mask (artificially hidden entries),
        not gt_mask (all observed entries).
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
# DRIO V2 TRAINER CLASS
# =============================================================================

class DRIOv2Trainer:
    """
    DRIO v2 Trainer with CSDI-style internal masking.

    Key differences from v1:
    1. Creates random cond_mask from gt_mask each batch (like CSDI's get_randmask)
    2. Model only sees cond_mask entries, must predict target_mask entries
    3. MSE computed on target_mask (artificially hidden), not gt_mask (all observed)
    4. Sinkhorn still computed on complete trajectories

    Parameters:
    -----------
    alpha : float
        Trade-off between reconstruction (α) and robustness (1-α). Default 0.5.
    gamma : float
        Robustness parameter controlling adversary's transport budget. Default 1.0.
    epsilon : float
        Entropic regularization for Sinkhorn. Default 0.1.
    tau : float
        Marginal relaxation for unbalanced OT. Default 10.0.
    inner_steps : int
        Number of gradient ascent steps for adversary. Default 5.
    inner_lr : float
        Learning rate for adversary update. Default 0.01.
    mask_ratio_range : tuple
        Range (min, max) for random mask ratio sampling. Default (0.0, 1.0) like CSDI.
    adaptive_epsilon : bool
        If True, compute epsilon adaptively per batch. Default True.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        tau: float = 10.0,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        mask_ratio_range: Tuple[float, float] = (0.0, 1.0),
        adaptive_epsilon: bool = True,
        epsilon_quant: float = 0.5,
        epsilon_mult: float = 0.05,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.mask_ratio_range = mask_ratio_range
        self.adaptive_epsilon = adaptive_epsilon
        self.epsilon_quant = epsilon_quant
        self.epsilon_mult = epsilon_mult

        # Create DRIO v2 loss module
        self.drio_loss = DRIOv2Loss(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            tau=tau,
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
        J(Z) = S(Q_Z, P_θ) - γ * C_Z
        """
        Z_current = Z.clone().detach().requires_grad_(True)

        # Compute per (d,t) bounds from batch
        batch_min = X_imputed_detached.min(dim=0).values
        batch_max = X_imputed_detached.max(dim=0).values

        center = (batch_min + batch_max) / 2
        half_range = (batch_max - batch_min) / 2
        scaled_half_range = (half_range * 1.05) + 1e-3

        z_min = center - scaled_half_range
        z_max = center + scaled_half_range

        z_min = torch.clamp(z_min, max=-1.645)
        z_max = torch.clamp(z_max, min=1.645)

        inner_obj_history = []

        for k in range(self.inner_steps):
            inner_obj = self.drio_loss.compute_inner_objective(
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

    def _mean_impute(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace missing values with mean per (d,t) position across samples."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)
        obs_count = mask.sum(dim=0, keepdim=True)
        position_mean = obs_sum / (obs_count + 1e-10)
        X_mean = X_obs * mask + (1 - mask) * position_mean
        return X_mean

    def train_step(
        self,
        model: nn.Module,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Complete DRIO v2 training step with CSDI-style internal masking.

        Key differences from v1:
        1. Create cond_mask by randomly masking gt_mask entries
        2. Model input uses cond_mask (not gt_mask)
        3. MSE computed on target_mask = gt_mask - cond_mask (artificially hidden)
        4. Sinkhorn on complete trajectories (using cond_mask for composition)

        Parameters:
        -----------
        model : nn.Module
            Imputer model G_θ
        observed_data : torch.Tensor, shape (B, D, T)
            Observed data
        observed_mask : torch.Tensor, shape (B, D, T)
            Binary mask indicating observed entries
        gt_mask : torch.Tensor, shape (B, D, T)
            Ground truth mask (what we know is observed for this sample)
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters

        Returns:
        --------
        metrics : dict
            Dictionary with training metrics
        """
        model.train()

        # Step 1: Create CSDI-style random conditioning mask
        # cond_mask = subset of gt_mask that model sees
        # target_mask = gt_mask - cond_mask = artificially hidden (for MSE)
        cond_mask = create_random_mask(gt_mask, self.mask_ratio_range)
        target_mask = gt_mask - cond_mask  # Artificially hidden entries

        # Step 2: Generate detached imputation for inner maximization
        # Model only sees cond_mask entries
        with torch.no_grad():
            predictions_detached = model(observed_data, cond_mask)
            # Compose: observed where cond_mask=1, predicted elsewhere
            X_imputed_detached = cond_mask * observed_data + (1 - cond_mask) * predictions_detached

        # Compute mean-imputed data (center of ambiguity set)
        X_mean = self._mean_impute(observed_data, cond_mask)

        # Step 3: Initialize adversary
        Z = self.initialize_adversary(observed_data, observed_mask)

        # Step 4: Inner maximization - find worst-case adversary
        Z_star, inner_info = self.inner_maximization(
            Z=Z,
            X_imputed_detached=X_imputed_detached,
            X_mean=X_mean
        )

        # Step 5: Outer minimization - update imputer
        optimizer.zero_grad()

        # Re-generate imputation with gradient
        # Model still only sees cond_mask
        predictions = model(observed_data, cond_mask)
        X_imputed = cond_mask * observed_data + (1 - cond_mask) * predictions

        # Compute DRIO v2 loss
        # MSE on target_mask (artificially hidden entries) - CSDI-style self-supervised
        # Sinkhorn on X_imputed (complete trajectories) vs Z_star
        loss, loss_dict = self.drio_loss(
            X_true=observed_data,
            X_pred=predictions,
            X_imputed=X_imputed,
            Z_adversary=Z_star,
            mse_mask=target_mask  # KEY DIFFERENCE: MSE on artificially hidden, not all observed
        )

        # Handle edge case: if target_mask is empty (mask_ratio=0), skip this batch
        if target_mask.sum() == 0:
            # No artificially masked entries, use gt_mask as fallback
            loss, loss_dict = self.drio_loss(
                X_true=observed_data,
                X_pred=predictions,
                X_imputed=X_imputed,
                Z_adversary=Z_star,
                mse_mask=gt_mask
            )

        # Backprop and update
        loss.backward()
        optimizer.step()

        # Compile metrics
        metrics = {
            'total_loss': loss_dict['total_loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'sinkhorn_divergence': loss_dict['sinkhorn_divergence'],
            'inner_obj_init': inner_info['inner_obj_init'],
            'inner_obj_final': inner_info['inner_obj_final'],
            'target_mask_ratio': target_mask.sum().item() / gt_mask.sum().item() if gt_mask.sum() > 0 else 0.0,
        }

        return metrics


def create_drio_v2_trainer(
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    tau: float = 10.0,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    mask_ratio_range: Tuple[float, float] = (0.0, 1.0),
    adaptive_epsilon: bool = True,
    epsilon_quant: float = 0.5,
    epsilon_mult: float = 0.05,
) -> DRIOv2Trainer:
    """
    Factory function to create a DRIO v2 trainer.

    Parameters:
    -----------
    alpha : float
        Trade-off between reconstruction and robustness. Default 0.5.
    gamma : float
        Robustness parameter. Default 1.0.
    epsilon : float
        Entropic regularization for Sinkhorn. Default 0.1.
    tau : float
        Marginal relaxation for unbalanced OT. Default 10.0.
    inner_steps : int
        Number of adversary update steps. Default 5.
    inner_lr : float
        Adversary learning rate. Default 0.01.
    mask_ratio_range : tuple
        Range (min, max) for random mask ratio. Default (0.0, 1.0).
    adaptive_epsilon : bool
        If True, compute epsilon adaptively per batch. Default True.

    Returns:
    --------
    trainer : DRIOv2Trainer
        Configured DRIO v2 trainer
    """
    return DRIOv2Trainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        tau=tau,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        mask_ratio_range=mask_ratio_range,
        adaptive_epsilon=adaptive_epsilon,
        epsilon_quant=epsilon_quant,
        epsilon_mult=epsilon_mult
    )
