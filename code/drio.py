"""
DRIO (Distributionally Robust Imputer Objective) - Self-contained Module

This module implements DRIO with UNBALANCED Sinkhorn divergence (marginal relaxation).
It is completely self-contained and does not depend on external archive modules.

Based on Algorithm 1 from the paper:
    min_θ α*R_θ + (1-α) * sup_Z (S_{ε,τ}(Q_Z, P_θ) - γ*C_Z)

where:
- R_θ: reconstruction error on observed entries
- Q_Z: empirical adversary distribution
- P_θ: empirical imputer distribution
- C_Z: transport cost penalty
- α: trade-off between reconstruction and robustness
- γ: robustness parameter (controls adversary's transport budget)
- S_{ε,τ}: UNBALANCED Sinkhorn divergence (reach=sqrt(tau))
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
    reach = tau ** (1.0 / p)  # UNBALANCED OT (marginal relaxation)

    return SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        reach=reach,        # UNBALANCED OT
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
# DRIO LOSS CLASS
# =============================================================================

class DRIOLoss(nn.Module):
    """
    DRIO Loss with UNBALANCED Sinkhorn divergence.

    Uses reach=sqrt(tau) in geomloss for marginal relaxation.
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
            reach = self.tau ** 0.5  # UNBALANCED

            if GEOMLOSS_AVAILABLE:
                loss_fn = SamplesLoss(
                    loss="sinkhorn", p=2, blur=blur, reach=reach,  # UNBALANCED
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
        Compute the full DRIO loss for imputer update.
        L(theta) = alpha * R_theta + (1 - alpha) * S_{ε,τ}(Q_Z*, P_theta)
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
# DRIO TRAINER CLASS
# =============================================================================


class DRIOTrainer:
    """
    DRIO Trainer implementing Algorithm 1 from the paper.

    Handles the alternating optimization:
    1. Inner maximization: Update adversary Z to find worst-case distribution
    2. Outer minimization: Update imputer θ to minimize DRIO objective

    Parameters:
    -----------
    alpha : float
        Trade-off between reconstruction (α) and robustness (1-α). Default 0.5.
    gamma : float
        Robustness parameter controlling adversary's transport budget. Default 1.0.
    epsilon : float
        Entropic regularization for Sinkhorn. Default 0.1.
        Only used when adaptive_epsilon=False.
    tau : float
        Marginal relaxation for unbalanced OT. Default 10.0.
    inner_steps : int
        Number of gradient ascent steps for adversary (K in Algorithm 1). Default 5.
    inner_lr : float
        Learning rate for adversary update (η_ζ in Algorithm 1). Default 0.01.
    adaptive_epsilon : bool
        If True, compute epsilon adaptively per batch using pick_epsilon heuristic
        from MissingDataOT. Default True.
    epsilon_quant : float
        Quantile for adaptive epsilon selection. Default 0.5 (median).
    epsilon_mult : float
        Multiplier for adaptive epsilon selection. Default 0.05.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        tau: float = 10.0,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
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
        self.adaptive_epsilon = adaptive_epsilon
        self.epsilon_quant = epsilon_quant
        self.epsilon_mult = epsilon_mult

        # Create DRIO loss module
        self.drio_loss = DRIOLoss(
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
        """
        Compute feature-wise mean from batch for adversary initialization.

        X_d,t = sum_i X^(i)_{obs,d,t} / sum_j M^(j)_{d,t}

        Parameters:
        -----------
        observed_data : torch.Tensor, shape (B, D, T)
            Observed data (zeros where missing)
        observed_mask : torch.Tensor, shape (B, D, T)
            Binary mask (1=observed, 0=missing)

        Returns:
        --------
        batch_mean : torch.Tensor, shape (D, T)
            Feature-temporal mean values
        """
        # Sum observed values and counts across batch
        obs_sum = (observed_data * observed_mask).sum(dim=0)  # (D, T)
        obs_count = observed_mask.sum(dim=0)  # (D, T)

        # Compute mean (avoid division by zero)
        batch_mean = obs_sum / (obs_count + 1e-10)

        return batch_mean

    def initialize_adversary(
        self,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize adversarial trajectories at batch mean (Algorithm 1, Line 6).

        ζ^(i) ← X_B for all samples i

        Parameters:
        -----------
        observed_data : torch.Tensor, shape (B, D, T)
        observed_mask : torch.Tensor, shape (B, D, T)

        Returns:
        --------
        Z : torch.Tensor, shape (B, D, T)
            Initial adversarial trajectories (requires_grad=True)
        """
        B, D, T = observed_data.shape

        # Compute batch mean
        batch_mean = self.compute_batch_mean(observed_data, observed_mask)  # (D, T)

        # Initialize all adversaries at batch mean
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
        Inner maximization: Find worst-case adversary (Algorithm 1, Lines 9-13).

        For k = 1 to K:
            J(Z_{k-1}) = S_{ε,τ}(Q_{Z_{k-1}}, P_θ) - γ * C_Z
            Z_k = Z_{k-1} + η_ζ * ∇_Z J(Z_{k-1})

        Parameters:
        -----------
        Z : torch.Tensor, shape (B, D, T)
            Initial adversarial trajectories (requires_grad=True)
        X_imputed_detached : torch.Tensor, shape (B, D, T)
            Imputed data (detached from computation graph)
        X_mean : torch.Tensor, shape (B, D, T)
            Mean-imputed data (center of ambiguity set)

        Returns:
        --------
        Z_star : torch.Tensor, shape (B, D, T)
            Worst-case adversarial trajectories
        info : dict
            Dictionary with optimization info (final inner objective, etc.)
        """
        # Clone Z to avoid modifying the original
        Z_current = Z.clone().detach().requires_grad_(True)

        # Compute per (d,t) min/max bounds from batch, scaled by 1.05
        # Shape: (D, T)
        batch_min = X_imputed_detached.min(dim=0).values  # (D, T)
        batch_max = X_imputed_detached.max(dim=0).values  # (D, T)

        # Scale bounds by 1.05 (expand range slightly)
        # Ensure min stays <= max after scaling
        center = (batch_min + batch_max) / 2
        half_range = (batch_max - batch_min) / 2
        scaled_half_range = (half_range * 1.05) + 1e-3

        z_min = center - scaled_half_range  # (D, T)
        z_max = center + scaled_half_range  # (D, T)

        # Ensure stability
        z_min = torch.clamp(z_min, max=-3.0)
        z_max = torch.clamp(z_max, min=3.0)

        inner_obj_history = []

        for k in range(self.inner_steps):
            # Compute inner objective: J(Z) = S(Q_Z, P_θ) - γ * C_Z
            inner_obj = self.drio_loss.compute_inner_objective(
                Z_adversary=Z_current,
                X_imputed=X_imputed_detached,
                X_mean=X_mean
            )

            inner_obj_history.append(inner_obj.item())

            # Compute gradient w.r.t. Z
            grad_Z = torch.autograd.grad(inner_obj, Z_current, create_graph=False)[0]

            # NaN/Inf safeguard
            if torch.isnan(grad_Z).any() or torch.isinf(grad_Z).any():
                break

            # Gradient ascent step with safeguards
            with torch.no_grad():
                # Clip gradients to prevent explosions
                grad_Z = torch.clamp(grad_Z, -1.0, 1.0)

                # Gradient ascent
                Z_current = Z_current + self.inner_lr * grad_Z

                # stability
                Z_current = torch.max(Z_current, z_min.unsqueeze(0))
                Z_current = torch.min(Z_current, z_max.unsqueeze(0))

            Z_current = Z_current.clone().requires_grad_(True)

        # Final Z* (detached)
        Z_star = Z_current.detach()

        info = {
            'inner_obj_init': inner_obj_history[0] if inner_obj_history else 0.0,
            'inner_obj_final': inner_obj_history[-1] if inner_obj_history else 0.0,
            'inner_steps': self.inner_steps
        }

        return Z_star, info

    def compute_drio_loss(
        self,
        X_true: torch.Tensor,
        X_pred: torch.Tensor,
        X_imputed: torch.Tensor,
        Z_star: torch.Tensor,
        mse_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DRIO loss for imputer update (Algorithm 1, Line 18).

        L(θ) = α * R_θ + (1-α) * S_{ε,τ}(Q_{Z*}, P_θ)

        Loss computation:
        - MSE (R_θ): Computed on mse_mask entries using raw predictions
        - Sinkhorn (S_{ε,τ}): Computed on X_imputed (composed) vs Z_star

        Parameters:
        -----------
        X_true : torch.Tensor, shape (B, D, T)
            Ground truth data
        X_pred : torch.Tensor, shape (B, D, T)
            Raw predictions from model (for MSE computation)
        X_imputed : torch.Tensor, shape (B, D, T)
            Composed imputation: gt_mask * X_true + (1-gt_mask) * X_pred (for Sinkhorn)
        Z_star : torch.Tensor, shape (B, D, T)
            Worst-case adversarial trajectories (fixed)
        mse_mask : torch.Tensor, shape (B, D, T)
            Binary mask for MSE computation (gt_mask for training)

        Returns:
        --------
        loss : torch.Tensor
            DRIO loss for backpropagation
        loss_dict : dict
            Dictionary with loss components
        """
        return self.drio_loss(X_true, X_pred, X_imputed, Z_star, mse_mask)

    def train_step(
        self,
        model: nn.Module,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Complete DRIO training step (Algorithm 1, one iteration).

        1. Generate imputation (detached) for inner maximization
        2. Initialize and update adversary Z
        3. Re-generate imputation (in graph) for outer minimization
        4. Compute DRIO loss and update imputer

        Loss computation:
        - MSE (R_θ): Computed on gt_mask entries (supervised training on known values)
        - Sinkhorn (S_{ε,τ}): Computed on complete trajectories (no masking)

        Parameters:
        -----------
        model : nn.Module
            Imputer model G_θ
        observed_data : torch.Tensor, shape (B, D, T)
            Observed data
        observed_mask : torch.Tensor, shape (B, D, T)
            Binary mask indicating observed entries
        gt_mask : torch.Tensor, shape (B, D, T)
            Ground truth mask (subset of observed for training)
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters

        Returns:
        --------
        metrics : dict
            Dictionary with training metrics
        """
        model.train()

        # Step 1: Generate detached imputation for inner maximization (Line 7)
        with torch.no_grad():
            predictions_detached = model(observed_data, gt_mask)
            X_imputed_detached = gt_mask * observed_data + (1 - gt_mask) * predictions_detached

        # Compute mean-imputed data (center of ambiguity set)
        X_mean = self._mean_impute(observed_data, gt_mask)

        # Step 2: Initialize adversary (Line 6)
        Z = self.initialize_adversary(observed_data, observed_mask)

        # Step 3: Inner maximization - find worst-case adversary (Lines 9-14)
        Z_star, inner_info = self.inner_maximization(
            Z=Z,
            X_imputed_detached=X_imputed_detached,
            X_mean=X_mean
        )

        # Step 4: Outer minimization - update imputer (Lines 16-19)
        optimizer.zero_grad()

        # Re-generate imputation with gradient (Line 16)
        predictions = model(observed_data, gt_mask)
        X_imputed = gt_mask * observed_data + (1 - gt_mask) * predictions

        # Compute DRIO loss (Line 18)
        # MSE on gt_mask using raw predictions, Sinkhorn on X_imputed vs Z_star
        loss, loss_dict = self.compute_drio_loss(
            X_true=observed_data,
            X_pred=predictions,      # Raw predictions for MSE
            X_imputed=X_imputed,     # Composed imputation for Sinkhorn
            Z_star=Z_star,
            mse_mask=gt_mask         # MSE on gt_mask entries (proper training)
        )

        # Backprop and update (Line 19)
        loss.backward()
        optimizer.step()

        # Compile metrics
        metrics = {
            'total_loss': loss_dict['total_loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'sinkhorn_divergence': loss_dict['sinkhorn_divergence'],
            'inner_obj_init': inner_info['inner_obj_init'],
            'inner_obj_final': inner_info['inner_obj_final'],
        }

        return metrics

    def _mean_impute(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace missing values with mean per (d,t) position across samples.

        x̄_{d,t} = Σ_i X^(i)_{obs,d,t} / Σ_j M^(j)_{d,t}

        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
        mask : torch.Tensor, shape (B, D, T)

        Returns:
        --------
        X_mean : torch.Tensor, shape (B, D, T)
        """
        # Compute mean for each (d,t) position across batch
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)

        position_mean = obs_sum / (obs_count + 1e-10)  # (1, D, T)

        X_mean = X_obs * mask + (1 - mask) * position_mean

        return X_mean


def create_drio_trainer(
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    tau: float = 10.0,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    adaptive_epsilon: bool = True,
    epsilon_quant: float = 0.5,
    epsilon_mult: float = 0.05,
) -> DRIOTrainer:
    """
    Factory function to create a DRIO trainer.

    Parameters:
    -----------
    alpha : float
        Trade-off between reconstruction and robustness. Default 0.5.
    gamma : float
        Robustness parameter. Default 1.0.
    epsilon : float
        Entropic regularization for Sinkhorn. Default 0.1.
        Only used when adaptive_epsilon=False.
    tau : float
        Marginal relaxation for unbalanced OT. Default 10.0.
    inner_steps : int
        Number of adversary update steps. Default 5.
    inner_lr : float
        Adversary learning rate. Default 0.01.
    adaptive_epsilon : bool
        If True, compute epsilon adaptively per batch. Default True.
    epsilon_quant : float
        Quantile for adaptive epsilon selection. Default 0.5 (median).
    epsilon_mult : float
        Multiplier for adaptive epsilon selection. Default 0.05.

    Returns:
    --------
    trainer : DRIOTrainer
        Configured DRIO trainer
    """
    return DRIOTrainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        tau=tau,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adaptive_epsilon=adaptive_epsilon,
        epsilon_quant=epsilon_quant,
        epsilon_mult=epsilon_mult
    )
