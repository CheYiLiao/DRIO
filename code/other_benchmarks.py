"""
Simple Benchmark Imputation Methods

These are simple baseline methods for comparison with learned imputation models.
All methods follow a consistent interface: fit_transform(X) where X has NaN for missing values.

Methods included:
1. Mean imputation (simple statistical baseline)
2. Matrix Factorization (neural network-based)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


# =============================================================================
# SIMPLE IMPUTATION METHODS
# =============================================================================

class MeanImputation(TransformerMixin):
    """
    Mean imputation for 3D time series data (N, T, D).

    For each (t, d) position, compute mean across samples N.
    If all samples miss this entry, impute with 0.

    Input shape: (N, T, D) where N=samples, T=time, D=features
    """

    def __init__(self):
        super().__init__()

    def fit_transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Apply mean imputation entry-wise across samples.

        Args:
            X: (N, T, D) array with NaN for missing values

        Returns:
            X_filled: (N, T, D) array with imputed values
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, T, D), got shape {X.shape}")

        X_filled = X.copy()
        N, T, D = X.shape

        # For each (t, d) position, compute mean across samples
        for t in range(T):
            for d in range(D):
                values = X[:, t, d]
                mask = np.isnan(values)
                if mask.any():
                    observed = values[~mask]
                    if len(observed) > 0:
                        fill_value = np.mean(observed)
                    else:
                        # All samples miss this entry, impute with 0
                        fill_value = 0.0
                    X_filled[mask, t, d] = fill_value

        return X_filled



# =============================================================================
# NEURAL NETWORK-BASED IMPUTATION METHODS
# =============================================================================

class MF(nn.Module):
    """
    Matrix Factorization model for imputation.

    Learns low-rank embeddings for samples and features,
    then reconstructs missing values as the inner product.
    """

    def __init__(self, sample_num: int, feature_num: int, embedding_size: int):
        super().__init__()
        self.sample_embedding = nn.Embedding(sample_num, embedding_size)
        self.sample_bias = nn.Embedding(sample_num, 1)
        self.feature_embedding = nn.Embedding(feature_num, embedding_size)
        self.feature_bias = nn.Embedding(feature_num, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict values for (sample_idx, feature_idx) pairs."""
        sample_embed = self.sample_embedding(x[:, 0])
        feature_embed = self.feature_embedding(x[:, 1])
        score = (sample_embed * feature_embed).sum(dim=1, keepdim=True)
        score = score + self.sample_bias(x[:, 0]) + self.feature_bias(x[:, 1])
        return score.squeeze()


class MFImputation(TransformerMixin):
    """
    Matrix Factorization imputation for 3D time series data (N, T, D).

    Flattens to 2D (N, T*D) and learns low-rank embeddings for samples and
    features to reconstruct the data matrix. Good for data with latent structure.

    Input shape: (N, T, D) where N=samples, T=time, D=features

    Args:
        embedding_size: Dimension of embeddings
        lr: Learning rate
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        reg: L2 regularization weight
        device: Device to use ('cuda' or 'cpu')
    """

    def __init__(
        self,
        embedding_size: int = 8,
        lr: float = 1e-2,
        n_epochs: int = 500,
        batch_size: int = 512,
        reg: float = 1e-3,
        device: Optional[str] = None
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg = reg

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None

    def fit_transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Fit MF model and impute missing values.

        Args:
            X: (N, T, D) array with NaN for missing values

        Returns:
            X_filled: (N, T, D) array with imputed values
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, T, D), got shape {X.shape}")

        N, T, D = X.shape

        # Flatten to 2D: (N, T*D)
        X_flat = X.reshape(N, T * D)
        n, d = X_flat.shape

        # Create model for this data size
        self.model = MF(n, d, self.embedding_size).to(self.device)

        # Convert to tensor
        X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(self.device)

        # Get indices of observed entries
        mask = ~torch.isnan(X_tensor)
        idx = torch.nonzero(mask, as_tuple=False)  # (num_observed, 2)

        # Get observed values
        observed_values = X_tensor[mask]

        # Adjust batch size if needed
        batch_size = min(self.batch_size, len(idx) // 2)
        batch_size = max(batch_size, 1)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            # Sample batch of observed entries
            perm = torch.randperm(len(idx))[:batch_size]
            batch_idx = idx[perm]
            batch_values = observed_values[perm]

            # Forward pass
            pred = self.model(batch_idx)

            # MSE loss
            loss = nn.functional.mse_loss(pred, batch_values)

            # Regularization
            reg_loss = (
                self.model.sample_embedding(batch_idx[:, 0]).pow(2).mean() +
                self.model.feature_embedding(batch_idx[:, 1]).pow(2).mean() +
                self.model.sample_bias(batch_idx[:, 0]).pow(2).mean() +
                self.model.feature_bias(batch_idx[:, 1]).pow(2).mean()
            )
            loss = loss + self.reg * reg_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reconstruct full matrix
        self.model.eval()
        with torch.no_grad():
            sample_idx = torch.arange(n, device=self.device)
            feature_idx = torch.arange(d, device=self.device)

            sample_embed = self.model.sample_embedding(sample_idx)  # (n, emb)
            feature_embed = self.model.feature_embedding(feature_idx)  # (d, emb)
            sample_bias = self.model.sample_bias(sample_idx)  # (n, 1)
            feature_bias = self.model.feature_bias(feature_idx)  # (d, 1)

            # Reconstruct: (n, d)
            X_reconstructed = torch.matmul(sample_embed, feature_embed.T)
            X_reconstructed = X_reconstructed + sample_bias + feature_bias.T

        # Fill missing values
        X_filled = X_tensor.clone()
        X_filled[~mask] = X_reconstructed[~mask]

        # Reshape back to 3D
        X_filled = X_filled.cpu().numpy().reshape(N, T, D)
        return X_filled


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_simple_imputer(method: str, **kwargs) -> TransformerMixin:
    """
    Factory function to create an imputer by name.

    Args:
        method: Name of imputation method
            - 'mean': Mean imputation
            - 'missforest': MissForest (Random Forest)
            - 'mf': Matrix factorization
        **kwargs: Additional arguments passed to the imputer

    Returns:
        Imputer instance
    """
    imputers = {
        'mean': MeanImputation,
        'mf': MFImputation,
    }

    if method.lower() not in imputers:
        raise ValueError(f"Unknown imputation method: {method}. "
                        f"Available: {list(imputers.keys())}")

    return imputers[method.lower()](**kwargs)
