"""
PSW-I (Proximal Spectrum Wasserstein Imputation) Wrapper

Adapts PSW-I from benchmark_psw to work with our unified training framework.

PSW-I is an optimal transport-based imputation method that uses spectral (FFT) distance
and unbalanced OT for time-series imputation.

Reference: FMLYD/PSW-I GitHub repository
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.base import TransformerMixin

# Add benchmark_psw to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_PSW_DIR = PROJECT_ROOT / 'benchmark_psw'


class MVAImputation(TransformerMixin):
    """
    Moving Window Average Imputation for initialization.
    Adapted from benchmark_psw/model_others.py.
    """
    def __init__(self, window_length=7, window_type='exponential', min_periods=1):
        super().__init__()
        self.window_length = window_length
        self.window_type = window_type
        self.min_periods = min_periods

    def fit_transform(self, X, *args, **kwargs):
        import pandas as pd
        X_df = pd.DataFrame(X)
        X_roll = X_df.rolling(
            window=self.window_length,
            min_periods=self.min_periods,
            center=True,
            win_type=self.window_type
        ).mean()
        X_df.fillna(X_roll, inplace=True)
        # Fill remaining NaN with column mean
        X_df.fillna(X_df.mean(), inplace=True)
        # If still NaN (all-NaN columns), fill with 0
        X_df.fillna(0, inplace=True)
        return X_df.to_numpy()


class PSWImputer(TransformerMixin):
    """
    PSW-I (Proximal Spectrum Wasserstein Imputation).

    Adapted from benchmark_psw/benchmark.py OTImputationIni class.

    Key features:
    - Uses FFT-based spectral distance for the OT cost matrix
    - Supports multiple OT solvers: sinkhorn, emd, uot, uot_mm (default)
    - Uses MVA (Moving Window Average) for initialization
    - Early stopping based on validation MAE

    Args:
        lr: Learning rate for imputed values optimization
        n_epochs: Maximum number of epochs
        batch_size: Batch size for OT computation
        n_pairs: Number of batch pairs per gradient update
        noise: Noise scale for initialization
        reg_sk: Sinkhorn regularization (for sinkhorn solver)
        numItermax: Max iterations for OT solver
        stopThr: Convergence threshold for OT solver
        normalize: Whether to normalize cost matrix (0 or 1)
        seq_length: Window size for time-series subsequences
        distance: Distance type for cost matrix ('fft', 'time', 'fft_mag', 'fft_mag_abs')
        ot_type: OT solver type ('sinkhorn', 'emd', 'uot', 'uot_mm')
        reg_m: KL divergence strength for unbalanced OT
        dropout: Dropout probability for spectral features
        mva_kernel: Kernel size for MVA initialization
        early_stop_patience: Patience for early stopping
        device: Device to use ('cuda' or 'cpu')
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 200,
        batch_size: int = 256,
        n_pairs: int = 2,
        noise: float = 1e-4,
        reg_sk: float = 0.005,
        numItermax: int = 1000,
        stopThr: float = 1e-6,
        normalize: int = 1,
        seq_length: int = 24,
        distance: str = 'fft',
        ot_type: str = 'uot_mm',
        reg_m: float = 1.0,
        dropout: float = 0.0,
        mva_kernel: int = 7,
        early_stop_patience: int = 10,
        device: str = 'cuda',
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize
        self.seq_length = seq_length
        self.distance = distance
        self.ot_type = ot_type
        self.reg_m = reg_m
        self.dropout = nn.Dropout(p=dropout)
        self.mva_kernel = mva_kernel
        self.early_stop_patience = early_stop_patience
        self.device = device

        # Initialize MVA for initial imputation
        self.initializer = MVAImputation(
            window_length=mva_kernel,
            window_type='exponential'
        )

    def _impute_single_sample(self, X_2d, gt_mask_2d=None, X_gt_2d=None, verbose=False,
                               internal_val_ratio=0.1):
        """
        Impute a single 2D time series (T, D).

        This is the core PSW-I algorithm that operates on a single time series.

        Following the original PSW-I implementation, we create an internal validation
        set by masking additional entries (default 10%) from the observed data.
        This internal validation is used for early stopping without peeking at test data.

        Args:
            X_2d: (T, D) array with observed values and NaN for missing
            gt_mask_2d: (T, D) mask (1=observed, 0=missing), optional
            X_gt_2d: (T, D) ground truth (NOT used for early stopping in fair comparison)
            verbose: Whether to print progress
            internal_val_ratio: Ratio of observed entries to hold out for internal validation

        Returns:
            X_imputed: (T, D) array with imputed values
        """
        import ot

        n, d = X_2d.shape

        # Create mask: True means missing (to impute)
        if gt_mask_2d is not None:
            mask = (gt_mask_2d == 0)
        else:
            mask = np.isnan(X_2d)

        # If no missing values, return as-is
        if not mask.any():
            return X_2d.copy().astype(np.float32)

        # Create internal validation set (following original PSW-I design)
        # Mask additional entries from observed data for early stopping
        observed_entries = ~mask
        n_observed = observed_entries.sum()
        n_internal_val = int(n_observed * internal_val_ratio)

        if n_internal_val > 0:
            # Randomly select entries from observed data for internal validation
            observed_indices = np.where(observed_entries.flatten())[0]
            internal_val_indices = np.random.choice(observed_indices, n_internal_val, replace=False)
            internal_val_mask = np.zeros(n * d, dtype=bool)
            internal_val_mask[internal_val_indices] = True
            internal_val_mask = internal_val_mask.reshape(n, d)

            # Training mask: original mask + internal validation entries
            train_mask = mask | internal_val_mask

            # Store original observed values for internal validation
            X_observed_vals = X_2d.copy()
        else:
            # Not enough observed entries for internal validation
            train_mask = mask
            internal_val_mask = np.zeros((n, d), dtype=bool)
            X_observed_vals = X_2d.copy()

        # Initialize with MVA using training mask (not seeing internal val)
        X_init = X_2d.copy()
        X_init[train_mask] = np.nan
        X_init = self.initializer.fit_transform(X_init)

        # Handle any remaining NaN after MVA
        while np.any(np.isnan(X_init)):
            temp_mask = np.isnan(X_init)
            X_init[temp_mask] = self.initializer.fit_transform(X_init)[temp_mask]

        # Convert to torch
        X_init = torch.tensor(X_init).double().to(self.device)
        train_mask_torch = torch.tensor(train_mask).to(self.device)

        # Initialize imputed values (for all entries we need to impute: train_mask)
        imps = (self.noise * torch.randn(train_mask.shape, device=self.device).double() + X_init)[train_mask_torch]
        imps.requires_grad = True

        optimizer = torch.optim.Adam([imps], lr=self.lr)

        # Adjust batch size if necessary (matching original: batch_size > n // 2)
        batch_size = self.batch_size
        if batch_size > n // 2:
            e = int(np.log2(n // 2)) if n > 2 else 0
            batch_size = max(2, 2 ** e)

        # Adjust seq_length if data is shorter
        actual_seq_length = min(self.seq_length, n)

        # Early stopping tracking
        best_val_mae = float('inf')
        tick = 0
        best_imps = imps.detach().clone()

        for epoch in range(self.n_epochs):
            # Fill in current imputations
            X_filled = X_init.detach().clone()
            X_filled[train_mask_torch] = imps.double()

            loss = torch.tensor(0.0, device=self.device, dtype=torch.double)

            for _ in range(self.n_pairs):
                optimizer.zero_grad()

                # Sample subsequence starting indices (matching original implementation)
                max_idx = n - actual_seq_length
                if max_idx <= 0:
                    idx1 = np.zeros(batch_size, dtype=int)
                    idx2 = np.zeros(batch_size, dtype=int)
                    seq_len = n
                else:
                    idx1 = np.random.choice(max_idx, batch_size, replace=True)
                    idx2 = np.random.choice(max_idx, batch_size, replace=True)
                    seq_len = actual_seq_length

                # Extract subsequences: (batch_size, seq_length, D)
                X1 = torch.stack([X_filled[idx:idx+seq_len] for idx in idx1], dim=0)
                X2 = torch.stack([X_filled[idx:idx+seq_len] for idx in idx2], dim=0)

                # Compute cost matrix based on distance type
                if self.distance == 'time':
                    M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
                elif self.distance == 'fft':
                    X1_fft = torch.fft.rfft(X1.transpose(1, 2))
                    X2_fft = torch.fft.rfft(X2.transpose(1, 2))
                    diff = self.dropout(((X1_fft.flatten(1)[:, None, :]) - X2_fft.flatten(1)[None, :, :]).abs())
                    M = diff.sum(-1)
                elif self.distance == 'fft_mag':
                    X1_fft = torch.fft.rfft(X1.transpose(1, 2)).abs()
                    X2_fft = torch.fft.rfft(X2.transpose(1, 2)).abs()
                    M = ot.dist(X1_fft.flatten(1), X2_fft.flatten(1), metric='sqeuclidean', p=2)
                elif self.distance == 'fft_mag_abs':
                    X1_fft = torch.fft.rfft(X1.transpose(1, 2)).abs()
                    X2_fft = torch.fft.rfft(X2.transpose(1, 2)).abs()
                    M = torch.norm(X1_fft.flatten(1)[:, None, :] - X2_fft.flatten(1)[None, :, :], p=1, dim=2)
                else:
                    raise ValueError(f"Unknown distance type: {self.distance}")

                # Normalize cost matrix
                if self.normalize == 1:
                    M = M / (M.max() + 1e-10)

                # Uniform marginals
                a = torch.ones(batch_size, device=M.device, dtype=torch.double) / batch_size
                b = torch.ones(batch_size, device=M.device, dtype=torch.double) / batch_size

                # Compute OT plan
                if self.ot_type == 'sinkhorn':
                    pi = ot.sinkhorn(a, b, M, reg=self.reg_sk, numItermax=self.numItermax).detach()
                elif self.ot_type == 'emd':
                    pi = ot.emd(a, b, M, numItermax=self.numItermax).detach()
                elif self.ot_type == 'uot':
                    pi = ot.unbalanced.sinkhorn_unbalanced(
                        a, b, M, reg=self.reg_sk, stopThr=self.stopThr,
                        numItermax=self.numItermax, reg_m=self.reg_m
                    ).detach()
                elif self.ot_type == 'uot_mm':
                    pi = ot.unbalanced.mm_unbalanced(
                        a, b, M, reg_m=self.reg_m, c=None, reg=0, div='kl',
                        G0=None, numItermax=self.numItermax, stopThr=self.stopThr
                    ).detach()
                else:
                    raise ValueError(f"Unknown ot_type: {self.ot_type}")

                loss = loss + (pi * M).sum() / self.n_pairs

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    break

            loss.backward()
            optimizer.step()

            # Early stopping based on internal validation MAE
            # (Following original PSW-I: use held-out entries from observed data)
            if internal_val_mask.sum() > 0:
                X_filled_eval = X_init.detach().clone()
                X_filled_eval[train_mask_torch] = imps.detach()
                X_imputed_np = X_filled_eval.cpu().numpy()
                # Compare imputed values on internal validation set with original observed values
                mae = np.abs(X_imputed_np[internal_val_mask] - X_observed_vals[internal_val_mask]).mean()

                if mae < best_val_mae:
                    best_val_mae = mae
                    best_imps = imps.detach().clone()
                    tick = 0
                else:
                    tick += 1

                if tick > self.early_stop_patience:
                    break
            else:
                # No internal validation possible, just keep the last imputations
                best_imps = imps.detach().clone()

        # Use best imputations - but only return imputations for original missing entries (mask)
        X_filled_final = X_init.detach().clone()
        X_filled_final[train_mask_torch] = best_imps
        X_result = X_filled_final.cpu().numpy().astype(np.float32)

        # Restore original observed values (including internal validation entries)
        # We only want to impute the original missing entries, not the internal validation
        X_result[~mask] = X_2d[~mask]

        return X_result

    def fit_transform(self, X, y=None, X_gt=None, gt_mask=None, verbose=True):
        """
        Fit and transform: impute missing values in X.

        For 3D input (N, T, D) with N independent samples, we impute each sample
        independently to avoid subsequences crossing sample boundaries.

        For 2D input (T, D), we treat it as a single time series.

        Args:
            X: Array with observed values and NaN for missing
               - 2D (T, D): single time series
               - 3D (N, T, D): N independent samples, each imputed separately
            y: Ignored (sklearn compatibility)
            X_gt: Ground truth for evaluation (optional, for early stopping)
            gt_mask: Mask indicating observed entries (1=observed, 0=missing)
            verbose: Whether to print progress

        Returns:
            X_imputed: Array with imputed values (same shape as X)
        """
        original_shape = X.shape
        is_3d = len(original_shape) == 3

        if is_3d:
            N, T, D = original_shape
            if verbose:
                print(f"  PSW-I: Input shape (N={N}, T={T}, D={D}) - imputing sample by sample")
                print(f"  seq_length={self.seq_length}, distance={self.distance}, ot_type={self.ot_type}")

            # Impute each sample independently
            X_imputed = np.zeros_like(X, dtype=np.float32)

            for i in range(N):
                X_i = X[i]  # (T, D)
                gt_mask_i = gt_mask[i] if gt_mask is not None else None
                X_gt_i = X_gt[i] if X_gt is not None else None

                X_imputed[i] = self._impute_single_sample(
                    X_i, gt_mask_i, X_gt_i, verbose=False
                )

                if verbose and (i + 1) % max(1, N // 10) == 0:
                    print(f"    Processed {i + 1}/{N} samples")

            if verbose:
                print(f"  PSW-I: Completed imputation for all {N} samples")

            return X_imputed
        else:
            # Single 2D time series
            if verbose:
                n, d = X.shape
                print(f"  PSW-I: Input shape (n={n}, d={d})")
                print(f"  seq_length={self.seq_length}, distance={self.distance}, ot_type={self.ot_type}")

            return self._impute_single_sample(X, gt_mask, X_gt, verbose=verbose)


def create_psw_imputer(config=None, device='cuda'):
    """
    Factory function to create a PSW imputer with given config.

    Args:
        config: Dictionary with PSW configuration (uses defaults if None)
        device: Device to use

    Returns:
        PSWImputer instance
    """
    from config import PSW_CONFIG

    if config is None:
        config = PSW_CONFIG

    return PSWImputer(
        lr=config.get('lr', 0.01),
        n_epochs=config.get('n_epochs', 200),
        batch_size=config.get('batch_size', 256),
        n_pairs=config.get('n_pairs', 2),
        noise=config.get('noise', 1e-4),
        reg_sk=config.get('reg_sk', 0.005),
        numItermax=config.get('numItermax', 1000),
        stopThr=config.get('stopThr', 1e-6),
        normalize=config.get('normalize', 1),
        seq_length=config.get('seq_length', 24),
        distance=config.get('distance', 'fft'),
        ot_type=config.get('ot_type', 'uot_mm'),
        reg_m=config.get('reg_m', 1.0),
        dropout=config.get('dropout', 0.0),
        mva_kernel=config.get('mva_kernel', 7),
        early_stop_patience=config.get('early_stop_patience', 10),
        device=device,
    )


# =============================================================================
# TRAINING AND PREDICTION FUNCTIONS FOR train_test_unified.py
# =============================================================================

def train_psw_model(train_dataset, val_dataset, save_dir, device):
    """
    Train PSW-I model (or rather, save config since PSW is optimization-based).

    PSW-I doesn't have a traditional training phase - it directly optimizes
    imputed values on each dataset. We save the config here.

    Args:
        train_dataset: Training dataset (ImputationDataset)
        val_dataset: Validation dataset (ImputationDataset)
        save_dir: Directory to save model config
        device: Device to use

    Returns:
        None (PSW is applied directly during prediction)
    """
    import json
    from config import PSW_CONFIG

    print(f"\n{'=' * 70}")
    print("PSW-I (Proximal Spectrum Wasserstein Imputation)")
    print("=" * 70)
    print(f"  Learning rate: {PSW_CONFIG['lr']}")
    print(f"  Max epochs: {PSW_CONFIG['n_epochs']}")
    print(f"  Batch size: {PSW_CONFIG['batch_size']}")
    print(f"  Seq length: {PSW_CONFIG['seq_length']}")
    print(f"  Distance: {PSW_CONFIG['distance']}")
    print(f"  OT type: {PSW_CONFIG['ot_type']}")

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(PSW_CONFIG, f, indent=2)

    print(f"\nPSW-I is an optimization-based method (no model training).")
    print(f"Imputation will be performed directly during evaluation.")

    return None


def generate_psw_predictions(dataset, device, split_name=""):
    """
    Generate predictions using PSW-I.

    PSW-I works on time-series data: (N, T, D).

    Args:
        dataset: ImputationDataset instance
        device: Device to use
        split_name: Name of the split for logging

    Returns:
        targets: (N, D, T) - ground truth (transposed for consistency with other models)
        predictions: (N, D, T) - imputed values
        gt_masks: (N, D, T) - gt_mask
        eval_masks: (N, D, T) - held-out entries mask
    """
    from config import PSW_CONFIG

    print(f"\nGenerating PSW-I predictions for {split_name}...")

    # Collect all data from dataset
    N = len(dataset)
    T = dataset[0]['observed_data'].shape[0]  # Time steps
    D = dataset[0]['observed_data'].shape[1]  # Features

    # Stack all data: (N, T, D)
    observed_data = np.stack([dataset[i]['observed_data'] for i in range(N)], axis=0)
    observed_mask = np.stack([dataset[i]['observed_mask'] for i in range(N)], axis=0)
    gt_mask = np.stack([dataset[i]['gt_mask'] for i in range(N)], axis=0)
    original_data = np.stack([dataset[i]['original_data'] for i in range(N)], axis=0)

    # Compute eval_mask
    eval_mask = observed_mask - gt_mask  # (N, T, D)

    print(f"  Input shape: (N={N}, T={T}, D={D})")

    # Create input with NaN for entries to impute (where gt_mask=0)
    X_input = observed_data.copy()
    X_input[gt_mask == 0] = np.nan

    # Create PSW imputer
    device_str = 'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
    imputer = create_psw_imputer(PSW_CONFIG, device=device_str)

    # Run imputation
    # NOTE: Do NOT pass ground truth for early stopping - that would be peeking at test data
    # PSW-I will run for fixed n_epochs without early stopping (fair comparison)
    X_imputed = imputer.fit_transform(
        X=X_input,
        X_gt=None,  # No ground truth for early stopping
        gt_mask=gt_mask,
        verbose=True
    )

    print(f"  Output shape: {X_imputed.shape}")

    # Transpose to (N, D, T) for consistency with other models
    targets = observed_data.transpose(0, 2, 1)        # (N, D, T)
    predictions = X_imputed.transpose(0, 2, 1)        # (N, D, T)
    gt_masks = gt_mask.transpose(0, 2, 1)             # (N, D, T)
    eval_masks = eval_mask.transpose(0, 2, 1)         # (N, D, T)

    return targets, predictions, gt_masks, eval_masks
