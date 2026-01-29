#!/usr/bin/env python3
"""
Unified Training and Evaluation Script for Imputation Models

Trains and evaluates multiple imputation models on preprocessed data from pkl files.

Available models:
- csdi: CSDI (Conditional Score-based Diffusion Imputation)
- mdot: MissingDataOT (OTimputer - Algorithm 1)
- brits: BRITS (Bidirectional Recurrent Imputation for Time Series)
- psw: PSW-I (Proximal Spectrum Wasserstein Imputation)
- drio_sttransformer: DRIO with Spatiotemporal Transformer
- drio_lstm: DRIO with LSTM
- drio_gat: DRIO with Graph Attention Network
- drio_mlp: DRIO with MLP
- drio_brits: DRIO with BRITS backbone
- mse_brits: MSE model with BRITS backbone
- bsh_drio_brits: bsh_drio model with BRITS backbone
- drio_v2_brits: DRIO-V2 with BRITS backbone
- notmiwae: Not-MIWAE (Missingness-Informed Variational Autoencoder)
- mf: Matrix Factorization
- mean: Mean Imputation


Usage:
    # Train all models on a dataset
    python train_test_unified.py --data-prefix data/processed/physionet/physionet_mnar_90pct_split70-10-20 --seed 42

    # Train DRIO without CV (use default alpha=0.5, gamma=1.0)
    python train_test_unified.py --data-prefix data/processed/physionet/physionet_mnar_90pct_split70-10-20 --seed 42 --models drio_sttransformer --no-drio-cv

    # Train DRIO with custom hyperparameter grid
    python train_test_unified.py --data-prefix data/processed/physionet/physionet_mnar_90pct_split70-10-20 --seed 42 --models drio_sttransformer --drio-alpha-grid 0.3 0.5 0.7 --drio-gamma-grid 0.5 1.0 2.0

    # List available models
    python train_test_unified.py --list-models
"""

import sys
import os
import json
import argparse
import pickle
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

# Setup paths
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
CSDI_BENCHMARK_DIR = PROJECT_ROOT / 'benchmark_CSDI'
MDOT_BENCHMARK_DIR = PROJECT_ROOT / 'benchmark_MissingDataOT'
BRITS_BENCHMARK_DIR = PROJECT_ROOT / 'benchmark_brits'
NOTMIWAE_BENCHMARK_DIR = PROJECT_ROOT / 'benchmark_notmiwae_torch'

# Only add CODE_DIR to path for our own modules
sys.path.insert(0, str(CODE_DIR))

from model_arch import create_imputer
from eval import evaluate_imputation
from drio import create_drio_trainer
from drio_v2 import create_drio_v2_trainer
from bsh_drio import create_bsh_drio_trainer
from brits_wrapper import BRITSForDRIO
from psw_wrapper import train_psw_model, generate_psw_predictions
from other_benchmarks import create_simple_imputer
from config import (
    AVAILABLE_MODELS,
    MSE_TRAIN_CONFIG,
    MEAN_CONFIG, MF_CONFIG,
    DRIO_MODEL_CONFIGS, DRIO_CONFIG, DRIO_DEFAULT_ALPHA_GRID, DRIO_DEFAULT_GAMMA_GRID,
    get_drio_cv_config, get_drio_epochs, get_drio_standalone_params,
    DRIO_V2_CONFIG, DRIO_V2_BRITS_CONFIG, DRIO_V2_DEFAULT_ALPHA_GRID, DRIO_V2_DEFAULT_GAMMA_GRID,
    BSH_DRIO_MODEL_CONFIGS, BSH_DRIO_CONFIG, BSH_DRIO_DEFAULT_ALPHA_GRID, BSH_DRIO_DEFAULT_GAMMA_GRID,
    CSDI_CONFIG, MDOT_CONFIG, BRITS_CONFIG, SILRTC_CONFIG, DRIO_BRITS_CONFIG, 
    NOTMIWAE_CONFIG, MSE_BRITS_CONFIG, BSH_DRIO_BRITS_CONFIG,
    N_CSDI_SAMPLES,
)


def load_module_from_path(module_name, file_path, add_parent_to_path=False, extra_paths=None):
    """
    Load a Python module from a specific file path.

    This avoids conflicts when multiple benchmark directories have files with the same name
    (e.g., both benchmark_CSDI and benchmark_MissingDataOT have utils.py).

    Args:
        module_name: Name to assign to the loaded module
        file_path: Path to the .py file
        add_parent_to_path: If True, temporarily add the parent directory to sys.path
                           while loading (needed when the module has internal imports
                           like 'from utils import ...')
        extra_paths: List of additional paths to temporarily add to sys.path
                    (useful when imports need to resolve from a different directory)
    """
    import importlib.util

    paths_added = []

    # Add parent directory if requested
    if add_parent_to_path:
        parent_dir = str(Path(file_path).parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            paths_added.append(parent_dir)

    # Add extra paths if provided
    if extra_paths:
        for path in extra_paths:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                paths_added.append(path_str)

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        # Remove the paths after loading to avoid conflicts with other benchmarks
        for path in paths_added:
            if path in sys.path:
                sys.path.remove(path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET CLASS
# =============================================================================

class ImputationDataset(Dataset):
    """
    Dataset for imputation tasks.

    Loads data from pkl files and handles normalization.
    Data shape: (N, T, D) where N=samples, T=time steps, D=features
    """

    def __init__(self, data_dict, feature_means, feature_stds, normalize=True):
        """
        Args:
            data_dict: Dictionary with 'observed_values', 'observed_mask', 'gt_mask'
            feature_means: Array of shape (D,) for normalization
            feature_stds: Array of shape (D,) for normalization
            normalize: Whether to normalize the data
        """
        self.observed_values = data_dict['observed_values'].astype(np.float32)  # (N, T, D)
        self.observed_mask = data_dict['observed_mask'].astype(np.float32)      # (N, T, D)
        self.gt_mask = data_dict['gt_mask'].astype(np.float32)                  # (N, T, D)

        self.feature_means = feature_means.astype(np.float32)
        self.feature_stds = feature_stds.astype(np.float32)
        # Avoid division by zero: replace 0 stds with 1
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.normalize = normalize

        # Store original values for evaluation
        self.original_values = self.observed_values.copy()

        if normalize:
            # Normalize: (x - mean) / std, only for observed entries
            # Shape: (N, T, D), means/stds: (D,)
            self.observed_values = (self.observed_values - self.feature_means) / self.feature_stds
            # Set non-observed entries to 0 after normalization
            self.observed_values = self.observed_values * self.observed_mask

    def __len__(self):
        return len(self.observed_values)

    def __getitem__(self, idx):
        return {
            'observed_data': self.observed_values[idx],      # (T, D)
            'observed_mask': self.observed_mask[idx],        # (T, D)
            'gt_mask': self.gt_mask[idx],                    # (T, D)
            'timepoints': np.arange(self.observed_values.shape[1]).astype(np.float32),  # (T,)
            'original_data': self.original_values[idx],      # (T, D) - for evaluation
        }

    def denormalize(self, data):
        """Denormalize data back to original scale."""
        # data shape: (N, D, T) or (N, T, D)
        if data.shape[-1] == len(self.feature_stds):
            # Shape is (N, T, D)
            return data * self.feature_stds + self.feature_means
        else:
            # Shape is (N, D, T) - transpose, denormalize, transpose back
            data_t = np.transpose(data, (0, 2, 1))  # (N, T, D)
            data_t = data_t * self.feature_stds + self.feature_means
            return np.transpose(data_t, (0, 2, 1))  # (N, D, T)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_prefix, seed):
    """
    Load train/val/test data from pkl files.

    Args:
        data_prefix: Path prefix like 'data/processed/physionet/physionet_mnar_90pct_split70-10-20'
        seed: Random seed used when creating the data

    Returns:
        train_data, val_data, test_data: Dictionaries with data
        metadata: Metadata including feature_means and feature_stds
    """
    data_prefix = Path(data_prefix)

    # Construct file paths
    train_path = Path(str(data_prefix) + f"_train_seed{seed}.pkl")
    val_path = Path(str(data_prefix) + f"_val_seed{seed}.pkl")
    test_path = Path(str(data_prefix) + f"_test_seed{seed}.pkl")

    # Check if files exist
    for path, split_name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if not path.exists():
            raise FileNotFoundError(f"{split_name} data file not found: {path}")

    # Load data
    print(f"\nLoading data from: {data_prefix}")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Get metadata from train data
    metadata = train_data['metadata']

    return train_data, val_data, test_data, metadata


def print_data_summary(train_data, val_data, test_data, metadata):
    """Print summary of loaded data."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"  Dataset:       {metadata['dataset']}")
    print(f"  Missing type:  {metadata['missing_type'].upper()}")
    print(f"  Missing ratio: {metadata['missing_ratio'] * 100:.0f}% (target)")
    print(f"  Actual missing: {metadata['actual_missing_ratio'] * 100:.1f}%")
    print(f"  Data shape:    {metadata['shape']} (N, T, D)")
    print(f"  - N (samples): {metadata['total_samples']}")
    print(f"  - T (time):    {metadata['shape'][1]}")
    print(f"  - D (features): {metadata['shape'][2]}")
    print(f"\n  Split ratios:  train={metadata['train_ratio']*100:.0f}%, "
          f"val={metadata['val_ratio']*100:.0f}%, "
          f"test={metadata['test_ratio']*100:.0f}%")
    print(f"  Train samples: {len(train_data['observed_values'])}")
    print(f"  Val samples:   {len(val_data['observed_values'])}")
    print(f"  Test samples:  {len(test_data['observed_values'])}")
    print("=" * 70)


def create_datasets(train_data, val_data, test_data, metadata):
    """Create PyTorch datasets."""
    feature_means = metadata['feature_means']
    feature_stds = metadata['feature_stds']

    train_dataset = ImputationDataset(train_data, feature_means, feature_stds, normalize=True)
    val_dataset = ImputationDataset(val_data, feature_means, feature_stds, normalize=True)
    test_dataset = ImputationDataset(test_data, feature_means, feature_stds, normalize=True)

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# MSE MODEL TRAINING
# =============================================================================

def train_mse_epoch(model, train_loader, optimizer, device):
    """Train MSE model for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        # Data comes in (B, T, D), model expects (B, D, T)
        observed_data = batch['observed_data'].float().to(device)  # (B, T, D)
        gt_mask = batch['gt_mask'].float().to(device)              # (B, T, D)

        # Transpose to (B, D, T)
        observed_data = observed_data.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(observed_data, gt_mask)

        # MSE loss on gt_mask entries (reconstruction error for training)
        loss = ((predictions - observed_data) ** 2 * gt_mask).sum() / (gt_mask.sum() + 1e-10)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_mse(model, val_loader, device):
    """
    Validate MSE model using self-supervised reconstruction error.

    IMPORTANT: Uses gt_mask (observed entries) for validation, NOT eval_mask (held-out entries).

    This is NOT data leakage because:
    - gt_mask = entries the model sees during training (observed values)
    - eval_mask = held-out entries for final testing (never seen during training/validation)
    - Validating on gt_mask tests the model's ability to reconstruct observed entries
      on NEW samples (validation set), which ensures generalizability without leaking
      test set information.

    This approach:
    1. Avoids test set leakage (eval_mask never used for model selection)
    2. Validates generalization (validation samples are unseen during training)
    3. Is deployable (doesn't require held-out ground truth for validation)
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            observed_data = batch['observed_data'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)  # Observed entries, NOT eval_mask

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            predictions = model(observed_data, gt_mask)

            # Self-supervised validation: reconstruction error on gt_mask entries
            # This validates generalization on NEW samples without using held-out test data
            loss = ((predictions - observed_data) ** 2 * gt_mask).sum() / (gt_mask.sum() + 1e-10)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_mse_model(model_name, model, train_loader, val_loader, save_dir, device):
    """Train MSE model (no early stopping - trains for all epochs)."""
    print(f"\n{'=' * 70}")
    print(f"Training {model_name}")
    print("=" * 70)

    config = MSE_TRAIN_CONFIG

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")

    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(config['epochs']), desc=f"Training {model_name}"):
        train_loss = train_mse_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate at every epoch (for logging only, no early stopping)
        val_loss = validate_mse(model, val_loader, device)
        val_losses.append(val_loss)

        tqdm.write(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Save final model (no early stopping)
    torch.save(model.state_dict(), save_dir / 'model.pth')

    # Save training curves
    np.savez(
        save_dir / 'training_curves.npz',
        train_losses=train_losses,
        val_losses=val_losses,
    )

    print(f"\nTraining complete! Final val loss: {val_losses[-1]:.6f}")

    return model


def generate_mse_predictions(model, data_loader, device):
    """
    Generate predictions using MSE model.

    Returns:
        targets: (N, D, T) - ground truth
        raw_predictions: (N, D, T) - raw model predictions (for train evaluation)
        gt_masks: (N, D, T) - gt_mask
        eval_masks: (N, D, T) - held-out entries mask
    """
    model.eval()

    all_targets = []
    all_raw_predictions = []
    all_gt_masks = []
    all_eval_masks = []

    with torch.no_grad():
        for batch in data_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            eval_mask = observed_mask - gt_mask

            # Raw model predictions (used for train evaluation on gt_mask)
            predictions = model(observed_data, gt_mask)

            all_targets.append(observed_data.cpu().numpy())
            all_raw_predictions.append(predictions.cpu().numpy())
            all_gt_masks.append(gt_mask.cpu().numpy())
            all_eval_masks.append(eval_mask.cpu().numpy())

    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_raw_predictions, axis=0),
        np.concatenate(all_gt_masks, axis=0),
        np.concatenate(all_eval_masks, axis=0),
    )


# =============================================================================
# CSDI MODEL TRAINING
# =============================================================================

def train_csdi_model(train_loader, val_loader, save_dir, device, d_features):
    """Train CSDI model."""
    # Import CSDI modules from benchmark_CSDI explicitly (avoid conflict with MDOT's utils)
    # add_parent_to_path=True needed because main_model.py imports from diff_models
    csdi_main_model = load_module_from_path("csdi_main_model", CSDI_BENCHMARK_DIR / "main_model.py", add_parent_to_path=True)
    csdi_utils = load_module_from_path("csdi_utils", CSDI_BENCHMARK_DIR / "utils.py", add_parent_to_path=True)

    CSDI_Physio = csdi_main_model.CSDI_Physio
    train_csdi = csdi_utils.train

    print(f"\n{'=' * 70}")
    print("Training CSDI")
    print("=" * 70)

    model = CSDI_Physio(CSDI_CONFIG, device, target_dim=d_features).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {CSDI_CONFIG['train']['epochs']}")
    print(f"  Batch size: {CSDI_CONFIG['train']['batch_size']}")

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(CSDI_CONFIG, f, indent=2)

    # Train using CSDI's train function
    train_csdi(
        model,
        CSDI_CONFIG["train"],
        train_loader,
        valid_loader=val_loader,
        valid_epoch_interval=10,
        foldername=str(save_dir) + "/"
    )

    print(f"\nTraining complete! Model saved to: {save_dir / 'model.pth'}")

    return model


def generate_csdi_predictions(model, data_loader, device, n_samples=N_CSDI_SAMPLES):
    """
    Generate predictions using CSDI model.

    Returns median of n_samples stochastic imputations.
    """
    model.eval()

    all_targets = []
    all_predictions = []
    all_gt_masks = []
    all_eval_masks = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating CSDI predictions"):
            # CSDI's evaluate method handles the data internally
            output = model.evaluate(batch, n_samples)
            samples, c_target, eval_points, observed_points, observed_time = output

            # samples: (B, n_samples, K, L) = (B, n_samples, D, T)
            # c_target: (B, K, L) = (B, D, T)
            # eval_points: (B, K, L) = (B, D, T)

            # Get median prediction
            samples_median = samples.median(dim=1).values  # (B, D, T)

            # Get gt_mask from batch
            gt_mask = batch['gt_mask'].to(device).float().permute(0, 2, 1)  # (B, D, T)

            all_targets.append(c_target.cpu().numpy())
            all_predictions.append(samples_median.cpu().numpy())
            all_gt_masks.append(gt_mask.cpu().numpy())
            all_eval_masks.append(eval_points.cpu().numpy())

    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_gt_masks, axis=0),
        np.concatenate(all_eval_masks, axis=0),
    )


# =============================================================================
# MISSINGDATAOT (OTimputer) TRAINING
# =============================================================================

def train_mdot_model(train_dataset, val_dataset, test_dataset, save_dir, device):
    """
    Train MissingDataOT (OTimputer - Algorithm 1).

    OTimputer works on 2D matrices (N, D), so we flatten time into samples.
    """
    # Import MDOT modules from benchmark_MissingDataOT explicitly
    # add_parent_to_path=True needed because imputers.py imports from utils
    mdot_imputers = load_module_from_path("mdot_imputers", MDOT_BENCHMARK_DIR / "imputers.py", add_parent_to_path=True)
    OTimputer = mdot_imputers.OTimputer

    print(f"\n{'=' * 70}")
    print("Training MissingDataOT (OTimputer - Algorithm 1)")
    print("=" * 70)
    print(f"  Iterations: {MDOT_CONFIG['niter']}")
    print(f"  Batch size: {MDOT_CONFIG['batchsize']}")
    print(f"  Learning rate: {MDOT_CONFIG['lr']}")

    # Save config
    config_to_save = MDOT_CONFIG.copy()
    config_to_save['opt'] = config_to_save['opt'].__name__
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # OTimputer doesn't have a traditional training phase
    # It directly optimizes imputed values on the data
    # We'll store the config and return None (imputation done at prediction time)

    print(f"\nMissingDataOT is a training-free method.")
    print(f"Imputation will be performed directly during evaluation.")

    return None


def generate_mdot_predictions(dataset, device, split_name=""):
    """
    Generate predictions using OTimputer (Algorithm 1).

    OTimputer works on 2D matrices (N, D). We flatten temporal and feature
    dimensions together: (N, T, D) -> (N, T*D), apply OTimputer, then reshape back.

    This treats each sample as a row with T*D combined features.

    Note: OTimputer runs on CPU only (the benchmark code doesn't handle GPU properly).

    Returns data in (N, D, T) format.
    """
    # Import MDOT modules from benchmark_MissingDataOT explicitly
    # add_parent_to_path=True needed because imputers.py imports from utils
    mdot_imputers = load_module_from_path("mdot_imputers", MDOT_BENCHMARK_DIR / "imputers.py", add_parent_to_path=True)
    OTimputer = mdot_imputers.OTimputer

    print(f"\nGenerating MissingDataOT predictions for {split_name}...")

    # Collect all data
    N = len(dataset)
    T = dataset[0]['observed_data'].shape[0]
    D = dataset[0]['observed_data'].shape[1]

    # Stack all data: (N, T, D)
    observed_data = np.stack([dataset[i]['observed_data'] for i in range(N)], axis=0)
    observed_mask = np.stack([dataset[i]['observed_mask'] for i in range(N)], axis=0)
    gt_mask = np.stack([dataset[i]['gt_mask'] for i in range(N)], axis=0)

    # Compute eval_mask
    eval_mask = observed_mask - gt_mask  # (N, T, D)

    # Flatten temporal and feature dimensions: (N, T, D) -> (N, T*D)
    observed_flat = observed_data.reshape(N, T * D)
    gt_mask_flat = gt_mask.reshape(N, T * D)

    print(f"  Input shape: (N={N}, T={T}, D={D})")
    print(f"  Flattened shape: (N={N}, T*D={T*D})")

    # Create input with NaN for entries to impute (where gt_mask=0)
    X_input = observed_flat.copy()
    X_input[gt_mask_flat == 0] = np.nan

    # Check for columns that are entirely NaN (can happen with high missing rate)
    # OTimputer initializes missing values using column means, which would be NaN
    # for all-NaN columns. We pre-fill these with normal(0,1) values.
    col_all_nan = np.all(np.isnan(X_input), axis=0)
    if np.any(col_all_nan):
        n_all_nan = np.sum(col_all_nan)
        print(f"  Warning: {n_all_nan} columns are entirely NaN, pre-filling with normal(0,1)")
        # Fill entire columns with standard normal values
        X_input[:, col_all_nan] = np.random.randn(N, n_all_nan)

    # Convert to torch - use CPU because OTimputer doesn't handle GPU properly
    # (it creates internal tensors on CPU that conflict with GPU input)
    X_torch = torch.from_numpy(X_input).double()  # CPU tensor

    # Create imputer and run on CPU
    imputer = OTimputer(**MDOT_CONFIG)
    X_imputed = imputer.fit_transform(X_torch, verbose=True, report_interval=50)

    # Convert back to numpy: (N, T*D)
    X_imputed_np = X_imputed.detach().cpu().numpy().astype(np.float32)

    # Reshape back to (N, T, D)
    predictions = X_imputed_np.reshape(N, T, D)
    targets = observed_data  # (N, T, D)
    gt_masks = gt_mask       # (N, T, D)
    eval_masks = eval_mask   # (N, T, D)

    print(f"  Output shape: (N={N}, T={T}, D={D})")

    # Transpose to (N, D, T) for consistency with other models
    return (
        targets.transpose(0, 2, 1),
        predictions.transpose(0, 2, 1),
        gt_masks.transpose(0, 2, 1),
        eval_masks.transpose(0, 2, 1),
    )


# =============================================================================
# BRITS MODEL TRAINING
# =============================================================================

def compute_deltas(observed_mask):
    """
    Compute time deltas (time since last observation) for BRITS.

    Args:
        observed_mask: (N, T, D) binary mask (1=observed, 0=missing)

    Returns:
        deltas: (N, T, D) time deltas
    """
    N, T, D = observed_mask.shape
    deltas = np.zeros((N, T, D), dtype=np.float32)

    for i in range(N):
        for d in range(D):
            last_observed = 0
            for t in range(T):
                if observed_mask[i, t, d] == 1:
                    deltas[i, t, d] = t - last_observed
                    last_observed = t
                else:
                    deltas[i, t, d] = t - last_observed

    return deltas


def prepare_brits_data(dataset):
    """
    Convert our dataset format to BRITS format.

    BRITS expects:
    - values: (N, T, D) with missing values filled with some value (we use 0)
    - masks: (N, T, D) binary (1=observed, 0=missing)
    - deltas: (N, T, D) time since last observation
    - evals: (N, T, D) ground truth for evaluation
    - eval_masks: (N, T, D) which values to evaluate
    - forward/backward: dict with above arrays
    - labels: (N,) for classification (we use dummy 0s)
    - is_train: (N,) whether sample is for training (always 1 in our case)

    Args:
        dataset: ImputationDataset

    Returns:
        brits_data: List of dicts in BRITS format
    """
    N = len(dataset)
    T = dataset[0]['observed_data'].shape[0]  # Time steps
    D = dataset[0]['observed_data'].shape[1]  # Features

    brits_samples = []

    for i in range(N):
        sample = dataset[i]

        # Get data: (T, D)
        observed_data = sample['observed_data']  # Already normalized
        observed_mask = sample['observed_mask']
        gt_mask = sample['gt_mask']

        # Eval mask = observed but not in gt (held-out for validation)
        eval_mask = observed_mask - gt_mask

        # Fill missing values with 0 (BRITS will learn to impute these)
        values = observed_data.copy()
        values[gt_mask == 0] = 0.0

        # Compute deltas
        deltas = compute_deltas(gt_mask[np.newaxis, :, :])[0]  # (T, D)

        # BRITS uses forward fill for initialization (not used in loss)
        forwards = np.zeros_like(values)
        for d in range(D):
            last_val = 0.0
            for t in range(T):
                if gt_mask[t, d] == 1:
                    last_val = values[t, d]
                    forwards[t, d] = last_val
                else:
                    forwards[t, d] = last_val

        # Create forward and backward sequences
        # Forward: normal order
        forward_dict = {
            'values': values,           # (T, D)
            'masks': gt_mask,          # (T, D) - what model sees as observed
            'deltas': deltas,          # (T, D)
            'forwards': forwards,      # (T, D)
            'evals': observed_data,    # (T, D) - ground truth
            'eval_masks': eval_mask,   # (T, D) - what to evaluate
        }

        # Backward: reversed order
        backward_dict = {
            'values': values[::-1, :].copy(),
            'masks': gt_mask[::-1, :].copy(),
            'deltas': compute_deltas(gt_mask[::-1, :][np.newaxis, :, :])[0],
            'forwards': forwards[::-1, :].copy(),
            'evals': observed_data[::-1, :].copy(),
            'eval_masks': eval_mask[::-1, :].copy(),
        }

        brits_sample = {
            'forward': forward_dict,
            'backward': backward_dict,
            'label': 0,  # Dummy label (no classification)
            'is_train': 1,  # Always 1 (we handle train/val/test separately)
        }

        brits_samples.append(brits_sample)

    return brits_samples


def brits_collate_fn(samples):
    """
    Collate function to batch BRITS samples.

    Args:
        samples: List of BRITS sample dicts

    Returns:
        batch: Dict of batched tensors
    """
    batch_size = len(samples)

    # Get dimensions from first sample
    T = samples[0]['forward']['values'].shape[0]
    D = samples[0]['forward']['values'].shape[1]

    # Initialize batch arrays
    def stack_dict(key):
        forward_list = [s['forward'][key] for s in samples]
        backward_list = [s['backward'][key] for s in samples]
        return {
            'forward': torch.FloatTensor(np.stack(forward_list, axis=0)),  # (N, T, D)
            'backward': torch.FloatTensor(np.stack(backward_list, axis=0)),  # (N, T, D)
        }

    batch = {
        'values': stack_dict('values'),
        'masks': stack_dict('masks'),
        'deltas': stack_dict('deltas'),
        'forwards': stack_dict('forwards'),
        'evals': stack_dict('evals'),
        'eval_masks': stack_dict('eval_masks'),
        'labels': torch.FloatTensor([s['label'] for s in samples]),
        'is_train': torch.FloatTensor([s['is_train'] for s in samples]),
    }

    # Restructure to match BRITS expected format
    brits_batch = {
        'forward': {
            'values': batch['values']['forward'],
            'masks': batch['masks']['forward'],
            'deltas': batch['deltas']['forward'],
            'forwards': batch['forwards']['forward'],
            'evals': batch['evals']['forward'],
            'eval_masks': batch['eval_masks']['forward'],
        },
        'backward': {
            'values': batch['values']['backward'],
            'masks': batch['masks']['backward'],
            'deltas': batch['deltas']['backward'],
            'forwards': batch['forwards']['backward'],
            'evals': batch['evals']['backward'],
            'eval_masks': batch['eval_masks']['backward'],
        },
        'labels': batch['labels'],
        'is_train': batch['is_train'],
    }

    return brits_batch


def train_brits_model(train_dataset, val_dataset, save_dir, device, d_features, d_time):
    """
    Train BRITS model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        save_dir: Directory to save model
        device: Device to use
        d_features: Number of features (D)
        d_time: Number of time steps (T)

    Returns:
        model: Trained BRITS model
    """
    # Import BRITS modules from benchmark_brits
    # Need to add both the models directory (for rits import) and the benchmark root (for utils, data_loader imports)
    brits_extra_paths = [BRITS_BENCHMARK_DIR / "models", BRITS_BENCHMARK_DIR]
    brits_module = load_module_from_path("brits_model", BRITS_BENCHMARK_DIR / "models" / "brits.py", extra_paths=brits_extra_paths)
    rits_module = load_module_from_path("rits_model", BRITS_BENCHMARK_DIR / "models" / "rits.py", extra_paths=brits_extra_paths)

    # Monkey-patch the hardcoded dimensions in BRITS
    # BRITS hardcodes SEQ_LEN and feature dimensions, we need to override them
    rits_module.SEQ_LEN = d_time
    brits_module.SEQ_LEN = d_time
    rits_module.RNN_HID_SIZE = BRITS_CONFIG['rnn_hid_size']
    brits_module.RNN_HID_SIZE = BRITS_CONFIG['rnn_hid_size']

    # Rebuild BRITS model classes with updated dimensions
    # We need to dynamically modify the model architecture
    # IMPORTANT: brits_module imports its own 'rits' module internally, so we must patch
    # the rits module that brits_module references (brits_module.rits), not rits_module directly
    rits_in_brits = brits_module.rits
    original_rits_build = rits_in_brits.Model.build

    def patched_rits_build(self):
        """Patched build method with dynamic dimensions."""
        self.rnn_cell = nn.LSTMCell(d_features * 2, BRITS_CONFIG['rnn_hid_size'])
        self.temp_decay_h = rits_in_brits.TemporalDecay(input_size=d_features, output_size=BRITS_CONFIG['rnn_hid_size'], diag=False)
        self.temp_decay_x = rits_in_brits.TemporalDecay(input_size=d_features, output_size=d_features, diag=True)
        self.hist_reg = nn.Linear(BRITS_CONFIG['rnn_hid_size'], d_features)
        self.feat_reg = rits_in_brits.FeatureRegression(d_features)
        self.weight_combine = nn.Linear(d_features * 2, d_features)
        self.dropout = nn.Dropout(p=0.25)
        self.out = nn.Linear(BRITS_CONFIG['rnn_hid_size'], 1)

    # Monkey-patch the build method on the rits module that brits uses
    rits_in_brits.Model.build = patched_rits_build
    # Also update SEQ_LEN on the rits module that brits uses
    rits_in_brits.SEQ_LEN = d_time

    # Create BRITS model
    model = brits_module.Model()
    model = model.to(device)

    # Restore original build method to avoid side effects
    rits_in_brits.Model.build = original_rits_build

    # Prepare data
    print("\nPreparing BRITS training data...")
    train_brits_data = prepare_brits_data(train_dataset)
    val_brits_data = prepare_brits_data(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_brits_data,
        batch_size=BRITS_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=brits_collate_fn
    )

    val_loader = DataLoader(
        val_brits_data,
        batch_size=BRITS_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=brits_collate_fn
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=BRITS_CONFIG['lr'])

    # Training loop
    print(f"\nTraining BRITS for {BRITS_CONFIG['epochs']} epochs...")
    print("Note: Using final model (no validation-based selection, matching original BRITS)")

    for epoch in range(BRITS_CONFIG['epochs']):
        model.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            for direction in ['forward', 'backward']:
                for key in data[direction]:
                    data[direction][key] = data[direction][key].to(device)
            data['labels'] = data['labels'].to(device)
            data['is_train'] = data['is_train'].to(device)

            # Forward pass
            ret = model.run_on_batch(data, optimizer)
            train_loss += ret['loss'].item()

        train_loss /= len(train_loader)

        # Validation (for monitoring only, not used for model selection)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_mae = 0.0
            val_count = 0

            with torch.no_grad():
                for data in val_loader:
                    # Move data to device
                    for direction in ['forward', 'backward']:
                        for key in data[direction]:
                            data[direction][key] = data[direction][key].to(device)
                    data['labels'] = data['labels'].to(device)
                    data['is_train'] = data['is_train'].to(device)

                    ret = model.run_on_batch(data, None)

                    # Compute MAE on eval_masks (for monitoring only)
                    evals = ret['evals'].cpu().numpy()
                    imputations = ret['imputations'].cpu().numpy()
                    eval_masks = ret['eval_masks'].cpu().numpy()

                    mae = np.abs(evals - imputations) * eval_masks
                    val_mae += mae.sum()
                    val_count += eval_masks.sum()

            val_mae = val_mae / (val_count + 1e-8)

            print(f"Epoch {epoch+1}/{BRITS_CONFIG['epochs']}: "
                  f"Train Loss = {train_loss:.6f}, Val MAE = {val_mae:.6f} (monitoring only)")

    # Save final model (no early stopping)
    torch.save(model.state_dict(), save_dir / 'model.pth')

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(BRITS_CONFIG, f, indent=2)

    return model


def generate_brits_predictions(model, dataset, device, d_features, d_time):
    """
    Generate predictions using trained BRITS model.

    Args:
        model: Trained BRITS model
        dataset: Dataset to predict on
        device: Device to use
        d_features: Number of features (D)
        d_time: Number of time steps (T)

    Returns:
        Tuple of (targets, predictions, gt_masks, eval_masks) in (N, D, T) format
    """
    # Prepare data
    brits_data = prepare_brits_data(dataset)

    # Create data loader
    data_loader = DataLoader(
        brits_data,
        batch_size=BRITS_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=brits_collate_fn
    )

    # Generate predictions
    model.eval()
    all_targets = []
    all_predictions = []
    all_gt_masks = []
    all_eval_masks = []

    with torch.no_grad():
        for data in data_loader:
            # Move data to device
            for direction in ['forward', 'backward']:
                for key in data[direction]:
                    data[direction][key] = data[direction][key].to(device)
            data['labels'] = data['labels'].to(device)
            data['is_train'] = data['is_train'].to(device)

            # Forward pass
            ret = model.run_on_batch(data, None)

            # Collect results
            # evals: (N, T, D) ground truth
            # imputations: (N, T, D) predictions
            # Use forward direction's eval_masks and evals (backward is reversed)
            targets = data['forward']['evals'].cpu().numpy()  # (N, T, D)
            predictions = ret['imputations'].cpu().numpy()    # (N, T, D)
            gt_masks = data['forward']['masks'].cpu().numpy()  # (N, T, D)
            eval_masks = data['forward']['eval_masks'].cpu().numpy()  # (N, T, D)

            all_targets.append(targets)
            all_predictions.append(predictions)
            all_gt_masks.append(gt_masks)
            all_eval_masks.append(eval_masks)

    # Concatenate batches: (N, T, D)
    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    gt_masks = np.concatenate(all_gt_masks, axis=0)
    eval_masks = np.concatenate(all_eval_masks, axis=0)

    # Transpose to (N, D, T) for consistency with other models
    return (
        targets.transpose(0, 2, 1),
        predictions.transpose(0, 2, 1),
        gt_masks.transpose(0, 2, 1),
        eval_masks.transpose(0, 2, 1),
    )


# =============================================================================
# SILRTC MODEL TRAINING (Simple Low Rank Tensor Completion)
# =============================================================================

def train_silrtc_model(train_dataset, val_dataset, save_dir, d_features, d_time):
    """
    "Train" SILRTC model (actually just save configuration).

    SILRTC is a direct optimization method like MDOT - it doesn't learn from training data.
    Instead, it solves an optimization problem directly on each dataset (train/val/test).
    This function only saves the configuration.

    Args:
        train_dataset: Training dataset (not used for training, SILRTC is direct optimization)
        val_dataset: Validation dataset (not used)
        save_dir: Directory to save config
        d_features: Number of features (D)
        d_time: Number of time steps (T)

    Returns:
        Dictionary with configuration only
    """
    print("\n" + "="*80)
    print("SILRTC Configuration (Direct Optimization - No Training)")
    print("="*80)
    print(f"Configuration:")
    print(f"  alphas: {SILRTC_CONFIG['alphas']} (uniform weights)")
    print(f"  beta: {SILRTC_CONFIG['beta']}")
    print(f"  max_iter: {SILRTC_CONFIG['max_iter']}")
    print(f"  tol: {SILRTC_CONFIG['tol']}")
    print("\nNote: SILRTC operates directly on each dataset (train/val/test)")
    print("      No learning from training data - pure optimization method")

    # Store configuration only (no training)
    model_dict = {
        'config': SILRTC_CONFIG.copy(),
        'd_features': d_features,
        'd_time': d_time,
    }

    # Save configuration
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model': 'silrtc',
            'config': SILRTC_CONFIG,
            'd_features': d_features,
            'd_time': d_time,
            'note': 'SILRTC is a direct optimization method - operates independently on each dataset'
        }, f, indent=2)

    print(f"\nConfiguration saved to {config_path}")
    print("="*80)

    return model_dict


def generate_silrtc_predictions(model_dict, dataset, split_name=''):
    """
    Generate predictions using SILRTC by running optimization directly on the dataset.

    Like MDOT, SILRTC is a direct optimization method that doesn't use training data.
    It solves the tensor completion problem independently for each dataset.

    Args:
        model_dict: Dictionary with configuration
        dataset: Dataset to run SILRTC on
        split_name: Name of split ('train', 'val', or 'test') for logging

    Returns:
        Tuple of (targets, predictions, gt_masks, eval_masks) in (N, D, T) format
    """
    # Choose GPU or CPU version based on config
    use_gpu = model_dict['config'].get('use_gpu', True)

    if use_gpu:
        try:
            import torch
            from silrtc_torch import silrtc_torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                print(f"  Using GPU-accelerated SILRTC (PyTorch on {torch.cuda.get_device_name(0)})")
            else:
                print(f"  Using CPU-accelerated SILRTC (PyTorch on CPU)")
            silrtc_fn = lambda *args, **kwargs: silrtc_torch(*args, **kwargs, device=device)
        except ImportError:
            print(f"  Warning: PyTorch not available, falling back to NumPy CPU version")
            from silrtc import silrtc
            silrtc_fn = silrtc
    else:
        from silrtc import silrtc
        print(f"  Using NumPy CPU version of SILRTC")
        silrtc_fn = silrtc

    N = len(dataset)
    d_time = model_dict['d_time']
    d_features = model_dict['d_features']

    # Collect data - stack all samples
    data = np.stack([dataset[i]['observed_data'] for i in range(N)], axis=0)  # (N, T, D)
    observed_masks = np.stack([dataset[i]['observed_mask'] for i in range(N)], axis=0)
    gt_masks = np.stack([dataset[i]['gt_mask'] for i in range(N)], axis=0)

    # Compute eval_mask (held-out entries for evaluation)
    eval_masks = observed_masks - gt_masks  # (N, T, D)

    # Run SILRTC directly on this dataset (no use of training data)
    print(f"\nRunning SILRTC on {split_name} set (N={N})...")
    print(f"  Observed entries (gt_mask): {gt_masks.sum()}/{gt_masks.size} ({100*gt_masks.mean():.1f}%)")

    # Prepare input: observed entries at their values, missing entries = 0
    # SILRTC will iteratively fill in the 0s via low-rank completion
    tensor_observed = data * gt_masks  # Missing entries become 0
    predictions = silrtc_fn(
        tensor_observed,
        gt_masks,
        alphas=model_dict['config']['alphas'],
        beta=model_dict['config']['beta'],
        max_iter=model_dict['config']['max_iter'],
        tol=model_dict['config']['tol'],
        verbose=(split_name == 'train')  # Verbose only for first split
    )

    # Transpose everything to (N, D, T) format for consistency
    targets = data.transpose(0, 2, 1)  # (N, T, D) -> (N, D, T)
    predictions = predictions.transpose(0, 2, 1)
    gt_masks = gt_masks.transpose(0, 2, 1)
    eval_masks = eval_masks.transpose(0, 2, 1)

    return targets, predictions, gt_masks, eval_masks


# =============================================================================
# not-MIWAE MODEL TRAINING (PyTorch version)
# =============================================================================

def train_notmiwae_model(train_dataset, val_dataset, save_dir, device):
    """
    Train not-MIWAE model (PyTorch implementation).

    not-MIWAE is a variational autoencoder that jointly models p(x,s) where:
    - x is the data
    - s is the missingness indicator (binary mask)

    This allows it to handle Missing Not At Random (MNAR) data by explicitly
    modeling the missing data mechanism.

    Like MDOT, not-MIWAE operates on flattened 2D data (N*T, D).

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        save_dir: Directory to save model
        device: Device to use (cuda or cpu)
    """
    # Import notMIWAE modules from PyTorch implementation
    notmiwae_module = load_module_from_path(
        "notmiwae_module",
        NOTMIWAE_BENCHMARK_DIR / "notMIWAE.py",
        add_parent_to_path=True
    )
    trainer_module = load_module_from_path(
        "notmiwae_trainer",
        NOTMIWAE_BENCHMARK_DIR / "trainer.py",
        add_parent_to_path=True
    )

    notMIWAE = notmiwae_module.notMIWAE
    train_fn = trainer_module.train

    print(f"\n{'=' * 70}")
    print("Training not-MIWAE (MIWAE with Missing Process Modeling) - PyTorch")
    print("=" * 70)

    # Get data dimensions
    N_train = len(train_dataset)
    N_val = len(val_dataset)
    T = train_dataset[0]['observed_data'].shape[0]
    D = train_dataset[0]['observed_data'].shape[1]

    print(f"  Original shape: N_train={N_train}, N_val={N_val}, T={T}, D={D}")

    # Flatten time series to 2D: (N, T, D) -> (N*T, D)
    # For not-MIWAE, we treat each time step as an independent sample
    train_data = np.stack([train_dataset[i]['observed_data'] for i in range(N_train)], axis=0)  # (N, T, D)
    train_mask = np.stack([train_dataset[i]['gt_mask'] for i in range(N_train)], axis=0)  # (N, T, D)

    val_data = np.stack([val_dataset[i]['observed_data'] for i in range(N_val)], axis=0)
    val_mask = np.stack([val_dataset[i]['gt_mask'] for i in range(N_val)], axis=0)

    # Flatten: (N, T, D) -> (N*T, D)
    X_train_flat = train_data.reshape(-1, D).astype(np.float32)  # (N*T, D)
    S_train_flat = train_mask.reshape(-1, D).astype(np.float32)  # (N*T, D)

    X_val_flat = val_data.reshape(-1, D).astype(np.float32)
    S_val_flat = val_mask.reshape(-1, D).astype(np.float32)

    print(f"  Flattened shape: train={X_train_flat.shape}, val={X_val_flat.shape}")

    # Prepare data: fill missing with 0 (notMIWAE PyTorch expects data with 0 for missing)
    X_train_z = X_train_flat.copy()
    X_train_z[S_train_flat == 0] = 0

    X_val_z = X_val_flat.copy()
    X_val_z[S_val_flat == 0] = 0

    # Get latent dimension (use n_latent from config, but cap at D-1)
    n_latent = min(NOTMIWAE_CONFIG['n_latent'], D - 1) if D > 1 else 1

    print(f"  Config: n_latent={n_latent}, n_hidden={NOTMIWAE_CONFIG['n_hidden']}, "
          f"n_samples={NOTMIWAE_CONFIG['n_samples']}")
    print(f"  Missing process: {NOTMIWAE_CONFIG['missing_process']}")
    print(f"  Max iterations: {NOTMIWAE_CONFIG['max_iter']}")
    print(f"  Device: {device}")

    # Save config
    config_to_save = {**NOTMIWAE_CONFIG, 'n_latent_actual': n_latent, 'T': T, 'D': D}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Create model save path
    model_save_path = str(save_dir / 'model.pth')

    # Create not-MIWAE model (PyTorch)
    model = notMIWAE(
        input_dim=D,
        n_latent=n_latent,
        n_hidden=NOTMIWAE_CONFIG['n_hidden'],
        n_samples=NOTMIWAE_CONFIG['n_samples'],
        out_dist=NOTMIWAE_CONFIG['out_dist'],
        missing_process=NOTMIWAE_CONFIG['missing_process'],
        device=device
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Train the model
    train_fn(
        model,
        X=X_train_z,
        S=S_train_flat,
        Xval=X_val_z,
        Sval=S_val_flat,
        batch_size=NOTMIWAE_CONFIG['batch_size'],
        max_iter=NOTMIWAE_CONFIG['max_iter'],
        save_path=model_save_path,
        device=device
    )

    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)

    print(f"\nTraining complete! Model saved to: {save_dir}")

    # Return model and metadata needed for prediction
    return {
        'model': model,
        'T': T,
        'D': D,
        'config': config_to_save,
    }


def generate_notmiwae_predictions(model_dict, dataset, device, split_name='test'):
    """
    Generate predictions using not-MIWAE model (PyTorch).

    Uses importance-weighted samples from the posterior to impute missing values.

    Args:
        model_dict: Dictionary containing model and metadata from train_notmiwae_model
        dataset: Dataset to generate predictions for
        device: Device to use
        split_name: Name of the split (for logging)

    Returns:
        Tuple of (targets, predictions, gt_masks, eval_masks) all in (N, D, T) format
    """
    # Import utils for batch imputation
    utils_module = load_module_from_path(
        "notmiwae_utils",
        NOTMIWAE_BENCHMARK_DIR / "utils.py",
        add_parent_to_path=True
    )
    batch_imputation = utils_module.batch_imputation

    model = model_dict['model']
    T = model_dict['T']
    D = model_dict['D']
    L = NOTMIWAE_CONFIG['L']

    N = len(dataset)
    print(f"\nGenerating not-MIWAE predictions for {split_name} (N={N})...")

    # Collect data
    data = np.stack([dataset[i]['observed_data'] for i in range(N)], axis=0)  # (N, T, D)
    gt_masks = np.stack([dataset[i]['gt_mask'] for i in range(N)], axis=0)  # (N, T, D)
    observed_masks = np.stack([dataset[i]['observed_mask'] for i in range(N)], axis=0)  # (N, T, D)

    # Compute eval_mask
    eval_masks = observed_masks - gt_masks

    # Flatten for not-MIWAE: (N, T, D) -> (N*T, D)
    data_flat = data.reshape(-1, D).astype(np.float32)
    gt_masks_flat = gt_masks.reshape(-1, D).astype(np.float32)

    # Prepare data with missing filled with 0
    Xz = data_flat.copy()
    Xz[gt_masks_flat == 0] = 0

    # Use batch imputation for efficiency
    XM = batch_imputation(model, Xz, gt_masks_flat, L=L, batch_size=100, device=device)

    # Reshape back: (N*T, D) -> (N, T, D)
    predictions = XM.reshape(N, T, D)
    targets = data  # Original data (N, T, D)

    # Transpose to (N, D, T) format for consistency with other models
    targets = targets.transpose(0, 2, 1)
    predictions = predictions.transpose(0, 2, 1)
    gt_masks = gt_masks.transpose(0, 2, 1)
    eval_masks = eval_masks.transpose(0, 2, 1)

    return targets, predictions, gt_masks, eval_masks


# =============================================================================
# DRIO MODEL TRAINING
# =============================================================================

def train_drio_epoch(model, drio_trainer, train_loader, optimizer, device):
    """Train DRIO model for one epoch using Algorithm 1."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_sinkhorn = 0
    n_batches = 0

    for batch in train_loader:
        # Data comes in (B, T, D), model expects (B, D, T)
        observed_data = batch['observed_data'].float().to(device)  # (B, T, D)
        observed_mask = batch['observed_mask'].float().to(device)  # (B, T, D)
        gt_mask = batch['gt_mask'].float().to(device)              # (B, T, D)

        # Transpose to (B, D, T)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        # Use DRIOTrainer's train_step
        metrics = drio_trainer.train_step(
            model=model,
            observed_data=observed_data,
            observed_mask=observed_mask,
            gt_mask=gt_mask,
            optimizer=optimizer
        )

        total_loss += metrics['total_loss']
        total_recon_loss += metrics['reconstruction_loss']
        total_sinkhorn += metrics['sinkhorn_divergence']
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'reconstruction_loss': total_recon_loss / n_batches,
        'sinkhorn_divergence': total_sinkhorn / n_batches,
    }


def validate_drio(model, val_loader, device):
    """
    Validate DRIO model using self-supervised reconstruction error.

    IMPORTANT: Uses gt_mask (observed entries) for model selection, NOT eval_mask (held-out entries).

    This is NOT data leakage because:
    - gt_mask = entries the model sees during training (observed values)
    - eval_mask = held-out entries for final testing (never seen during training/validation)
    - Validating on gt_mask tests the model's ability to reconstruct observed entries
      on NEW samples (validation set), which ensures generalizability without leaking
      test set information.

    This approach:
    1. Avoids test set leakage (eval_mask never used for model selection)
    2. Validates generalization (validation samples are unseen during training)
    3. Is deployable (doesn't require held-out ground truth for validation)

    Returns:
        gt_loss: Self-supervised loss on gt_mask (used for model selection)
        eval_loss: Loss on eval_mask (for monitoring only, not used for selection)
    """
    model.eval()
    total_gt_loss = 0
    total_eval_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)  # Observed entries, NOT eval_mask

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            # Eval mask: held-out entries (for monitoring only)
            eval_mask = observed_mask - gt_mask

            predictions = model(observed_data, gt_mask)

            # Self-supervised validation: reconstruction error on gt_mask entries (used for model selection)
            # This validates generalization on NEW samples without using held-out test data
            gt_loss = ((predictions - observed_data) ** 2 * gt_mask).sum() / (gt_mask.sum() + 1e-10)

            # Eval loss: error on held-out entries (for monitoring only, NOT used for selection)
            eval_loss = ((predictions - observed_data) ** 2 * eval_mask).sum() / (eval_mask.sum() + 1e-10)

            total_gt_loss += gt_loss.item()
            total_eval_loss += eval_loss.item()
            n_batches += 1

    return total_gt_loss / n_batches, total_eval_loss / n_batches


def drio_hyperparameter_search(
    train_loader,
    val_loader,
    device,
    metadata,
    model_config,
    alpha_grid=None,
    gamma_grid=None,
    cv_epochs=None,
    model_class=None,
    model_kwargs=None,
):
    """
    Single-fold hyperparameter search for DRIO.

    Trains a model for each (alpha, gamma) combination on the training set
    and evaluates on the validation set. Returns the best hyperparameters.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        metadata: Data metadata with shape info
        model_config: Architecture configuration dict with 'model_type' and arch params
        alpha_grid: List of alpha values to search (default: DRIO_DEFAULT_ALPHA_GRID)
        gamma_grid: List of gamma values to search (default: DRIO_DEFAULT_GAMMA_GRID)
        cv_epochs: Number of epochs for CV (default: use dataset-specific lookup)
        model_class: Optional custom model class (if None, uses create_imputer)
        model_kwargs: Optional kwargs for custom model class

    Returns:
        best_alpha: Best alpha value
        best_gamma: Best gamma value
        cv_results: Dict mapping (alpha, gamma) -> dict with val_loss, mse, w2
        best_model_state: State dict of the best model
    """
    # Get optimized CV config from lookup table (alpha_grid, gamma_grid, epochs)
    lookup_alpha_grid, lookup_gamma_grid, lookup_epochs = get_drio_cv_config(
        metadata['dataset'],
        metadata['missing_type'],
        metadata['missing_ratio']
    )

    # Use lookup values if not overridden by arguments
    if alpha_grid is None:
        alpha_grid = lookup_alpha_grid
    if gamma_grid is None:
        gamma_grid = lookup_gamma_grid
    if cv_epochs is None:
        cv_epochs = lookup_epochs

    d_features = metadata['shape'][2]
    d_time = metadata['shape'][1]
    train_config = DRIO_CONFIG['train'].copy()
    drio_config = DRIO_CONFIG['drio']

    # Use cv_epochs (either from argument or lookup table)
    train_config['epochs'] = cv_epochs

    # Extract model_type and architecture params
    arch_config = model_config.copy()
    model_type = arch_config.pop('model_type')

    print(f"\n{'=' * 70}")
    print(f"DRIO Hyperparameter Search (Single-Fold) - {model_type}")
    print("=" * 70)
    print(f"Alpha grid: {alpha_grid}")
    print(f"Gamma grid: {gamma_grid}")
    print(f"Total combinations: {len(alpha_grid) * len(gamma_grid)}")
    print(f"Epochs per model: {train_config['epochs']}")

    cv_results = {}
    best_eval_loss = float('inf')
    best_alpha = alpha_grid[0]
    best_gamma = gamma_grid[0]
    best_model_state = None

    combo_idx = 0
    total_combos = len(alpha_grid) * len(gamma_grid)

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{total_combos}] Testing alpha={alpha}, gamma={gamma}...")

            # Create fresh model
            if model_class is not None:
                # Use custom model class (e.g., BRITSForDRIO)
                model = model_class(**model_kwargs).to(device)
            else:
                # Use standard create_imputer
                model = create_imputer(
                    model_type=model_type,
                    d_features=d_features,
                    d_time=d_time,
                    **arch_config
                ).to(device)

            # Create DRIO trainer with current hyperparameters
            drio_trainer = create_drio_trainer(
                alpha=alpha,
                gamma=gamma,
                epsilon=drio_config['epsilon'],
                tau=drio_config['tau'],
                inner_steps=drio_config['inner_steps'],
                inner_lr=drio_config['inner_lr'],
                adaptive_epsilon=drio_config['adaptive_epsilon'],
                epsilon_quant=drio_config['epsilon_quant'],
                epsilon_mult=drio_config['epsilon_mult'],
            )

            # Create optimizer
            optimizer = Adam(
                model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config['weight_decay']
            )

            # Train for full epochs with validation at every epoch
            # Best epoch is selected based on val_eval_loss (eval_mask) for CV
            best_epoch_val_loss = float('inf')
            best_epoch_val_eval = float('inf')
            best_epoch_num = 0
            best_epoch_model_state = None
            for epoch in range(train_config['epochs']):
                train_metrics = train_drio_epoch(model, drio_trainer, train_loader, optimizer, device)

                # Validate at every epoch for DRIO CV
                val_gt_loss, val_eval_loss = validate_drio(model, val_loader, device)
                print(f"    Epoch {epoch+1}/{train_config['epochs']}: "
                      f"total_loss={train_metrics['total_loss']:.6f}, "
                      f"recon={train_metrics['reconstruction_loss']:.6f}, "
                      f"sinkhorn={train_metrics['sinkhorn_divergence']:.6f}, "
                      f"val_gt={val_gt_loss:.6f}, val_eval={val_eval_loss:.6f}")
                # Select best epoch based on val_eval_loss (eval_mask)
                if val_eval_loss < best_epoch_val_eval:
                    best_epoch_val_loss = val_gt_loss
                    best_epoch_val_eval = val_eval_loss
                    best_epoch_num = epoch + 1
                    best_epoch_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Load best model state for this combination to compute final metrics
            if best_epoch_model_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_epoch_model_state.items()})

            # Compute MSE and W2 on validation set for this combination
            targets, predictions, gt_masks, eval_masks = generate_drio_predictions(model, val_loader, device)
            observed_mask = gt_masks + eval_masks
            val_eval_results = evaluate_imputation(
                X_true=targets,
                X_pred=predictions,
                mask=eval_masks,
                min_samples_w2=5,
                gt_mask=gt_masks,
                observed_mask=observed_mask
            )
            mse = val_eval_results['mse']
            w2_global = val_eval_results['w2_global']
            # CVaR: 90th percentile of squared errors (tail risk metric)
            cvar = val_eval_results['se_summary']['p90']

            # Store results (selection is based on val_eval from eval_mask)
            cv_results[(alpha, gamma)] = {
                'val_loss': best_epoch_val_loss,
                'val_eval': best_epoch_val_eval,
                'epoch': best_epoch_num,
                'mse': mse,
                'w2_global': w2_global,
                'cvar': cvar,
            }
            print(f"    Final: val_loss={best_epoch_val_loss:.6f}, val_eval={best_epoch_val_eval:.6f}, "
                  f"epoch={best_epoch_num}, MSE={mse:.6f}, W2_global={w2_global:.6f}, CVaR={cvar:.6f}")

            # Selection based on val_eval (MSE on eval_mask)
            if best_epoch_val_eval < best_eval_loss:
                best_eval_loss = best_epoch_val_eval
                best_alpha = alpha
                best_gamma = gamma
                best_model_state = best_epoch_model_state

    # Print summary table of all combinations (sorted by val_eval - MSE on eval_mask)
    print(f"\n{'=' * 100}")
    print("CV Results Summary (sorted by val_eval - MSE on eval_mask)")
    print("=" * 100)
    print(f"{'Alpha':>8} | {'Gamma':>8} | {'Val Loss':>12} | {'Val Eval':>12} | {'Epoch':>6} | {'MSE':>12} | {'W2_global':>12} | {'CVaR':>12}")
    print("-" * 100)

    # Sort by val_eval (selection criterion)
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['val_eval'])
    for (alpha, gamma), metrics in sorted_results:
        marker = " *" if alpha == best_alpha and gamma == best_gamma else ""
        print(f"{alpha:>8.2f} | {gamma:>8.2f} | {metrics['val_loss']:>12.6f} | {metrics['val_eval']:>12.6f} | {metrics['epoch']:>6} | {metrics['mse']:>12.6f} | {metrics['w2_global']:>12.6f} | {metrics['cvar']:>12.6f}{marker}")

    print("-" * 100)
    print(f"* Best hyperparameters (by val_eval): alpha={best_alpha}, gamma={best_gamma}")
    print(f"  Best validation eval loss: {best_eval_loss:.6f}")
    print("=" * 100)

    # Print second table sorted by CVaR (90th percentile of squared errors)
    print(f"\n{'=' * 100}")
    print("CV Results Summary (sorted by CVaR - 90th percentile of squared errors)")
    print("=" * 100)
    print(f"{'Alpha':>8} | {'Gamma':>8} | {'Val Loss':>12} | {'Val Eval':>12} | {'Epoch':>6} | {'MSE':>12} | {'W2_global':>12} | {'CVaR':>12}")
    print("-" * 100)

    # Sort by CVaR
    sorted_by_cvar = sorted(cv_results.items(), key=lambda x: x[1]['cvar'])
    best_cvar_alpha, best_cvar_gamma = sorted_by_cvar[0][0]
    best_cvar_value = sorted_by_cvar[0][1]['cvar']
    for (alpha, gamma), metrics in sorted_by_cvar:
        marker = " *" if alpha == best_cvar_alpha and gamma == best_cvar_gamma else ""
        print(f"{alpha:>8.2f} | {gamma:>8.2f} | {metrics['val_loss']:>12.6f} | {metrics['val_eval']:>12.6f} | {metrics['epoch']:>6} | {metrics['mse']:>12.6f} | {metrics['w2_global']:>12.6f} | {metrics['cvar']:>12.6f}{marker}")

    print("-" * 100)
    print(f"* Best hyperparameters (by CVaR): alpha={best_cvar_alpha}, gamma={best_cvar_gamma}")
    print(f"  Best CVaR (p90 of SE): {best_cvar_value:.6f}")
    print("=" * 100)

    return best_alpha, best_gamma, cv_results, best_model_state


def train_drio_model(model, train_loader, val_loader, save_dir, device, alpha=None, gamma=None, metadata=None):
    """
    Train DRIO model using Algorithm 1: Alternating optimization.

    1. Inner maximization: Update adversary Z to find worst-case distribution
    2. Outer minimization: Update imputer θ to minimize DRIO objective

    Args:
        model: The imputer model
        train_loader: Training data loader
        val_loader: Validation data loader
        save_dir: Directory to save model and results
        device: Device to use
        alpha: DRIO alpha parameter (default: use DRIO_CONFIG)
        gamma: DRIO gamma parameter (default: use DRIO_CONFIG)
        metadata: Data metadata for epoch lookup (optional)
    """
    print(f"\n{'=' * 70}")
    print("Training DRIO (Distributionally Robust Imputer Objective)")
    print("=" * 70)

    train_config = DRIO_CONFIG['train'].copy()
    drio_config = DRIO_CONFIG['drio']

    # Get epochs: standalone override > lookup table > default
    if metadata is not None:
        if '_standalone_epochs' in metadata:
            # Standalone mode with explicit epochs from lookup
            train_config['epochs'] = metadata['_standalone_epochs']
        else:
            # CV mode - use epochs from CV lookup table
            optimal_epochs = get_drio_epochs(
                metadata['dataset'],
                metadata['missing_type'],
                metadata['missing_ratio']
            )
            train_config['epochs'] = optimal_epochs

    # Use provided alpha/gamma or fall back to config defaults
    alpha = alpha if alpha is not None else drio_config['alpha']
    gamma = gamma if gamma is not None else drio_config['gamma']

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Learning rate: {train_config['lr']}")
    print(f"  DRIO alpha: {alpha}")
    print(f"  DRIO gamma: {gamma}")
    print(f"  Inner steps: {drio_config['inner_steps']}")

    # Create DRIO trainer
    drio_trainer = create_drio_trainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=drio_config['epsilon'],
        tau=drio_config['tau'],
        inner_steps=drio_config['inner_steps'],
        inner_lr=drio_config['inner_lr'],
        adaptive_epsilon=drio_config['adaptive_epsilon'],
        epsilon_quant=drio_config['epsilon_quant'],
        epsilon_mult=drio_config['epsilon_mult'],
    )

    optimizer = Adam(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )

    train_losses = []
    val_gt_losses = []
    val_eval_losses = []

    for epoch in tqdm(range(train_config['epochs']), desc="Training DRIO"):
        train_metrics = train_drio_epoch(model, drio_trainer, train_loader, optimizer, device)
        train_losses.append(train_metrics['total_loss'])

        # Validate at every epoch
        val_gt_loss, val_eval_loss = validate_drio(model, val_loader, device)
        val_gt_losses.append(val_gt_loss)
        val_eval_losses.append(val_eval_loss)

        tqdm.write(f"  Epoch {epoch+1}: total_loss={train_metrics['total_loss']:.6f}, "
                  f"recon={train_metrics['reconstruction_loss']:.6f}, "
                  f"sinkhorn={train_metrics['sinkhorn_divergence']:.6f}, "
                  f"val_gt={val_gt_loss:.6f}, val_eval={val_eval_loss:.6f}")

    # Save the final (last epoch) model - no early stopping in standalone training
    torch.save(model.state_dict(), save_dir / 'model.pth')

    # Save training curves
    np.savez(
        save_dir / 'training_curves.npz',
        train_losses=train_losses,
        val_gt_losses=val_gt_losses,
        val_eval_losses=val_eval_losses,
    )

    print(f"\nTraining complete! Final val_gt loss: {val_gt_losses[-1]:.6f}, val_eval loss: {val_eval_losses[-1]:.6f}")

    return model


def generate_drio_predictions(model, data_loader, device):
    """
    Generate predictions using DRIO model.

    Returns:
        targets: (N, D, T) - ground truth
        raw_predictions: (N, D, T) - raw model predictions
        gt_masks: (N, D, T) - gt_mask
        eval_masks: (N, D, T) - held-out entries mask
    """
    model.eval()

    all_targets = []
    all_raw_predictions = []
    all_gt_masks = []
    all_eval_masks = []

    with torch.no_grad():
        for batch in data_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            eval_mask = observed_mask - gt_mask

            # Raw model predictions
            predictions = model(observed_data, gt_mask)

            all_targets.append(observed_data.cpu().numpy())
            all_raw_predictions.append(predictions.cpu().numpy())
            all_gt_masks.append(gt_mask.cpu().numpy())
            all_eval_masks.append(eval_mask.cpu().numpy())

    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_raw_predictions, axis=0),
        np.concatenate(all_gt_masks, axis=0),
        np.concatenate(all_eval_masks, axis=0),
    )


# =============================================================================
# BSH_DRIO (BALANCED SINKHORN DRIO) MODEL TRAINING
# =============================================================================

def train_bsh_drio_epoch(model, bsh_drio_trainer, train_loader, optimizer, device):
    """Train BSH_DRIO model for one epoch using Algorithm 1 (balanced Sinkhorn)."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_sinkhorn = 0
    n_batches = 0

    for batch in train_loader:
        # Data comes in (B, T, D), model expects (B, D, T)
        observed_data = batch['observed_data'].float().to(device)  # (B, T, D)
        observed_mask = batch['observed_mask'].float().to(device)  # (B, T, D)
        gt_mask = batch['gt_mask'].float().to(device)              # (B, T, D)

        # Transpose to (B, D, T)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        # Use BSH_DRIOTrainer's train_step
        metrics = bsh_drio_trainer.train_step(
            model=model,
            observed_data=observed_data,
            observed_mask=observed_mask,
            gt_mask=gt_mask,
            optimizer=optimizer
        )

        total_loss += metrics['total_loss']
        total_recon_loss += metrics['reconstruction_loss']
        total_sinkhorn += metrics['sinkhorn_divergence']
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'reconstruction_loss': total_recon_loss / n_batches,
        'sinkhorn_divergence': total_sinkhorn / n_batches,
    }


def validate_bsh_drio(model, val_loader, device):
    """
    Validate BSH_DRIO model using self-supervised reconstruction error.

    IMPORTANT: Uses gt_mask (observed entries) for model selection, NOT eval_mask (held-out entries).

    This is NOT data leakage because:
    - gt_mask = entries the model sees during training (observed values)
    - eval_mask = held-out entries for final testing (never seen during training/validation)
    - Validating on gt_mask tests the model's ability to reconstruct observed entries
      on NEW samples (validation set), which ensures generalizability without leaking
      test set information.

    This approach:
    1. Avoids test set leakage (eval_mask never used for model selection)
    2. Validates generalization (validation samples are unseen during training)
    3. Is deployable (doesn't require held-out ground truth for validation)

    Returns:
        gt_loss: Self-supervised loss on gt_mask (used for model selection)
        eval_loss: Loss on eval_mask (for monitoring only, not used for selection)
    """
    model.eval()
    total_gt_loss = 0
    total_eval_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)  # Observed entries, NOT eval_mask

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            # Eval mask: held-out entries (for monitoring only)
            eval_mask = observed_mask - gt_mask

            predictions = model(observed_data, gt_mask)

            # Self-supervised validation: reconstruction error on gt_mask entries (used for model selection)
            # This validates generalization on NEW samples without using held-out test data
            gt_loss = ((predictions - observed_data) ** 2 * gt_mask).sum() / (gt_mask.sum() + 1e-10)

            # Eval loss: error on held-out entries (for monitoring only, NOT used for selection)
            eval_loss = ((predictions - observed_data) ** 2 * eval_mask).sum() / (eval_mask.sum() + 1e-10)

            total_gt_loss += gt_loss.item()
            total_eval_loss += eval_loss.item()
            n_batches += 1

    return total_gt_loss / n_batches, total_eval_loss / n_batches


def bsh_drio_hyperparameter_search(
    train_loader,
    val_loader,
    device,
    metadata,
    model_config,
    alpha_grid=None,
    gamma_grid=None,
    cv_epochs=None,
    model_class=None,
    model_kwargs=None,
):
    """
    Single-fold hyperparameter search for BSH_DRIO (Balanced Sinkhorn DRIO).

    Trains a model for each (alpha, gamma) combination on the training set
    and evaluates on the validation set. Returns the best hyperparameters.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        metadata: Data metadata with shape info
        model_config: Architecture configuration dict with 'model_type' and arch params
        alpha_grid: List of alpha values to search (default: BSH_DRIO_DEFAULT_ALPHA_GRID)
        gamma_grid: List of gamma values to search (default: BSH_DRIO_DEFAULT_GAMMA_GRID)
        cv_epochs: Number of epochs for CV (default: use BSH_DRIO_CONFIG)
        model_class: Optional custom model class (if None, uses create_imputer)
        model_kwargs: Optional kwargs for custom model class

    Returns:
        best_alpha: Best alpha value
        best_gamma: Best gamma value
        cv_results: Dict mapping (alpha, gamma) -> dict with val_loss, mse, w2
        best_model_state: State dict of the best model
    """
    if alpha_grid is None:
        alpha_grid = BSH_DRIO_DEFAULT_ALPHA_GRID
    if gamma_grid is None:
        gamma_grid = BSH_DRIO_DEFAULT_GAMMA_GRID

    d_features = metadata['shape'][2]
    d_time = metadata['shape'][1]
    train_config = BSH_DRIO_CONFIG['train'].copy()
    bsh_drio_config = BSH_DRIO_CONFIG['bsh_drio']

    # Override epochs if cv_epochs is provided
    if cv_epochs is not None:
        train_config['epochs'] = cv_epochs

    # Extract model_type and architecture params
    arch_config = model_config.copy()
    model_type = arch_config.pop('model_type')

    print(f"\n{'=' * 70}")
    print(f"BSH_DRIO Hyperparameter Search (Single-Fold) - {model_type}")
    print("=" * 70)
    print(f"Alpha grid: {alpha_grid}")
    print(f"Gamma grid: {gamma_grid}")
    print(f"Total combinations: {len(alpha_grid) * len(gamma_grid)}")
    print(f"Epochs per model: {train_config['epochs']}")
    print("Note: Using balanced Sinkhorn (no tau parameter)")

    cv_results = {}
    best_val_loss = float('inf')
    best_alpha = alpha_grid[0]
    best_gamma = gamma_grid[0]
    best_model_state = None

    combo_idx = 0
    total_combos = len(alpha_grid) * len(gamma_grid)

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{total_combos}] Testing alpha={alpha}, gamma={gamma}...")

            # Create fresh model
            if model_class is not None:
                # Use custom model class (e.g., BRITSForDRIO)
                model = model_class(**model_kwargs).to(device)
            else:
                # Use standard create_imputer
                model = create_imputer(
                    model_type=model_type,
                    d_features=d_features,
                    d_time=d_time,
                    **arch_config
                ).to(device)

            # Create BSH_DRIO trainer with current hyperparameters (no tau!)
            bsh_drio_trainer = create_bsh_drio_trainer(
                alpha=alpha,
                gamma=gamma,
                epsilon=bsh_drio_config['epsilon'],
                inner_steps=bsh_drio_config['inner_steps'],
                inner_lr=bsh_drio_config['inner_lr'],
                adaptive_epsilon=bsh_drio_config['adaptive_epsilon'],
                epsilon_quant=bsh_drio_config['epsilon_quant'],
                epsilon_mult=bsh_drio_config['epsilon_mult'],
            )

            # Create optimizer
            optimizer = Adam(
                model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config['weight_decay']
            )

            # Train for full epochs with validation at every epoch
            # Note: val_eval_loss is logged for monitoring but NOT used for model selection
            best_epoch_val_loss = float('inf')
            best_epoch_model_state = None
            for epoch in range(train_config['epochs']):
                train_metrics = train_bsh_drio_epoch(model, bsh_drio_trainer, train_loader, optimizer, device)

                # Validate at every epoch for BSH_DRIO CV
                val_gt_loss, val_eval_loss = validate_bsh_drio(model, val_loader, device)
                print(f"    Epoch {epoch+1}/{train_config['epochs']}: "
                      f"total_loss={train_metrics['total_loss']:.6f}, "
                      f"recon={train_metrics['reconstruction_loss']:.6f}, "
                      f"sinkhorn={train_metrics['sinkhorn_divergence']:.6f}, "
                      f"val_gt={val_gt_loss:.6f}, val_eval={val_eval_loss:.6f}")
                if val_gt_loss < best_epoch_val_loss:
                    best_epoch_val_loss = val_gt_loss
                    best_epoch_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Load best model state for this combination to compute final metrics
            if best_epoch_model_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_epoch_model_state.items()})

            # Compute MSE and W2 on validation set for this combination
            targets, predictions, gt_masks, eval_masks = generate_bsh_drio_predictions(model, val_loader, device)
            observed_mask = gt_masks + eval_masks
            val_eval_results = evaluate_imputation(
                X_true=targets,
                X_pred=predictions,
                mask=eval_masks,
                min_samples_w2=5,
                gt_mask=gt_masks,
                observed_mask=observed_mask
            )
            mse = val_eval_results['mse']
            w2_global = val_eval_results['w2_global']

            # Store results (selection is based on val_loss, but we track mse and w2 too)
            cv_results[(alpha, gamma)] = {
                'val_loss': best_epoch_val_loss,
                'mse': mse,
                'w2_global': w2_global,
            }
            print(f"    Final: val_loss={best_epoch_val_loss:.6f}, MSE={mse:.6f}, W2_global={w2_global:.6f}")

            # Selection based on val_loss (MSE on eval_mask during training)
            if best_epoch_val_loss < best_val_loss:
                best_val_loss = best_epoch_val_loss
                best_alpha = alpha
                best_gamma = gamma
                best_model_state = best_epoch_model_state

    # Print summary table of all combinations
    print(f"\n{'=' * 70}")
    print("BSH_DRIO CV Results Summary (sorted by val_loss)")
    print("=" * 70)
    print(f"{'Alpha':>8} | {'Gamma':>8} | {'Val Loss':>12} | {'MSE':>12} | {'W2_global':>12}")
    print("-" * 70)

    # Sort by val_loss
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['val_loss'])
    for (alpha, gamma), metrics in sorted_results:
        marker = " *" if alpha == best_alpha and gamma == best_gamma else ""
        print(f"{alpha:>8.2f} | {gamma:>8.2f} | {metrics['val_loss']:>12.6f} | {metrics['mse']:>12.6f} | {metrics['w2_global']:>12.6f}{marker}")

    print("-" * 70)
    print(f"* Best hyperparameters: alpha={best_alpha}, gamma={best_gamma}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print("=" * 70)

    return best_alpha, best_gamma, cv_results, best_model_state


def train_bsh_drio_model(model, train_loader, val_loader, save_dir, device, alpha=None, gamma=None):
    """
    Train BSH_DRIO model using Algorithm 1 with balanced Sinkhorn.

    1. Inner maximization: Update adversary Z to find worst-case distribution
    2. Outer minimization: Update imputer θ to minimize BSH_DRIO objective

    Args:
        model: The imputer model
        train_loader: Training data loader
        val_loader: Validation data loader
        save_dir: Directory to save model and results
        device: Device to use
        alpha: BSH_DRIO alpha parameter (default: use BSH_DRIO_CONFIG)
        gamma: BSH_DRIO gamma parameter (default: use BSH_DRIO_CONFIG)
    """
    print(f"\n{'=' * 70}")
    print("Training BSH_DRIO (Balanced Sinkhorn DRIO)")
    print("=" * 70)

    train_config = BSH_DRIO_CONFIG['train']
    bsh_drio_config = BSH_DRIO_CONFIG['bsh_drio']

    # Use provided alpha/gamma or fall back to config defaults
    alpha = alpha if alpha is not None else bsh_drio_config['alpha']
    gamma = gamma if gamma is not None else bsh_drio_config['gamma']

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Learning rate: {train_config['lr']}")
    print(f"  BSH_DRIO alpha: {alpha}")
    print(f"  BSH_DRIO gamma: {gamma}")
    print(f"  Inner steps: {bsh_drio_config['inner_steps']}")
    print("  Note: Using balanced Sinkhorn (no tau)")

    # Create BSH_DRIO trainer (no tau parameter!)
    bsh_drio_trainer = create_bsh_drio_trainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=bsh_drio_config['epsilon'],
        inner_steps=bsh_drio_config['inner_steps'],
        inner_lr=bsh_drio_config['inner_lr'],
        adaptive_epsilon=bsh_drio_config['adaptive_epsilon'],
        epsilon_quant=bsh_drio_config['epsilon_quant'],
        epsilon_mult=bsh_drio_config['epsilon_mult'],
    )

    optimizer = Adam(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )

    train_losses = []
    val_gt_losses = []
    val_eval_losses = []

    for epoch in tqdm(range(train_config['epochs']), desc="Training BSH_DRIO"):
        train_metrics = train_bsh_drio_epoch(model, bsh_drio_trainer, train_loader, optimizer, device)
        train_losses.append(train_metrics['total_loss'])

        # Validate at every epoch
        val_gt_loss, val_eval_loss = validate_bsh_drio(model, val_loader, device)
        val_gt_losses.append(val_gt_loss)
        val_eval_losses.append(val_eval_loss)

        tqdm.write(f"  Epoch {epoch+1}: total_loss={train_metrics['total_loss']:.6f}, "
                  f"recon={train_metrics['reconstruction_loss']:.6f}, "
                  f"sinkhorn={train_metrics['sinkhorn_divergence']:.6f}, "
                  f"val_gt={val_gt_loss:.6f}, val_eval={val_eval_loss:.6f}")

    # Save the final (last epoch) model - no early stopping in standalone training
    torch.save(model.state_dict(), save_dir / 'model.pth')

    # Save training curves
    np.savez(
        save_dir / 'training_curves.npz',
        train_losses=train_losses,
        val_gt_losses=val_gt_losses,
        val_eval_losses=val_eval_losses,
    )

    print(f"\nTraining complete! Final val_gt loss: {val_gt_losses[-1]:.6f}, val_eval loss: {val_eval_losses[-1]:.6f}")

    return model


def generate_bsh_drio_predictions(model, data_loader, device):
    """
    Generate predictions using BSH_DRIO model.

    Returns:
        targets: (N, D, T) - ground truth
        raw_predictions: (N, D, T) - raw model predictions
        gt_masks: (N, D, T) - gt_mask
        eval_masks: (N, D, T) - held-out entries mask
    """
    model.eval()

    all_targets = []
    all_raw_predictions = []
    all_gt_masks = []
    all_eval_masks = []

    with torch.no_grad():
        for batch in data_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            eval_mask = observed_mask - gt_mask

            # Raw model predictions
            predictions = model(observed_data, gt_mask)

            all_targets.append(observed_data.cpu().numpy())
            all_raw_predictions.append(predictions.cpu().numpy())
            all_gt_masks.append(gt_mask.cpu().numpy())
            all_eval_masks.append(eval_mask.cpu().numpy())

    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_raw_predictions, axis=0),
        np.concatenate(all_gt_masks, axis=0),
        np.concatenate(all_eval_masks, axis=0),
    )


# =============================================================================
# DRIO V2 (WITH CSDI-STYLE INTERNAL MASKING) MODEL TRAINING
# =============================================================================

def train_drio_v2_epoch(model, drio_v2_trainer, train_loader, optimizer, device):
    """Train DRIO v2 model for one epoch using CSDI-style internal masking."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_sinkhorn = 0
    total_mask_ratio = 0
    n_batches = 0

    for batch in train_loader:
        # Data comes in (B, T, D), model expects (B, D, T)
        observed_data = batch['observed_data'].float().to(device)  # (B, T, D)
        observed_mask = batch['observed_mask'].float().to(device)  # (B, T, D)
        gt_mask = batch['gt_mask'].float().to(device)              # (B, T, D)

        # Transpose to (B, D, T)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        # Use DRIOv2Trainer's train_step (with internal masking)
        metrics = drio_v2_trainer.train_step(
            model=model,
            observed_data=observed_data,
            observed_mask=observed_mask,
            gt_mask=gt_mask,
            optimizer=optimizer
        )

        total_loss += metrics['total_loss']
        total_recon_loss += metrics['reconstruction_loss']
        total_sinkhorn += metrics['sinkhorn_divergence']
        total_mask_ratio += metrics.get('target_mask_ratio', 0.5)
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'reconstruction_loss': total_recon_loss / n_batches,
        'sinkhorn_divergence': total_sinkhorn / n_batches,
        'avg_mask_ratio': total_mask_ratio / n_batches,
    }


def validate_drio_v2(model, val_loader, device):
    """
    Validate DRIO v2 model using CSDI-style validation approach.

    Like CSDI, validation uses gt_mask directly as cond_mask (no random masking).
    Computes loss on both gt_mask and eval_mask.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to use

    Returns:
        val_gt_loss: Average loss on gt_mask (for model selection during CV)
        val_eval_loss: Average loss on eval_mask (held-out test entries, for reference)
    """
    model.eval()
    total_gt_loss = 0
    total_eval_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            observed_data = batch['observed_data'].float().to(device)
            observed_mask = batch['observed_mask'].float().to(device)
            gt_mask = batch['gt_mask'].float().to(device)

            # Transpose to (B, D, T)
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            eval_mask = observed_mask - gt_mask

            # Use gt_mask as cond_mask (like CSDI validation)
            cond_mask = gt_mask
            predictions = model(observed_data, cond_mask)

            # Loss on gt_mask (observed entries - for model selection)
            gt_loss = ((predictions - observed_data) ** 2 * gt_mask).sum() / (gt_mask.sum() + 1e-10)
            total_gt_loss += gt_loss.item()

            # Loss on eval_mask (held-out test entries - for reference)
            eval_loss = ((predictions - observed_data) ** 2 * eval_mask).sum() / (eval_mask.sum() + 1e-10)
            total_eval_loss += eval_loss.item()

            n_batches += 1

    return total_gt_loss / n_batches, total_eval_loss / n_batches


def drio_v2_hyperparameter_search(
    train_loader,
    val_loader,
    device,
    metadata,
    model_config,
    alpha_grid=None,
    gamma_grid=None,
    cv_epochs=None,
    model_class=None,
    model_kwargs=None,
):
    """
    Single-fold hyperparameter search for DRIO v2 (with CSDI-style internal masking).

    Same structure as DRIO v1 hyperparameter search, but uses DRIOv2Trainer.
    Uses DRIO lookup tables for alpha/gamma/epochs (same as drio_hyperparameter_search).
    """
    # Get optimized CV config from DRIO lookup table (alpha_grid, gamma_grid, epochs)
    lookup_alpha_grid, lookup_gamma_grid, lookup_epochs = get_drio_cv_config(
        metadata['dataset'],
        metadata['missing_type'],
        metadata['missing_ratio']
    )

    # Use lookup values if not overridden by arguments
    if alpha_grid is None:
        alpha_grid = lookup_alpha_grid
    if gamma_grid is None:
        gamma_grid = lookup_gamma_grid
    if cv_epochs is None:
        cv_epochs = lookup_epochs

    d_features = metadata['shape'][2]
    d_time = metadata['shape'][1]

    # Use cv_epochs (either from argument or lookup table)
    n_epochs = cv_epochs

    print(f"\n{'=' * 100}")
    print("DRIO V2 Hyperparameter Search (with CSDI-style internal masking)")
    print("=" * 100)
    print(f"Alpha grid: {alpha_grid}")
    print(f"Gamma grid: {gamma_grid}")
    print(f"Epochs per combination: {n_epochs}")
    print(f"Total combinations: {len(alpha_grid) * len(gamma_grid)}")
    print("=" * 100)

    cv_results = {}
    best_val_loss = float('inf')
    best_alpha = None
    best_gamma = None
    best_model_state = None

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            print(f"\n--- Training with alpha={alpha}, gamma={gamma} ---")

            # Create fresh model for each combination
            if model_class is not None:
                model = model_class(**model_kwargs).to(device)
            else:
                model_config_copy = model_config.copy()
                model_type = model_config_copy.pop('model_type')
                model = create_imputer(
                    model_type=model_type,
                    d_features=d_features,
                    d_time=d_time,
                    **model_config_copy
                ).to(device)

            # Create DRIO v2 trainer
            drio_v2_config = DRIO_V2_CONFIG['drio_v2']
            drio_v2_trainer = create_drio_v2_trainer(
                alpha=alpha,
                gamma=gamma,
                epsilon=drio_v2_config['epsilon'],
                tau=drio_v2_config['tau'],
                inner_steps=drio_v2_config['inner_steps'],
                inner_lr=drio_v2_config['inner_lr'],
                mask_ratio_range=drio_v2_config['mask_ratio_range'],
                adaptive_epsilon=drio_v2_config['adaptive_epsilon'],
                epsilon_quant=drio_v2_config['epsilon_quant'],
                epsilon_mult=drio_v2_config['epsilon_mult'],
            )

            optimizer = Adam(model.parameters(), lr=DRIO_V2_CONFIG['train']['lr'],
                           weight_decay=DRIO_V2_CONFIG['train']['weight_decay'])

            # Train for full epochs (no early stopping / best epoch tracking)
            for epoch in range(n_epochs):
                train_metrics = train_drio_v2_epoch(model, drio_v2_trainer, train_loader, optimizer, device)
                val_gt_loss, val_eval_loss = validate_drio_v2(model, val_loader, device)

                print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_metrics['total_loss']:.6f}, "
                      f"val_gt={val_gt_loss:.6f}, val_eval={val_eval_loss:.6f}, mask_ratio={train_metrics['avg_mask_ratio']:.2f}")

            # Use final model state after full training
            final_val_loss, final_val_eval = validate_drio_v2(model, val_loader, device)
            final_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Compute additional metrics on validation set
            targets, preds, gt_masks, eval_masks = generate_drio_predictions(model, val_loader, device)
            observed_mask = gt_masks + eval_masks
            eval_results = evaluate_imputation(
                targets, preds, eval_masks, min_samples_w2=5,
                gt_mask=gt_masks, observed_mask=observed_mask
            )

            cv_results[(alpha, gamma)] = {
                'val_loss': final_val_loss,
                'val_eval': final_val_eval,
                'epochs': n_epochs,
                'mse': eval_results['mse'],
                'w2_global': eval_results['w2_global'],
                'cvar': eval_results['se_summary']['p90'],
            }

            print(f"  Final (epoch {n_epochs}): val_loss: {final_val_loss:.6f}, val_eval: {final_val_eval:.6f}, "
                  f"mse: {eval_results['mse']:.6f}, w2_global: {eval_results['w2_global']:.6f}")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_alpha = alpha
                best_gamma = gamma
                best_model_state = final_model_state

    # Print CV results summary (sorted by val_loss)
    print(f"\n{'=' * 100}")
    print("DRIO V2 CV Results Summary (sorted by val_loss)")
    print("=" * 100)
    print(f"{'Alpha':>8} | {'Gamma':>8} | {'Val Loss':>12} | {'Val Eval':>12} | {'Epochs':>6} | {'MSE':>12} | {'W2_global':>12} | {'CVaR':>12}")
    print("-" * 100)

    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['val_loss'])
    for (alpha, gamma), metrics in sorted_results:
        marker = " *" if alpha == best_alpha and gamma == best_gamma else ""
        print(f"{alpha:>8.2f} | {gamma:>8.2f} | {metrics['val_loss']:>12.6f} | {metrics['val_eval']:>12.6f} | {metrics['epochs']:>6} | {metrics['mse']:>12.6f} | {metrics['w2_global']:>12.6f} | {metrics['cvar']:>12.6f}{marker}")

    print("-" * 100)
    print(f"* Best hyperparameters (by val_loss): alpha={best_alpha}, gamma={best_gamma}")
    print("=" * 100)

    # Print CV results summary (sorted by CVaR)
    print(f"\n{'=' * 100}")
    print("DRIO V2 CV Results Summary (sorted by CVaR)")
    print("=" * 100)
    print(f"{'Alpha':>8} | {'Gamma':>8} | {'Val Loss':>12} | {'Val Eval':>12} | {'Epochs':>6} | {'MSE':>12} | {'W2_global':>12} | {'CVaR':>12}")
    print("-" * 100)

    sorted_by_cvar = sorted(cv_results.items(), key=lambda x: x[1]['cvar'])
    best_cvar_alpha, best_cvar_gamma = sorted_by_cvar[0][0]
    best_cvar_value = sorted_by_cvar[0][1]['cvar']
    for (alpha, gamma), metrics in sorted_by_cvar:
        marker = " *" if alpha == best_cvar_alpha and gamma == best_cvar_gamma else ""
        print(f"{alpha:>8.2f} | {gamma:>8.2f} | {metrics['val_loss']:>12.6f} | {metrics['val_eval']:>12.6f} | {metrics['epochs']:>6} | {metrics['mse']:>12.6f} | {metrics['w2_global']:>12.6f} | {metrics['cvar']:>12.6f}{marker}")

    print("-" * 100)
    print(f"* Best hyperparameters (by CVaR): alpha={best_cvar_alpha}, gamma={best_cvar_gamma}")
    print(f"  Best CVaR (p90 of SE): {best_cvar_value:.6f}")
    print("=" * 100)

    return best_alpha, best_gamma, cv_results, best_model_state


def train_drio_v2_model(model, train_loader, val_loader, save_dir, device, alpha=None, gamma=None, metadata=None):
    """
    Train DRIO v2 model using CSDI-style internal masking.

    Key difference from DRIO v1:
    - Uses random internal masking during training (like CSDI)
    - MSE computed on artificially masked entries
    - Sinkhorn on complete trajectories
    """
    print(f"\n{'=' * 70}")
    print("Training DRIO V2 (with CSDI-style Internal Masking)")
    print("=" * 70)

    train_config = DRIO_V2_CONFIG['train'].copy()
    drio_v2_config = DRIO_V2_CONFIG['drio_v2']

    # Get epochs: standalone override > lookup table > default
    if metadata is not None:
        if '_standalone_epochs' in metadata:
            # Standalone mode with explicit epochs from lookup
            train_config['epochs'] = metadata['_standalone_epochs']
        else:
            # CV mode - use epochs from CV lookup table
            optimal_epochs = get_drio_epochs(
                metadata['dataset'],
                metadata['missing_type'],
                metadata['missing_ratio']
            )
            train_config['epochs'] = optimal_epochs
        print(f"Using dataset-specific epochs: {train_config['epochs']}")

    # Use provided alpha/gamma or defaults
    if alpha is None:
        alpha = drio_v2_config['alpha']
    if gamma is None:
        gamma = drio_v2_config['gamma']

    # Create DRIO v2 trainer
    drio_v2_trainer = create_drio_v2_trainer(
        alpha=alpha,
        gamma=gamma,
        epsilon=drio_v2_config['epsilon'],
        tau=drio_v2_config['tau'],
        inner_steps=drio_v2_config['inner_steps'],
        inner_lr=drio_v2_config['inner_lr'],
        mask_ratio_range=drio_v2_config['mask_ratio_range'],
        adaptive_epsilon=drio_v2_config['adaptive_epsilon'],
        epsilon_quant=drio_v2_config['epsilon_quant'],
        epsilon_mult=drio_v2_config['epsilon_mult'],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Learning rate: {train_config['lr']}")
    print(f"  Alpha: {alpha}, Gamma: {gamma}")
    print(f"  Mask ratio range: {drio_v2_config['mask_ratio_range']}")

    optimizer = Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])

    train_losses = []
    val_gt_losses = []
    val_eval_losses = []
    best_val_loss = float('inf')

    pbar = tqdm(range(train_config['epochs']), desc="Training")
    for epoch in pbar:
        train_metrics = train_drio_v2_epoch(model, drio_v2_trainer, train_loader, optimizer, device)
        train_losses.append(train_metrics['total_loss'])

        val_gt_loss, val_eval_loss = validate_drio_v2(model, val_loader, device)
        val_gt_losses.append(val_gt_loss)
        val_eval_losses.append(val_eval_loss)

        pbar.set_postfix({
            'train': f"{train_metrics['total_loss']:.4f}",
            'val_gt': f"{val_gt_loss:.4f}",
            'val_eval': f"{val_eval_loss:.4f}",
            'mask_r': f"{train_metrics['avg_mask_ratio']:.2f}"
        })

        if val_gt_loss < best_val_loss:
            best_val_loss = val_gt_loss
            torch.save(model.state_dict(), save_dir / 'model.pth')
            tqdm.write(f"  Epoch {epoch+1}: val_gt={val_gt_loss:.6f} (best), val_eval={val_eval_loss:.6f}")

    # Save training curves
    np.savez(
        save_dir / 'training_curves.npz',
        train_losses=train_losses,
        val_gt_losses=val_gt_losses,
        val_eval_losses=val_eval_losses,
    )

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load(save_dir / 'model.pth', map_location=device))

    return model


def generate_drio_v2_predictions(model, data_loader, device):
    """
    Generate predictions using DRIO v2 model.

    Same as DRIO v1 - at inference time, no internal masking is used.
    """
    return generate_drio_predictions(model, data_loader, device)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(targets, predictions, mask, dataset, description="",
                         gt_mask=None, observed_mask=None):
    """
    Evaluate predictions and return metrics.

    Args:
        targets: (N, D, T) - ground truth (normalized)
        predictions: (N, D, T) - predictions (normalized)
        mask: (N, D, T) - mask for evaluation (eval_mask for val/test, gt_mask for train)
        dataset: ImputationDataset for denormalization
        description: Description string for printing
        gt_mask: (N, D, T) - mask of entries used for conditioning (1 = observed by model).
                 Required for global W2 computation.
        observed_mask: (N, D, T) - mask of all entries with ground truth (1 = not naturally missing).
                       Required for global W2 computation.

    Returns:
        results: Dictionary with metrics (computed on normalized scale, consistent with CSDI benchmark)
    """
    # Evaluate on normalized scale (consistent with CSDI benchmark which uses scaler=1)
    results = evaluate_imputation(
        X_true=targets,
        X_pred=predictions,
        mask=mask,
        min_samples_w2=5,
        gt_mask=gt_mask,
        observed_mask=observed_mask
    )

    return results


def create_results_table(all_results, split_name):
    """
    Create markdown tables for results with all percentiles.

    Args:
        all_results: Dictionary mapping model_name -> results dict
        split_name: 'train', 'val', or 'test'

    Returns:
        Markdown table string with MSE, MAE, and W2 statistics
    """
    lines = []
    lines.append(f"## {split_name.capitalize()} Error Metrics (Normalized Scale)\n")

    # Summary table with key metrics
    lines.append("### Summary\n")
    lines.append("| Model | RMSE | MAE | W2 (global) |")
    lines.append("|-------|------|-----|-------------|")

    for model_name, results in sorted(all_results.items()):
        rmse = results.get('rmse', np.nan)
        mae = results['mae']
        w2_global = results.get('w2_global', np.nan)
        w2_global_str = f"{w2_global:.6f}" if not np.isnan(w2_global) else "N/A"
        lines.append(f"| {model_name} | {rmse:.6f} | {mae:.6f} | {w2_global_str} |")

    lines.append("")

    # MSE/SE table
    lines.append("### Squared Error (SE) Statistics\n")
    lines.append("| Model | Mean (MSE) | Median | p90 | p95 | p99 |")
    lines.append("|-------|------------|--------|-----|-----|-----|")

    for model_name, results in sorted(all_results.items()):
        mse = results['mse']
        se_summary = results.get('se_summary', {})
        se_p50 = se_summary.get('p50', np.nan)
        se_p90 = se_summary.get('p90', np.nan)
        se_p95 = se_summary.get('p95', np.nan)
        se_p99 = se_summary.get('p99', np.nan)
        lines.append(f"| {model_name} | {mse:.6f} | {se_p50:.6f} | {se_p90:.6f} | {se_p95:.6f} | {se_p99:.6f} |")

    lines.append("")

    # MAE/AE table
    lines.append("### Absolute Error (AE) Statistics\n")
    lines.append("| Model | Mean (MAE) | Median | p90 | p95 | p99 |")
    lines.append("|-------|------------|--------|-----|-----|-----|")

    for model_name, results in sorted(all_results.items()):
        mae = results['mae']
        ae_summary = results.get('ae_summary', {})
        ae_p50 = ae_summary.get('p50', np.nan)
        ae_p90 = ae_summary.get('p90', np.nan)
        ae_p95 = ae_summary.get('p95', np.nan)
        ae_p99 = ae_summary.get('p99', np.nan)
        lines.append(f"| {model_name} | {mae:.6f} | {ae_p50:.6f} | {ae_p90:.6f} | {ae_p95:.6f} | {ae_p99:.6f} |")

    return "\n".join(lines)


def save_predictions(predictions, targets, mask, save_dir, model_name, split_name, shape_info):
    """
    Save predictions in npz and csv formats.

    Args:
        predictions: (N, D, T) array
        targets: (N, D, T) array
        mask: (N, D, T) array
        save_dir: Directory to save
        model_name: Model name
        split_name: 'train', 'val', or 'test'
        shape_info: Dict with N, D, T info
    """
    # Save as npz (original shape)
    npz_path = save_dir / f"predictions_{split_name}.npz"
    np.savez(
        npz_path,
        predictions=predictions,
        targets=targets,
        mask=mask,
        shape_info=f"Shape: (N={shape_info['N']}, D={shape_info['D']}, T={shape_info['T']}) - "
                   f"N=samples, D=features, T=time"
    )

    # Save as csv (flattened)
    csv_path = save_dir / f"predictions_{split_name}.csv"
    predictions_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Create header with shape info
    header = (f"# Predictions for {model_name} on {split_name} set\n"
              f"# Original shape: (N={shape_info['N']}, D={shape_info['D']}, T={shape_info['T']})\n"
              f"# N=samples (batch dimension), D=features, T=time steps\n"
              f"# Flattened order: iterate over N, then D, then T\n"
              f"prediction,target,mask")

    data = np.column_stack([predictions_flat, targets_flat, mask_flat])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.8f')

    print(f"  Saved {split_name} predictions: {npz_path}, {csv_path}")


# =============================================================================
# MAIN TRAINING AND EVALUATION
# =============================================================================

def run_model(model_name, train_dataset, val_dataset, test_dataset,
              train_loader, val_loader, test_loader,
              save_dir, device, metadata,
              drio_cv=True, drio_alpha_grid=None, drio_gamma_grid=None, drio_cv_epochs=None,
              bsh_drio_cv=True, bsh_drio_alpha_grid=None, bsh_drio_gamma_grid=None):
    """
    Run training and evaluation for a single model.

    Args:
        model_name: Name of the model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        save_dir: Directory to save results
        device: Device to use
        metadata: Data metadata
        drio_cv: Whether to run hyperparameter search for DRIO (default: True)
        drio_alpha_grid: Grid of alpha values for DRIO CV (default: DRIO_DEFAULT_ALPHA_GRID)
        drio_gamma_grid: Grid of gamma values for DRIO CV (default: DRIO_DEFAULT_GAMMA_GRID)
        drio_cv_epochs: Number of epochs for DRIO CV (default: use dataset-specific lookup)
        bsh_drio_cv: Whether to run hyperparameter search for BSH_DRIO (default: True)
        bsh_drio_alpha_grid: Grid of alpha values for BSH_DRIO CV (default: BSH_DRIO_DEFAULT_ALPHA_GRID)
        bsh_drio_gamma_grid: Grid of gamma values for BSH_DRIO CV (default: BSH_DRIO_DEFAULT_GAMMA_GRID)

    Returns:
        results: Dict with train/val/test results
    """
    d_features = metadata['shape'][2]  # D
    d_time = metadata['shape'][1]      # T

    model_save_dir = save_dir / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    results = {'train': None, 'val': None, 'test': None}

    # Training and prediction generation
    if model_name == 'mse_brits':
        # MSE_BRITS: BRITS architecture with MSE training objective
        # Create BRITSForDRIO model (wrapper that adapts BRITS to MSE interface)
        model = BRITSForDRIO(
            d_features=d_features,
            d_time=d_time,
            rnn_hid_size=MSE_BRITS_CONFIG['rnn_hid_size']
        ).to(device)

        # Save config
        config_to_save = {
            'train': MSE_TRAIN_CONFIG,
            'model': {'type': 'mse_brits', **MSE_BRITS_CONFIG},
        }
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Train with MSE objective
        model = train_mse_model(model_name, model, train_loader, val_loader, model_save_dir, device)

        # Generate predictions
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_mse_predictions(model, train_loader, device)
        val_preds = generate_mse_predictions(model, val_loader, device)
        test_preds = generate_mse_predictions(model, test_loader, device)


    elif model_name == 'csdi':
        # CSDI model
        model = train_csdi_model(train_loader, val_loader, model_save_dir, device, d_features)

        # Generate predictions
        print(f"\nGenerating predictions for CSDI...")
        train_preds = generate_csdi_predictions(model, train_loader, device)
        val_preds = generate_csdi_predictions(model, val_loader, device)
        test_preds = generate_csdi_predictions(model, test_loader, device)

    elif model_name == 'mdot':
        # MissingDataOT
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float64)

        train_mdot_model(train_dataset, val_dataset, test_dataset, model_save_dir, device)

        # Generate predictions (imputation happens here)
        print(f"\nGenerating predictions for MissingDataOT...")
        train_preds = generate_mdot_predictions(train_dataset, device, "train")
        val_preds = generate_mdot_predictions(val_dataset, device, "val")
        test_preds = generate_mdot_predictions(test_dataset, device, "test")

        # Reset dtype
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float32)

    elif model_name == 'brits':
        # BRITS model
        model = train_brits_model(train_dataset, val_dataset, model_save_dir, device, d_features, d_time)

        # Generate predictions
        print(f"\nGenerating predictions for BRITS...")
        train_preds = generate_brits_predictions(model, train_dataset, device, d_features, d_time)
        val_preds = generate_brits_predictions(model, val_dataset, device, d_features, d_time)
        test_preds = generate_brits_predictions(model, test_dataset, device, d_features, d_time)


    elif model_name == 'psw':
        # PSW-I (Proximal Spectrum Wasserstein Imputation) model
        # PSW-I is optimization-based: no training needed, directly impute test data
        train_psw_model(train_dataset, val_dataset, model_save_dir, device)

        # Only impute test data (PSW doesn't learn from training data)
        print(f"\nGenerating predictions for PSW-I (test set only)...")
        test_preds = generate_psw_predictions(test_dataset, device, split_name='test')

        # Set train/val to None (not needed for PSW)
        train_preds = None
        val_preds = None

    elif model_name == 'notmiwae':
        # not-MIWAE: MIWAE with missing process modeling (PyTorch)
        # Handles Missing Not At Random (MNAR) data by jointly modeling p(x,s)
        model = train_notmiwae_model(train_dataset, val_dataset, model_save_dir, device)

        # Generate predictions for all splits
        print(f"\nGenerating predictions for not-MIWAE...")
        train_preds = generate_notmiwae_predictions(model, train_dataset, device, split_name='train')
        val_preds = generate_notmiwae_predictions(model, val_dataset, device, split_name='val')
        test_preds = generate_notmiwae_predictions(model, test_dataset, device, split_name='test')

    elif model_name in ['mean',  'mf']:
        # Simple baseline methods (mean,  Matrix Factorization)
        # These methods don't have a separate train phase - they fit_transform directly

        print(f"\n{'=' * 70}")
        print(f"Running {model_name.upper()} imputation")
        print("=" * 70)

        # Get config for the method
        if model_name == 'mean':
            config = MEAN_CONFIG
        elif model_name == 'mf':
            config = MF_CONFIG.copy()
            config['device'] = str(device)

        # Save config
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        def generate_simple_predictions(dataset, method, config, split_name):
            """Generate predictions using simple imputation methods."""
            N = len(dataset)
            T = dataset[0]['observed_data'].shape[0]
            D = dataset[0]['observed_data'].shape[1]

            # Collect data in (N, T, D) format
            observed_data = np.stack([dataset[i]['observed_data'] for i in range(N)], axis=0)
            observed_mask = np.stack([dataset[i]['observed_mask'] for i in range(N)], axis=0)
            gt_mask = np.stack([dataset[i]['gt_mask'] for i in range(N)], axis=0)

            # Compute eval_mask
            eval_mask = observed_mask - gt_mask  # (N, T, D)

            # Create input with NaN for entries to impute (where gt_mask=0)
            X_input = observed_data.copy()
            X_input[gt_mask == 0] = np.nan

            print(f"\n  Imputing {split_name} set (N={N}, T={T}, D={D})...")

            # Create imputer and run
            imputer = create_simple_imputer(method, **config)
            X_imputed = imputer.fit_transform(X_input)

            # Transpose to (N, D, T) format for consistency with other models
            targets = observed_data.transpose(0, 2, 1)
            predictions = X_imputed.transpose(0, 2, 1)
            gt_masks = gt_mask.transpose(0, 2, 1)
            eval_masks = eval_mask.transpose(0, 2, 1)

            return targets, predictions, gt_masks, eval_masks

        # Generate predictions for all splits
        print(f"\nGenerating predictions for {model_name.upper()}...")
        train_preds = generate_simple_predictions(train_dataset, model_name, config, 'train')
        val_preds = generate_simple_predictions(val_dataset, model_name, config, 'val')
        test_preds = generate_simple_predictions(test_dataset, model_name, config, 'test')

    elif model_name == 'drio_brits':
        # DRIO-BRITS: BRITS architecture with DRIO training objective
        # Create BRITSForDRIO model (wrapper that adapts BRITS to DRIO interface)
        model = BRITSForDRIO(
            d_features=d_features,
            d_time=d_time,
            rnn_hid_size=DRIO_BRITS_CONFIG['rnn_hid_size']
        ).to(device)

        # Hyperparameter selection (same as other DRIO models)
        if drio_cv:
            # Run single-fold hyperparameter search
            # Note: We pass model_config=None since BRITSForDRIO handles its own architecture
            best_alpha, best_gamma, cv_results, best_model_state = drio_hyperparameter_search(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                metadata=metadata,
                model_config={'model_type': 'brits_for_drio', 'rnn_hid_size': DRIO_BRITS_CONFIG['rnn_hid_size']},
                alpha_grid=drio_alpha_grid,
                gamma_grid=drio_gamma_grid,
                cv_epochs=drio_cv_epochs,
                model_class=BRITSForDRIO,
                model_kwargs={'d_features': d_features, 'd_time': d_time, 'rnn_hid_size': DRIO_BRITS_CONFIG['rnn_hid_size']},
            )

            # Save CV results
            cv_results_serializable = {str(k): v for k, v in cv_results.items()}
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_gamma': best_gamma,
                    'alpha_grid': drio_alpha_grid or DRIO_DEFAULT_ALPHA_GRID,
                    'gamma_grid': drio_gamma_grid or DRIO_DEFAULT_GAMMA_GRID,
                    'results': cv_results_serializable,
                }, f, indent=2)

            # Load best state from CV
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            torch.save(model.state_dict(), model_save_dir / 'model.pth')
            print(f"\nLoaded best model from CV search (alpha={best_alpha}, gamma={best_gamma})")

        else:
            # No CV - use lookup table or defaults
            standalone_params = get_drio_standalone_params(
                metadata['dataset'], metadata['missing_type'], metadata['missing_ratio']
            )

            if standalone_params is not None:
                best_alpha, best_gamma, standalone_epochs = standalone_params
                print(f"\nUsing STANDALONE LOOKUP: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")
            else:
                # Default parameters when not in lookup table
                best_alpha, best_gamma, standalone_epochs = 0.99, 1, 30
                print(f"\nUsing DEFAULT standalone parameters: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")
            # Override epochs in metadata for train_drio_model
            metadata_with_epochs = metadata.copy()
            metadata_with_epochs['_standalone_epochs'] = standalone_epochs

            # Train with DRIO objective
            model = train_drio_model(
                model, train_loader, val_loader, model_save_dir, device,
                alpha=best_alpha, gamma=best_gamma, metadata=metadata_with_epochs
            )

            # Load training curves to get final validation losses
            training_curves = np.load(model_save_dir / 'training_curves.npz')
            final_val_loss = float(training_curves['val_gt_losses'][-1])
            final_val_eval = float(training_curves['val_eval_losses'][-1])

            # Save cv_results.json for compatibility with analysis scripts
            cv_results_standalone = {
                'best_alpha': best_alpha,
                'best_gamma': best_gamma,
                'alpha_grid': [best_alpha],
                'gamma_grid': [best_gamma],
                'results': {
                    f"({best_alpha}, {best_gamma})": {
                        'val_loss': final_val_loss,
                        'val_eval': final_val_eval,
                        'epoch': standalone_epochs,
                        'mse': final_val_eval,
                    }
                },
                'standalone_mode': True,
            }
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump(cv_results_standalone, f, indent=2)

        # Save config
        config_to_save = {
            'train': DRIO_CONFIG['train'],
            'drio': DRIO_CONFIG['drio'].copy(),
            'model': {'type': 'drio_brits', **DRIO_BRITS_CONFIG},
        }
        config_to_save['drio']['alpha'] = best_alpha
        config_to_save['drio']['gamma'] = best_gamma
        config_to_save['cv_enabled'] = drio_cv
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Generate predictions (same as other DRIO models)
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_drio_predictions(model, train_loader, device)
        val_preds = generate_drio_predictions(model, val_loader, device)
        test_preds = generate_drio_predictions(model, test_loader, device)

    elif model_name == 'drio_v2_brits':
        # DRIO V2 with BRITS: BRITS architecture with DRIO v2 training (CSDI-style internal masking)
        # Create BRITSForDRIO model (wrapper that adapts BRITS to DRIO interface)
        model = BRITSForDRIO(
            d_features=d_features,
            d_time=d_time,
            rnn_hid_size=DRIO_V2_BRITS_CONFIG['rnn_hid_size']
        ).to(device)

        # Hyperparameter selection (same as DRIO v1 but uses v2 trainer)
        if drio_cv:
            # Run single-fold hyperparameter search with DRIO v2 trainer
            best_alpha, best_gamma, cv_results, best_model_state = drio_v2_hyperparameter_search(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                metadata=metadata,
                model_config={'model_type': 'brits_for_drio', 'rnn_hid_size': DRIO_V2_BRITS_CONFIG['rnn_hid_size']},
                alpha_grid=drio_alpha_grid,
                gamma_grid=drio_gamma_grid,
                cv_epochs=drio_cv_epochs,
                model_class=BRITSForDRIO,
                model_kwargs={'d_features': d_features, 'd_time': d_time, 'rnn_hid_size': DRIO_V2_BRITS_CONFIG['rnn_hid_size']},
            )

            # Save CV results
            cv_results_serializable = {str(k): v for k, v in cv_results.items()}
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_gamma': best_gamma,
                    'alpha_grid': drio_alpha_grid or DRIO_V2_DEFAULT_ALPHA_GRID,
                    'gamma_grid': drio_gamma_grid or DRIO_V2_DEFAULT_GAMMA_GRID,
                    'results': cv_results_serializable,
                }, f, indent=2)

            # Load best state from CV
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            torch.save(model.state_dict(), model_save_dir / 'model.pth')
            print(f"\nLoaded best model from CV search (alpha={best_alpha}, gamma={best_gamma})")

        else:
            # No CV - use DRIO lookup table or defaults (same as drio_brits)
            standalone_params = get_drio_standalone_params(
                metadata['dataset'], metadata['missing_type'], metadata['missing_ratio']
            )

            if standalone_params is not None:
                best_alpha, best_gamma, standalone_epochs = standalone_params
                print(f"\nUsing DRIO STANDALONE LOOKUP: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")
            else:
                # Default parameters when not in lookup table
                best_alpha, best_gamma, standalone_epochs = 0.99, 1, 30
                print(f"\nUsing DEFAULT standalone parameters: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")

            # Override epochs in metadata for train_drio_v2_model
            metadata_with_epochs = metadata.copy()
            metadata_with_epochs['_standalone_epochs'] = standalone_epochs

            # Train with DRIO v2 objective
            model = train_drio_v2_model(
                model, train_loader, val_loader, model_save_dir, device,
                alpha=best_alpha, gamma=best_gamma, metadata=metadata_with_epochs
            )

            # Load training curves to get final validation losses
            training_curves = np.load(model_save_dir / 'training_curves.npz')
            final_val_loss = float(training_curves['val_gt_losses'][-1])
            final_val_eval = float(training_curves['val_eval_losses'][-1])

            # Save cv_results.json for compatibility with analysis scripts
            cv_results_standalone = {
                'best_alpha': best_alpha,
                'best_gamma': best_gamma,
                'alpha_grid': [best_alpha],
                'gamma_grid': [best_gamma],
                'results': {
                    f"({best_alpha}, {best_gamma})": {
                        'val_loss': final_val_loss,
                        'val_eval': final_val_eval,
                        'epoch': standalone_epochs,
                    }
                },
                'standalone_mode': True,
                'lookup_table_used': standalone_params is not None,
            }
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump(cv_results_standalone, f, indent=2)

        # Save config
        config_to_save = {
            'train': DRIO_V2_CONFIG['train'],
            'drio_v2': DRIO_V2_CONFIG['drio_v2'].copy(),
            'model': {'type': 'drio_v2_brits', **DRIO_V2_BRITS_CONFIG},
        }
        config_to_save['drio_v2']['alpha'] = best_alpha
        config_to_save['drio_v2']['gamma'] = best_gamma
        config_to_save['cv_enabled'] = drio_cv
        with open(model_save_dir / 'config.json', 'w') as f:
            # Convert tuple to list for JSON serialization
            config_json = config_to_save.copy()
            config_json['drio_v2'] = config_to_save['drio_v2'].copy()
            config_json['drio_v2']['mask_ratio_range'] = list(config_to_save['drio_v2']['mask_ratio_range'])
            json.dump(config_json, f, indent=2)

        # Generate predictions (same as other DRIO models)
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_drio_v2_predictions(model, train_loader, device)
        val_preds = generate_drio_v2_predictions(model, val_loader, device)
        test_preds = generate_drio_v2_predictions(model, test_loader, device)

    elif model_name.startswith('drio'):
        # DRIO variants (drio_transformer, drio_lstm, drio_gat, drio_mlp, drio_sttransformer)
        model_config = DRIO_MODEL_CONFIGS[model_name].copy()
        model_type = model_config.pop('model_type')

        # Hyperparameter selection
        if drio_cv:
            # Run single-fold hyperparameter search
            best_alpha, best_gamma, cv_results, best_model_state = drio_hyperparameter_search(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                metadata=metadata,
                model_config=DRIO_MODEL_CONFIGS[model_name],
                alpha_grid=drio_alpha_grid,
                gamma_grid=drio_gamma_grid,
                cv_epochs=drio_cv_epochs,
            )

            # Save CV results
            cv_results_serializable = {str(k): v for k, v in cv_results.items()}
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_gamma': best_gamma,
                    'alpha_grid': drio_alpha_grid or DRIO_DEFAULT_ALPHA_GRID,
                    'gamma_grid': drio_gamma_grid or DRIO_DEFAULT_GAMMA_GRID,
                    'results': cv_results_serializable,
                }, f, indent=2)

            # Create model and load best state from CV (no retraining needed)
            model = create_imputer(
                model_type=model_type,
                d_features=d_features,
                d_time=d_time,
                **model_config
            ).to(device)
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

            # Save the model
            torch.save(model.state_dict(), model_save_dir / 'model.pth')
            print(f"\nLoaded best model from CV search (alpha={best_alpha}, gamma={best_gamma})")

        else:
            # No CV - use lookup table or defaults
            standalone_params = get_drio_standalone_params(
                metadata['dataset'], metadata['missing_type'], metadata['missing_ratio']
            )

            if standalone_params is not None:
                best_alpha, best_gamma, standalone_epochs = standalone_params
                print(f"\nUsing STANDALONE LOOKUP: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")
            else:
                # Default parameters when not in lookup table
                best_alpha, best_gamma, standalone_epochs = 0.99, 1, 30
                print(f"\nUsing DEFAULT standalone parameters: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")

            # Override epochs in metadata for train_drio_model
            metadata_with_epochs = metadata.copy()
            metadata_with_epochs['_standalone_epochs'] = standalone_epochs

            # Create model
            model = create_imputer(
                model_type=model_type,
                d_features=d_features,
                d_time=d_time,
                **model_config
            ).to(device)

            # Train with DRIO objective using lookup hyperparameters
            model = train_drio_model(
                model, train_loader, val_loader, model_save_dir, device,
                alpha=best_alpha, gamma=best_gamma, metadata=metadata_with_epochs
            )

            # Load training curves to get final validation losses
            training_curves = np.load(model_save_dir / 'training_curves.npz')
            final_val_loss = float(training_curves['val_gt_losses'][-1])
            final_val_eval = float(training_curves['val_eval_losses'][-1])

            # Save cv_results.json for compatibility with analysis scripts
            cv_results_standalone = {
                'best_alpha': best_alpha,
                'best_gamma': best_gamma,
                'alpha_grid': [best_alpha],
                'gamma_grid': [best_gamma],
                'results': {
                    f"({best_alpha}, {best_gamma})": {
                        'val_loss': final_val_loss,
                        'val_eval': final_val_eval,
                        'epoch': standalone_epochs,
                        'mse': final_val_eval,
                    }
                },
                'standalone_mode': True,
            }
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump(cv_results_standalone, f, indent=2)

        # Save config with selected hyperparameters
        config_to_save = {
            'train': DRIO_CONFIG['train'],
            'drio': DRIO_CONFIG['drio'].copy(),
            'model': DRIO_MODEL_CONFIGS[model_name],
        }
        config_to_save['drio']['alpha'] = best_alpha
        config_to_save['drio']['gamma'] = best_gamma
        config_to_save['cv_enabled'] = drio_cv
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Generate predictions
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_drio_predictions(model, train_loader, device)
        val_preds = generate_drio_predictions(model, val_loader, device)
        test_preds = generate_drio_predictions(model, test_loader, device)

    elif model_name == 'bsh_drio_brits':
        # BSH_BRITS: BRITS architecture with BSH_DRIO training objective
        # Uses DRIO lookup tables for alpha/gamma/epochs (same as drio_brits)
        # Create BRITSForDRIO model (wrapper that adapts BRITS to BSH_DRIO interface)
        model = BRITSForDRIO(
            d_features=d_features,
            d_time=d_time,
            rnn_hid_size=BSH_DRIO_BRITS_CONFIG['rnn_hid_size']
        ).to(device)

        # Hyperparameter selection using DRIO lookup tables (same as drio_brits)
        if bsh_drio_cv:
            # Run single-fold hyperparameter search using DRIO grids
            # Get dataset-specific grids from DRIO lookup
            drio_alpha_grid_lookup, drio_gamma_grid_lookup, drio_cv_epochs_lookup = get_drio_cv_config(
                metadata['dataset'], metadata['missing_type'], metadata['missing_ratio']
            )
            # Use provided grids if available, otherwise use DRIO lookup
            alpha_grid_to_use = bsh_drio_alpha_grid or drio_alpha_grid_lookup
            gamma_grid_to_use = bsh_drio_gamma_grid or drio_gamma_grid_lookup

            best_alpha, best_gamma, cv_results, best_model_state = bsh_drio_hyperparameter_search(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                metadata=metadata,
                model_config={'model_type': 'brits_for_drio', 'rnn_hid_size': BSH_DRIO_BRITS_CONFIG['rnn_hid_size']},
                alpha_grid=alpha_grid_to_use,
                gamma_grid=gamma_grid_to_use,
                cv_epochs=drio_cv_epochs_lookup,
                model_class=BRITSForDRIO,
                model_kwargs={'d_features': d_features, 'd_time': d_time, 'rnn_hid_size': BSH_DRIO_BRITS_CONFIG['rnn_hid_size']},
            )

            # Save CV results
            cv_results_serializable = {str(k): v for k, v in cv_results.items()}
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_gamma': best_gamma,
                    'alpha_grid': alpha_grid_to_use,
                    'gamma_grid': gamma_grid_to_use,
                    'results': cv_results_serializable,
                }, f, indent=2)

            # Load best state from CV
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            torch.save(model.state_dict(), model_save_dir / 'model.pth')
            print(f"\nLoaded best model from CV search (alpha={best_alpha}, gamma={best_gamma})")

        else:
            # No CV - use DRIO standalone lookup table (same as drio_brits)
            standalone_params = get_drio_standalone_params(
                metadata['dataset'], metadata['missing_type'], metadata['missing_ratio']
            )

            if standalone_params is not None:
                best_alpha, best_gamma, standalone_epochs = standalone_params
                print(f"\nUsing DRIO STANDALONE LOOKUP for BSH_BRITS: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")
            else:
                # Default parameters when not in lookup table
                best_alpha, best_gamma, standalone_epochs = 0.99, 1, 30
                print(f"\nUsing DEFAULT standalone parameters for BSH_BRITS: alpha={best_alpha}, gamma={best_gamma}, epochs={standalone_epochs}")

            # Override epochs in metadata for train_bsh_drio_model
            metadata_with_epochs = metadata.copy()
            metadata_with_epochs['_standalone_epochs'] = standalone_epochs

            # Train with BSH_DRIO objective
            model = train_bsh_drio_model(
                model, train_loader, val_loader, model_save_dir, device,
                alpha=best_alpha, gamma=best_gamma
            )

            # Load training curves to get final validation losses
            training_curves = np.load(model_save_dir / 'training_curves.npz')
            final_val_loss = float(training_curves['val_gt_losses'][-1])
            final_val_eval = float(training_curves['val_eval_losses'][-1])

            # Save cv_results.json for compatibility with analysis scripts
            cv_results_standalone = {
                'best_alpha': best_alpha,
                'best_gamma': best_gamma,
                'alpha_grid': [best_alpha],
                'gamma_grid': [best_gamma],
                'results': {
                    f"({best_alpha}, {best_gamma})": {
                        'val_loss': final_val_loss,
                        'val_eval': final_val_eval,
                        'epoch': standalone_epochs,
                        'mse': final_val_eval,
                    }
                },
                'standalone_mode': True,
            }
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump(cv_results_standalone, f, indent=2)

        # Save config
        config_to_save = {
            'train': BSH_DRIO_CONFIG['train'],
            'bsh_drio': BSH_DRIO_CONFIG['bsh_drio'].copy(),
            'model': {'type': 'bsh_drio_brits', **BSH_DRIO_BRITS_CONFIG},
        }
        config_to_save['bsh_drio']['alpha'] = best_alpha
        config_to_save['bsh_drio']['gamma'] = best_gamma
        config_to_save['cv_enabled'] = bsh_drio_cv
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Generate predictions
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_bsh_drio_predictions(model, train_loader, device)
        val_preds = generate_bsh_drio_predictions(model, val_loader, device)
        test_preds = generate_bsh_drio_predictions(model, test_loader, device)

    elif model_name.startswith('bsh_drio'):
        # BSH_DRIO variants (bsh_drio_transformer, bsh_drio_lstm, bsh_drio_gat, bsh_drio_mlp, bsh_drio_sttransformer)
        model_config = BSH_DRIO_MODEL_CONFIGS[model_name].copy()
        model_type = model_config.pop('model_type')

        # Hyperparameter selection
        if bsh_drio_cv:
            # Run single-fold hyperparameter search
            best_alpha, best_gamma, cv_results, best_model_state = bsh_drio_hyperparameter_search(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                metadata=metadata,
                model_config=BSH_DRIO_MODEL_CONFIGS[model_name],
                alpha_grid=bsh_drio_alpha_grid,
                gamma_grid=bsh_drio_gamma_grid,
            )

            # Save CV results
            cv_results_serializable = {str(k): v for k, v in cv_results.items()}
            with open(model_save_dir / 'cv_results.json', 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_gamma': best_gamma,
                    'alpha_grid': bsh_drio_alpha_grid or BSH_DRIO_DEFAULT_ALPHA_GRID,
                    'gamma_grid': bsh_drio_gamma_grid or BSH_DRIO_DEFAULT_GAMMA_GRID,
                    'results': cv_results_serializable,
                }, f, indent=2)

            # Create model and load best state from CV (no retraining needed)
            model = create_imputer(
                model_type=model_type,
                d_features=d_features,
                d_time=d_time,
                **model_config
            ).to(device)
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

            # Save the model
            torch.save(model.state_dict(), model_save_dir / 'model.pth')
            print(f"\nLoaded best model from CV search (alpha={best_alpha}, gamma={best_gamma})")

        else:
            # Use default hyperparameters - need to train
            best_alpha = BSH_DRIO_CONFIG['bsh_drio']['alpha']
            best_gamma = BSH_DRIO_CONFIG['bsh_drio']['gamma']
            print(f"\nUsing default BSH_DRIO hyperparameters: alpha={best_alpha}, gamma={best_gamma}")

            # Create model
            model = create_imputer(
                model_type=model_type,
                d_features=d_features,
                d_time=d_time,
                **model_config
            ).to(device)

            # Train with BSH_DRIO objective using default hyperparameters
            model = train_bsh_drio_model(
                model, train_loader, val_loader, model_save_dir, device,
                alpha=best_alpha, gamma=best_gamma
            )

        # Save config with selected hyperparameters
        config_to_save = {
            'train': BSH_DRIO_CONFIG['train'],
            'bsh_drio': BSH_DRIO_CONFIG['bsh_drio'].copy(),
            'model': BSH_DRIO_MODEL_CONFIGS[model_name],
        }
        config_to_save['bsh_drio']['alpha'] = best_alpha
        config_to_save['bsh_drio']['gamma'] = best_gamma
        config_to_save['cv_enabled'] = bsh_drio_cv
        with open(model_save_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Generate predictions
        print(f"\nGenerating predictions for {model_name}...")
        train_preds = generate_bsh_drio_predictions(model, train_loader, device)
        val_preds = generate_bsh_drio_predictions(model, val_loader, device)
        test_preds = generate_bsh_drio_predictions(model, test_loader, device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Evaluate and save predictions for each split
    shape_info = {'N': None, 'D': d_features, 'T': d_time}

    for split_name, preds, dataset in [
        ('train', train_preds, train_dataset),
        ('val', val_preds, val_dataset),
        ('test', test_preds, test_dataset)
    ]:
        # Skip if predictions are None (e.g., PSW only evaluates test)
        if preds is None:
            print(f"\n  Skipping {split_name} evaluation (no predictions generated)")
            continue

        targets, predictions, gt_masks, eval_masks = preds
        shape_info['N'] = targets.shape[0]

        # Compute observed_mask = gt_mask + eval_mask (all entries with ground truth)
        observed_mask = gt_masks + eval_masks

        # For training error, evaluate on gt_mask (what model was trained on)
        # For val/test, evaluate on eval_mask (held-out entries)
        if split_name == 'train':
            mask = gt_masks
            print(f"\n  Evaluating {model_name} on {split_name} (gt_mask - training reconstruction)...")
        else:
            mask = eval_masks
            print(f"\n  Evaluating {model_name} on {split_name} (eval_mask - held-out entries)...")

        # Evaluate
        split_results = evaluate_predictions(
            targets, predictions, mask, dataset, split_name,
            gt_mask=gt_masks, observed_mask=observed_mask
        )
        results[split_name] = split_results

        # Print metrics including global W2
        w2_global_str = f"{split_results['w2_global']:.6f}" if not np.isnan(split_results['w2_global']) else "N/A"
        print(f"    RMSE: {split_results['rmse']:.6f}, MAE: {split_results['mae']:.6f}, W2_global: {w2_global_str}")

        # Save predictions (denormalized)
        targets_denorm = dataset.denormalize(targets)
        predictions_denorm = dataset.denormalize(predictions)
        save_predictions(predictions_denorm, targets_denorm, mask,
                        model_save_dir, model_name, split_name, shape_info)

    # Save individual model results
    results_json = {}
    for split_name in ['train', 'val', 'test']:
        r = results.get(split_name)
        if r is None:
            continue  # Skip splits that weren't evaluated (e.g., PSW only evaluates test)
        results_json[split_name] = {
            'mse': float(r['mse']),
            'rmse': float(r['rmse']),
            'mae': float(r['mae']),
            'w2_global': float(r['w2_global']) if not np.isnan(r['w2_global']) else None,
            'ae_summary': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                          for k, v in r['ae_summary'].items()},
            'se_summary': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                          for k, v in r['se_summary'].items()},
        }

    with open(model_save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified training and evaluation for imputation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-prefix",
        type=str,
        required=False,
        help="Path prefix for data files (e.g., data/processed/physionet/physionet_mnar_90pct_split70-10-20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when creating the data"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help="Models to train and evaluate"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<dataset>/<data_file>/)"
    )

    # DRIO-specific arguments
    parser.add_argument(
        "--drio-cv",
        action="store_true",
        default=True,
        help="Enable single-fold cross-validation for DRIO hyperparameter selection (default: True)"
    )
    parser.add_argument(
        "--no-drio-cv",
        action="store_true",
        help="Disable DRIO cross-validation and use default hyperparameters"
    )
    parser.add_argument(
        "--drio-alpha-grid",
        type=float,
        nargs='+',
        default=None,
        help=f"Grid of alpha values for DRIO CV (default: {DRIO_DEFAULT_ALPHA_GRID})"
    )
    parser.add_argument(
        "--drio-gamma-grid",
        type=float,
        nargs='+',
        default=None,
        help=f"Grid of gamma values for DRIO CV (default: {DRIO_DEFAULT_GAMMA_GRID})"
    )
    parser.add_argument(
        "--drio-cv-epochs",
        type=int,
        default=None,
        help="Number of epochs for DRIO CV (default: use dataset-specific lookup table)"
    )

    # BSH_DRIO-specific arguments (Balanced Sinkhorn DRIO)
    parser.add_argument(
        "--bsh-drio-cv",
        action="store_true",
        default=True,
        help="Enable single-fold cross-validation for BSH_DRIO hyperparameter selection (default: True)"
    )
    parser.add_argument(
        "--no-bsh-drio-cv",
        action="store_true",
        help="Disable BSH_DRIO cross-validation and use default hyperparameters"
    )
    parser.add_argument(
        "--bsh-drio-alpha-grid",
        type=float,
        nargs='+',
        default=None,
        help=f"Grid of alpha values for BSH_DRIO CV (default: {BSH_DRIO_DEFAULT_ALPHA_GRID})"
    )
    parser.add_argument(
        "--bsh-drio-gamma-grid",
        type=float,
        nargs='+',
        default=None,
        help=f"Grid of gamma values for BSH_DRIO CV (default: {BSH_DRIO_DEFAULT_GAMMA_GRID})"
    )

    args = parser.parse_args()

    # Handle --no-drio-cv flag
    if args.no_drio_cv:
        args.drio_cv = False

    # Handle --no-bsh-drio-cv flag
    if args.no_bsh_drio_cv:
        args.bsh_drio_cv = False

    # List models and exit
    if args.list_models:
        print("\nAvailable models:")
        for model in AVAILABLE_MODELS:
            print(f"  - {model}")
        print("\nUsage example:")
        print("  python train_test_unified.py --data-prefix data/processed/physionet/physionet_mnar_90pct_split70-10-20 --seed 42 --models csdi mse_transformer")
        return

    if args.data_prefix is None:
        parser.error("--data-prefix is required unless --list-models is specified")

    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # For full reproducibility (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'=' * 70}")
    print("UNIFIED TRAINING AND EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Data prefix: {args.data_prefix}")
    print(f"Seed: {args.seed} (set for np, torch, random)")
    if any(m.startswith('drio') for m in args.models):
        print(f"DRIO CV: {args.drio_cv}")
        if args.drio_cv:
            alpha_grid = args.drio_alpha_grid or DRIO_DEFAULT_ALPHA_GRID
            gamma_grid = args.drio_gamma_grid or DRIO_DEFAULT_GAMMA_GRID
            print(f"  Alpha grid: {alpha_grid}")
            print(f"  Gamma grid: {gamma_grid}")
    if any(m.startswith('bsh_drio') for m in args.models):
        print(f"BSH_DRIO CV: {args.bsh_drio_cv}")
        if args.bsh_drio_cv:
            bsh_alpha_grid = args.bsh_drio_alpha_grid or BSH_DRIO_DEFAULT_ALPHA_GRID
            bsh_gamma_grid = args.bsh_drio_gamma_grid or BSH_DRIO_DEFAULT_GAMMA_GRID
            print(f"  Alpha grid: {bsh_alpha_grid}")
            print(f"  Gamma grid: {bsh_gamma_grid}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    train_data, val_data, test_data, metadata = load_data(args.data_prefix, args.seed)
    print_data_summary(train_data, val_data, test_data, metadata)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data, metadata
    )

    # Check which types of models we need loaders for
    need_mse_loaders = any(m.startswith('mse_') for m in args.models)
    need_csdi_loaders = 'csdi' in args.models
    need_drio_loaders = any(m.startswith('drio') for m in args.models)
    need_bsh_drio_loaders = any(m.startswith('bsh_drio') for m in args.models)

    # Create data loaders only if needed
    train_loader, val_loader, test_loader = None, None, None
    csdi_train_loader, csdi_val_loader, csdi_test_loader = None, None, None
    drio_train_loader, drio_val_loader, drio_test_loader = None, None, None
    bsh_drio_train_loader, bsh_drio_val_loader, bsh_drio_test_loader = None, None, None

    if need_mse_loaders:
        train_loader = DataLoader(train_dataset, batch_size=MSE_TRAIN_CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=MSE_TRAIN_CONFIG['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=MSE_TRAIN_CONFIG['batch_size'], shuffle=False)

    if need_csdi_loaders:
        csdi_train_loader = DataLoader(train_dataset, batch_size=CSDI_CONFIG['train']['batch_size'], shuffle=True)
        csdi_val_loader = DataLoader(val_dataset, batch_size=CSDI_CONFIG['train']['batch_size'], shuffle=False)
        csdi_test_loader = DataLoader(test_dataset, batch_size=CSDI_CONFIG['train']['batch_size'], shuffle=False)

    if need_drio_loaders:
        drio_train_loader = DataLoader(train_dataset, batch_size=DRIO_CONFIG['train']['batch_size'], shuffle=True)
        drio_val_loader = DataLoader(val_dataset, batch_size=DRIO_CONFIG['train']['batch_size'], shuffle=False)
        drio_test_loader = DataLoader(test_dataset, batch_size=DRIO_CONFIG['train']['batch_size'], shuffle=False)

    if need_bsh_drio_loaders:
        bsh_drio_train_loader = DataLoader(train_dataset, batch_size=BSH_DRIO_CONFIG['train']['batch_size'], shuffle=True)
        bsh_drio_val_loader = DataLoader(val_dataset, batch_size=BSH_DRIO_CONFIG['train']['batch_size'], shuffle=False)
        bsh_drio_test_loader = DataLoader(test_dataset, batch_size=BSH_DRIO_CONFIG['train']['batch_size'], shuffle=False)

    # Setup output directory
    if args.output_dir is None:
        data_name = Path(args.data_prefix).name + f"_seed{args.seed}"
        output_dir = PROJECT_ROOT / 'results' / metadata['dataset'] / data_name
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Train and evaluate each model
    all_train_results = {}
    all_val_results = {}
    all_test_results = {}

    for model_name in args.models:
        print(f"\n\n{'#' * 70}")
        print(f"# PROCESSING MODEL: {model_name}")
        print("#" * 70)

        # Use appropriate loaders for each model type
        if model_name == 'csdi':
            results = run_model(
                model_name, train_dataset, val_dataset, test_dataset,
                csdi_train_loader, csdi_val_loader, csdi_test_loader,
                output_dir, device, metadata
            )
        elif model_name.startswith('bsh_drio'):
            results = run_model(
                model_name, train_dataset, val_dataset, test_dataset,
                bsh_drio_train_loader, bsh_drio_val_loader, bsh_drio_test_loader,
                output_dir, device, metadata,
                bsh_drio_cv=args.bsh_drio_cv,
                bsh_drio_alpha_grid=args.bsh_drio_alpha_grid,
                bsh_drio_gamma_grid=args.bsh_drio_gamma_grid,
            )
        elif model_name.startswith('drio'):
            results = run_model(
                model_name, train_dataset, val_dataset, test_dataset,
                drio_train_loader, drio_val_loader, drio_test_loader,
                output_dir, device, metadata,
                drio_cv=args.drio_cv,
                drio_alpha_grid=args.drio_alpha_grid,
                drio_gamma_grid=args.drio_gamma_grid,
                drio_cv_epochs=args.drio_cv_epochs,
            )
        else:
            results = run_model(
                model_name, train_dataset, val_dataset, test_dataset,
                train_loader, val_loader, test_loader,
                output_dir, device, metadata
            )

        # Store results (may be None for models that don't evaluate all splits)
        all_train_results[model_name] = results.get('train')
        all_val_results[model_name] = results.get('val')
        all_test_results[model_name] = results.get('test')

        # Save summary for this model in its own folder
        model_save_dir = output_dir / model_name
        print(f"\n{'=' * 70}")
        print(f"GENERATING SUMMARY FOR {model_name}")
        print("=" * 70)

        summary_lines = []
        summary_lines.append(f"# Imputation Evaluation Results - {model_name}\n")
        summary_lines.append(f"**Model:** `{model_name}`\n")
        summary_lines.append(f"**Data prefix:** `{args.data_prefix}`\n")
        summary_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add detailed data information section
        summary_lines.append("## Data Information\n")
        summary_lines.append("| Property | Value |")
        summary_lines.append("|----------|-------|")
        summary_lines.append(f"| Dataset | {metadata['dataset']} |")
        summary_lines.append(f"| Missing type | {metadata['missing_type'].upper()} |")
        summary_lines.append(f"| Target missing ratio | {metadata['missing_ratio']*100:.0f}% |")
        summary_lines.append(f"| Actual missing ratio | {metadata['actual_missing_ratio']*100:.1f}% |")
        summary_lines.append(f"| Random seed | {args.seed} |")
        summary_lines.append(f"| Total samples | {metadata['total_samples']} |")
        summary_lines.append(f"| Train samples | {len(train_data['observed_values'])} ({metadata['train_ratio']*100:.0f}%) |")
        summary_lines.append(f"| Val samples | {len(val_data['observed_values'])} ({metadata['val_ratio']*100:.0f}%) |")
        summary_lines.append(f"| Test samples | {len(test_data['observed_values'])} ({metadata['test_ratio']*100:.0f}%) |")
        summary_lines.append(f"| Time steps (T) | {metadata['shape'][1]} |")
        summary_lines.append(f"| Features (D) | {metadata['shape'][2]} |")
        summary_lines.append(f"| Data shape | (N, T, D) = (N, {metadata['shape'][1]}, {metadata['shape'][2]}) |")
        if metadata.get('mnar_steepness') is not None:
            summary_lines.append(f"| MNAR steepness | {metadata['mnar_steepness']} |")
        summary_lines.append("\n")

        # Add note about evaluation
        summary_lines.append("## Evaluation Notes\n")
        summary_lines.append("- **Train metrics:** Computed on `gt_mask` (reconstruction error on observed entries used for training)\n")
        summary_lines.append("- **Val/Test metrics:** Computed on `eval_mask` (held-out entries not seen during training)\n")
        summary_lines.append("- **All metrics:** Computed on normalized (z-score) scale, consistent with CSDI benchmark\n\n")

        # Create tables for this model only (filter out None results)
        model_train_results = {model_name: results.get('train')} if results.get('train') is not None else {}
        model_val_results = {model_name: results.get('val')} if results.get('val') is not None else {}
        model_test_results = {model_name: results.get('test')} if results.get('test') is not None else {}

        train_table = create_results_table(model_train_results, 'train')
        val_table = create_results_table(model_val_results, 'val')
        test_table = create_results_table(model_test_results, 'test')

        summary_lines.append(train_table)
        summary_lines.append("\n\n")
        summary_lines.append(val_table)
        summary_lines.append("\n\n")
        summary_lines.append(test_table)

        # Save summary to model's folder
        summary_path = model_save_dir / 'results_summary.md'
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))

        print(f"Summary saved to: {summary_path}")

        # Print tables
        print("\n" + train_table)
        print("\n" + val_table)
        print("\n" + test_table)

    print(f"\n{'=' * 70}")
    print("COMPLETED")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
