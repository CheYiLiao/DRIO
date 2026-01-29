"""
Configuration file for unified training and evaluation of imputation models.

Contains all model configurations, training hyperparameters, and dataset-specific settings.
"""

import torch

# =============================================================================
# AVAILABLE MODELS
# =============================================================================

AVAILABLE_MODELS = [
    # Simple baselines (no learning / non-neural)
    'mean',        # Mean imputation (simple statistical baseline)
    'mf',          # Matrix Factorization (neural low-rank embedding)
    # Benchmark methods
    'csdi',
    'mdot',
    'brits',
    'psw',
    'notmiwae',  # not-MIWAE (MIWAE with missing process modeling)
    'drio_sttransformer',
    'drio_lstm',
    'drio_gat',
    'drio_mlp',
    'drio_brits',
    'drio_v2_brits',  # DRIO v2 with Internal masking
    'bsh_drio_brits',  # BSH_DRIO with BRITS backbone (uses DRIO lookup tables)
    'mse_brits',  # MSE with BRITS backbone
]



# MSE training hyperparameters
MSE_TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'val_interval': 10,
}

# =============================================================================
# SIMPLE BASELINE CONFIGURATIONS
# =============================================================================

# Mean imputation (no hyperparameters - just fill with mean)
MEAN_CONFIG = {}


# Matrix Factorization configuration
MF_CONFIG = {
    'embedding_size': 8,      # Dimension of embeddings
    'lr': 1e-2,               # Learninsg rate
    'n_epochs': 500,          # Number of training epochs
    'batch_size': 512,        # Batch size for training
    'reg': 1e-3,              # L2 regularization weight
}

# =============================================================================
# DRIO MODEL CONFIGURATIONS
# =============================================================================
DRIO_CV_LOOKUP = {}
DRIO_MODEL_CONFIGS = {
    'drio_sttransformer': {
        'model_type': 'spatiotemporal_transformer',
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'dropout': 0.1
    },
    'drio_lstm': {
        'model_type': 'lstm',
        'd_hidden': 128,
        'num_layers': 2,
        'dropout': 0.1
    },
    'drio_gat': {
        'model_type': 'gat',
        'd_hidden': 128,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1
    },
    'drio_mlp': {
        'model_type': 'mlp',
        'd_hidden': 128,
        'n_layers': 3,
        'dropout': 0.1
    }
}

# DRIO configuration (from train-eval-drio.py)
DRIO_CONFIG = {
    'train': {
        'epochs': 30,  # Epochs for CV and standalone training
        'batch_size': 32,
        'lr': 5e-4,
        'weight_decay': 1e-6,
        'val_interval': 10,  # Not used - DRIO validates every epoch
    },
    'drio': {
        'alpha': 0.5,       # Trade-off between reconstruction and robustness
        'gamma': 1.0,       # Robustness parameter (adversary transport budget)
        'epsilon': 0.1,     # Entropic regularization for Sinkhorn
        'tau': 10.0,        # Marginal relaxation for unbalanced OT
        'inner_steps': 5,   # Gradient ascent steps for adversary (K in Algorithm 1)
        'inner_lr': 0.01,   # Adversary learning rate
        'adaptive_epsilon': True,  # Compute epsilon adaptively per batch
        'epsilon_quant': 0.5,      # Quantile for adaptive epsilon
        'epsilon_mult': 0.05,      # Multiplier for adaptive epsilon
    },
    'model': {
        # Spatiotemporal Transformer architecture (same as mse_sttransformer)
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
    }
}

# Default hyperparameter grids for DRIO cross-validation
# Fallback grid when dataset-specific lookup is not available
DRIO_DEFAULT_ALPHA_GRID = [0.99]
DRIO_DEFAULT_GAMMA_GRID = [0.05, 1.0]


# These parameters were selected using val_loss (MSE on gt_mask) as the criterion,
# which is the deployable, non-cheating approach (doesn't require ground truth of missing values).
# Selected from full CV grid search: alpha in [0.01, 0.25, 0.5, 0.75, 0.99], gamma in [0.1, 1.0, 5.0, 10.0]

DRIO_STANDALONE_LOOKUP = {
    # airquality dataset
    ('airquality', 'mcar', 10): (0.99, 0.1, 30),
    ('airquality', 'mcar', 50): (0.99, 0.1, 30),
    ('airquality', 'mcar', 90): (0.99, 10.0, 30),
    ('airquality', 'mnar', 10): (0.99, 0.1, 30),
    ('airquality', 'mnar', 50): (0.99, 5.0, 30),
    ('airquality', 'mnar', 90): (0.99, 10.0, 30),
    # cmapss dataset
    ('cmapss', 'mcar', 10): (0.99, 0.1, 30),
    ('cmapss', 'mcar', 50): (0.99, 0.1, 30),
    ('cmapss', 'mcar', 90): (0.99, 0.1, 30),
    ('cmapss', 'mnar', 10): (0.99, 10.0, 30),
    ('cmapss', 'mnar', 50): (0.99, 10.0, 30),
    ('cmapss', 'mnar', 90): (0.99, 1.0, 30),
    # cnnpred dataset
    ('cnnpred', 'mcar', 10): (0.75, 5.0, 30),
    ('cnnpred', 'mcar', 50): (0.99, 10.0, 30),
    ('cnnpred', 'mcar', 90): (0.75, 10.0, 30),
    ('cnnpred', 'mnar', 10): (0.99, 10.0, 30),
    ('cnnpred', 'mnar', 50): (0.75, 5.0, 30),
    ('cnnpred', 'mnar', 90): (0.75, 5.0, 30),
    # gait dataset
    ('gait', 'mcar', 10): (0.99, 0.1, 30),
    ('gait', 'mcar', 50): (0.99, 5.0, 30),
    ('gait', 'mcar', 90): (0.99, 10.0, 30),
    ('gait', 'mnar', 10): (0.99, 0.1, 30),
    ('gait', 'mnar', 50): (0.99, 5.0, 30),
    ('gait', 'mnar', 90): (0.99, 5.0, 30),
    # gassensor dataset
    ('gassensor', 'mcar', 10): (0.99, 5.0, 30),
    ('gassensor', 'mcar', 50): (0.99, 0.1, 28),
    ('gassensor', 'mcar', 90): (0.99, 0.1, 28),
    ('gassensor', 'mnar', 10): (0.99, 5.0, 30),
    ('gassensor', 'mnar', 50): (0.99, 5.0, 30),
    ('gassensor', 'mnar', 90): (0.99, 1.0, 29),
    # har dataset
    ('har', 'mcar', 10): (0.99, 10.0, 28),
    ('har', 'mcar', 50): (0.99, 10.0, 24),
    ('har', 'mcar', 90): (0.99, 10.0, 30),  # No CV data (poor student no budget), use default
    ('har', 'mnar', 10): (0.99, 10.0, 29),
    ('har', 'mnar', 50): (0.99, 5.0, 22),
    ('har', 'mnar', 90): (0.99, 5.0, 30),  # No CV data (poor student no budget), use default
    # pems04 dataset
    ('pems04', 'mcar', 10): (0.99, 10.0, 30),
    ('pems04', 'mcar', 50): (0.99, 0.1, 30),
    ('pems04', 'mcar', 90): (0.99, 0.1, 30),
    ('pems04', 'mnar', 10): (0.99, 10.0, 30),
    ('pems04', 'mnar', 50): (0.99, 0.1, 30),
    ('pems04', 'mnar', 90): (0.99, 0.1, 30),
    # pems08 dataset
    ('pems08', 'mcar', 10): (0.99, 10.0, 30),
    ('pems08', 'mcar', 50): (0.99, 10.0, 30),
    ('pems08', 'mcar', 90): (0.99, 1.0, 30),
    ('pems08', 'mnar', 10): (0.99, 10.0, 30),
    ('pems08', 'mnar', 50): (0.99, 1.0, 30),
    ('pems08', 'mnar', 90): (0.99, 0.1, 30),
    # physionet dataset
    ('physionet', 'mcar', 10): (0.75, 10.0, 27),
    ('physionet', 'mcar', 50): (0.99, 10.0, 30),
    ('physionet', 'mcar', 90): (0.99, 5.0, 30),
    ('physionet', 'mnar', 10): (0.99, 10.0, 28),
    ('physionet', 'mnar', 50): (0.99, 5.0, 23),
    ('physionet', 'mnar', 90): (0.99, 0.1, 28),
    # pm25 dataset
    ('pm25', 'mcar', 10): (0.99, 10.0, 30),
    ('pm25', 'mcar', 50): (0.99, 10.0, 30),
    ('pm25', 'mcar', 90): (0.99, 10.0, 30),
    ('pm25', 'mnar', 10): (0.99, 0.1, 30),
    ('pm25', 'mnar', 50): (0.99, 10.0, 30),
    ('pm25', 'mnar', 90): (0.99, 10.0, 30),
}


def get_drio_standalone_params(dataset, missing_type, missing_ratio):
    """
    Get fixed (alpha, gamma, epochs) for standalone DRIO training (no CV).

    Args:
        dataset: Dataset name (e.g., 'har', 'cmapss')
        missing_type: 'mcar' or 'mnar'
        missing_ratio: Missing ratio as decimal (e.g., 0.1, 0.5, 0.9)

    Returns:
        Tuple of (alpha, gamma, epochs) or None if not in lookup
    """
    missing_pct = int(round(missing_ratio * 100))
    key = (dataset.lower(), missing_type.lower(), missing_pct)

    if key in DRIO_STANDALONE_LOOKUP:
        return DRIO_STANDALONE_LOOKUP[key]

    # Not in lookup - return None to signal use of defaults
    return None




# BSH_DRIO configuration (same as DRIO but no tau - balanced Sinkhorn)
BSH_DRIO_CONFIG = {
    'train': {
        'epochs': 30,  # Epochs for CV and standalone training
        'batch_size': 32,
        'lr': 5e-4,
        'weight_decay': 1e-6,
        'val_interval': 10,  # Not used - BSH_DRIO validates every epoch
    },
    'bsh_drio': {
        'alpha': 0.5,       # Trade-off between reconstruction and robustness
        'gamma': 1.0,       # Robustness parameter (adversary transport budget)
        'epsilon': 0.1,     # Entropic regularization for Sinkhorn
        # No tau - balanced Sinkhorn uses reach=None
        'inner_steps': 5,   # Gradient ascent steps for adversary (K in Algorithm 1)
        'inner_lr': 0.01,   # Adversary learning rate
        'adaptive_epsilon': True,  # Compute epsilon adaptively per batch
        'epsilon_quant': 0.5,      # Quantile for adaptive epsilon
        'epsilon_mult': 0.05,      # Multiplier for adaptive epsilon
    },
}

# Default hyperparameter grids for BSH_DRIO cross-validation
BSH_DRIO_DEFAULT_ALPHA_GRID = [ 0.99]
BSH_DRIO_DEFAULT_GAMMA_GRID = [0.1, 1.0]

# =============================================================================
# BASELINE MODEL CONFIGURATIONS
# =============================================================================

# CSDI configuration
# ORIGINAL SETTINGS: Matches benchmark_CSDI defaults exactly
CSDI_CONFIG = {
    "train": {
        "epochs": 30,           
        "batch_size": 16,       # Original CSDI default
        "lr": 1.0e-3,          # Original CSDI default
        "itr_per_epoch": 1.0e+8 # Original CSDI default
    },
    "diffusion": {
        "layers": 4,                      # Original CSDI default
        "channels": 64,                   # Original CSDI default
        "nheads": 8,                      # Original CSDI default
        "diffusion_embedding_dim": 128,   # Original CSDI default
        "beta_start": 0.0001,             # Original CSDI default
        "beta_end": 0.5,                  # Original CSDI default
        "num_steps": 50,                  # Original CSDI default
        "schedule": "quad",               # Original CSDI default
        "is_linear": False                # Original CSDI default
    },
    "model": {
        "is_unconditional": False,    # Original CSDI default
        "timeemb": 128,               # Original CSDI default
        "featureemb": 16,             # Original CSDI default
        "target_strategy": "random"   # Original CSDI default
    }
}

# MissingDataOT configuration (Algorithm 1 - OTimputer)
# ORIGINAL SETTINGS: Matches benchmark_MissingDataOT/imputers.py defaults exactly
MDOT_CONFIG = {
    'eps': 0.01,                   # Original MDOT default
    'lr': 1e-2,                    # Original MDOT default
    'opt': torch.optim.RMSprop,    # Original MDOT default
    'niter': 1000,                 # Reduce to 1000 for efficiency
    'batchsize': 128,              # Original MDOT default
    'n_pairs': 1,                  # Original MDOT default
    'noise': 0.1,                  # Original MDOT default
    'scaling': 0.9                 # Original MDOT default
}

# BRITS configuration (Bidirectional Recurrent Imputation for Time Series)
# ORIGINAL SETTINGS: Matches benchmark_brits/main.py defaults (except label_weight for pure imputation)
BRITS_CONFIG = {
    'epochs': 30,          # set to 30
    'batch_size': 32,      # Original BRITS default
    'lr': 1e-3,            # Original BRITS default (Adam optimizer)
    'rnn_hid_size': 64,    # Original BRITS default
    'impute_weight': 1.0,  # Weight for imputation loss (x_loss / SEQ_LEN in original)
    'label_weight': 0.0,   # Classification weight (0.3 in original, but we set to 0 for pure imputation)
}

# DRIO_BRITS configuration (BRITS architecture trained with DRIO loss)
# Uses BRITS architecture (rnn_hid_size) but DRIO training params (alpha, gamma, etc.)
DRIO_BRITS_CONFIG = {
    'rnn_hid_size': 64,    # BRITS architecture parameter
    # Training params are inherited from DRIO_CONFIG
}

# MSE_BRITS configuration (BRITS architecture trained with MSE loss)
# Uses BRITS architecture but standard MSE training
MSE_BRITS_CONFIG = {
    'rnn_hid_size': 64,    # BRITS architecture parameter
    # Training params are inherited from MSE_TRAIN_CONFIG
}

# BSH_DRIO_BRITS configuration (BRITS architecture trained with BSH_DRIO loss)
# Uses BRITS architecture but BSH_DRIO training with DRIO lookup tables for alpha/gamma/epochs
BSH_DRIO_BRITS_CONFIG = {
    'rnn_hid_size': 64,    # BRITS architecture parameter
    # Training params are inherited from BSH_DRIO_CONFIG
    # Hyperparameters (alpha, gamma, epochs) use DRIO lookup tables (DRIO_STANDALONE_LOOKUP, DRIO_CV_LOOKUP)
}

# =============================================================================
# DRIO V2 CONFIGURATIONS (with CSDI-style internal masking)
# =============================================================================

# DRIO V2 configuration
# Key difference: Uses internal masking during training
# - Randomly masks ~50% of observed entries (like CSDI's get_randmask)
# - MSE computed on artificially masked entries (self-supervised)
# - Sinkhorn still on complete trajectories
DRIO_V2_CONFIG = {
    'train': {
        'epochs': 30,  # Epochs for CV and standalone training
        'batch_size': 128,  # Increased from 32 for faster training
        'lr': 5e-4,
        'weight_decay': 1e-6,
        'val_interval': 10,
    },
    'drio_v2': {
        'alpha': 0.5,       # Trade-off between reconstruction and robustness
        'gamma': 1.0,       # Robustness parameter (adversary transport budget)
        'epsilon': 0.1,     # Entropic regularization for Sinkhorn
        'tau': 10.0,        # Marginal relaxation for unbalanced OT
        'inner_steps': 5,   # Gradient ascent steps for adversary
        'inner_lr': 0.01,   # Adversary learning rate
        'mask_ratio_range': (0.0, 1.0),  # CSDI-style: uniform [0,1] mask ratio
        'adaptive_epsilon': True,
        'epsilon_quant': 0.5,
        'epsilon_mult': 0.05,
    },
}

# DRIO_V2_BRITS configuration (BRITS architecture trained with DRIO v2 loss)
DRIO_V2_BRITS_CONFIG = {
    'rnn_hid_size': 64,    # BRITS architecture parameter
    # Training params are inherited from DRIO_V2_CONFIG
}

# Default hyperparameter grids for DRIO V2 cross-validation
# Same as DRIO defaults - can be tuned later based on experiments
DRIO_V2_DEFAULT_ALPHA_GRID = [0.99]
DRIO_V2_DEFAULT_GAMMA_GRID = [0.1, 1.0]

# PSW-I configuration (Proximal Spectrum Wasserstein Imputation)
# ORIGINAL SETTINGS: Based on benchmark_psw/benchmark.py OTImputationIni
# Key innovation: Uses FFT-based spectral distance for OT cost matrix
PSW_CONFIG = {
    'lr': 0.002,               # Learning rate for imputed values optimization (argparser default line 163)
    'n_epochs': 30,            # Maximum number of epochs (set to 30 for consistency with other models)
    'batch_size': 200,         # Batch size for OT computation (argparser default line 182)
    'n_pairs': 2,              # Number of batch pairs per gradient update (argparser default line 172)
    'noise': 1e-4,             # Noise scale for initialization (model instantiation line 427)
    'reg_sk': 0.005,           # Sinkhorn regularization (argparser default line 177)
    'numItermax': 1000,        # Max iterations for OT solver (model instantiation line 427)
    'stopThr': 1e-6,           # Convergence threshold for OT solver (model instantiation line 427)
    'normalize': 1,            # Whether to normalize cost matrix (argparser default line 187)
    'seq_length': 24,          # Window size for time-series subsequences (argparser default line 192)
    'distance': 'fft',         # Distance type: 'fft' (spectral), 'time', 'fft_mag', 'fft_mag_abs' (argparser default line 202)
    'ot_type': 'uot_mm',       # OT solver: 'sinkhorn', 'emd', 'uot', 'uot_mm' (argparser default line 207)
    'reg_m': 1.0,              # KL divergence strength for unbalanced OT (argparser default line 212)
    'dropout': 0.0,            # Dropout probability for spectral features (argparser default line 218)
    'mva_kernel': 7,           # Kernel size for MVA initialization (argparser default line 197)
    'early_stop_patience': 10, # Patience for early stopping (line 412: if tick>10: break)
}

# not-MIWAE configuration (MIWAE with missing process modeling)
# ORIGINAL SETTINGS: Based on benchmark_notmiwae_torch/task01.py and notMIWAE.py
# Paper: "not-MIWAE: Deep Generative Modelling with Missing not at Random Data"
# Key innovation: Models the missing process p(s|x) jointly with data distribution
NOTMIWAE_CONFIG = {
    'n_latent': 50,            # Latent dimension (task01.py line 78: dl = D - 1, capped at D-1 in code)
    'n_hidden': 128,           # Hidden layer size (task01.py line 51)
    'n_samples': 20,           # Number of importance samples (task01.py line 52)
    'batch_size': 16,          # Batch size for training (task01.py line 54)
    'max_iter': 1000,          # Maximum training iterations (reduced to match MDOT niter)
    'out_dist': 'gauss',       # Output distribution: 'gauss', 'bern', 't'
    'missing_process': 'selfmasking_known',  # Missing mechanism (task01.py line 58)
    'L': 10000,                # Number of samples for imputation (task01.py line 55)
}

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Number of imputation samples for CSDI evaluation
N_CSDI_SAMPLES = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_drio_cv_config(dataset, missing_type, missing_ratio):
    """
    Get the optimized CV grid and epochs for DRIO based on dataset.

    Args:
        dataset: Dataset name (e.g., 'airquality', 'cmapss')
        missing_type: 'mcar' or 'mnar'
        missing_ratio: Missing ratio as decimal (e.g., 0.1, 0.5, 0.9)

    Returns:
        Tuple of (alpha_grid, gamma_grid, epochs)
    """
    missing_pct = int(round(missing_ratio * 100))
    key = (dataset.lower(), missing_type.lower(), missing_pct)

    if key in DRIO_CV_LOOKUP:
        return DRIO_CV_LOOKUP[key]

    # Fallback: use default grids and epochs from DRIO_CONFIG
    return (
        DRIO_DEFAULT_ALPHA_GRID,
        DRIO_DEFAULT_GAMMA_GRID,
        DRIO_CONFIG['train']['epochs']
    )


def get_drio_epochs(dataset, missing_type, missing_ratio):
    """
    Get the epochs for DRIO based on dataset (backward compatibility).

    Args:
        dataset: Dataset name (e.g., 'airquality', 'cmapss')
        missing_type: 'mcar' or 'mnar'
        missing_ratio: Missing ratio as decimal (e.g., 0.1, 0.5, 0.9)

    Returns:
        Number of epochs to use
    """
    _, _, epochs = get_drio_cv_config(dataset, missing_type, missing_ratio)
    return epochs
