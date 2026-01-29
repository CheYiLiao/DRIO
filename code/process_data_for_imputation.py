#!/usr/bin/env python3
"""
Data Processing for Imputation Tasks

Downloads data, processes it, generates missing masks, splits into train/val/test,
and saves the processed datasets.

KEY DESIGN: The test set is deterministic given (dataset, missing_type, missing_ratio,
test_ratio, seed). This ensures fair comparison across different train/val splits.

The splitting strategy:
1. First, use a dedicated RNG (seeded by seed + test_ratio) to select test indices
2. Then, use a separate RNG (seeded by seed + train_ratio + val_ratio) to split remaining data
3. Missing masks are generated using yet another RNG to avoid coupling

This guarantees: for the same dataset, missing type, missing ratio, and test proportion,
the test set will be identical regardless of how train/val are split.

Supported datasets:
- pems04: California District 4 traffic data (307 sensors x 59 days x 3 features)
- pems08: California District 8 traffic data (170 sensors x 62 days x 3 features)
- cmapss: NASA turbofan engine degradation (mixed FD001-FD004, engines with >=207 cycles)
- gassensor: UCI Gas Sensor Array (290 chunks x 150 time steps x 16 sensors)
- airquality: UCI Air Quality (389 days x 24 hours x 13 features)
- cnnpred: UCI CNNpred Stock Market (165 quarters x 60 trading days x 79 features)
- gait: UCI Multivariate Gait Data (300 cycles x 101 time points x 6 joint angles)
- har: UCI Human Activity Recognition (2947 samples x 128 time steps x 9 sensor channels)
- pm25: Beijing PM2.5 air quality data (260 weeks x 168 hours x 7 features)
- physionet: PhysioNet Challenge 2012 ICU data (48 hours x 35 attributes)

Note: PEMS data is aggregated to daily resolution (Flow=sum, Occupy/Speed=mean).
      Gas Sensor data is downsampled 10x then chunked (58 exp × 5 chunks = 290 samples).
      HAR features: body_acc (xyz), body_gyro (xyz), total_acc (xyz).
      PM2.5 features: pm2.5, DEWP, TEMP, PRES, Iws, Is, Ir.
      CMAPSS combines FD001-FD004, filtered to engines with >=207 cycles (median).
      Air Quality has natural missing values (marked as -200 in raw data, converted to NaN).
      CNNpred combines 5 US indices into quarterly chunks; features NOT normalized.
      Gait: 10 subjects x 3 conditions x 10 cycles; 6 bilateral joint angles (ankle/hip/knee).

Usage:
    # Default settings (physionet, MNAR, 90% missing, 70/10/20 split)
    python process_data_for_imputation.py

    # Custom settings
    python process_data_for_imputation.py --dataset physionet --missing-type mcar --missing-ratio 0.5

    # Different train/val splits with SAME test set (20% test)
    python process_data_for_imputation.py --train-ratio 0.7 --val-ratio 0.1  # 70/10/20
    python process_data_for_imputation.py --train-ratio 0.6 --val-ratio 0.2  # 60/20/20
    # Both will have the SAME test set!
"""

import os
import sys
import argparse
import pickle
import hashlib
import numpy as np
from pathlib import Path
from scipy.stats import norm
from datetime import datetime

# Setup paths
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# Default paths
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Supported datasets
SUPPORTED_DATASETS = ["physionet", "pems04", "pems08", "har", "pm25", "cmapss", "gassensor", "airquality", "cnnpred", "gait"]


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process data for imputation tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="physionet",
        choices=SUPPORTED_DATASETS,
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw data directory. If None, uses default path based on dataset."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save processed data. If None, uses default path."
    )
    parser.add_argument(
        "--missing-type",
        type=str,
        default="mnar",
        choices=["mcar", "mnar"],
        help="Missing pattern type: MCAR (Missing Completely At Random) or MNAR (Missing Not At Random)"
    )
    parser.add_argument(
        "--missing-ratio",
        type=float,
        default=0.9,
        help="Fraction of observed values to mask (must be in (0, 1))"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (must be in (0, 1))"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (must be in (0, 1))"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--mnar-steepness",
        type=float,
        default=3.0,
        help="Steepness parameter for MNAR missing pattern"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading data (assumes data already exists)"
    )

    args = parser.parse_args()

    # Validate ratios
    if not 0 < args.missing_ratio < 1:
        parser.error(f"missing-ratio must be in (0, 1), got {args.missing_ratio}")
    if not 0 < args.train_ratio < 1:
        parser.error(f"train-ratio must be in (0, 1), got {args.train_ratio}")
    if not 0 < args.val_ratio < 1:
        parser.error(f"val-ratio must be in (0, 1), got {args.val_ratio}")

    # Check test ratio is valid
    test_ratio = 1 - args.train_ratio - args.val_ratio
    if not 0 < test_ratio < 1:
        parser.error(
            f"test-ratio (1 - train_ratio - val_ratio) must be in (0, 1), "
            f"got {test_ratio} (train={args.train_ratio}, val={args.val_ratio})"
        )

    return args


# ============================================================================
# Seed Generation for Reproducible Splits
# ============================================================================

def generate_deterministic_seed(base_seed, *args):
    """
    Generate a deterministic seed from base_seed and additional arguments.

    This uses hashing to combine multiple values into a single seed,
    ensuring the same inputs always produce the same output seed.

    Args:
        base_seed: The base random seed
        *args: Additional values to incorporate (strings, floats, ints)

    Returns:
        int: A deterministic seed value
    """
    # Create a string representation of all arguments
    seed_str = f"{base_seed}"
    for arg in args:
        if isinstance(arg, float):
            # Use fixed precision for floats to avoid floating point issues
            seed_str += f"_{arg:.10f}"
        else:
            seed_str += f"_{arg}"

    # Hash the string and convert to an integer seed
    hash_bytes = hashlib.sha256(seed_str.encode()).digest()
    # Use first 4 bytes to get a 32-bit integer
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed


# ============================================================================
# Missing Pattern Functions
# ============================================================================

def apply_mcar_missing(observed_mask, missing_ratio, rng):
    """
    Apply MCAR (Missing Completely At Random) missing pattern.

    Randomly selects a fraction of observed values to be masked as missing.

    Args:
        observed_mask: Boolean/float array indicating observed values (1=observed, 0=missing)
        missing_ratio: Fraction of observed values to mask
        rng: NumPy random generator

    Returns:
        gt_mask: Ground-truth mask after applying MCAR pattern
    """
    # Handle both 2D (T, D) and 3D (N, T, D) arrays
    original_shape = observed_mask.shape
    masks = observed_mask.reshape(-1).copy().astype(bool)

    obs_indices = np.where(masks)[0]
    n_to_mask = int(len(obs_indices) * missing_ratio)
    miss_indices = rng.choice(obs_indices, n_to_mask, replace=False)
    masks[miss_indices] = False

    gt_mask = masks.reshape(original_shape).astype("float32")
    return gt_mask


def apply_mnar_missing(observed_values, observed_mask, missing_ratio,
                       feature_means, feature_stds, steepness=3.0, rng=None):
    """
    Apply MNAR (Missing Not At Random) missing pattern.

    Missing probability depends on the value itself

    where z_d,t = (X_d,t - mean_d) / std_d is the z-score.
    Higher absolute values have higher probability of being missing.

    Args:
        observed_values: Array with observed values (N, T, D) or (T, D)
        observed_mask: Boolean/float array indicating observed values
        missing_ratio: Target fraction of observed values to mask
        feature_means: Array of mean value per feature
        feature_stds: Array of std value per feature
        steepness: Parameter w controlling steepness of sigmoid
        rng: NumPy random generator

    Returns:
        gt_mask: Ground-truth mask after applying MNAR pattern
    """
    if rng is None:
        rng = np.random.default_rng()

    original_shape = observed_values.shape

    # Handle both 2D and 3D arrays
    if observed_values.ndim == 2:
        # (T, D) -> add batch dimension
        observed_values = observed_values[np.newaxis, ...]
        observed_mask = observed_mask[np.newaxis, ...]

    N, T, D = observed_values.shape
    gt_mask = observed_mask.copy().astype(bool)

    # Compute missing probability for each observed value
    missing_probs = np.zeros_like(observed_values)
    for d in range(D):
        if feature_stds[d] > 0:
            z_score = (observed_values[:, :, d] - feature_means[d]) / feature_stds[d]
            # Use CDF of normal at |z|: extreme values in either direction have higher missing prob
            missing_probs[:, :, d] = norm.cdf(np.abs(z_score))

    # Only consider observed values
    missing_probs = missing_probs * observed_mask

    # Flatten and sample
    obs_indices = np.where(observed_mask.reshape(-1))[0]
    probs_flat = missing_probs.reshape(-1)[obs_indices]

    # Normalize probabilities
    if probs_flat.sum() > 0:
        probs_flat = probs_flat / probs_flat.sum()
    else:
        probs_flat = np.ones_like(probs_flat) / len(probs_flat)

    # Sample indices to mask
    total_observed = len(obs_indices)
    n_to_mask = min(int(total_observed * missing_ratio), len(obs_indices))
    miss_indices = rng.choice(obs_indices, size=n_to_mask, replace=False, p=probs_flat)

    gt_mask_flat = gt_mask.reshape(-1)
    gt_mask_flat[miss_indices] = False
    gt_mask = gt_mask_flat.reshape(N, T, D)

    # Remove batch dimension if input was 2D
    if len(original_shape) == 2:
        gt_mask = gt_mask[0]

    return gt_mask.astype("float32")


def generate_masks(observed_values, observed_mask, missing_type, missing_ratio,
                   feature_means=None, feature_stds=None, mnar_steepness=3.0, rng=None):
    """
    Generate ground-truth masks based on missing type.

    Args:
        observed_values: Array with observed values
        observed_mask: Original observed mask from data
        missing_type: "mcar" or "mnar"
        missing_ratio: Fraction of observed values to mask
        feature_means: Required for MNAR
        feature_stds: Required for MNAR
        mnar_steepness: Steepness for MNAR sigmoid
        rng: NumPy random generator

    Returns:
        gt_mask: Ground-truth mask
    """
    if missing_type == "mcar":
        return apply_mcar_missing(observed_mask, missing_ratio, rng)
    elif missing_type == "mnar":
        if feature_means is None or feature_stds is None:
            raise ValueError("feature_means and feature_stds required for MNAR")
        return apply_mnar_missing(
            observed_values, observed_mask, missing_ratio,
            feature_means, feature_stds, mnar_steepness, rng
        )
    else:
        raise ValueError(f"Unknown missing_type: {missing_type}")


# ============================================================================
# PhysioNet Dataset Functions
# ============================================================================

def download_physionet(data_dir):
    """Download PhysioNet Challenge 2012 dataset."""
    import tarfile
    import wget

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if (data_dir / "set-a").exists():
        print(f"PhysioNet data already exists at {data_dir}")
        return data_dir

    print("Downloading PhysioNet Challenge 2012 dataset...")
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"

    output_file = data_dir / "set-a.tar.gz"
    wget.download(url, out=str(output_file))

    print("\nExtracting...")
    with tarfile.open(output_file, "r:gz") as t:
        t.extractall(path=data_dir)

    print(f"PhysioNet data saved to {data_dir}")
    return data_dir


def get_physionet_attributes():
    """Return list of PhysioNet attribute names."""
    return [
        'DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH',
        'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3',
        'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS', 'Cholesterol',
        'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine',
        'NIMAP', 'Creatinine', 'ALP'
    ]


def load_physionet_raw(data_dir):
    """
    Load raw PhysioNet data.

    Args:
        data_dir: Path to physionet data directory (containing set-a/)

    Returns:
        observed_values: Array of shape (N, T, D) with values
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import re
    import pandas as pd

    data_dir = Path(data_dir)
    attributes = get_physionet_attributes()
    num_attributes = len(attributes)

    def extract_hour(x):
        h, _ = map(int, x.split(":"))
        return h

    def parse_data(x):
        x = x.set_index("Parameter").to_dict()["Value"]
        values = []
        for attr in attributes:
            if attr in x:
                values.append(x[attr])
            else:
                values.append(np.nan)
        return values

    # Get patient IDs
    patient_ids = []
    for filename in os.listdir(data_dir / "set-a"):
        match = re.search(r"\d{6}", filename)
        if match:
            patient_ids.append(match.group())
    patient_ids = sorted(patient_ids)

    print(f"Found {len(patient_ids)} patients")

    all_values = []
    all_masks = []

    for patient_id in patient_ids:
        try:
            data = pd.read_csv(data_dir / "set-a" / f"{patient_id}.txt")
            data["Time"] = data["Time"].apply(extract_hour)

            # Create data for 48 hours x 35 attributes
            patient_values = []
            for h in range(48):
                patient_values.append(parse_data(data[data["Time"] == h]))
            patient_values = np.array(patient_values)  # (48, 35)
            patient_mask = ~np.isnan(patient_values)

            # Replace NaN with 0 for storage
            patient_values = np.nan_to_num(patient_values)

            all_values.append(patient_values)
            all_masks.append(patient_mask)
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue

    observed_values = np.array(all_values).astype("float32")  # (N, T, D)
    observed_mask = np.array(all_masks).astype("float32")      # (N, T, D)

    return observed_values, observed_mask, attributes


# ============================================================================
# PEMS Dataset Functions
# ============================================================================

def download_pems(data_dir, dataset):
    """
    Download PEMS04 or PEMS08 from ASTGCN GitHub repository.

    Args:
        data_dir: Base directory for PEMS data
        dataset: "pems04" or "pems08"

    Returns:
        Path to data directory
    """
    import wget

    data_dir = Path(data_dir)
    dataset_upper = dataset.upper()  # PEMS04 or PEMS08
    dataset_lower = dataset.lower()  # pems04 or pems08

    dataset_dir = data_dir / dataset_upper
    os.makedirs(dataset_dir, exist_ok=True)

    npz_file = dataset_dir / f"{dataset_lower}.npz"
    if npz_file.exists():
        print(f"{dataset_upper} data already exists at {dataset_dir}")
        return data_dir

    BASE_URL = "https://raw.githubusercontent.com/wanhuaiyu/ASTGCN/master/data"

    # Download .npz file
    url = f"{BASE_URL}/{dataset_upper}/{dataset_lower}.npz"
    print(f"Downloading {dataset_upper} traffic data...")
    wget.download(url, out=str(npz_file))

    # Download distance.csv (adjacency info)
    dist_file = dataset_dir / "distance.csv"
    if not dist_file.exists():
        url = f"{BASE_URL}/{dataset_upper}/distance.csv"
        print(f"\nDownloading {dataset_upper} distance matrix...")
        wget.download(url, out=str(dist_file))

    print(f"\n{dataset_upper} data saved to {dataset_dir}")
    return data_dir


def get_pems_attributes():
    """Return list of PEMS feature names."""
    return ['Flow', 'Occupy', 'Speed']


def load_pems_raw(data_dir, dataset):
    """
    Load raw PEMS data, aggregated to daily resolution.

    PEMS data is collected at 5-minute intervals (288 per day).
    We aggregate to daily resolution using:
    - Flow: sum (total daily vehicle count)
    - Occupy: mean (average daily occupancy %)
    - Speed: mean (average daily speed)

    Args:
        data_dir: Path to pems data directory
        dataset: "pems04" or "pems08"

    Returns:
        observed_values: Array of shape (N, T, D) - N=sensors, T=days, D=3 features
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    data_dir = Path(data_dir)
    dataset_upper = dataset.upper()
    dataset_lower = dataset.lower()

    npz_path = data_dir / dataset_upper / f"{dataset_lower}.npz"
    data = np.load(npz_path)['data']  # Shape: (T, N, F) - 5-min intervals

    # Aggregate to daily resolution
    # 1 day = 24 hours * 12 intervals/hour = 288 intervals
    intervals_per_day = 288
    n_days = data.shape[0] // intervals_per_day

    # Truncate to complete days
    data_truncated = data[:n_days * intervals_per_day, :, :]

    # Reshape to (n_days, intervals_per_day, N, F)
    data_reshaped = data_truncated.reshape(n_days, intervals_per_day, data.shape[1], data.shape[2])

    # Aggregate: Flow (sum), Occupy (mean), Speed (mean)
    # Feature order: [Flow, Occupy, Speed] = indices [0, 1, 2]
    flow_daily = data_reshaped[:, :, :, 0].sum(axis=1)      # Sum over intervals
    occupy_daily = data_reshaped[:, :, :, 1].mean(axis=1)   # Mean over intervals
    speed_daily = data_reshaped[:, :, :, 2].mean(axis=1)    # Mean over intervals

    # Stack features: (n_days, N, 3)
    data_daily = np.stack([flow_daily, occupy_daily, speed_daily], axis=-1)

    # Transpose to (N, T, F) to match other datasets (samples, time, features)
    # Here each sensor becomes a "sample", days become "time"
    data_daily = data_daily.transpose(1, 0, 2)  # (N, n_days, 3)

    attributes = get_pems_attributes()

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(data_daily)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(data_daily).astype("float32")

    print(f"  Loaded {dataset_upper}: {observed_values.shape[0]} sensors, "
          f"{observed_values.shape[1]} days, {observed_values.shape[2]} features")
    print(f"  (Aggregated from {data.shape[0]} 5-min intervals to daily resolution)")

    return observed_values, observed_mask, attributes


# ============================================================================
# HAR Dataset Functions
# ============================================================================

def download_har(data_dir):
    """
    Download UCI HAR dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    extract_dir = data_dir / "UCI HAR Dataset"
    if extract_dir.exists():
        print(f"HAR data already exists at {extract_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    zip_path = data_dir / "har.zip"

    if not zip_path.exists():
        print("Downloading HAR dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract outer zip (contains UCI HAR Dataset.zip inside)
    inner_zip = data_dir / "UCI HAR Dataset.zip"
    if not inner_zip.exists():
        print("Extracting outer zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

    # Extract inner zip (contains the actual data)
    if not extract_dir.exists():
        print("Extracting inner zip...")
        with zipfile.ZipFile(inner_zip, 'r') as z:
            z.extractall(data_dir)

    print(f"HAR data saved to {extract_dir}")
    return data_dir


def get_har_attributes():
    """Return list of HAR signal names."""
    return [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]


def load_har_raw(data_dir):
    """
    Load raw HAR inertial signals.

    Args:
        data_dir: Path to HAR data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=2947 samples, T=128, D=9
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)
    har_dir = data_dir / "UCI HAR Dataset"
    test_dir = har_dir / "test" / "Inertial Signals"

    attributes = get_har_attributes()

    # Load all 9 signals
    signal_files = [
        'body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
        'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
        'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt'
    ]

    arrays = []
    for f in signal_files:
        data = pd.read_csv(test_dir / f, sep=r'\s+', header=None).values
        arrays.append(data)

    # Stack: each array is (N, T=128), stack to (N, T, D=9)
    observed_values = np.stack(arrays, axis=-1).astype("float32")

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded HAR: {observed_values.shape[0]} samples, "
          f"{observed_values.shape[1]} time steps, {observed_values.shape[2]} features")

    return observed_values, observed_mask, attributes


# ============================================================================
# PM2.5 Dataset Functions
# ============================================================================

def download_pm25(data_dir):
    """
    Download Beijing PM2.5 dataset from UCI.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if CSV already exists
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"PM2.5 data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
    zip_path = data_dir / "pm25.zip"

    if not zip_path.exists():
        print("Downloading PM2.5 dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    print(f"PM2.5 data saved to {data_dir}")
    return data_dir


def get_pm25_attributes():
    """Return list of PM2.5 feature names."""
    return ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']


def load_pm25_raw(data_dir):
    """
    Load Beijing PM2.5 data, reshaped to weekly chunks.

    Args:
        data_dir: Path to PM2.5 data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=260 weeks, T=168 hours, D=7
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)

    # Find CSV file
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    df = pd.read_csv(csv_files[0])

    attributes = get_pm25_attributes()

    # Extract feature columns
    data_matrix = df[attributes].values

    # Reshape to weekly chunks (168 hours per week = 7 days * 24 hours)
    hours_per_week = 168
    n_weeks = len(data_matrix) // hours_per_week

    # Truncate to complete weeks
    data_truncated = data_matrix[:n_weeks * hours_per_week]

    # Reshape to (N_weeks, T=168, D=7)
    observed_values = data_truncated.reshape(n_weeks, hours_per_week, len(attributes))

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded PM2.5: {observed_values.shape[0]} weeks, "
          f"{observed_values.shape[1]} hours, {observed_values.shape[2]} features")

    return observed_values, observed_mask, attributes


# ============================================================================
# CMAPSS Dataset Functions
# ============================================================================

def download_cmapss(data_dir):
    """
    Download NASA CMAPSS jet engine dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if already extracted
    train_file = data_dir / "train_FD001.txt"
    if train_file.exists():
        print(f"CMAPSS data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    zip_path = data_dir / "CMAPSSData.zip"

    if not zip_path.exists():
        print("Downloading CMAPSS dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    print(f"CMAPSS data saved to {data_dir}")
    return data_dir


def get_cmapss_attributes():
    """
    Return list of CMAPSS sensor names (varying sensors only).

    Note: The actual varying sensors are determined dynamically during loading
    based on the combined FD001-FD004 data. This list is representative.
    """
    return [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7', 'sensor_8',
        'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
        'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
    ]


def load_cmapss_raw(data_dir):
    """
    Load CMAPSS data from all FD001-FD004 datasets, filtered to engines with >= 207 cycles.

    Combines all four datasets (FD001, FD002, FD003, FD004) and keeps only engines
    with at least 207 cycles (the median across all datasets). Each engine is truncated
    to the first 207 cycles.

    Args:
        data_dir: Path to CMAPSS data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=engines with >=207 cycles, T=207, D=varying sensors
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)
    MIN_CYCLES = 207  # Median cycle length across all FD001-FD004

    # Column names
    col_names = ['unit_id', 'cycle'] + \
                [f'op_setting_{i}' for i in range(1, 4)] + \
                [f'sensor_{i}' for i in range(1, 22)]

    # Load and combine all FD001-FD004 datasets
    all_dfs = []
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        train_path = data_dir / f'train_{fd}.txt'
        if train_path.exists():
            df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
            # Add dataset identifier and create globally unique engine ID
            df['dataset'] = fd
            df['global_engine_id'] = fd + '_' + df['unit_id'].astype(str)
            all_dfs.append(df)
            print(f"    Loaded {fd}: {df['unit_id'].nunique()} engines")

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_engines = combined_df['global_engine_id'].nunique()
    print(f"    Combined: {total_engines} total engines")

    # Calculate cycles per engine and filter
    cycles_per_engine = combined_df.groupby('global_engine_id')['cycle'].max()
    engines_to_keep = cycles_per_engine[cycles_per_engine >= MIN_CYCLES].index.tolist()
    filtered_df = combined_df[combined_df['global_engine_id'].isin(engines_to_keep)].copy()

    print(f"    Filtered to >= {MIN_CYCLES} cycles: {len(engines_to_keep)} engines")

    # Get varying sensors from the filtered combined data
    all_sensors = [f'sensor_{i}' for i in range(1, 22)]
    varying_sensors = [col for col in all_sensors if filtered_df[col].std() > 1e-6]
    attributes = varying_sensors

    # Reshape data: (N_engines, T_cycles, D_sensors)
    n_engines = len(engines_to_keep)
    n_features = len(attributes)
    fixed_length = MIN_CYCLES

    observed_values = np.zeros((n_engines, fixed_length, n_features), dtype=np.float32)

    for i, engine_id in enumerate(engines_to_keep):
        engine_data = filtered_df[filtered_df['global_engine_id'] == engine_id][attributes].values
        # Take first MIN_CYCLES cycles
        observed_values[i] = engine_data[:fixed_length]

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    # Print breakdown by dataset
    print(f"  Loaded CMAPSS (combined FD001-FD004):")
    print(f"    Shape: {observed_values.shape} (N, T, D)")
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        n = len([e for e in engines_to_keep if e.startswith(fd)])
        print(f"    {fd}: {n} engines")

    return observed_values, observed_mask, attributes


# ============================================================================
# Gas Sensor Dataset Functions
# ============================================================================

def download_gassensor(data_dir):
    """
    Download UCI Gas Sensor Array Under Flow Modulation dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile
    import gzip

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if already extracted
    rawdata_path = data_dir / "rawdata.csv"
    if rawdata_path.exists():
        print(f"Gas Sensor data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip"
    zip_path = data_dir / "gassensor.zip"

    if not zip_path.exists():
        print("Downloading Gas Sensor dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract zip
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    # Extract gzipped rawdata file if needed
    rawdata_gz = data_dir / "rawdata.csv.gz"
    if rawdata_gz.exists() and not rawdata_path.exists():
        print("Extracting rawdata.csv.gz...")
        with gzip.open(rawdata_gz, 'rb') as f_in:
            with open(rawdata_path, 'wb') as f_out:
                f_out.write(f_in.read())

    print(f"Gas Sensor data saved to {data_dir}")
    return data_dir


def get_gassensor_attributes():
    """Return list of Gas Sensor channel names."""
    return [f'sensor_{i}' for i in range(1, 17)]


def load_gassensor_raw(data_dir):
    """
    Load UCI Gas Sensor Array data, downsampled and chunked.

    The raw data has 7500 time steps at 25Hz (5 minutes per experiment).
    Processing:
    1. Downsample 10x (strided) -> 750 steps at 2.5Hz
    2. Chunk each experiment into 5 segments of 150 time steps
    3. Final shape: 58 experiments * 5 chunks = 290 samples

    Args:
        data_dir: Path to Gas Sensor data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=290 chunks, T=150, D=16 sensors
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)

    # Load raw data
    rawdata_path = data_dir / "rawdata.csv"
    rawdata_gz = data_dir / "rawdata.csv.gz"

    if rawdata_path.exists():
        raw_df = pd.read_csv(rawdata_path)
    elif rawdata_gz.exists():
        raw_df = pd.read_csv(rawdata_gz, compression='gzip')
    else:
        raise FileNotFoundError(f"No raw data file found in {data_dir}")

    # Get time series columns
    time_cols = [c for c in raw_df.columns if c.startswith('dR_t')]

    # Get unique experiments and sensors
    experiments = sorted(raw_df['exp'].unique())
    sensors = sorted(raw_df['sensor'].unique())

    n_experiments = len(experiments)
    n_sensors = len(sensors)
    n_time_steps_raw = len(time_cols)

    # Reshape raw data into (N, T, D) format
    data_raw = np.zeros((n_experiments, n_time_steps_raw, n_sensors), dtype=np.float32)

    exp_to_idx = {exp_id: i for i, exp_id in enumerate(experiments)}
    sensor_to_idx = {sensor_id: j for j, sensor_id in enumerate(sensors)}

    for _, row in raw_df.iterrows():
        exp_idx = exp_to_idx[row['exp']]
        sensor_idx = sensor_to_idx[row['sensor']]
        data_raw[exp_idx, :, sensor_idx] = row[time_cols].values

    # Downsample 10x (strided): 7500 -> 750 time steps
    downsample_factor = 10
    data_downsampled = data_raw[:, ::downsample_factor, :]  # (58, 750, 16)

    # Chunk each experiment into 5 segments of 150 time steps
    chunk_size = 150
    n_chunks_per_exp = data_downsampled.shape[1] // chunk_size  # 750 // 150 = 5

    # Reshape to (N_exp, n_chunks, chunk_size, D) then flatten first two dims
    data_chunked = data_downsampled[:, :n_chunks_per_exp * chunk_size, :]
    data_chunked = data_chunked.reshape(n_experiments, n_chunks_per_exp, chunk_size, n_sensors)
    observed_values = data_chunked.reshape(-1, chunk_size, n_sensors)  # (290, 150, 16)

    attributes = get_gassensor_attributes()

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded Gas Sensor: {observed_values.shape[0]} chunks "
          f"({n_experiments} experiments × {n_chunks_per_exp} chunks), "
          f"{observed_values.shape[1]} time steps, {observed_values.shape[2]} sensors")

    return observed_values, observed_mask, attributes


# ============================================================================
# Air Quality Dataset Functions
# ============================================================================

def download_airquality(data_dir):
    """
    Download UCI Air Quality dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if CSV already exists
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"Air Quality data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    zip_path = data_dir / "airquality.zip"

    if not zip_path.exists():
        print("Downloading Air Quality dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    print(f"Air Quality data saved to {data_dir}")
    return data_dir


def get_airquality_attributes():
    """Return list of Air Quality feature names."""
    return [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]


def load_airquality_raw(data_dir):
    """
    Load UCI Air Quality data, reshaped to daily chunks.

    The raw data has hourly measurements from March 2004 to February 2005.
    Missing values are marked as -200 in the original data and converted to NaN.

    Args:
        data_dir: Path to Air Quality data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=~389 days, T=24 hours, D=13 features
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)

    # Find CSV file
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    # Load CSV (uses semicolon separator and comma as decimal - European format)
    df = pd.read_csv(csv_files[0], sep=';', decimal=',')

    # Drop empty columns (CSV sometimes has trailing semicolons)
    df = df.dropna(axis=1, how='all')

    # Create datetime index
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )

    # Drop rows with invalid datetime
    df = df.dropna(subset=['datetime'])

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    attributes = get_airquality_attributes()

    # Replace -200 with NaN (missing value marker in this dataset)
    for col in attributes:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df.loc[df[col] == -200, col] = np.nan

    # Extract feature columns
    data_matrix = df[attributes].values

    # Reshape to daily chunks (24 hours per day)
    hours_per_day = 24
    n_days = len(data_matrix) // hours_per_day

    # Truncate to complete days
    data_truncated = data_matrix[:n_days * hours_per_day]

    # Reshape to (N_days, T=24, D=13)
    observed_values = data_truncated.reshape(n_days, hours_per_day, len(attributes))

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage (same as other datasets)
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded Air Quality: {observed_values.shape[0]} days, "
          f"{observed_values.shape[1]} hours, {observed_values.shape[2]} features")

    return observed_values, observed_mask, attributes


# ============================================================================
# CNNpred Stock Market Dataset Functions
# ============================================================================

def download_cnnpred(data_dir):
    """
    Download UCI CNNpred Stock Market dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if CSV files already exist
    csv_files = list(data_dir.glob("Processed_*.csv"))
    if csv_files:
        print(f"CNNpred data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/554/cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables.zip"
    zip_path = data_dir / "cnnpred.zip"

    if not zip_path.exists():
        print("Downloading CNNpred dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    print(f"CNNpred data saved to {data_dir}")
    return data_dir


def get_cnnpred_attributes():
    """
    Return list of CNNpred feature names.

    Note: This returns common numeric features across all 5 indices.
    The actual features are determined dynamically during loading.
    """
    # These are representative features - actual list determined at load time
    return [
        'AAPL', 'AMZN', 'AUD', 'Brent', 'CAC-F', 'CAD', 'CHF', 'CNY', 'CTB1Y',
        'CTB3M', 'CTB6M', 'Close', 'DAAA', 'DBAA', 'DE1', 'DE2', 'DE4', 'DE5',
        'DE6', 'DGS10', 'DGS5', 'DJI-F', 'DTB3', 'DTB4WK', 'DTB6', 'DAX-F',
        'EMA_10', 'EMA_20', 'EMA_200', 'EMA_50', 'EUR', 'FCHI', 'FTSE', 'FTSE-F',
        'GAS-F', 'GBP', 'GDAXI', 'GE', 'Gold', 'HSI', 'HSI-F', 'JNJ', 'JPM',
        'JPY', 'KOSPI-F', 'MSFT', 'NASDAQ-F', 'NZD', 'Nikkei-F', 'Oil',
        'ROC_10', 'ROC_15', 'ROC_20', 'ROC_5', 'RUSSELL-F', 'S&P-F', 'SSEC',
        'TE1', 'TE2', 'TE3', 'TE5', 'TE6', 'Volume', 'WFC', 'WIT-oil', 'XAG',
        'XAU', 'XOM', 'copper-F', 'gold-F', 'mom', 'mom1', 'mom2', 'mom3',
        'silver-F', 'wheat-F'
    ]


def load_cnnpred_raw(data_dir):
    """
    Load UCI CNNpred Stock Market data, reshaped to quarterly chunks.

    Combines 5 US market indices (S&P 500, NASDAQ, DJI, RUSSELL, NYSE)
    into non-overlapping quarterly chunks of 60 trading days each.

    Args:
        data_dir: Path to CNNpred data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=165 quarters, T=60 days, D=~79 features
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)
    QUARTER_SIZE = 60  # ~3 months of trading days

    # Load all CSV files
    csv_files = sorted(data_dir.glob("Processed_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Processed_*.csv files found in {data_dir}")

    datasets = {}
    for csv_path in csv_files:
        name = csv_path.stem.replace("Processed_", "")
        df = pd.read_csv(csv_path)
        datasets[name] = df

    # Find common numeric columns across all indices
    all_columns = [set(df.columns) for df in datasets.values()]
    common_cols = set.intersection(*all_columns)

    # Get numeric columns only (excludes 'Date', 'Name', etc.)
    sample_df = datasets[list(datasets.keys())[0]]
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter: must be common AND numeric
    feature_cols = sorted([c for c in common_cols if c in numeric_cols])
    attributes = feature_cols

    # Create quarterly chunks for each index
    all_quarters = []
    for name, df in datasets.items():
        data = df[feature_cols].values.astype(np.float32)
        n_quarters = len(data) // QUARTER_SIZE
        data_truncated = data[:n_quarters * QUARTER_SIZE]
        quarters = data_truncated.reshape(n_quarters, QUARTER_SIZE, len(feature_cols))
        all_quarters.append(quarters)

    # Stack all quarters
    observed_values = np.concatenate(all_quarters, axis=0)

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded CNNpred: {observed_values.shape[0]} quarters, "
          f"{observed_values.shape[1]} trading days, {observed_values.shape[2]} features")
    print(f"  (Combined from {len(datasets)} indices: {list(datasets.keys())})")

    return observed_values, observed_mask, attributes


# ============================================================================
# Gait Dataset Functions
# ============================================================================

def download_gait(data_dir):
    """
    Download UCI Multivariate Gait dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to data directory
    """
    import wget
    import zipfile

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Check if CSV already exists
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"Gait data already exists at {data_dir}")
        return data_dir

    # Download
    url = "https://archive.ics.uci.edu/static/public/760/multivariate+gait+data.zip"
    zip_path = data_dir / "gait.zip"

    if not zip_path.exists():
        print("Downloading Gait dataset...")
        wget.download(url, out=str(zip_path))
        print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    print(f"Gait data saved to {data_dir}")
    return data_dir


def get_gait_attributes():
    """Return list of Gait feature names (bilateral joint angles)."""
    return [
        'Left_Ankle', 'Left_Hip', 'Left_Knee',
        'Right_Ankle', 'Right_Hip', 'Right_Knee'
    ]


def load_gait_raw(data_dir):
    """
    Load UCI Multivariate Gait data, reshaped to (N, T, D) format.

    Combines 10 subjects x 3 conditions x 10 cycles into N=300 samples.
    Each sample has T=101 time points and D=6 joint angle features.

    Args:
        data_dir: Path to Gait data directory

    Returns:
        observed_values: Array of shape (N, T, D) - N=300 cycles, T=101 time, D=6 joints
        observed_mask: Array of shape (N, T, D) with 1=observed, 0=missing
        attributes: List of attribute names
    """
    import pandas as pd

    data_dir = Path(data_dir)

    # Find CSV file
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    df = pd.read_csv(csv_files[0])

    # Column mappings (integers to readable names)
    leg_map = {1: 'Left', 2: 'Right'}
    joint_map = {1: 'Ankle', 2: 'Hip', 3: 'Knee'}

    # Create readable feature column
    df['leg_joint'] = df['leg'].map(leg_map) + '_' + df['joint'].map(joint_map)

    # Feature names
    attributes = get_gait_attributes()

    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=['subject', 'condition', 'replication', 'time'],
        columns='leg_joint',
        values='angle',
        aggfunc='first'
    ).reset_index()

    # Get dimensions
    sample_groups = pivot_df.groupby(['subject', 'condition', 'replication'])
    n_samples = len(sample_groups)
    n_time = df['time'].nunique()
    n_features = len(attributes)

    # Build 3D array
    observed_values = np.zeros((n_samples, n_time, n_features), dtype=np.float32)

    for i, ((subj, cond, rep), group) in enumerate(sample_groups):
        group_sorted = group.sort_values('time')
        observed_values[i] = group_sorted[attributes].values

    # Create observed mask (1 where data is not NaN)
    observed_mask = (~np.isnan(observed_values)).astype("float32")

    # Replace NaN with 0 for storage
    observed_values = np.nan_to_num(observed_values).astype("float32")

    print(f"  Loaded Gait: {observed_values.shape[0]} cycles, "
          f"{observed_values.shape[1]} time points, {observed_values.shape[2]} joint angles")
    print(f"  (10 subjects × 3 conditions × 10 replications)")

    return observed_values, observed_mask, attributes


def compute_feature_stats(observed_values, observed_mask):
    """
    Compute mean and std per feature from observed values only.

    Args:
        observed_values: Array of shape (N, T, D)
        observed_mask: Array of shape (N, T, D)

    Returns:
        feature_means: Array of shape (D,)
        feature_stds: Array of shape (D,)
    """
    N, T, D = observed_values.shape
    feature_means = np.zeros(D)
    feature_stds = np.zeros(D)

    for d in range(D):
        mask_d = observed_mask[:, :, d].reshape(-1)
        values_d = observed_values[:, :, d].reshape(-1)
        if mask_d.sum() > 0:
            feature_means[d] = values_d[mask_d > 0].mean()
            feature_stds[d] = values_d[mask_d > 0].std()
            if feature_stds[d] == 0:
                feature_stds[d] = 1.0

    return feature_means, feature_stds


# ============================================================================
# Dataset Registry
# ============================================================================

def download_dataset(dataset, data_dir, skip_download=False):
    """
    Download dataset if needed.

    Args:
        dataset: Dataset name
        data_dir: Directory to save data
        skip_download: If True, skip downloading

    Returns:
        Path to data directory
    """
    if dataset == "physionet":
        data_dir = Path(data_dir) / "physio"
        if not skip_download:
            download_physionet(data_dir)
        return data_dir
    elif dataset in ["pems04", "pems08"]:
        data_dir = Path(data_dir) / "pems"
        if not skip_download:
            download_pems(data_dir, dataset)
        return data_dir
    elif dataset == "har":
        data_dir = Path(data_dir) / "har"
        if not skip_download:
            download_har(data_dir)
        return data_dir
    elif dataset == "pm25":
        data_dir = Path(data_dir) / "pm25"
        if not skip_download:
            download_pm25(data_dir)
        return data_dir
    elif dataset == "cmapss":
        data_dir = Path(data_dir) / "cmapss"
        if not skip_download:
            download_cmapss(data_dir)
        return data_dir
    elif dataset == "gassensor":
        data_dir = Path(data_dir) / "gassensor"
        if not skip_download:
            download_gassensor(data_dir)
        return data_dir
    elif dataset == "airquality":
        data_dir = Path(data_dir) / "airquality"
        if not skip_download:
            download_airquality(data_dir)
        return data_dir
    elif dataset == "cnnpred":
        data_dir = Path(data_dir) / "cnnpred"
        if not skip_download:
            download_cnnpred(data_dir)
        return data_dir
    elif dataset == "gait":
        data_dir = Path(data_dir) / "gait"
        if not skip_download:
            download_gait(data_dir)
        return data_dir
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_dataset(dataset, data_dir):
    """
    Load raw dataset.

    Args:
        dataset: Dataset name
        data_dir: Path to data directory

    Returns:
        observed_values: Array of shape (N, T, D)
        observed_mask: Array of shape (N, T, D)
        attributes: List of attribute names
    """
    if dataset == "physionet":
        return load_physionet_raw(data_dir)
    elif dataset in ["pems04", "pems08"]:
        return load_pems_raw(data_dir, dataset)
    elif dataset == "har":
        return load_har_raw(data_dir)
    elif dataset == "pm25":
        return load_pm25_raw(data_dir)
    elif dataset == "cmapss":
        return load_cmapss_raw(data_dir)
    elif dataset == "gassensor":
        return load_gassensor_raw(data_dir)
    elif dataset == "airquality":
        return load_airquality_raw(data_dir)
    elif dataset == "cnnpred":
        return load_cnnpred_raw(data_dir)
    elif dataset == "gait":
        return load_gait_raw(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================================
# Train-Val-Test Split with Deterministic Test Set
# ============================================================================

def split_data_deterministic(observed_values, observed_mask, gt_mask,
                              train_ratio, val_ratio, base_seed,
                              dataset, missing_type, missing_ratio):
    """
    Split data into train/val/test sets with DETERMINISTIC test set.

    The test set is determined ONLY by:
    - dataset name
    - missing_type
    - missing_ratio
    - test_ratio (= 1 - train_ratio - val_ratio)
    - base_seed

    This ensures that for the same dataset, missing pattern, and test proportion,
    the test set will be IDENTICAL regardless of how train/val are split.

    Args:
        observed_values: Array of shape (N, T, D)
        observed_mask: Array of shape (N, T, D)
        gt_mask: Array of shape (N, T, D)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        base_seed: Base random seed
        dataset: Dataset name (for deterministic seed generation)
        missing_type: Missing pattern type (for deterministic seed generation)
        missing_ratio: Missing ratio (for deterministic seed generation)

    Returns:
        Dictionary with train/val/test data
    """
    N = len(observed_values)
    test_ratio = 1 - train_ratio - val_ratio

    # STEP 1: Generate test set using a seed that depends ONLY on test-related params
    # This seed is independent of train_ratio and val_ratio
    test_seed = generate_deterministic_seed(
        base_seed,
        "test_split",  # fixed string to identify this as test split seed
        dataset,
        missing_type,
        missing_ratio,
        test_ratio
    )
    test_rng = np.random.default_rng(test_seed)

    # Shuffle all indices with the test RNG
    all_indices = np.arange(N)
    test_rng.shuffle(all_indices)

    # Select test indices (last portion after shuffle)
    n_test = int(N * test_ratio)
    test_idx = all_indices[-n_test:]  # Take from the end
    remaining_idx = all_indices[:-n_test]  # Everything else for train/val

    # STEP 2: Split remaining data into train/val using a different seed
    # This seed depends on train_ratio and val_ratio, so different splits
    # will give different train/val partitions of the remaining data
    trainval_seed = generate_deterministic_seed(
        base_seed,
        "trainval_split",
        dataset,
        missing_type,
        missing_ratio,
        train_ratio,
        val_ratio
    )
    trainval_rng = np.random.default_rng(trainval_seed)

    # Shuffle remaining indices
    trainval_rng.shuffle(remaining_idx)

    # Split into train and val
    # Adjust ratios for the remaining data
    remaining_ratio = train_ratio + val_ratio
    train_ratio_adjusted = train_ratio / remaining_ratio
    n_train = int(len(remaining_idx) * train_ratio_adjusted)

    train_idx = remaining_idx[:n_train]
    val_idx = remaining_idx[n_train:]

    # Sort indices for reproducibility in output (optional, but nice for debugging)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    splits = {
        "train": {
            "observed_values": observed_values[train_idx],
            "observed_mask": observed_mask[train_idx],
            "gt_mask": gt_mask[train_idx],
            "indices": train_idx,
        },
        "val": {
            "observed_values": observed_values[val_idx],
            "observed_mask": observed_mask[val_idx],
            "gt_mask": gt_mask[val_idx],
            "indices": val_idx,
        },
        "test": {
            "observed_values": observed_values[test_idx],
            "observed_mask": observed_mask[test_idx],
            "gt_mask": gt_mask[test_idx],
            "indices": test_idx,
        },
    }

    return splits


# ============================================================================
# Save Functions
# ============================================================================

def get_output_filename(dataset, missing_type, missing_ratio, train_ratio, val_ratio, split_name, seed):
    """
    Generate output filename.

    Format: {dataset}_{missing_type}_{missing_ratio}pct_split{train}-{val}-{test}_{split}_seed{seed}.pkl
    """
    missing_str = f"{int(missing_ratio * 100)}pct"
    train_pct = int(train_ratio * 100)
    val_pct = int(val_ratio * 100)
    test_pct = 100 - train_pct - val_pct
    split_str = f"split{train_pct}-{val_pct}-{test_pct}"
    return f"{dataset}_{missing_type}_{missing_str}_{split_str}_{split_name}_seed{seed}.pkl"


def save_split(split_data, output_dir, dataset, missing_type, missing_ratio,
               train_ratio, val_ratio, split_name, seed, attributes, metadata):
    """
    Save a single split to disk.

    Args:
        split_data: Dictionary with observed_values, observed_mask, gt_mask
        output_dir: Output directory
        dataset: Dataset name
        missing_type: Missing pattern type
        missing_ratio: Missing ratio
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        split_name: "train", "val", or "test"
        seed: Random seed
        attributes: List of attribute names
        metadata: Additional metadata to save
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    filename = get_output_filename(dataset, missing_type, missing_ratio, train_ratio, val_ratio, split_name, seed)
    filepath = output_dir / filename

    save_data = {
        "observed_values": split_data["observed_values"],
        "observed_mask": split_data["observed_mask"],
        "gt_mask": split_data["gt_mask"],
        "indices": split_data["indices"],
        "attributes": attributes,
        "metadata": metadata,
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)

    print(f"  Saved {split_name}: {filepath}")
    print(f"    Samples: {len(split_data['observed_values'])}")

    return filepath


# ============================================================================
# Main Processing Function
# ============================================================================

def process_data(args):
    """
    Main data processing function.

    Args:
        args: Parsed command line arguments
    """
    print("\n" + "=" * 70)
    print("Data Processing for Imputation")
    print("=" * 70)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Missing type:  {args.missing_type.upper()}")
    print(f"  Missing ratio: {args.missing_ratio * 100:.0f}%")
    print(f"  Train ratio:   {args.train_ratio * 100:.0f}%")
    print(f"  Val ratio:     {args.val_ratio * 100:.0f}%")
    test_ratio = 1 - args.train_ratio - args.val_ratio
    print(f"  Test ratio:    {test_ratio * 100:.0f}%")
    print(f"  Seed:          {args.seed}")
    if args.missing_type == "mnar":
        print(f"  MNAR steepness: {args.mnar_steepness}")
    print(f"  Start time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Set data path
    if args.data_path is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(args.data_path)

    # Set output path
    if args.output_path is None:
        output_dir = DEFAULT_OUTPUT_DIR / args.dataset
    else:
        output_dir = Path(args.output_path)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Download data
    print("\n" + "-" * 70)
    print("Step 1: Downloading data")
    print("-" * 70)
    data_dir = download_dataset(args.dataset, data_dir, skip_download=args.no_download)

    # Step 2: Load raw data
    print("\n" + "-" * 70)
    print("Step 2: Loading raw data")
    print("-" * 70)
    observed_values, observed_mask, attributes = load_dataset(args.dataset, data_dir)
    print(f"  Loaded {len(observed_values)} samples")
    print(f"  Shape: {observed_values.shape} (N, T, D)")
    print(f"  Attributes: {len(attributes)}")

    # Step 3: Compute feature statistics for MNAR mask generation (uses full data)
    # NOTE: These stats are ONLY used for MNAR masking, NOT for normalization
    print("\n" + "-" * 70)
    print("Step 3: Computing feature statistics (full data, for MNAR masking)")
    print("-" * 70)
    full_feature_means, full_feature_stds = compute_feature_stats(observed_values, observed_mask)
    print(f"  Computed mean/std for {len(full_feature_means)} features (full data)")

    # Step 4: Generate masks using a dedicated seed
    # This seed depends on dataset, missing_type, missing_ratio, and base seed
    # but NOT on train/val ratios
    print("\n" + "-" * 70)
    print("Step 4: Generating masks")
    print("-" * 70)
    print(f"  Pattern: {args.missing_type.upper()}")
    print(f"  Target missing ratio: {args.missing_ratio * 100:.0f}%")

    mask_seed = generate_deterministic_seed(
        args.seed,
        "mask_generation",
        args.dataset,
        args.missing_type,
        args.missing_ratio
    )
    mask_rng = np.random.default_rng(mask_seed)

    gt_mask = generate_masks(
        observed_values, observed_mask,
        args.missing_type, args.missing_ratio,
        full_feature_means, full_feature_stds,  # Use full-data stats for MNAR
        args.mnar_steepness, mask_rng
    )

    # Report actual missing ratio
    total_observed = observed_mask.sum()
    total_remaining = gt_mask.sum()
    actual_missing = 1 - (total_remaining / total_observed)
    print(f"  Original observed: {int(total_observed)}")
    print(f"  After masking: {int(total_remaining)}")
    print(f"  Actual missing ratio: {actual_missing * 100:.1f}%")

    # Step 5: Split data with deterministic test set
    print("\n" + "-" * 70)
    print("Step 5: Splitting data (deterministic test set)")
    print("-" * 70)
    print(f"  NOTE: Test set is determined by (dataset, missing_type, missing_ratio, test_ratio, seed)")
    print(f"        Different train/val splits with same test_ratio will have SAME test set!")

    splits = split_data_deterministic(
        observed_values, observed_mask, gt_mask,
        args.train_ratio, args.val_ratio, args.seed,
        args.dataset, args.missing_type, args.missing_ratio
    )
    print(f"  Train: {len(splits['train']['observed_values'])} samples")
    print(f"  Val:   {len(splits['val']['observed_values'])} samples")
    print(f"  Test:  {len(splits['test']['observed_values'])} samples")

    # Step 5b: Compute normalization statistics from TRAINING set only (no data leakage)
    print("\n" + "-" * 70)
    print("Step 5b: Computing normalization statistics (train set only)")
    print("-" * 70)
    train_values = splits['train']['observed_values']
    train_mask = splits['train']['observed_mask']
    feature_means, feature_stds = compute_feature_stats(train_values, train_mask)
    print(f"  Computed mean/std for {len(feature_means)} features (from train set only)")

    # Step 6: Save data
    print("\n" + "-" * 70)
    print("Step 6: Saving data")
    print("-" * 70)

    # Metadata to save with each file
    metadata = {
        "dataset": args.dataset,
        "missing_type": args.missing_type,
        "missing_ratio": args.missing_ratio,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": test_ratio,
        "seed": args.seed,
        "mnar_steepness": args.mnar_steepness if args.missing_type == "mnar" else None,
        "feature_means": feature_means,  # Computed from train set only
        "feature_stds": feature_stds,    # Computed from train set only
        "normalization_source": "train_only",  # No data leakage
        "total_samples": len(observed_values),
        "shape": observed_values.shape,
        "actual_missing_ratio": actual_missing,
        "created_at": datetime.now().isoformat(),
        "deterministic_test_set": True,  # Flag indicating this uses the new splitting strategy
    }

    saved_files = []
    for split_name in ["train", "val", "test"]:
        filepath = save_split(
            splits[split_name], output_dir,
            args.dataset, args.missing_type, args.missing_ratio,
            args.train_ratio, args.val_ratio,
            split_name, args.seed, attributes, metadata
        )
        saved_files.append(filepath)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Missing pattern: {args.missing_type.upper()} ({args.missing_ratio * 100:.0f}%)")
    print(f"Splits: train ({args.train_ratio*100:.0f}%), val ({args.val_ratio*100:.0f}%), test ({test_ratio*100:.0f}%)")
    print(f"\nDETERMINISTIC TEST SET: For fair comparison, the test set is fixed")
    print(f"for the same (dataset, missing_type, missing_ratio, test_ratio, seed).")
    print(f"\nSaved files:")
    for f in saved_files:
        print(f"  {f}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return saved_files


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    process_data(args)
