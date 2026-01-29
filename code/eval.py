"""
Evaluation Metrics for Imputation Models

This module provides evaluation metrics for comparing imputed values against ground truth:
- MSE (Mean Squared Error): Point-wise accuracy on held-out entries
- MAE (Mean Absolute Error): Point-wise accuracy on held-out entries
- Error summary statistics: percentiles (50th, 90th, 95th, 99th), std, min, max
- Global W2: Wasserstein-2 distance on completed dataset
"""

import numpy as np
from typing import Dict, Optional


def compute_mse(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute Mean Squared Error on held-out entries.

    Args:
        X_true: Ground truth values, shape (N, D, T)
        X_pred: Predicted values, same shape as X_true
        mask: Binary mask where 1 = held-out entry to evaluate

    Returns:
        MSE value (scalar)
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.nan

    errors = (X_true[mask] - X_pred[mask]) ** 2
    return float(np.mean(errors))


def compute_rmse(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute Root Mean Squared Error on held-out entries.
    """
    mse = compute_mse(X_true, X_pred, mask)
    return float(np.sqrt(mse)) if not np.isnan(mse) else np.nan


def compute_mae(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error on held-out entries.
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.nan

    errors = np.abs(X_true[mask] - X_pred[mask])
    return float(np.mean(errors))


def compute_error_summary(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray,
    percentiles: tuple = (50, 90, 95, 99)
) -> Dict[str, float]:
    """
    Compute summary statistics of absolute errors on held-out entries.

    Args:
        X_true: Ground truth values, shape (N, D, T)
        X_pred: Predicted values, same shape as X_true
        mask: Binary mask where 1 = held-out entry to evaluate
        percentiles: Tuple of percentiles to compute (default: 50, 90, 95, 99)

    Returns:
        Dictionary with:
        - 'mean': Mean absolute error
        - 'std': Standard deviation of absolute errors
        - 'min': Minimum absolute error
        - 'max': Maximum absolute error
        - 'p{X}': Xth percentile of absolute errors (for each X in percentiles)
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        result = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        for p in percentiles:
            result[f'p{p}'] = np.nan
        return result

    errors = np.abs(X_true[mask] - X_pred[mask])

    result = {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
    }

    for p in percentiles:
        result[f'p{p}'] = float(np.percentile(errors, p))

    return result


def compute_squared_error_summary(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray,
    percentiles: tuple = (50, 90, 95, 99)
) -> Dict[str, float]:
    """
    Compute summary statistics of squared errors on held-out entries.

    Args:
        X_true: Ground truth values, shape (N, D, T)
        X_pred: Predicted values, same shape as X_true
        mask: Binary mask where 1 = held-out entry to evaluate
        percentiles: Tuple of percentiles to compute (default: 50, 90, 95, 99)

    Returns:
        Dictionary with:
        - 'mean': Mean squared error (MSE)
        - 'std': Standard deviation of squared errors
        - 'min': Minimum squared error
        - 'max': Maximum squared error
        - 'p{X}': Xth percentile of squared errors (for each X in percentiles)
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        result = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        for p in percentiles:
            result[f'p{p}'] = np.nan
        return result

    errors = (X_true[mask] - X_pred[mask]) ** 2

    result = {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
    }

    for p in percentiles:
        result[f'p{p}'] = float(np.percentile(errors, p))

    return result


def compute_global_w2(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    gt_mask: np.ndarray,
    observed_mask: np.ndarray
) -> float:
    """
    Compute global Wasserstein-2 distance on the completed dataset.

    Compares the distribution of the completed data (observed values + imputed values)
    against the full ground truth, restricted to entries where ground truth exists.

    This measures whether the imputation preserves the overall data distribution,
    not just point-wise accuracy.

    Args:
        X_true: Ground truth values, shape (N, D, T)
        X_pred: Predicted values, shape (N, D, T)
        gt_mask: Mask of entries used for conditioning (1 = observed by model)
        observed_mask: Mask of all entries with ground truth (1 = not naturally missing)

    Returns:
        W2 distance (scalar)
    """
    # Create completed data: keep observed values (gt_mask=1), use predictions elsewhere
    X_completed = X_true * gt_mask + X_pred * (1 - gt_mask)

    # Only compare where we have ground truth (observed_mask = 1)
    mask = observed_mask.astype(bool)

    y_true = X_true[mask]
    y_completed = X_completed[mask]

    if len(y_true) == 0:
        return np.nan

    # Sort and compute W2
    y_true_sorted = np.sort(y_true)
    y_completed_sorted = np.sort(y_completed)

    w2 = np.sqrt(np.mean((y_true_sorted - y_completed_sorted) ** 2))
    return float(w2)


def evaluate_imputation(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray,
    min_samples_w2: int = 10,
    percentiles: tuple = (50, 90, 95, 99),
    gt_mask: Optional[np.ndarray] = None,
    observed_mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Run full evaluation on imputation results.

    Args:
        X_true: Ground truth values, shape (N, D, T)
        X_pred: Predicted values, shape (N, D, T)
        mask: Binary mask where 1 = held-out entry to evaluate, shape (N, D, T)
        min_samples_w2: Minimum samples for W2 computation (unused, kept for API compatibility)
        percentiles: Tuple of percentiles to compute for error summaries
        gt_mask: Optional mask of entries used for conditioning (1 = observed by model).
                 Required for global W2 computation.
        observed_mask: Optional mask of all entries with ground truth (1 = not naturally missing).
                       Required for global W2 computation.

    Returns:
        Dictionary with all metrics:
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'ae_summary': Absolute error summary stats (mean, std, min, max, percentiles)
        - 'se_summary': Squared error summary stats (mean, std, min, max, percentiles)
        - 'w2_global': Global W2 on completed dataset (if gt_mask and observed_mask provided)
    """
    mse = compute_mse(X_true, X_pred, mask)
    rmse = compute_rmse(X_true, X_pred, mask)
    mae = compute_mae(X_true, X_pred, mask)

    # Error summary statistics
    ae_summary = compute_error_summary(X_true, X_pred, mask, percentiles)
    se_summary = compute_squared_error_summary(X_true, X_pred, mask, percentiles)

    # Compute global W2 if masks are provided
    if gt_mask is not None and observed_mask is not None:
        w2_global = compute_global_w2(X_true, X_pred, gt_mask, observed_mask)
    else:
        w2_global = np.nan

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'ae_summary': ae_summary,
        'se_summary': se_summary,
        'w2_global': w2_global,
    }


def print_evaluation_results(results: Dict, feature_names: Optional[list] = None):
    """
    Pretty print evaluation results.

    Args:
        results: Dictionary from evaluate_imputation
        feature_names: Optional list of feature names for display (unused)
    """
    print("=" * 60)
    print("Imputation Evaluation Results")
    print("=" * 60)
    print(f"\nPoint-wise Metrics (on held-out entries):")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")

    # Print absolute error summary
    if 'ae_summary' in results:
        ae = results['ae_summary']
        print(f"\nAbsolute Error Summary:")
        print(f"  Mean:   {ae['mean']:.6f}")
        print(f"  Std:    {ae['std']:.6f}")
        print(f"  Min:    {ae['min']:.6f}")
        print(f"  Max:    {ae['max']:.6f}")
        # Print percentiles
        percentile_keys = sorted([k for k in ae.keys() if k.startswith('p')])
        if percentile_keys:
            print(f"  Percentiles:")
            for k in percentile_keys:
                print(f"    {k}: {ae[k]:.6f}")

    # Print squared error summary
    if 'se_summary' in results:
        se = results['se_summary']
        print(f"\nSquared Error Summary:")
        print(f"  Mean (MSE): {se['mean']:.6f}")
        print(f"  Std:        {se['std']:.6f}")
        print(f"  Min:        {se['min']:.6f}")
        print(f"  Max:        {se['max']:.6f}")
        # Print percentiles
        percentile_keys = sorted([k for k in se.keys() if k.startswith('p')])
        if percentile_keys:
            print(f"  Percentiles:")
            for k in percentile_keys:
                print(f"    {k}: {se[k]:.6f}")

    # Print Global W2
    print(f"\nDistributional Metrics:")
    if 'w2_global' in results and not np.isnan(results['w2_global']):
        print(f"  Global W2: {results['w2_global']:.6f}")
    else:
        print(f"  Global W2: N/A (gt_mask and observed_mask not provided)")

    print("=" * 60)
