"""
BRITS Wrapper for DRIO Training

This module provides a wrapper class that adapts the BRITS (Bidirectional Recurrent
Imputation for Time Series) model to work with the DRIO training interface.

The wrapper handles:
1. Data format conversion: (B, D, T) DRIO format <-> (B, T, D) BRITS format
2. Delta computation: Time since last observation for each feature
3. Forward/backward sequence preparation for bidirectional processing
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# =============================================================================
# BRITS COMPONENTS (Self-contained to avoid benchmark import issues)
# =============================================================================

class FeatureRegression(nn.Module):
    """Feature regression layer from BRITS."""

    def __init__(self, input_size):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        # Mask to prevent self-regression
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    """Temporal decay layer from BRITS."""

    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        return torch.exp(-gamma)


class RITS(nn.Module):
    """Single-direction RITS model (used by BRITS for forward/backward)."""

    def __init__(self, d_features, d_time, rnn_hid_size=64):
        super().__init__()
        self.d_features = d_features
        self.d_time = d_time
        self.rnn_hid_size = rnn_hid_size

        # LSTM cell
        self.rnn_cell = nn.LSTMCell(d_features * 2, rnn_hid_size)

        # Temporal decay
        self.temp_decay_h = TemporalDecay(d_features, rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(d_features, d_features, diag=True)

        # Regression layers
        self.hist_reg = nn.Linear(rnn_hid_size, d_features)
        self.feat_reg = FeatureRegression(d_features)
        self.weight_combine = nn.Linear(d_features * 2, d_features)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, values, masks, deltas, return_raw=False):
        """
        Forward pass for single direction.

        Args:
            values: (B, T, D) - observed values (0 where missing)
            masks: (B, T, D) - binary mask (1=observed)
            deltas: (B, T, D) - time since last observation
            return_raw: If True, return raw model estimates (c_h) instead of masked (c_c)

        Returns:
            imputations: (B, T, D) - imputed values (raw estimates if return_raw=True)
        """
        B, T, D = values.shape
        device = values.device

        h = torch.zeros(B, self.rnn_hid_size, device=device)
        c = torch.zeros(B, self.rnn_hid_size, device=device)

        imputations = []

        for t in range(T):
            x = values[:, t, :]     # (B, D)
            m = masks[:, t, :]      # (B, D)
            d = deltas[:, t, :]     # (B, D)

            # Temporal decay
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            # Decay hidden state
            h = h * gamma_h

            # History-based estimation
            x_h = self.hist_reg(h)

            # Combine observed and estimated (for feature regression input)
            x_c = m * x + (1 - m) * x_h

            # Feature-based estimation
            z_h = self.feat_reg(x_c)

            # Weighted combination - this is the RAW model estimate
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h

            # Masked combination (for LSTM input and standard output)
            c_c = m * x + (1 - m) * c_h

            # LSTM update (always uses masked combination)
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            # Output: raw estimates (c_h) for DRIO loss, or masked (c_c) for standard use
            if return_raw:
                imputations.append(c_h.unsqueeze(1))
            else:
                imputations.append(c_c.unsqueeze(1))

        imputations = torch.cat(imputations, dim=1)  # (B, T, D)
        return imputations


class BRITS(nn.Module):
    """Bidirectional RITS model."""

    def __init__(self, d_features, d_time, rnn_hid_size=64):
        super().__init__()
        self.d_features = d_features
        self.d_time = d_time

        self.rits_f = RITS(d_features, d_time, rnn_hid_size)
        self.rits_b = RITS(d_features, d_time, rnn_hid_size)

    def forward(self, data, return_raw=False):
        """
        Forward pass.

        Args:
            data: dict with 'forward' and 'backward' keys, each containing:
                - values: (B, T, D)
                - masks: (B, T, D)
                - deltas: (B, T, D)
            return_raw: If True, return raw model estimates for DRIO loss computation

        Returns:
            dict with 'imputations': (B, T, D)
        """
        # Forward direction
        imp_f = self.rits_f(
            data['forward']['values'],
            data['forward']['masks'],
            data['forward']['deltas'],
            return_raw=return_raw
        )

        # Backward direction (reverse, process, reverse back)
        imp_b_rev = self.rits_b(
            data['backward']['values'],
            data['backward']['masks'],
            data['backward']['deltas'],
            return_raw=return_raw
        )
        # Reverse back to forward order
        imp_b = torch.flip(imp_b_rev, dims=[1])

        # Average forward and backward
        imputations = (imp_f + imp_b) / 2

        return {'imputations': imputations}


# =============================================================================
# BRITS WRAPPER FOR DRIO
# =============================================================================

class BRITSForDRIO(nn.Module):
    """
    Wrapper that adapts BRITS to the DRIO interface.

    DRIO expects:
        model(observed_data, gt_mask) -> predictions
        where shapes are (B, D, T)

    BRITS expects:
        model(data_dict) -> {'imputations': (B, T, D)}
        where data_dict has forward/backward with values, masks, deltas
    """

    def __init__(self, d_features, d_time, rnn_hid_size=64):
        """
        Initialize BRITS wrapper.

        Args:
            d_features: Number of features (D)
            d_time: Number of time steps (T)
            rnn_hid_size: Hidden size for LSTM cells
        """
        super().__init__()
        self.d_features = d_features
        self.d_time = d_time
        self.rnn_hid_size = rnn_hid_size

        # Create BRITS model
        self.brits = BRITS(d_features, d_time, rnn_hid_size)

    def _compute_deltas(self, mask):
        """
        Compute time deltas (time since last observation).

        Args:
            mask: (B, T, D) binary mask (1=observed)

        Returns:
            deltas: (B, T, D) time deltas
        """
        B, T, D = mask.shape
        device = mask.device

        deltas = torch.zeros(B, T, D, device=device)

        for t in range(T):
            if t == 0:
                deltas[:, t, :] = 0
            else:
                # Time since last observation
                deltas[:, t, :] = 1 + (1 - mask[:, t-1, :]) * deltas[:, t-1, :]

        return deltas

    def _prepare_brits_input(self, observed_data, gt_mask):
        """
        Convert DRIO format to BRITS dict format.

        Args:
            observed_data: (B, D, T) - observed data
            gt_mask: (B, D, T) - mask for what model sees as observed

        Returns:
            data_dict: BRITS-compatible input dict
        """
        # Permute: (B, D, T) -> (B, T, D)
        values_btd = observed_data.permute(0, 2, 1)  # (B, T, D)
        masks_btd = gt_mask.permute(0, 2, 1)         # (B, T, D)

        # Zero out missing values (BRITS expects 0 where missing)
        values_btd = values_btd * masks_btd

        # Compute deltas
        deltas_fwd = self._compute_deltas(masks_btd)

        # Backward: flip along time dimension
        values_bwd = torch.flip(values_btd, dims=[1])
        masks_bwd = torch.flip(masks_btd, dims=[1])
        deltas_bwd = self._compute_deltas(masks_bwd)

        data_dict = {
            'forward': {
                'values': values_btd,
                'masks': masks_btd,
                'deltas': deltas_fwd,
            },
            'backward': {
                'values': values_bwd,
                'masks': masks_bwd,
                'deltas': deltas_bwd,
            }
        }

        return data_dict

    def forward(self, observed_data, gt_mask):
        """
        Forward pass compatible with DRIO interface.

        Args:
            observed_data: (B, D, T) - observed data
            gt_mask: (B, D, T) - mask for what model sees as observed

        Returns:
            predictions: (B, D, T) - RAW model estimates (not masked with observed values)

        Note:
            For DRIO training, we need raw model estimates so that reconstruction
            loss can be computed on all positions. The original BRITS outputs
            masked values (observed values at observed positions), which would
            result in zero reconstruction loss at those positions.

            The internal LSTM computation still uses masked values (c_c) as input,
            preserving the original BRITS architecture. Only the OUTPUT changes
            to return raw estimates (c_h) for proper DRIO loss computation.
        """
        # Convert to BRITS format
        brits_input = self._prepare_brits_input(observed_data, gt_mask)

        # Run BRITS with return_raw=True to get raw model estimates
        ret = self.brits(brits_input, return_raw=True)

        # Permute back: (B, T, D) -> (B, D, T)
        imputations = ret['imputations'].permute(0, 2, 1)

        return imputations


def create_brits_for_drio(d_features, d_time, rnn_hid_size=64):
    """
    Factory function to create a BRITS model wrapped for DRIO training.

    Args:
        d_features: Number of features (D)
        d_time: Number of time steps (T)
        rnn_hid_size: Hidden size for LSTM cells

    Returns:
        BRITSForDRIO model
    """
    return BRITSForDRIO(d_features, d_time, rnn_hid_size)
