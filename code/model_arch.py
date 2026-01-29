"""
Model Architectures for Distributionally Robust Imputation

This module implements the imputer model backbones described in the paper:
1. Spatiotemporal Transformer - main architecture with factorized attention
2. LSTM baseline
3. GNN baseline (Graph Attention Network)

The imputer signature: G_theta: R^{D×T} × {0,1}^{D×T} → R^{D×T}
- Input: (X_obs, M) - observed data and mask
- Output: X_hat - complete imputed trajectory

Architecture from Appendix C:
- Input Embedding: Linear projection + positional encoding
- Factorized Attention: Temporal attention → Spatial attention
- Output Projection: Linear back to data space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for temporal and spatial dimensions.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Parameters:
        -----------
        x : torch.Tensor, shape (B, L, d_model)

        Returns:
        --------
        x : torch.Tensor, shape (B, L, d_model)
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (B, L, d_model)
        mask : torch.Tensor, optional, shape (B, L) or (B, L, L)

        Returns:
        --------
        output : torch.Tensor, shape (B, L, d_model)
        """
        B, L, _ = x.shape

        # Linear projections
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, num_heads, L, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, num_heads, L, L)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (B, num_heads, L, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with pre-norm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# SPATIOTEMPORAL TRANSFORMER (Main Architecture)
# =============================================================================

class SpatiotemporalTransformer(nn.Module):
    """
    Spatiotemporal Transformer for multivariate time series imputation.

    Architecture from Appendix C:
    1. Input: Concatenate mean-imputed data and mask along channel dimension
    2. Linear projection to d_model with positional encoding
    3. L layers of factorized attention:
       - Temporal attention: attend across time for each feature
       - Spatial attention: attend across features for each time step
    4. Output projection back to data space

    Parameters:
    -----------
    d_features : int
        Number of features D
    d_time : int
        Number of time steps T
    d_model : int
        Latent embedding dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of encoder layers
    d_ff : int
        Feed-forward hidden dimension
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_features = d_features
        self.d_time = d_time
        self.d_model = d_model

        # Input embedding: (data + mask) -> d_model
        # Input shape: (B, 2, D, T) -> flatten to (B, D*T, 2) -> project
        self.input_proj = nn.Linear(2, d_model)

        # Learnable positional encodings for spatial and temporal dimensions
        self.temporal_pos = LearnablePositionalEncoding(d_model, max_len=d_time)
        self.spatial_pos = LearnablePositionalEncoding(d_model, max_len=d_features)

        # Factorized attention layers
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Spatiotemporal Transformer.

        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
            Observed data (missing entries can be any value, will be replaced)
        mask : torch.Tensor, shape (B, D, T)
            Binary mask (1 = observed, 0 = missing)

        Returns:
        --------
        X_imputed : torch.Tensor, shape (B, D, T)
            Imputed complete trajectory
        """
        B, D, T = X_obs.shape

        # Mean imputation for initial values
        X_mean = self._mean_impute(X_obs, mask)

        # Stack data and mask: (B, D, T, 2)
        x = torch.stack([X_mean, mask], dim=-1)

        # Reshape for processing: (B, D, T, 2) -> (B, D*T, 2)
        x = x.view(B, D * T, 2)

        # Project to d_model: (B, D*T, d_model)
        h = self.input_proj(x)

        # Reshape for factorized attention: (B, D, T, d_model)
        h = h.view(B, D, T, self.d_model)

        # Apply factorized attention layers
        for temp_layer, spat_layer in zip(self.temporal_layers, self.spatial_layers):
            # Temporal attention: attend across T for each feature
            # Reshape: (B*D, T, d_model)
            h_temp = h.view(B * D, T, self.d_model)
            h_temp = self.temporal_pos(h_temp)
            h_temp = temp_layer(h_temp)
            h = h_temp.view(B, D, T, self.d_model)

            # Spatial attention: attend across D for each time step
            # Reshape: (B*T, D, d_model)
            h_spat = h.permute(0, 2, 1, 3).contiguous().view(B * T, D, self.d_model)
            h_spat = self.spatial_pos(h_spat)
            h_spat = spat_layer(h_spat)
            h = h_spat.view(B, T, D, self.d_model).permute(0, 2, 1, 3).contiguous()

        # Output projection: (B, D, T, d_model) -> (B, D, T, 1) -> (B, D, T)
        h = self.output_norm(h)
        output = self.output_proj(h).squeeze(-1)

        # Return raw predictions - let caller handle masking
        return output

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

        # Position-wise mean (avoid division by zero)
        position_mean = obs_sum / (obs_count + 1e-10)

        # Replace missing with mean
        X_mean = X_obs * mask + (1 - mask) * position_mean

        return X_mean


# =============================================================================
# LSTM BASELINE
# =============================================================================

class LSTMImputer(nn.Module):
    """
    Bidirectional LSTM for time series imputation.
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_features = d_features
        self.d_time = d_time

        # Input: (data + mask) for each feature
        self.input_proj = nn.Linear(2 * d_features, d_hidden)

        self.lstm = nn.LSTM(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_proj = nn.Linear(2 * d_hidden, d_features)

    def forward(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
        mask : torch.Tensor, shape (B, D, T)

        Returns:
        --------
        X_imputed : torch.Tensor, shape (B, D, T)
        """
        B, D, T = X_obs.shape

        # Mean imputation
        X_mean = self._mean_impute(X_obs, mask)

        # Concatenate data and mask: (B, T, 2*D)
        x = torch.cat([X_mean.permute(0, 2, 1), mask.permute(0, 2, 1)], dim=-1)

        # Project and process with LSTM
        h = self.input_proj(x)  # (B, T, d_hidden)
        h, _ = self.lstm(h)  # (B, T, 2*d_hidden)

        # Output projection
        output = self.output_proj(h)  # (B, T, D)
        output = output.permute(0, 2, 1)  # (B, D, T)

        # Return raw predictions - let caller handle masking
        return output

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per (d,t) position across batch."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)
        position_mean = obs_sum / (obs_count + 1e-10)
        return X_obs * mask + (1 - mask) * position_mean


# =============================================================================
# GNN BASELINE (Graph Attention Network)
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for spatial dependencies.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        super().__init__()

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (B, N, in_features)

        Returns:
        --------
        output : torch.Tensor, shape (B, N, out_features)
        """
        B, N, _ = x.shape

        # Linear transformation
        h = self.W(x)  # (B, N, out_features)

        # Compute attention coefficients
        # Concatenate all pairs: (B, N, N, 2*out_features)
        h_i = h.unsqueeze(2).expand(B, N, N, -1)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)
        concat = torch.cat([h_i, h_j], dim=-1)

        # Attention scores
        e = self.leaky_relu((concat * self.a).sum(dim=-1))  # (B, N, N)
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        # Apply attention
        output = torch.bmm(attention, h)  # (B, N, out_features)

        return output


class GATImputer(nn.Module):
    """
    Graph Attention Network for spatiotemporal imputation.

    Treats features as nodes and learns attention-based aggregation.
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_hidden: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_features = d_features
        self.d_time = d_time

        # Input projection: (T * 2) -> d_hidden (data + mask for each time)
        self.input_proj = nn.Linear(2 * d_time, d_hidden)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                nn.ModuleList([
                    GraphAttentionLayer(d_hidden, d_hidden // num_heads, dropout)
                    for _ in range(num_heads)
                ])
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_hidden) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_hidden, d_time)

    def forward(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
        mask : torch.Tensor, shape (B, D, T)

        Returns:
        --------
        X_imputed : torch.Tensor, shape (B, D, T)
        """
        B, D, T = X_obs.shape

        # Mean imputation
        X_mean = self._mean_impute(X_obs, mask)

        # Concatenate and project: (B, D, 2*T) -> (B, D, d_hidden)
        x = torch.cat([X_mean, mask], dim=-1)  # (B, D, 2*T)
        h = self.input_proj(x)  # (B, D, d_hidden)

        # Apply GAT layers
        for gat_heads, norm in zip(self.gat_layers, self.layer_norms):
            # Multi-head attention
            head_outputs = [head(h) for head in gat_heads]
            h_new = torch.cat(head_outputs, dim=-1)  # (B, D, d_hidden)
            h = norm(h + h_new)  # Residual + LayerNorm

        # Output projection
        output = self.output_proj(h)  # (B, D, T)

        # Return raw predictions - let caller handle masking
        return output

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per (d,t) position across batch."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)
        position_mean = obs_sum / (obs_count + 1e-10)
        return X_obs * mask + (1 - mask) * position_mean


# =============================================================================
# SIMPLE TRANSFORMER BASELINE (without factorization)
# =============================================================================

class SimpleTransformer(nn.Module):
    """
    Standard Transformer without factorized spatial/temporal attention.
    Flattens D×T into a single sequence.
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_features = d_features
        self.d_time = d_time
        self.seq_len = d_features * d_time

        # Input embedding
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=self.seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        B, D, T = X_obs.shape

        # Mean imputation
        X_mean = self._mean_impute(X_obs, mask)

        # Stack and flatten: (B, D*T, 2)
        x = torch.stack([X_mean, mask], dim=-1).view(B, D * T, 2)

        # Project and add positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Output projection
        h = self.output_norm(h)
        output = self.output_proj(h).squeeze(-1).view(B, D, T)

        # Return raw predictions - let caller handle masking
        return output

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per (d,t) position across batch."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)
        position_mean = obs_sum / (obs_count + 1e-10)
        return X_obs * mask + (1 - mask) * position_mean


# =============================================================================
# MLP IMPUTER
# =============================================================================

class MLPImputer(nn.Module):
    """
    Simple MLP-based imputation model.

    Processes each time step independently with a fully connected network.
    Input is concatenated [observed_data, mask] for each time step.

    Parameters:
    -----------
    d_features : int
        Number of features D
    d_time : int
        Number of time steps T (not used, for interface consistency)
    d_hidden : int
        Hidden dimension size
    n_layers : int
        Number of hidden layers
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_features = d_features
        self.d_time = d_time
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        # Input: [data, mask] concatenated = 2*D features per time step
        # Output: D features per time step

        layers = []

        # Input layer
        layers.append(nn.Linear(2 * d_features, d_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(d_hidden, d_features))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        X_obs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
            Observed data (with mean imputation for missing values)
        mask : torch.Tensor, shape (B, D, T)
            Binary mask (1=observed, 0=missing)

        Returns:
        --------
        output : torch.Tensor, shape (B, D, T)
            Raw predictions for all entries
        """
        B, D, T = X_obs.shape

        # Mean imputation for initial values
        X_filled = self._mean_impute(X_obs, mask)

        # Transpose to (B, T, D) for processing each time step
        X_filled = X_filled.permute(0, 2, 1)  # (B, T, D)
        mask_t = mask.permute(0, 2, 1)  # (B, T, D)

        # Concatenate data and mask: (B, T, 2*D)
        x = torch.cat([X_filled, mask_t], dim=-1)

        # Reshape to (B*T, 2*D) for MLP processing
        x = x.reshape(B * T, 2 * D)

        # Process through MLP: (B*T, 2*D) -> (B*T, D)
        output = self.mlp(x)

        # Reshape back to (B, T, D) and transpose to (B, D, T)
        output = output.reshape(B, T, D).permute(0, 2, 1)

        # Return raw predictions - let caller handle masking
        return output

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per (d,t) position across batch."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)
        position_mean = obs_sum / (obs_count + 1e-10)
        return X_obs * mask + (1 - mask) * position_mean


# =============================================================================
# CSDI-STYLE TRANSFORMER
# =============================================================================

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """1D convolution with Kaiming initialization (from CSDI)."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CSDIResidualBlock(nn.Module):
    """
    CSDI-style residual block with factorized time/feature attention and gated activation.

    Simplified from CSDI's diffusion model to work with standard MSE training.
    """

    def __init__(
        self,
        channels: int,
        nheads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.channels = channels

        # Projections (using Conv1d like CSDI)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # Time and feature transformers (like CSDI)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nheads,
            dim_feedforward=64,  # CSDI uses small feedforward
            activation="gelu",
            dropout=dropout,
            batch_first=False
        )
        self.time_layer = nn.TransformerEncoder(time_encoder_layer, num_layers=1)

        feature_encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nheads,
            dim_feedforward=64,
            activation="gelu",
            dropout=dropout,
            batch_first=False
        )
        self.feature_layer = nn.TransformerEncoder(feature_encoder_layer, num_layers=1)

    def forward_time(self, y, base_shape):
        """Apply time-wise attention (from CSDI)."""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # Standard transformer format: (L, B, E)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """Apply feature-wise attention (from CSDI)."""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # Standard transformer format: (K, B, E)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (B, channels, K, L)
            where K=features, L=time

        Returns:
        --------
        output : torch.Tensor, shape (B, channels, K, L)
        skip : torch.Tensor, shape (B, channels, K, L)
        """
        B, channel, K, L = x.shape
        base_shape = x.shape
        x_flat = x.reshape(B, channel, K * L)

        # Apply time and feature transformers
        y = self.forward_time(x_flat, base_shape)
        y = self.forward_feature(y, base_shape)

        # Projection
        y = self.mid_projection(y)  # (B, 2*channel, K*L)

        # Gated activation (like CSDI)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, channel, K*L)

        # Output projection and residual
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        # Residual connection with normalization (like CSDI)
        output = (x_flat + residual) / math.sqrt(2.0)
        return output.reshape(base_shape), skip.reshape(base_shape)


class CSDIStyleTransformer(nn.Module):
    """
    CSDI-style transformer for imputation.

    Uses CSDI's architecture with:
    - Factorized time/feature attention
    - Residual blocks with gated activation
    - Skip connections
    """

    def __init__(
        self,
        d_features: int,
        d_time: int,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_features = d_features
        self.d_time = d_time
        self.channels = d_model

        # Input projection (like CSDI)
        self.input_projection = Conv1d_with_init(2, self.channels, 1)  # [data, mask] -> channels

        # Residual blocks (like CSDI)
        self.residual_layers = nn.ModuleList([
            CSDIResidualBlock(channels=self.channels, nheads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection (like CSDI)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

    def forward(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        X_obs : torch.Tensor, shape (B, D, T)
        mask : torch.Tensor, shape (B, D, T)

        Returns:
        --------
        output : torch.Tensor, shape (B, D, T)
        """
        B, D, T = X_obs.shape

        # Mean imputation
        X_mean = self._mean_impute(X_obs, mask)

        # Stack [data, mask] and flatten: (B, 2, D*T)
        x = torch.stack([X_mean, mask], dim=1).reshape(B, 2, D * T)

        # Input projection
        x = self.input_projection(x)  # (B, channels, D*T)
        x = F.relu(x)
        x = x.reshape(B, self.channels, D, T)  # (B, channels, K=D, L=T)

        # Apply residual blocks and collect skip connections (like CSDI)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip.append(skip_connection)

        # Aggregate skip connections (like CSDI)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # Output projection
        x = x.reshape(B, self.channels, D * T)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B, 1, D*T)
        x = x.reshape(B, D, T)

        return x

    def _mean_impute(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per (d,t) position across batch."""
        obs_sum = (X_obs * mask).sum(dim=0, keepdim=True)  # (1, D, T)
        obs_count = mask.sum(dim=0, keepdim=True)  # (1, D, T)
        position_mean = obs_sum / (obs_count + 1e-10)
        return X_obs * mask + (1 - mask) * position_mean


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_imputer(
    model_type: str,
    d_features: int,
    d_time: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create imputer models.

    Parameters:
    -----------
    model_type : str
        One of: 'spatiotemporal_transformer', 'lstm', 'gat', 'transformer'
    d_features : int
        Number of features D
    d_time : int
        Number of time steps T
    **kwargs
        Additional model-specific arguments

    Returns:
    --------
    model : nn.Module
        Imputer model
    """
    models = {
        'spatiotemporal_transformer': SpatiotemporalTransformer,
        'lstm': LSTMImputer,
        'gat': GATImputer,
        'transformer': SimpleTransformer,
        'mlp': MLPImputer
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(models.keys())}")

    return models[model_type](d_features, d_time, **kwargs)
