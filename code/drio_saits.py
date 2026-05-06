"""
drio_saits_minimal.py — Minimal single-file reproducible implementation of DRIO-SAITS.

DRIO trains an imputer with a distributionally-robust objective:

    L = α · R_θ(X)  +  (1 - α) · S_{ε,τ}(Z*, X̂)

where R_θ is the SAITS reconstruction loss (joint ORT + MIT, L1), S_{ε,τ} is the
unbalanced Sinkhorn divergence with entropic regularization ε and marginal
relaxation τ, and Z* is the worst-case adversary obtained by K steps of
gradient ascent on the inner objective S_{ε,τ}(Z, X̂) − γ · ||Z − X̄||².

The two terms in the outer loss are EMA-normalized so that α truly controls the
balance between point-wise reconstruction and distributional alignment.

Dependencies: torch, numpy, geomloss
    pip install torch numpy geomloss tqdm

Data format (one .pkl per split: train / val / test):
    {
        'observed_values': float32 ndarray (N, T, D) -- raw values
        'observed_mask':   float32 ndarray (N, T, D) -- 1 where ground truth exists
        'gt_mask':         float32 ndarray (N, T, D) -- 1 where the entry is fed
                                                         to the model at training
                                                         time (a subset of
                                                         observed_mask; the
                                                         complement on
                                                         observed_mask is the
                                                         held-out evaluation set)
        'metadata': {
            'feature_means': ndarray (D,),
            'feature_stds':  ndarray (D,),
            ...
        }
    }

Usage:
    python drio_saits_minimal.py \\
        --data_dir data/processed/cnnpred \\
        --data_prefix cnnpred_mnar_10pct_split70-10-20 \\
        --output_dir runs/drio_saits/cnnpred_mnar_10pct \\
        --alpha 0.5 --gamma 1.0 --epochs 80
"""

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from geomloss import SamplesLoss
except ImportError as e:
    raise SystemExit("geomloss is required: pip install geomloss") from e


# =============================================================================
# SAITS BACKBONE
#   Du, W., Cote, D., Liu, Y. "SAITS: Self-Attention-based Imputation for Time
#   Series." Expert Systems with Applications 219, 119619 (2023).
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, n_position: int = 1024):
        super().__init__()
        position = np.arange(n_position)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((n_position, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.register_buffer("pe", torch.from_numpy(pe).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with optional pairwise mask."""

    def __init__(self, d_model: int, n_head: int, d_k: int, d_v: int, dropout: float):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.scale = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        B, L, _ = x.shape
        q = self.w_q(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_head, self.d_v).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        return self.fc(out), attn


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with diagonal-masked self-attention."""

    def __init__(self, d_time: int, d_model: int, d_inner: int,
                 n_head: int, d_k: int, d_v: int, dropout: float):
        super().__init__()
        self.d_time = d_time
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Diagonal mask: prevent each token from attending to itself,
        # forcing the imputer to reconstruct each (d, t) entry from context.
        diag = torch.eye(self.d_time, dtype=torch.bool, device=x.device)
        attn_mask = diag.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        h, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(h)
        h2 = self.fc2(F.relu(self.fc1(self.norm2(x))))
        return x + self.dropout(h2), attn


class SAITS(nn.Module):
    """
    Two-block SAITS imputer with attention-weighted combination.

    Forward:
        X    : (B, T, D)  data with arbitrary fill at masked positions
        mask : (B, T, D)  binary mask, 1 = observed (visible to the model)

    Returns:
        X_completed : (B, T, D)  observed entries restored, masked entries imputed
        (X_tilde_1, X_tilde_2, X_tilde_3) : intermediates used in the SAITS loss
    """

    def __init__(self, d_time: int, d_feature: int,
                 n_layers: int = 5, d_model: int = 256, d_inner: int = 512,
                 n_head: int = 8, d_k: int = 32, d_v: int = 32, dropout: float = 0.0):
        super().__init__()
        actual_d = 2 * d_feature  # data + mask channels

        self.embed1 = nn.Linear(actual_d, d_model)
        self.embed2 = nn.Linear(actual_d, d_model)
        self.pos = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(dropout)

        self.block1 = nn.ModuleList([
            EncoderLayer(d_time, d_model, d_inner, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.block2 = nn.ModuleList([
            EncoderLayer(d_time, d_model, d_inner, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])

        self.reduce_z = nn.Linear(d_model, d_feature)
        self.reduce_beta = nn.Linear(d_model, d_feature)
        self.reduce_gamma = nn.Linear(d_feature, d_feature)
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        # ---- Block 1: coarse imputation ----
        h = self.dropout(self.pos(self.embed1(torch.cat([X, mask], dim=-1))))
        for layer in self.block1:
            h, _ = layer(h)
        X_tilde_1 = self.reduce_z(h)
        X_prime = mask * X + (1 - mask) * X_tilde_1  # observed entries restored

        # ---- Block 2: refinement ----
        h2 = self.pos(self.embed2(torch.cat([X_prime, mask], dim=-1)))
        attn_w = None
        for layer in self.block2:
            h2, attn_w = layer(h2)
        X_tilde_2 = self.reduce_gamma(F.relu(self.reduce_beta(h2)))

        # ---- Combine via attention-weighted gate ----
        # attn_w : (B, n_head, T, T) -> head-mean -> (B, T, T)
        attn_avg = attn_w.mean(dim=1)
        eta = torch.sigmoid(self.weight_combine(torch.cat([mask, attn_avg], dim=-1)))
        X_tilde_3 = (1 - eta) * X_tilde_2 + eta * X_tilde_1
        X_completed = mask * X + (1 - mask) * X_tilde_3
        return X_completed, (X_tilde_1, X_tilde_2, X_tilde_3)


# =============================================================================
# DRIO LOSS
# =============================================================================

def saits_recon_loss(X: torch.Tensor, mask: torch.Tensor, indicating_mask: torch.Tensor,
                     X1: torch.Tensor, X2: torch.Tensor, X3: torch.Tensor) -> torch.Tensor:
    """SAITS joint loss: average L1 over the three intermediates on observed entries
    (ORT) plus L1 on the artificially-held-out entries via the third estimate (MIT)."""
    def masked_mae(p, t, m):
        return (torch.abs(p - t) * m).sum() / (m.sum().clamp_min(1e-9))
    ort = (masked_mae(X1, X, mask) + masked_mae(X2, X, mask) + masked_mae(X3, X, mask)) / 3.0
    mit = masked_mae(X3, X, indicating_mask)
    return ort + mit


def make_sinkhorn(epsilon: float = 0.1, tau: float = 10.0) -> SamplesLoss:
    """Unbalanced Sinkhorn divergence (squared Euclidean, debiased)."""
    return SamplesLoss(
        loss="sinkhorn", p=2,
        blur=epsilon ** 0.5,    # geomloss: blur = epsilon^(1/p)
        reach=tau ** 0.5,       # geomloss: reach = sqrt(tau) for the L2 marginal
        scaling=0.9, debias=True,
        backend="tensorized",
    )


def flatten_traj(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, T*D) so each entire trajectory is one point in R^{T*D}.
    The batch becomes the point cloud Sinkhorn compares against another batch."""
    return x.reshape(x.size(0), -1) if x.dim() > 2 else x


def pick_epsilon(x: torch.Tensor, y: torch.Tensor,
                 quant: float = 0.5, mult: float = 0.05,
                 max_points: int = 500) -> float:
    """Adaptive entropic-reg heuristic from MissingDataOT (Muzellec et al., 2020):
    epsilon = mult * quantile_q(pairwise squared L2 / 2) over the union of x and y,
    each row treated as one point. Inputs must already be 2D (B, *)."""
    combined = torch.cat([x, y], dim=0)
    n = combined.shape[0]
    if n > max_points:
        idx = torch.randperm(n, device=combined.device)[:max_points]
        combined = combined[idx]
    d2 = ((combined[:, None] - combined) ** 2).sum(dim=2) / 2.0
    d2 = d2.flatten()
    d2 = d2[d2 > 0]
    if d2.numel() == 0:
        return 0.1
    return max(torch.quantile(d2, quant).item() * mult, 1e-4)


def inner_maximization(Z_init: torch.Tensor, X_imputed_det: torch.Tensor,
                       X_mean: torch.Tensor, sinkhorn_fn, gamma: float,
                       K: int, lr: float, z_min: float, z_max: float):
    """K steps of projected gradient ascent on
        F(Z) = S_{eps,tau}(Z, X_imputed_det) - gamma * ||Z - X_mean||_2^2
    Returns the final adversary Z* (detached). X_imputed_det is the model's
    current imputation, frozen during the inner step."""
    Z = Z_init.detach().clone().requires_grad_(True)
    X_imp_flat = flatten_traj(X_imputed_det)
    for _ in range(K):
        s = sinkhorn_fn(flatten_traj(Z), X_imp_flat)  # scalar (B treated as point cloud)
        cost = ((Z - X_mean) ** 2).mean()
        obj = s - gamma * cost
        grad = torch.autograd.grad(obj, Z)[0]
        with torch.no_grad():
            Z = (Z + lr * grad).clamp(min=z_min, max=z_max)
        Z = Z.detach().requires_grad_(True)
    return Z.detach()


# =============================================================================
# DATA
# =============================================================================

def load_split(data_dir: str, prefix: str, split: str, seed: int):
    p = Path(data_dir) / f"{prefix}_{split}_seed{seed}.pkl"
    with open(p, "rb") as f:
        d = pickle.load(f)
    X = d["observed_values"].astype(np.float32)
    obs = d["observed_mask"].astype(np.float32)
    gt = d["gt_mask"].astype(np.float32)
    fmean = np.asarray(d["metadata"]["feature_means"], dtype=np.float32)
    fstd = np.asarray(d["metadata"]["feature_stds"], dtype=np.float32)
    return X, obs, gt, fmean, fstd


def normalize_per_feature(X: np.ndarray, fmean: np.ndarray, fstd: np.ndarray) -> np.ndarray:
    return (X - fmean[None, None, :]) / (fstd[None, None, :] + 1e-8)


def per_position_mean(X_norm: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Mean of observed entries at each (t, d) over the training set."""
    obs_sum = (X_norm * gt_mask).sum(axis=0, keepdims=True)
    obs_cnt = gt_mask.sum(axis=0, keepdims=True)
    return obs_sum / np.maximum(obs_cnt, 1.0)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_mse(model: SAITS, X: np.ndarray, gt_mask: np.ndarray, obs_mask: np.ndarray,
                 pos_mean_t: torch.Tensor, batch_size: int, device: torch.device,
                 also_mae: bool = False):
    """MSE / MAE on artificially held-out entries (observed_mask=1 AND gt_mask=0)."""
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    gt_t = torch.from_numpy(gt_mask).to(device)
    obs_t = torch.from_numpy(obs_mask).to(device)
    se = ae = 0.0
    cnt = 0
    for i in range(0, Xt.size(0), batch_size):
        Xb = Xt[i:i + batch_size]
        gb = gt_t[i:i + batch_size]
        ob = obs_t[i:i + batch_size]
        mean_b = pos_mean_t.expand(Xb.size(0), -1, -1)
        # Feed observed entries; fill held-out / missing with the per-position mean
        X_in = Xb * gb + mean_b * (1 - gb)
        X_complete, _ = model(X_in, gb)
        eval_mask = ob * (1 - gb)
        diff = (X_complete - Xb) * eval_mask
        se += (diff ** 2).sum().item()
        ae += diff.abs().sum().item()
        cnt += eval_mask.sum().item()
    cnt = max(cnt, 1)
    if also_mae:
        return se / cnt, ae / cnt
    return se / cnt


# =============================================================================
# TRAIN
# =============================================================================

def train_drio_saits(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Data
    X_tr, obs_tr, gt_tr, fmean, fstd = load_split(args.data_dir, args.data_prefix, "train", args.seed)
    X_va, obs_va, gt_va, _, _ = load_split(args.data_dir, args.data_prefix, "val", args.seed)
    X_te, obs_te, gt_te, _, _ = load_split(args.data_dir, args.data_prefix, "test", args.seed)
    X_tr = normalize_per_feature(X_tr, fmean, fstd)
    X_va = normalize_per_feature(X_va, fmean, fstd)
    X_te = normalize_per_feature(X_te, fmean, fstd)
    N, T, D = X_tr.shape
    print(f"[data] N_train={N}, T={T}, D={D}, "
          f"N_val={X_va.shape[0]}, N_test={X_te.shape[0]}")

    pos_mean_np = per_position_mean(X_tr, gt_tr)              # (1, T, D)
    pos_mean_t = torch.from_numpy(pos_mean_np).to(device)
    z_min, z_max = float(X_tr.min()), float(X_tr.max())

    # Model + optimizer
    model = SAITS(
        d_time=T, d_feature=D,
        n_layers=args.n_layers, d_model=args.d_model, d_inner=args.d_inner,
        n_head=args.n_head, d_k=args.d_k, d_v=args.d_v, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fixed_sinkhorn_fn = make_sinkhorn(epsilon=args.epsilon, tau=args.tau)

    # Training loader (gt_mask is the input mask; MIT holdout drawn fresh per batch)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(gt_tr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    ema_recon = ema_sink = None
    decay = 0.99
    best_val = math.inf
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        ep_total = ep_recon = ep_sink = 0.0
        n_batches = 0
        for X_batch, gt_batch in train_loader:
            X_batch = X_batch.to(device)
            gt_batch = gt_batch.to(device)
            B = X_batch.size(0)
            mean_b = pos_mean_t.expand(B, -1, -1)

            # SAITS-style MIT: artificially mask 20% of currently-observed entries
            indicating_mask = (torch.rand_like(gt_batch) < args.mit_rate) * gt_batch
            input_mask = gt_batch * (1 - indicating_mask)
            X_input = X_batch * input_mask + mean_b * (1 - input_mask)

            X_completed, (X1, X2, X3) = model(X_input, input_mask)

            # Reconstruction loss (point-wise, MIT + ORT)
            recon = saits_recon_loss(X_batch, input_mask, indicating_mask, X1, X2, X3)

            # Build the Sinkhorn divergence used in this batch.
            # Adaptive epsilon (MissingDataOT heuristic) recomputed per batch
            # from the current mean_b vs. the imputer's output (both flattened).
            if args.adaptive_epsilon:
                eps_b = pick_epsilon(
                    flatten_traj(mean_b.detach()), flatten_traj(X_completed.detach()),
                    quant=args.epsilon_quant, mult=args.epsilon_mult,
                )
                sinkhorn_fn = make_sinkhorn(epsilon=eps_b, tau=args.tau)
            else:
                sinkhorn_fn = fixed_sinkhorn_fn

            # Inner maximization for the worst-case adversary Z*
            Z_init = mean_b.clone()
            Z_star = inner_maximization(
                Z_init, X_completed.detach(), mean_b, sinkhorn_fn,
                gamma=args.gamma, K=args.K, lr=args.inner_lr,
                z_min=z_min, z_max=z_max,
            )
            sink = sinkhorn_fn(flatten_traj(Z_star), flatten_traj(X_completed))

            # EMA-normalize each term so alpha controls the true balance
            with torch.no_grad():
                ema_recon = recon.item() if ema_recon is None else decay * ema_recon + (1 - decay) * recon.item()
                ema_sink = max(sink.item(), 1e-6) if ema_sink is None \
                    else decay * ema_sink + (1 - decay) * max(sink.item(), 1e-6)

            total = (
                args.alpha * (recon / max(ema_recon, 1e-6))
                + (1.0 - args.alpha) * (sink / max(ema_sink, 1e-6))
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            optimizer.step()

            ep_total += total.item()
            ep_recon += recon.item()
            ep_sink += sink.item()
            n_batches += 1

        ep_total /= n_batches
        ep_recon /= n_batches
        ep_sink /= n_batches
        val_mse = evaluate_mse(model, X_va, gt_va, obs_va, pos_mean_t,
                               args.batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "loss": ep_total, "recon": ep_recon, "sink": ep_sink, "val_mse": val_mse,
        })
        print(f"[epoch {epoch + 1:3d}/{args.epochs}] "
              f"loss={ep_total:.4f}  recon={ep_recon:.4f}  sink={ep_sink:.4f}  "
              f"val_mse={val_mse:.5f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    runtime = time.time() - t0

    # Test on the best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_mae = evaluate_mse(
        model, X_te, gt_te, obs_te, pos_mean_t, args.batch_size, device, also_mae=True,
    )

    print()
    print(f"[final] best val MSE = {best_val:.5f}")
    print(f"[final] test MSE     = {test_mse:.5f}")
    print(f"[final] test MAE     = {test_mae:.5f}")
    print(f"[final] runtime      = {runtime:.1f} s")

    # Persist
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out / "model.pth")
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    results = {
        "val": {"mse": best_val},
        "test": {"mse": test_mse, "mae": test_mae},
        "runtime_seconds": runtime,
        "history": history,
    }
    with open(out / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out}")
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal DRIO-SAITS reproducible script")
    # Data / IO
    p.add_argument("--data_dir", required=True,
                   help="Directory containing the .pkl splits.")
    p.add_argument("--data_prefix", required=True,
                   help="File prefix, e.g. cnnpred_mnar_10pct_split70-10-20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_norm", type=float, default=5.0)
    # SAITS architecture
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_inner", type=int, default=512)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_k", type=int, default=32)
    p.add_argument("--d_v", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mit_rate", type=float, default=0.2,
                   help="Fraction of observed entries artificially masked each batch for the SAITS MIT loss.")
    # DRIO objective
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Reconstruction-vs-Sinkhorn weight in [0, 1].")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Transport-cost penalty in the inner maximization.")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Sinkhorn entropic regularization.")
    p.add_argument("--tau", type=float, default=10.0,
                   help="Marginal-relaxation parameter for unbalanced Sinkhorn.")
    p.add_argument("--K", type=int, default=8,
                   help="Number of inner gradient-ascent steps.")
    p.add_argument("--inner_lr", type=float, default=0.01,
                   help="Inner gradient-ascent step size.")
    p.add_argument("--adaptive_epsilon", action=argparse.BooleanOptionalAction, default=True,
                   help="Recompute epsilon per batch via the MissingDataOT heuristic. "
                        "Pass --no-adaptive_epsilon to disable.")
    p.add_argument("--epsilon_quant", type=float, default=0.5,
                   help="Quantile of pairwise squared distances for adaptive epsilon.")
    p.add_argument("--epsilon_mult", type=float, default=0.05,
                   help="Multiplier on the quantile for adaptive epsilon.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_drio_saits(args)


if __name__ == "__main__":
    main()
