import torch
import numpy as np


def imputationRMSE(model, Xorg, Xz, Xnan, S, L, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute imputation RMSE for MIWAE model.

    Args:
        model: Trained MIWAE model
        Xorg: Original complete data [N, D]
        Xz: Data with missing values filled with 0 [N, D]
        Xnan: Data with NaN for missing values [N, D]
        S: Missingness mask (1=observed, 0=missing) [N, D]
        L: Number of importance samples
        device: Device to use

    Returns:
        RMSE, Imputed matrix
    """
    model.eval()
    N = len(Xz)

    # Convert to tensors
    Xz_t = torch.tensor(Xz, dtype=torch.float32, device=device)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    XM = np.zeros_like(Xorg)

    with torch.no_grad():
        for i in range(N):
            xz = Xz_t[i:i+1]
            s = S_t[i:i+1]

            # Get importance-weighted imputation
            xm = _imp_miwae(model, xz, s, L)
            XM[i, :] = xm.cpu().numpy()

            if i % 100 == 0:
                print(f'{i} / {N}')

    # Compute RMSE only on missing values
    rmse = np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S))

    return rmse, XM


def _imp_miwae(model, xz, s, L):
    """
    Importance-weighted imputation for MIWAE.

    Args:
        model: MIWAE model
        xz: Input with missing filled with 0 [1, D]
        s: Missingness mask [1, D]
        L: Number of importance samples

    Returns:
        Imputed values for missing entries [D]
    """
    # Encode
    q_mu, q_log_sig2 = model.encode(xz, s)
    q_std = torch.sqrt(torch.exp(q_log_sig2))

    # Sample z
    q_z = torch.distributions.Normal(q_mu, q_std)
    z = q_z.rsample((L,)).permute(1, 0, 2)  # [1, L, n_latent]

    # Decode
    dec_params = model.decode(z)

    if model.out_dist in ['gauss', 'normal', 'truncated_normal']:
        mu, std = dec_params
        p_x_given_z = torch.distributions.Normal(mu, std)
        l_out_mu = mu
    elif model.out_dist == 'bern':
        logits, = dec_params
        l_out_mu = torch.sigmoid(logits)
        p_x_given_z = torch.distributions.Bernoulli(logits=logits)
    elif model.out_dist in ['t', 't-distribution']:
        mu, log_sig2, df = dec_params
        scale = torch.nn.functional.softplus(log_sig2) + 0.0001
        df_transformed = 3 + torch.nn.functional.softplus(df)
        p_x_given_z = torch.distributions.StudentT(df_transformed, mu, scale)
        l_out_mu = mu

    # Compute importance weights
    log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(xz.unsqueeze(1))).sum(dim=-1)

    q_z_expanded = torch.distributions.Normal(q_mu.unsqueeze(1), q_std.unsqueeze(1))
    log_q_z_given_x = q_z_expanded.log_prob(z).sum(dim=-1)

    prior = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    log_p_z = prior.log_prob(z).sum(dim=-1)

    # Softmax weights
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
    w = torch.softmax(log_w, dim=1)  # [1, L]

    # Weighted average
    xm = (l_out_mu * w.unsqueeze(-1)).sum(dim=1)  # [1, D]

    # Mix with observed
    x_imputed = xz * s + xm * (1 - s)

    return x_imputed.squeeze(0)


def not_imputationRMSE(model, Xorg, Xz, Xnan, S, L, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute imputation RMSE for not-MIWAE model.

    Args:
        model: Trained notMIWAE model
        Xorg: Original complete data [N, D]
        Xz: Data with missing values filled with 0 [N, D]
        Xnan: Data with NaN for missing values [N, D]
        S: Missingness mask (1=observed, 0=missing) [N, D]
        L: Number of importance samples
        device: Device to use

    Returns:
        RMSE, Imputed matrix
    """
    model.eval()
    N = len(Xz)

    # Convert to tensors
    Xz_t = torch.tensor(Xz, dtype=torch.float32, device=device)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    XM = np.zeros_like(Xorg)

    with torch.no_grad():
        for i in range(N):
            xz = Xz_t[i:i+1]
            s = S_t[i:i+1]

            # Get importance-weighted imputation
            xm = _imp_not_miwae(model, xz, s, L)
            XM[i, :] = xm.cpu().numpy()

            if i % 100 == 0:
                print(f'{i} / {N}')

    # Compute RMSE only on missing values
    rmse = np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S))

    return rmse, XM


def _imp_not_miwae(model, xz, s, L):
    """
    Importance-weighted imputation for not-MIWAE.

    Args:
        model: notMIWAE model
        xz: Input with missing filled with 0 [1, D]
        s: Missingness mask [1, D]
        L: Number of importance samples

    Returns:
        Imputed values for missing entries [D]
    """
    # Encode
    q_mu, q_log_sig2 = model.encode(xz, s)
    q_std = torch.sqrt(torch.exp(q_log_sig2))

    # Sample z
    q_z = torch.distributions.Normal(q_mu, q_std)
    z = q_z.rsample((L,)).permute(1, 0, 2)  # [1, L, n_latent]

    # Decode
    dec_params = model.decode(z)

    if model.out_dist in ['gauss', 'normal', 'truncated_normal']:
        mu, std = dec_params
        p_x_given_z = torch.distributions.Normal(mu, std)
        l_out_mu = mu
    elif model.out_dist == 'bern':
        logits, = dec_params
        l_out_mu = torch.sigmoid(logits)
        p_x_given_z = torch.distributions.Bernoulli(logits=logits)
    elif model.out_dist in ['t', 't-distribution']:
        mu, log_sig2, df = dec_params
        scale = torch.nn.functional.softplus(log_sig2) + 0.0001
        df_transformed = 3 + torch.nn.functional.softplus(df)
        p_x_given_z = torch.distributions.StudentT(df_transformed, mu, scale)
        l_out_mu = mu

    # Compute log p(x|z)
    log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(xz.unsqueeze(1))).sum(dim=-1)

    # Mix observed with sampled missing values (use mean for stability)
    l_out_mixed = l_out_mu * (1 - s).unsqueeze(1) + xz.unsqueeze(1) * s.unsqueeze(1)

    # Missing process p(s|x)
    logits_miss = model.missing_decoder(l_out_mixed)
    p_s_given_x = torch.distributions.Bernoulli(logits=logits_miss)
    log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(1)).sum(dim=-1)

    # log q(z|x) and log p(z)
    q_z_expanded = torch.distributions.Normal(q_mu.unsqueeze(1), q_std.unsqueeze(1))
    log_q_z_given_x = q_z_expanded.log_prob(z).sum(dim=-1)

    prior = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    log_p_z = prior.log_prob(z).sum(dim=-1)

    # not-MIWAE importance weights
    log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
    w = torch.softmax(log_w, dim=1)  # [1, L]

    # Weighted average
    xm = (l_out_mu * w.unsqueeze(-1)).sum(dim=1)  # [1, D]

    # Mix with observed
    x_imputed = xz * s + xm * (1 - s)

    return x_imputed.squeeze(0)


def compute_rmse(Xorg, Ximputed, S):
    """
    Compute RMSE on missing values only.

    Args:
        Xorg: Original complete data
        Ximputed: Imputed data
        S: Missingness mask (1=observed, 0=missing)

    Returns:
        RMSE value
    """
    return np.sqrt(np.sum((Xorg - Ximputed) ** 2 * (1 - S)) / np.sum(1 - S))


def batch_imputation(model, Xz, S, L, batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Batch imputation for efficiency on larger datasets.

    Args:
        model: MIWAE or notMIWAE model
        Xz: Data with missing values filled with 0 [N, D]
        S: Missingness mask [N, D]
        L: Number of importance samples
        batch_size: Batch size for imputation
        device: Device to use

    Returns:
        Imputed data [N, D]
    """
    model.eval()
    N = len(Xz)

    Xz_t = torch.tensor(Xz, dtype=torch.float32, device=device)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    X_imputed = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xz_batch = Xz_t[i:i+batch_size]
            s_batch = S_t[i:i+batch_size]

            x_imp = model.impute(xz_batch, s_batch, n_samples=L)
            X_imputed.append(x_imp.cpu().numpy())

            if i % 500 == 0:
                print(f'{i} / {N}')

    return np.vstack(X_imputed)
