import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np


class notMIWAE(nn.Module):
    """
    PyTorch implementation of not-MIWAE: Deep generative modelling with missing not at random data.

    Paper: https://arxiv.org/abs/2006.12871
    """

    def __init__(self, input_dim, n_latent=50, n_hidden=100, n_samples=1,
                 activation=nn.Tanh,
                 out_dist='gauss',
                 out_activation=None,
                 learnable_imputation=False,
                 permutation_invariance=False,
                 embedding_size=20,
                 code_size=20,
                 missing_process='selfmasking',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(notMIWAE, self).__init__()

        self.d = input_dim
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.out_dist = out_dist
        self.out_activation = out_activation
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.missing_process = missing_process
        self.device = device
        self.learnable_imputation = learnable_imputation
        self.permutation_invariance = permutation_invariance
        self.eps = np.finfo(float).eps

        # Learnable imputation values
        if learnable_imputation:
            self.imp = nn.Parameter(torch.zeros(1, self.d))

        # Permutation invariant embedding
        if permutation_invariance:
            self.E = nn.Parameter(torch.randn(self.d, self.embedding_size))
            self.h_layer = nn.Linear(self.embedding_size + 1, self.code_size)
            encoder_input_dim = self.code_size
        else:
            encoder_input_dim = self.d

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_hidden),
            activation(),
        )
        self.q_mu = nn.Linear(n_hidden, n_latent)
        self.q_log_sig2 = nn.Linear(n_hidden, n_latent)

        # Decoder for data process
        if out_dist in ['gauss', 'normal', 'truncated_normal']:
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
            )
            self.dec_mu = nn.Linear(n_hidden, self.d)
            self.dec_std = nn.Linear(n_hidden, self.d)

        elif out_dist == 'bern':
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
            )
            self.dec_logits = nn.Linear(n_hidden, self.d)

        elif out_dist in ['t', 't-distribution']:
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
            )
            self.dec_mu = nn.Linear(n_hidden, self.d)
            self.dec_log_sig2 = nn.Linear(n_hidden, self.d)
            self.dec_df = nn.Linear(n_hidden, self.d)

        # Missing process decoder
        if missing_process == 'selfmasking':
            self.W_miss = nn.Parameter(torch.ones(1, 1, self.d))
            self.b_miss = nn.Parameter(torch.zeros(1, 1, self.d))
        elif missing_process == 'selfmasking_known':
            self.W_miss_raw = nn.Parameter(torch.ones(1, 1, self.d))
            self.b_miss = nn.Parameter(torch.zeros(1, 1, self.d))
        elif missing_process == 'linear':
            self.miss_linear = nn.Linear(self.d, self.d)
        elif missing_process == 'nonlinear':
            self.miss_net = nn.Sequential(
                nn.Linear(self.d, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, self.d),
            )

        self.to(device)

    def permutation_invariant_embedding(self, x, s):
        """EDDI-style permutation invariant embedding."""
        # Multiply E and s to zero unobserved dimensions in E
        Es = s.unsqueeze(2) * self.E.unsqueeze(0)  # [batch, d, embedding_size]

        # Concatenate with x
        Esx = torch.cat([Es, x.unsqueeze(2)], dim=2)  # [batch, d, embedding_size + 1]

        # Reshape for shared network
        batch_size = x.shape[0]
        Esxr = Esx.view(-1, self.embedding_size + 1)  # [batch * d, embedding_size + 1]

        # Nonlinear mapping
        h = torch.relu(self.h_layer(Esxr))  # [batch * d, code_size]

        # Reshape back
        hr = h.view(batch_size, self.d, self.code_size)  # [batch, d, code_size]

        # Zero dimensions with no observations
        hz = s.unsqueeze(2) * hr  # [batch, d, code_size]

        # Permutation invariant aggregation (sum over feature dimension)
        g = hz.sum(dim=1)  # [batch, code_size]

        return g

    def encode(self, x, s):
        """Encode input to latent distribution parameters."""
        if self.learnable_imputation:
            x_in = x + (1 - s) * self.imp
        elif self.permutation_invariance:
            x_in = self.permutation_invariant_embedding(x, s)
        else:
            x_in = x

        h = self.encoder(x_in)
        mu = self.q_mu(h)
        log_sig2 = torch.clamp(self.q_log_sig2(h), -10, 10)
        return mu, log_sig2

    def decode(self, z):
        """Decode latent samples to output distribution parameters."""
        h = self.decoder(z)

        if self.out_dist in ['gauss', 'normal', 'truncated_normal']:
            mu = self.dec_mu(h)
            if self.out_activation is not None:
                mu = self.out_activation(mu)
            std = nn.functional.softplus(self.dec_std(h))
            return mu, std

        elif self.out_dist == 'bern':
            logits = self.dec_logits(h)
            return logits,

        elif self.out_dist in ['t', 't-distribution']:
            mu = self.dec_mu(h)
            if self.out_activation is not None:
                mu = self.out_activation(mu)
            log_sig2 = torch.clamp(self.dec_log_sig2(h), -10, 10)
            df = self.dec_df(h)
            return mu, log_sig2, df

    def missing_decoder(self, x_mixed):
        """Compute logits for missing process p(s|x)."""
        if self.missing_process == 'selfmasking':
            logits = -self.W_miss * (x_mixed - self.b_miss)
        elif self.missing_process == 'selfmasking_known':
            W = nn.functional.softplus(self.W_miss_raw)
            logits = -W * (x_mixed - self.b_miss)
        elif self.missing_process == 'linear':
            logits = self.miss_linear(x_mixed)
        elif self.missing_process == 'nonlinear':
            logits = self.miss_net(x_mixed)
        else:
            raise ValueError(f"Unknown missing_process: {self.missing_process}")
        return logits

    def forward(self, x, s, n_samples=None):
        """
        Forward pass computing the not-MIWAE ELBO.

        Args:
            x: Input data with missing values filled (e.g., with 0) [batch_size, d]
            s: Missingness mask (1 = observed, 0 = missing) [batch_size, d]
            n_samples: Number of importance samples (default: self.n_samples)

        Returns:
            Dictionary containing loss and various log probabilities
        """
        if n_samples is None:
            n_samples = self.n_samples

        batch_size = x.shape[0]

        # Encode
        q_mu, q_log_sig2 = self.encode(x, s)
        q_std = torch.sqrt(torch.exp(q_log_sig2))

        # Sample z using reparameterization
        q_z = dist.Normal(q_mu, q_std)
        z = q_z.rsample((n_samples,))  # [n_samples, batch_size, n_latent]
        z = z.permute(1, 0, 2)  # [batch_size, n_samples, n_latent]

        # Decode
        dec_params = self.decode(z)

        # Compute log p(x|z)
        if self.out_dist in ['gauss', 'normal']:
            mu, std = dec_params
            p_x_given_z = dist.Normal(mu, std)
            log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(x.unsqueeze(1))).sum(dim=-1)
            l_out_mu = mu
            l_out_sample = p_x_given_z.rsample()

        elif self.out_dist == 'truncated_normal':
            mu, std = dec_params
            # Approximate truncated normal with normal for simplicity
            p_x_given_z = dist.Normal(mu, std)
            log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(x.unsqueeze(1))).sum(dim=-1)
            l_out_mu = mu
            l_out_sample = torch.clamp(p_x_given_z.rsample(), 0.0, 1.0)

        elif self.out_dist == 'bern':
            logits, = dec_params
            p_x_given_z = dist.Bernoulli(logits=logits)
            log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(x.unsqueeze(1))).sum(dim=-1)
            l_out_mu = torch.sigmoid(logits)
            l_out_sample = p_x_given_z.sample()

        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sig2, df = dec_params
            scale = nn.functional.softplus(log_sig2) + 0.0001
            df_transformed = 3 + nn.functional.softplus(df)
            p_x_given_z = dist.StudentT(df_transformed, mu, scale)
            log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(x.unsqueeze(1))).sum(dim=-1)
            l_out_mu = mu
            l_out_sample = p_x_given_z.rsample()

        # Mix observed with sampled missing values
        l_out_mixed = l_out_sample * (1 - s).unsqueeze(1) + x.unsqueeze(1) * s.unsqueeze(1)

        # Missing process p(s|x)
        logits_miss = self.missing_decoder(l_out_mixed)
        p_s_given_x = dist.Bernoulli(logits=logits_miss)
        log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(1)).sum(dim=-1)

        # log q(z|x)
        q_z_expanded = dist.Normal(q_mu.unsqueeze(1), q_std.unsqueeze(1))
        log_q_z_given_x = q_z_expanded.log_prob(z).sum(dim=-1)

        # log p(z)
        prior = dist.Normal(torch.zeros_like(z), torch.ones_like(z))
        log_p_z = prior.log_prob(z).sum(dim=-1)

        # not-MIWAE ELBO
        log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
        log_sum_w = torch.logsumexp(log_w, dim=1)
        notMIWAE = (log_sum_w - np.log(n_samples)).mean()

        # MIWAE (for comparison / testing)
        log_w_miwae = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_sum_w_miwae = torch.logsumexp(log_w_miwae, dim=1)
        MIWAE = (log_sum_w_miwae - np.log(n_samples)).mean()

        return {
            'loss': -notMIWAE,
            'notMIWAE': notMIWAE,
            'MIWAE': MIWAE,
            'log_p_x_given_z': log_p_x_given_z.mean(),
            'log_p_s_given_x': log_p_s_given_x.mean(),
            'log_q_z_given_x': log_q_z_given_x.mean(),
            'log_p_z': log_p_z.mean(),
            'l_out_mu': l_out_mu,
            'l_out_sample': l_out_sample,
        }

    def impute(self, x, s, n_samples=1000):
        """
        Impute missing values using importance-weighted sampling.

        Args:
            x: Input data with missing values filled [batch_size, d]
            s: Missingness mask [batch_size, d]
            n_samples: Number of importance samples

        Returns:
            Imputed data [batch_size, d]
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]

            # Encode
            q_mu, q_log_sig2 = self.encode(x, s)
            q_std = torch.sqrt(torch.exp(q_log_sig2))

            # Sample z
            q_z = dist.Normal(q_mu, q_std)
            z = q_z.rsample((n_samples,)).permute(1, 0, 2)  # [batch, n_samples, n_latent]

            # Decode
            dec_params = self.decode(z)

            if self.out_dist in ['gauss', 'normal', 'truncated_normal']:
                mu, std = dec_params
                p_x_given_z = dist.Normal(mu, std)
                l_out_mu = mu
            elif self.out_dist == 'bern':
                logits, = dec_params
                l_out_mu = torch.sigmoid(logits)
                p_x_given_z = dist.Bernoulli(logits=logits)
            elif self.out_dist in ['t', 't-distribution']:
                mu, log_sig2, df = dec_params
                scale = nn.functional.softplus(log_sig2) + 0.0001
                df_transformed = 3 + nn.functional.softplus(df)
                p_x_given_z = dist.StudentT(df_transformed, mu, scale)
                l_out_mu = mu

            # Compute log probabilities
            log_p_x_given_z = (s.unsqueeze(1) * p_x_given_z.log_prob(x.unsqueeze(1))).sum(dim=-1)

            # Sample for missing process
            l_out_sample = l_out_mu  # Use mean for more stable imputation
            l_out_mixed = l_out_sample * (1 - s).unsqueeze(1) + x.unsqueeze(1) * s.unsqueeze(1)

            # Missing process
            logits_miss = self.missing_decoder(l_out_mixed)
            p_s_given_x = dist.Bernoulli(logits=logits_miss)
            log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(1)).sum(dim=-1)

            # log q(z|x) and log p(z)
            q_z_expanded = dist.Normal(q_mu.unsqueeze(1), q_std.unsqueeze(1))
            log_q_z_given_x = q_z_expanded.log_prob(z).sum(dim=-1)
            prior = dist.Normal(torch.zeros_like(z), torch.ones_like(z))
            log_p_z = prior.log_prob(z).sum(dim=-1)

            # Importance weights (not-MIWAE)
            log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
            w = torch.softmax(log_w, dim=1)  # [batch, n_samples]

            # Weighted average of decoder outputs
            xm = (l_out_mu * w.unsqueeze(-1)).sum(dim=1)  # [batch, d]

            # Mix with observed values
            x_imputed = x * s + xm * (1 - s)

        return x_imputed

    def get_llh_estimate(self, x, n_samples=100):
        """Estimate log-likelihood using MIWAE bound."""
        s = (~torch.isnan(x)).float()
        x_filled = x.clone()
        x_filled[torch.isnan(x_filled)] = 0

        self.eval()
        with torch.no_grad():
            result = self.forward(x_filled, s, n_samples=n_samples)
        return result['MIWAE'].item()
