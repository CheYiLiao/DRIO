"""
PyTorch implementation: Use the MIWAE and not-MIWAE on UCI data

This script replicates the original TensorFlow task01.py using PyTorch.
"""
import numpy as np
import pandas as pd
import os
import torch

from MIWAE import MIWAE
from notMIWAE import notMIWAE
import trainer
import utils

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def introduce_missing(X):
    """
    Introduce MNAR missingness: values above the mean are set to NaN
    in the first D/2 dimensions.
    """
    N, D = X.shape
    Xnan = X.copy()

    # MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    # Create zero-filled version
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz


def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data settings
    save_dir = '/tmp/uci/task01/torch'
    os.makedirs(save_dir, exist_ok=True)

    n_hidden = 128
    n_samples = 20
    max_iter = 100000
    batch_size = 16
    L = 10000  # Importance samples for imputation

    # Choose the missing model
    mprocess = 'selfmasking_known'  # Options: 'linear', 'selfmasking', 'selfmasking_known', 'nonlinear'

    # Number of runs
    runs = 1
    RMSE_miwae = []
    RMSE_notmiwae = []
    RMSE_mean = []
    RMSE_mice = []
    RMSE_RF = []

    for run in range(runs):
        print(f"\n=== Run {run + 1}/{runs} ===\n")

        # Load data (UCI White Wine)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
        # Drop the classification attribute
        data = data[:, :-1]

        N, D = data.shape
        dl = D - 1  # Latent dimension

        # Standardize data
        data = data - np.mean(data, axis=0)
        data = data / np.std(data, axis=0)

        # Random permutation
        p = np.random.permutation(N)
        data = data[p, :]

        Xtrain = data.copy()
        Xval_org = data.copy()

        # Introduce missing process
        Xnan, Xz = introduce_missing(Xtrain)
        S = np.array(~np.isnan(Xnan), dtype=np.float32)
        Xval, Xvalz = introduce_missing(Xval_org)
        Sval = np.array(~np.isnan(Xval), dtype=np.float32)

        # ------------------- #
        # ---- fit MIWAE ---- #
        # ------------------- #
        print("\n--- Training MIWAE ---")
        miwae = MIWAE(
            input_dim=D,
            n_latent=dl,
            n_samples=n_samples,
            n_hidden=n_hidden,
            device=device
        )

        trainer.train(
            miwae, Xz, S, Xvalz, Sval,
            batch_size=batch_size,
            max_iter=max_iter,
            save_path=os.path.join(save_dir, 'miwae_best.pt'),
            device=device
        )

        # Load best model
        miwae.load_state_dict(torch.load(os.path.join(save_dir, 'miwae_best.pt')))

        # Find imputation RMSE
        rmse_miwae, _ = utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L, device=device)
        RMSE_miwae.append(rmse_miwae)

        # ---------------------- #
        # ---- fit not-MIWAE---- #
        # ---------------------- #
        print("\n--- Training not-MIWAE ---")
        notmiwae = notMIWAE(
            input_dim=D,
            n_latent=dl,
            n_samples=n_samples,
            n_hidden=n_hidden,
            missing_process=mprocess,
            device=device
        )

        trainer.train(
            notmiwae, Xz, S, Xvalz, Sval,
            batch_size=batch_size,
            max_iter=max_iter,
            save_path=os.path.join(save_dir, 'notmiwae_best.pt'),
            device=device
        )

        # Load best model
        notmiwae.load_state_dict(torch.load(os.path.join(save_dir, 'notmiwae_best.pt')))

        # Find imputation RMSE
        rmse_notmiwae, _ = utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L, device=device)
        RMSE_notmiwae.append(rmse_notmiwae)

        # ------------------------- #
        # ---- mean imputation ---- #
        # ------------------------- #
        print("\n--- Mean Imputation ---")
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(Xnan)
        Xrec = imp.transform(Xnan)
        RMSE_mean.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))

        # ------------------------- #
        # ---- MICE imputation ---- #
        # ------------------------- #
        print("\n--- MICE Imputation ---")
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(Xnan)
        Xrec = imp.transform(Xnan)
        RMSE_mice.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))

        # ------------------------------- #
        # ---- missForest imputation ---- #
        # ------------------------------- #
        print("\n--- MissForest Imputation ---")
        estimator = RandomForestRegressor(n_estimators=100)
        imp = IterativeImputer(estimator=estimator)
        imp.fit(Xnan)
        Xrec = imp.transform(Xnan)
        RMSE_RF.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))

        print(f'\nRMSE, MIWAE {RMSE_miwae[-1]:.5f}, notMIWAE {RMSE_notmiwae[-1]:.5f}, '
              f'MEAN {RMSE_mean[-1]:.5f}, MICE {RMSE_mice[-1]:.5f}, missForest {RMSE_RF[-1]:.5f}')

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"RMSE_miwae = {np.mean(RMSE_miwae):.5f} +- {np.std(RMSE_miwae):.5f}")
    print(f"RMSE_notmiwae = {np.mean(RMSE_notmiwae):.5f} +- {np.std(RMSE_notmiwae):.5f}")
    print(f"RMSE_mean = {np.mean(RMSE_mean):.5f} +- {np.std(RMSE_mean):.5f}")
    print(f"RMSE_mice = {np.mean(RMSE_mice):.5f} +- {np.std(RMSE_mice):.5f}")
    print(f"RMSE_missForest = {np.mean(RMSE_RF):.5f} +- {np.std(RMSE_RF):.5f}")


if __name__ == '__main__':
    main()
