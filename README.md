# Multivariate Time Series Data Imputation via Distributionally Robust Regularization

Contains reproducible implementation of DRIO paired with the SAITS backbone, following the parameter setting from the paper *Multivariate Time Series Data Imputation via Distributionally Robust Regularization*.

## Layout

```
git_nips/
├── code/
│   └── drio_saits.py     # the entire model + training loop
├── data/
│   └── cmapss/           # one example scenario for the reproducibility check
│       ├── cmapss_mcar_10pct_split70-10-20_{train,val,test}_seed42.pkl
│       └── cmapss_mnar_10pct_split70-10-20_{train,val,test}_seed42.pkl
└── README.md
```

The included `data/cmapss/` is **one scenario** of one dataset (NASA C-MAPSS, 10% missing, MCAR and MNAR) so the script can be run out of the box for a reproducibility check. Other datasets and missing ratios used in the paper are not bundled.

## Install

```bash
pip install torch numpy geomloss
```

## Run

The two scripts below run DRIO-SAITS on the provided C-MAPSS 10% scenarios. `--alpha` and `--gamma` are the only knobs you would normally vary; everything else (architecture, optimiser, inner-maximisation steps, Sinkhorn `epsilon`/`tau`, EMA loss normalisation) is at the paper's default value.

```bash
# MCAR 10%
python code/drio_saits.py \
    --data_dir    data/cmapss \
    --data_prefix cmapss_mcar_10pct_split70-10-20 \
    --seed 42 \
    --output_dir  runs/cmapss_mcar_10pct \
    --alpha 0.5 --gamma 1.0

# MNAR 10%
python code/drio_saits.py \
    --data_dir    data/cmapss \
    --data_prefix cmapss_mnar_10pct_split70-10-20 \
    --seed 42 \
    --output_dir  runs/cmapss_mnar_10pct \
    --alpha 0.5 --gamma 1.0
```

The script writes `model.pth`, `config.json`, and `evaluation_results.json` (val/test MSE+MAE, per-epoch history, runtime) to `--output_dir`. Run `python code/drio_saits.py -h` for the full list of flags.

## Data format

Each scenario is a triple of pickle files:

```
data/<dataset>/<dataset>_<missing_type>_<missing_ratio>pct_split70-10-20_{train,val,test}_seed<seed>.pkl
```

Each pickle contains:

| key | shape | meaning |
|---|---|---|
| `observed_values` | `(N, T, D)` | raw values (any fill at masked positions) |
| `observed_mask`   | `(N, T, D)` | `1` where ground truth exists |
| `gt_mask`         | `(N, T, D)` | `1` where the entry is fed to the model at training time (a subset of `observed_mask`); held-out evaluation entries are `observed_mask=1 & gt_mask=0` |
| `metadata.feature_means` | `(D,)` | per-feature normalisation mean (computed on the train split) |
| `metadata.feature_stds`  | `(D,)` | per-feature normalisation std (computed on the train split) |

## Preparing your own data (NASA C-MAPSS as an example)

The bundled `data/cmapss/` files were produced from the NASA C-MAPSS turbofan-engine dataset, available at:

> NASA Prognostics Data Repository — C-MAPSS Jet Engine Simulated Data
> <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>

Steps to reproduce the .pkl format from the raw download:

1. **Download and unzip** `CMAPSSData.zip` so `train_FD00{1,2,3,4}.txt` are next to each other in a directory.
2. **Combine** the four sub-datasets `FD001`-`FD004` into a single table; each row is one (engine, cycle) tuple with 21 sensor columns.
3. **Filter** to engines with at least **207 cycles** (the median across the four sub-datasets), and **truncate** each kept engine to its first 207 cycles. This gives a uniform `(N, T=207, D=varying-sensors)` tensor where `D` is the subset of sensor channels that vary in the combined corpus (typically 15 channels: sensor_{2,3,4,6,7,8,9,11,12,13,14,15,17,20,21}).
4. **Split** along `N` into 70 / 10 / 20 for train / val / test.
5. **Compute** per-feature mean and std on the **training split** only; store them in `metadata.feature_means` / `metadata.feature_stds`. (The .pkl values themselves are stored on the original scale; the script handles normalisation at load time.)
6. **Generate masks**:
   - `observed_mask = 1` everywhere (C-MAPSS has no native missingness).
   - For each scenario, draw a `gt_mask` by holding out a fraction (e.g. 10%) of the entries with the chosen mechanism (MCAR = uniform Bernoulli, MNAR = held-out probability depends on the value, e.g. through a logistic with the value as input).
7. **Save** one pickle per `(dataset, mechanism, missing_ratio, split, seed)` combination, with the keys listed in the *Data format* table above.

The same recipe (download, normalise on the train split, hold out entries via MCAR/MNAR, save splits) applies to any tabular multivariate-time-series source. As long as each pickle has the listed keys and shapes, `drio_saits.py` will train and evaluate without any code changes.

## Citation

If you use this code, please cite:

```bibtex
@article{liao2026multivariate,
  title={Multivariate Time Series Data Imputation via Distributionally Robust Regularization},
  author={Liao, Che-Yi and Dong, Zheng and Garcia, Gian-Gabriel and Paynabar, Kamran},
  journal={arXiv preprint arXiv:2602.00844},
  year={2026}
}
```

## Author

Che-Yi Liao &mdash; <cyliao@umich.edu>

Issues, questions, and pull requests are welcome.
