# Multivariate Time Series Data Imputation via Distributionally Robust Regularization

## Data

Preprocessed datasets are available in `DRIO/data/processed/`. We provide 10 datasets with MCAR and MNAR missingness at 10%, 50%, and 90% missing rates:

- **pems04, pems08**: Traffic flow data
- **physionet**: ICU clinical time series
- **har**: Human activity recognition
- **pm25, airquality**: Environmental sensor data
- **cmapss**: Turbofan engine degradation
- **gassensor**: Gas sensor array drift
- **cnnpred**: Stock market prediction
- **gait**: Gait analysis

Data format: `{dataset}_{missing_type}_{missing_ratio}pct_split70-10-20_{split}_seed42.pkl`

## Code

### Benchmark Model Sources

The `benchmark_*` directories contain implementations based on the following repositories:

| Model | Original Repository | Notes |
|-------|---------------------|-------|
| BRITS | [caow13/BRITS](https://github.com/caow13/BRITS) | Wrapper implementation |
| CSDI | [ermongroup/CSDI](https://github.com/ermongroup/CSDI) | - |
| MissingDataOT | [BorisMuzellec/MissingDataOT](https://github.com/BorisMuzellec/MissingDataOT) | - |
| PSW | [FMLYD/PSW-I](https://github.com/FMLYD/PSW-I) | Wrapper implementation |
| not-MIWAE | [marineLM/not-MIWAE](https://github.com/marineLM/not-MIWAE) | Reimplemented in PyTorch |

Model configurations are stored in `code/config/`.

The `DRIO/code/` directory contains:
- `train_test_unified.py`: Unified wrapper for training and evaluation
- `drio.py`, `drio_v2.py`: DRIO implementation. `drio.py` implments `drio` using reconstruction-based training (without internal masking); `drio_v2.py` trains with internal masking. 
- Ablation models: `bsh_drio.py` (balanced Sinkhorn), `mse_brits` (MSE + BRITS backbone)

## Reproducing Results

Pre-computed results are available in `DRIO/results/`. Run the Jupyter notebook to generate tables and figures:

```bash
jupyter notebook DRIO/generate_results.ipynb
```

## Running Experiments

### Data Preprocessing (optional)

```bash
# Example: Process physionet with MCAR 10% missing
python code/process_data_for_imputation.py \
    --dataset physionet \
    --missing-type mcar \
    --missing-ratio 0.1 \
    --train-ratio 0.7 \
    --val-ratio 0.1 \
    --seed 42
```

See `DRIO/data.sbatch` for all configurations.

### Training Models

```bash
# Example: Run DRIO with BRITS backbone on physionet MCAR 10%
python code/train_test_unified.py \
    --data-prefix data/processed/physionet/physionet_mcar_10pct_split70-10-20 \
    --seed 42 \
    --models drio_brits
```

Available models: `mean`, `mf`, `brits`, `csdi`, `mdot`, `psw`, `notmiwae`, `drio_brits`, `bsh_drio_brits`, `mse_brits`