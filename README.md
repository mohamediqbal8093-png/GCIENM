# GCIENM Reproducibility Package

Reproducibility code for the manuscript **“Multivariate Time Series Data Analysis Using a Generative Convolutional Information Encoding Network Model.”**

This repository implements the manuscript’s proposed **Generative Convolutional Information Encoding Network Model (GCIENM)** for multivariate wind-power forecasting. The implementation is intentionally organized as a **flat repository**: all files are placed in the repository root and no source-code folders are required.

## 1. Scope

The code implements:

- leakage-aware multivariate time-series preprocessing;
- interpolation, normalization, chronological data splitting, and walk-forward cross-validation;
- positional information encoding;
- dilated causal 1-D convolution;
- an Attention-based Transformer Module (ATM);
- the manuscript’s nonlinear tanh-based attention formulation;
- standard scaled dot-product attention as an ablation/reference option;
- Multivariate Point-wise Neural Network (MVPNN) blocks using 1 × 1 convolutions;
- GCIENM encoder-decoder forecasting;
- Improved Differential Evolution (IDE) hyperparameter search;
- fair baseline models: CNN, TCN, RNN, GRU, LSTM, and Transformer;
- MAE, RMSE, MAPE, sMAPE, and normalized RMSE;
- repeated runs with fixed random seeds;
- mean, standard deviation, bootstrap confidence intervals, Friedman tests, paired Wilcoxon tests, Holm correction, and rank-biserial effect size;
- parameter sensitivity and architectural ablation experiments;
- automated reproduction through a single command.

## 2. Important dataset note

The supplied manuscript describes the experimental data as historical SCADA wind-turbine measurements combined with meteorological observations, but it does not provide an unambiguous public dataset identifier, exact downloadable URL, complete observation period, definitive sampling interval, or exact sample count.

For scientific integrity, this repository **does not fabricate those missing details**.

Place the exact dataset used by the study at the path configured in `config.yaml`. The default expected columns are:

- `timestamp`
- `wind_speed`
- `wind_direction`
- `temperature`
- `humidity`
- `power`

Additional numeric SCADA/meteorological variables can be included by editing `feature_columns` in `config.yaml`.

See `data_schema.md` for the complete input contract.

## 3. Environment

Recommended:

- Python 3.10 or later
- Linux, Windows, or macOS
- CUDA-capable GPU optional

Install dependencies:

```bash
python -m venv .venv
```

Activate the environment and run:

```bash
pip install -r requirements.txt
```

## 4. Configure the experiment

Edit:

```text
config.yaml
```

At minimum, set:

```yaml
data:
  csv_path: "wind_scada.csv"
```

and verify the feature/target column names.

The configuration controls:

- time-series look-back;
- forecasting horizon;
- train/validation/test fractions;
- number of chronological folds;
- convolution filters;
- kernel size;
- dilation rates;
- attention heads;
- latent dimension;
- batch size;
- learning rate;
- epochs;
- early stopping;
- random seeds;
- IDE search space;
- sensitivity-analysis values.

## 5. Fair comparison protocol

All models are evaluated using the **same**:

1. cleaned input table;
2. chronological folds;
3. look-back window;
4. forecasting horizon;
5. normalization fitted only on training data;
6. training/validation/test indices;
7. random seeds;
8. maximum epoch budget;
9. early-stopping rule;
10. target variable;
11. error metrics.

This prevents the unfair situation in which different models are evaluated at different forecasting horizons.

Implemented baselines:

```text
CNN
TCN
RNN
GRU
LSTM
Transformer
GCIENM
```

## 6. Single-command reproduction

After configuring the real dataset:

```bash
python reproduce.py --config config.yaml
```

Optional stages:

```bash
python reproduce.py --config config.yaml --skip-ide
python reproduce.py --config config.yaml --skip-sensitivity
python reproduce.py --config config.yaml --skip-statistics
```

The pipeline performs:

```text
load data
→ validate schema
→ clean/interpolate
→ construct chronological folds
→ fit preprocessing on training fold only
→ train all baselines
→ train GCIENM
→ evaluate common forecasting horizon
→ aggregate repeated runs
→ statistical tests
→ sensitivity/ablation analysis
```

## 7. Individual commands

### Train GCIENM

```bash
python train.py --config config.yaml --model gcienm
```

### Train a baseline

```bash
python train.py --config config.yaml --model tcn
```

Valid model names:

```text
gcienm, cnn, tcn, rnn, gru, lstm, transformer
```

### Evaluate saved predictions

```bash
python evaluate.py --predictions predictions.csv
```

### IDE search

```bash
python ide_optimizer.py --config config.yaml
```

### Sensitivity/ablation study

```bash
python sensitivity.py --config config.yaml
```

### Statistical analysis

```bash
python statistics.py --results repeated_results.csv
```

### Architecture/dimension checks

```bash
python test_shapes.py
```

## 8. GCIENM data flow

For an input tensor:

```text
X ∈ R^(B × L × F)
```

where:

- `B` = batch size,
- `L` = input sequence length,
- `F` = number of multivariate features,

the implementation uses the following pipeline:

```text
multivariate sequence
→ input projection
→ positional encoding
→ dilated causal convolution
→ nonlinear/standard multi-head attention
→ MVPNN 1×1 feature mixing
→ residual connection + layer normalization
→ encoder representation
→ decoder/query tokens
→ cross/self attention
→ point-wise feature refinement
→ horizon-wise power prediction
```

Attention tensors are represented as:

```text
Q     : [B, H, Lq, Dk]
K     : [B, H, Lk, Dk]
V     : [B, H, Lk, Dv]
Score : [B, H, Lq, Lk]
C     : [B, H, Lq, Dv]
```

where `H` is the number of heads.

## 9. Attention formulations

### Standard scaled dot-product attention

```text
Score(Q,K) = QK^T / sqrt(Dk)
A = softmax(Score)
C = AV
```

### Manuscript-motivated nonlinear attention

The repository also implements a bounded nonlinear score:

```text
S_ij = tanh(w_q^T q_i + w_k^T k_j + b)
A_ij = softmax(S_ij)
C_i  = Σ_j A_ij v_j
```

This formulation preserves the manuscript’s stated nonlinear/tanh attention concept while retaining a fully defined tensor implementation.

The `attention_mode` parameter can be set to:

```yaml
attention_mode: "nonlinear"
```

or:

```yaml
attention_mode: "scaled_dot"
```

This enables a direct ablation requested by the reviewer.

## 10. IDE optimization

The IDE implementation optimizes a configurable subset of:

- look-back;
- hidden dimension;
- number of convolution filters;
- kernel size;
- dilation;
- number of attention heads;
- dropout;
- learning rate.

The fitness function is validation MSE.

The implementation contains:

- population initialization;
- DE/best/1 mutation;
- generation-dependent mutation scaling;
- logistic chaotic crossover control;
- binomial crossover;
- boundary repair;
- greedy selection;
- best-individual tracking.

The search is reproducible under the configured seed.

## 11. Statistical validation

Repeated results are stored with one row per:

```text
model × seed × fold
```

`statistics.py` reports:

- mean;
- standard deviation;
- median;
- bootstrap 95% confidence interval;
- average rank;
- Friedman omnibus test;
- pairwise Wilcoxon signed-rank tests against GCIENM;
- Holm-adjusted p-values;
- rank-biserial effect sizes.

A statistical claim should only be made when the experimental results generated from the real study data support it.

## 12. Sensitivity and ablation experiments

The repository can vary:

- kernel size;
- dilation;
- look-back;
- hidden dimension;
- selected feature subsets;
- attention formulation.

Architectural ablations include:

- no ATM;
- no MVPNN;
- no dilated causal block;
- standard attention instead of nonlinear attention.

These experiments are intended to show which GCIENM components are responsible for performance changes.

## 13. Output files

Running experiments creates files in the working directory, for example:

```text
repeated_results.csv
predictions_<model>_fold<k>_seed<s>.csv
best_model_<model>_fold<k>_seed<s>.pt
statistics_summary.csv
pairwise_statistics.csv
sensitivity_results.csv
ide_best.json
```

These are generated artifacts and are excluded by `.gitignore`.

## 14. Reproducibility checklist

Before archiving a release:

- configure the exact real dataset;
- verify all column names;
- document the true sampling interval;
- document the exact observation period;
- verify the real forecasting horizon;
- run `python test_shapes.py`;
- run `python reproduce.py --config config.yaml`;
- retain the generated repeated-run results;
- verify that all reported manuscript values can be traced to generated result files;
- create a GitHub release;
- archive that exact release in a DOI-assigning repository such as Zenodo;
- insert the resulting code DOI into the manuscript’s Code Availability section.

## 15. Code availability wording after archival

After a DOI has actually been assigned, a manuscript statement can use the following structure:

> The source code used to implement and evaluate the GCIENM forecasting framework, including preprocessing, baseline models, IDE optimization, sensitivity analysis, and statistical validation, is publicly available from the associated GitHub repository. A versioned archival snapshot of the code has been deposited in Zenodo and is accessible through the DOI reported in the final manuscript.

Do not add a DOI until the archival record exists.

## 16. Files

```text
README.md
requirements.txt
config.yaml
data_schema.md
preprocess.py
gcienm.py
ide_optimizer.py
baselines.py
train.py
evaluate.py
sensitivity.py
statistics.py
reproduce.py
test_shapes.py
.gitignore
```
