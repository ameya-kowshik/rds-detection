# Synthetic Data Generation Workflow — RDS Neonatal Dataset
## Using CTGAN / TVAE (SDV Library)

---

## Overview

This workflow covers generating a synthetic version of the RDS neonatal dataset using
Conditional Tabular GAN (CTGAN) or Tabular VAE (TVAE) from the [SDV library](https://sdv.dev/).
The goal is to produce a statistically faithful synthetic dataset that preserves the
distributions and relationships in the real data, without exposing real patient records.

**Source file:** `rds_data_clean.csv` (output of `preprocess.ipynb`)  
**Target column:** `Target` (positive / negative for RDS)  
**Dataset size:** ~1369 rows, mixed categorical + continuous features

---

## When to Use CTGAN vs TVAE

| | CTGAN | TVAE |
|---|---|---|
| Architecture | GAN (generator + discriminator) | Variational Autoencoder |
| Best for | Imbalanced, complex distributions | Smaller datasets, faster training |
| Training stability | Can be unstable (mode collapse risk) | More stable |
| Quality ceiling | Generally higher with tuning | Good out-of-the-box |
| Recommendation | Try first; tune if needed | Fallback if CTGAN is unstable |

Given the class imbalance (positive: 1000, negative: 369) and moderate dataset size (~1369 rows),
**start with CTGAN** and fall back to TVAE if training is unstable.

---

## Step 1 — Install Dependencies

```bash
pip install sdv pandas scikit-learn matplotlib seaborn
```

Verify:
```python
import sdv
print(sdv.__version__)  # should be >= 1.0
```

---

## Step 2 — Prepare the Training Data

Load `rds_data_clean.csv` and select only the columns relevant for synthesis.
Drop identifier columns — they carry no statistical signal and could leak patient identity.

```python
import pandas as pd

df = pd.read_csv('rds_data_clean.csv')

# Drop identifiers — not useful for synthesis
drop_cols = ['Sl.No', 'Ref No Data', 'Entry code', 'IP No',
             'Heart rate', 'Respiratory rate', 'SpO2',       # raw text versions
             'Other Complications/Info']                      # high-cardinality free text

df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Keep the cleaned numeric vitals instead
# Heart_rate_bpm, Resp_rate_cpm, SpO2_pct are already extracted

print(df_model.shape)
print(df_model.dtypes)
```

**Columns to retain for synthesis:**

| Column | Type |
|---|---|
| Birth Term | Categorical |
| Gestational Age (Weeks) | Continuous |
| Category | Categorical (simplify to AGA/SGA/LGA) |
| Place | Categorical |
| Gender | Categorical |
| Type | Categorical |
| Presentation | Categorical |
| Birth weight (gms) | Continuous |
| Apgar at 1 | Continuous |
| Apgar at 5 | Continuous |
| Head circumference | Continuous |
| Length | Continuous |
| Mother Age | Continuous |
| Mother Hb | Continuous |
| Mother Blood Group | Categorical |
| Parity | Categorical (simplify — see note below) |
| Arterial Blood Ph | Continuous |
| Arterial Blood Pco2 | Continuous |
| Arterial Blood Po2 | Continuous |
| Heart_rate_bpm | Continuous |
| Resp_rate_cpm | Continuous |
| SpO2_pct | Continuous |
| Target | Categorical (label) |

> **Note on Parity:** The raw column has 50+ unique string values (e.g., `g2p1l1`, `primigravida`).
> Before synthesis, simplify it to a numeric gravida count or broad category
> (primigravida / multigravida) to avoid the model learning noise.

---

## Step 3 — Further Pre-processing for SDV

SDV handles missing values internally, but you should:

1. Ensure categorical columns are `str` dtype (not mixed)
2. Cap high-cardinality categoricals
3. Confirm numeric columns are `float`

```python
import numpy as np

# Simplify Category to core 3 classes
def simplify_category(val):
    if pd.isna(val):
        return np.nan
    v = str(val).upper()
    if 'SGA' in v:
        return 'SGA'
    if 'LGA' in v:
        return 'LGA'
    if 'AGA' in v:
        return 'AGA'
    return np.nan  # drop rare/compound entries

df_model['Category'] = df_model['Category'].apply(simplify_category)

# Simplify Parity
def simplify_parity(val):
    if pd.isna(val):
        return np.nan
    v = str(val).lower()
    if 'primi' in v or v == 'g1':
        return 'primigravida'
    return 'multigravida'

if 'Parity' in df_model.columns:
    df_model['Parity'] = df_model['Parity'].apply(simplify_parity)

# Ensure correct dtypes
cat_cols = ['Birth Term', 'Category', 'Place', 'Gender', 'Type',
            'Presentation', 'Mother Blood Group', 'Parity', 'Target']
for col in cat_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype('object')  # SDV expects object, not StringDtype

num_cols = ['Gestational Age (Weeks)', 'Birth weight (gms)', 'Apgar at 1', 'Apgar at 5',
            'Head circumference', 'Length', 'Mother Age', 'Mother Hb',
            'Arterial Blood Ph', 'Arterial Blood Pco2', 'Arterial Blood Po2',
            'Heart_rate_bpm', 'Resp_rate_cpm', 'SpO2_pct']
for col in num_cols:
    if col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
```

---

## Step 4 — Define Metadata

SDV requires a metadata object that describes column types.

```python
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_model)

# Manually correct any misdetected columns
for col in cat_cols:
    if col in df_model.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

for col in num_cols:
    if col in df_model.columns:
        metadata.update_column(column_name=col, sdtype='numerical')

metadata.validate()
print(metadata.to_dict())
```

---

## Step 5 — Train the Model

### Option A — CTGAN

```python
from sdv.single_table import CTGANSynthesizer

model = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True
)

model.fit(df_model)
model.save('ctgan_rds.pkl')
```

### Option B — TVAE

```python
from sdv.single_table import TVAESynthesizer

model = TVAESynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    verbose=True
)

model.fit(df_model)
model.save('tvae_rds.pkl')
```

---

## Step 6 — Generate Synthetic Data

```python
# Load saved model (if needed)
# model = CTGANSynthesizer.load('ctgan_rds.pkl')

# Generate — aim for 2-3x the original size
synthetic_df = model.sample(num_rows=3000)
synthetic_df.to_csv('rds_synthetic.csv', index=False)

print(synthetic_df.shape)
print(synthetic_df['Target'].value_counts())
```

> To preserve the original class ratio, use conditional sampling:
> ```python
> from sdv.sampling import Condition
> cond_pos = Condition(num_rows=2000, column_values={'Target': 'positive'})
> cond_neg = Condition(num_rows=1000, column_values={'Target': 'negative'})
> synthetic_df = model.sample_from_conditions([cond_pos, cond_neg])
> ```

---

## Step 7 — Evaluate Synthetic Data Quality

### 7.1 SDV Built-in Report

```python
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

diagnostic = run_diagnostic(real_data=df_model, synthetic_data=synthetic_df, metadata=metadata)
quality_report = evaluate_quality(real_data=df_model, synthetic_data=synthetic_df, metadata=metadata)

print(quality_report.get_score())
quality_report.get_details('Column Shapes')
quality_report.get_details('Column Pair Trends')
```

Scores are 0–1. Aim for > 0.8 on Column Shapes and > 0.7 on Column Pair Trends.

### 7.2 Visual Comparison

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compare distributions for key numeric columns
for col in ['Gestational Age (Weeks)', 'Birth weight (gms)', 'Arterial Blood Ph']:
    if col in df_model.columns:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        df_model[col].dropna().hist(ax=ax[0], bins=20, color='steelblue')
        ax[0].set_title(f'Real — {col}')
        synthetic_df[col].dropna().hist(ax=ax[1], bins=20, color='coral')
        ax[1].set_title(f'Synthetic — {col}')
        plt.tight_layout()
        plt.show()

# Compare Target distribution
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
df_model['Target'].value_counts().plot(kind='bar', ax=axes[0], title='Real Target')
synthetic_df['Target'].value_counts().plot(kind='bar', ax=axes[1], title='Synthetic Target')
plt.tight_layout()
plt.show()
```

### 7.3 Train-on-Synthetic, Test-on-Real (TSTR)

The gold standard for utility evaluation.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def prepare_for_clf(data, cat_cols, num_cols, target='Target'):
    d = data.copy()
    for col in cat_cols:
        if col in d.columns and col != target:
            le = LabelEncoder()
            d[col] = le.fit_transform(d[col].astype(str))
    keep = [c for c in cat_cols + num_cols if c in d.columns]
    X = d[keep].fillna(-1)
    y = (d[target] == 'positive').astype(int)
    return X, y

X_syn, y_syn = prepare_for_clf(synthetic_df, cat_cols, num_cols)
X_real, y_real = prepare_for_clf(df_model, cat_cols, num_cols)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_syn, y_syn)
y_pred = clf.predict(X_real)

print(classification_report(y_real, y_pred))
```

A model trained on synthetic data should achieve comparable F1 to one trained on real data.

---

## Step 8 — Hyperparameter Tuning (if quality is low)

| Parameter | CTGAN default | Try |
|---|---|---|
| `epochs` | 300 | 500–1000 |
| `batch_size` | 500 | 256, 1000 |
| `generator_dim` | (256,256) | (128,128,128) |
| `discriminator_dim` | (256,256) | (128,128,128) |
| `pac` | 10 | 1, 5 |

For TVAE, increasing `epochs` and reducing `compress_dims` often helps with small datasets.

---

## Common Issues & Fixes

| Issue | Likely Cause | Fix |
|---|---|---|
| Mode collapse (synthetic data looks identical) | CTGAN instability | Reduce `pac`, lower learning rate, try TVAE |
| Poor categorical distributions | High cardinality | Simplify categories before training |
| NaN in synthetic output | SDV imputation mismatch | Ensure dtypes are correct before `fit()` |
| Low Column Pair Trends score | Weak correlation learning | Increase epochs, use larger network dims |
| Class imbalance not preserved | Unconstrained sampling | Use `Condition` sampling (Step 6) |

---

## Output Files

| File | Description |
|---|---|
| `rds_data_clean.csv` | Cleaned real data (input to model) |
| `ctgan_rds.pkl` / `tvae_rds.pkl` | Saved trained model |
| `rds_synthetic.csv` | Generated synthetic dataset |

---

## References

- [SDV Documentation](https://docs.sdv.dev/)
- [CTGAN Paper — Xu et al. 2019](https://arxiv.org/abs/1907.00503)
- [TVAE Paper — Xu et al. 2019](https://arxiv.org/abs/1907.00503)
- [SDV Evaluation Guide](https://docs.sdv.dev/sdv/single-table-data/evaluation)
