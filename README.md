# Neonatal RDS Prediction

A machine learning pipeline for early prediction of Respiratory Distress Syndrome (RDS) in neonates, built on a real-world clinical dataset.

## Project Overview

Respiratory Distress Syndrome is a leading cause of morbidity in preterm neonates. This project builds a pipeline that cleans a noisy, manually-entered clinical dataset, generates synthetic data to augment it, and prepares it for ML classification to assist in early identification of at-risk newborns.

## Project Structure

```
├── src/
│   ├── preprocess_1.ipynb       # EDA, initial cleaning, feature extraction
│   └── preprocess_2.ipynb       # Full cleaning pipeline (outlier removal, harmonization, imputation)
├── docs/
│   └── synthetic_data_workflow.md  # Step-by-step guide for CTGAN/TVAE synthetic data generation
├── .gitignore
└── README.md
```

> Dataset files (`.csv`, `.xlsx`) are excluded from this repository as they contain sensitive patient data.

## Pipeline Summary

**1. Exploratory Data Analysis (`preprocess_1.ipynb`)**
- Loaded raw clinical data (1369 records, 29 features)
- Assessed missing values and unique value distributions per column
- Extracted numeric values from free-text fields (vitals, ABG values)

**2. Data Cleaning (`preprocess_2.ipynb`)**
- Dropped identifier columns and sparse/free-text fields
- Clipped physiologically impossible values to NaN across 10 numeric columns
- Harmonized categorical columns:
  - `Birth Term`: 22 variants → 5 standard classes
  - `Category`: 39 variants → AGA / SGA / LGA
  - `Type`: 27 variants → lscs / nvd
  - `Mother Blood Group`: 57 variants → 8 standard blood groups
  - `Parity`: 141 variants → gravida number
- Imputed missing values (mode for categoricals, median for numerics)
- Output: 1369 rows × 22 columns, zero nulls

**3. Synthetic Data Generation (`docs/synthetic_data_workflow.md`)**
- Workflow for generating synthetic data using CTGAN and TVAE (SDV library)
- Covers metadata definition, model training, conditional sampling, and quality evaluation

## Dataset

The dataset contains neonatal clinical records with features including gestational age, birth weight, Apgar scores, arterial blood gas values, maternal parameters, and an RDS diagnosis label (`Target`: positive / negative).

The dataset is **not included** in this repository due to patient privacy constraints.

## Requirements

```bash
pip install pandas numpy scikit-learn sdv matplotlib seaborn
```

## Target Variable

`Target` — binary RDS diagnosis
- `positive`: 1000 cases (73%)
- `negative`: 369 cases (27%)
