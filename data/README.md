# Data folder

This folder is intentionally split into `raw/` and `processed/` so that the
original dataset is never overwritten.

## raw/

Put the downloaded dataset file here, **unmodified**:

```
data/raw/5year.arff
```

### Where to get it

1. Go to the UCI Machine Learning Repository page for the
   **"Polish companies bankruptcy data"** dataset (Zięba, Tomczak & Tomczak,
   contributed 2016).
2. Download the dataset archive (it is distributed as a ZIP containing five
   ARFF files: `1year.arff`, `2year.arff`, `3year.arff`, `4year.arff`,
   `5year.arff` — one per forecasting horizon).
3. Extract **only `5year.arff`** and place it at `data/raw/5year.arff`.

The "5year" file corresponds to financial data collected 1 year before the
bankruptcy status was evaluated, which is why this project treats it as
"predict distress ~1 year ahead." Do not rename the file — `config.py`
expects the exact name `5year.arff`.

### Do NOT

- Do not open the `.arff` file in Excel and re-save it (this can corrupt the
  ARFF header/format).
- Do not manually edit, delete, or reorder rows/columns.
- Do not commit this file to git — it is excluded via `.gitignore` because
  of UCI's redistribution terms and file size.

## processed/

This folder is generated automatically by the code
(`src/preprocessing.py` and `src/feature_engineering.py`). You do not need
to create anything here manually. It contains:

- `cleaned_5year.csv` — the raw data after basic cleaning (missing value
  handling, infinite value handling), with no feature engineering and no
  scaling applied yet.
- `features_5year.csv` — the cleaned data plus the small set of engineered
  financial ratio features, ready for modeling.

Both files are derived artifacts and are also excluded from git via
`.gitignore` (they can always be regenerated from the raw file).
