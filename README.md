# CreditWatch — Early Financial Risk Monitoring System

A student ML prototype that screens companies for early signs of
financial distress using historical financial-ratio data, with
interpretable, explainable risk signals for analysts to investigate.


---

## 1. Problem Statement

Given a company's financial ratios, can we identify — approximately one
year in advance — whether the company shows signs of financial
deterioration serious enough to warrant an analyst's attention?

## 2. Why This Problem Is Relevant to Credit-Risk Monitoring

Lenders and credit-risk teams don't just evaluate a company once at the
time of loan approval — they monitor the health of their portfolio on an
ongoing basis, because a company that looked healthy at approval time can
deteriorate later. Early warning systems exist to flag this deterioration
*before* it turns into an actual default, so analysts can investigate
proactively rather than reactively.

## 3. Dataset

**UCI Machine Learning Repository — Polish Companies Bankruptcy Data**
(Zięba, Tomczak & Tomczak, 2016), specifically the **`5year.arff`** file.

- The dataset contains **64 financial-ratio attributes** (`Attr1` ...
  `Attr64`) computed from real Polish companies' financial statements, and
  a binary **`class`** target: `1` if the company went bankrupt, `0` if
  not.
- The "5year" file uses financial data collected from the period ~5 years
  before the bankruptcy *evaluation window* began — in practice this file
  is commonly used to frame the prediction as **"predict bankruptcy
  roughly one year ahead"** of the forecasting point used for that file,
  which is why this project describes the task as early (~1-year-ahead)
  risk screening. See the dataset's official documentation for the exact
  definition before quoting this elsewhere.

### Where to download it / where it goes

See [`data/README.md`](data/README.md) for full instructions. Short
version:

1. Download the "Polish companies bankruptcy data" dataset from the UCI
   Machine Learning Repository (it comes as a ZIP with five `.arff`
   files).
2. Extract **only** `5year.arff`.
3. Place it at: **`data/raw/5year.arff`**
4. Do not rename it, edit it, or re-save it from Excel.

### Do I need to convert ARFF to CSV myself?

**No.** `src/data_loader.py` loads the `.arff` file directly using
`scipy.io.arff`, and produces a cleaned CSV copy for your own inspection
in `data/processed/` — you never need to manually convert anything.

## 4. Dataset Characteristics

To be filled in after you run the pipeline (numbers depend on your actual
downloaded copy of the dataset):

- Number of rows: `<run src/data_loader.py to see>`
- Number of raw feature columns: 64 (`Attr1`...`Attr64`)
- Class balance: `<run src/data_loader.py to see — expected to be heavily
  imbalanced toward "not bankrupt">`
- Columns dropped for excessive missingness: `<printed by
  src/preprocessing.py when you run it>`

## 5. Methodology

```
Raw ARFF file (data/raw/5year.arff)
        |
        v
 data_loader.py        -> load into DataFrame, validate target column
        |
        v
 preprocessing.py       -> replace +/-inf with NaN
                          -> drop columns with >30% missing values
                          -> (saved to data/processed/cleaned_5year.csv)
        |
        v
 feature_engineering.py -> add small set of composite ratio features
                          -> (saved to data/processed/features_5year.csv)
        |
        v
 preprocessing.py       -> stratified train/test split
                          -> fit imputer + scaler on TRAIN ONLY
                          -> apply to train and test
        |
        v
 train_models.py        -> train Logistic Regression, Random Forest,
                             XGBoost (supervised), Isolation Forest
                             (unsupervised)
        |
        v
 evaluate_models.py     -> precision/recall/F1/PR-AUC/ROC-AUC,
                             confusion matrices, threshold tuning
        |
        v
 explainability.py      -> SHAP global + per-company explanations
        |
        v
 predict.py             -> apply saved pipeline to new/selected rows,
                             assign risk category
        |
        v
 app/streamlit_app.py   -> dashboard for all of the above
```

## 6. Models

| Model | Role |
|---|---|
| Logistic Regression | Simple, interpretable linear baseline |
| Random Forest | Nonlinear tree-based baseline |
| XGBoost | Primary supervised model (used for SHAP + the dashboard's prediction page) |
| Isolation Forest | Unsupervised anomaly detector — flags unusual financial profiles without ever seeing bankruptcy labels during training |

The project compares whether models that **learn historical bankruptcy
patterns** (the three supervised models) behave differently from a model
that simply looks for **unusual financial profiles** (Isolation Forest),
regardless of whether that profile historically led to bankruptcy. See
Model Evaluation results once you run the pipeline.


## 7. Risk Score

The predicted bankruptcy probability from XGBoost is bucketed into three
**project-defined** categories (see `config.py`):

| Category | Probability range |
|---|---|
| Low Risk | `< 0.30` |
| Medium Risk | `0.30 – 0.60` |
| High Risk | `>= 0.60` |

**These thresholds are not official Vivriti (or any institution's)
categories.** They were chosen as simple, round, interpretable cut-points
that roughly divide the probability range into thirds; they are a
reasonable starting point for a prototype, not a calibrated business
policy. No probability-calibration step (e.g. Platt scaling / isotonic
regression) is currently applied — XGBoost's raw predicted probabilities
are used directly. This is listed under Limitations / Future
Improvements below.

## 15. Project Structure

```
CreditWatch/
├── data/
│   ├── raw/                  # put 5year.arff here (untouched, gitignored)
│   ├── processed/            # generated cleaned/feature CSVs (gitignored)
│   └── README.md
├── notebooks/
│   └── 01_exploration.ipynb  # initial dataset inspection
├── src/
│   ├── data_loader.py        # loads raw ARFF -> DataFrame
│   ├── preprocessing.py      # cleaning, leakage-safe split, imputer/scaler
│   ├── feature_engineering.py# small set of composite ratio features
│   ├── train_models.py       # trains all 4 models
│   ├── evaluate_models.py    # metrics, threshold tuning, PR curves
│   ├── explainability.py     # SHAP global + individual explanations
│   └── predict.py            # applies saved pipeline to new records
├── app/
│   └── streamlit_app.py      # 3-page dashboard
├── models/                   # saved .joblib artifacts (gitignored)
├── tests/
│   └── test_pipeline.py      # end-to-end sanity checks
├── requirements.txt
├── README.md
├── .gitignore
└── config.py                 # all paths, seeds, thresholds in one place
```