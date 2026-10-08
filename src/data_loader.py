"""
data_loader.py

Responsible for ONE thing: loading the raw UCI "Polish companies
bankruptcy data" (5year.arff) file into a pandas DataFrame, without
modifying its content.

The UCI dataset ships as ARFF (Attribute-Relation File Format), which is
the standard export format for Weka. Each attribute in the 5year.arff file
is:
    Attr1, Attr2, ..., Attr64   -> numeric financial ratios
    class                       -> nominal target, values '0' or '1'

We use scipy.io.arff to parse this, because it is already a dependency-free
way to read ARFF files (no extra package needed beyond scipy, which
scikit-learn already depends on).

IMPORTANT: This file does NOT clean, impute, or engineer anything. It only
loads and does the minimum type conversion needed to get a usable
DataFrame (e.g. decoding the nominal class column from bytes to a string,
then to int). All real cleaning happens in preprocessing.py.
"""

import os
import sys
import pandas as pd
from scipy.io import arff

# Allow this file to be run either as `python src/data_loader.py` (script
# mode, where only src/ is on sys.path by default) or imported as part of
# the package (e.g. from app/streamlit_app.py). Either way, config.py at
# the project root needs to be importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import RAW_ARFF_PATH, RAW_TARGET_COLUMN, TARGET_COLUMN


def load_raw_dataset(arff_path: str = RAW_ARFF_PATH) -> pd.DataFrame:
    """
    Load the raw 5year.arff file into a pandas DataFrame.

    Parameters
    ----------
    arff_path : str
        Path to the .arff file. Defaults to config.RAW_ARFF_PATH, i.e.
        data/raw/5year.arff.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Attr1..Attr64 (floats) and a "bankrupt"
        column (int, 0 or 1). Row order matches the original file.

    Raises
    ------
    FileNotFoundError
        If the file has not been downloaded yet. The error message tells
        the user exactly where to put it.
    """
    if not os.path.exists(arff_path):
        raise FileNotFoundError(
            f"Could not find the raw dataset at '{arff_path}'.\n"
            "Please download '5year.arff' from the UCI Machine Learning "
            "Repository's 'Polish companies bankruptcy data' dataset and "
            "place it at that exact path. See data/README.md for details."
        )

    # scipy.io.arff.loadarff returns (data, meta):
    #   data -> a numpy structured array
    #   meta -> attribute metadata (names, types) -- useful for sanity checks
    raw_data, meta = arff.loadarff(arff_path)

    df = pd.DataFrame(raw_data)

    # Sanity check: make sure the target column we expect actually exists.
    # We deliberately do NOT guess a target column -- if this fails, it
    # means the file is not the dataset we expect, and we want a loud error
    # rather than silently training on the wrong column.
    if RAW_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a column named '{RAW_TARGET_COLUMN}' in the ARFF "
            f"file but found columns: {list(df.columns)}. This does not "
            "look like the expected UCI 5year.arff file."
        )

    # The nominal target column comes back as bytes (e.g. b'0', b'1')
    # because ARFF nominal attributes are read as byte strings by scipy.
    # Decode to a plain string, then to int.
    df[RAW_TARGET_COLUMN] = df[RAW_TARGET_COLUMN].apply(
        lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
    )
    df[RAW_TARGET_COLUMN] = df[RAW_TARGET_COLUMN].astype(int)

    # Rename to a more readable name used throughout the rest of the
    # project. We keep this rename (and only this rename) in one place so
    # it's easy to audit -- we are not silently renaming any other column.
    df = df.rename(columns={RAW_TARGET_COLUMN: TARGET_COLUMN})

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of raw predictor columns (everything except the
    target). Kept as a small helper so other scripts don't need to repeat
    the "all columns except target" logic.
    """
    return [c for c in df.columns if c != TARGET_COLUMN]


if __name__ == "__main__":
    # Quick manual check: `python src/data_loader.py` from the project root
    # should print the shape and target distribution of the raw data.
    data = load_raw_dataset()
    print("Loaded raw dataset with shape:", data.shape)
    print("\nTarget distribution (0 = not bankrupt, 1 = bankrupt):")
    print(data[TARGET_COLUMN].value_counts())
    print("\nFirst few rows:")
    print(data.head())
