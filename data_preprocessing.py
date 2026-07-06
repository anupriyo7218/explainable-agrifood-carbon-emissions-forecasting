"""
data_preprocessing.py

Project: Explainable Machine Learning for Forecasting Agri-food Carbon
         Emissions Using FAO/IPCC Environmental Data.

Purpose
-------
Loads the raw FAO/IPCC agri-food CO2 emissions dataset and produces a
leakage-safe, model-ready train/test split, saved to disk as CSV files.

This script performs ONLY preprocessing. No model training occurs here.

Why "leakage-safe" matters for this dataset
--------------------------------------------
`total_emission` in the raw data is not an independently measured
quantity -- it is the arithmetic sum of ~23 other columns in the dataset
(Crop Residues, Food Transport, Manure Management, Forestland, etc.).
Using any of those columns as predictors would let the model "predict"
the target by addition rather than by learning a genuine emissions
relationship, producing misleadingly high accuracy. This script removes
every column that is a component of that sum and keeps only variables
that plausibly *drive* emissions without being mathematically derived
from them: population figures, country identity, and year.

Author: Prepared for the Explainable Agri-food CO2 Forecasting MVP
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants
#
# Centralising these at module level (rather than burying them inside
# functions) makes the pipeline easy to audit and adjust without hunting
# through function bodies -- a small thing that matters when someone else
# (or future-you) reviews this on GitHub.
# ---------------------------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/Agrofood_co2_emission.csv")
PROCESSED_DATA_DIR = Path("data/processed")

TARGET_COLUMN = "total_emission"

# These are the ONLY columns approved as predictors. They were selected
# because none of them are mathematical components of `total_emission`,
# so they carry no leakage risk. See module docstring and
# `LEAKAGE_COLUMNS` below for the full reasoning.
APPROVED_PREDICTOR_COLUMNS = [
    "Area",
    "Year",
    "Rural population",
    "Urban population",
    "Total Population - Male",
    "Total Population - Female",
]

# Every column below sums exactly to `total_emission` (verified
# numerically: summing these 23 columns per row reproduces
# `total_emission` to within floating-point precision across the full
# dataset). They are excluded categorically -- not filtered by
# correlation or feature importance -- because the relationship is
# definitional, not statistical, and no amount of regularisation fixes
# that.
LEAKAGE_COLUMNS = [
    "Savanna fires",
    "Forest fires",
    "Crop Residues",
    "Rice Cultivation",
    "Drained organic soils (CO2)",
    "Pesticides Manufacturing",
    "Food Transport",
    "Forestland",
    "Net Forest conversion",
    "Food Household Consumption",
    "Food Retail",
    "On-farm Electricity Use",
    "Food Packaging",
    "Agrifood Systems Waste Disposal",
    "Food Processing",
    "Fertilizers Manufacturing",
    "IPPU",
    "Manure applied to Soils",
    "Manure left on Pasture",
    "Manure Management",
    "Fires in organic soils",
    "Fires in humid tropical forests",
    "On-farm energy use",
]

# Temporal split boundary. Chosen (rather than a random 80/20 split)
# because this dataset is a country-year panel intended for
# *forecasting*: a random split would let the model train on a
# country's 2019 data while being tested on that same country's 2017
# data, which is not a realistic forecasting scenario and inflates
# apparent performance through temporal autocorrelation.
TRAIN_YEAR_MAX = 2015
TEST_YEAR_MIN = 2016

# Valid range for the Year column itself, independent of the train/test
# split boundary above. This dataset covers 1990-2020 by construction
# (FAO/IPCC agri-food emissions coverage); any value outside that range
# indicates a data quality problem upstream (e.g. a corrupted row or a
# unit/format mismatch) rather than a legitimate new observation, and
# should stop the pipeline rather than be silently included.
VALID_YEAR_MIN = 1990
VALID_YEAR_MAX = 2020

NUMERIC_PREDICTOR_COLUMNS = [
    "Rural population",
    "Urban population",
    "Total Population - Male",
    "Total Population - Female",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the raw dataset from disk.

    Parameters
    ----------
    path : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Place Agrofood_co2_emission.csv under data/raw/ before running."
        )

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Confirm that every column this pipeline depends on is actually present.

    Failing loudly and early here (rather than letting a KeyError surface
    deep inside a later function) makes the script safer to hand off to
    someone else, or to run again after the raw CSV changes.
    """
    required_columns = set(APPROVED_PREDICTOR_COLUMNS) | {TARGET_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {sorted(missing)}. "
            "Cannot proceed with preprocessing."
        )
    logger.info("Schema validation passed: all required columns present.")


def detect_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect exact duplicate rows, log the count, and remove them if found.

    Rationale: this dataset is assembled by merging roughly a dozen
    separate FAO/IPCC source tables (per the dataset documentation). A
    merge/join step like that is a common place for a country-year
    observation to be accidentally duplicated. Duplicate rows would
    silently over-represent those country-years in training, biasing
    the model toward whatever pattern the duplicated rows happen to
    show. Detecting and removing them is treated as a distinct,
    logged step -- rather than a side effect of some other function --
    so the decision is visible and auditable in the pipeline output.
    """
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.warning(
            "Found and removed %d duplicate row(s). %d rows remain.",
            n_duplicates, len(df),
        )
    else:
        logger.info("No duplicate rows found.")
    return df


def validate_year_column(df: pd.DataFrame) -> None:
    """
    Validate the integrity of the `Year` column before it is used as the
    temporal split key.

    Two checks are enforced:

    1. `Year` must be numeric. A non-numeric Year (e.g. a stray string
       or date-formatted value) would silently break the `<=` / `>=`
       comparisons used later in `temporal_train_test_split`, producing
       a wrong split rather than a visible error.
    2. Every value must fall within the dataset's known valid range
       (VALID_YEAR_MIN-VALID_YEAR_MAX, i.e. 1990-2020). A value outside
       that range points to a data quality issue upstream (e.g. a
       corrupted row, a typo, or a unit mismatch) and should be
       surfaced immediately rather than allowed to quietly land in
       either the training or test set.

    This function is validation-only: it does not modify or return the
    dataframe. It raises `ValueError` with the offending values listed
    if either check fails, so a broken raw file fails loudly at the
    start of the pipeline rather than producing a subtly wrong split
    downstream.
    """
    if not pd.api.types.is_numeric_dtype(df["Year"]):
        raise ValueError(
            "'Year' column must be numeric. Found dtype: "
            f"{df['Year'].dtype}. Check the raw CSV for corrupted or "
            "non-numeric Year values before proceeding."
        )

    invalid_mask = (df["Year"] < VALID_YEAR_MIN) | (df["Year"] > VALID_YEAR_MAX)
    if invalid_mask.any():
        invalid_values = sorted(df.loc[invalid_mask, "Year"].unique().tolist())
        raise ValueError(
            f"'Year' column contains value(s) outside the expected "
            f"{VALID_YEAR_MIN}-{VALID_YEAR_MAX} range: {invalid_values}. "
            "Investigate these rows in the raw CSV before proceeding."
        )

    logger.info(
        "Year column validated: numeric, all values within %d-%d.",
        VALID_YEAR_MIN, VALID_YEAR_MAX,
    )


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that mathematically sum to the target.

    These columns are dropped unconditionally and *before* any feature
    selection step, so there is no code path later in the pipeline that
    could accidentally reintroduce them.
    """
    present_leakage_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df_clean = df.drop(columns=present_leakage_cols)
    logger.info(
        "Dropped %d leakage column(s) that sum to '%s'.",
        len(present_leakage_cols),
        TARGET_COLUMN,
    )
    return df_clean


def select_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataframe to the approved predictors plus the target.

    This is a deliberate allow-list (rather than a deny-list of "drop
    everything bad") because it's the safer default for a dataset this
    wide: any future column added to the raw CSV is excluded by default
    until someone explicitly reviews and approves it as a predictor.
    """
    keep_columns = APPROVED_PREDICTOR_COLUMNS + [TARGET_COLUMN]
    df_selected = df[keep_columns].copy()
    logger.info("Selected %d approved predictor column(s) + target.", len(APPROVED_PREDICTOR_COLUMNS))
    return df_selected


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the approved feature set.

    Decisions (documented per-column so the reasoning survives a GitHub
    code review, even though this specific dataset happens to contain
    zero missing values in the approved columns as of the version this
    pipeline was built against):

    - Numeric population columns (Rural/Urban population, Total
      Population - Male/Female): imputed with the column MEDIAN rather
      than the mean. Population figures span several orders of
      magnitude across countries (from small island nations to
      billion-plus countries), so the mean is heavily skewed by a small
      number of very large countries. The median is robust to that
      skew and is a safer default for a "no information available"
      fallback.
    - Area (categorical, country/region name): imputed with the literal
      string "Unknown" rather than dropped, so that a missing country
      label doesn't silently delete an otherwise-valid emissions
      observation. "Unknown" becomes its own one-hot category rather
      than being conflated with any real country.
    - Year: NOT imputed. A missing year cannot be safely guessed --
      doing so risks placing a row on the wrong side of the temporal
      train/test split, which would quietly break the forecasting
      evaluation this pipeline is built around. Rows with a missing
      Year are dropped instead, with a warning logged.
    - total_emission (target): rows with a missing target are dropped.
      A model cannot learn from, or be evaluated against, a label that
      doesn't exist, and imputing a target value would fabricate
      ground truth.
    """
    df = df.copy()

    rows_before = len(df)
    df = df.dropna(subset=["Year", TARGET_COLUMN])
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.warning(
            "Dropped %d row(s) with missing Year or missing target.", rows_dropped
        )

    for col in NUMERIC_PREDICTOR_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info(
                "Imputed %d missing value(s) in '%s' with median (%.2f).",
                n_missing, col, median_value,
            )

    n_missing_area = df["Area"].isna().sum()
    if n_missing_area > 0:
        df["Area"] = df["Area"].fillna("Unknown")
        logger.info("Imputed %d missing value(s) in 'Area' with 'Unknown'.", n_missing_area)

    logger.info("Missing value handling complete. %d rows remain.", len(df))
    return df


def encode_area_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the `Area` (country/region) column.

    Encoding strategy and justification
    ------------------------------------
    `Area` is a NOMINAL categorical variable -- countries have no
    inherent order or numeric relationship to one another. Two encoding
    strategies were considered and rejected in favour of one-hot:

    1. Label/ordinal encoding (assigning each country an arbitrary
       integer) was rejected because it implies a false ordinal
       relationship (e.g. that "Country code 40" is numerically "between"
       codes 39 and 41 in some meaningful sense) that tree-based models
       can exploit in ways that are difficult to interpret with SHAP --
       a SHAP value attached to "country_code <= 87.5" is far less
       meaningful to a reader than one attached to "is_Country_X = 1".

    2. Target/mean encoding (replacing each country with a statistic of
       its own target values) was rejected because it leaks target
       information directly into the feature itself, which is exactly
       the kind of leakage this pipeline is designed to avoid elsewhere.

    One-hot encoding avoids both problems: every country becomes an
    independent binary indicator, preserves no false ordering, and
    injects no target information. With ~236 unique countries and
    ~7,000 rows, the resulting feature matrix is wider but remains a
    manageable size for the tree-based models (Random Forest, XGBoost,
    Gradient Boosting) planned for this project, all of which handle
    sparse binary indicator features natively.

    Implementation note: category levels are established from the full
    dataset (train + test combined) *before* the temporal split, not
    from the training set alone. This is safe and does not leak target
    information -- the set of countries that exist is fixed, observable
    metadata, not a statistic derived from `total_emission`. Fitting
    categories on the training set alone would instead risk a real bug:
    a country appearing only in 2016-2020 test rows would silently
    produce an all-zero encoding no training data ever addressed.
    """
    df = df.copy()
    area_dummies = pd.get_dummies(df["Area"], prefix="Area", dtype=int)
    df = pd.concat([df.drop(columns=["Area"]), area_dummies], axis=1)
    logger.info(
        "One-hot encoded 'Area' into %d indicator columns.", area_dummies.shape[1]
    )
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by Year rather than randomly.

    Training set:  Year <= TRAIN_YEAR_MAX (1990-2015)
    Test set:       Year >= TEST_YEAR_MIN (2016-2020)

    Rationale: this dataset is a country-year panel explicitly intended
    to support forecasting. A random row-wise split would place, e.g.,
    France-2019 in training and France-2017 in test -- the model would
    then be evaluated on a year it has effectively already "seen" the
    neighbourhood of, via other years from the same country. That
    inflates apparent performance and does not reflect the real task
    (predicting emissions in years not yet observed). A strict temporal
    cutoff is the standard, more honest evaluation design for panel /
    time-series data.
    """
    train_df = df[df["Year"] <= TRAIN_YEAR_MAX].copy()
    test_df = df[df["Year"] >= TEST_YEAR_MIN].copy()

    logger.info(
        "Temporal split: %d training rows (<= %d), %d test rows (>= %d).",
        len(train_df), TRAIN_YEAR_MAX, len(test_df), TEST_YEAR_MIN,
    )
    return train_df, test_df


def scale_features_if_required(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Document the scaling decision for this pipeline.

    Decision: NO scaling is applied to the numeric predictors.

    Justification: the three models planned for this project -- Random
    Forest, XGBoost, and Gradient Boosting -- are all tree-based
    ensembles. Tree splits are based on threshold comparisons on raw
    feature values, so they are invariant to monotonic transformations
    like standardisation or min-max scaling. Scaling would add a
    preprocessing step (and an artifact -- a fitted scaler -- that must
    be saved and consistently re-applied at inference time) with no
    modelling benefit for this model family.

    This decision would need to be revisited if a distance-based or
    gradient-based linear model (e.g. Ridge, SVR) or a neural network
    (e.g. MLPRegressor) is added later, since those ARE sensitive to
    feature scale. That is flagged here deliberately so the decision is
    visible to a future contributor rather than silently assumed.

    This function performs no transformation; it exists purely to make
    the "we considered scaling and chose not to do it" decision
    explicit and testable in the pipeline, rather than an implicit
    omission.
    """
    logger.info(
        "Scaling skipped: all planned models (RF, XGBoost, Gradient "
        "Boosting) are tree-based and scale-invariant. Revisit if a "
        "linear or neural model is added."
    )


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a dataframe into feature matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def save_feature_names(feature_columns: list[str], output_dir: Path) -> None:
    """
    Save the final model-ready feature names to `feature_names.csv`.

    Rationale: after one-hot encoding, the feature set expands from 6
    approved predictors to ~241 columns (5 numeric + ~236 `Area_*`
    indicators), and the exact set/order of those columns is an
    artifact of whatever countries happen to be present in the raw
    data. Persisting the final feature list makes that artifact
    explicit and reusable -- downstream scripts (model training, SHAP
    explainability, the Streamlit app) can load this file to align
    input columns at inference time, rather than each independently
    re-deriving the one-hot schema and risking a mismatch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_columns, name="feature_name").to_csv(
        output_dir / "feature_names.csv", index=False
    )
    logger.info(
        "Saved %d feature name(s) to '%s'.",
        len(feature_columns), output_dir / "feature_names.csv",
    )


def save_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Persist the processed train/test splits to disk as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logger.info("Saved processed datasets to '%s'.", output_dir)
    logger.info(
        "  X_train: %s | X_test: %s | y_train: %s | y_test: %s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )


def main() -> None:
    """Run the full preprocessing pipeline end to end."""
    logger.info("Starting preprocessing pipeline.")

    df = load_data(RAW_DATA_PATH)
    validate_schema(df)
    df = detect_and_remove_duplicates(df)
    validate_year_column(df)

    df = drop_leakage_columns(df)
    df = select_features_and_target(df)
    df = handle_missing_values(df)
    df = encode_area_column(df)

    train_df, test_df = temporal_train_test_split(df)
    scale_features_if_required(train_df, test_df)

    X_train, y_train = split_features_and_target(train_df)
    X_test, y_test = split_features_and_target(test_df)

    save_feature_names(X_train.columns.tolist(), PROCESSED_DATA_DIR)
    save_datasets(X_train, X_test, y_train, y_test, PROCESSED_DATA_DIR)

    logger.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
