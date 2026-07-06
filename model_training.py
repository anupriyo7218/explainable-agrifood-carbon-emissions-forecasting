"""
model_training.py

Project: Explainable Machine Learning for Forecasting Agri-food Carbon
         Emissions Using FAO/IPCC Environmental Data.

Purpose
-------
Loads the leakage-safe, temporally-split train/test data produced by
`data_preprocessing.py`, trains three regression models (Random Forest,
Gradient Boosting, XGBoost), evaluates and times each of them, and
saves both the trained models and a comparison table to disk.

Explicitly OUT OF SCOPE for this script (by design, deferred to later
stages of the project): SHAP explainability, feature importance
analysis, hyperparameter tuning, cross-validation, and the Streamlit
dashboard. Keeping this script narrowly scoped to "train, evaluate,
compare, save" makes each stage of the pipeline independently
reviewable and re-runnable.

Author: Prepared for the Explainable Agri-food CO2 Forecasting MVP
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration constants
#
# Kept at module level, matching data_preprocessing.py, so the pipeline
# stays easy to audit and adjust without hunting through function bodies.
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODEL_COMPARISON_PATH = Path("reports/model_comparison.csv")
PREDICTIONS_DIR = Path("reports/predictions")

# A single random seed constant, reused everywhere a model or splitting
# operation accepts one. Centralising it here (rather than typing `42`
# in three separate model constructors) makes it a one-line change if
# the seed ever needs to be revisited, and makes reproducibility claims
# in the README easy to verify against the actual code.
RANDOM_STATE = 42

TARGET_COLUMN = "total_emission"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_datasets(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the processed train/test splits produced by data_preprocessing.py.

    y_train.csv / y_test.csv are saved as single-column CSVs (see
    data_preprocessing.py's `save_datasets`). They are loaded here and
    squeezed into a 1-D pd.Series rather than left as single-column
    DataFrames, because every sklearn/XGBoost `.fit(X, y)` call expects
    a 1-D target -- passing a DataFrame works in some sklearn versions
    but silently triggers shape-mismatch warnings in others, so this is
    made explicit rather than left to version-dependent behaviour.
    """
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")

    logger.info(
        "Loaded processed data: X_train %s, X_test %s, y_train %s, y_test %s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )

    # Defensive check: if X and y row counts don't match, every metric
    # computed later would silently be wrong (misaligned predictions vs
    # ground truth). Fail loudly here instead.
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(
            "Row count mismatch between features and target: "
            f"train ({len(X_train)} vs {len(y_train)}), "
            f"test ({len(X_test)} vs {len(y_test)}). "
            "Re-run data_preprocessing.py to regenerate consistent files."
        )

    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """
    Instantiate the three regression models this project compares.

    Hyperparameter choices are deliberately left at (near-)library
    defaults, with only mild complexity constraints applied. This is a
    conscious decision, not an oversight: hyperparameter tuning is
    explicitly deferred to a later stage of this project (see module
    docstring), so introducing tuned or hand-picked values here would
    misrepresent this script's role and make the eventual "before/after
    tuning" comparison meaningless.

    `random_state=RANDOM_STATE` is set on every model that accepts it,
    so re-running this script produces identical trained models and
    identical evaluation numbers -- a basic reproducibility requirement
    for a project meant to be reviewed on GitHub.

    A note on model choice: all three models are tree-based ensembles.
    This is intentional and consistent with the scaling decision made
    in data_preprocessing.py (`scale_features_if_required`) -- none of
    these models require feature scaling, so the unscaled processed
    data can be fed to all three without any additional transformation
    step in this script.
    """
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,  # use all available cores; purely a speed decision,
                        # does not affect model output or reproducibility.
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",  # explicit rather than relying on
                                            # the library default, so the
                                            # loss function being optimised
                                            # is visible in code, not just
                                            # implied by the library version.
            eval_metric="rmse",  # explicit rather than relying on the
                                 # library default eval metric. RMSE is
                                 # set here to match the RMSE this script
                                 # already reports for every model in
                                 # `evaluate_model`, so XGBoost's internal
                                 # notion of error and this project's
                                 # reported comparison metric agree.
        ),
    }
    logger.info("Instantiated %d model(s): %s", len(models), list(models.keys()))
    return models


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[object, float]:
    """
    Fit a single model and measure wall-clock training time.

    Training time is measured here (rather than estimated from
    algorithmic complexity) because actual wall-clock time is what
    matters for a project narrative about model trade-offs -- e.g.
    "XGBoost trains faster than Random Forest on this dataset" is a
    claim that should be backed by a measurement, not an assumption
    about the algorithms in the abstract.
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict, np.ndarray]:
    """
    Evaluate a trained model on the held-out test set.

    Metrics computed:
    - R²: proportion of variance in total_emission explained by the
      model. Chosen as the headline metric per this project's
      requirement to select the "best" model by R².
    - MAE: mean absolute error, in the same units as total_emission
      (kt CO2-eq). Included because it's directly interpretable
      ("the model is off by X kt on average") in a way R² is not.
    - RMSE: root mean squared error. Included alongside MAE because it
      penalises large errors more heavily -- comparing MAE and RMSE
      side by side reveals whether a model's errors are fairly uniform
      or dominated by a few large misses (RMSE >> MAE implies the
      latter).

    Prediction time is also measured, since a model that is only
    marginally more accurate but meaningfully slower at inference is a
    relevant trade-off to surface for a project that ends in a live
    Streamlit dashboard making on-demand predictions.

    Returns both the metrics dict AND the raw predictions array (rather
    than metrics alone) so that `main()` can persist per-model
    predictions to disk via `save_predictions` without recomputing
    `model.predict(X_test)` a second time -- avoiding a duplicate,
    wasteful inference pass and, for models with any non-determinism,
    guaranteeing the saved predictions are exactly the ones the
    reported metrics were computed from.
    """
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "Prediction_Time_Seconds": prediction_time,
    }
    return metrics, y_pred


def save_predictions(
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Save a model's test-set predictions alongside the actual values.

    This is saved per-model (rather than one combined file) so each
    file is a self-contained record of exactly what that model
    predicted on the held-out 2016-2020 test set. Keeping actual and
    predicted values together in one file (rather than only the
    predictions) makes each CSV independently useful for later error
    analysis or plotting, without needing to re-join it against
    y_test.csv first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": y_pred,
    })
    predictions_df["error"] = predictions_df["actual"] - predictions_df["predicted"]

    filename = f"{model_name.lower()}_predictions.csv"
    output_path = output_dir / filename
    predictions_df.to_csv(output_path, index=False)
    logger.info("Saved %s test-set predictions to '%s'.", model_name, output_path)
    return output_path


def save_model(model, model_name: str, output_dir: Path) -> Path:
    """
    Persist a trained model to disk with joblib.

    joblib is used instead of pickle because it is more efficient for
    objects containing large NumPy arrays (the fitted trees inside RF /
    Gradient Boosting / XGBoost), which is the standard, documented
    recommendation from scikit-learn itself for persisting fitted
    estimators.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Filenames are derived from the model name via a simple lowercase
    # mapping rather than hardcoded per-model, so adding a fourth model
    # to `build_models()` later does not require a matching change here.
    filename = f"{model_name.lower()}.pkl"
    model_path = output_dir / filename
    joblib.dump(model, model_path)
    logger.info("Saved trained model '%s' to '%s'.", model_name, model_path)
    return model_path


def compare_models(results: dict) -> pd.DataFrame:
    """
    Assemble per-model timing and evaluation results into a single
    comparison table, sorted by R² (descending) so the best-performing
    model is always the first row.
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"
    comparison_df = comparison_df.sort_values("R2", ascending=False)

    # Reorder columns for readability: performance metrics first, timing
    # metrics last, since performance is the primary axis of comparison
    # this project's requirements evaluate models on.
    column_order = ["R2", "MAE", "RMSE", "Training_Time_Seconds", "Prediction_Time_Seconds"]
    comparison_df = comparison_df[[c for c in column_order if c in comparison_df.columns]]

    return comparison_df


def identify_best_model(comparison_df: pd.DataFrame) -> tuple[str, pd.Series]:
    """
    Identify the best-performing model by R², and log a clear report.

    R² is used as the sole selection criterion here because the project
    requirements explicitly specify "best-performing model based on R²"
    -- MAE/RMSE are reported for interpretability and error-shape
    context, but are not used to override the R²-based ranking in this
    script. (A more nuanced multi-metric selection process, weighing
    error tolerance against the dashboard's needs, is a reasonable
    future refinement but is out of scope here.)
    """
    best_model_name = comparison_df["R2"].idxmax()
    best_row = comparison_df.loc[best_model_name]

    logger.info("=" * 60)
    logger.info("BEST-PERFORMING MODEL (by R2): %s", best_model_name)
    logger.info(
        "  R2=%.4f | MAE=%.2f | RMSE=%.2f | Train time=%.2fs | Predict time=%.4fs",
        best_row["R2"], best_row["MAE"], best_row["RMSE"],
        best_row["Training_Time_Seconds"], best_row["Prediction_Time_Seconds"],
    )
    logger.info("=" * 60)

    return best_model_name, best_row


def save_comparison_table(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Persist the model comparison table to disk as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path)
    logger.info("Saved model comparison table to '%s'.", output_path)


def main() -> None:
    """Run the full model training and evaluation pipeline end to end."""
    logger.info("Starting model training pipeline.")

    X_train, X_test, y_train, y_test = load_datasets(PROCESSED_DATA_DIR)
    models = build_models()

    results = {}
    for model_name, model in models.items():
        logger.info("Training %s...", model_name)
        trained_model, training_time = train_model(model, X_train, y_train)

        logger.info("Evaluating %s...", model_name)
        metrics, y_pred = evaluate_model(trained_model, X_test, y_test)
        metrics["Training_Time_Seconds"] = training_time

        logger.info(
            "%s -> R2=%.4f | MAE=%.2f | RMSE=%.2f | Train time=%.2fs | Predict time=%.4fs",
            model_name, metrics["R2"], metrics["MAE"], metrics["RMSE"],
            metrics["Training_Time_Seconds"], metrics["Prediction_Time_Seconds"],
        )

        results[model_name] = metrics
        save_model(trained_model, model_name, MODELS_DIR)
        save_predictions(model_name, y_test, y_pred, PREDICTIONS_DIR)

    comparison_df = compare_models(results)
    save_comparison_table(comparison_df, MODEL_COMPARISON_PATH)
    identify_best_model(comparison_df)

    logger.info("Model training pipeline complete.")


if __name__ == "__main__":
    main()
