"""
shap_analysis.py

Project: Explainable Machine Learning for Forecasting Agri-food Carbon
         Emissions Using FAO/IPCC Environmental Data.

Purpose
-------
Loads the best-performing trained model (Gradient Boosting, selected by
`model_training.py`'s R²-based comparison) and computes SHAP values on
the held-out test set to explain its predictions. Saves the summary
(beeswarm) plot, the bar-chart feature importance plot, and the raw
SHAP values to disk for reuse in later reporting or the Streamlit
dashboard.

This script does NOT retrain any model and does NOT modify the
preprocessing pipeline -- it only loads artifacts already produced by
`data_preprocessing.py` and `model_training.py` and explains them.

Author: Prepared for the Explainable Agri-food CO2 Forecasting MVP
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Configuration constants
#
# Kept at module level, matching data_preprocessing.py and
# model_training.py, so the pipeline stays easy to audit and adjust
# without hunting through function bodies.
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SHAP_OUTPUT_DIR = Path("reports/shap")

# The model this script explains. Hardcoded to the specific model that
# model_training.py's R²-based comparison selected as best, rather than
# re-deriving "best" here -- this script's job is to explain a given
# model, not to re-run model selection. If a different model becomes
# best after a future re-run of model_training.py, this constant is
# the single line to update.
MODEL_NAME = "GradientBoosting"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME.lower()}.pkl"

# Cap on the number of test-set rows used for SHAP computation. Set to
# None to use the entire test set. Exposed as a constant (rather than
# hardcoded inline) so it can be lowered if the test set grows large
# enough that full-dataset SHAP computation becomes a runtime concern --
# not needed at this project's current test-set size (~1,100 rows), but
# worth having as a documented lever rather than an implicit assumption.
MAX_SHAP_ROWS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """
    Load the trained Gradient Boosting model saved by model_training.py.

    Loaded with joblib rather than pickle directly, matching how
    model_training.py's `save_model` wrote it -- joblib.load is the
    documented, correct counterpart to joblib.dump for scikit-learn
    estimators.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run model_training.py first to produce this file."
        )
    model = joblib.load(model_path)
    logger.info("Loaded trained model '%s' from '%s'.", MODEL_NAME, model_path)
    return model


def load_test_features_and_names(
    processed_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load X_test.csv and the saved feature_names.csv.

    feature_names.csv was written by data_preprocessing.py's
    `save_feature_names` specifically so downstream scripts like this
    one don't need to independently re-derive the one-hot-encoded
    feature schema -- it is loaded here as the single source of truth
    for "what columns should X_test have, in what order."
    """
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    feature_names = pd.read_csv(processed_dir / "feature_names.csv")["feature_name"].tolist()

    logger.info(
        "Loaded X_test %s and %d saved feature name(s).",
        X_test.shape, len(feature_names),
    )
    return X_test, feature_names


def validate_feature_alignment(X_test: pd.DataFrame, feature_names: list[str]) -> None:
    """
    Confirm that X_test's columns exactly match the saved feature names,
    in both membership and order, before any SHAP computation runs.

    This check exists because SHAP values are only meaningful if each
    column of X_test lines up with the same column the model was
    trained on. A silent mismatch here (e.g. a column dropped, renamed,
    or reordered somewhere upstream) would not necessarily raise an
    error in sklearn/SHAP -- it could instead silently attribute SHAP
    values to the WRONG feature names, producing a plausible-looking
    but incorrect explanation. Failing loudly here is far preferable to
    a quietly mislabeled beeswarm plot.
    """
    X_test_columns = list(X_test.columns)

    if X_test_columns == feature_names:
        logger.info("Feature alignment validated: X_test columns match saved feature names exactly.")
        return

    # Provide a specific, actionable diagnosis rather than a generic
    # "mismatch" message -- distinguish between a pure ordering
    # difference (fixable by reordering) and an actual membership
    # difference (a real upstream problem).
    missing_from_X_test = set(feature_names) - set(X_test_columns)
    unexpected_in_X_test = set(X_test_columns) - set(feature_names)

    if not missing_from_X_test and not unexpected_in_X_test:
        raise ValueError(
            "X_test contains the correct set of features but in a "
            "different order than feature_names.csv. Reorder X_test "
            "columns to match feature_names.csv before computing SHAP "
            "values, rather than proceeding with a silently misaligned "
            "explanation."
        )

    raise ValueError(
        "X_test columns do not match the saved feature_names.csv. "
        f"Missing from X_test: {sorted(missing_from_X_test)}. "
        f"Unexpected in X_test: {sorted(unexpected_in_X_test)}. "
        "Re-run data_preprocessing.py to regenerate consistent files "
        "before computing SHAP values."
    )


def compute_shap_values(model, X_test: pd.DataFrame) -> tuple[shap.TreeExplainer, np.ndarray]:
    """
    Compute SHAP values for the test set using shap.TreeExplainer.

    TreeExplainer is used (rather than the model-agnostic
    KernelExplainer) because Gradient Boosting is a tree-based ensemble.
    TreeExplainer computes EXACT Shapley values for tree models via a
    polynomial-time algorithm specific to trees, rather than KernelExplainer's
    sampling-based approximation -- so this is both the faster and the
    more accurate choice available for this specific model, not merely
    the more convenient one.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info(
        "Computed SHAP values for %d test row(s) x %d feature(s).",
        shap_values.shape[0], shap_values.shape[1],
    )
    return explainer, shap_values


def save_shap_values(shap_values: np.ndarray, feature_names: list[str], output_dir: Path) -> Path:
    """
    Persist the raw SHAP values to disk as a .npy file for later reuse.

    Saved as a plain NumPy array (rather than only as plots) so later
    stages of the project -- the Streamlit dashboard's live waterfall
    plot, or further written analysis -- can load the exact already-
    computed values instead of recomputing them, which matters if
    TreeExplainer computation time ever becomes non-trivial on a larger
    dataset. The corresponding feature name order is saved alongside it
    so the array is never ambiguous about which column is which.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values_path = output_dir / "shap_values.npy"
    np.save(shap_values_path, shap_values)

    feature_order_path = output_dir / "shap_feature_order.csv"
    pd.Series(feature_names, name="feature_name").to_csv(feature_order_path, index=False)

    logger.info("Saved raw SHAP values to '%s'.", shap_values_path)
    logger.info("Saved corresponding feature order to '%s'.", feature_order_path)
    return shap_values_path


def save_summary_plot(shap_values: np.ndarray, X_test: pd.DataFrame, output_dir: Path) -> Path:
    """
    Generate and save the SHAP summary (beeswarm) plot.

    The beeswarm plot is this project's primary global-explainability
    artifact: it shows, for every feature, the direction and magnitude
    of its effect across every test-set prediction simultaneously --
    which single-number bar charts cannot convey (e.g. whether a
    feature's high values consistently push predictions up, down, or
    both depending on context).

    `show=False` is passed to shap.summary_plot so it draws into the
    current matplotlib figure without opening an interactive window --
    necessary for this script to run non-interactively (e.g. in CI or
    from the command line) and still produce a saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shap_summary_beeswarm.png"

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary (beeswarm) plot to '%s'.", output_path)
    return output_path


def save_bar_plot(shap_values: np.ndarray, X_test: pd.DataFrame, output_dir: Path) -> Path:
    """
    Generate and save the SHAP bar plot (mean absolute SHAP value per feature).

    This is the simpler, single-number-per-feature companion to the
    beeswarm plot: it ranks features by average impact magnitude
    without showing direction, making it a faster-to-read "what matters
    most" reference -- useful for the Streamlit dashboard's model
    comparison view, where a dense beeswarm plot per model would be
    visually overwhelming.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shap_bar_importance.png"

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP bar (feature importance) plot to '%s'.", output_path)
    return output_path


def main() -> None:
    """Run the full SHAP explainability pipeline end to end."""
    logger.info("Starting SHAP analysis for model '%s'.", MODEL_NAME)

    model = load_model(MODEL_PATH)
    X_test, feature_names = load_test_features_and_names(PROCESSED_DATA_DIR)

    validate_feature_alignment(X_test, feature_names)

    if MAX_SHAP_ROWS is not None and len(X_test) > MAX_SHAP_ROWS:
        logger.info(
            "Limiting SHAP computation to the first %d of %d test rows "
            "(MAX_SHAP_ROWS is set).", MAX_SHAP_ROWS, len(X_test),
        )
        X_test = X_test.iloc[:MAX_SHAP_ROWS].reset_index(drop=True)

    _, shap_values = compute_shap_values(model, X_test)

    save_shap_values(shap_values, feature_names, SHAP_OUTPUT_DIR)
    save_summary_plot(shap_values, X_test, SHAP_OUTPUT_DIR)
    save_bar_plot(shap_values, X_test, SHAP_OUTPUT_DIR)

    logger.info("SHAP analysis complete. All outputs saved to '%s'.", SHAP_OUTPUT_DIR)


if __name__ == "__main__":
    main()
