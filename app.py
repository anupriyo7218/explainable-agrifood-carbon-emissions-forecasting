"""
app.py

Project: Explainable Machine Learning for Forecasting Agri-food Carbon
         Emissions Using FAO/IPCC Environmental Data.

Purpose
-------
A Streamlit dashboard presenting the completed ML pipeline: project
overview, model performance comparison, an interactive emission
prediction tool, and SHAP-based explainability.

This app performs NO model training and NO SHAP computation. It only
loads artifacts already produced by:
    - data_preprocessing.py  (feature_names.csv, processed data)
    - model_training.py      (gradientboosting.pkl, model_comparison.csv)
    - shap_analysis.py       (shap_summary_beeswarm.png, shap_bar_importance.png)

Run with:
    streamlit run app.py

Author: Prepared for the Explainable Agri-food CO2 Forecasting MVP
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration constants
#
# Centralised at module level, matching the style of data_preprocessing.py,
# model_training.py, and shap_analysis.py elsewhere in this project, so
# every script in the repo is auditable the same way.
# ---------------------------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/Agrofood_co2_emission.csv")
FEATURE_NAMES_PATH = Path("data/processed/feature_names.csv")
MODEL_PATH = Path("models/gradientboosting.pkl")
MODEL_COMPARISON_PATH = Path("reports/model_comparison.csv")
SHAP_SUMMARY_PLOT_PATH = Path("reports/shap/shap_summary_beeswarm.png")
SHAP_BAR_PLOT_PATH = Path("reports/shap/shap_bar_importance.png")

MODEL_NAME = "Gradient Boosting"

# Valid year range this app allows for prediction input. 1990-2020 is
# the range the underlying data actually covers (see
# data_preprocessing.py's VALID_YEAR_MIN/MAX); years up to 2030 are
# still accepted here to allow limited forward-looking exploration, but
# are clearly labeled in the UI as extrapolation beyond the model's
# training and test range rather than presented as equally reliable.
MIN_INPUT_YEAR = 1990
MAX_TRAINED_YEAR = 2020
MAX_INPUT_YEAR = 2030

# The five approved numeric/categorical predictors this model was
# trained on (excluding Area, which is handled separately via one-hot
# encoding). Kept as a named constant, matching the allow-list pattern
# used in data_preprocessing.py, rather than inferring them implicitly
# from feature_names.csv each time.
NUMERIC_INPUT_COLUMNS = [
    "Year",
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


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_required_files() -> list[Path]:
    """
    Check that every artifact this app depends on actually exists on
    disk, before any page tries to load one.

    Returns a list of missing paths (empty if everything is present).
    Checking all of them up front, rather than letting each page fail
    independently when a user navigates to it, gives a single clear
    diagnostic instead of a different cryptic error per page.
    """
    required_paths = [
        RAW_DATA_PATH,
        FEATURE_NAMES_PATH,
        MODEL_PATH,
        MODEL_COMPARISON_PATH,
        SHAP_SUMMARY_PLOT_PATH,
        SHAP_BAR_PLOT_PATH,
    ]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        logger.error("Missing required artifact(s): %s", missing)
    return missing


# ---------------------------------------------------------------------------
# Cached loaders
#
# st.cache_data is used for plain data (DataFrames, lists) that Streamlit
# can hash and serialize. st.cache_resource is used for the model object,
# which is the documented Streamlit pattern for non-serializable resources
# (open connections, loaded ML models) that should be loaded once per
# session rather than re-read from disk on every widget interaction.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading raw dataset...")
def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the original FAO/IPCC CSV, used only for the Country dropdown
    and for pre-filling sensible default population values -- never fed
    directly to the model."""
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %s", df.shape)
    return df


@st.cache_data(show_spinner="Loading feature schema...")
def load_feature_names(path: Path) -> list[str]:
    """Load the exact, ordered feature list the model was trained on."""
    feature_names = pd.read_csv(path)["feature_name"].tolist()
    logger.info("Loaded %d feature name(s).", len(feature_names))
    return feature_names


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path: Path):
    """
    Load the trained Gradient Boosting model.

    Cached with st.cache_resource (not st.cache_data) because a fitted
    scikit-learn estimator is a live Python object, not plain data --
    this is the Streamlit-documented distinction between the two cache
    decorators.
    """
    model = joblib.load(path)
    logger.info("Loaded trained model from '%s'.", path)
    return model


@st.cache_data(show_spinner="Loading model comparison results...")
def load_model_comparison(path: Path) -> pd.DataFrame:
    """Load the model comparison table produced by model_training.py."""
    df = pd.read_csv(path, index_col="Model")
    logger.info("Loaded model comparison table: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature engineering for live prediction
#
# This mirrors, at inference time, the exact encoding scheme
# data_preprocessing.py applied at training time. It intentionally does
# NOT reimplement or import from data_preprocessing.py's pipeline
# functions, since those operate on a full historical dataframe (fit-time
# batch encoding); this instead builds a single-row feature vector
# directly against the already-fixed `feature_names` schema, which is
# the simpler and more robust approach for one live prediction request.
# ---------------------------------------------------------------------------

def validate_feature_alignment(model, feature_names: list[str]) -> None:
    """
    Confirm the loaded model actually expects the same feature schema
    saved in feature_names.csv, before any prediction is attempted.

    This guards against a stale or mismatched artifact pairing (e.g. an
    old feature_names.csv left over from a previous preprocessing run
    sitting alongside a newer retrained model) -- exactly the kind of
    silent mismatch that would otherwise produce a prediction without
    any error, just a wrong one.
    """
    if not hasattr(model, "n_features_in_"):
        # Some estimator types don't expose this attribute; skip the
        # check rather than fail on something outside this app's control.
        logger.warning("Loaded model has no n_features_in_ attribute; skipping alignment check.")
        return

    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: the loaded model expects "
            f"{model.n_features_in_} feature(s), but feature_names.csv "
            f"lists {len(feature_names)}. This usually means the model "
            "and feature_names.csv were produced by different runs of "
            "the pipeline. Re-run data_preprocessing.py and "
            "model_training.py together, then retry."
        )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_names:
            raise ValueError(
                "Feature name/order mismatch between the loaded model "
                "and feature_names.csv. Re-run data_preprocessing.py and "
                "model_training.py together to regenerate consistent "
                "artifacts before predicting."
            )

    logger.info("Feature alignment validated: model and feature_names.csv agree.")


def build_feature_vector(
    feature_names: list[str],
    country: str,
    year: int,
    rural_population: float,
    urban_population: float,
    male_population: float,
    female_population: float,
) -> pd.DataFrame:
    """
    Construct a single-row, model-ready feature vector from user inputs.

    Built directly against `feature_names` (rather than, say, one-hot
    encoding a fresh single-row DataFrame with pd.get_dummies) because
    pd.get_dummies on a single row would only ever produce ONE dummy
    column -- it has no way to know about the other 235 countries the
    model was trained on. Starting from a zero-filled row with every
    training-time column already present, then setting only the
    relevant ones, is the only correct way to reproduce the training-time
    schema for a single new observation.
    """
    area_column = f"Area_{country}"
    if area_column not in feature_names:
        raise ValueError(
            f"'{country}' does not have a corresponding one-hot column "
            f"('{area_column}') in the trained feature schema. This "
            "country may not have been present in the training data."
        )

    # Zero-filled row: every Area_* indicator starts at 0, which is the
    # correct default for every country except the one selected below.
    row = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)

    row.at[0, "Year"] = year
    row.at[0, "Rural population"] = rural_population
    row.at[0, "Urban population"] = urban_population
    row.at[0, "Total Population - Male"] = male_population
    row.at[0, "Total Population - Female"] = female_population
    row.at[0, area_column] = 1.0

    return row


def get_country_defaults(raw_df: pd.DataFrame, country: str) -> dict:
    """
    Pre-fill sensible default input values for a selected country, using
    its most recent available historical record.

    This is a UX convenience, not a modeling step: it saves the user
    from having to look up realistic population figures themselves, by
    defaulting to the country's latest known values, which they can
    then adjust. If a country has no rows at all (shouldn't happen given
    the dropdown is built from this same dataframe, but checked
    defensively), safe zero defaults are returned instead of raising,
    since this only affects form pre-fill, not correctness of an
    eventual prediction.
    """
    country_rows = raw_df[raw_df["Area"] == country].sort_values("Year")
    if country_rows.empty:
        logger.warning("No historical rows found for '%s'; using zero defaults.", country)
        return {
            "year": MAX_TRAINED_YEAR,
            "rural_population": 0.0,
            "urban_population": 0.0,
            "male_population": 0.0,
            "female_population": 0.0,
        }

    latest = country_rows.iloc[-1]
    return {
        "year": int(latest["Year"]),
        "rural_population": float(latest["Rural population"]),
        "urban_population": float(latest["Urban population"]),
        "male_population": float(latest["Total Population - Male"]),
        "female_population": float(latest["Total Population - Female"]),
    }


def predict_emission(model, feature_vector: pd.DataFrame) -> float:
    """Run the model's prediction on a single prepared feature vector."""
    prediction = model.predict(feature_vector)
    return float(prediction[0])


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_home_page(raw_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    st.title("Explainable Machine Learning for Forecasting Agri-food Carbon Emissions")
    st.caption("Using FAO/IPCC Environmental Data")

    st.markdown(
        """
        ### Project Overview
        This project builds an interpretable machine learning pipeline to
        forecast national agri-food sector carbon emissions from
        population and demographic indicators, using a country-year
        panel dataset assembled from FAO and IPCC sources. Rather than
        optimizing purely for predictive accuracy, the project treats
        model explainability as a first-class requirement: every
        prediction can be traced back to the feature contributions that
        produced it via SHAP (SHapley Additive exPlanations).

        ### Research Objective
        To what extent can agri-food sector carbon emissions be
        forecast from demographic drivers (population size, urban/rural
        distribution) alone, and which of these drivers matters most,
        and why? This framing deliberately excludes the emission-source
        columns that mechanically sum to the target (see **About** for
        the full leakage-avoidance rationale), so that any predictive
        signal found reflects a genuine demographic relationship rather
        than arithmetic.
        """
    )

    st.markdown("### Machine Learning Pipeline")
    pipeline_steps = pd.DataFrame({
        "Stage": [
            "1. Data Preprocessing",
            "2. Model Training",
            "3. Explainability",
            "4. Dashboard",
        ],
        "Description": [
            "Leakage-column removal, missing-value handling, one-hot encoding of country, temporal train/test split (1990-2015 / 2016-2020).",
            "Random Forest, Gradient Boosting, and XGBoost regressors trained and compared on R², MAE, RMSE, and timing.",
            "SHAP TreeExplainer applied to the best-performing model to generate global feature-importance explanations.",
            "This Streamlit app: performance comparison, live prediction, and SHAP visualization -- no retraining or recomputation.",
        ],
    })
    st.table(pipeline_steps)

    st.markdown("### Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries/Regions", f"{raw_df['Area'].nunique()}")
    col2.metric("Years Covered", f"{int(raw_df['Year'].min())}\u2013{int(raw_df['Year'].max())}")
    col3.metric("Total Rows", f"{len(raw_df):,}")
    col4.metric("Target Variable", "total_emission")

    st.markdown("### Best-Performing Model")
    best_model_name = comparison_df["R2"].idxmax()
    best_row = comparison_df.loc[best_model_name]
    st.success(
        f"**{best_model_name}** achieved the highest R\u00b2 "
        f"({best_row['R2']:.4f}) among the three models compared. "
        "See the **Model Performance** page for the full comparison."
    )


def render_model_performance_page(comparison_df: pd.DataFrame) -> None:
    st.title("Model Performance")
    st.markdown(
        "All three models were evaluated on the same held-out, "
        "temporally separated test set (2016\u20132020), never seen during training."
    )

    best_model_name = comparison_df["R2"].idxmax()
    best_row = comparison_df.loc[best_model_name]

    st.markdown(f"### Best Model: **{best_model_name}**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("R\u00b2", f"{best_row['R2']:.4f}")
    col2.metric("MAE", f"{best_row['MAE']:,.0f}")
    col3.metric("RMSE", f"{best_row['RMSE']:,.0f}")
    col4.metric("Training Time", f"{best_row['Training_Time_Seconds']:.2f}s")
    col5.metric("Prediction Time", f"{best_row['Prediction_Time_Seconds']:.4f}s")

    st.markdown("### Full Comparison")
    display_df = comparison_df.copy()
    display_df.columns = ["R\u00b2", "MAE", "RMSE", "Training Time (s)", "Prediction Time (s)"]
    st.dataframe(
        display_df.style.format({
            "R\u00b2": "{:.4f}",
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "Training Time (s)": "{:.2f}",
            "Prediction Time (s)": "{:.4f}",
        }).highlight_max(subset=["R\u00b2"], color="#d4f7dc"),
        use_container_width=True,
    )

    st.caption(
        "R\u00b2, MAE, and RMSE are computed on the test set. Training and "
        "prediction time are wall-clock measurements from the training run "
        "and are hardware-dependent, not intrinsic model properties."
    )


def render_prediction_page(
    raw_df: pd.DataFrame,
    feature_names: list[str],
    model,
) -> None:
    st.title("Emission Prediction")
    st.markdown(
        f"Predict total agri-food sector carbon emissions (kt CO\u2082-eq) "
        f"using the **{MODEL_NAME}** model."
    )

    st.info(
        "This model was trained on five approved, leakage-free predictors: "
        "**Area (country), Year, Rural population, Urban population, "
        "Total Population - Male, and Total Population - Female.** "
        "`Average Temperature \u00b0C` exists in the raw dataset but was "
        "deliberately excluded from the approved feature set during "
        "preprocessing -- it has no established causal role as a *driver* "
        "of a single country's agri-food emissions, so it is not collected "
        "as a prediction input here. See the **About** page for the full "
        "feature-selection rationale.",
        icon="\u2139\ufe0f",
    )

    countries = sorted(raw_df["Area"].unique().tolist())
    country = st.selectbox("Country", countries)

    defaults = get_country_defaults(raw_df, country)

    st.markdown(
        f"Input fields below are pre-filled with **{country}**'s most "
        f"recently recorded values (as of {defaults['year']}) as a "
        "starting point \u2014 adjust any of them to explore a different scenario."
    )

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input(
            "Year", min_value=MIN_INPUT_YEAR, max_value=MAX_INPUT_YEAR,
            value=min(max(defaults["year"], MIN_INPUT_YEAR), MAX_INPUT_YEAR), step=1,
        )
        rural_population = st.number_input(
            "Rural Population", min_value=0.0, value=defaults["rural_population"], step=1000.0,
        )
        urban_population = st.number_input(
            "Urban Population", min_value=0.0, value=defaults["urban_population"], step=1000.0,
        )
    with col2:
        male_population = st.number_input(
            "Male Population", min_value=0.0, value=defaults["male_population"], step=1000.0,
        )
        female_population = st.number_input(
            "Female Population", min_value=0.0, value=defaults["female_population"], step=1000.0,
        )

    if year > MAX_TRAINED_YEAR:
        st.warning(
            f"Year {int(year)} is beyond the model's training/test range "
            f"(1990\u2013{MAX_TRAINED_YEAR}). This prediction is an "
            "extrapolation and should be treated with reduced confidence.",
            icon="\u26a0\ufe0f",
        )

    if st.button("Predict Total Emission", type="primary"):
        try:
            validate_feature_alignment(model, feature_names)
            feature_vector = build_feature_vector(
                feature_names=feature_names,
                country=country,
                year=int(year),
                rural_population=rural_population,
                urban_population=urban_population,
                male_population=male_population,
                female_population=female_population,
            )
            prediction = predict_emission(model, feature_vector)
            logger.info(
                "Prediction made for %s, year %d: %.2f kt CO2-eq.",
                country, year, prediction,
            )
            st.metric("Predicted Total Emission", f"{prediction:,.2f} kt CO\u2082-eq")
        except ValueError as exc:
            logger.error("Prediction failed: %s", exc)
            st.error(f"Could not generate a prediction: {exc}")
        except Exception as exc:  # noqa: BLE001 -- surface any unexpected
                                   # failure to the user rather than crash
                                   # the app silently.
            logger.exception("Unexpected error during prediction.")
            st.error(f"An unexpected error occurred while predicting: {exc}")


def render_shap_page() -> None:
    st.title("SHAP Explainability")
    st.markdown(
        f"These visualizations were precomputed by `shap_analysis.py` "
        f"against the **{MODEL_NAME}** model's held-out test set "
        "predictions, using `shap.TreeExplainer`. They are loaded here "
        "as static images -- SHAP values are not recomputed by this app."
    )

    st.markdown("### Feature Importance (Bar Plot)")
    st.image(str(SHAP_BAR_PLOT_PATH), use_container_width=True)
    st.caption(
        "Ranks each feature by its **mean absolute SHAP value** across "
        "every test-set prediction -- a single number per feature "
        "representing average impact magnitude, regardless of direction. "
        "Longer bars indicate features that, on average, move the "
        "model's prediction further from the baseline."
    )

    st.markdown("### Summary Plot (Beeswarm)")
    st.image(str(SHAP_SUMMARY_PLOT_PATH), use_container_width=True)
    st.caption(
        "Each dot is one test-set prediction. Horizontal position shows "
        "that feature's SHAP value (impact on the predicted emission) "
        "for that specific row; color shows whether the feature's actual "
        "value was high (red) or low (blue). This reveals *direction*, "
        "which the bar plot above does not -- e.g. whether high urban "
        "population consistently pushes predicted emissions up, down, or "
        "depends on other context."
    )


def render_about_page() -> None:
    st.title("About This Project")

    st.markdown(
        """
        ### Project Description
        An end-to-end, explainability-first machine learning pipeline
        forecasting agri-food sector carbon emissions at the country-year
        level, built as a portfolio project demonstrating applied AI
        methodology relevant to Life Cycle Assessment (LCA) research.

        ### Dataset Source
        FAO (Food and Agriculture Organization of the United Nations) and
        IPCC (Intergovernmental Panel on Climate Change) agri-food CO\u2082
        emissions data, 1990\u20132020, distributed publicly via Kaggle
        (*Agrofood_co2_emission.csv*).

        ### Feature Selection & Leakage Avoidance
        The raw dataset includes ~23 emission-source columns (Crop
        Residues, Food Transport, Manure Management, etc.) that sum
        exactly to `total_emission` by construction. These were excluded
        entirely from the model's feature set, since using them would
        let the model "predict" the target by arithmetic rather than by
        learning a genuine relationship. The five approved predictors
        (Area, Year, Rural/Urban population, Male/Female population)
        contain no such mechanical relationship to the target.

        ### Algorithms Used
        - Random Forest Regressor
        - Gradient Boosting Regressor (selected as best-performing)
        - XGBoost Regressor

        ### Explainability
        SHAP (SHapley Additive exPlanations), via `shap.TreeExplainer`,
        applied to the best-performing model.

        ### Python Libraries
        `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `streamlit`,
        `matplotlib`, `joblib`

        ### Author
        Anupriyo Chakraborty

        
        """
    )


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Agri-food Carbon Emission Forecasting",
        page_icon="\U0001F33E",
        layout="wide",
    )

    missing_files = validate_required_files()
    if missing_files:
        st.error(
            "This app cannot start because the following required "
            "artifact(s) are missing:\n\n"
            + "\n".join(f"- `{p}`" for p in missing_files)
            + "\n\nRun `data_preprocessing.py`, `model_training.py`, and "
            "`shap_analysis.py` (in that order) to generate them before "
            "launching this app."
        )
        st.stop()

    try:
        raw_df = load_raw_dataset(RAW_DATA_PATH)
        feature_names = load_feature_names(FEATURE_NAMES_PATH)
        model = load_model(MODEL_PATH)
        comparison_df = load_model_comparison(MODEL_COMPARISON_PATH)
    except Exception as exc:  # noqa: BLE001 -- any load failure here is
                               # fatal to every page, so it is surfaced
                               # once, clearly, rather than per-page.
        logger.exception("Failed to load required artifacts.")
        st.error(f"Failed to load required project artifacts: {exc}")
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Model Performance", "Emission Prediction", "SHAP Explainability", "About"],
        label_visibility="collapsed",
    )

    if page == "Home":
        render_home_page(raw_df, comparison_df)
    elif page == "Model Performance":
        render_model_performance_page(comparison_df)
    elif page == "Emission Prediction":
        render_prediction_page(raw_df, feature_names, model)
    elif page == "SHAP Explainability":
        render_shap_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
