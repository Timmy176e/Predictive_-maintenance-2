"""
Predictive Maintenance Dashboard - Streamlit app.

Upload a CSV of current asset readings; the app runs the 4 trained models and
reports, per asset: recommended maintenance type, failure risk score,
remaining useful life, and the predicted next maintenance due date.

The 4 model files are too large to keep in the Git repo, so they're hosted on
Google Drive and downloaded into a local `models/` folder the first time the
app starts (then cached for the life of the running app).

Run locally:   streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Google Drive file IDs for each model (the id between /d/ and /view in the share link).
# Replace these if you re-upload a retrained model to Drive later.
MODEL_FILES = {
    "type": ("1yF-zgkKA_fX9qPkVeMRuAL6N9EafVlwL", "maintenance_type_classifier.joblib"),
    "risk": ("1oNpbkDPf8W5LlWumroQlBWmHgiy6K0mo", "failure_risk_regressor.joblib"),
    "rul":  ("1BjAYSTWLqKSktTWri1mNzxSIF0mMHbcW", "rul_regressor.joblib"),
    "due":  ("1wqk3O2bw_cTIZQw9leNhoTQ3NSv4ZxV9", "days_until_due_regressor.joblib"),
}

NUMERIC_FEATURES = [
    "age_days", "fault_count_last_30d", "fault_count_last_90d",
    "downtime_minutes_last_90d", "days_since_last_maintenance", "degradation_index",
]
CATEGORICAL_FEATURES = ["comp_type", "section", "last_maintenance_type", "line"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OPTIONAL_ID_COLS = ["component_id", "area_code", "component_name"]
DATE_COL = "snapshot_date"

TYPE_COLORS = {"Corrective": "#f8d7da", "Predictive": "#fff3cd", "Preventive": "#d4edda"}

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide", page_icon="🔧")


@st.cache_resource
def load_models():
    models = {}
    for key, (file_id, filename) in MODEL_FILES.items():
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            gdown.download(id=file_id, output=str(local_path), quiet=False)
        models[key] = joblib.load(local_path)
    return models


def run_predictions(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Validate columns, run all 4 models, and return df with prediction columns added."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.copy()
    df["line"] = df["line"].astype(str)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    else:
        df[DATE_COL] = pd.Timestamp(datetime.today().date())

    X = df[FEATURE_COLS]

    df["predicted_maintenance_type"] = models["type"].predict(X)
    df["predicted_failure_risk_score"] = np.round(models["risk"].predict(X).astype(float), 3)
    df["predicted_remaining_useful_life_days"] = np.round(models["rul"].predict(X)).astype(int)
    days_until_due = np.round(models["due"].predict(X)).astype(int)
    df["predicted_days_until_due"] = days_until_due
    df["predicted_next_maintenance_due_date"] = df[DATE_COL] + pd.to_timedelta(days_until_due, unit="D")
    return df


def highlight_type(val):
    return f"background-color: {TYPE_COLORS.get(val, '')}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔧 Predictive Maintenance Dashboard")
st.caption(
    "Upload a CSV of current asset readings to get a recommended maintenance type, "
    "failure risk score, remaining useful life, and next maintenance due date per asset."
)

try:
    with st.spinner("Loading models (first run downloads them from Google Drive)..."):
        models = load_models()
except Exception as e:
    st.error("Failed to load models.")
    st.exception(e)
    st.stop()

with st.expander("Expected CSV columns", expanded=False):
    st.markdown(f"**Required:** `{'`, `'.join(FEATURE_COLS)}`")
    st.markdown(f"**Optional** (carried through for labeling): `{'`, `'.join(OPTIONAL_ID_COLS + [DATE_COL])}`")
    st.markdown(
        "- `comp_type`: Mixer, Forming, Oven, Packaging, Palletizer, Laser\n"
        "- `last_maintenance_type`: Install, Preventive, Corrective\n"
        f"- `{DATE_COL}`: when the readings were taken (if omitted, today's date is used)"
    )
    sample_path = Path(__file__).parent / "sample_input.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button("Download a sample CSV", f, file_name="sample_input.csv", mime="text/csv")


uploaded = st.file_uploader(
    "Upload asset data (CSV)",
    type=["csv"],
    key="maintenance_csv"
)

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)

    if raw_df.empty:
        st.error("The uploaded CSV contains no data.")
        st.stop()

    st.success("CSV loaded successfully.")
    st.write("Dataset shape:", raw_df.shape)
    st.dataframe(raw_df.head())

    results = run_predictions(raw_df, models)

except ValueError as e:
    st.error(f"Input validation failed: {e}")
    st.stop()

except Exception as e:
    st.error(f"Couldn't process the uploaded file: {e}")
    st.stop()
    

# ---- summary cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets analyzed", len(results))
c2.metric("Needing Corrective action", int((results["predicted_maintenance_type"] == "Corrective").sum()))
c3.metric("Average failure risk", f"{results['predicted_failure_risk_score'].mean():.2f}")
c4.metric("Soonest due date", results["predicted_next_maintenance_due_date"].min().date().isoformat())

# ---- results table ----
st.subheader("Predictions")
display_cols = (
    [c for c in OPTIONAL_ID_COLS if c in results.columns]
    + [DATE_COL, "predicted_maintenance_type", "predicted_failure_risk_score",
       "predicted_remaining_useful_life_days", "predicted_next_maintenance_due_date"]
)
table = results[display_cols].sort_values("predicted_next_maintenance_due_date").reset_index(drop=True)

st.dataframe(
    table.style.map(highlight_type, subset=["predicted_maintenance_type"]),
    use_container_width=True,
    height=min(80 + 35 * len(table), 500),
)

st.download_button(
    "Download predictions as CSV",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="maintenance_predictions.csv",
    mime="text/csv",
)

# ---- breakdown chart ----
st.subheader("Maintenance type breakdown")
st.bar_chart(results["predicted_maintenance_type"].value_counts())

st.caption(
    "Models are Random Forests trained on a synthetic dataset built to mirror this plant's real "
    "shift maintenance reports - treat outputs as a proof-of-concept until retrained on real "
    "historical maintenance records."
)
