import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Hotel Cancellation Analysis", layout="wide")
st.title("Hotel Cancellation Analysis")

# ------------------------------------------------------------
# HELPERS: LOAD DATA & MODEL
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the hotel bookings dataset from the most likely paths.
    """
    candidates = [
        # Parquet
        "cleaned_output_2025-12-11_11-48-46.parquet",
        "sample_data/cleaned_output_2025-12-11_11-48-46.parquet",
        # CSVs
        "df_ready_processed.csv",
        "sample_data/df_ready_processed.csv",
        "hotel_bookings_clean.csv",
        "sample_data/hotel_bookings_clean.csv",
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            try:
                if path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                st.info(f"Loaded data from **{path}**")
                break
            except Exception as e:
                st.warning(f"Could not read {path}: {e}")

    if df is None:
        st.error("No dataset found. Please upload a CSV/Parquet next to this app.")
        st.stop()

    # light cleaning
    if "is_canceled" in df.columns:
        df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    for col in ["adr", "lead_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


@st.cache_resource
def load_model():
    """
    Load the trained RandomForest pipeline.
    """
    candidates = [
        "rf_booking_pipeline_2025-12-11_11-53-26.joblib",
        "sample_data/rf_booking_pipeline_2025-12-11_11-53-26.joblib",
    ]
    model = None
    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.info(f"Loaded model from **{path}**")
                break
            except Exception as e:
                st.warning(f"Could not load model {path}: {e}")

    if model is None:
        st.error("Trained model (.joblib) not found. Put it next to this app.")
        st.stop()

    # feature names (for building X correctly)
    required = getattr(model, "feature_names_in_", None)
    if required is None:
        st.error("Model has no feature_names_in_. It must be trained on a pandas DataFrame.")
        st.stop()

    return model, list(required)


df = load_data()
model, REQUIRED = load_model()

# ------------------------------------------------------------
# SIDEBAR – LIGHT FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Simple key filters only
if "country" in df.columns:
    country = st.sidebar.selectbox(
        "Country", ["All"] + sorted(df["country"].dropna().unique().tolist())
    )
else:
    country = "All"

if "hotel" in df.columns:
    hotel = st.sidebar.selectbox(
        "Hotel", ["All"] + sorted(df["hotel"].dropna().unique().tolist())
    )
else:
    hotel = "All"

if "arrival_date_month" in df.columns:
    month = st.sidebar.selectbox(
        "Arrival month", ["All"] + sorted(df["arrival_date_month"].dropna().unique().tolist())
    )
else:
    month = "All"

# Lead time & ADR sliders (if present)
lead_range = None
adr_range = None

if "lead_time" in df.columns:
    lt_min, lt_max = float(df["lead_time"].min()), float(df["lead_time"].max())
    lead_range = st.sidebar.slider(
        "Lead time (days)",
        min_value=lt_min,
        max_value=lt_max,
        value=(lt_min, lt_max),
    )

if "adr" in df.columns:
    adr_min, adr_max = float(df["adr"].min()), float(df["adr"].max())
    adr_range = st.sidebar.slider(
        "Average Daily Rate (ADR)",
        min_value=adr_min,
        max_value=adr_max,
        value=(adr_min, adr_max),
    )

# Cancellation status filter
cancel_filter = None
if "is_canceled" in df.columns:
    cancel_filter = st.sidebar.radio(
        "Cancellation status", ["All", "Canceled", "Not canceled"], index=0
    )

# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------
mask = pd.Series(True, index=df.index)

if country != "All" and "country" in df.columns:
    mask &= df["country"] == country

if hotel != "All" and "hotel" in df.columns:
    mask &= df["hotel"] == hotel

if month != "All" and "arrival_date_month" in df.columns:
    mask &= df["arrival_date_month"] == month

if lead_range and "lead_time" in df.columns:
    lo, hi = lead_range
    mask &= df["lead_time"].between(lo, hi)

if adr_range and "adr" in df.columns:
    lo, hi = adr_range
    mask &= df["adr"].between(lo, hi)

if cancel_filter == "Canceled":
    mask &= df["is_canceled"] == 1
elif cancel_filter == "Not canceled":
    mask &= df["is_canceled"] == 0

df_f = df.loc[mask].copy()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", len(df_f.columns))

if "is_canceled" in df_f.columns and len(df_f):
    c3.metric("Cancellation rate", f"{df_f['is_canceled'].mean() * 100:.1f}%")
else:
    c3.metric("Cancellation rate", "—")

if "adr" in df_f.columns and len(df_f):
    c4.metric("Average ADR", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average ADR", "—")

st.markdown("---")

# ------------------------------------------------------------
# VISUALS (SIMPLE)
# ------------------------------------------------------------
st.subheader("Visualisations")

charts = []

# Cancellation rate by month
if {"arrival_date_month", "is_canceled"}.issubset(df_f.columns) and len(df_f):
    tmp = df_f.copy()
    # sort by real month order if full names are used
    try:
        tmp["_month_num"] = pd.to_datetime(tmp["arrival_date_month"], format="%B").dt.month
    except Exception:
        tmp["_month_num"] = pd.factorize(tmp["arrival_date_month"])[0] + 1

    tmp = (
        tmp.groupby(["_month_num", "arrival_date_month"])["is_canceled"]
        .mean()
        .reset_index()
        .sort_values("_month_num")
    )

    ch1 = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("arrival_date_month:N", title="Month"),
            y=alt.Y("is_canceled:Q", title="Cancellation rate", axis=alt.Axis(format=".0%")),
            tooltip=["arrival_date_month", "is_canceled"],
        )
        .properties(height=260)
    )
    charts.append(("Cancellation Rate by Month", ch1))

# ADR by hotel
if {"hotel", "adr"}.issubset(df_f.columns) and len(df_f):
    tmp2 = (
        df_f.groupby("hotel", dropna=False)["adr"]
        .mean()
        .reset_index()
        .sort_values("hotel")
    )
    ch2 = (
        alt.Chart(tmp2)
        .mark_bar()
        .encode(
            x=alt.X("hotel:N", title="Hotel"),
            y=alt.Y("adr:Q", title="Average ADR"),
            tooltip=["hotel", "adr"],
        )
        .properties(height=260)
    )
    charts.append(("Average ADR by Hotel", ch2))

if charts:
    cols = st.columns(len(charts))
    for col, (title, ch) in zip(cols, charts):
        with col:
            st.markdown(f"### {title}")
            st.altair_chart(ch, use_container_width=True)
else:
    st.info("Not enough data columns to draw charts (need arrival_date_month, is_canceled, hotel, adr).")

st.markdown("---")

# ------------------------------------------------------------
# SIMPLE DATA TABLE
# ------------------------------------------------------------
st.subheader("Sample of filtered data")
st.dataframe(df_f.head(200), use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# ML PREDICTION SECTION
# ------------------------------------------------------------
st.subheader("🔮 Cancellation Prediction")

st.markdown(
    "Select a booking from the filtered data and let the trained model "
    "predict whether it will be **cancelled** or **not cancelled**."
)

if len(df_f) == 0:
    st.warning("No rows available after filtering. Remove some filters to enable prediction.")
else:
    # let user pick a row by index
    idx = st.number_input(
        "Choose row index from filtered data",
        min_value=int(df_f.index.min()),
        max_value=int(df_f.index.max()),
        value=int(df_f.index.min()),
        step=1,
    )

    if st.button("Predict cancellation for this booking"):
        try:
            row = df_f.loc[idx]

            # build feature vector in the exact order the model expects
            missing = [c for c in REQUIRED if c not in row.index]
            if missing:
                st.error(f"Dataset is missing required features for the model: {missing}")
            else:
                X = pd.DataFrame([row[REQUIRED].values], columns=REQUIRED)

                pred = int(model.predict(X)[0])
                proba = float(model.predict_proba(X)[0][1])  # prob of class '1' (canceled)

                # map 0/1 to Yes/No text (1 = will cancel, 0 = will not cancel)
                if pred == 1:
                    label = "YES – the booking is likely to be **CANCELLED**."
                else:
                    label = "NO – the booking is likely to **NOT be cancelled**."

                st.success(label)
                st.write(f"Estimated probability of cancellation: **{proba:.3f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Model: RandomForest pipeline loaded from rf_booking_pipeline_2025-12-11_11-53-26.joblib")
