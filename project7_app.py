import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Hotel Cancellation Analysis Interface", layout="wide")
st.title("Hotel Cancellation Analysis Interface")

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load hotel bookings data.

    Priority:
      1) cleaned_output_2025-12-11_11-48-46.parquet
      2) hotel_bookings_clean.csv
      3) hotel_bookings_clean(1).csv

    If nothing is found, fall back to synthetic data.
    """
    df = None

    # ---- 1) Try PARQUET ----
    parquet_file = "cleaned_output_2025-12-11_11-48-46.parquet"
    if os.path.exists(parquet_file):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            st.warning(f"[Parquet load failed: {parquet_file}] {e}")

    # ---- 2) Try CSV files ----
    if df is None:
        for csv_file in ["hotel_bookings_clean.csv", "hotel_bookings_clean(1).csv"]:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    break
                except Exception as e:
                    st.warning(f"[CSV load failed: {csv_file}] {e}")

    # ---- 3) Synthetic fallback ----
    if df is None:
        st.warning(
            "⚠️ Real dataset not found — loading synthetic 2000-row dataset. "
            "Place cleaned_output_2025-12-11_11-48-46.parquet or "
            "hotel_bookings_clean.csv next to project7_app.py."
        )
        rng = np.random.default_rng(7)
        n = 2000
        df = pd.DataFrame({
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], size=n),
            "arrival_date_month": rng.choice(
                [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ],
                size=n
            ),
            "lead_time": rng.integers(0, 365, size=n),
            "adults": rng.integers(1, 4, size=n),
            "children": rng.integers(0, 3, size=n),
            "stays_in_weekend_nights": rng.integers(0, 3, size=n),
            "stays_in_week_nights": rng.integers(0, 5, size=n),
            "adr": np.abs(rng.normal(100, 35, size=n)).round(2),
            "deposit_type": rng.choice(["No Deposit", "Non Refund", "Refundable"], size=n),
            "customer_type": rng.choice(["Transient", "Group", "Contract"], size=n),
            "is_canceled": rng.integers(0, 2, size=n),
        })

    # ---- Cleaning Step ----
    # make sure key numeric columns are numeric
    numeric_cols = [
        "adults", "children", "babies", "lead_time",
        "stays_in_weekend_nights", "stays_in_week_nights",
        "adr", "is_canceled"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows missing important numeric values
    keep_for_dropna = [c for c in ["adults", "children", "adr"] if c in df.columns]
    if keep_for_dropna:
        df = df.dropna(subset=keep_for_dropna)

    # clip and cast integer counts
    for col in ["adults", "children", "babies"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).clip(lower=0)

    # ADR non-negative
    if "adr" in df.columns:
        df["adr"] = df["adr"].fillna(0)
        df = df[df["adr"] >= 0]

    # is_canceled must be 0 or 1
    if "is_canceled" in df.columns:
        df["is_canceled"] = df["is_canceled"].fillna(0).astype(int).clip(0, 1)

    return df.reset_index(drop=True)


df = load_data()

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")

# Reset button
if st.sidebar.button("🔄 Reset all filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Identify categorical and numeric columns
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ---- CATEGORICAL FILTERS ----
selected_cats = {}
pretty_cat_label = {
    "arrival_date_month": "Arrival month",
    "customer_type": "Customer type",
    "deposit_type": "Deposit type",
    "hotel": "Hotel",
}

for col in sorted(cat_cols):
    values = sorted(df[col].dropna().astype(str).unique())
    if not values:
        continue
    label = pretty_cat_label.get(col, col)
    selected = st.sidebar.multiselect(label, values, key=f"cat_{col}")
    if selected:
        selected_cats[col] = set(selected)

# ---- NUMERIC FILTERS ----
range_filters = {}
skip_numeric = {"is_canceled"}  # we'll control this with radio instead

pretty_num_label = {
    "lead_time": "Lead time (days)",
    "adr": "Average Daily Rate",
    "adults": "Adults",
    "children": "Children",
}

for col in num_cols:
    if col in skip_numeric:
        continue

    vmin, vmax = float(df[col].min()), float(df[col].max())
    label = pretty_num_label.get(col, col)

    # ADR: numeric inputs
    if col == "adr":
        st.sidebar.markdown("**Average Daily Rate (ADR)**")
        min_val = st.sidebar.number_input(
            "Min ADR", value=float(vmin), step=1.0, key="adr_min"
        )
        max_val = st.sidebar.number_input(
            "Max ADR", value=float(vmax), step=1.0, key="adr_max"
        )
        if max_val < min_val:
            max_val = min_val
        range_filters[col] = (min_val, max_val)
        continue

    # Lead time: numeric inputs
    if col == "lead_time":
        st.sidebar.markdown("**Lead time (days)**")
        min_val = st.sidebar.number_input(
            "Min lead time", value=float(vmin), step=1.0, key="lead_min"
        )
        max_val = st.sidebar.number_input(
            "Max lead time", value=float(vmax), step=1.0, key="lead_max"
        )
        if max_val < min_val:
            max_val = min_val
        range_filters[col] = (min_val, max_val)
        continue

    # Other numeric: sliders
    lo, hi = st.sidebar.slider(
        label,
        min_value=float(vmin),
        max_value=float(vmax),
        value=(float(vmin), float(vmax)),
        key=f"range_{col}",
    )
    range_filters[col] = (lo, hi)

# ---- CANCELLATION FILTER ----
cancel_filter = None
if "is_canceled" in df.columns:
    cancel_filter = st.sidebar.radio(
        "Cancellation status",
        options=["All", "Canceled", "Not canceled"],
        index=0,
        key="cancel_radio",
    )

# ---- SEARCH & SORT ----
query = st.sidebar.text_input("Search in text columns", key="search")
sort_col = st.sidebar.selectbox("Sort by column", options=list(df.columns), index=0)
ascending = st.sidebar.checkbox("Ascending sort", value=True, key="asc")

# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------
mask = pd.Series(True, index=df.index)

# Apply categorical filters
for col, values in selected_cats.items():
    mask &= df[col].astype(str).isin(values)

# Apply numeric filters
for col, (lo, hi) in range_filters.items():
    mask &= df[col].between(lo, hi)

# Apply cancellation filter
if cancel_filter == "Canceled":
    mask &= df["is_canceled"] == 1
elif cancel_filter == "Not canceled":
    mask &= df["is_canceled"] == 0

df_f = df.loc[mask].copy()

# Apply search
if query:
    search_mask = pd.Series(False, index=df_f.index)
    for col in cat_cols:
        search_mask |= df_f[col].astype(str).str.contains(query, case=False, na=False)
    df_f = df_f[search_mask]

# Apply sort
if sort_col in df_f.columns:
    df_f = df_f.sort_values(by=sort_col, ascending=ascending)

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
    c4.metric("Average Daily Rate", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average Daily Rate", "—")

st.divider()

# ------------------------------------------------------------
# VISUALISATIONS
# ------------------------------------------------------------
st.subheader("Visualizations")
charts = []

# --- Cancellation rate by month (handles full month names) ---
if {"arrival_date_month", "is_canceled"}.issubset(df_f.columns) and len(df_f):
    df_tmp = df_f.copy()

    # Convert full month names to month numbers for proper ordering
    try:
        df_tmp["_month_num"] = pd.to_datetime(
            df_tmp["arrival_date_month"], format="%B"
        ).dt.month
    except Exception:
        df_tmp["_month_num"] = pd.factorize(df_tmp["arrival_date_month"])[0] + 1

    tmp = (
        df_tmp.groupby(["_month_num", "arrival_date_month"])["is_canceled"]
        .mean()
        .reset_index()
        .sort_values("_month_num")
    )

    chart1 = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("arrival_date_month:N", title="Month"),
            y=alt.Y(
                "is_canceled:Q",
                title="Cancellation rate",
                axis=alt.Axis(format=".0%")
            ),
            tooltip=["arrival_date_month", "is_canceled"],
        )
        .properties(height=280, width="container")
    )
    charts.append(("Cancellation Rate by Month", chart1))

# --- Average ADR by hotel ---
if {"hotel", "adr"}.issubset(df_f.columns) and len(df_f):
    tmp2 = (
        df_f.groupby("hotel", dropna=False)["adr"]
        .mean()
        .reset_index()
        .sort_values("hotel")
    )

    chart2 = (
        alt.Chart(tmp2)
        .mark_bar()
        .encode(
            x=alt.X("hotel:N", title="Hotel"),
            y=alt.Y("adr:Q", title="Average Daily Rate"),
            tooltip=["hotel", "adr"],
        )
        .properties(height=280, width="container")
    )
    charts.append(("Average Daily Rate by Hotel", chart2))

# Display charts
if charts:
    cols = st.columns(len(charts))
    for col, (title, chart) in zip(cols, charts):
        with col:
            st.markdown(f"### {title}")
            st.altair_chart(chart)
else:
    st.info(
        "Adjust filters to display charts (need arrival_date_month, "
        "is_canceled, hotel, adr in the current filtered data)."
    )

st.divider()

# ------------------------------------------------------------
# DATA TABLE & EXPORT
# ------------------------------------------------------------
st.subheader("Data Table")

if len(df_f):
    st.dataframe(df_f, use_container_width=True, height=400)
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="filtered_output.csv",
        mime="text/csv",
    )
else:
    st.warning("No rows match the current filters. Try resetting or relaxing filters.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.caption("Run locally with:  `streamlit run project7_app.py`")
