import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------------------
#       STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Hotel Cancellation Analysis Interface", layout="wide")
st.title("Hotel Cancellation Analysis Interface")

# ------------------------------------------------------------
#       DATA LOADING
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Attempt to load the real dataset from local files.
    Priority:
        1) cleaned_output_2025-12-11_11-48-46.parquet
        2) hotel_bookings_clean.csv
    If both fail, fallback to synthetic 2,000 rows.
    """

    # ---- 1) Try PARQUET ----
    parquet_file = "cleaned_output_2025-12-11_11-48-46.parquet"
    if os.path.exists(parquet_file):
        try:
            df = pd.read_parquet(parquet_file)
            return df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"[Parquet load failed] {e}")

    # ---- 2) Try CSV ----
    csv_file = "hotel_bookings_clean.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"[CSV load failed] {e}")

    # ---- 3) Synthetic fallback ----
    st.warning(
        "⚠️ Real dataset not found. Loading synthetic 2,000-row dataset. "
        "Ensure cleaned_output_2025-12-11_11-48-46.parquet "
        "or hotel_bookings_clean(1).csv is in the same directory as project7_app.py."
    )

    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], size=n),
        "arrival_date_month": rng.choice(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
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
        "is_canceled": rng.integers(0, 2, size=n)
    })
    return df

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

# Column type detection
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ---- CATEGORY FILTERS ----
selected_cats = {}
label_map = {
    "arrival_date_month": "Arrival month",
    "customer_type": "Customer type",
    "deposit_type": "Deposit type",
    "hotel": "Hotel"
}

for c in sorted(cat_cols):
    values = sorted(df[c].dropna().unique().astype(str))
    label = label_map.get(c, c)
    selected = st.sidebar.multiselect(label, values, key=f"cat_{c}")
    if selected:
        selected_cats[c] = set(selected)

# ---- NUMERIC FILTERS ----
range_filters = {}
skip_cols = {"is_canceled"}

pretty_label = {
    "lead_time": "Lead time (days)",
    "adr": "Average Daily Rate",
    "adults": "Adults",
    "children": "Children"
}

for c in num_cols:
    if c in skip_cols:
        continue

    vmin, vmax = float(df[c].min()), float(df[c].max())
    label = pretty_label.get(c, c)

    # ADR manual input
    if c == "adr":
        minv = st.sidebar.number_input("Min ADR", value=vmin, step=1.0)
        maxv = st.sidebar.number_input("Max ADR", value=vmax, step=1.0)
        if maxv < minv:
            maxv = minv
        range_filters[c] = (minv, maxv)
        continue

    # Lead time manual input
    if c == "lead_time":
        minv = st.sidebar.number_input("Min lead time", value=vmin, step=1.0)
        maxv = st.sidebar.number_input("Max lead time", value=vmax, step=1.0)
        if maxv < minv:
            maxv = minv
        range_filters[c] = (minv, maxv)
        continue

    # Normal slider
    lo, hi = st.sidebar.slider(
        label, min_value=vmin, max_value=vmax, value=(vmin, vmax), key=f"range_{c}"
    )
    range_filters[c] = (lo, hi)

# ---- CANCELLATION FILTER ----
cancel_filter = None
if "is_canceled" in df.columns:
    cancel_filter = st.sidebar.radio(
        "Cancellation",
        options=["All", "Canceled", "Not canceled"],
        index=0
    )

# ---- SEARCH & SORT ----
query = st.sidebar.text_input("Search (string columns)")
sort_col = st.sidebar.selectbox("Sort by", df.columns)
ascending = st.sidebar.checkbox("Ascending", True)

# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------
mask = pd.Series(True, index=df.index)

# Category filters
for col, vals in selected_cats.items():
    mask &= df[col].astype(str).isin(vals)

# Numeric filters
for col, (lo, hi) in range_filters.items():
    mask &= df[col].between(lo, hi)

# Cancellation filter
if cancel_filter == "Canceled":
    mask &= df["is_canceled"] == 1
elif cancel_filter == "Not canceled":
    mask &= df["is_canceled"] == 0

df_f = df[mask].copy()

# Text search
if query:
    search_mask = pd.Series(False, index=df_f.index)
    for col in cat_cols:
        search_mask |= df_f[col].astype(str).str.contains(query, case=False, na=False)
    df_f = df_f[search_mask]

# Sorting
df_f = df_f.sort_values(by=sort_col, ascending=ascending)

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", len(df_f.columns))
c3.metric("Cancellation rate", f"{df_f['is_canceled'].mean() * 100:.1f}%" if "is_canceled" in df_f else "—")
c4.metric("Average Daily Rate", f"{df_f['adr'].mean():.2f}" if "adr" in df_f else "—")

st.divider()

# ------------------------------------------------------------
# VISUALISATIONS
# ------------------------------------------------------------
st.subheader("Visualizations")

charts = []

# Cancellation rate by month
if {"arrival_date_month", "is_canceled"}.issubset(df_f.columns):
    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    tmp = df_f.groupby("arrival_date_month")["is_canceled"].mean().reindex(order).reset_index().dropna()

    chart1 = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("arrival_date_month:N", title="Month"),
            y=alt.Y("is_canceled:Q", title="Cancellation rate", axis=alt.Axis(format=".0%")),
            tooltip=["arrival_date_month", "is_canceled"]
        )
        .properties(height=280, width="container")
    )
    charts.append(("Cancellation Rate by Month", chart1))

# ADR by hotel
if {"hotel", "adr"}.issubset(df_f.columns):
    tmp2 = df_f.groupby("hotel")["adr"].mean().reset_index()

    chart2 = (
        alt.Chart(tmp2)
        .mark_bar()
        .encode(
            x=alt.X("hotel:N", title="Hotel"),
            y=alt.Y("adr:Q", title="Average ADR"),
            tooltip=["hotel", "adr"]
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
    st.info("Adjust filters to display charts.")

st.divider()

# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------
st.subheader("Data Table")

if len(df_f):
    st.dataframe(df_f, use_container_width=True, height=400)
    st.download_button(
        "Download filtered CSV",
        df_f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_output.csv",
        mime="text/csv",
    )
else:
    st.warning("No rows match the current filters.")

st.caption("Run with:  `streamlit run project7_app.py`")
