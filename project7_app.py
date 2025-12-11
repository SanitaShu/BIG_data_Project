import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Hotel Cancellation Analysis Interface", layout="wide")
st.title("Hotel Cancellation Analysis Interface")


# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    """
    Load hotel bookings data. If a real CSV is available, use it.
    Otherwise generate a synthetic demo dataset with a similar schema.
    """
    candidates = [
        "hotel_bookings_clean.csv",                 # your main file
        "cleaned_output.csv",
        "cleaned_output_2025-11-10_18-38-15.csv",
        "df_ready_processed.csv",
        "hotel_bookings.csv",
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                pass

    # Fallback synthetic data (only if no real file was found)
    if df is None:
        rng = np.random.default_rng(7)
        n = 2000
        df = pd.DataFrame({
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], size=n),
            "arrival_date_year": rng.integers(2015, 2018, size=n),
            "arrival_date_month": rng.integers(1, 13, size=n),
            "lead_time": rng.integers(0, 365, size=n),
            "adults": rng.integers(1, 4, size=n),
            "children": rng.integers(0, 3, size=n),
            "stays_in_weekend_nights": rng.integers(0, 3, size=n),
            "stays_in_week_nights": rng.integers(0, 5, size=n),
            "adr": np.abs(rng.normal(100, 35, size=n)).round(2),
            "deposit_type": rng.choice(["No Deposit", "Non Refund", "Refundable"], size=n),
            "customer_type": rng.choice(["Transient", "Group", "Contract"], size=n),
            "distribution_channel": rng.choice(["Direct", "TA/TO", "Corporate"], size=n),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Groups"], size=n),
            "country": rng.choice(["PRT", "GBR", "ESP", "USA", "LTU"], size=n),
            "agent": rng.integers(1, 400, size=n),
            "is_canceled": rng.integers(0, 2, size=n),  # 0 = not canceled, 1 = canceled
        })

    # Basic cleaning
    for c in ["adults", "children", "lead_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")

    # drop rows with missing key values
    keep_cols = [c for c in ["adults", "children", "adr"] if c in df.columns]
    if keep_cols:
        df = df.dropna(subset=keep_cols)

    if "adults" in df.columns:
        df["adults"] = df["adults"].astype(int).clip(lower=0)
    if "children" in df.columns:
        df["children"] = df["children"].astype(int).clip(lower=0)
    if "adr" in df.columns:
        df = df[df["adr"] >= 0]

    return df.reset_index(drop=True)


df = load_data()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filters")

# Global reset
if st.sidebar.button("🔄 Reset all filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Split categorical / numeric
cat_cols = [
    c for c in df.columns
    if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
]
num_cols = [
    c for c in df.columns
    if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
]

# ---- CATEGORICAL FILTERS ----
selected_cats = {}
for c in sorted(cat_cols):
    vals = sorted(map(str, df[c].dropna().unique().tolist()))
    if not vals:
        continue

    label_map = {
        "arrival_date_month": "Arrival month",
        "arrival_date_year": "Arrival year",
        "customer_type": "Customer type",
        "deposit_type": "Deposit type",
        "hotel": "Hotel",
        "country": "Country",
        "distribution_channel": "Distribution channel",
        "market_segment": "Market segment",
        "assigned_room_type": "Assigned room type",
        "agent": "Agent",
        "lead_bucket": "Lead-time bucket",
    }
    label = label_map.get(c, c.replace("_", " ").title())

    sel = st.sidebar.multiselect(label, vals, key=f"cat_{c}")
    if sel:
        selected_cats[c] = set(sel)

# ---- NUMERIC FILTERS ----
range_filters = {}
skip_numeric = {"stays_in_week_nights", "stays_in_weekend_nights", "is_canceled"}

pretty_label = {
    "adr": "Average Daily Rate",
    "adults": "Adults",
    "children": "Children",
    "lead_time": "Stay duration (days)",
}

lead_min = int(df["lead_time"].min()) if "lead_time" in df.columns else 0
lead_max = int(df["lead_time"].max()) if "lead_time" in df.columns else 365

for c in sorted(num_cols):
    if c in skip_numeric:
        continue

    # --- ADR manual inputs ---
    if c == "adr":
        st.sidebar.markdown("**Average Daily Rate**")
        adr_min = st.sidebar.number_input(
            "Min ADR",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=1.0,
            key="adr_min",
        )
        adr_max = st.sidebar.number_input(
            "Max ADR",
            min_value=0.0,
            max_value=500.0,
            value=500.0,
            step=1.0,
            key="adr_max",
        )
        if adr_max < adr_min:
            adr_max = adr_min
        range_filters["adr"] = (adr_min, adr_max)
        continue

    # --- lead_time manual inputs ---
    if c == "lead_time":
        st.sidebar.markdown("**Stay duration (days)**")
        lt_min = st.sidebar.number_input(
            "Min duration",
            min_value=float(lead_min),
            max_value=float(lead_max),
            value=float(lead_min),
            step=1.0,
            key="lt_min",
        )
        lt_max = st.sidebar.number_input(
            "Max duration",
            min_value=float(lead_min),
            max_value=float(lead_max),
            value=float(lead_max),
            step=1.0,
            key="lt_max",
        )
        if lt_max < lt_min:
            lt_max = lt_min
        range_filters["lead_time"] = (lt_min, lt_max)
        continue

    # --- other numeric sliders ---
    vmin = float(np.nanmin(df[c].values))
    vmax = float(np.nanmax(df[c].values))
    label = pretty_label.get(c, f"{c} range".replace("_", " ").title())

    if c in {"adults", "children"}:
        if c == "adults":
            vmin, vmax = 1, 4
        if c == "children":
            vmin, vmax = 0, 7

        lo, hi = st.sidebar.slider(
            label,
            min_value=int(vmin),
            max_value=int(vmax),
            value=(int(vmin), int(vmax)),
            step=1,
            key=f"num_{c}",
        )
    else:
        lo, hi = st.sidebar.slider(
            label,
            min_value=float(vmin),
            max_value=float(vmax),
            value=(float(vmin), float(vmax)),
            key=f"num_{c}",
        )

    range_filters[c] = (lo, hi)

# ---- CANCELLATION FILTER ----
cancel_option = None
if "is_canceled" in df.columns:
    cancel_option = st.sidebar.radio(
        "Cancellation",
        options=["All", "Not canceled", "Canceled"],
        index=0,
        key="cancel_radio"
    )

# ---- SEARCH & SORT ----
query = st.sidebar.text_input("Search text (string columns)", key="q")
sort_col = st.sidebar.selectbox("Sort by", options=list(df.columns), index=0, key="sort_col")
ascending = st.sidebar.checkbox("Ascending", value=True, key="asc")

# ----------------- APPLY FILTERS -----------------
mask = pd.Series(True, index=df.index)

for c, vals in selected_cats.items():
    mask &= df[c].astype(str).isin(vals)

for c, (lo, hi) in range_filters.items():
    if c in {"adults", "children"}:
        mask &= df[c].between(int(lo), int(hi))
    else:
        mask &= df[c].between(lo, hi)

if cancel_option and cancel_option != "All":
    if cancel_option == "Canceled":
        mask &= df["is_canceled"] == 1
    elif cancel_option == "Not canceled":
        mask &= df["is_canceled"] == 0

df_f = df.loc[mask].copy()

if query and cat_cols:
    contains = pd.Series(False, index=df_f.index)
    for c in cat_cols:
        contains |= df_f[c].astype(str).str.contains(query, case=False, na=False)
    df_f = df_f[contains]

if sort_col in df_f.columns:
    df_f = df_f.sort_values(by=sort_col, ascending=ascending)

# ----------------- KPIs -----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", f"{df.shape[1]}")

if "is_canceled" in df_f.columns and len(df_f):
    c3.metric("Cancellation rate", f"{df_f['is_canceled'].mean() * 100:.1f}%")
else:
    c3.metric("Cancellation rate", "—")

if "adr" in df_f.columns and len(df_f):
    c4.metric("Average Daily Rate", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average Daily Rate", "—")

st.divider()

# ----------------- VISUALISATIONS -----------------
st.subheader("Visualizations")

charts = []

if "arrival_date_month" in df_f.columns and len(df_f):

    tmp = df_f.copy()

    # map month to names, whether numeric or already string
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    if pd.api.types.is_numeric_dtype(tmp["arrival_date_month"]):
        month_map = {i + 1: name for i, name in enumerate(month_order)}
        tmp["month_name"] = tmp["arrival_date_month"].astype(int).map(month_map)
    else:
        tmp["month_name"] = tmp["arrival_date_month"].astype(str)

    # --- 1) Cancellation rate by month ---
    if "is_canceled" in tmp.columns:
        canc = (
            tmp.groupby("month_name")["is_canceled"]
            .mean()
            .reset_index()
        )
        canc["month_name"] = pd.Categorical(
            canc["month_name"], categories=month_order, ordered=True
        )
        canc = canc.sort_values("month_name").dropna()

        if len(canc):
            chart1 = alt.Chart(canc).mark_bar().encode(
                x=alt.X("month_name:N", title="Month"),
                y=alt.Y(
                    "is_canceled:Q",
                    title="Cancellation rate",
                    axis=alt.Axis(format=".0%")
                ),
                tooltip=["month_name", "is_canceled"]
            ).properties(height=240)

            charts.append(("Cancellation Rate by Month", chart1))

    # --- 2) Average Daily Rate by month ---
    if "adr" in tmp.columns:
        adr = (
            tmp.groupby("month_name")["adr"]
            .mean()
            .reset_index()
        )
        adr["month_name"] = pd.Categorical(
            adr["month_name"], categories=month_order, ordered=True
        )
        adr = adr.sort_values("month_name").dropna()

        if len(adr):
            chart2 = alt.Chart(adr).mark_bar().encode(
                x=alt.X("month_name:N", title="Month"),
                y=alt.Y("adr:Q", title="Average Daily Rate"),
                tooltip=["month_name", "adr"]
            ).properties(height=240)

            charts.append(("Average Daily Rate by Month", chart2))

# Render charts side by side
if charts:
    cols = st.columns(len(charts))
    for col, (title, ch) in zip(cols, charts):
        with col:
            st.markdown(f"### **{title}**")
            st.altair_chart(ch, use_container_width=True)
else:
    st.info(
        "Adjust filters to display charts. "
        "(Need columns like 'arrival_date_month', 'is_canceled', 'adr'.)"
    )

st.divider()

# ----------------- DATA TABLE & EXPORT -----------------
st.subheader("Data Table")
if len(df_f):
    st.dataframe(df_f, use_container_width=True, height=380)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        csv,
        file_name="filtered_output.csv",
        mime="text/csv"
    )
else:
    st.warning("No rows match the current filters. Use 'Reset all filters' or relax constraints.")

# ----------------- FILTER STATE EXPORT -----------------
st.sidebar.download_button(
    "Download current filter state (JSON)",
    data=json.dumps({
        "selected_cats": {k: list(v) for k, v in selected_cats.items()},
        "range_filters": range_filters,
        "query": query,
        "sort_by": sort_col,
        "ascending": ascending,
        "cancellation_filter": cancel_option,
    }, indent=2),
    file_name="ui_state.json",
    mime="application/json",
)

st.caption("Run with:  `streamlit run project7_app.py`")
