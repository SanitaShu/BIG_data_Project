import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Hotel Cancellation Analysis Interface", layout="wide")
st.title("Hotel Cancellation Analysis Interface")


# Data Loading
@st.cache_data
def load_data():
    # Try to load a real file; otherwise synthesise a small demo set
    candidates = [
        "cleaned_output_2025-11-10_18-38-15.csv",
        "cleaned_output.csv",
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

    # Fallback synthetic data with the SAME schema as the real dataset
    if df is None:
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
            "customer_type": rng.choice(["Transient","Group","Contract"], size=n),
            "is_canceled": rng.integers(0, 2, size=n)  # 0 or 1
        })

    # Cleaning rules
    for c in ["adults", "children"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")

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

# Sidebar Controls
st.sidebar.header("Filters")

# Global reset
if st.sidebar.button("🔄 Reset all filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Categorical / numeric columns
cat_cols = [c for c in df.columns
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
num_cols = [c for c in df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]

selected_cats = {}
for c in sorted(cat_cols)[:6]:
    vals = sorted(map(str, df[c].dropna().unique().tolist()))
    if vals:
        sel = st.sidebar.multiselect(c, vals, key=f"cat_{c}")
        if sel:
            selected_cats[c] = set(sel)

# Numeric ranges; integer step for count-like columns
range_filters = {}
count_like = {
    "adults", "children", "week_nights", "weekend_nights",
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights"
}
for c in sorted(num_cols)[:8]:
    vmin = float(np.nanmin(df[c].values))
    vmax = float(np.nanmax(df[c].values))
    is_count = c.lower() in count_like
    if is_count:
        lo, hi = st.sidebar.slider(
            f"{c} range",
            min_value=int(vmin),
            max_value=int(vmax),
            value=(int(vmin), int(vmax)),
            step=1,
            key=f"num_{c}",
        )
    else:
        lo, hi = st.sidebar.slider(
            f"{c} range",
            min_value=vmin,
            max_value=vmax,
            value=(vmin, vmax),
            key=f"num_{c}",
        )
    range_filters[c] = (lo, hi)

# Keep ADR > 0 (already mostly cleaned; toggle kept for clarity)
only_pos_adr = st.sidebar.checkbox("ADR > 0 only", value=True)
if "adr" in df.columns and only_pos_adr:
    df = df[df["adr"] > 0]

# Text search & sorting
query = st.sidebar.text_input("Search text (string columns)", key="q")
sort_col = st.sidebar.selectbox("Sort by", options=list(df.columns), index=0, key="sort_col")
ascending = st.sidebar.checkbox("Ascending", value=True, key="asc")

# Apply filters
mask = pd.Series(True, index=df.index)
for c, vals in selected_cats.items():
    mask &= df[c].astype(str).isin(vals)
for c, (lo, hi) in range_filters.items():
    if c.lower() in count_like:
        mask &= df[c].between(int(lo), int(hi))
    else:
        mask &= df[c].between(lo, hi)

df_f = df.loc[mask].copy()

if query and cat_cols:
    contains = pd.Series(False, index=df_f.index)
    for c in cat_cols:
        contains |= df_f[c].astype(str).str.contains(query, case=False, na=False)
    df_f = df_f[contains]

if sort_col in df_f.columns:
    df_f = df_f.sort_values(by=sort_col, ascending=ascending)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", f"{df.shape[1]}")

if "is_canceled" in df_f.columns and len(df_f):
    c3.metric("Cancellation Rate", f"{df_f['is_canceled'].mean() * 100:.1f}%")
else:
    c3.metric("Cancellation Rate", "—")

if "adr" in df_f.columns and len(df_f):
    c4.metric("Average Daily Rate", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average Daily Rate", "—")

st.divider()

# Visualisations
st.subheader("Visualizations")
charts = []

if "is_canceled" in df_f.columns and "arrival_date_month" in df_f.columns and len(df_f):
    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    tmp = (df_f.groupby("arrival_date_month")["is_canceled"]
           .mean()
           .reindex(order)
           .reset_index()
           .dropna())
    if len(tmp):
        chart1 = alt.Chart(tmp).mark_bar().encode(
            x=alt.X("arrival_date_month:N", title="Month"),
            y=alt.Y("is_canceled:Q",
                    title="Cancellation Rate",
                    axis=alt.Axis(format=".0%")),
            tooltip=["arrival_date_month","is_canceled"]
        ).properties(height=240)
        charts.append(("Cancellation Rate by Month", chart1))

if "hotel" in df_f.columns and "adr" in df_f.columns and len(df_f):
    tmp2 = df_f.groupby("hotel", dropna=False)["adr"].mean().reset_index()
    chart2 = alt.Chart(tmp2).mark_bar().encode(
        x=alt.X("hotel:N", title="Hotel"),
        y=alt.Y("adr:Q", title="Average ADR"),
        tooltip=["hotel","adr"]
    ).properties(height=240)
    charts.append(("Average ADR by Hotel", chart2))

if charts:
    cols = st.columns(len(charts))
    for col, (title, ch) in zip(cols, charts):
        with col:
            st.caption(title)
            st.altair_chart(ch, use_container_width=True)
else:
    st.info("Adjust filters to display charts. (Need columns like 'arrival_date_month', 'is_canceled', 'hotel', 'adr'.)")

st.divider()

# Data Table & Exports
st.subheader("Data Table")
if len(df_f):
    st.dataframe(df_f, use_container_width=True, height=380)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv,
                       file_name="filtered_output.csv", mime="text/csv")
else:
    st.warning("No rows match the current filters. Use 'Reset all filters' in the sidebar or relax constraints.")

# State export for M2M-style use
st.sidebar.download_button(
    "Download current filter state (JSON)",
    data=json.dumps({
        "selected_cats": {k: list(v) for k, v in selected_cats.items()},
        "range_filters": range_filters,
        "query": query,
        "sort_by": sort_col,
        "ascending": ascending,
        "only_positive_adr": only_pos_adr,
    }, indent=2),
    file_name="ui_state.json",
    mime="application/json",
)

st.caption("Run with:  `streamlit run project7_app.py`")
