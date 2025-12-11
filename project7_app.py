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

    If nothing is found, fall back to a small synthetic dataset.
    """
    df = None

    # 1) Parquet
    parquet_file = "cleaned_output_2025-12-11_11-48-46.parquet"
    if os.path.exists(parquet_file):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            st.warning(f"Could not read {parquet_file}: {e}")

    # 2) CSV main
    if df is None:
        for csv_file in ["hotel_bookings_clean.csv", "hotel_bookings_clean(1).csv"]:
            if os.path.exists(csv_file):
                try:
                    df = pd
