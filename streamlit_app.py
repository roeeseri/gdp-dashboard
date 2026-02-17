# streamlit_app.py
import os
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Page config
# =========================
st.set_page_config(page_title="Spotify Israel – 7.10 Impact", layout="wide")


# =========================
# Helpers: column detection
# =========================
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _coerce_week(series: pd.Series) -> pd.Series:
    """
    Coerce week column to datetime.
    Your data likely uses day-first dates; we try dayfirst=True first.
    """
    s = pd.to_datetime(series, dayfirst=True, errors="coerce")
    # If too many NaT, try non-dayfirst
    if s.isna().mean() > 0.6:
        s2 = pd.to_datetime(series, dayfirst=False, errors="coerce")
        if s2.isna().mean() < s.isna().mean():
            return s2
    return s


def _make_period(week: pd.Series, cutoff="2023-10-07") -> pd.Series:
    cutoff_ts = pd.Timestamp(cutoff)
    return np.where(week < cutoff_ts, "Before 7.10", "After 7.10")


def _detect_hebrew(text_series: pd.Series) -> pd.Series:
    # Hebrew unicode range
    heb_re = re.compile(r"[\u0590-\u05FF]")
    return text_series.astype(str).apply(lambda x: bool(heb_re.search(x)))


# =========================
# Load data
# =========================
@st.cache_data
def load_main_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # Detect key columns
    col_week = _find_col(df, ["week", "date", "chart_week", "week_start"])
    col_streams = _find_col(df, ["streams", "stream", "num_streams"])
    col_rank = _find_col(df, ["rank", "position"])
    col_track = _find_col(df, ["track_name", "name_track", "track", "song", "title"])
    col_artist = _find_col(df, ["artist_names", "names_artist", "artist", "artists"])

    # Coerce week
    if col_week:
        df[col_week] = _coerce_week(df[col_week])
        df.rename(columns={col_week: "week"}, inplace=True)
    else:
        # If no week, create a dummy date column to avoid crashing tabs; user should fix.
        df["week"] = pd.NaT

    # Standardize streams/rank if exist
    if col_streams:
        df.rename(columns={col_streams: "streams"}, inplace=True)
    if col_rank:
        df.rename(columns={col_rank: "rank"}, inplace=True)
    if col_track:
        df.rename(columns={col_track: "track_name"}, inplace=True)
    if col_artist:
        df.rename(columns={col_artist: "artist_names"}, inplace=True)

    # Period before/after 7.10
    if "week" in df.columns and df["week"].notna().any():
        df["period"] = _make_period(df["week"])
    else:
        df["period"] = "Unknown"

    # Hebrew detection (track or artist)
    if "track_name" in df.columns or "artist_names" in df.columns:
        t = df["track_name"] if "track_name" in df.columns else ""
        a = df["artist_names"] if "artist_names" in df.columns else ""
        heb = _detect_hebrew(pd.Series(t).astype(str)) | _detect_hebrew(pd.Series(a).astype(str))
        df["is_hebrew_track"] = heb.astype(int)
    else:
        df["is_hebrew_track"] = np.nan

    # Primary artist for join (split on comma / '&' / 'feat' etc.)
    if "artist_names" in df.columns:
        primary = (
            df["artist_names"]
            .astype(str)
            .str.split(r",|&|feat\.|ft\.", regex=True)
            .str[0]
            .str.strip()
        )
        df["primary_artist"] = primary
    else:
        df["primary_artist"] = np.nan

    return df


@st.cache_data
def load_artist_meta_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    meta = pd.read_excel(path)

    # Try to find artist name column
    col_artist = _find_col(meta, ["artist", "artist_name", "name", "names_artist"])
    if col_artist:
        meta.rename(columns={col_artist: "artist"}, inplace=True)
    else:
        meta["artist"] = np.nan

    # Try to infer group flags
    # These are guesses: adjust in your sheet if needed.
    col_pro = _find_col(meta, ["pro_israel", "pro", "support_israel", "support"])
    col_anti = _find_col(meta, ["anti_israel", "anti", "against_israel", "boycott"])
    col_neutral = _find_col(meta, ["neutral", "unknown", "other", "unclassified"])

    def infer_group(row):
        def _is_true(x):
            try:
                return int(x) == 1
            except Exception:
                return str(x).strip().lower() in {"true", "yes", "y", "1"}

        if col_pro and _is_true(row.get(col_pro)):
            return "Pro-Israel"
        if col_anti and _is_true(row.get(col_anti)):
            return "Anti-Israel"
        if col_neutral and _is_true(row.get(col_neutral)):
            return "Neutral"
        return "Unknown"

    meta["artist_group"] = meta.apply(infer_group, axis=1)

    # Keep minimal columns
    meta = meta[["artist", "artist_group"]].dropna(subset=["artist"])
    meta["artist"] = meta["artist"].astype(str).str.strip()

    return meta


def safe_load():
    main_path = "data/merged_all_weeks.csv"
    meta_path = "data/artist_meta_filled.xlsx"

    # תמיד נגדיר df_meta כדי למנוע UnboundLocalError
    df_meta = pd.DataFrame(columns=["artist", "artist_group"])

    # טעינת הקובץ הראשי
    try:
        df_main = load_main_csv(main_path)
    except Exception as e:
        st.error(f"Failed to load main CSV: {e}")
        st.info("Make sure you placed merged_all_weeks.csv inside the data/ folder.")
        st.stop()

    # ניסיון לטעון קובץ מטא – אם לא מצליח לא מפילים את האפליקציה
    try:
        df_meta = load_artist_meta_xlsx(meta_path)
    except Exception as e:
        st.warning(f"Artist meta XLSX not loaded ({e}). Continuing with artist_group='Unknown'.")

    # Merge אם העמודות קיימות
    if "primary_artist" in df_main.columns and "artist" in df_meta.columns:
        merged = df_main.merge(
            df_meta,
            left_on="primary_artist",
            right_on="artist",
            how="left",
        )
        merged.drop(columns=["artist"], inplace=True, errors="ignore")
    else:
        merged = df_main.copy()

    # לוודא שקיימת עמודת artist_group
    if "artist_group" not in merged.columns:
        merged["artist_group"] = "Unknown"

    merged["artist_group"] = merged["artist_group"].fillna("Unknown")

    return merged

df = safe_load()


# =========================
# Sidebar filters
# =========================
st.sidebar.title("Filters")

# Date range
if df["week"].notna().any():
    min_date = df["week"].min().date()
    max_date = df["week"].max().date()
    dr = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(dr[0])
    end_date = pd.to_datetime(dr[1])
    dff = df[(df["week"] >= start_date) & (df["week"] <= end_date)].copy()
else:
    dff = df.copy()
    st.sidebar.warning("No valid 'week' dates detected in CSV.")

# Period filter
periods = [p for p in ["Before 7.10", "After 7.10"] if p in dff["period"].unique()]
if periods:
    period_sel = st.sidebar.multiselect("Period", options=sorted(dff["period"].unique()), default=periods)
    dff = dff[dff["period"].isin(period_sel)]

# Hebrew filter
if "is_hebrew_track" in dff.columns and dff["is_hebrew_track"].notna().any():
    heb_opt = st.sidebar.selectbox("Hebrew track?", ["All", "Hebrew", "Non-Hebrew"])
    if heb_opt == "Hebrew":
        dff = dff[dff["is_hebrew_track"] == 1]
    elif heb_opt == "Non-Hebrew":
        dff = dff[dff["is_hebrew_track"] == 0]

# Rank filter
if "rank" in dff.columns and dff["rank"].notna().any():
    top_n = st.sidebar.slider("Top N (by rank)", 50, 200, 200, step=10)
    dff = dff[dff["rank"] <= top_n]

# Artist group filter
if "artist_group" in dff.columns:
    groups = sorted(dff["artist_group"].fillna("Unknown").unique().tolist())
    group_sel = st.sidebar.multiselect("Artist Group", options=groups, default=groups)
    dff = dff[dff["artist_group"].fillna("Unknown").isin(group_sel)]


# =========================
# Header & KPIs
# =========================
st.title("Spotify Israel – Listening Patterns Around 7.10")
st.caption("Dashboard built from merged_all_weeks.csv + artist_meta_filled.xlsx")

k1, k2, k3, k4 = st.columns(4)

if "streams" in dff.columns and dff["streams"].notna().any():
    k1.metric("Total Streams", f"{int(dff['streams'].sum()):,}")
else:
    k1.metric("Total Streams", "N/A")

if "track_name" in dff.columns:
    k2.metric("Unique Tracks", f"{dff['track_name'].nunique():,}")
else:
    k2.metric("Unique Tracks", "N/A")

if "artist_names" in dff.columns:
    k3.metric("Unique Artists", f"{dff['primary_artist'].nunique():,}" if "primary_artist" in dff.columns else f"{dff['artist_names'].nunique():,}")
else:
    k3.metric("Unique Artists", "N/A")

if dff["week"].notna().any():
    k4.metric("Weeks Covered", f"{dff['week'].dt.to_period('W').nunique():,}")
else:
    k4.metric("Weeks Covered", "N/A")


# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Overview",
    "Audio Features Over Time",
    "Hebrew vs Non-Hebrew",
    "Artist Groups (Indexed Monthly)",
])

cutoff_date = pd.Timestamp("2023-10-07")


# -------------------------
# Tab 0: Overview
# -------------------------
with tabs[0]:
    st.subheader("Overview")
    st.write("Quick look at streams over time and distribution before/after 7.10.")

    if dff["week"].notna().any() and "streams" in dff.columns and dff["streams"].notna().any():
        weekly = dff.groupby(pd.Grouper(key="week", freq="W"))["streams"].sum().reset_index()
        fig = px.line(weekly, x="week", y="streams", markers=False)
        fig.add_vline(x=cutoff_date, line_dash="dash")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        if "period" in dff.columns:
            w2 = dff.groupby(["period", pd.Grouper(key="week", freq="W")])["streams"].sum().reset_index()
            fig2 = px.box(w2, x="period", y="streams", points="outliers")
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Need valid 'week' and 'streams' columns to show overview charts.")

    st.divider()
    st.write("Sample of filtered data:")
    st.dataframe(dff.head(30), use_container_width=True)


# -------------------------
# Tab 1: Audio features
# -------------------------
with tabs[1]:
    st.subheader("Audio Features Trends Over Time")
    st.write("Weekly average of selected audio features; dashed line marks 7.10.")

    # Common Spotify audio features
    audio_candidates = [
        "danceability", "energy", "valence", "tempo", "loudness",
        "acousticness", "instrumentalness", "speechiness", "liveness"
    ]
    audio_cols = [c for c in audio_candidates if c in dff.columns]

    if not audio_cols:
        st.warning("No audio feature columns found in CSV (danceability/energy/valence/etc.).")
    elif not dff["week"].notna().any():
        st.warning("No valid 'week' column found/parsed.")
    else:
        default_features = [c for c in ["energy", "valence", "danceability"] if c in audio_cols]
        chosen = st.multiselect("Choose features", audio_cols, default=default_features or audio_cols[:3])

        weekly = dff.groupby(pd.Grouper(key="week", freq="W"))[chosen].mean().reset_index()
        long = weekly.melt(id_vars="week", var_name="feature", value_name="value")

        fig = px.line(long, x="week", y="value", color="feature")
        fig.add_vline(x=cutoff_date, line_dash="dash")
        fig.update_layout(height=450, legend_title_text="Feature")
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Tab 2: Hebrew vs Non-Hebrew
# -------------------------
with tabs[2]:
    st.subheader("Hebrew vs Non-Hebrew Consumption")

    if "is_hebrew_track" not in dff.columns or dff["is_hebrew_track"].isna().all():
        st.warning("Hebrew detection column is missing. Ensure track/artist name exists so we can infer Hebrew.")
    elif "streams" not in dff.columns or dff["streams"].isna().all():
        st.warning("Missing 'streams' column.")
    elif not dff["week"].notna().any():
        st.warning("No valid 'week' column.")
    else:
        tmp = dff.copy()
        tmp["track_lang"] = np.where(tmp["is_hebrew_track"] == 1, "Hebrew", "Non-Hebrew")

        weekly = tmp.groupby([pd.Grouper(key="week", freq="W"), "track_lang"])["streams"].sum().reset_index()
        fig = px.line(weekly, x="week", y="streams", color="track_lang")
        fig.add_vline(x=cutoff_date, line_dash="dash")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.write("Distribution comparison (weekly totals):")
        if "period" in tmp.columns:
            box = tmp.groupby(["period", "track_lang", pd.Grouper(key="week", freq="W")])["streams"].sum().reset_index()
            fig2 = px.box(box, x="track_lang", y="streams", color="period", points="outliers")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)


# -------------------------
# Tab 3: Artist Groups (Indexed Monthly)
# -------------------------
with tabs[3]:
    st.subheader("Monthly Listening by Artist Group (Indexed)")
    st.write("Indexed view: baseline is Oct 2023 = 100 for each group (if data exists).")

    if "artist_group" not in dff.columns:
        st.warning("artist_group column missing after merge. Check artist_meta_filled.xlsx mapping.")
    elif "streams" not in dff.columns or dff["streams"].isna().all():
        st.warning("Missing 'streams' column.")
    elif not dff["week"].notna().any():
        st.warning("No valid 'week' column.")
    else:
        tmp = dff.copy()
        tmp["month"] = tmp["week"].dt.to_period("M").dt.to_timestamp()

        monthly = tmp.groupby(["artist_group", "month"])["streams"].sum().reset_index()

        base_month = pd.Timestamp("2023-10-01")
        base = monthly[monthly["month"] == base_month].set_index("artist_group")["streams"]

        def calc_index(row):
            b = base.get(row["artist_group"], np.nan)
            if pd.isna(b) or b == 0:
                return np.nan
            return (row["streams"] / b) * 100

        monthly["index"] = monthly.apply(calc_index, axis=1)

        if monthly["index"].notna().any():
            fig = px.bar(monthly, x="month", y="index", color="artist_group", barmode="group")
            fig.add_vline(x=base_month, line_dash="dash")
            fig.add_hline(y=100, line_dash="dot")
            fig.update_layout(height=450, yaxis_title="Index (Oct 2023 = 100)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Couldn't compute index (baseline Oct 2023 missing or streams=0). "
                "Try expanding date range to include Oct 2023, or verify the data."
            )
            st.dataframe(monthly.sort_values(["artist_group", "month"]).head(50), use_container_width=True)