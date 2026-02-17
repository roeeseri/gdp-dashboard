# streamlit_app.py
import os
import re
import io
import base64
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Page config
# =========================
st.set_page_config(page_title="Spotify Israel – 7.10 Impact", layout="wide")

# Base dir = folder of this file (works reliably on Git remote)
BASE_DIR = Path(__file__).resolve().parent


# =========================
# Helpers: robust path resolving
# =========================
def resolve_repo_path(p: str) -> str:
    """
    Resolve a path that is stored in META (relative to repo root / script dir).
    If p is absolute -> return as-is.
    If p is relative -> resolve relative to BASE_DIR.
    """
    p = (p or "").strip()
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((BASE_DIR / pp).resolve())


# =========================
# Helpers: column detection
# =========================
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _coerce_week(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if s.isna().mean() > 0.6:
        s2 = pd.to_datetime(series, dayfirst=False, errors="coerce")
        if s2.isna().mean() < s.isna().mean():
            return s2
    return s


def _make_period(week: pd.Series, cutoff="2023-10-07") -> pd.Series:
    cutoff_ts = pd.Timestamp(cutoff)
    return np.where(week < cutoff_ts, "Before 7.10", "After 7.10")


def _detect_hebrew(text_series: pd.Series) -> pd.Series:
    heb_re = re.compile(r"[\u0590-\u05FF]")
    return text_series.astype(str).apply(lambda x: bool(heb_re.search(x)))


# =========================
# Cache controls
# =========================
st.sidebar.title("Filters")
if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()


# =========================
# Load data
# =========================
@st.cache_data
def load_main_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    col_week = _find_col(df, ["week", "date", "chart_week", "week_start"])
    col_streams = _find_col(df, ["streams", "stream", "num_streams"])
    col_rank = _find_col(df, ["rank", "position"])
    col_track = _find_col(df, ["track_name", "name_track", "track", "song", "title"])
    col_artist = _find_col(df, ["artist_names", "names_artist", "artist", "artists"])

    if col_week:
        df[col_week] = _coerce_week(df[col_week])
        df.rename(columns={col_week: "week"}, inplace=True)
    else:
        df["week"] = pd.NaT

    if col_streams:
        df.rename(columns={col_streams: "streams"}, inplace=True)
    if col_rank:
        df.rename(columns={col_rank: "rank"}, inplace=True)
    if col_track:
        df.rename(columns={col_track: "track_name"}, inplace=True)
    if col_artist:
        df.rename(columns={col_artist: "artist_names"}, inplace=True)

    if "week" in df.columns and df["week"].notna().any():
        df["period"] = _make_period(df["week"])
    else:
        df["period"] = "Unknown"

    if "track_name" in df.columns or "artist_names" in df.columns:
        t = df["track_name"] if "track_name" in df.columns else ""
        a = df["artist_names"] if "artist_names" in df.columns else ""
        heb = _detect_hebrew(pd.Series(t).astype(str)) | _detect_hebrew(pd.Series(a).astype(str))
        df["is_hebrew_track"] = heb.astype(int)
    else:
        df["is_hebrew_track"] = np.nan

    # Primary artist for join (split on comma / '&' / feat etc.)
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
def load_artist_meta_groups(path: str) -> pd.DataFrame:
    """Small meta for the sidebar 'Artist Group' filter."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    meta = pd.read_excel(path, engine="openpyxl")

    col_artist = _find_col(meta, ["artist", "artist_name", "name", "names_artist"])
    if col_artist:
        meta.rename(columns={col_artist: "artist"}, inplace=True)
    else:
        meta["artist"] = np.nan

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
    meta = meta[["artist", "artist_group"]].dropna(subset=["artist"])
    meta["artist"] = meta["artist"].astype(str).str.strip()
    return meta


@st.cache_data
def load_artist_meta_full(path: str) -> pd.DataFrame:
    """
    Full meta for dumbbell:
    - index by artist
    - image_path resolved later
    - pro_israel / anti_israel / neutral
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    meta = pd.read_excel(path, engine="openpyxl")

    for col in ["artist", "image_path", "anti_israel", "pro_israel", "neutral"]:
        if col not in meta.columns:
            meta[col] = "" if col in ["artist", "image_path"] else 0

    meta["artist"] = meta["artist"].astype(str).str.strip()
    meta["image_path"] = meta["image_path"].fillna("").astype(str).str.strip()

    for c in ["anti_israel", "pro_israel", "neutral"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce").fillna(0).astype(int)

    # IMPORTANT: resolve image paths relative to streamlit_app.py location
    meta["image_path_abs"] = meta["image_path"].apply(resolve_repo_path)

    return meta.drop_duplicates("artist").set_index("artist")


def safe_load():
    # resolve paths robustly
    main_path = resolve_repo_path("data/merged_all_weeks.csv")
    # NOTE: use the UPDATED meta you uploaded into repo
    meta_path = resolve_repo_path("/workspaces/gdp-dashboard/artist_meta_filled_UPDATED.xlsx")

    # load main
    try:
        df_main = load_main_csv(main_path)
    except Exception as e:
        st.error(f"Failed to load main CSV: {e}")
        st.info("Expected: data/merged_all_weeks.csv (relative to repo).")
        st.stop()

    # load groups meta (for filter) - if fails, continue
    try:
        df_meta_groups = load_artist_meta_groups(meta_path)
    except Exception as e:
        st.warning(f"Artist meta not loaded for groups ({e}). Continuing with artist_group='Unknown'.")
        df_meta_groups = pd.DataFrame(columns=["artist", "artist_group"])

    # merge on primary_artist (only for the group filter)
    if "primary_artist" in df_main.columns and "artist" in df_meta_groups.columns:
        merged = df_main.merge(df_meta_groups, left_on="primary_artist", right_on="artist", how="left")
        merged.drop(columns=["artist"], inplace=True, errors="ignore")
    else:
        merged = df_main.copy()

    if "artist_group" not in merged.columns:
        merged["artist_group"] = "Unknown"
    merged["artist_group"] = merged["artist_group"].fillna("Unknown")
    return merged, meta_path


df, META_PATH_RESOLVED = safe_load()


# =========================
# Sidebar filters
# =========================
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
st.caption("Dashboard built from merged_all_weeks.csv + artist_meta_filled_UPDATED.xlsx")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Streams", f"{int(dff['streams'].sum()):,}" if "streams" in dff.columns and dff["streams"].notna().any() else "N/A")
k2.metric("Unique Tracks", f"{dff['track_name'].nunique():,}" if "track_name" in dff.columns else "N/A")
k3.metric("Unique Artists", f"{dff['primary_artist'].nunique():,}" if "primary_artist" in dff.columns else "N/A")
k4.metric("Weeks Covered", f"{dff['week'].dt.to_period('W').nunique():,}" if dff["week"].notna().any() else "N/A")


# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Overview",
    "Audio Features Over Time",
    "Hebrew vs Non-Hebrew",
    "Before vs After (Dumbbell + Avatars)",
])

cutoff_date = pd.Timestamp("2023-10-07")


# -------------------------
# Tab 0: Overview
# -------------------------
with tabs[0]:
    st.subheader("Overview")

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

    audio_candidates = [
        "danceability", "energy", "valence", "tempo", "loudness",
        "acousticness", "instrumentalness", "speechiness", "liveness"
    ]
    audio_cols = [c for c in audio_candidates if c in dff.columns]

    if not audio_cols:
        st.warning("No audio feature columns found in CSV.")
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
        st.warning("Hebrew detection column is missing.")
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

        if "period" in tmp.columns:
            box = tmp.groupby(["period", "track_lang", pd.Grouper(key="week", freq="W")])["streams"].sum().reset_index()
            fig2 = px.box(box, x="track_lang", y="streams", color="period", points="outliers")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)


# -------------------------
# Tab 3: Dumbbell + Avatars
# -------------------------
with tabs[3]:
    st.subheader("Before vs After — Median Monthly Index per Artist (baseline month = 100)")

    # ---- Controls ----
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        baseline_month = st.selectbox(
            "Baseline month (index = 100)",
            options=pd.date_range("2023-01-01", "2024-12-01", freq="MS"),
            index=list(pd.date_range("2023-01-01", "2024-12-01", freq="MS")).index(pd.Timestamp("2023-10-01"))
        )
    with c2:
        split_month = st.selectbox(
            "Before/After split month",
            options=pd.date_range("2023-01-01", "2024-12-01", freq="MS"),
            index=list(pd.date_range("2023-01-01", "2024-12-01", freq="MS")).index(pd.Timestamp("2023-10-01"))
        )
    with c3:
        top_k = st.slider("Top K artists", 5, 40, 23)
    with c4:
        show_avatars = st.checkbox("Show avatars", value=True)

    # stance colors
    RING_GREEN  = "rgba(0,158,115,1)"     # pro
    RING_ORANGE = "rgba(230,159,0,1)"     # neutral
    RING_RED    = "rgba(213,94,0,1)"      # anti
    RING_GRAY   = "rgba(120,120,120,0.9)" # unknown

    def ring_color(g: str) -> str:
        if g == "pro": return RING_GREEN
        if g == "anti": return RING_RED
        if g == "neutral": return RING_ORANGE
        return RING_GRAY

    def group_of_artist(meta_full: pd.DataFrame, a: str) -> str:
        if a not in meta_full.index:
            return "unknown"
        if int(meta_full.loc[a, "pro_israel"]) == 1: return "pro"
        if int(meta_full.loc[a, "anti_israel"]) == 1: return "anti"
        if int(meta_full.loc[a, "neutral"]) == 1: return "neutral"
        return "unknown"

    def split_artists(s):
        if pd.isna(s):
            return []
        return [p.strip() for p in str(s).split(",") if p.strip()]

    def bytes_to_data_uri_png(png_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    def pil_circle_avatar(image_path_abs: str, size_px: int = 120):
        """
        image_path_abs must be absolute path (we precompute it).
        Returns PNG bytes or None.
        """
        if not image_path_abs or not os.path.exists(image_path_abs):
            return None
        try:
            from PIL import Image, ImageOps, ImageDraw
            im = Image.open(image_path_abs)  # keep format support
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGBA")

            w, h = im.size
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            im = im.crop((left, top, left + m, top + m))
            im = im.resize((size_px, size_px), Image.LANCZOS)

            mask = Image.new("L", (size_px, size_px), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size_px - 1, size_px - 1), fill=255)

            out = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
            out.paste(im, (0, 0), mask=mask)

            buf = io.BytesIO()
            out.save(buf, format="PNG")  # always output PNG bytes for Plotly
            return buf.getvalue()
        except Exception:
            return None

    # Load full meta (absolute image paths computed inside)
    try:
        meta_full = load_artist_meta_full(META_PATH_RESOLVED)
    except Exception as e:
        st.error(f"Failed to load META: {e}")
        st.stop()

    # Debug summary: how many image paths exist
    with st.expander("Debug: META + image paths"):
        st.write("META path:", META_PATH_RESOLVED)
        if "image_path_abs" in meta_full.columns:
            exists = meta_full["image_path_abs"].apply(lambda p: Path(p).exists())
            st.write("Images found on server:", int(exists.sum()), "/", len(exists))
            st.dataframe(
                meta_full.reset_index()[["artist", "image_path", "image_path_abs"]].assign(img_exists=exists.values)
                .loc[lambda d: ~d["img_exists"]]
                .head(30),
                use_container_width=True
            )

    if "artist_names" not in dff.columns or "streams" not in dff.columns or not dff["week"].notna().any():
        st.warning("Need valid week/streams/artist_names to build the dumbbell chart.")
        st.stop()

    # Build weekly artist streams (split collabs evenly)
    base_df = dff[["week", "artist_names", "streams"]].copy()
    base_df["artist_list"] = base_df["artist_names"].apply(split_artists)
    base_df = base_df[base_df["artist_list"].map(len) > 0].copy()
    base_df["n_artists"] = base_df["artist_list"].map(len)
    base_df = base_df.explode("artist_list", ignore_index=True).rename(columns={"artist_list": "artist"})
    base_df["artist"] = base_df["artist"].astype(str).str.strip()
    base_df["artist_streams"] = base_df["streams"] / base_df["n_artists"]

    weekly = (
        base_df.groupby(["week", "artist"], as_index=False)["artist_streams"]
              .sum()
              .rename(columns={"artist_streams": "streams"})
    )

    weekly["month"] = weekly["week"].dt.to_period("M").dt.to_timestamp()
    monthly = weekly.groupby(["month", "artist"], as_index=False)["streams"].sum()
    wide = (
        monthly.pivot_table(index="month", columns="artist", values="streams", aggfunc="sum")
              .sort_index()
              .fillna(0)
    )

    if wide.empty:
        st.warning("No monthly data after filters.")
        st.stop()

    # Snap baseline month if missing
    if baseline_month not in wide.index:
        nearest = wide.index[np.argmin(np.abs((wide.index - baseline_month).days))]
        st.info(f"Baseline month not found. Using nearest: {nearest.strftime('%Y-%m')}")
        baseline_month = nearest

    baseline = wide.loc[baseline_month].replace(0, np.nan)
    indexed = (wide.divide(baseline, axis=1) * 100.0).replace([np.inf, -np.inf], np.nan)

    idx_long = indexed.stack(dropna=False).reset_index()
    idx_long.columns = ["month", "artist", "index_value"]
    idx_long = idx_long.dropna(subset=["index_value"]).copy()
    idx_long["period2"] = np.where(idx_long["month"] < split_month, "Before", "After")

    summary = (
        idx_long.groupby(["artist", "period2"], as_index=False)["index_value"]
                .median()
                .pivot(index="artist", columns="period2", values="index_value")
                .reset_index()
                .dropna(subset=["Before", "After"])
    )

    if summary.empty:
        st.warning("No artists have both Before and After data with a valid baseline.")
        st.stop()

    summary["group"] = summary["artist"].apply(lambda a: group_of_artist(meta_full, a))
    summary["color"] = summary["group"].apply(ring_color)
    summary["delta"] = summary["After"] - summary["Before"]
    summary["importance"] = summary["Before"] + summary["After"]

    summary = summary.sort_values("importance", ascending=False).head(top_k).copy()
    summary = summary.sort_values("delta", ascending=True).reset_index(drop=True)
    summary["y"] = np.arange(len(summary))

    fig = go.Figure()

    for _, r in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["Before"], r["After"]],
            y=[r["y"], r["y"]],
            mode="lines",
            line=dict(width=4, color=r["color"]),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=summary["Before"],
        y=summary["y"],
        mode="markers",
        marker=dict(size=10, color="rgba(80,80,80,0.6)"),
        name="Before (median index)",
        hovertemplate="<b>%{customdata}</b><br>Before median index: %{x:.1f}<extra></extra>",
        customdata=summary["artist"]
    ))

    fig.add_trace(go.Scatter(
        x=summary["After"],
        y=summary["y"],
        mode="markers",
        marker=dict(size=14, color="rgba(255,255,255,0.95)", line=dict(width=4, color=summary["color"])),
        name="After (median index)",
        hovertemplate="<b>%{customdata[0]}</b><br>After median index: %{x:.1f}<br>Δ: %{customdata[1]:.1f}<extra></extra>",
        customdata=np.stack([summary["artist"], summary["delta"]], axis=1)
    ))

    # Avatars
    missing_avatars = 0
    failed_open = 0
    if show_avatars and "image_path_abs" in meta_full.columns:
        x_min = float(np.nanmin(summary[["Before", "After"]].values))
        x_max = float(np.nanmax(summary[["Before", "After"]].values))
        x_range = max(1.0, x_max - x_min)
        x_sizex = max(3.0, 0.06 * x_range)
        x_offset = 0.03 * x_range

        for _, r in summary.iterrows():
            a = r["artist"]
            if a not in meta_full.index:
                missing_avatars += 1
                continue

            img_abs = str(meta_full.loc[a, "image_path_abs"])
            if not img_abs or not Path(img_abs).exists():
                missing_avatars += 1
                continue

            avatar_bytes = pil_circle_avatar(img_abs, size_px=120)
            if avatar_bytes is None:
                failed_open += 1
                continue

            uri = bytes_to_data_uri_png(avatar_bytes)
            fig.add_layout_image(dict(
                source=uri,
                xref="x", yref="y",
                x=float(r["After"]) + x_offset,
                y=float(r["y"]),
                sizex=x_sizex,
                sizey=0.85,
                xanchor="left", yanchor="middle",
                layer="above",
                opacity=1.0
            ))

    fig.update_yaxes(
        tickmode="array",
        tickvals=summary["y"],
        ticktext=summary["artist"],
        title="Artist"
    )
    fig.update_xaxes(
        title=f"Monthly streams index (baseline: {pd.Timestamp(baseline_month).strftime('%b %Y')} = 100)",
        zeroline=False
    )

    # Legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(width=4, color=RING_GREEN),
        marker=dict(size=12, color="white", line=dict(width=4, color=RING_GREEN)),
        name="Pro-Israel / Israeli"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(width=4, color=RING_ORANGE),
        marker=dict(size=12, color="white", line=dict(width=4, color=RING_ORANGE)),
        name="Neutral / not stated"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(width=4, color=RING_RED),
        marker=dict(size=12, color="white", line=dict(width=4, color=RING_RED)),
        name="Anti-Israel"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(width=4, color=RING_GRAY),
        marker=dict(size=12, color="white", line=dict(width=4, color=RING_GRAY)),
        name="Unknown"
    ))

    fig.update_layout(
        title="Median monthly streams — Before vs After (indexed baseline = 100)",
        template="plotly_white",
        height=max(650, 34 * len(summary)),
        margin=dict(l=260, r=260, t=80, b=60),
        legend=dict(
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Artists shown", f"{len(summary)}")
    m2.metric("Median Δ (After−Before)", f"{summary['delta'].median():.1f}")
    m3.metric("Missing image files", f"{missing_avatars}" if show_avatars else "—")
    m4.metric("Failed to open images", f"{failed_open}" if show_avatars else "—")

    if show_avatars and failed_open > 0:
        st.warning(
            "Some images exist but could not be opened. "
            "If the failing ones are .webp, your server Pillow build may not support WEBP. "
            "Best fix: convert WEBP to JPG/PNG in the repo and update META paths."
        )

    with st.expander("Show summary table"):
        st.dataframe(
            summary[["artist", "Before", "After", "delta", "group"]].sort_values("delta", ascending=False),
            use_container_width=True
        )
