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
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# =========================
# Page config
# =========================
st.set_page_config(page_title="Spotify Israel – 7.10 Impact", layout="wide")

# =========================
# Spotify-ish palette (Streamlit UI + Plotly)
# - UI text: spotify green
# - Plot text inside charts: black
# - Controls + borders: greys
# =========================
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_GRAY  = "#B3B3B3"
SPOTIFY_GRAY2 = "#E6E6E6"
SPOTIFY_GRAY3 = "#6B6B6B"

# Streamlit UI styling (white background, green headings, GREY widgets accents)
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background: #FFFFFF;
        color: {SPOTIFY_GREEN};
    }}

    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid rgba(0,0,0,0.08);
    }}

    /* Titles/headers in Streamlit: green */
    h1, h2, h3, h4, h5, h6 {{
        color: {SPOTIFY_GREEN} !important;
    }}

    /* Regular text in Streamlit: spotify black */
    p, span, div, label, .stCaption, .stMarkdown, .stText, .stMetricLabel {{
        color: {SPOTIFY_BLACK} !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.10);
        padding: 12px 12px;
        border-radius: 14px;
    }}

    /* Plot containers */
    div[data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 14px;
        padding: 10px 10px;
    }}

    /* Make slider/selection accents grey-ish (Streamlit varies by version; best-effort) */
    div[data-baseweb="slider"] * {{
        color: {SPOTIFY_BLACK} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Key dates & events
# =========================
CUTOFF_DATE = pd.Timestamp("2023-10-07")

EVENTS = {
    "Protests after the reasonableness standard repeal": "2023-07-30",
    "Rosh Hashanah": "2023-09-17",
    "Evacuation of communities in the North & South": "2023-10-15",
    "First hostage deal": "2023-11-24",
    "Hanukkah": "2023-12-10",
    "Shooting of 6 hostages": "2023-12-15",
    "Building collapse killing 21 soldiers": "2024-01-21",
    "Rescue operation of 2 hostages": "2024-02-12",
    "Hostage deal talks": "2024-03-11",
    "Escalation in the North": "2024-03-23",
}

BASE_DIR = Path(__file__).resolve().parent


# =========================
# Path utils
# =========================
def resolve_repo_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((BASE_DIR / pp).resolve())


def norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\bfeat\.?\b|\bft\.?\b", " ", s)
    s = re.sub(r"[^0-9a-z\u0590-\u05ff]+", "", s)
    return s


@st.cache_data
def build_avatar_lookup(rel_dir: str = "artists_photos") -> dict:
    folder = Path(resolve_repo_path(rel_dir))
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    lookup = {}
    if folder.exists() and folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                lookup[norm_key(p.stem)] = str(p.resolve())
                lookup[norm_key(p.name)] = str(p.resolve())
    return lookup


# =========================
# Helpers: column detection
# =========================
CANONICAL_ARTIST = {
    "ness": "Ness & Stilla",
    "stilla": "Ness & Stilla",
    "nessandstilla": "Ness & Stilla",
    "nessstilla": "Ness & Stilla",
    "theweekend": "The Weeknd",
    "theweeknd": "The Weeknd",
    "tylerthecreator": "Tyler, The Creator",
}


def canon_artist(name: str) -> str:
    k = norm_key(name)
    return CANONICAL_ARTIST.get(k, name.strip() if isinstance(name, str) else name)


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


def add_event_lines(fig, selected_event_labels, *, show_labels=False, y_for_label=None):
    for label in selected_event_labels:
        dt = pd.to_datetime(EVENTS.get(label), errors="coerce")
        if pd.isna(dt):
            continue

        fig.add_vline(
            x=dt,
            line_dash="dot",
            line_color="rgba(0,0,0,0.35)",
            line_width=1
        )

        if show_labels and (y_for_label is not None):
            fig.add_annotation(
                x=dt,
                y=y_for_label,
                text=label,
                showarrow=True,
                arrowhead=2,
                ax=18,
                ay=-22,
                font=dict(size=11, color=SPOTIFY_BLACK),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )

    fig.add_vline(
        x=CUTOFF_DATE,
        line_dash="dash",
        line_color="rgba(0,0,0,0.95)",
        line_width=3
    )
    return fig


def apply_spotify_plotly_layout(fig: go.Figure):
    # ✅ chart text black, not green
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=SPOTIFY_BLACK),
        title_font=dict(color=SPOTIFY_BLACK),
        legend_font=dict(color=SPOTIFY_BLACK),
    )
    fig.update_xaxes(
        title_font=dict(color=SPOTIFY_BLACK),
        tickfont=dict(color=SPOTIFY_BLACK),
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.10)"
    )
    fig.update_yaxes(
        title_font=dict(color=SPOTIFY_BLACK),
        tickfont=dict(color=SPOTIFY_BLACK),
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.10)"
    )
    return fig


# =========================
# Plotly hover-zoom for layout_image (Tab 3)
# =========================
def render_plotly_with_hover_image_zoom(fig: go.Figure, height: int, zoom: float = 2.35):
    """
    Bigger hover zoom (default 2.35)
    """
    div_id = "plotly-dumbbell-hoverzoom"
    fig_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="include",
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )

    html = f"""
    <div style="width:100%; height:{height}px;">
      {fig_html}
    </div>

    <script>
      (function() {{
        const div = document.getElementById('{div_id}');
        if (!div) return;

        function cloneOrig(images) {{
          return (images || []).map(im => ({{
            sizex: im.sizex, sizey: im.sizey, x: im.x, y: im.y
          }}));
        }}

        let orig = null;
        let lastIdx = null;

        function ensureOrig() {{
          if (!orig && div.layout && div.layout.images) {{
            orig = cloneOrig(div.layout.images);
          }}
        }}

        div.on('plotly_hover', function(ev) {{
          ensureOrig();
          if (!orig) return;
          if (!ev || !ev.points || !ev.points.length) return;

          const cd = ev.points[0].customdata;
          if (cd === undefined || cd === null) return;

          const idx = Array.isArray(cd) ? cd[0] : cd;
          if (idx === undefined || idx === null) return;
          if (!div.layout || !div.layout.images || !div.layout.images[idx]) return;

          if (lastIdx !== null && orig[lastIdx] && div.layout.images[lastIdx]) {{
            div.layout.images[lastIdx].sizex = orig[lastIdx].sizex;
            div.layout.images[lastIdx].sizey = orig[lastIdx].sizey;
            div.layout.images[lastIdx].x = orig[lastIdx].x;
            div.layout.images[lastIdx].y = orig[lastIdx].y;
          }}

          const o = orig[idx];
          const im = div.layout.images[idx];
          im.sizex = o.sizex * {zoom};
          im.sizey = o.sizey * {zoom};
          im.x = o.x;
          im.y = o.y;

          lastIdx = idx;
          Plotly.relayout(div, {{images: div.layout.images}});
        }});

        div.on('plotly_unhover', function(ev) {{
          ensureOrig();
          if (!orig) return;
          if (lastIdx === null) return;
          if (!div.layout || !div.layout.images || !div.layout.images[lastIdx]) return;

          const o = orig[lastIdx];
          const im = div.layout.images[lastIdx];
          im.sizex = o.sizex;
          im.sizey = o.sizey;
          im.x = o.x;
          im.y = o.y;

          Plotly.relayout(div, {{images: div.layout.images}});
          lastIdx = null;
        }});
      }})();
    </script>
    """
    components.html(html, height=height, scrolling=True)


# =========================
# Cache control
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
    col_uri = _find_col(df, ["uri", "track_uri", "spotify_uri", "track_id", "id"])

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
    if col_uri:
        df.rename(columns={col_uri: "uri"}, inplace=True)

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

    meta["artist_key"] = meta["artist"].apply(norm_key)
    meta["image_path_abs"] = meta["image_path"].apply(resolve_repo_path)

    meta = meta.drop_duplicates("artist_key").set_index("artist_key")
    return meta


# =========================
# Genres pipeline (KMeans + Rules)
# =========================
FEATS = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "acousticness", "instrumentalness", "speechiness", "liveness"
]

HEB_RE = re.compile(r"[\u0590-\u05FF]")


def _has_hebrew(x) -> bool:
    return bool(HEB_RE.search(str(x))) if pd.notna(x) else False


def _wavg(g: pd.DataFrame, col: str, w: str = "streams") -> float:
    ww = g[w].fillna(0).to_numpy()
    xx = g[col].to_numpy()
    s = ww.sum()
    return float((xx * ww).sum() / s) if s > 0 else float(np.mean(xx))


def _name_cluster(row: pd.Series) -> str:
    e = row["energy"]
    dnc = row["danceability"]
    ac = row["acousticness"]
    inst = row["instrumentalness"]
    sp = row["speechiness"]
    val = row["valence"]
    tmp = row["tempo"]
    loud = row["loudness"]

    if ac > 0.55 and e < 0.45 and inst < 0.35:
        return "Classic / Acoustic"
    if inst > 0.55 and sp < 0.08:
        return "Instrumental / Ambient"
    if sp > 0.20 and e > 0.50:
        return "Rap / Hip-Hop"
    if e > 0.70 and loud > -7 and dnc < 0.55:
        return "Rock"
    if dnc > 0.72 and e > 0.65 and tmp > 115:
        return "Pop / EDM"
    if val < 0.40 and e < 0.55 and ac > 0.35:
        return "Ballad / Sad Pop"
    return "Pop"


@st.cache_data
def add_genres(df_in: pd.DataFrame, k: int = 8, random_state: int = 42):
    d = df_in.copy()

    need_cols = ["uri", "track_name", "artist_names", "week"] + FEATS
    missing = [c for c in need_cols if c not in d.columns]
    if missing:
        return d, {"ok": False, "reason": f"Missing columns: {missing}"}

    d["week"] = pd.to_datetime(d["week"], dayfirst=True, errors="coerce")

    if "streams" not in d.columns:
        d["streams"] = np.nan

    for c in FEATS + ["streams"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    track = d.dropna(subset=["uri"] + FEATS).copy()
    if track.empty:
        return d, {"ok": False, "reason": "No rows with uri + audio features."}

    track_song = (
        track.groupby("uri")
        .apply(lambda g: pd.Series({
            "track_name": g["track_name"].iloc[0],
            "artist_names": g["artist_names"].iloc[0],
            **{f: _wavg(g, f) for f in FEATS}
        }))
        .reset_index()
    )

    X = track_song[FEATS].to_numpy()
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=int(k), random_state=int(random_state), n_init="auto")
    track_song["cluster"] = kmeans.fit_predict(Xz)

    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATS
    )
    centroids["cluster"] = range(int(k))
    cluster2name = centroids.set_index("cluster").apply(_name_cluster, axis=1).to_dict()

    def _tag_from_centroid(row: pd.Series, base_name: str) -> str:
        e = float(row.get("energy", np.nan))
        dnc = float(row.get("danceability", np.nan))
        ac = float(row.get("acousticness", np.nan))
        inst = float(row.get("instrumentalness", np.nan))
        sp = float(row.get("speechiness", np.nan))
        val = float(row.get("valence", np.nan))
        tmp = float(row.get("tempo", np.nan))
        loud = float(row.get("loudness", np.nan))

        tags = []
        if sp >= 0.22: tags.append("Rap")
        if inst >= 0.55: tags.append("Instrumental")
        if ac >= 0.60: tags.append("Acoustic")
        if e >= 0.75: tags.append("High Energy")
        if e <= 0.40: tags.append("Chill")

        if val >= 0.65: tags.append("Upbeat")
        elif val <= 0.35: tags.append("Sad")

        if dnc >= 0.72: tags.append("Dance")
        if tmp >= 125: tags.append("Fast Tempo")
        elif tmp <= 95: tags.append("Slow Tempo")
        if loud >= -6.5: tags.append("Loud")

        if base_name == "Rock" and e >= 0.70 and loud >= -7:
            tags.insert(0, "Rock")

        return tags[0] if tags else "Style"

    base2clusters = {}
    for c in range(int(k)):
        base = str(cluster2name.get(c, "Other")).strip() or "Other"
        base2clusters.setdefault(base, []).append(c)

    cluster2unique = {}
    for base_name, clusters in base2clusters.items():
        if len(clusters) == 1:
            cluster2unique[clusters[0]] = base_name
            continue

        used = set()
        for c in clusters:
            row = centroids.loc[centroids["cluster"] == c].iloc[0]
            tag = _tag_from_centroid(row, base_name)

            if tag in used:
                candidates = ["Upbeat", "Sad", "Dance", "Fast Tempo", "Slow Tempo", "Loud", "Style"]
                picked = next((cand for cand in candidates if cand not in used), f"Variant {len(used)+1}")
                tag = picked

            used.add(tag)
            cluster2unique[c] = f"{base_name} ({tag})"

    track_song["genre_base"] = track_song["cluster"].map(cluster2name)
    track_song["genre"] = track_song["cluster"].astype(int).map(cluster2unique)

    track_song["is_hebrew"] = (
        track_song["track_name"].apply(_has_hebrew) |
        track_song["artist_names"].apply(_has_hebrew)
    )

    mask_mizrahi = (
        track_song["is_hebrew"] &
        (track_song["danceability"] > 0.62) &
        (track_song["energy"] > 0.55)
    )
    track_song["is_mizrahi"] = mask_mizrahi.astype(int)

    dfg = d.merge(track_song[["uri", "cluster", "genre_base", "genre", "is_mizrahi"]], on="uri", how="left")

    report = {"ok": True, "k": int(k), "inertia": float(kmeans.inertia_), "n_iter": int(kmeans.n_iter_)}
    try:
        report["silhouette"] = float(silhouette_score(Xz, track_song["cluster"]))
        report["calinski_harabasz"] = float(calinski_harabasz_score(Xz, track_song["cluster"]))
        report["davies_bouldin"] = float(davies_bouldin_score(Xz, track_song["cluster"]))
    except Exception as e:
        report["metrics_note"] = f"Metrics skipped: {e}"

    cluster_sizes = track_song["cluster"].value_counts().sort_index().rename("n_songs").reset_index()
    cluster_sizes = cluster_sizes.rename(columns={"index": "cluster"})
    centroids_out = centroids.copy()
    centroids_out["genre_base"] = centroids_out["cluster"].map(cluster2name)
    centroids_out = centroids_out.merge(cluster_sizes, on="cluster", how="left")

    report["cluster_sizes"] = cluster_sizes
    report["centroids"] = centroids_out.sort_values("n_songs", ascending=False)

    return dfg, report


# =========================
# safe_load
# =========================
def safe_load():
    main_path = resolve_repo_path("data/merged_all_weeks.csv")
    meta_path = resolve_repo_path("/workspaces/gdp-dashboard/artist_meta.xlsx")

    try:
        df_main = load_main_csv(main_path)
    except Exception as e:
        st.error(f"Failed to load main CSV: {e}")
        st.info("Expected: data/merged_all_weeks.csv")
        st.stop()

    try:
        df_meta_groups = load_artist_meta_groups(meta_path)
    except Exception as e:
        st.warning(f"Artist meta XLSX not loaded for group filter ({e}). Using Unknown.")
        df_meta_groups = pd.DataFrame(columns=["artist", "artist_group"])

    if "primary_artist" in df_main.columns and "artist" in df_meta_groups.columns:
        merged = df_main.merge(df_meta_groups, left_on="primary_artist", right_on="artist", how="left")
        merged.drop(columns=["artist"], inplace=True, errors="ignore")
    else:
        merged = df_main.copy()

    if "artist_group" not in merged.columns:
        merged["artist_group"] = "Unknown"
    merged["artist_group"] = merged["artist_group"].fillna("Unknown")

    return merged, meta_path


df, META_PATH = safe_load()

# =========================
# Sidebar filters
# =========================
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

periods = [p for p in ["Before 7.10", "After 7.10"] if p in dff["period"].unique()]
if periods:
    period_sel = st.sidebar.multiselect("Period", options=sorted(dff["period"].unique()), default=periods)
    dff = dff[dff["period"].isin(period_sel)]

if "rank" in dff.columns and dff["rank"].notna().any():
    top_n = st.sidebar.slider("Top N (by rank)", 50, 200, 200, step=10)
    dff = dff[dff["rank"] <= top_n]

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

st.sidebar.divider()
st.sidebar.subheader("Event lines")

show_events = st.sidebar.checkbox("Show event lines", value=False)
event_labels = list(EVENTS.keys())
default_events = [
    "Evacuation of communities in the North & South",
    "First hostage deal",
    "Building collapse killing 21 soldiers",
]
selected_events = st.sidebar.multiselect(
    "Choose events",
    options=event_labels,
    default=default_events if show_events else []
)
show_event_labels = st.sidebar.checkbox("Show event labels on chart", value=False)


# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Overview",
    "Audio Features Over Time",
    "Genres (Before vs After)",
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
        weekly = weekly.sort_values("week")

        fig = px.line(weekly, x="week", y="streams", markers=False)
        fig.update_layout(
            height=420,
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            hovermode="x unified",
            xaxis_title="Week",
            yaxis_title="Streams",
        )

        y_for_label = float(weekly["streams"].max()) if len(weekly) else None
        fig = add_event_lines(
            fig,
            selected_events if (show_events and selected_events) else [],
            show_labels=show_event_labels,
            y_for_label=y_for_label
        )
        fig = apply_spotify_plotly_layout(fig)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need valid 'week' and 'streams' columns to show overview chart.")


# -------------------------
# Tab 1: Audio features (keep clean + spotify-ish lines)
# -------------------------
with tabs[1]:
    st.subheader("Audio Features Trends Over Time (Min-Max Normalized)")

    audio_cols = [c for c in FEATS if c in dff.columns]
    if not audio_cols:
        st.warning("No audio feature columns found in CSV.")
        st.stop()
    if not dff["week"].notna().any():
        st.warning("No valid 'week' column found/parsed.")
        st.stop()

    cA, cB, cC = st.columns([1.6, 1.0, 1.0])
    with cA:
        default_features = [c for c in ["energy", "valence", "danceability", "acousticness"] if c in audio_cols]
        chosen = st.multiselect("Choose features", audio_cols, default=default_features or audio_cols[:3])
    with cB:
        use_rolling = st.checkbox("Rolling smoothing", value=True)
    with cC:
        rolling_window = st.slider("Rolling window (weeks)", 1, 8, 3) if use_rolling else 1

    tmp = dff[["week"] + chosen].copy()
    tmp["week"] = pd.to_datetime(tmp["week"], errors="coerce")
    for c in chosen:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna(subset=["week"] + chosen)

    weekly_means = tmp.groupby(pd.Grouper(key="week", freq="W"))[chosen].mean().reset_index()
    weekly_means = weekly_means.sort_values("week")

    weekly_scaled = weekly_means.copy()
    for f in chosen:
        x = weekly_means[f].to_numpy(dtype=float)
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        weekly_scaled[f] = (x - mn) / denom

    weekly_plot = weekly_scaled.copy()
    if use_rolling:
        for f in chosen:
            weekly_plot[f] = weekly_plot[f].rolling(rolling_window, min_periods=1).mean()

    weekly_long = weekly_plot.melt(id_vars="week", value_vars=chosen, var_name="feature", value_name="value")

    # Use a spotify-ish, muted color mapping:
    # First series green, then greys. (Stable mapping by order)
    series_colors = [SPOTIFY_GREEN, "#4A4A4A", "#7A7A7A", "#A0A0A0", "#C0C0C0", "#2B2B2B", "#8A8A8A", "#B8B8B8"]
    color_map = {f: series_colors[i % len(series_colors)] for i, f in enumerate(chosen)}

    fig = px.line(
        weekly_long,
        x="week",
        y="value",
        color="feature",
        color_discrete_map=color_map,
        title=None
    )
    fig.update_traces(line=dict(width=3), mode="lines")

    y_max = float(weekly_long["value"].max()) if len(weekly_long) else 1.0
    fig.add_annotation(
        x=cutoff_date, y=y_max,
        text="7.10",
        showarrow=True,
        arrowhead=2,
        ax=20, ay=-30,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(color=SPOTIFY_BLACK)
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Week",
        yaxis_title="Normalized value (0-1)",
        legend_title_text="Feature",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=20, b=50),
        height=480
    )
    fig.update_xaxes(showgrid=True, tickformat="%b %Y", tickangle=-30)
    fig.update_yaxes(showgrid=True, zeroline=False)

    y_for_label = float(weekly_long["value"].max()) if len(weekly_long) else None
    fig = add_event_lines(
        fig,
        selected_events if (show_events and selected_events) else [],
        show_labels=show_event_labels,
        y_for_label=y_for_label
    )
    fig = apply_spotify_plotly_layout(fig)

    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Tab 2: Genres (bar facets) - switch to spotify-ish grey palette
# -------------------------
with tabs[2]:
    st.subheader("Weekly Streams per Genre — Before (lighter) vs After (darker) 7.10")

    if not dff["week"].notna().any():
        st.warning("No valid 'week' column.")
        st.stop()
    if "streams" not in dff.columns or dff["streams"].isna().all():
        st.warning("Missing 'streams' column.")
        st.stop()

    genre_k = st.slider("Number of clusters (K)", 5, 14, 8)

    dff_gen, genre_report = add_genres(dff, k=genre_k, random_state=42)
    if not genre_report.get("ok", False):
        st.warning(f"Genre pipeline not available: {genre_report.get('reason', 'Unknown reason')}")
        st.stop()

    clusters_available = sorted([int(x) for x in pd.Series(dff_gen["cluster"]).dropna().unique().tolist()])
    cluster_sel = st.multiselect(
        "Filter clusters (optional)",
        options=clusters_available,
        default=clusters_available
    )
    dd = dff_gen[dff_gen["cluster"].isin(cluster_sel)].copy()

    if "genre" not in dd.columns:
        st.warning("No 'genre' column found (unexpected).")
        st.stop()

    dd["week"] = pd.to_datetime(dd["week"], errors="coerce")
    dd = dd.dropna(subset=["week", "genre", "streams"])
    dd["period"] = np.where(dd["week"] < cutoff_date, "Before 7-Oct", "After 7-Oct")

    weekly = (
        dd.groupby(["week", "genre", "period"], as_index=False)["streams"].sum()
        .sort_values("week")
    )

    weekly["week_label"] = weekly["week"].dt.strftime("%Y-%m-%d")
    weekly["genre_period"] = weekly["genre"].astype(str) + " | " + weekly["period"].astype(str)

    # Spotify-ish greys per genre with one accent green
    genres = sorted(weekly["genre"].unique().tolist())

    # build deterministic greys, then assign green to the top (by total streams) genre
    genre_order = (
        weekly.groupby("genre")["streams"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    top_genre = genre_order[0] if genre_order else (genres[0] if genres else None)

    greys = ["#2B2B2B", "#4A4A4A", "#6B6B6B", "#8A8A8A", "#A0A0A0", "#B3B3B3", "#C8C8C8"]
    base_color = {}
    for i, g in enumerate(genre_order):
        base_color[g] = greys[i % len(greys)]
    if top_genre is not None:
        base_color[top_genre] = SPOTIFY_GREEN

    def to_rgba_hex(hex_color: str, alpha: float):
        hc = hex_color.lstrip("#")
        r = int(hc[0:2], 16)
        g = int(hc[2:4], 16)
        b = int(hc[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    color_map = {}
    for g in genre_order:
        c = base_color[g]
        color_map[f"{g} | After 7-Oct"] = to_rgba_hex(c, 0.90)
        color_map[f"{g} | Before 7-Oct"] = to_rgba_hex(c, 0.30)

    weekly["genre"] = pd.Categorical(weekly["genre"], categories=genre_order, ordered=True)

    week_order_asc = (
        weekly[["week", "week_label"]]
        .drop_duplicates()
        .sort_values("week", ascending=True)["week_label"]
        .tolist()
    )

    fig = px.bar(
        weekly,
        x="streams",
        y="week_label",
        color="genre_period",
        color_discrete_map=color_map,
        facet_row="genre",
        orientation="h",
        barmode="overlay",
        opacity=0.95,
        title=None,
        height=170 * max(1, len(genre_order))
    )

    fig.update_layout(
        template="plotly_white",
        legend_title_text="",
        margin=dict(l=90, r=20, t=20, b=40),
        bargap=0.02,
        bargroupgap=0.0,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=week_order_asc, autorange="reversed")
    fig.update_yaxes(title_text="")
    fig.update_yaxes(matches=None)
    fig.layout.grid = dict(rows=len(genre_order), columns=1, roworder="top to bottom", ygap=0.02)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(tickformat="~s")

    fig = apply_spotify_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Genre pipeline debug (KMeans report)"):
        st.write({k: v for k, v in genre_report.items() if k not in ["cluster_sizes", "centroids"]})
        if "cluster_sizes" in genre_report:
            st.dataframe(genre_report["cluster_sizes"], use_container_width=True)
        if "centroids" in genre_report:
            st.dataframe(genre_report["centroids"], use_container_width=True)


# -------------------------
# Tab 3: Dumbbell + Avatars (HOVER = GROW CIRCLE A LOT)
# -------------------------
with tabs[3]:
    st.subheader("Before vs After — Median Monthly Index per Artist (baseline month = 100)")

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

    # keep your ring palette as-is (it encodes meaning), but make it less saturated
    RING_GREEN = "rgba(29,185,84,1)"     # spotify green
    RING_ORANGE = "rgba(180,180,180,1)"  # grey
    RING_RED = "rgba(90,90,90,1)"        # dark grey
    RING_GRAY = "rgba(140,140,140,0.9)"  # mid grey

    def ring_color(g: str) -> str:
        if g == "pro": return RING_GREEN
        if g == "anti": return RING_RED
        if g == "neutral": return RING_ORANGE
        return RING_GRAY

    try:
        meta_full = load_artist_meta_full(META_PATH)
    except Exception as e:
        st.error(f"Failed to load META: {e}")
        st.stop()

    avatar_lookup = build_avatar_lookup("artists_photos")

    def group_of_artist(artist_name: str) -> str:
        k = norm_key(artist_name)
        if k not in meta_full.index:
            return "unknown"
        if int(meta_full.loc[k, "pro_israel"]) == 1: return "pro"
        if int(meta_full.loc[k, "anti_israel"]) == 1: return "anti"
        if int(meta_full.loc[k, "neutral"]) == 1: return "neutral"
        return "unknown"

    def bytes_to_data_uri_png(png_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    def pil_circle_avatar(image_abs: str, size_px: int = 180):
        if not image_abs or not os.path.exists(image_abs):
            return None
        try:
            from PIL import Image, ImageOps, ImageDraw
            im = Image.open(image_abs)
            im = ImageOps.exif_transpose(im).convert("RGBA")

            w, h = im.size
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            im = im.crop((left, top, left + m, top + m)).resize((size_px, size_px), Image.LANCZOS)

            mask = Image.new("L", (size_px, size_px), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size_px - 1, size_px - 1), fill=255)

            out = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
            out.paste(im, (0, 0), mask=mask)

            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    if "artist_names" not in dff.columns or "streams" not in dff.columns or not dff["week"].notna().any():
        st.warning("Need valid week/streams/artist_names to build the dumbbell chart.")
        st.stop()

    def split_artists(s):
        if pd.isna(s):
            return []
        return [p.strip() for p in str(s).split(",") if p.strip()]

    base_df = dff[["week", "artist_names", "streams"]].copy()
    base_df["artist_list"] = base_df["artist_names"].apply(split_artists)
    base_df = base_df[base_df["artist_list"].map(len) > 0].copy()
    base_df["n_artists"] = base_df["artist_list"].map(len)
    base_df = base_df.explode("artist_list", ignore_index=True).rename(columns={"artist_list": "artist"})
    base_df["artist"] = base_df["artist"].astype(str).str.strip()
    base_df["artist"] = base_df["artist"].apply(canon_artist)
    base_df["artist"] = base_df["artist"].apply(canon_artist)

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

    summary["group"] = summary["artist"].apply(group_of_artist)
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

    missing_files = 0
    failed_open = 0

    hover_x = []
    hover_y = []
    hover_img_idx = []

    if show_avatars:
        x_min = float(np.nanmin(summary[["Before", "After"]].values))
        x_max = float(np.nanmax(summary[["Before", "After"]].values))
        x_range = max(1.0, x_max - x_min)

        # base thumbnails
        x_sizex = max(5.6, 0.105 * x_range)
        x_offset = 0.03 * x_range
        sizey = 1.15

        for _, r in summary.iterrows():
            a = r["artist"]
            k = norm_key(a)

            img_abs = ""
            if k in meta_full.index:
                cand = str(meta_full.loc[k, "image_path_abs"])
                if cand and Path(cand).exists():
                    img_abs = cand

            if not img_abs:
                img_abs = avatar_lookup.get(k, "")

            if not img_abs or not Path(img_abs).exists():
                missing_files += 1
                continue

            avatar_bytes = pil_circle_avatar(img_abs, size_px=190)
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
                sizey=sizey,
                xanchor="left", yanchor="middle",
                layer="above",
                opacity=1.0
            ))

            img_idx = len(fig.layout.images) - 1

            hx = float(r["After"]) + x_offset + 0.50 * x_sizex
            hy = float(r["y"])
            hover_x.append(hx)
            hover_y.append(hy)
            hover_img_idx.append(img_idx)

    # ✅ make the invisible hover "hit area" MUCH bigger
    if hover_x:
        fig.add_trace(go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            showlegend=False,
            marker=dict(size=70, color="rgba(0,0,0,0)"),  # << BIGGER hover area
            customdata=np.array(hover_img_idx, dtype=int),
            hovertemplate="<extra></extra>"
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
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        )
    )

    fig = apply_spotify_plotly_layout(fig)

    chart_h = max(650, 34 * len(summary))
    render_plotly_with_hover_image_zoom(fig, height=chart_h, zoom=3.10)  # << MUCH bigger image on hover

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Artists shown", f"{len(summary)}")
    m2.metric("Median Δ (After−Before)", f"{summary['delta'].median():.1f}")
    m3.metric("Missing image files", f"{missing_files}" if show_avatars else "—")
    m4.metric("Failed to open images", f"{failed_open}" if show_avatars else "—")
