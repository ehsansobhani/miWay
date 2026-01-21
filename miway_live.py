import os
import zipfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
from google.transit import gtfs_realtime_pb2

# 
# way_live.py (fancy demo basemap + fallback)
import zipfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
from google.transit import gtfs_realtime_pb2

import streamlit as st

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .badge-mask {
        position: fixed;
        right: 0;
        bottom: 0;
        width: 260px;
        height: 120px;
        background: #ffffff;
        z-index: 999999;
        pointer-events: none;
        display: flex;
        align-items: flex-end;
        justify-content: flex-end;
        padding: 12px;
        font-family: Arial, sans-serif;
        font-weight: 700;
        color: #333;
    }
    </style>

    <div class="badge-mask">MiWay Live</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* --- Kill ALL Streamlit Cloud floating badges/decoration --- */

    /* 1) Hide Streamlit's floating decoration container (new + old builds) */
    [data-testid="stDecoration"],
    [data-testid="stAppDecoration"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* 2) Hide any fixed bottom-right elements Streamlit injects */
    div[style*="position: fixed"][style*="bottom: 0"][style*="right: 0"] {
        display: none !important;
    }

    /* 3) Hide any streamlit.io marketing links/badges */
    a[href*="streamlit.io"],
    a[href*="share.streamlit.io"] {
        display: none !important;
    }

    /* 4) Hide footer + remove extra padding */
    footer {display: none !important;}
    .block-container {padding-bottom: 0rem !important; padding-top: 0rem !important;}

    /* 5) Hide header space if present */
    header {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Hide Streamlit header */
    header {visibility: hidden;}

    /* Hide footer */
    footer {visibility: hidden;}

    /* Remove top padding */
    .block-container {
        padding-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* --- Hide Streamlit Cloud bottom-right branding --- */
    a[href*="streamlit.io"] {
        display: none !important;
    }

    /* Hide 'Hosted with Streamlit' badge */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Hide footer completely */
    footer {
        display: none !important;
    }

    /* Remove extra bottom padding */
    .block-container {
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# MiWay config
# -----------------------------
AGENCY_NAME = "MiWay (Mississauga)"
GTFS_STATIC_ZIP_URL = "https://www.miapp.ca/GTFS/google_transit.zip"
GTFSRT_VEHICLES_URL = "https://www.miapp.ca/GTFS_RT/Vehicle/VehiclePositions.pb"
GTFSRT_TRIPS_URL    = "https://www.miapp.ca/GTFS_RT/TripUpdate/TripUpdates.pb"
GTFSRT_ALERTS_URL   = "https://www.miapp.ca/gtfs_rt/Alerts/Alerts.pb"  # optional

DEFAULT_CENTER = (43.5890, -79.6441)
DEFAULT_ZOOM = 11.8

# Where to store static GTFS after download/extract
DATA_DIR = Path(__file__).parent / "data"
GTFS_FOLDER = DATA_DIR / "miway_gtfs"


# -----------------------------
# Basemap styles
# -----------------------------
# MAPBOX_STYLES = {
#     "Mapbox Light (clean)": "mapbox://styles/mapbox/light-v11",
#     "Mapbox Dark (demo)": "mapbox://styles/mapbox/dark-v11",
#     "Mapbox Streets": "mapbox://styles/mapbox/streets-v12",
#     "Mapbox Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
# }
MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"

# Free (no token) fallback styles that usually work anywhere
# CARTO_STYLES = {
#     "Carto Voyager (free)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
#     "Carto Positron (free)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#     "Carto DarkMatter (free)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
# }


# -----------------------------
# Utilities
# -----------------------------
def _parse_feed(url: str, timeout: int = 20) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    r = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (MiWay GTFS-RT Viewer)",
            "Accept": "application/x-protobuf, application/octet-stream;q=0.9, */*;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    r.raise_for_status()
    feed.ParseFromString(r.content)
    return feed


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def ensure_static_gtfs(gtfs_zip_url: str, out_folder: str) -> str:
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    zip_path = out_path / "gtfs.zip"
    r = requests.get(gtfs_zip_url, timeout=90)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_path)

    needed = ["routes.txt", "trips.txt", "shapes.txt"]
    missing = [f for f in needed if not (out_path / f).exists()]
    if missing:
        raise RuntimeError(f"GTFS extracted but missing files: {missing}")

    return str(out_path)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_static_gtfs(gtfs_folder: str):
    gtfs_path = Path(gtfs_folder)
    routes = pd.read_csv(gtfs_path / "routes.txt")
    trips  = pd.read_csv(gtfs_path / "trips.txt")
    shapes = pd.read_csv(gtfs_path / "shapes.txt")

    for c in ["route_id"]:
        if c in routes.columns: routes[c] = routes[c].astype("string")
        if c in trips.columns:  trips[c]  = trips[c].astype("string")
    if "trip_id" in trips.columns:  trips["trip_id"]  = trips["trip_id"].astype("string")
    if "shape_id" in trips.columns: trips["shape_id"] = trips["shape_id"].astype("string")
    if "shape_id" in shapes.columns: shapes["shape_id"] = shapes["shape_id"].astype("string")

    return routes, trips, shapes


@st.cache_data(ttl=24*3600, show_spinner=False)
def load_trip_direction_map(gtfs_folder: str) -> dict:
    trips_path = Path(gtfs_folder) / "trips.txt"
    trips = pd.read_csv(trips_path)

    if "trip_id" not in trips.columns:
        return {}

    trips["trip_id"] = trips["trip_id"].astype("string")

    if "direction_id" in trips.columns:
        t = trips[["trip_id", "direction_id"]].copy()
        return dict(zip(t["trip_id"], t["direction_id"]))
    return {}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def build_route_paths(gtfs_folder: str, route_id: str) -> pd.DataFrame:
    routes, trips, shapes = load_static_gtfs(gtfs_folder)

    t = trips[trips["route_id"] == str(route_id)].copy()
    if t.empty:
        return pd.DataFrame(columns=["direction_id", "shape_id", "path", "name"])

    if "direction_id" not in t.columns:
        t["direction_id"] = 0

    reps = (
        t.dropna(subset=["shape_id"])
         .groupby(["direction_id", "shape_id"])
         .size()
         .reset_index(name="n")
         .sort_values(["direction_id", "n"], ascending=[True, False])
    )
    if reps.empty:
        return pd.DataFrame(columns=["direction_id", "shape_id", "path", "name"])

    reps = reps.groupby("direction_id", as_index=False).head(1)

    rrow = routes[routes["route_id"] == str(route_id)]
    rname = str(route_id)
    if not rrow.empty:
        short = rrow.iloc[0].get("route_short_name", "")
        long  = rrow.iloc[0].get("route_long_name", "")
        if pd.notna(short) and str(short).strip():
            rname = str(short).strip()
        elif pd.notna(long) and str(long).strip():
            rname = str(long).strip()

    out_rows = []
    for _, row in reps.iterrows():
        sid = str(row["shape_id"])
        did = int(row["direction_id"])

        s = shapes[shapes["shape_id"] == sid].copy()
        if s.empty:
            continue
        if "shape_pt_sequence" in s.columns:
            s = s.sort_values("shape_pt_sequence")

        path = s[["shape_pt_lon", "shape_pt_lat"]].astype(float).values.tolist()
        out_rows.append({
            "direction_id": did,
            "shape_id": sid,
            "path": path,
            "name": f"Route {rname} (dir {did})",
        })

    return pd.DataFrame(out_rows)


# -----------------------------
# Realtime fetchers
# -----------------------------
def fetch_vehicle_positions(timeout: int = 20) -> pd.DataFrame:
    feed = _parse_feed(GTFSRT_VEHICLES_URL, timeout=timeout)
    feed_ts = int(feed.header.timestamp) if feed.header and feed.header.timestamp else None

    rows = []
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        if not v.HasField("position"):
            continue

        pos = v.position
        if pos.latitude == 0 or pos.longitude == 0:
            continue

        trip_id = v.trip.trip_id if v.HasField("trip") else None
        route_id = v.trip.route_id if v.HasField("trip") else None

        direction_id = None
        if v.HasField("trip") and v.trip.HasField("direction_id"):
            direction_id = int(v.trip.direction_id)

        ts = int(v.timestamp) if v.HasField("timestamp") and v.timestamp else feed_ts

        rows.append({
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
            "trip_id": trip_id,
            "route_id": route_id,
            "direction_id": direction_id,
            "lat": float(pos.latitude),
            "lon": float(pos.longitude),
            "bearing": float(pos.bearing) if pos.HasField("bearing") else None,
            "speed": float(pos.speed) if pos.HasField("speed") else None,
            "ts": ts,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["route_id"] = df["route_id"].astype("string")
        df["trip_id"]  = df["trip_id"].astype("string")
    return df


def fetch_trip_updates_delay(timeout: int = 20) -> pd.DataFrame:
    feed = _parse_feed(GTFSRT_TRIPS_URL, timeout=timeout)

    rows = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue

        tu = ent.trip_update
        trip_id = tu.trip.trip_id if tu.HasField("trip") else None
        route_id = tu.trip.route_id if tu.HasField("trip") else None

        delay_s = None
        if tu.HasField("delay"):
            delay_s = int(tu.delay)
        else:
            delays = []
            for stu in tu.stop_time_update:
                if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                    delays.append(int(stu.arrival.delay))
                if stu.HasField("departure") and stu.departure.HasField("delay"):
                    delays.append(int(stu.departure.delay))
            if delays:
                delay_s = int(np.median(delays))

        rows.append({
            "trip_id": trip_id,
            "route_id": route_id,
            "delay_s": delay_s,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["route_id"] = df["route_id"].astype("string")
        df["trip_id"]  = df["trip_id"].astype("string")
        df["delay_s"]  = pd.to_numeric(df["delay_s"], errors="coerce")
    return df


@st.cache_data(ttl=6, show_spinner=False)
def get_live_positions() -> pd.DataFrame:
    return fetch_vehicle_positions()


@st.cache_data(ttl=6, show_spinner=False)
def get_live_delays() -> pd.DataFrame:
    return fetch_trip_updates_delay()


# -----------------------------
# Delay class + colors
# -----------------------------
def add_delay_class_and_color(df: pd.DataFrame, ontime_window_s: int = 60) -> pd.DataFrame:
    out = df.copy()
    out["delay_s"] = pd.to_numeric(out["delay_s"], errors="coerce")

    def cls(d):
        if pd.isna(d):
            return "unknown"
        if d < -ontime_window_s:
            return "early"
        if d > ontime_window_s:
            return "late"
        return "on_time"

    out["delay_class"] = out["delay_s"].apply(cls)

    color_map = {
        "early":   [30, 90, 220],
        "on_time": [30, 170, 80],
        "late":    [220, 50, 50],
        "unknown": [160, 160, 160],
    }
    out["color"] = out["delay_class"].map(color_map)
    return out


# -----------------------------
# Headway estimate (dominant corridor ordering)
# -----------------------------
def _corridor_order_score(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    X = np.column_stack([lon, lat]).astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        axis = vt[0]
    except Exception:
        axis = np.array([0.0, 1.0])
    return X @ axis


def add_headway_estimates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out["corridor_pos"] = np.nan
    out["headway_min"] = np.nan

    for (route_id, direction_id), g in out.groupby(["route_id", "direction_id"], dropna=False):
        idx = g.index
        score = _corridor_order_score(g["lat"].to_numpy(), g["lon"].to_numpy())
        out.loc[idx, "corridor_pos"] = score

        if g["ts"].notna().sum() < 2:
            continue

        gg = g.copy()
        gg["corridor_pos"] = score
        gg = gg.sort_values("corridor_pos")

        ts = pd.to_numeric(gg["ts"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(ts) < 2:
            continue

        diffs = np.diff(ts)
        diffs = diffs[(diffs > 0) & (diffs < 3600)]
        if len(diffs) == 0:
            continue

        out.loc[idx, "headway_min"] = float(np.median(diffs) / 60.0)

    return out


def compute_online_metrics(df: pd.DataFrame) -> dict:
    out = {"vehicles": int(len(df))}
    d = pd.to_numeric(df["delay_s"], errors="coerce").dropna()
    if len(d):
        out["median_delay_s"] = float(np.median(d))
        out["pct_late"] = float((d > 60).mean() * 100)
        out["pct_early"] = float((d < -60).mean() * 100)
        out["pct_ontime"] = float((np.abs(d) <= 60).mean() * 100)
    else:
        out["median_delay_s"] = None
        out["pct_late"] = None
        out["pct_early"] = None
        out["pct_ontime"] = None
    h = pd.to_numeric(df["headway_min"], errors="coerce").dropna()
    out["headway_min"] = float(np.median(h)) if len(h) else None
    return out


# -----------------------------
# Streamlit app (fancy demo)
# -----------------------------
st.set_page_config(layout="wide")
st.title(f"Live {AGENCY_NAME} Vehicles — Route, Delay, Headway")

# Faster refresh for demo animation
st_autorefresh(interval=5_000, key="refresh")

with st.sidebar:
    # st.subheader("Basemap (fancy)")
    # mapbox_token = st.text_input(
    #     "Mapbox token (recommended)",
    #     value=os.getenv("MAPBOX_TOKEN", "pk.eyJ1IjoiZWhzYW5zb2JoYW5pIiwiYSI6ImNtZGQzYzI3djAwZnEya3B3eWxpY3V1dHoifQ.aBQj-dW2JwCll1GituWVsg"),
    #     type="password",
    #     help="Set MAPBOX_TOKEN env var or paste token here.",
    # )

    # basemap_family = st.radio("Basemap provider", ["Mapbox (needs token)", "Carto (free, no token)"], index=0)

    # if basemap_family.startswith("Mapbox"):
    #     style_name = st.selectbox("Mapbox style", list(MAPBOX_STYLES.keys()), index=1)
    #     map_style = MAPBOX_STYLES[style_name]
    # else:
    #     style_name = st.selectbox("Carto style", list(CARTO_STYLES.keys()), index=0)
    #     map_style = CARTO_STYLES[style_name]

    st.divider()

    # st.subheader("Static")
    use_route_lines = st.checkbox("Show route", value=True)

    # st.subheader("Filters")
    only_with_trip = st.checkbox("Only vehicles with trip_id", value=True)

    # st.subhead/ = st.checkbox("Demo mode (pulse + labels)", value=True)

    # st.subheader("Table")
    show_table = st.checkbox("Show table", value=True)


# Static GTFS (cached)
gtfs_folder = ensure_static_gtfs(GTFS_STATIC_ZIP_URL, str(GTFS_FOLDER))
trip_dir_map = load_trip_direction_map(gtfs_folder)

# Realtime
df_pos = get_live_positions()
df_delay = get_live_delays()

if df_pos.empty:
    st.error("No vehicles returned right now.")
    st.stop()

if only_with_trip:
    df_pos = df_pos[df_pos["trip_id"].notna()].copy()

# Merge delays
if not df_delay.empty:
    df = df_pos.merge(df_delay[["trip_id", "delay_s"]], on="trip_id", how="left")
else:
    df = df_pos.copy()
    df["delay_s"] = np.nan

df["delay_s"] = pd.to_numeric(df["delay_s"], errors="coerce")
df["direction_id"] = df["direction_id"].fillna(df["trip_id"].map(trip_dir_map))

# Route filter
route_ids = sorted([r for r in df["route_id"].dropna().unique().tolist()])
selected_route = st.selectbox("Route", ["ALL"] + route_ids)

if selected_route != "ALL":
    df = df[df["route_id"] == selected_route].copy()

# Route line
route_paths_df = pd.DataFrame()
if use_route_lines and selected_route != "ALL":
    try:
        route_paths_df = build_route_paths(gtfs_folder, selected_route)
    except Exception as e:
        st.warning(f"Could not load shapes for route {selected_route}: {e}")
        route_paths_df = pd.DataFrame()

# Headway + colors
df = add_headway_estimates(df)
df = add_delay_class_and_color(df, ontime_window_s=60)
metrics = compute_online_metrics(df)

# Center
if len(df):
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
else:
    center_lat, center_lon = DEFAULT_CENTER

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles online", f"{metrics['vehicles']}")
c2.metric("Median delay", "—" if metrics["median_delay_s"] is None else f"{metrics['median_delay_s']:.0f}s")
c3.metric("Headway", "—" if metrics["headway_min"] is None else f"{metrics['headway_min']:.1f} min")
c4.metric("On-time / Late / Early", "—" if metrics["pct_late"] is None else f"{metrics['pct_ontime']:.0f}% / {metrics['pct_late']:.0f}% / {metrics['pct_early']:.0f}%")

# Layers
layers = []

# Route line (thicker and clearer for demo)
if use_route_lines and not route_paths_df.empty:
    layers.append(
        pdk.Layer(
            "PathLayer",
            route_paths_df,
            get_path="path",
            get_color=[255, 60, 60],
            get_width=5,
            width_scale=1,
            width_min_pixels=2,
            width_max_pixels=6,
            opacity=0.85,
            pickable=False,
        )
    )

# Vehicles
layers.append(
    pdk.Layer(
    "ScatterplotLayer",
    df,
    get_position="[lon, lat]",
    get_radius=7,              # <-- IMPORTANT: make points visible
    radius_units="meters",      # <-- IMPORTANT: stable sizing
    get_fill_color="color",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1,        
    opacity=0.95,
    pickable=True,        
)
)



# Labels (fancy)
# if fancy_mode:
#     df_lbl = df.copy()
#     df_lbl["label"] = df_lbl["route_id"].fillna("") + "  " + df_lbl["vehicle_id"].fillna("")
#     layers.append(
#         pdk.Layer(
#             "TextLayer",
#             df_lbl,
#             get_position="[lon, lat]",
#             get_text="label",
#             get_size=12,
#             get_color=[240, 240, 240],
#             get_angle=0,
#             get_text_anchor="'start'",
#             get_alignment_baseline="'center'",
#             pickable=False,
#         )
#     )

tooltip = {
    "text": (
        "Vehicle {vehicle_id}\n"
        "Route {route_id}\n"
        "Trip {trip_id}\n"
        "Dir {direction_id}\n"
        "Delay {delay_s}s ({delay_class})\n"
        "Headway {headway_min} min"
    )
}

# Deck (Mapbox token only affects Mapbox styles; Carto works without token)
deck = pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=DEFAULT_ZOOM if selected_route == "ALL" else 12.8,
        # pitch=45 if (fancy_mode and basemap_family.startswith("Mapbox")) else 0,
        bearing=0,
    ),
    map_style=MAP_STYLE,

    tooltip=tooltip,
)

# Ensure pydeck has token if Mapbox chosen
# if basemap_family.startswith("Mapbox") and mapbox_token.strip():
#     pdk.settings.mapbox_api_key = mapbox_token.strip()

st.pydeck_chart(deck, use_container_width=True)

if show_table:
    view_df = df.copy()
    view_df["delay_min"] = (pd.to_numeric(view_df["delay_s"], errors="coerce") / 60.0).round(2)
    cols = ["vehicle_id", "route_id", "direction_id", "trip_id", "delay_s", "delay_min", "headway_min", "lat", "lon", "ts"]
    cols = [c for c in cols if c in view_df.columns]
    st.dataframe(
        view_df[cols].sort_values(["route_id", "direction_id", "vehicle_id"], na_position="last"),
        height=360,
        use_container_width=True
    )
