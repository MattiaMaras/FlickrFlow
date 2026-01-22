import streamlit as st
import pydeck as pdk
import pandas as pd
import plotly.express as px
import sys
import os
import math
from shapely.geometry import MultiPoint, mapping

try:
    from backend.config import ROME_ROIS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from backend.config import ROME_ROIS

try:
    from data_loader import load_clusters
except ImportError as e:
    st.error(f"Errore di importazione: {e}")
    st.stop()

# Centroidi
KNOWN_CENTROIDS = {}
for name, coords in ROME_ROIS.items():
    center_lat = (coords["lat_min"] + coords["lat_max"]) / 2
    center_lon = (coords["lon_min"] + coords["lon_max"]) / 2
    KNOWN_CENTROIDS[name] = (center_lat, center_lon)

# Colori per i cluster
CLUSTER_COLORS = {
    0: [255, 0, 0, 140], 1: [0, 255, 0, 140], 2: [0, 0, 255, 140],
    3: [255, 255, 0, 140], 4: [0, 255, 255, 140], 5: [255, 0, 255, 140],
    6: [255, 128, 0, 140], 7: [128, 0, 255, 140], 8: [0, 128, 128, 140],
    9: [128, 128, 128, 140]
}


def get_color(cluster_id):
    cid = int(cluster_id)
    return CLUSTER_COLORS.get(cid % 10, [200, 200, 200, 140])


def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def get_cluster_label(cluster_df, cluster_id):
    center_lat = float(cluster_df["lat"].mean())
    center_lon = float(cluster_df["lon"].mean())

    closest_roi = "Zona Periferica"
    min_dist = float("inf")

    for roi_name, (roi_lat, roi_lon) in KNOWN_CENTROIDS.items():
        dist = calculate_distance(center_lat, center_lon, roi_lat, roi_lon)
        if dist < min_dist:
            min_dist = dist
            closest_roi = roi_name

    # Soglia 1.5km
    if min_dist > 0.015:
        return f"Zona Esterna (Cl. {cluster_id})"

    return f"{closest_roi} (Cl. {cluster_id})"


def render():
    st.title("🧠 Analisi Cluster")

    df = load_clusters()

    if df.empty:
        st.error("Dataset cluster mancante")
        return

    # Normalizzazione Nomi
    rename_map = {"prediction": "cluster", "latitude": "lat", "longitude": "lon"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.dropna(subset=["lat", "lon"])

    df["cluster"] = df["cluster"].apply(lambda x: int(x))
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Logica Naming
    meta = []
    cluster_labels_map = {}
    unique_clusters = sorted(df["cluster"].unique())  # unique() ritorna numpy array

    for c_numpy in unique_clusters:
        c = int(c_numpy)
        subset = df[df["cluster"] == c]

        roi_name = get_cluster_label(subset, c)
        cluster_labels_map[c] = roi_name

        rgb = CLUSTER_COLORS.get(c % 10, [200, 200, 200])
        meta.append({
            "ID": c,
            "Zona Identificata": roi_name,
            "Foto": int(len(subset)),
            "Colore": f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        })

    stats_df = pd.DataFrame(meta).sort_values("Foto", ascending=False)

    # Selezione
    col1, col2 = st.columns([2, 1])
    with col2:
        st.subheader("📍 Riconoscimento")
        st.dataframe(
            stats_df[["Zona Identificata", "Foto"]],
            width='stretch',
            height=350,
            column_config={"Foto": st.column_config.ProgressColumn(format="%d", max_value=int(stats_df["Foto"].max()))}
        )
        sel = st.selectbox("Isola Cluster", ["Tutti"] + list(stats_df["Zona Identificata"]))

    # Filtro Cluster
    if sel != "Tutti":
        target_id = int(stats_df[stats_df["Zona Identificata"] == sel]["ID"].values[0])
        with st.spinner(f"🔍 Isolamento cluster {target_id}..."):
            map_df = df[df["cluster"] == target_id].copy()
        zoom = 13
    else:
        map_df = df.copy()
        zoom = 11

    # Sampling
    if len(map_df) > 40000:
        with st.spinner("⚙️ Ottimizzazione rendering (sampling intelligente)..."):
            map_df = map_df.sample(40000, random_state=42)
        st.info(f"📊 Visualizzati 40.000 punti rappresentativi su {len(df):,} totali")


    clean_map_data = []
    for _, row in map_df.iterrows():
        c_id = int(row["cluster"])
        clean_map_data.append({
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "cluster": c_id,
            "color": get_color(c_id),
            "label": cluster_labels_map.get(c_id, f"Cluster {c_id}")
        })

    # Generazione Poligoni
    features = []
    unique_visible_clusters = set(d["cluster"] for d in clean_map_data)

    for c in unique_visible_clusters:
        pts = [p for p in clean_map_data if p["cluster"] == c]
        coords = [(p["lon"], p["lat"]) for p in pts]

        if len(coords) < 4: continue

        try:
            poly = MultiPoint(coords).convex_hull
            if poly.geom_type == 'Polygon':
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "cluster": c,
                        "color": get_color(c),
                        "name": cluster_labels_map.get(c, f"Cl {c}")
                    }
                })
        except Exception:
            continue

    # Visualizzazione
    layers = []

    # Poligoni
    if features:
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            {"type": "FeatureCollection", "features": features},
            stroked=True,
            filled=True,
            get_fill_color="properties.color",
            get_line_color=[255, 255, 255],
            get_line_width=20,
            opacity=0.2,
            pickable=False
        ))

    # Punti
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=clean_map_data,  # Usiamo i dati puliti
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=30,
        pickable=True,
        opacity=0.6
    ))

    view_lat = float(map_df["lat"].mean())
    view_lon = float(map_df["lon"].mean())

    tooltip = {
        "html": "<b>{label}</b>",
        "style": {"backgroundColor": "#111", "color": "white"}
    }

    with col1:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=view_lat,
                longitude=view_lon,
                zoom=zoom
            ),
            layers=layers,
            tooltip=tooltip
        ))

    # Grafico
    st.markdown("---")
    fig = px.bar(
        stats_df,
        x="Zona Identificata",
        y="Foto",
        text="Foto",
        title="Densità dei Cluster Identificati"
    )
    fig.update_traces(marker_color=stats_df["Colore"], textposition='auto')
    fig.update_layout(xaxis_title=None, showlegend=False)
    st.plotly_chart(fig, width='stretch')