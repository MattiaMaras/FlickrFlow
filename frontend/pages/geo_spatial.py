import streamlit as st
import pydeck as pdk
import pandas as pd
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_loader import load_main_data
except ImportError as e:
    st.error(f"Errore di importazione: {e}")
    st.stop()

def render():
    st.title("🌍 Analisi Geo-Spaziale 3D")

    # 1. Caricamento Dati
    df = load_main_data()
    if df.empty:
        st.error("⚠️ Dati mancanti.")
        return

    df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'roi_name': 'roi'})
    if 'roi' not in df.columns:
        st.error("Errore: Colonna ROI mancante.")
        return

    df = df[df['roi'] != 'Unknown']
    df = df.dropna(subset=['lat', 'lon'])

    with st.sidebar:
        st.header("🔧 Filtri Dati")

        # Filtro Anni
        if 'year' in df.columns:
            all_years = sorted(df['year'].unique())
            sel_years = st.multiselect("Seleziona Anni", all_years, default=all_years)
        else:
            sel_years = []

        # Filtro Mesi
        if 'month' in df.columns:
            all_months = sorted(df['month'].unique())
            sel_months = st.multiselect("Seleziona Mesi", all_months, default=all_months)
        else:
            sel_months = []

        # Filtro Utenti
        if 'user_type' in df.columns:
            all_types = df['user_type'].unique()
            sel_types = st.multiselect("Tipo Utente", all_types, default=all_types)
        else:
            sel_types = []

        st.markdown("---")
        st.header("🔦 Spotlight ROI")

        # Tendina ROI Specifica
        unique_rois = sorted(df['roi'].unique())
        spotlight = st.selectbox("Evidenzia Monumento", ["Tutta Roma"] + unique_rois)

        st.markdown("---")
        st.header("📐 Geometria")
        radius = st.slider("Raggio Esagono", 10, 80, 20)
        elevation_scale = st.slider("Altezza Torri", 1, 50, 15)

    # Filtri
    filtered_df = df.copy()

    if sel_years:
        filtered_df = filtered_df[filtered_df['year'].isin(sel_years)]
    if sel_months:
        filtered_df = filtered_df[filtered_df['month'].isin(sel_months)]
    if sel_types:
        filtered_df = filtered_df[filtered_df['user_type'].isin(sel_types)]

    # Visualizzazione con dati ridotti per evitare crash o troppo uso di memoria
    MAX_POINTS = 500000
    original_count = len(filtered_df)

    if spotlight != "Tutta Roma":
        filtered_df = filtered_df[filtered_df['roi'] == spotlight]
        st.info(f"📍 Focus su: **{spotlight}** ({len(filtered_df)} foto)")
    else:
        if original_count > MAX_POINTS:
            filtered_df = filtered_df.sample(n=MAX_POINTS, random_state=42)
            st.info(
                f"📊 Densità elevata: trovate **{original_count:,}** foto. Visualizzate le **{MAX_POINTS:,}** più rilevanti.")
        else:
            st.success(f"✅ Visualizzando tutte le **{original_count:,}** foto filtrate.")


    map_data = filtered_df[['lat', 'lon', 'roi']].copy()

    # Avendo fatto .copy posso modificare le colonne senza warning
    map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
    map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')

    # Rimozione eventuali NaN generati dalla conversione
    map_data = map_data.dropna()

    roi_centers = map_data.groupby("roi")[["lat", "lon"]].mean().reset_index()

    layers = []

    # Esagoni
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=map_data,
        get_position=["lon", "lat"],
        radius=radius,
        elevation_scale=elevation_scale,
        elevation_range=[0, 800],
        pickable=False,
        extruded=True,
        coverage=1,
        auto_highlight=True,
        material={"ambient": 0.6, "diffuse": 0.6, "shininess": 32}
    )
    layers.append(hex_layer)

    # Scatterplot
    roi_layer = pdk.Layer(
        "ScatterplotLayer",
        roi_centers,
        get_position=["lon", "lat"],
        get_radius=80,
        get_fill_color=[0, 128, 255, 200],
        get_line_color=[255, 255, 255],
        get_line_width=20,
        pickable=True,
        opacity=0.9,
        stroked=True
    )
    layers.append(roi_layer)

    # Testo
    if spotlight == "Tutta Roma":
        top_labels = map_data['roi'].value_counts().head(50).index.tolist()
        text_data = roi_centers[roi_centers['roi'].isin(top_labels)]
    else:
        text_data = roi_centers

    text_layer = pdk.Layer(
        "TextLayer",
        text_data,
        get_position=["lon", "lat"],
        get_text="roi",
        get_color=[255, 255, 255],
        get_size=14,
        get_alignment_baseline="'top'",
        get_background_color=[0, 0, 0, 150],
        background=True
    )
    layers.append(text_layer)

    tooltip_html = """
    <div style="background:#1E1E1E; padding:8px; color:white; border-radius:4px; border:1px solid #333;">
        <b style="font-size:1.1em">🏛️ {roi}</b><br>
        <span style="color:#aaa; font-size:0.9em">📍 ROI Center</span>
    </div>
    """

    # Mappa
    view_lat = map_data['lat'].mean() if not map_data.empty else 41.89
    view_lon = map_data['lon'].mean() if not map_data.empty else 12.49

    zoom_level = 14 if spotlight != "Tutta Roma" else 12

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=view_lat,
            longitude=view_lon,
            zoom=zoom_level,
            pitch=50,
            bearing=0
        ),
        layers=layers,
        tooltip={"html": tooltip_html}
    ))

    # Footer
    c1, c2, c3 = st.columns([3, 1.5, 1.5])

    with c1:
        st.caption("🔴 Densità  · 🔵 Region Of Interest · 🏷️ Etichette")

    with c2:
        if spotlight == "Tutta Roma":
            st.caption("🖱️ Shift + Drag per ruotare")

    with c3:
        st.caption("🔍 Scroll per zoom")