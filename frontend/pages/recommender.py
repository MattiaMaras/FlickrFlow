import streamlit as st
import pandas as pd
import pydeck as pdk
import sys
import os

try:
    from backend.config import ROI_COORDS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from backend.config import ROI_COORDS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def load_rules():
    path = os.path.join(parent_dir, "..", "data", "flickr_rules.parquet")
    if os.path.exists(path): return pd.read_parquet(path)
    return pd.DataFrame()


def render():
    st.title("🔮 AI Predictor: Next-Step")
    st.markdown("""
    **Motore Prescrittivo:** Questo modulo utilizza le regole di associazione (FP-Growth) 
    per suggerire la prossima tappa ideale in base alla tua posizione attuale.
    """)

    df_rules = load_rules()

    if df_rules.empty:
        st.error("⚠️ Regole mancanti. Esegui Fase 6 del Backend.")
        return

    col_input, col_pred = st.columns([1, 2])

    with col_input:
        st.markdown("### 📍 Dove ti trovi?")
        available_sources = set()

        for _, row in df_rules.iterrows():
            for item in row['antecedent']:
                if item in ROI_COORDS:
                    available_sources.add(item)

        if not available_sources:
            st.warning("Nessuna delle tue Regioni di Interesse ha regole di partenza sufficienti.")
            return

        current_loc = st.selectbox("Seleziona posizione:", sorted(list(available_sources)))

        st.info("L'AI calcola la probabilità di spostamento basandosi sui flussi storici degli utenti.")

    # Motore di Raccomandazione
    suggestions = []

    # Regole dove l'antecedente contiene la posizione attuale
    for _, row in df_rules.iterrows():
        ant = row['antecedent']
        cons = row['consequent']

        if current_loc in ant:
            target = cons[0]  # Prima destinazione
            if target in ROI_COORDS and target != current_loc:
                suggestions.append({
                    "target": target,
                    "prob": row['confidence'],
                    "lift": row['lift'],
                    "coords": ROI_COORDS[target]
                })

    # Ordinamento per probabilità, prendiamo i migliori unici
    suggestions = sorted(suggestions, key=lambda x: x['prob'], reverse=True)

    # Se ci sono più regole che portano allo stesso posto, consideriamo la migliore
    seen = set()
    top_picks = []
    for s in suggestions:
        if s['target'] not in seen:
            top_picks.append(s)
            seen.add(s['target'])

    # Top 3
    top_picks = top_picks[:3]

    with col_pred:
        if not top_picks:
            st.warning(f"Non ho abbastanza dati storici per predire dove andare dopo **{current_loc}**.")
        else:
            best = top_picks[0]

            # Card Principale
            st.success(f"### 🚀 Consigliato: **{best['target']}**")
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Probabilità", f"{best['prob'] * 100:.1f}%")
            kpi2.metric("Lift (Forza)", f"{best['lift']:.2f}x")

            # Alternative
            if len(top_picks) > 1:
                st.markdown("**Alternative interessanti:**")
                for alt in top_picks[1:]:
                    st.text(f"• {alt['target']} ({alt['prob'] * 100:.1f}%)")

    # Mappa Predittiva 3D
    if top_picks:
        start_coords = ROI_COORDS[current_loc]

        arc_layers = []
        points = [{"pos": start_coords, "name": "TU SEI QUI", "color": [255, 0, 0, 255], "radius": 200}]

        for pick in top_picks:
            # Colore arco: Verde acceso se prob > 25%, Giallo altrimenti
            color = [0, 255, 128, 180] if pick['prob'] > 0.25 else [255, 200, 0, 180]

            arc_layers.append({
                "source": start_coords,
                "target": pick['coords'],
                "name": f"Verso {pick['target']}",
                "width": pick['prob'] * 15,  # Spessore dinamico
                "color": color
            })

            # Punto di arrivo
            points.append({
                "pos": pick['coords'],
                "name": pick['target'],
                "color": [0, 255, 128, 255],
                "radius": 150
            })

        # Layer Archi
        layer_arcs = pdk.Layer(
            "ArcLayer",
            data=arc_layers,
            get_source_position="source",
            get_target_position="target",
            get_source_color=[255, 0, 0, 200],
            get_target_color="color",
            get_width="width",
            get_tilt=15,
        )

        # Layer Punti
        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position="pos",
            get_fill_color="color",
            get_radius="radius",
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=start_coords[1],
            longitude=start_coords[0],
            zoom=12.5,
            pitch=45,
            bearing=0
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer_arcs, layer_points],
            initial_view_state=view_state,
            tooltip={"html": "<b>{name}</b>"}
        ))