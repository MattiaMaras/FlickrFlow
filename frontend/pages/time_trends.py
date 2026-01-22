import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

@st.cache_data
def load_time_data():
    base_path = os.path.join(parent_dir, "..", "data", "flickr_time_features.parquet")
    if not os.path.exists(base_path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(base_path)
        return df
    except Exception as e:
        st.error(f"Errore lettura parquet: {e}")
        return pd.DataFrame()


def render():
    st.title("📈 Analisi Temporale (Time Trends)")
    st.markdown("Esplorazione dei pattern temporali: **Stagionalità**, **Orari di punta** e **Abitudini settimanali**.")

    df = load_time_data()

    if df.empty:
        st.error("⚠️ File 'flickr_time_features.parquet' non trovato. Esegui Fase 5 del Backend.")
        return

    if 'roi_name' in df.columns:
        df = df.rename(columns={'roi_name': 'roi'})

    with st.sidebar:
        st.header("⏳ Filtri Temporali")

        unique_rois = sorted([x for x in df['roi'].unique() if x != 'Unknown'])
        selected_rois = st.multiselect("📍 Filtra per Luogo (ROI)", unique_rois,
                                       default=unique_rois[:2] if len(unique_rois) > 1 else unique_rois)

        # Filtro Anno
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            selected_years = st.multiselect("📅 Filtra Anni", years, default=years)
        else:
            selected_years = []

    filtered_df = df.copy()

    # Filtro ROI
    if selected_rois:
        filtered_df = filtered_df[filtered_df['roi'].isin(selected_rois)]
    else:
        st.warning("Seleziona almeno un Luogo dalla sidebar.")
        return

    # Filtro Anni
    if selected_years and 'year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['year'].astype(str).isin([str(y) for y in selected_years])]

    st.info(f"Analisi su **{len(filtered_df):,}** foto filtrate.")

    # Grafico
    st.subheader("🗓️ Stagionalità: Distribuzione Mensile")

    # Raggruppamento per mese e ROI
    monthly_counts = filtered_df.groupby(['month', 'roi']).size().reset_index(name='count')

    fig_month = px.line(
        monthly_counts,
        x='month',
        y='count',
        color='roi',
        markers=True,
        title="Trend Mensile (Turismo Estivo vs Invernale)",
        labels={'month': 'Mese', 'count': 'Numero Foto', 'roi': 'Luogo'}
    )
    fig_month.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    st.plotly_chart(fig_month, width='stretch')

    # Ritmo Giornaliero
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⏰ Ritmo Giornaliero (Ore)")
        hourly_counts = filtered_df.groupby(['hour', 'roi']).size().reset_index(name='count')

        fig_hour = px.area(
            hourly_counts,
            x='hour',
            y='count',
            color='roi',
            title="Distribuzione Oraria (Giorno vs Notte)",
            labels={'hour': 'Ora del Giorno (0-23)', 'count': 'Volume Foto'}
        )
        st.plotly_chart(fig_hour, width='stretch')

    with c2:
        st.subheader("📅 Giorni della Settimana")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        daily_counts = filtered_df.groupby(['day_name']).size().reindex(days_order).reset_index(name='count')

        fig_day = px.bar(
            daily_counts,
            x='day_name',
            y='count',
            color='count',  # Gradiente
            color_continuous_scale='Blues',
            title="Popolarità per Giorno della Settimana"
        )
        st.plotly_chart(fig_day, width='stretch')

    # Heatmap (Giorno x Ora)
    st.markdown("---")
    st.subheader("🔥 Heatmap: Quando scattano le foto?")
    st.caption("Incrocio tra Giorno della Settimana e Ora del Giorno. I colori caldi indicano i momenti di massima affluenza.")

    heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')

    heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').reindex(days_order).fillna(0)

    fig_heat = px.imshow(
        heatmap_pivot,
        labels=dict(x="Ora del Giorno", y="Giorno", color="Densità"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_heat, width='stretch')