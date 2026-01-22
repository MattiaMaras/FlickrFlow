import streamlit as st
import sys
import os
import plotly.express as px
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_loader import load_main_data, load_user_profiles
except ImportError as e:
    st.error(f"Errore di importazione: {e}")
    st.stop()


def render():
    st.title("📊 Project Overview")

    # Caricamento Dati
    try:
        df = load_main_data()
        df_users = load_user_profiles()
    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        return

    if df.empty:
        st.error("⚠️ Dati principali non trovati (df vuoto). Verifica data_loader.py")
        return

    # Dettagli Principali
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📸 Totale Foto", f"{len(df):,}")
    col2.metric("👥 Utenti Unici", f"{df['user_id'].nunique():,}")

    # Calcolo ROI mappate (escludendo Unknown)
    roi_col = 'roi' if 'roi' in df.columns else 'roi_name'
    valid_rois = df[df[roi_col] != 'Unknown'][roi_col].nunique()
    col3.metric("📍 Luoghi Mappati", valid_rois)

    # Copertura Temporale
    if 'year' in df.columns:
        years = pd.to_numeric(df['year'], errors='coerce').dropna().unique()
        # Filtriamo dal 1500 in poi
        years = years[years >= 1500]
        if len(years) > 0:
            col4.metric("📅 Periodo", f"{int(years.min())} – {int(years.max())}")
        else:
            col4.metric("📅 Periodo", "Dati non validi")

    st.markdown("---")

    # Grafici

    # Trend Annuale
    st.subheader("📈 Evoluzione Temporale (2000-2016)")
    if 'year' in df.columns:
        # Pulizia e raggruppamento
        yearly_counts = df['year'].value_counts().reset_index()
        yearly_counts.columns = ['Anno', 'Foto']
        yearly_counts = yearly_counts.sort_values('Anno')
        yearly_counts = yearly_counts[yearly_counts['Anno'].astype(int) > 1999]

        fig_year = px.area(
            yearly_counts,
            x='Anno',
            y='Foto',
            markers=True,
            line_shape='spline',
            title="Crescita del volume dati negli anni",
            color_discrete_sequence=['#FF4B4B']
        )
        st.plotly_chart(fig_year, width='stretch')

    #cTop ROI e Utenti
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("🏆 Top 10 Monumenti")
        # Logica che replica Query 1 di analytics.py
        top_rois = df[df[roi_col] != 'Unknown'][roi_col].value_counts().head(10).reset_index()
        top_rois.columns = ['Monumento', 'Foto']
        fig = px.bar(top_rois, x='Foto', y='Monumento', orientation='h', color='Foto', color_continuous_scale='Viridis')
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("👥 Turisti vs Residenti")
        if not df_users.empty and 'user_type' in df_users.columns:
            user_counts = df_users['user_type'].value_counts().reset_index()
            user_counts.columns = ['Tipo', 'Utenti']
            fig_pie = px.pie(user_counts, values='Utenti', names='Tipo', hole=0.6,
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, width='stretch')