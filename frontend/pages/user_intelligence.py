import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def load_profiles():
    path = os.path.join(parent_dir, "..", "data", "flickr_user_profiles.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def render():
    st.title("👥 User Intelligence")
    st.markdown("Profilazione avanzata degli utenti: **Residenti, Turisti e Visitatori Ricorrenti**.")

    df = load_profiles()

    if df.empty:
        st.error("⚠️ Profili utente mancanti. Esegui Fase 5 del backend.")
        return

    # Overview Metriche
    total_users = len(df)
    residents = len(df[df['user_type'] == 'Resident'])
    tourists = len(df[df['user_type'] == 'Tourist'])

    k1, k2, k3 = st.columns(3)
    k1.metric("Utenti Totali", f"{total_users:,}")
    k2.metric("Turisti", f"{tourists:,}", delta=f"{tourists / total_users:.1%}")
    k3.metric("Residenti", f"{residents:,}", delta=f"{residents / total_users:.1%}")

    st.divider()

    # Grafici
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tipologia Utenti")
        fig_pie = px.pie(
            df,
            names='user_type',
            title="Segmentazione Utenza",
            color='user_type',
            color_discrete_map={'Tourist': '#FF4B4B', 'Resident': '#00CC96', 'Recurring Visitor': '#FFAA00'}
        )
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.subheader("Durata Permanenza")
        clean_days = df[df['days_active'] < 365]
        fig_hist = px.histogram(
            clean_days,
            x="days_active",
            nbins=30,
            title="Distribuzione Giorni di Attività",
            labels={'days_active': 'Giorni Attivi'},
            color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig_hist, width='stretch')

    # Super Users / Anomalie
    st.subheader("🚨 Top Contributors (Potential Bots/Archivists)")
    st.caption("Utenti con volume di foto anomalo o attività estremamente prolungata.")

    top_users = df.sort_values("total_photos", ascending=False).head(10)

    st.dataframe(
        top_users[['user_id', 'user_type', 'total_photos', 'unique_rois', 'days_active']],
        column_config={
            "user_id": "User ID",
            "user_type": "Classe",
            "total_photos": st.column_config.ProgressColumn("Volume Foto", format="%d",
                                                            max_value=int(df['total_photos'].max())),
            "unique_rois": "Zone Visitate",
            "days_active": "Giorni Attivi"
        },
        width='stretch',
        hide_index=True
    )

    # Correlazione
    st.markdown("---")
    st.subheader("🔍 Pattern Esplorativo")
    fig_scat = px.scatter(
        df.sample(min(5000, len(df))),  # Sample per performance
        x="days_active",
        y="unique_rois",
        color="user_type",
        size="total_photos",
        log_y=True,
        title="Mobilità vs Permanenza (Bubble Size = Num. Foto)",
        labels={'unique_rois': 'Zone Uniche Visitate', 'days_active': 'Giorni di Permanenza'}
    )
    st.plotly_chart(fig_scat, width='stretch')