import streamlit as st
import sys
import os
import plotly.express as px


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import data_loader as dl
except ImportError as e:
    st.error(f"Errore di importazione: {e}")
    st.stop()


def render():
    st.title("🔍 Query Explorer")
    st.markdown("""
    Analisi puntuali su **orari ottimali** e **gemme nascoste** (Local vs Tourist).
    """)

    tab_time, tab_gems = st.tabs(["⏰ Orari Ideali", "💎 Hidden Gems"])

    # Orari
    with tab_time:
        st.subheader("Quando visitare per evitare la folla?")
        df_hm = dl.load_heatmap()

        if df_hm.empty:
            st.warning("⚠️ Dati orari mancanti (Eseguire Fase 7 Backend).")
        else:
            rois = sorted(df_hm['roi'].unique())

            col_sel, col_kpi = st.columns([1, 3])
            with col_sel:
                sel_roi = st.selectbox("Seleziona Luogo:", rois, key="hm_roi")

            # Filtro
            subset = df_hm[df_hm['roi'] == sel_roi]

            peak_hour = subset.loc[subset['count'].idxmax()]['hour']
            with col_kpi:
                st.info(f"💡 Il picco di affluenza per **{sel_roi}** è alle ore **{peak_hour}:00**.")

            # Grafico Linea Curva
            fig_line = px.line(
                subset,
                x='hour',
                y='count',
                color='user_type',
                markers=True,
                title=f"Affluenza Oraria: {sel_roi}",
                labels={'hour': 'Ora (0-23)', 'count': 'Volume Foto', 'user_type': 'Utente'},
                line_shape='spline'  # Linea curva morbida
            )
            # Colori personalizzati per contrasto dark mode
            fig_line.update_traces(line=dict(width=3))
            fig_line.update_layout(xaxis=dict(tickmode='linear', dtick=2))

            st.plotly_chart(fig_line, width='stretch')

    # Gemme Nascoste
    with tab_gems:
        st.subheader("💎 I luoghi preferiti dai Romani")
        st.caption("Classifica basata sull' **indice di località**: (1.0 = Solo Residenti), (0.0 = Solo Turisti)")

        df_gems = dl.load_gems()

        if df_gems.empty:
            st.warning("⚠️ Dati Gems mancanti (Esegui Fase 7 Backend).")
        else:
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig_scatter = px.scatter(
                    df_gems,
                    x="Tourist",
                    y="Resident",
                    size="total",
                    color="localness_index",
                    hover_name="roi",
                    color_continuous_scale="Viridis",
                    title="Mappa Localness",
                    labels={'localness_index': 'Score Locale'}
                )
                fig_scatter.update_layout(height=450)
                st.plotly_chart(fig_scatter, width='stretch')

            with col_table:
                top_gems = df_gems.sort_values("localness_index", ascending=False).head(10)
                st.write("**Top 10 'Local' Spots**")
                st.dataframe(
                    top_gems[['roi', 'localness_index']],
                    column_config={
                        "localness_index": st.column_config.ProgressColumn(
                            "Score", format="%.2f", min_value=0, max_value=1,
                        ),
                        "roi": "Luogo"
                    },
                    hide_index=True,
                    width='stretch',
                    height=400
                )