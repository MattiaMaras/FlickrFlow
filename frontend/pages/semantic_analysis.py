import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def load_rules():
    path = os.path.join(parent_dir, "..", "data", "flickr_rules.parquet")
    if os.path.exists(path): return pd.read_parquet(path)
    return pd.DataFrame()


def load_tags():
    path = os.path.join(parent_dir, "..", "data", "flickr_roi_tags.parquet")
    if os.path.exists(path): return pd.read_parquet(path)
    return pd.DataFrame()


def render():
    st.title("🔗 Semantic Analysis")
    st.markdown("Analisi avanzata: **Dove vanno** (Traiettorie) e **Di cosa parlano** (NLP) gli utenti.")

    # Tab Strutturati
    tab_flow, tab_nlp = st.tabs(["🔄 Traiettorie (FP-Growth)", "☁️ Contenuto (NLP & Tag)"])

    # Sankey Diagram
    with tab_flow:
        df_rules = load_rules()

        if df_rules.empty:
            st.error("⚠️ Nessuna regola trovata. Esegui Fase 6 Backend")
        else:
            st.subheader("🌊 Flussi Turistici Sequenziali")

            # Layout Filtri + Legenda
            col_filter, col_legend = st.columns([1, 2])

            with col_filter:
                st.markdown("**⚙️ Configurazione**")
                min_conf = st.slider("Confidenza Minima", 0.05, 1.0, 0.20, 0.05)
                limit = st.number_input("Max Flussi visualizzati", 5, 50, 20)

            with col_legend:
                with st.expander("ℹ️ Guida alla lettura", expanded=True):
                    st.markdown("""
                    ### 🧭 Flussi turistici sequenziali

                    Le frecce rappresentano **percorsi A → B** estratti dalle traiettorie utente (FP-Growth).

                    ---
                    **➡️ Frecce**
                    * Flussi duplicati **aggregati**
                    * **Spessore** = numero di utenti

                    ---
                    **📊 Confidenza – P(B | A)**
                    * 🔼 Alta → percorsi **frequenti**
                    * 🔽 Bassa → percorsi **rari** (più rumore)

                    ---
                    **🔗 Lift**
                    * **> 1** → legame significativo
                    * **≈ 1** → casuale

                    💡 *Confidenza = stabilità del flusso*  
                    💡 *Lift = forza della relazione*
                    """)

            filtered_df = df_rules[df_rules['confidence'] >= min_conf].sort_values("lift", ascending=False)

            if not filtered_df.empty:
                filtered_df['src_name'] = filtered_df['antecedent'].apply(lambda x: x[0])
                filtered_df['dst_name'] = filtered_df['consequent'].apply(lambda x: x[0])

                # Unione flussi duplicati
                sankey_data = filtered_df.groupby(['src_name', 'dst_name'], as_index=False).agg({
                    'confidence': 'max',  # Probabilità massima
                    'lift': 'mean',  # Media del lift
                    'support': 'sum'  # Somma del volume (supporto)
                }).sort_values("confidence", ascending=False).head(limit)


                all_nodes = list(set(sankey_data['src_name']).union(set(sankey_data['dst_name'])))
                node_map = {name: i for i, name in enumerate(all_nodes)}

                node_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
                nodes_color_list = [node_colors[i % len(node_colors)] for i in range(len(all_nodes))]

                fig_sankey = go.Figure(data=[go.Sankey(
                    textfont=dict(color="black", size=12, family="Arial Black"),

                    node=dict(
                        pad=20, thickness=25,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color=nodes_color_list
                    ),
                    link=dict(
                        source=[node_map[x] for x in sankey_data['src_name']],
                        target=[node_map[x] for x in sankey_data['dst_name']],
                        # Moltiplichiamo per dare spessore visibile
                        value=(sankey_data['confidence'] * 100).astype(int),
                        customdata=sankey_data['lift'],
                        hovertemplate='Da: %{source.label}<br>A: %{target.label}<br>Probabilità: %{value}%<br>Lift: %{customdata:.2f}<extra></extra>'
                    )
                )])

                fig_sankey.update_layout(title_text=f"Top {len(sankey_data)} Flussi Consolidati", font_size=14,
                                         height=550)
                st.plotly_chart(fig_sankey, width='stretch')

                # Tabella sotto
                with st.expander("📊 Dati Dettagliati (Aggregati)"):
                    st.dataframe(
                        sankey_data[['src_name', 'dst_name', 'confidence', 'lift']],
                        column_config={
                            "src_name": "Partenza",
                            "dst_name": "Arrivo",
                            "confidence": st.column_config.ProgressColumn("Probabilità", format="%.2f", max_value=1),
                            "lift": st.column_config.NumberColumn("Forza (Lift)", format="%.2f")
                        },
                        width='stretch'
                    )
            else:
                st.warning("Nessuna regola trovata con questi filtri. Prova ad abbassare la Confidenza.")


    # WordClouds e Treemap
    with tab_nlp:
        df_tags = load_tags()

        if df_tags.empty:
            st.warning("⚠️ Dati Tag mancanti. Esegui Fase 7 del backend")
        else:
            st.subheader("🗣️ Analisi Lessicale")
            st.markdown("Analisi dei **Tag** associati alle foto. Capiamo come gli utenti percepiscono il luogo.")

            # Selezione ROI
            rois = sorted(df_tags['roi'].unique())
            col_sel, col_space = st.columns([1, 3])
            with col_sel:
                sel_roi = st.selectbox("Seleziona Zona da Analizzare:", rois)

            # Filtro dati ROI
            subset = df_tags[df_tags['roi'] == sel_roi].sort_values("count", ascending=False).head(50)

            if not subset.empty:
                # WordCloud
                st.markdown("### 1. Nuvola di Parole (WordCloud)")
                text_data = dict(zip(subset['word'], subset['count']))

                wc = WordCloud(
                    width=900, height=400,
                    background_color='#0E1117',
                    colormap='Wistia',
                    max_words=60,
                    contour_width=0,
                ).generate_from_frequencies(text_data)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                fig.patch.set_facecolor('#0E1117')
                st.pyplot(fig)

                st.divider()

                # Treemap
                st.markdown("### 2. Gerarchia dei Temi (Treemap)")

                fig_tree = px.treemap(
                    subset,
                    path=['word'],
                    values='count',
                    color='count',
                    color_continuous_scale='Greens',
                    title=f"Paesaggio Semantico: {sel_roi}",
                    hover_data=['count']
                )
                fig_tree.update_layout(height=500)
                st.plotly_chart(fig_tree, width='stretch')

            else:
                st.info("Nessun tag disponibile per questa zona.")