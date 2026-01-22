import streamlit as st
import pandas as pd
import sys
import os
import requests


from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
from langchain_ollama import ChatOllama


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Configurazione AI
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434", timeout=0.5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

#AI Locale (Mistral) o Mock
@st.cache_resource
def get_llm_engine():
    if check_ollama_status():
        return ChatOllama(model="mistral", temperature=0.5), True
    else:
        simulated_response = (
            "📄 **REPORT STRATEGICO (MODALITÀ BACKUP)**\n\n"
            "### 🏆 Analisi dei Poli Turistici\n"
            "L'analisi K-Means (Silhouette: 0.52) su 2.2 milioni di foto rivela una discrepanza. "
            "Mentre il **Vaticano** è la ROI definita più popolare, i cluster non supervisionati **1 e 8** "
            "(Centro Storico e Colosseo allargato) aggregano insieme oltre **1.32 milioni di punti**, "
            "dimostrando che il turismo reale è molto più diffuso dei confini ufficiali.\n\n"
            "### ⏳ Pattern Comportamentali\n"
            "I dati storici (**2000-2016**) mostrano una crescita organica. La presenza di 'Super-Users' "
            "indica una forte componente di archiviazione automatica o bot che va filtrata per analisi comportamentali pure.\n\n"
            "### 🚀 Proposta Operativa\n"
            "Sfruttare la regola associativa forte (Lift > 3.8) tra **Trastevere/Vaticano** e **Piazza di Spagna** "
            "per creare percorsi pedonali guidati che decongestionino il trasporto pubblico su questa tratta."
        )
        return FakeListLLM(responses=[simulated_response]), False


#Caricamento Dati

def load_summary_data():
    data_path = os.path.join(parent_dir, "..", "data")
    df_c = pd.DataFrame()
    df_t = pd.DataFrame()

    clusters_path = os.path.join(data_path, "flickr_clusters.parquet")
    if os.path.exists(clusters_path):
        df_c = pd.read_parquet(clusters_path)

    time_path = os.path.join(data_path, "flickr_time_features.parquet")
    if os.path.exists(time_path):
        df_t = pd.read_parquet(time_path)

    return df_c, df_t


def prepare_stats_text(df_c, df_t):
    text = "DATI GENERALI:\n"
    total_rows = len(df_c) if not df_c.empty else 0
    text += f"- Totale Foto Dataset: {total_rows:,} (Periodo 2000-2016)\n"

    # Analisi Cluster (K-Means)
    if not df_c.empty:
        if 'prediction' in df_c.columns: df_c.rename(columns={'prediction': 'cluster'}, inplace=True)

        # Top 3 Cluster per volume
        top_clusters = df_c['cluster'].value_counts().head(3)
        text += "\nTOP 3 CLUSTER (Raggruppamento AI Non Supervisionato):\n"
        for cid, count in top_clusters.items():
            share = (count / total_rows) * 100
            zone_hint = ""
            if cid == 1:
                zone_hint = "(Probabile Centro Storico/Pantheon)"
            elif cid == 8:
                zone_hint = "(Probabile Area Archeologica/Colosseo)"
            elif cid == 6:
                zone_hint = "(Probabile Vaticano)"
            text += f"- Cluster {cid} {zone_hint}: {count:,} foto ({share:.1f}%)\n"

        # ROI definite (Spatial Enrichment)
        if 'roi' in df_c.columns:
            # Escludiamo Unknown per vedere le ROI ufficiali
            known_rois = df_c[df_c['roi'] != 'Unknown']['roi'].value_counts().head(3)
            mapped_count = df_c[df_c['roi'] != 'Unknown'].shape[0]

            text += f"\nROI DEFINITE (Spatial Enrichment):\n"
            text += f"- Totale foto mappate in ROI specifiche: {mapped_count:,} (su {total_rows:,} totali)\n"
            for roi, count in known_rois.items():
                text += f"- {roi}: {count:,} foto\n"

    # Analisi Temporale e Utenti
    if not df_t.empty:
        text += "\nDATI TEMPORALI E UTENTI:\n"
        if 'year' in df_t.columns:
            peak_year = df_t['year'].mode()[0]
            text += f"- Anno di picco attività: {peak_year}\n"

        if 'user_id' in df_t.columns:
            top_user_count = df_t['user_id'].value_counts().iloc[0]
            text += f"- Top User (attività anomala): ha scattato {top_user_count:,} foto da solo.\n"

    return text


def render():
    st.title("🤖 AI Strategic Advisor")
    st.markdown("Generazione di report strategici basati sui dati reali di FlickrFlow (2.2M foto).")

    st.divider()

    df_c, df_t = load_summary_data()
    if df_c.empty:
        st.error("Dati mancanti. Esegui il backend.")
        return

    stats_summary = prepare_stats_text(df_c, df_t)

    col1, col2 = st.columns([1, 3])

    with col1:
        ollama_active = check_ollama_status()
        status_color = "🟢" if ollama_active else "🟠"
        status_text = "Online" if ollama_active else "Offline (Mock)"

        st.image("assets/ollama.png",width=300)
        st.caption(f"**Status:** {status_color} {status_text}")

        generate_btn = st.button("🧠 Genera Report", type="primary")

    with col2:
        if generate_btn:
            with st.spinner("Analisi dei cluster e generazione insight..."):

                llm, is_real = get_llm_engine()

                # Prompt con dati ricavati dal backend
                template = """
                Sei un Senior Data Analyst per il Turismo di Roma.
                Hai analizzato un dataset di 2.2 Milioni di foto Flickr (2000-2016).

                Ecco i dati estratti dal sistema:
                {stats_data}

                TASK:
                Scrivi un report strategico breve (in italiano, max 12 righe totali) focalizzandoti su:
                1. **Confronto Cluster vs ROI**: Nota come i Cluster 1 e 8 (AI) siano giganti (circa 660k foto ciascuno) rispetto alle singole ROI definite (es. Colosseo). Cosa significa? (Turismo diffuso vs Punti precisi).
                2. **Il Paradosso Vaticano**: Il Vaticano (Cluster 6) ha circa 433k foto reali, confermandosi un polo massivo ma distinto dal centro storico.
                3. **Anomalie**: Menziona il "Top User" con 115k foto come potenziale bot/archivio.

                Usa Markdown. Sii analitico e professionale.
                """

                prompt = PromptTemplate(
                    input_variables=["stats_data"],
                    template=template
                )

                chain = prompt | llm

                try:
                    response = chain.invoke({"stats_data": stats_summary})
                    final_text = response.content if hasattr(response, 'content') else str(response)

                    st.markdown(final_text)

                    if is_real:
                        st.success("Report generato da Mistral (Locale) su dati reali.")
                    else:
                        st.warning("Report generato dal sistema di Backup.")

                except Exception as e:
                    st.error(f"Errore: {e}")

        else:
            st.info("Premi il pulsante per avviare l'analisi AI.")
            with st.expander("🔍 Vedi i dati grezzi inviati all'AI"):
                st.text(stats_summary)