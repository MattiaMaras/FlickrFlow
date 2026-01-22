# 🏛️ FlickrFlow | Rome Big Data Analytics

Progetto di Big Data per l'analisi dei flussi turistici a Roma basato su dataset Flickr (2.2M di foto).

## 🚀 Architettura

Il progetto segue una pipeline ETL moderna basata su **Apache Spark** per il backend e **Streamlit** per il frontend.

### 1. Backend (Spark Engine)
Tutti gli script di elaborazione si trovano nella cartella `backend/`:
- **ETL & Cleaning**: `data_cleaning.py` converte i JSON grezzi in Parquet ottimizzati.
- **Enrichment**: `spatial_enrichment.py` mappa le coordinate GPS su Regioni di Interesse (ROI) come Colosseo, Vaticano, ecc.
- **Profiling**: `feature_engineering.py` classifica gli utenti in Turisti, Residenti o Visitatori Ricorrenti.
- **AI & ML**:
  - `clustering.py`: K-Means per scoprire zone turistiche non ufficiali.
  - `trajectory_mining.py`: FPGrowth per analizzare i percorsi sequenziali (es. Colosseo -> Trevi).
  - `tag_analysis.py`: NLP per estrarre il contesto semantico dai tag delle foto.

### 2. Frontend (Streamlit App)
L'interfaccia utente è gestita da `frontend/app.py` che orchestra diverse pagine:
- **Overview**: KPI generali e trend temporali.
- **Query Explorer**: Analisi flussi, heatmap orarie e gemme nascoste.
- **Geo Spatial 3D**: Mappe esagonali interattive (PyDeck).
- **Cluster Analysis**: Visualizzazione dei cluster AI vs ROI reali.
- **AI Analyst**: Generazione report strategici automatici (LLM).

## 🛠️ Installazione

1. **Requisiti**:
   - Python 3.10+
   - Java 17 (per Spark)

2. **Setup Ambiente**:
    Controllare le librerie richieste dentro il file `requirements.txt` ed eventualmente scaricare quelle mancanti dentro il proprio environment per procedere.

## ▶️ Esecuzione

### 1. Pipeline Dati (Backend)
Per rigenerare tutti i dati (necessario alla prima esecuzione):
```bash
python run_pipeline.py
```
*Questo script eseguirà in sequenza pulizia, arricchimento, ML e preparazione viste.*

### 2. Dashboard (Frontend)
Per avviare l'interfaccia web:
```bash
streamlit run frontend/app.py
```

## 📊 Struttura Dati
I dati processati vengono salvati in `data/` in formato **Parquet**:
- `flickr_enriched.parquet`: Dataset principale con ROI.
- `flickr_user_profiles.parquet`: Classificazione utenti.
- `view_*.parquet`: Viste pre-calcolate per la dashboard (performance ottimizzate).


## Troubleshooting
- **Dataset mancante**: Scaricare `flickr2x.json` da `https://drive.google.com/drive/folders/1g1a-l_K3u_HLY5fXXfRJc0LIln34fZ3a?usp=sharing`

---
**Autore:** Mattia Marasco
**Corso:** Modelli e tecniche per BigData
