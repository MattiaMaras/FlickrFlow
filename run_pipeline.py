import os
import sys
import subprocess
import time

def run_step(script_path, description):
    """Esegue uno script Python e gestisce gli errori."""

    full_path = os.path.join("backend", script_path)
    if not os.path.exists(full_path):
        print(f"Errore: Script {full_path} non trovato.")
        sys.exit(1)

    start_time = time.time()
    try:
        # Usa sys.executable per garantire che venga usato lo stesso interprete Python
        subprocess.run([sys.executable, full_path], check=True)
        duration = time.time() - start_time
        print(f"✅ COMPLETATO in {duration:.2f} secondi.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERRORE durante l'esecuzione di {script_path}.")
        print(f"Codice uscita: {e.returncode}")
        sys.exit(1)

def main():
    # Controllo versione Python per evitare errori con PySpark < 3.5 su Python 3.12+
    if sys.version_info >= (3, 12):
        print("❌ Ambiente Python non supportato per PySpark. Usa Python <= 3.11 (es. bigdata_env).")
        sys.exit(1)

    print("🚀 FLICKR FLOW - BIG DATA PIPELINE ORCHESTRATOR")
    print("Avvio sequenza di elaborazione completa...\n")

    # 1. Pulizia Dati (Phase 2)
    run_step("phase2/data_cleaning.py", "Data Cleaning (JSON -> Parquet, Filtri Roma)")

    # 2. Arricchimento Spaziale (Phase 3)
    run_step("phase3/spatial_enrichment.py", "Spatial Enrichment (GPS -> ROI)")

    # 3. Feature Engineering (Phase 5)
    run_step("phase5/feature_engineering.py", "Feature Engineering (User Type, Time)")

    # 4. Clustering (Phase 5)
    run_step("phase5/clustering.py", "Unsupervised Clustering (K-Means)")

    # 5. Trajectory Mining (Phase 6)
    run_step("phase6/trajectory_mining.py", "Trajectory Mining (Sequenze di visita)")

    # 6. NLP Tag Analysis (Phase 7)
    run_step("phase7/tag_analysis.py", "NLP Tag Analysis (Estrazione Keyword)")

    # 7. Preparazione Viste Dashboard (Phase 7)
    run_step("phase7/prepare_queries.py", "Dashboard Views Optimization")

    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETATA CON SUCCESSO!")
    print("Ora puoi avviare la dashboard con: streamlit run frontend/app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
