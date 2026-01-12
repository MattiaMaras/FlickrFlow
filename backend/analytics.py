import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, count


def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_analytics():
    init_environment()

    # Inizializzo la sessione Spark
    spark = SparkSession.builder \
        .appName("FlickrFlow_Phase4_Analytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- FASE 4: ANALYTICS ---")

    #percorso del dataset arricchito creato nella Fase 3
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")

    if not os.path.exists(input_path):
        print("Errore: Dataset arricchito non trovato.") #da eseguire fase 3
        return

    # Carico il dataset in formato Parquet (tipi di dato preservati)
    df = spark.read.parquet(input_path)

    # ---Fase 3Bis, ottimizzazione---
    # Eseguo il caching del DataFrame in memoria.
    # Poiche' eseguiro' 3 query diverse sullo stesso dataset,
    # voglio evitare di leggere il file da disco 3 volte.
    df.cache()

    # Eseguo una count() per forzare la materializzazione della cache ora
    print(f"Dataset caricato e cachato in RAM. Totale righe: {df.count()}")

    # --- QUERY 1: Classifica dei Monumenti (Top RoI) ---
    print("\n[QUERY 1] Top 10 Regioni di Interesse (RoI)")
    # Raggruppo per RoI, conto le occorrenze e ordino in modo decrescente
    # Escludo 'Unknown' per focalizzarmi sui punti di interesse identificati
    top_rois = df.filter(col("roi") != "Unknown") \
        .groupBy("roi") \
        .count() \
        .orderBy(col("count").desc())

    top_rois.show(truncate=False)

    # --- QUERY 2: Analisi Temporale (Trend Annuale) ---
    print("\n[QUERY 2] Distribuzione temporale delle foto")
    # Filtro gli anni > 2000 per escludere i dati con data di default (anno 0001)
    # mantenuti nella fase di cleaning per non perdere i dati spaziali
    yearly_trend = df.filter(year(col("timestamp")) > 2000) \
        .groupBy(year(col("timestamp")).alias("anno")) \
        .count() \
        .orderBy("anno")

    yearly_trend.show(25)

    # --- QUERY 3: Analisi Utenti ---
    print("\n[QUERY 3] Top 5 Utenti per attivita'")
    # Identifico gli utenti che hanno pubblicato piu' foto
    top_users = df.groupBy("user_id") \
        .count() \
        .orderBy(col("count").desc())

    top_users.show(5, truncate=False)

    # Rilascio la memoria alla fine dello script
    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_analytics()