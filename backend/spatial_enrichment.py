import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.storagelevel import StorageLevel

# --- CONFIGURAZIONE ROI (Regions of Interest) ---
# Mappiamo coordinate GPS grezze su etichette semantiche.
# Usiamo un dizionario Python che verrà trasmesso via Broadcast.
ROME_ROIS = {
    "Colosseo": {"lat_min": 41.889, "lat_max": 41.891, "lon_min": 12.491, "lon_max": 12.494},
    "Vaticano": {"lat_min": 41.900, "lat_max": 41.906, "lon_min": 12.450, "lon_max": 12.460},
    "Pantheon": {"lat_min": 41.897, "lat_max": 41.899, "lon_min": 12.475, "lon_max": 12.478},
    "FontanaTrevi": {"lat_min": 41.900, "lat_max": 41.902, "lon_min": 12.482, "lon_max": 12.484},
    "Trastevere": {"lat_min": 41.886, "lat_max": 41.892, "lon_min": 12.465, "lon_max": 12.472},
    "StazioneTermini": {"lat_min": 41.900, "lat_max": 41.903, "lon_min": 12.498, "lon_max": 12.504},
    "PiazzaNavona": {"lat_min": 41.898, "lat_max": 41.900, "lon_min": 12.472, "lon_max": 12.474},
    "PiazzaSpagna": {"lat_min": 41.905, "lat_max": 41.907, "lon_min": 12.481, "lon_max": 12.484},
    "VillaBorghese": {"lat_min": 41.910, "lat_max": 41.918, "lon_min": 12.475, "lon_max": 12.495}
}


def init_environment():
    # Setup Java 17
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_spatial_enrichment():
    init_environment()

    # 1. Spark Session
    spark = SparkSession.builder \
        .appName("FlickrFlow_Phase3_Enrichment") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n" + "=" * 60)
    print("FASE 3: SPATIAL ENRICHMENT (Broadcast & UDF)")
    print("=" * 60)

    # Path Management
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_cleaned.parquet")
    output_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")

    if not os.path.exists(input_path):
        print("ERRORE CRITICO: Input Parquet non trovato. Esegui la Fase 2.")
        return

    # 2. Caricamento Dataset Ottimizzato (Parquet)
    print(f"Lettura Parquet: {input_path}")
    df = spark.read.parquet(input_path)

    # 3. Ottimizzazione: Broadcast Variable
    # Inviamo la mappa delle ROI a tutti i nodi worker per evitare shuffle
    print("Broadcasting RoI Map...")
    rois_broadcast = spark.sparkContext.broadcast(ROME_ROIS)

    # 4. Definizione Logica Spaziale (UDF)
    def find_roi_logic(lat, lon):
        # Accesso ai dati in broadcast
        rois = rois_broadcast.value
        for roi_name, coords in rois.items():
            if (coords["lat_min"] <= lat <= coords["lat_max"]) and \
                    (coords["lon_min"] <= lon <= coords["lon_max"]):
                return roi_name
        return "Unknown"  # Punto fuori dalle zone note, le ROME_ROIS che ho definito sopra

    find_roi_udf = udf(find_roi_logic, StringType())

    # 5. Arricchimento
    print("Mapping GPS -> Zone Semantiche...")
    enriched_df = df.withColumn("roi", find_roi_udf(col("latitude"), col("longitude")))

    # 6. Optimization Checkpoint: Caching
    # Persistiamo in memoria perché questo DF verrà usato per K-Means e Traiettorie
    enriched_df.persist(StorageLevel.MEMORY_AND_DISK)

    # Materializzazione
    total = enriched_df.count()
    mapped = enriched_df.filter(col("roi") != "Unknown").count()

    print(f"Processo Completato.")
    print(f"   --> Totale Foto: {total}")
    print(f"   --> Foto 'catturate' nelle RoI: {mapped}")

    if mapped > 0:
        print("\nEsempio dati arricchiti:")
        enriched_df.filter(col("roi") != "Unknown") \
            .select("photo_id", "roi", "timestamp") \
            .show(5, truncate=False)

    # 7. Salvataggio Finale
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"Scrittura Parquet Arricchito: {output_path}")
    enriched_df.write.parquet(output_path)

    print("=" * 60 + "\n")
    spark.stop()


if __name__ == "__main__":
    run_spatial_enrichment()