import os
import shutil

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.storagelevel import StorageLevel

from backend.config import ROME_ROIS
from backend.utils import get_spark_session

def run_spatial_enrichment():
    spark = get_spark_session("FlickrFlow_Phase3_Enrichment")

    spark.sparkContext.setLogLevel("ERROR")
    print("\n" + "=" * 60)
    print("--- FASE 3: SPATIAL ENRICHMENT (Broadcast & UDF) ---")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr_cleaned.parquet"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr_enriched.parquet"))

    if not os.path.exists(input_path):
        print("ERRORE: Input Parquet non trovato. Esegui la Fase 2.")
        return

    # 2. Caricamento Dataset (Parquet)
    print(f"Lettura Parquet: {input_path}")
    df = spark.read.parquet(input_path)

    # 3. Ottimizzazione: Broadcast Variable
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
        return "Unknown"  # Punto fuori dalle zone note, le ROME_ROIS che ho definito in config.py

    find_roi_udf = udf(find_roi_logic, StringType())

    # 5. Arricchimento
    print("Mapping GPS -> Zone Semantiche...")
    enriched_df = df.withColumn("roi", find_roi_udf(col("latitude"), col("longitude")))

    # 6. Caching
    # Salvo in memoria perché questo DF verrà usato per K-Means e Traiettorie
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