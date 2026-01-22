import os
import shutil

from pyspark.sql.functions import col, to_timestamp, coalesce, lit
from backend.config import ANALYSIS_CONFIG
from backend.utils import get_spark_session

def run_cleaning_pipeline():
    conf = ANALYSIS_CONFIG

    spark = get_spark_session(conf["APP_NAME"])

    spark.sparkContext.setLogLevel("ERROR")
    print("\n" + "=" * 60)
    print("--- FASE 2: DATA CLEANING ---")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr2x.json"))
    output_path = os.path.abspath(os.path.join(base_dir, "..","..", "data", "flickr_cleaned.parquet"))

    print(f"Leggo il dataset da: {input_path}")
    if not os.path.exists(input_path):
        print("ERRORE: File di input non trovato.")
        return

    # Lettura file JSON
    raw_df = spark.read.json(input_path)
    count_raw = raw_df.count()
    print(f"Totale record grezzi trovati: {count_raw}")

    # 3. Nomi Colonne (Case Sensitivity)
    available_columns = raw_df.columns
    print(f"Colonne rilevate (primi 10): {available_columns[:10]}")

    # Riferimento sicuro alla colonna data.
    # Se c'è 'dateTaken' uso quella, altrimenti 'DateTaken' o 'datetaken' se esistono.
    def get_col_safe(name_variations):
        for name in name_variations:
            if name in available_columns:
                return col(name)
        return lit(None)  # Se nessuna esiste, restituisce null

    target_date_col = get_col_safe(["dateTaken", "DateTaken", "datetaken"])

    # 4. Parsing Date (Formati Multipli)
    fmt_iso = "yyyy-MM-dd HH:mm:ss"
    fmt_verbose = "MMM d, yyyy h:mm:ss a"

    timestamp_col = coalesce(
        to_timestamp(target_date_col, fmt_iso),
        to_timestamp(target_date_col, fmt_verbose)
    )

    # 5. Flattening
    # Anche per geoData e owner uso la stessa logica difensiva se necessario,
    # ma dallo schema inspection erano minuscoli, quindi mi fido.
    flat_df = raw_df.select(
        col("id").alias("photo_id"),
        col("owner.id").alias("user_id"),
        col("title"),
        timestamp_col.alias("timestamp"),
        col("geoData.latitude").cast("double").alias("latitude"),
        col("geoData.longitude").cast("double").alias("longitude"),
        col("tags.value").alias("tags")
    )

    # 6. Filtering
    print("Applico i filtri di pulizia...")
    cleaned_df = flat_df.filter(
        col("user_id").isNotNull() &
        col("timestamp").isNotNull() &
        col("latitude").isNotNull() &
        col("longitude").isNotNull() &
        (col("latitude") != 0.0) & (col("longitude") != 0.0)
    )

    if conf["USE_GEOFENCE"]:
        bbox = conf["BOUNDING_BOX"]
        print(f"Filtro geografico attivo: Roma ({bbox['min_lat']} - {bbox['max_lat']})")
        cleaned_df = cleaned_df.filter(
            (col("latitude").between(bbox["min_lat"], bbox["max_lat"])) &
            (col("longitude").between(bbox["min_lon"], bbox["max_lon"]))
        )

    count_clean = cleaned_df.count()
    dropped = count_raw - count_clean
    drop_pct = (dropped / count_raw) * 100 if count_raw > 0 else 0

    print(f"Record validi finali: {count_clean}")
    print(f"Record scartati: {dropped} ({drop_pct:.2f}%)")

    # 7. Salvataggio Parquet
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if count_clean > 0:
        print(f"Salvo il risultato in Parquet: {output_path}")
        cleaned_df.coalesce(1).write.parquet(output_path)
    else:
        print("ATTENZIONE: Nessun record valido rimasto dopo i filtri.")

    spark.stop()


if __name__ == "__main__":
    run_cleaning_pipeline()