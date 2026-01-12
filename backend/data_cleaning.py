import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, coalesce, lit

# --- CONFIGURAZIONE ---
ANALYSIS_CONFIG = {
    "APP_NAME": "FlickrFlow_Cleaning",
    "CITY_NAME": "Roma",
    "BOUNDING_BOX": {
        "min_lat": 41.6,
        "max_lat": 42.2,
        "min_lon": 12.2,
        "max_lon": 12.8
    },
    "USE_GEOFENCE": True #Mettendo a false analizziamo tutto il mondo e non Roma
}


def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_cleaning_pipeline():
    init_environment()
    conf = ANALYSIS_CONFIG

    # 1. Spark Session (Case Sensitive Enabled)
    # Impostiamo 'spark.sql.caseSensitive' a True per evitare l'errore "Column already exists"
    # se il JSON contiene chiavi miste (es. dateTaken vs DateTaken)
    spark = SparkSession.builder \
        .appName(conf["APP_NAME"]) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.ansi.enabled", "false") \
        .config("spark.sql.caseSensitive", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- AVVIO DATA CLEANING (Fase 2) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr2x.json")
    output_path = os.path.join(base_dir, "..", "data", "flickr_cleaned.parquet")

    # 2. Ingestion
    print(f"Leggo il dataset da: {input_path}")
    if not os.path.exists(input_path):
        print("ERRORE: File di input non trovato.")
        return

    # Lettura standard JSON Lines
    raw_df = spark.read.json(input_path)
    count_raw = raw_df.count()
    print(f"Totale record grezzi trovati: {count_raw}")

    # 3. Gestione Nomi Colonne (Case Sensitivity Strategy)
    # Cerchiamo se esistono varianti del nome 'dateTaken' nel DataFrame caricato
    available_columns = raw_df.columns
    print(f"Colonne rilevate (primi 10): {available_columns[:10]}")

    # Creiamo un riferimento sicuro alla colonna data.
    # Se c'è 'dateTaken' usiamo quella, altrimenti proviamo 'DateTaken' o 'datetaken' se esistono.
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

    # 5. Flattening & Application
    # Anche per geoData e owner usiamo la stessa logica difensiva se necessario,
    # ma lo schema inspection diceva che erano minuscoli, quindi ci fidiamo.
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