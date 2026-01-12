import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, year
from pyspark.ml.fpm import FPGrowth


def init_environment():
    # Setup Java 17
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_trajectory_mining():
    init_environment()

    # Inizializzo Spark Session
    spark = SparkSession.builder \
        .appName("FlickrFlow_Phase6_FPGrowth") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.crossJoin.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- FASE 6: TRAJECTORY MINING---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")

    if not os.path.exists(input_path):
        print("Errore: Dataset non trovato. (Eseguire spatial_enrichment.py)")
        return

    df = spark.read.parquet(input_path)

    # 1. Preparazione Dati
    print("Preparazione delle traiettorie utente...")

    filtered_df = df.filter(
        (col("roi") != "Unknown") &
        (year(col("timestamp")) > 2000)
    )

    # FIX: Uso collect_set invece di collect_list
    # FPGrowth richiede che gli elementi in una transazione siano unici.
    # Se un utente ha fatto 10 foto al Vaticano, ci interessa il concetto "Ha visitato il Vaticano",
    # non la ripetizione dell'evento.
    transactions_df = filtered_df.groupBy("user_id") \
        .agg(collect_set("roi").alias("items"))

    # Caching per performance
    transactions_df.cache()

    count_traj = transactions_df.count()
    print(f"Numero di utenti/traiettorie uniche analizzate: {count_traj}")

    if count_traj == 0:
        print("Nessuna traiettoria valida trovata.")
        return

    # 2. Configurazione FPGrowth
    # Parametri ottimizzati per dataset turistici:
    # - minSupport=0.01: Un pattern deve apparire in almeno l'1% delle traiettorie per essere interessante.
    # - minConfidence=0.05: La regola deve essere vera almeno il 5% delle volte.
    print(f"Addestramento modello FPGrowth su {count_traj} transazioni...")
    fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.05)

    model = fp_growth.fit(transactions_df)

    # 3. Risultati: Frequent Itemsets
    # Quali sono le combinazioni di luoghi più visitate?
    print("\n[FREQUENT ITEMSETS] Combinazioni frequenti:")
    model.freqItemsets.sort(col("freq").desc()).show(10, truncate=False)

    # 4. Risultati: Association Rules
    # Quali regole possiamo dedurre? (Se visiti A -> visiti B)
    print("\n[ASSOCIATION RULES] Regole di movimento (Top Generali):")
    model.associationRules.filter(col("lift") > 1.0) \
        .sort(col("confidence").desc()) \
        .show(5, truncate=False)

    print("\n[ASSOCIATION RULES] Regole Alternative (Escluso Vaticano):")
    # Filtriamo via le regole che hanno "Vaticano" come conseguente per vedere i flussi secondari
    from pyspark.sql.functions import array_contains
    model.associationRules.filter(
        (col("lift") > 1.0) &
        (~array_contains(col("consequent"), "Vaticano"))
    ) \
        .sort(col("confidence").desc()) \
        .select("antecedent", "consequent", "confidence", "lift") \
        .show(10, truncate=False)

    transactions_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_trajectory_mining()