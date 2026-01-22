import os
import shutil

from pyspark.sql.functions import col, collect_set, year, array_contains
from pyspark.ml.fpm import FPGrowth

from backend.utils import get_spark_session


def run_trajectory_mining():

    spark = get_spark_session("FlickrFlow_Phase6_FPGrowth")
    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "=" * 60)
    print("--- FASE 6: TRAJECTORY MINING ---")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr_enriched.parquet"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr_rules.parquet"))

    if not os.path.exists(input_path):
        print("Errore: Dataset non trovato. Esegui la fase 3")
        return

    df = spark.read.parquet(input_path)

    print("Preparazione delle traiettorie utente...")
    filtered_df = df.filter(
        (col("roi") != "Unknown") &
        (year(col("timestamp")) > 1999) #Nel data cleaning ho lasciato anni come 0001 per non perdere informazioni utili in altri campi
    )

    transactions_df = filtered_df.groupBy("user_id") \
        .agg(collect_set("roi").alias("items"))

    transactions_df.cache()
    count_traj = transactions_df.count()
    print(f"Numero di utenti/traiettorie uniche analizzate: {count_traj}")

    if count_traj == 0:
        print("Nessuna traiettoria valida trovata.")
        return

    # Configurazione FP-Growth
    print(f"Addestramento modello FPGrowth su {count_traj} transazioni...")
    fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.05)
    model = fp_growth.fit(transactions_df)

    #Frequent Itemsets
    print("\n[FREQUENT ITEMSETS] Top 10 combinazioni frequenti:")
    model.freqItemsets.sort(col("freq").desc()).show(10, truncate=False)

    #Association Rules
    rules = model.associationRules
    print("\n[ASSOCIATION RULES] Top 10 regole generali (ordinato per lift):")
    rules.sort(col("lift").desc()).show(10, truncate=False)

    # Regole più interessanti: lift > 1 e non Vaticano come conseguente
    print("\n[ASSOCIATION RULES] Regole alternative (lift>1, escludendo Vaticano):")
    filtered_rules = rules.filter(
        (col("lift") > 1.0) & (~array_contains(col("consequent"), "Vaticano"))
    ).sort(col("confidence").desc())

    filtered_rules.select("antecedent", "consequent", "confidence", "lift").show(10, truncate=False)

    #Salvataggio regole in parquet
    print(f"\nSalvataggio delle regole in: {output_path} ...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    rules.write.parquet(output_path)
    print("Salvataggio completato.")

    transactions_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_trajectory_mining()