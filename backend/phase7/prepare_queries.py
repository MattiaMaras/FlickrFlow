import os

from pyspark.sql.functions import col, lead, desc
from pyspark.sql.window import Window

from backend.utils import get_spark_session


def run_prepare_queries():
    spark = get_spark_session("FlickrFlow_Phase7_Query")

    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "=" * 60)
    print("--- PREPARAZIONE VISTE PER DASHBOARD ---")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "data"))
    
    enriched_path = os.path.join(data_dir, "flickr_enriched.parquet")
    profiles_path = os.path.join(data_dir, "flickr_user_profiles.parquet") # Contiene user_type

    if not os.path.exists(enriched_path) or not os.path.exists(profiles_path):
        print("Errore: Dataset mancanti. Eseguire fasi 3 e 5.")
        return

    df = spark.read.parquet(enriched_path)
    profiles = spark.read.parquet(profiles_path)

    # Join con profili per avere user_type nel dataset principale
    full_df = df.join(profiles.select("user_id", "user_type"), "user_id", "left")
    full_df.cache()

    # 1. VIEW: Hourly Heatmap (ROI x Ora)
    print("1. Generazione Hourly Heatmap...")
    hourly_view = full_df.filter(col("roi") != "Unknown") \
        .withColumn("hour", col("timestamp").cast("timestamp").substr(12, 2).cast("int")) \
        .groupBy("roi", "hour", "user_type") \
        .count() \
        .orderBy("roi", "hour")
    
    hourly_view.write.mode("overwrite").parquet(os.path.join(data_dir, "view_hourly_heatmap.parquet"))

    # 2. VIEW: Origin-Destination Matrix
    print("2. Generazione Matrice O/D (Flussi)...")
    window_spec = Window.partitionBy("user_id").orderBy("timestamp")
    
    #colonna 'next_roi'
    traj_df = full_df.filter(col("roi") != "Unknown") \
        .select("user_id", "roi", "timestamp", "user_type") \
        .withColumn("next_roi", lead("roi", 1).over(window_spec)) \
        .filter(col("next_roi").isNotNull()) \
        .filter(col("roi") != col("next_roi"))
    
    od_matrix = traj_df.groupBy("roi", "next_roi", "user_type") \
        .count() \
        .withColumnRenamed("roi", "source") \
        .withColumnRenamed("next_roi", "target") \
        .withColumnRenamed("count", "weight") \
        .orderBy(desc("weight"))
        
    od_matrix.write.mode("overwrite").parquet(os.path.join(data_dir, "view_od_matrix.parquet"))

    # 3. VIEW: Hidden Gems (Luoghi con alto ratio Residenti/Totale)
    print("3. Calcolo Hidden Gems...")
    roi_stats = full_df.filter(col("roi") != "Unknown") \
        .groupBy("roi") \
        .pivot("user_type", ["Tourist", "Resident"]) \
        .count() \
        .na.fill(0)
    
    # Calcolo indice "Localness"
    # (Resident + 1) / (Tourist + 1) per evitare divisioni per zero
    hidden_gems = roi_stats.withColumn("total", col("Tourist") + col("Resident")) \
        .withColumn("localness_index", (col("Resident") + 1) / (col("total") + 1)) \
        .orderBy(desc("localness_index"))

    hidden_gems.write.mode("overwrite").parquet(os.path.join(data_dir, "view_hidden_gems.parquet"))

    print("Viste preparate correttamente.")
    full_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    run_prepare_queries()
