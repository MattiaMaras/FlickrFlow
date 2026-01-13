import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, desc
from pyspark.sql.window import Window

def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")

def run_prepare_queries():
    init_environment()
    spark = SparkSession.builder \
        .appName("FlickrFlow_QueryPrep") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- PREPARAZIONE VISTE PER DASHBOARD ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Input files
    enriched_path = os.path.join(data_dir, "flickr_enriched.parquet")
    profiles_path = os.path.join(data_dir, "flickr_user_profiles.parquet") # Contiene user_type

    if not os.path.exists(enriched_path) or not os.path.exists(profiles_path):
        print("Errore: Dataset mancanti (eseguire feature_engineering.py prima).")
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

    # 2. VIEW: Origin-Destination Matrix (Flussi)
    print("2. Generazione Matrice O/D (Flussi)...")
    # Ordiniamo per utente e tempo
    window_spec = Window.partitionBy("user_id").orderBy("timestamp")
    
    #colonna 'next_roi'
    traj_df = full_df.filter(col("roi") != "Unknown") \
        .select("user_id", "roi", "timestamp", "user_type") \
        .withColumn("next_roi", lead("roi", 1).over(window_spec)) \
        .filter(col("next_roi").isNotNull()) \
        .filter(col("roi") != col("next_roi")) # Rimuoviamo self-loop immediati
    
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
