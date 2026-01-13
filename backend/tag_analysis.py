import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, desc

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")

def run_tag_analysis():
    init_environment()
    spark = SparkSession.builder \
        .appName("FlickrFlow_TagAnalysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- AVVIO ANALISI TAG (NLP) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")
    
    if not os.path.exists(input_path):
        print("Errore: Dataset enriched non trovato.")
        return

    df = spark.read.parquet(input_path)

    # 1. Parsing dei Tag
    # I tag in Flickr spesso sono stringhe separate da spazio
    print("Parsing e pulizia dei tag...")
    
    # Filtro foto senza ROI (Unknown) per focalizzarmi sui luoghi
    roi_df = df.filter(col("roi") != "Unknown").select("roi", "tags")

    exploded_df = roi_df.select("roi", explode(col("tags")).alias("word"))

    # 2. Pulizia
    # Convertiamo in lower case e rimuoviamo caratteri strani
    clean_df = exploded_df.withColumn("word", lower(col("word"))) \
        .filter(
        (col("word") != "") & 
        (col("word").rlike("^[a-z]+$")) & # Solo lettere
        (col("word").isin(["rome", "roma", "italy", "italia", "lazio", "photo", "flickr"]) == False)
    )

    # 3. Aggregazione per ROI
    print("Calcolo frequenza tag per RoI...")
    roi_tags = clean_df.groupBy("roi", "word") \
        .count() \
        .orderBy("roi", desc("count"))


    window_spec = Window.partitionBy("roi").orderBy(desc("count"))
    top_tags = roi_tags.withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") <= 20) \
        .drop("rank")

    # 4. Salvataggio
    output_path = os.path.join(base_dir, "..", "data", "flickr_roi_tags.parquet")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    top_tags.write.parquet(output_path)
    print(f"Analisi Tag completata. Dati salvati in: {output_path}")
    
    # Preview
    top_tags.show(10)
    spark.stop()

if __name__ == "__main__":
    run_tag_analysis()
