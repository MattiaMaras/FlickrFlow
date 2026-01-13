import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, dayofweek, date_format, min, max, datediff, count, countDistinct, \
    when, stddev
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_advanced_features():
    init_environment()
    spark = SparkSession.builder \
        .appName("FlickrFlow_Ultimate_Features") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- AVVIO FEATURE ENGINEERING AVANZATO ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")
    df = spark.read.parquet(input_path)

    # 1. FEATURE TEMPORALI (Per pag. Trend)
    print("1. Estrazione feature temporali...")
    df_time = df.withColumn("hour", hour("timestamp")) \
        .withColumn("month", month("timestamp")) \
        .withColumn("day_of_week", dayofweek("timestamp")) \
        .withColumn("day_name", date_format("timestamp", "EEEE")) \
        .withColumn("year", date_format("timestamp", "yyyy"))

    # Salvo dataset temporale
    trend_output = os.path.join(base_dir, "..", "data", "flickr_time_features.parquet")
    if os.path.exists(trend_output): shutil.rmtree(trend_output)
    df_time.write.parquet(trend_output)

    # 2. USER PROFILING (Per pag. Utenti & ML)
    print("2. Creazione profili utente complessi...")

    # Calcolo metriche base
    user_stats = df_time.groupBy("user_id").agg(
        min("timestamp").alias("first_seen"),
        max("timestamp").alias("last_seen"),
        count("photo_id").alias("total_photos"),
        countDistinct("roi").alias("unique_rois"),
        # Calcoliamo deviazione standard dell'ora (per vedere se scattano solo di notte o sempre)
        stddev("hour").alias("hour_stddev")
    )

    # Feature Engineering
    user_features = user_stats.withColumn("days_active", datediff(col("last_seen"), col("first_seen"))) \
        .withColumn("photos_per_day", col("total_photos") / (col("days_active") + 1)) \
        .na.fill(0)

    # Labeling: Local se attivo > 30gg - O se ha visitato > 5 RoI diverse
    # Aggiungo colonna user_type esplicita per la UI
    user_labeled = user_features.withColumn("label",
                                            when((col("days_active") > 30) | (col("unique_rois") > 4), 1.0).otherwise(
                                                0.0)
                                            ).withColumn("user_type",
                                                         when(col("days_active") > 30, "Resident")
                                                         .when(col("days_active") <= 7, "Tourist")
                                                         .otherwise("Recurring Visitor")
                                                         )

    # 3. ML CLASSIFICATION (Per pag. Classificazione)
    print("3. Addestramento Modello Classificazione (Random Forest)...")

    assembler = VectorAssembler(
        inputCols=["total_photos", "unique_rois", "days_active", "photos_per_day"],
        outputCol="features"
    )
    ml_data = assembler.transform(user_labeled)

    # Train/Test Split
    train, test = ml_data.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    model = rf.fit(train)

    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    acc = evaluator.evaluate(predictions)
    print(f"   Accuracy Modello: {acc:.2%}")

    # Salvo profili
    user_output = os.path.join(base_dir, "..", "data", "flickr_user_profiles.parquet")
    if os.path.exists(user_output): shutil.rmtree(user_output)
    user_labeled.write.parquet(user_output)

    print("Feature Engineering completato.")
    spark.stop()


if __name__ == "__main__":
    run_advanced_features()