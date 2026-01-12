import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def init_environment():
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def run_clustering():
    init_environment()

    # Inizializzo Spark Session
    spark = SparkSession.builder \
        .appName("FlickrFlow_Phase5_KMeans") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("\n--- FASE 5: K-MEANS CLUSTERING ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "flickr_enriched.parquet")

    if not os.path.exists(input_path):
        print("Errore: Dataset non trovato.") #Da eseguire Fase 3 (spatial_enrichment.py)
        return

    # Carico i dati
    df = spark.read.parquet(input_path)

    # 1. Feature Engineering
    # Spark MLlib richiede che le feature di input siano in una singola colonna di tipo Vector.
    # Uso VectorAssembler per combinare Latitude e Longitude.
    print("Preparo i dati per il Machine Learning (VectorAssembler)...")
    vec_assembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")

    # Trasformo il dataset aggiungendo la colonna 'features'
    vector_df = vec_assembler.transform(df)

    # Ottimizzazione: Caching del dataset vettorizzato
    vector_df.cache()

    # 2. Configurazione Modello K-Means
    # Scelgo k=10 per identificare i 10 poli turistici principali di Roma.
    # setSeed garantisce la riproducibilità dei risultati.
    k_value = 10
    print(f"Addestramento modello K-Means con k={k_value}...")

    kmeans = KMeans().setK(k_value).setSeed(1).setFeaturesCol("features")

    # 3. Training
    model = kmeans.fit(vector_df)

    # 4. Evaluation
    # Calcolo il Silhouette Score per misurare la qualità dei cluster.
    # Valori vicini a 1 indicano cluster ben separati.
    predictions = model.transform(vector_df)
    evaluator = ClusteringEvaluator()

    # Nota: Il calcolo della Silhouette su 2 milioni di punti è molto oneroso (O(N^2)).
    print("Calcolo Silhouette Score...")
    # Uso un sample del 20% per la valutazione per non bloccare il pc per minuti
    silhouette = evaluator.evaluate(predictions.sample(False, 0.2))
    print(f"Silhouette = {silhouette:.4f}")

    # 5. Risultati: I Centroidi
    print("\nCentroidi dei Cluster identificati (Lat, Lon):")
    centers = model.clusterCenters()
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")

    # 6. Interpretazione
    print("\nDistribuzione punti per Cluster:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()

    # Cleanup
    vector_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_clustering()