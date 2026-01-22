import os

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from backend.utils import get_spark_session


def run_clustering():
    spark = get_spark_session("FlickrFlow_Phase5_KMeans")

    spark.sparkContext.setLogLevel("ERROR")
    print("\n" + "=" * 60)
    print("--- FASE 5: K-MEANS CLUSTERING ---")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "..","data", "flickr_enriched.parquet"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "flickr_clusters.parquet"))

    if not os.path.exists(input_path):
        print("Errore: Dataset non trovato. Esegui la fase 3")
        return

    df = spark.read.parquet(input_path)

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

    print("Calcolo Silhouette Score...")
    # Uso un sample del 20% per la valutazione per non bloccare il pc per minuti
    silhouette = evaluator.evaluate(predictions.sample(False, 0.2))
    print(f"Silhouette = {silhouette:.4f}")

    # 5. Risultati
    print("\nCentroidi dei Cluster identificati (Lat, Lon):")
    centers = model.clusterCenters()
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")

    # 6. Interpretazione
    print("\nDistribuzione punti per Cluster:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()

    print("Salvo i cluster su disco...")
    predictions.write.mode("overwrite").parquet(output_path)

    vector_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_clustering()