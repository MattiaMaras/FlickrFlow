import os
from pyspark.sql import SparkSession


def init_environment():
    # Percorso di installazione standard di OpenJDK 17 via Homebrew
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")
    else:
        print(f"Warning: JDK 17 non trovata in {jdk_path}. Verificare l'installazione.")


def inspect_dataset():
    init_environment()

    #Spark Session Initialization
    # Utilizziamo 'local[*]' per simulare un cluster utilizzando tutti i thread disponibili sulla macchina
    try:
        spark = SparkSession.builder \
            .appName("FlickrFlow_ETL_Test") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

    except Exception as e:
        print(f"Critical Error during Spark Driver initialization: {e}")
        return

    # Path Resolution
    # Risoluzione dinamica del percorso relativo per garantire portabilità del codice
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "data", "flickr2x.json")

    #Schema Inference
    if os.path.exists(json_path):
        try:
            print(f"Reading dataset from: {json_path}")
            # L'opzione multiline è necessaria per gestire correttamente i record JSON distribuiti su più righe
            df = spark.read.option("multiline", "true").json(json_path)

            print("\n--- INFERRED SCHEMA ---")
            df.printSchema()

        except Exception as e:
            print(f"Error reading DataFrame: {e}")
    else:
        print("Error: Dataset file not found in 'data' directory.")

    spark.stop()


if __name__ == "__main__":
    inspect_dataset()