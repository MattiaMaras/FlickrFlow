import os
from backend.utils import get_spark_session


def inspect_dataset():
    #Spark Session Initialization
    # 'local[*]' per simulare un cluster utilizzando tutti i thread disponibili sulla macchina
    try:
        spark = get_spark_session("FlickrFlow_InspectSchema")
        spark.sparkContext.setLogLevel("ERROR")

    except Exception as e:
        print(f"Critical Error during Spark Driver initialization: {e}")
        return

    # Risoluzione dinamica del percorso relativo per garantire portabilità del codice
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "..", "data", "flickr2x.json")

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