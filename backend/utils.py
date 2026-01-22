import os
from pyspark.sql import SparkSession

def init_environment():
    java_home = os.environ.get("JAVA_HOME")

    if not java_home:
        possible_paths = [
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
            "/usr/lib/jvm/java-17-openjdk-amd64",
            "/usr/lib/jvm/default-java"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                java_home = path
                break

    if java_home:
        os.environ["JAVA_HOME"] = java_home
        if java_home not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{java_home}/bin:" + os.environ.get("PATH", "")
        print(f"Environment: JAVA_HOME set to {java_home}")
    else:
        print("WARNING: JAVA_HOME non trovata. Assicurati che Java 17 sia installato e configurato.")
        print("PySpark potrebbe fallire se non trova Java.")

def get_spark_session(app_name, memory="4g"):
    init_environment()
    
    builder = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", memory) \
        .config("spark.sql.ansi.enabled", "false") \
        .config("spark.sql.caseSensitive", "true") 
        
    return builder.getOrCreate()
