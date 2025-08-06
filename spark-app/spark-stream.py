import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, FloatType

# 📁 Dossier surveillé
input_dir = "../import/"
table_name = "zh12"
batch_size = 1000
scan_interval = 30  # secondes entre chaque scan

# 🔐 Connexions PostgreSQL
pg_url = "jdbc:postgresql://postgres_db:5432/ai_db"
pg_properties = {
    "user": "ai_user",
    "password": "ai_pass",
    "driver": "org.postgresql.Driver"
}

# 🚀 Démarrage Spark
spark = SparkSession.builder \
    .appName("CSVWatcher") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.2.27") \
    .getOrCreate()

# 📀 Schéma complet
schema = StructType([
    StructField("Matricule", StringType(), True),
    StructField("Date", TimestampType(), True),
    StructField("# de tâche", IntegerType(), True),
    StructField("Code tâche", StringType(), True),
    StructField("Jira", StringType(), True),
    StructField("Code Service", StringType(), True),
    StructField("Heures déclarées", StringType(), True),
    StructField("Durée tâche (heures)", FloatType(), True),
    StructField("Unité quantité base", StringType(), True),
    StructField("Créé par", StringType(), True),
    StructField("Saisi le", TimestampType(), True),
    StructField("Heure de création", StringType(), True),
    StructField("Confirmé", StringType(), True),
    StructField("Commentaire", StringType(), True)
])

# 🛡️ Création de la table si elle n'existe pas
def ensure_table_exists():
    try:
        spark.read.jdbc(pg_url, table_name, properties=pg_properties).limit(1).collect()
    except:
        spark.createDataFrame(spark.sparkContext.emptyRDD(), schema) \
            .write.mode("overwrite").jdbc(pg_url, table_name, properties=pg_properties)

# 🔨 Prétraitement du DataFrame
def preprocess_dataframe(df):
    df = df.dropna(subset=["Matricule", "Date", "# de tâche"])
    df = df.withColumn("Date", to_timestamp("Date"))
    df = df.filter(col("Date").isNotNull())
    return df

# 🔁 Liste des fichiers déjà vus
deja_vus = set()
print("🚀 En attente de nouveaux fichiers CSV dans :", input_dir)

# ⚐ Boucle de surveillance
while True:
    fichiers = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    nouveaux = [f for f in fichiers if f not in deja_vus]

    for file in nouveaux:
        file_path = os.path.join(input_dir, file)
        print(f"\n📄 Nouveau fichier détecté : {file}")

        try:
            df = spark.read.format("csv") \
                .option("header", "true") \
                .schema(schema) \
                .load(file_path)

            df = preprocess_dataframe(df)
            total_rows = df.count()

            ensure_table_exists()
            try:
                existing_df = spark.read.jdbc(pg_url, table_name, properties=pg_properties)
                print(f"✅ Table `{table_name}` existante, suppression/remplacement des doublons...")

                join_cond = [
                    existing_df["Matricule"] == df["Matricule"],
                    existing_df["Date"] == df["Date"],
                    existing_df["# de tâche"] == df["# de tâche"]
                ]

                updated_df = existing_df.join(df, join_cond, "left_anti").unionByName(df)
                updated_df.write.mode("overwrite").jdbc(pg_url, table_name, properties=pg_properties)
            except Exception as e:
                print(f"❌ Erreur pendant l'update : {e}")
                df.write.mode("overwrite").jdbc(pg_url, table_name, properties=pg_properties)

            print(f"✅ {total_rows} lignes insérées pour : {file}")
            os.remove(file_path)
            print(f"🗑️ Fichier supprimé : {file}")
            deja_vus.add(file)

        except Exception as e:
            print(f"❌ Erreur avec {file} : {e}")

    time.sleep(scan_interval)
