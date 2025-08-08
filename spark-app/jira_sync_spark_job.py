import time
import gc
from pyspark.sql import SparkSession
from config import JIRA_URL, JIRA_EMAIL, JIRA_TOKEN
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from pymongo import MongoClient, UpdateOne

# Spark init avec mémoire augmentée
spark = SparkSession.builder \
    .appName("jira_sync_job") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.mongodb.output.uri", "mongodb://root:exemplePass@mongodb:27017/jiradb.issues?authSource=admin") \
    .getOrCreate()

# Connexion Mongo
client = MongoClient("mongodb://root:exemplePass@mongodb:27017/?authSource=admin")
mongo_db = client["jiradb"]
collection = mongo_db["issues"]

PROJECTS = ["SCMF", "ULIS"]
BATCH_LIMIT = 10000

def fetch_issues(jql_query, max_results=100, max_retries=3):
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
    headers = {"Accept": "application/json"}
    all_issues = []
    start_at = 0

    while True:
        params = {"jql": jql_query, "startAt": start_at, "maxResults": max_results}
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(JIRA_URL, headers=headers, params=params, auth=auth)
                if response.status_code != 200:
                    raise RuntimeError(f"❌ API Jira : {response.status_code} {response.text}")
                break  # Sortie de la boucle si OK
            except (requests.ConnectionError, requests.exceptions.RequestException) as e:
                attempt += 1
                print(f"⚠️ Connexion échouée ({attempt}/{max_retries}) : {e}")
                if attempt < max_retries:
                    print("⏳ Attente de 5 secondes avant nouvel essai...")
                    time.sleep(5)
                else:
                    raise RuntimeError("❌ Nombre maximal de tentatives atteint.")

        data = response.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        start_at += max_results
        if start_at >= data.get("total", 0):
            break
    return all_issues

# --------- Synchronisation ---------
today_str = datetime.now().strftime('%Y-%m-%d')
collection_exists = "issues" in mongo_db.list_collection_names()
collection_empty = collection.count_documents({}) == 0 if collection_exists else True

for project in PROJECTS:
    if collection_empty:
        jql = f"project = {project}"
        print(f"\n📥 Collection vide, récupération complète des tickets du projet {project}...")
    else:
        jql = f"project = {project} AND updated >= {today_str}"
        print(f"\n🔄 Projet {project} — Tickets mis à jour aujourd'hui ({today_str})...")

    try:
        issues = fetch_issues(jql)
    except RuntimeError as err:
        print(f"❌ Abandon récupération pour le projet {project} : {err}")
        continue

    if not issues:
        print(f"⚠️ Aucun ticket trouvé pour le projet {project}.")
        continue

    print(f"🔍 {len(issues)} tickets récupérés depuis Jira.")
    existing_keys = set(collection.distinct("key"))
    new_batch = []
    update_ops = []

    for i, issue in enumerate(issues, 1):
        key = issue["key"]
        if key in existing_keys:
            update_ops.append(UpdateOne({"key": key}, {"$set": issue}, upsert=True))
        else:
            new_batch.append(issue)

        if len(new_batch) >= BATCH_LIMIT:
            print(f"🚨 Insertion Spark de {len(new_batch)} tickets")
            spark.createDataFrame(new_batch).write.format("mongo").mode("append").save()
            new_batch.clear()
            gc.collect()

        if len(update_ops) >= BATCH_LIMIT:
            print(f"🚨 Update Mongo de {len(update_ops)} tickets")
            collection.bulk_write(update_ops)
            update_ops.clear()
            gc.collect()

    if new_batch:
        print(f"📦 Insertion finale de {len(new_batch)} tickets")
        spark.createDataFrame(new_batch).write.format("mongo").mode("append").save()
    if update_ops:
        print(f"📦 Update final de {len(update_ops)} tickets")
        collection.bulk_write(update_ops)

    print(f"✅ Projet {project} synchronisé avec succès.")

spark.stop()
