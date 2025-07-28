import pandas as pd
import joblib
import os
from datetime import datetime
from config import DB_BACKEND, SQLITE_PATH, POSTGRES_CONFIG
from apscheduler.schedulers.background import BackgroundScheduler
from module import (
    get_connection,
    get_jira_tickets_dataframe,
    get_filtered_jira_issues,
    traiter_historique_1an_bis,
    preprocess_libelle,
    vectoriser_tfidf_merged_df
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def train_model():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \U0001f501 Démarrage du job de mise à jour du modèle...")
    conn = None
    try:
        conn = get_connection()
        print("✅ Connexion à la base établie.")
        cursor = conn.cursor()

        # Vérification de la table zh12
        print("🔍 Vérification de la présence de la table 'zh12'...")
        if DB_BACKEND == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='zh12';")
        else:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'zh12'
                );
            """)
        exists = cursor.fetchone()
        if not exists or (isinstance(exists, tuple) and not exists[0]):
            print("❌ La table 'zh12' est introuvable dans la base.")
            return
        print("✅ Table 'zh12' détectée.")

        # Création de data_entrain si nécessaire
        print("🛠️ Création de la table 'data_entrain' si absente...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_entrain (
                matricule TEXT,
                ticket TEXT,
                check TEXT DEFAULT NULL
            );
        """)
        conn.commit()

        # Tickets manquants
        print("🔎 Récupération des tickets non encore entraînés...")
        query = """
            SELECT DISTINCT z.matricule, z.ticket
            FROM zh12 z
            WHERE NOT EXISTS (
                SELECT 1 FROM data_entrain d
                WHERE d.matricule = z.matricule AND d.ticket = z.ticket AND d.check != 'X'
            );
        """
        df_train = pd.read_sql_query(query, conn)

        if df_train.empty:
            print("⚠️ Aucun nouveau ticket à entraîner.")
            return

        print(f"✅ {len(df_train)} ticket(s) à traiter.")
        tickets = df_train["ticket"].tolist()

        # Récup Jira
        print("📥 Chargement des tickets Jira...")
        df = get_jira_tickets_dataframe(tickets)
        df_all_ticket = get_filtered_jira_issues(df)
        df_merged = pd.concat([df, df_all_ticket], ignore_index=True).drop_duplicates(subset="key")

        if df_merged.empty:
            print("⚠️ Aucun ticket Jira enrichi trouvé.")
            return

        print("🧬 Fusion et enrichissement ZH12...")
        df = traiter_historique_1an_bis(df_merged=df_merged)

        # Nettoyage texte
        print("🧹 Prétraitement texte...")
        df["nb_key"] = df["key"].str.extract(r"-(\d+)", expand=False).astype(int)
        df["fields.summary"] = df["fields.summary"].fillna("")
        df["summary"] = df["fields.summary"].apply(preprocess_libelle)
        df["fields.description"] = df["fields.description"].fillna("")
        df["description"] = df["fields.description"].apply(preprocess_libelle)

        df["target"] = df["Type_Étendu"]
        cat_cols = ["Matricule", "fields.project.key", "Type_Étendu"]

        # Charger encodeur si existant
        encoder_path = "xgb_encoder_nlp.pkl"
        if os.path.exists(encoder_path):
            print("🔁 Chargement de l'encodeur existant...")
            encoder = joblib.load(encoder_path)
            df[cat_cols] = encoder.transform(df[cat_cols].fillna(""))
        else:
            print("✨ Nouveau encodeur créé.")
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[cat_cols] = encoder.fit_transform(df[cat_cols].fillna(""))

        print("🧠 TF-IDF vectorisation...")
        df_vect = vectoriser_tfidf_merged_df(df)
        exclude = ["target", "key", "fields.summary", "fields.description", "summary", "description"]
        features = df_vect.drop(columns=[c for c in exclude if c in df_vect.columns], errors='ignore')

        X = features.values
        y = df["target"]

        print("🔀 Split train/test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Charger ou créer le modèle
        model_path = "xgb_pipeline_nlp.pkl"
        if os.path.exists(model_path):
            print("🔁 Chargement du modèle existant pour mise à jour...")
            model = XGBClassifier()
            model.load_model(model_path)
            model.fit(X_train, y_train, xgb_model=model)  # mise à jour
        else:
            print("✨ Création d'un nouveau modèle...")
            model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            model.fit(X_train, y_train)

        print("📊 Évaluation...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        print("💾 Sauvegarde du modèle et de l'encodeur...")
        model.save_model(model_path)
        joblib.dump(encoder, encoder_path)
        print("✅ Mise à jour réussie.")

    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement : {e}")

    finally:
        if conn:
            conn.close()
            print("🔒 Connexion fermée.")


if __name__ == "__main__":
    train_model()  # premier lancement
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model, trigger='interval', hours=24)
    scheduler.start()

    print("📆 Scheduler actif. Ctrl+C pour stopper.")
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("🛑 Scheduler arrêté.")
