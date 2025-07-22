import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from xgboost import XGBRegressor
from tqdm import tqdm
import unicodedata
import subprocess
import importlib
import swifter 
import joblib
import spacy
import ijson
import nltk
import json
import re
import os

merged_df = pd.read_csv("merged_libelle_clean.csv", sep=";", encoding="utf-8")

# 🧹 Supprimer toutes les colonnes dont le nom commence par "tfidf_"
merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith("tfidf_")]

# Concaténer les colonnes nettoyées
textes_concat = (merged_df["summary"].fillna("") + " " + merged_df["description"].fillna("") + " " + merged_df["commentaire_filtré_clean"].fillna("")).str.strip()

max_len = textes_concat.str.len().max()
print("Longueur maximale :", max_len)

# Initialiser le vecteur TF-IDF
tfidf = TfidfVectorizer(max_features=50000)

# Appliquer le modèle
tfidf_matrix = tfidf.fit_transform(textes_concat)

# Créer un DataFrame à partir des vecteurs
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
)

# 🧠 Colonnes catégorielles à encoder
cat_features_nlp = ["Matricule", "fields.project.key", "Type_Étendu"]
enc_nlp = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_cat = enc_nlp.fit_transform(merged_df[cat_features_nlp])
encoded_df = pd.DataFrame(encoded_cat, columns=cat_features_nlp)

# 🔢 Variables numériques supplémentaires
cols_num = [
    "nb_key", "Historique_1an",
    # "annee_commence", "mois_commence", "jour_semaine_commence",
    "annee_creation", "mois_creation", "jour_semaine",
    "delai_creation_action_h"
]

# 🔄 Fusion finale des features pour entraînement
x_nlp = pd.concat([
    encoded_df.reset_index(drop=True),
    merged_df[cols_num].reset_index(drop=True),
    tfidf_df.reset_index(drop=True)
], axis=1)

y_nlp = merged_df["Duree"].copy()
del merged_df

# 📊 Split
x_train_nlp, x_test_nlp, y_train_nlp, y_test_nlp = train_test_split(
    x_nlp, y_nlp, test_size=0.2, random_state=42
)

# 🧱 Pipeline : scaler + XGBoost
pipeline_nlp = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(n_estimators=100, random_state=42, verbosity=1))
])

# 🚀 Entraînement
pipeline_nlp.fit(x_train_nlp, y_train_nlp)

# 📈 Évaluation
pred_nlp = pipeline_nlp.predict(x_test_nlp)
errors = abs(y_test_nlp - pred_nlp)

# 📈 Résumé
mae = mean_absolute_error(y_test_nlp, pred_nlp)
rmse = np.sqrt(mean_squared_error(y_test_nlp, pred_nlp))
min_err = errors.min()
max_err = errors.max()

print(f"📊 MAE avec NLP + Dates + TF-IDF : {mae:.4f} heures")
print(f"📊 RMSE avec NLP + Dates + TF-IDF : {rmse:.4f} heures")
print(f"✅ Erreur minimale : {min_err:.4f} heures")
print(f"❌ Erreur maximale : {max_err:.4f} heures")

# ⚠️ Assure-toi d’avoir déjà défini `errors = abs(y_test_nlp - pred_nlp)` avant ce bloc
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='steelblue', edgecolor='black')
plt.title("📈 Distribution des erreurs absolues de prédiction")
plt.xlabel("Erreur absolue (heures)")
plt.ylabel("Nombre de tickets")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 💾 Sauvegardes
joblib.dump(pipeline_nlp, "models/xgb_pipeline_nlp.pkl")
joblib.dump(enc_nlp, "models/xgb_encoder_nlp.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
