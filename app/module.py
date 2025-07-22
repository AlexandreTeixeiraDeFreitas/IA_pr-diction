from sklearn.feature_extraction.text import TfidfVectorizer
from config import JIRA_URL, JIRA_EMAIL, JIRA_TOKEN
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from spacy.util import is_package
from spacy.cli import download
import pandas as pd
import unicodedata
import subprocess
import importlib
import requests
import swifter
import sqlite3
import joblib
import spacy
import ijson
import nltk
import json
import time
import ast
import re
import os

tfidf = joblib.load("tfidf_vectorizer.pkl")

# 📦 Téléchargement auto des stopwords NLTK
try:
    stop_fr = set(stopwords.words('french'))
    stop_en = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_fr = set(stopwords.words('french'))
    stop_en = set(stopwords.words('english'))

# 🧠 Initialisation des correcteurs d'orthographe
spell_fr = SpellChecker(language='fr')
spell_en = SpellChecker(language='en')

# 🧠 Fonction de chargement sécurisé des modèles spaCy
def safe_spacy_load(model_name):
    if is_package(model_name):
        return spacy.load(model_name)
    else:
        print(f"🔄 Modèle {model_name} non installé. Téléchargement en cours...")
        download(model_name)
        print(f"✅ Modèle {model_name} téléchargé. 🔁 Veuillez redémarrer le kernel, puis relancer la cellule.")
        return None

# 📦 Chargement des modèles linguistiques
nlp_fr = safe_spacy_load("fr_core_news_sm")
nlp_en = safe_spacy_load("en_core_web_sm")

if nlp_fr is None or nlp_en is None:
    raise SystemExit("🛑 Redémarre le kernel avant de continuer. Modèles téléchargés.")

# Caches pour optimiser
correction_cache = {}
lemmatisation_cache = {}

# Liste blanche de mots techniques/acronymes à préserver
whitelist = {"idoc", "sap", "vpn", "teams", "jira", "sql", "pdf", "ssl"}

def get_jira_tickets_dataframe(ticket_ids):
    """
    Récupère une liste de tickets Jira et retourne un DataFrame à plat.
    
    Args:
        ticket_ids (List[str]): Liste d'identifiants Jira (ex: ["PROJ-1", "PROJ-2"])
    
    Returns:
        pd.DataFrame: Données Jira à plat par ticket
    """
    auth = (JIRA_EMAIL, JIRA_TOKEN)
    headers = {"Accept": "application/json"}
    all_data = []

    for ticket_id in ticket_ids:
        url = f"{JIRA_URL}/rest/api/latest/issue/{ticket_id}"
        response = requests.get(url, auth=auth, headers=headers)

        if response.status_code == 200:
            data = response.json()
            flat_data = pd.json_normalize(data)
            all_data.append(flat_data)
        else:
            print(f"Erreur {response.status_code} pour le ticket {ticket_id}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def clean_encoding(text):
    try:
        text = text.encode('latin1').decode('utf-8')
    except:
        pass
    text = text.replace("�", "")
    return unicodedata.normalize('NFKC', text)

def corriger_orthographe(mot):
    if mot in correction_cache:
        return correction_cache[mot]
    corr = spell_fr.correction(mot)
    if corr is None or corr == mot:
        corr = spell_en.correction(mot)
    correction_cache[mot] = corr if corr else mot
    return correction_cache[mot]

def lemmatiser_mot(mot):
    if mot in lemmatisation_cache:
        return lemmatisation_cache[mot]
    doc_fr = nlp_fr(mot)
    if doc_fr and doc_fr[0].lemma_ != mot:
        lemmatisation_cache[mot] = doc_fr[0].lemma_
        return lemmatisation_cache[mot]
    doc_en = nlp_en(mot)
    lemmatisation_cache[mot] = doc_en[0].lemma_ if doc_en else mot
    return lemmatisation_cache[mot]

def preprocess_libelle(text):
    if not isinstance(text, str):
        return ""
    # Nettoyage de l'encodage et de la ponctuation
    text = clean_encoding(text)
    text = re.sub(r"[^\w\s]", " ", text)
    mots = text.split()

    mots_cles = []
    for m in mots:
        m_lower = m.lower()
        # Option 1 : ignorer les majuscules (acronymes)
        if m.isupper() and len(m) > 2:
            mots_cles.append(m_lower)
            continue
        # Option 2 : ignorer les mots en whitelist
        if m_lower in whitelist:
            mots_cles.append(m)
            continue
        # Option 3 : stop words + correction/lemmatisation
        if m not in stop_fr and m not in stop_en and len(m) > 2:
            mot_corr = corriger_orthographe(m_lower)
            mot_lem = lemmatiser_mot(mot_corr)
            mots_cles.append(mot_lem)
    return " ".join(mots_cles)

def get_filtered_jira_issues(df_tickets):
    """
    Récupère les tickets Jira correspondant aux composants, types et champs personnalisés d'un DataFrame :
    - dont "fields.resolution.name" == "Done"
    - dont "fields.components" contient au moins un composant présent dans df_tickets
    - dont "fields.customfield_10116.value" contient une valeur présente dans df_tickets (filtré après)
    - dont "fields.issuetype.name" correspond à un type présent dans df_tickets
    - créés dans l'année précédente par rapport à leur propre date de création

    Args:
        df_tickets (pd.DataFrame): Un DataFrame avec des colonnes Jira standards

    Returns:
        pd.DataFrame: Tickets filtrés
    """
    if "key" not in df_tickets.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'key' avec les identifiants Jira")

    auth = (JIRA_EMAIL, JIRA_TOKEN)
    headers = {"Accept": "application/json"}
    results = []

    # Extraction des composants, types d'issue et valeurs custom à réutiliser
    component_set = set()
    issuetype_set = set()
    custom_value_set = set()

    if "fields.components" in df_tickets.columns:
        df_tickets["fields.components"].dropna().apply(
            lambda lst: [component_set.add(comp["name"]) for comp in lst if isinstance(comp, dict)]
        )
    if "fields.issuetype.name" in df_tickets.columns:
        issuetype_set = set(df_tickets["fields.issuetype.name"].dropna().tolist())
    if "fields.customfield_10116.value" in df_tickets.columns:
        custom_value_set = set(df_tickets["fields.customfield_10116.value"].dropna().tolist())

    url = f"{JIRA_URL}/rest/api/latest/search"

    for issuetype in issuetype_set:
        for component in component_set:
            jql = f"created >= -365d AND component = \"{component}\" AND issuetype = \"{issuetype}\" AND resolution = Done"
            start_at = 0

            while True:
                params = {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": 100
                }

                response = requests.get(url, auth=auth, headers=headers, params=params)

                if response.status_code != 200:
                    print(f"⚠️ Erreur API Jira : {response.status_code} {response.text}")
                    break

                data = response.json()
                issues = data.get("issues", [])
                total = data.get("total", 0)

                for issue in issues:
                    # 🔍 Vérifie que les champs sont bien présents dans l'issue
                    fields = issue.get("fields")
                    if fields is None:
                        print(f"❌ Problème : 'fields' est None dans issue : {json.dumps(issue, indent=2)}")
                        continue  # Évite l'erreur en sautant l'itération

                    # ✅ Extraction sécurisée des champs Jira
                    resolution = ""
                    components = []
                    custom_value = None
                    issuetype_value = None

                    try:
                        if isinstance(fields.get("resolution"), dict):
                            resolution = fields["resolution"].get("name", "")

                        if isinstance(fields.get("components"), list):
                            components = fields["components"]

                        if isinstance(fields.get("customfield_10116"), dict):
                            custom_value = fields["customfield_10116"].get("value", None)

                        if isinstance(fields.get("issuetype"), dict):
                            issuetype_value = fields["issuetype"].get("name", None)

                    except Exception as e:
                        print(f"⚠️ Erreur inattendue lors de l'extraction des champs Jira : {e}")
                        print("Contenu de fields :", fields)


                    created = fields.get("created", "")

                    if not created:
                        continue

                    created_date = datetime.strptime(created[:10], "%Y-%m-%d")
                    date_min = created_date - timedelta(days=365)

                    component_names = [comp.get("name") for comp in components if isinstance(comp, dict)]
                    has_common_component = bool(component_set.intersection(component_names))
                    has_common_custom_value = custom_value in custom_value_set
                    has_common_issuetype = issuetype_value in issuetype_set

                    if (resolution == "Done" and
                        has_common_component and
                        has_common_issuetype and
                        date_min <= created_date <= created_date):

                        flat = pd.json_normalize(issue)
                        if has_common_custom_value:
                            results.append(flat)

                start_at += len(issues)
                if start_at >= total:
                    break

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def lire_zh12_depuis_sqlite(sqlite_path, table_name, jira_keys):
    """
    Charge uniquement les lignes de la base SQLite dont la colonne Jira est dans jira_keys.
    """
    conn = sqlite3.connect(sqlite_path)

    # Convertir les clés Jira en tuple SQL-safe pour l'IN clause
    placeholders = ','.join(['?'] * len(jira_keys))
    query = f"SELECT * FROM {table_name} WHERE Jira IN ({placeholders})"

    zh12 = pd.read_sql_query(query, conn, params=tuple(jira_keys))
    conn.close()
    return zh12

def tester_requete_sql(sqlite_path: str, table: str, tickets):
    """
    Exécute une requête SELECT * sur une table SQLite filtrée par une ou plusieurs clés Jira.
    
    Args:
        sqlite_path (str): Chemin vers la base SQLite
        table (str): Nom de la table (ex: "zh12")
        tickets (str | list[str]): Ticket Jira ou liste de tickets
        max_lignes (int): Nombre max de lignes à afficher
    """
    try:
        conn = sqlite3.connect(sqlite_path)
        print(f"📡 Connexion ouverte vers {sqlite_path}")

        # Préparation des tickets
        if isinstance(tickets, str):
            tickets = [tickets]
        tickets = [str(t).strip() for t in tickets if t]

        # Clause WHERE Jira IN (?, ?, ...)
        placeholders = ','.join(['?'] * len(tickets))
        requete_sql = f"SELECT * FROM {table} WHERE Jira IN ({placeholders})"

        print("🔎 Requête préparée :")
        print(f"{requete_sql}")
        print("🔑 Paramètres :", tickets)

        df = pd.read_sql_query(requete_sql, conn, params=tuple(tickets))
        conn.close()

        print(f"✅ {len(df)} ligne(s) récupérée(s).")
        return df

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return pd.DataFrame()


def construire_type_etendu(row, colonnes):
    valeurs = []
    for col in colonnes:
        val = str(row.get(col, "")).strip()
        if val:
            valeurs.append(val)
    return "__".join(valeurs) if valeurs else None

def calculer_historique_1an(df, df_reference, type_col="Type_Étendu", date_col="fields.created", projet_col="fields.project.key", duree_col="Duree"):
    history = []
    for idx, row in df.iterrows():
        typ = row[type_col]
        date = row[date_col]
        projet = row[projet_col]

        if pd.isna(typ) or pd.isna(date) or pd.isna(projet):
            history.append(0)
            continue

        date_min = date - timedelta(days=365)
        past = df_reference[
            (df_reference[projet_col] == projet) &
            (df_reference[date_col] < date) &
            (df_reference[date_col] >= date_min)
            # (df_reference[type_col] == typ)
        ]
        print(f"🔍 Historique pour {typ} dans {projet} entre {date_min} et {date}: {len(past)} entrées trouvées")
        mean_val = past[duree_col].mean() if not past.empty else 0
        history.append(mean_val)

    return history

# --------- MAIN PIPELINE ---------

def traiter_historique_1an(df_ticket: pd.DataFrame, df_merged: pd.DataFrame, zh12_path: str, date_commencement: str, matricule: str):
    """
    Calcule l'historique glissant 1 an pour les tickets Jira (df_ticket) enrichis via df_merged.
    Ajoute les tickets manquants dans ZH12 avec durée = 0.
    """
    # Copie des tickets enrichis
    jira_df = df_merged.copy()

    # Charger ZH12 uniquement pour les clés présentes dans Jira
    keep_keys = list(set(jira_df["key"]))
    zh12 = lire_zh12_depuis_sqlite(zh12_path, "zh12", keep_keys)
    print(f"📥 Chargement de {len(zh12)} lignes de ZH12 pour {len(keep_keys)} tickets Jira")

    # Agrégation ZH12 si non vide
    if not zh12.empty:
        zh12["Durée tâche (heures)"] = pd.to_numeric(zh12["Durée tâche (heures)"], errors="coerce")
        zh12["Date"] = pd.to_datetime(zh12["Date"], errors="coerce")
        aggr = zh12.groupby(["Matricule", "Jira"], as_index=False).agg({
            "Durée tâche (heures)": "sum",
            "Code Service": "first",
            "Date": "min"
        }).rename(columns={"Date": "Date commence"})
    else:
        aggr = pd.DataFrame(columns=["Matricule", "Jira", "Durée tâche (heures)", "Code Service", "Date commence"])

    # Liste des tickets à remplacer
    keys_to_replace = df_ticket["key"].unique().tolist()

    # 🧹 Supprimer les lignes existantes de aggr avec ces Jira
    aggr = aggr[~aggr["Jira"].isin(keys_to_replace)]

    # ➕ Ajouter les nouvelles lignes pour ces tickets
    rows_to_add = pd.DataFrame({
        "Matricule": [matricule] * len(keys_to_replace),
        "Jira": keys_to_replace,
        "Durée tâche (heures)": [0] * len(keys_to_replace),
        "Code Service": [None] * len(keys_to_replace),
        "Date commence": [date_commencement] * len(keys_to_replace)
    })

    # 🔁 Fusionner
    aggr = pd.concat([aggr, rows_to_add], ignore_index=True)

    # Merge avec les données Jira
    merged_df = aggr.merge(jira_df, left_on="Jira", right_on="key", how="left").copy()
    print(f"🔗 Fusion des données Jira et ZH12 : {len(merged_df)} lignes après fusion")

    merged_df["fields.created"] = pd.to_datetime(merged_df["fields.created"], errors="coerce", utc=True).dt.tz_convert(None)

    # Calcul du masque
    date_val = pd.to_datetime(date_commencement)
    keys_ticket = set(df_ticket["key"])
    mask = merged_df["key"].isin(keys_ticket)

    # Nettoyage des composants uniquement pour mask
    merged_df.loc[mask, "fields.components"] = merged_df.loc[mask, "fields.components"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    merged_df.loc[mask, "components"] = merged_df.loc[mask, "fields.components"].apply(
        lambda comps: " / ".join(
            [c["name"] for c in comps if isinstance(c, dict) and "name" in c]
        ) if isinstance(comps, list) else None
    )

    # Construction Type_Étendu uniquement pour mask
    colonnes_type = ["fields.issuetype.name", "components", "fields.customfield_10116.value"]
    merged_df.loc[mask, "Type_Étendu"] = merged_df.loc[mask].apply(
        lambda r: construire_type_etendu(r, colonnes_type), axis=1
    )
    
    merged_df.loc[mask, "Date commence"] = date_val
    merged_df["Date commence"] = pd.to_datetime(merged_df["Date commence"], errors="coerce", utc=True).dt.tz_convert(None)
    # Colonnes temporelles uniquement pour mask
    merged_df.loc[mask, "annee_creation"] = merged_df.loc[mask, "fields.created"].dt.year
    merged_df.loc[mask, "mois_creation"] = merged_df.loc[mask, "fields.created"].dt.month
    merged_df.loc[mask, "jour_semaine"] = merged_df.loc[mask, "fields.created"].dt.weekday
    merged_df.loc[mask, "delai_creation_action_h"] = (
        (merged_df.loc[mask, "Date commence"] - merged_df.loc[mask, "fields.created"]).dt.total_seconds() / 3600
    )

    merged_df.rename(columns={"Durée tâche (heures)": "Duree"}, inplace=True)

    # Calcul Historique uniquement sur les tickets initiaux
    merged_df["Historique_1an"] = 0
    merged_df[mask]["Historique_1an"] = calculer_historique_1an(merged_df[mask], merged_df)
    print(merged_df["Date commence"].head())
    return merged_df[mask]

def vectoriser_tfidf_merged_df(df: pd.DataFrame) -> pd.DataFrame:
    textes_concat = (df["summary"].fillna("") + " " + df["description"].fillna("")).str.strip() # + " " + df["commentaire_filtré_clean"].fillna("")
    tfidf_matrix = tfidf.transform(textes_concat)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    )
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

def convertir_type_sql(pandas_dtype):
    if pd.api.types.is_integer_dtype(pandas_dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(pandas_dtype):
        return "REAL"
    else:
        return "TEXT"

def importer_excel_dans_sqlite(excel_path, sqlite_path, table_name="zh12"):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["Matricule", "Date", "# de tâche"])

    # Construire le schéma SQL dynamiquement
    schema_sql = []
    for col in df.columns:
        sql_type = convertir_type_sql(df[col].dtype)
        schema_sql.append(f"[{col}] {sql_type}")
    schema_sql.append("PRIMARY KEY (Matricule, Date, [# de tâche])")

    create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  " + ",\n  ".join(schema_sql) + "\n)"

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute(create_stmt)
    conn.commit()

    # Supprimer les doublons existants
    for _, row in df.iterrows():
        cursor.execute(f"""
            DELETE FROM {table_name}
            WHERE Matricule = ? AND Date = ? AND [# de tâche] = ?
        """, (str(row["Matricule"]), str(row["Date"]), str(row["# de tâche"])))

    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return len(df)



def get_jira_comments(issue_key, auth, base_url="https://arvato-scs.atlassian.net"):
    comments = []
    start_at = 0
    max_results = 100

    while True:
        url = f"{base_url}/rest/api/latest/issue/{issue_key}/comment"
        params = {"startAt": start_at, "maxResults": max_results}
        response = requests.get(url, params=params, auth=auth)

        if response.status_code != 200:
            print(f"⚠️ Erreur pour {issue_key} : {response.status_code}")
            break

        data = response.json()
        page_comments = data.get("comments", [])

        if not page_comments:  # 👈 Aucun commentaire ? On quitte.
            break

        comments.extend(page_comments)

        if start_at + max_results >= data.get("total", 0):
            break

        start_at += max_results
        time.sleep(0.2)  # éviter d'abuser de l'API

    return comments

def nettoyer_commentaire(x):
    import re
    x = re.sub(r"https?://\S+", "lien", x)  # remplace les URL
    x = re.sub(r"\[\~accountid:[^\]]+\]", "", x)  # supprime les mentions [~accountid:...]
    return x


def extraire_commentaire_depuis_api(row, keep="before", debug=False):
    """
    keep:
        "before" -> commentaires avant 'Date commence' ou le premier du même jour
        "after"  -> commentaires après la date
        "all"    -> tous les commentaires
    """
    import pandas as pd, re
    auth = (JIRA_EMAIL, JIRA_TOKEN)
    issue_key = row.get("Jira")
    date_ref = row.get("Date commence")

    if debug:
        print(f"\n🔹 Ticket : {issue_key} | Date référence : {date_ref}")

    if not isinstance(issue_key, str) or (keep != "all" and pd.isna(date_ref)):
        if debug:
            print("⛔ Clé Jira manquante ou date_ref NaN")
        return ""

    commentaires = get_jira_comments(issue_key, auth=auth)
    if not commentaires:
        if debug:
            print("⚠️ Aucun commentaire récupéré.")
        return ""

    # ✅ Création du DataFrame
    df = pd.DataFrame(commentaires)

    # ✅ Conversion de la colonne 'updated' en datetime sans fuseau horaire
    df["updated"] = pd.to_datetime(df["updated"], errors="coerce", utc=True)
    df = df.dropna(subset=["updated"])
    df["updated"] = df["updated"].dt.tz_convert(None)

    # ✅ Séparation date / heure
    df["date_updated"] = df["updated"].dt.date
    df["heure_updated"] = df["updated"].dt.time

    if debug:
        print(f"→ {len(df)} commentaires valides")
        print(df[["date_updated", "heure_updated", "body"]].head())

    # ✅ Conversion de la date référence
    date_ref = pd.to_datetime(date_ref)
    date_ref_only = date_ref.date()

    # ✅ Filtrage logique
    if keep == "before":
        mask_before = df["date_updated"] < date_ref_only
        same_day = df["date_updated"] == date_ref_only
        first_same_day = same_day & (df["updated"] == df[same_day]["updated"].min())
        df = df[mask_before | first_same_day]
    elif keep == "after":
        df = df[df["updated"] >= date_ref]
    # sinon keep == "all" : pas de filtre

    if debug:
        print(f"✔️ {len(df)} commentaire(s) conservé(s) après filtre")

    # ✅ Nettoyage des textes
    textes = df["body"].fillna("").apply(nettoyer_commentaire).tolist()

    return " ".join(textes).strip()
