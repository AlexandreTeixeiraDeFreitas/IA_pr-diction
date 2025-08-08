from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from module import get_jira_tickets_dataframe, get_filtered_jira_issues, traiter_historique_1an, preprocess_libelle, vectoriser_tfidf_merged_df, importer_excel_dans_sqlite, extraire_commentaire_depuis_api, sauvegarder_predictions
import joblib
import uuid
import os

# Chargement du modèle
encoder = joblib.load("xgb_encoder_nlp.pkl")
model = joblib.load("xgb_pipeline_nlp.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def index():

    return render_template("index.html")

@app.route("/tickets", methods=["GET", "POST"])
def tickets_form():
    predictions = None
    error = None
    matricule = ""
    date_commencement = ""
    ticket_list = []

    if request.method == "POST":
        ticket_list = request.form.getlist("tickets")
        matricule = request.form.get("matricule", "").strip()
        date_commencement = request.form.get("date_commencement", "").strip()

        try:
            if not ticket_list or not matricule or not date_commencement:
                error = "Tous les champs sont requis."
            else:
                df = get_jira_tickets_dataframe(ticket_list)
                df_all_ticket = get_filtered_jira_issues(df)
                df_all_ticket = pd.concat([df, df_all_ticket], ignore_index=True).drop_duplicates(subset="key")

                df = traiter_historique_1an(
                    df_ticket=df,
                    df_merged=df_all_ticket,
                    date_commencement=date_commencement,
                    matricule=matricule
                )

                df["nb_key"] = df["key"].str.extract(r"-(\d+)", expand=False).astype(int)
                df["fields.summary"] = df["fields.summary"].fillna("")
                df["summary"] = df["fields.summary"].apply(preprocess_libelle)
                df["fields.description"] = df["fields.description"].fillna("")
                df["description"] = df["fields.description"].apply(preprocess_libelle)
                # # ✅ Garde les commentaires AVANT la date
                # df["commentaire_filtré"] = df.apply(
                #     lambda r: extraire_commentaire_depuis_api(r, keep="before"), axis=1
                # )
                # # Nettoyage NLP
                # df["commentaire_filtré_clean"] = df["commentaire_filtré"].apply(preprocess_libelle)

                cat_cols = ["Matricule", "fields.project.key", "Type_Étendu"]
                df_prédict = df.copy()
                df_prédict[cat_cols] = encoder.transform(df_prédict[cat_cols])
                df_prédict = vectoriser_tfidf_merged_df(df_prédict)
                features = model.feature_names_in_
                df_input = df_prédict[features]
                df["predict"] = model.predict(df_input)
                predictions = df.to_dict(orient="records")

                now = datetime.now()
                sauvegarder_predictions(df, now)

        except Exception as e:
            error = str(e)

    return render_template("tickets.html", predictions=predictions, error=error, matricule=matricule, date_commencement=date_commencement, tickets=ticket_list)

@app.route("/api/tickets", methods=["GET", "POST"])
def tickets():
    if request.method == "POST":
        try:
            data = request.get_json(force=True)  # Force JSON même si Content-Type est mal défini

            ticket_list = data.get("tickets", [])
            if isinstance(ticket_list, str):
                ticket_list = ticket_list.split()

            matricule = data.get("matricule", "").strip()
            date_commencement = data.get("date_commencement", "").strip()

            if not ticket_list or not matricule or not date_commencement:
                return jsonify({"error": "Champs tickets, matricule et date_commencement requis."}), 400

            # 1. Récupération des tickets
            df = get_jira_tickets_dataframe(ticket_list)

            # 2. Recherche des tickets similaires
            df_all_ticket = get_filtered_jira_issues(df, "mongo")
            df_all_ticket = pd.concat([df, df_all_ticket], ignore_index=True).drop_duplicates(subset="key")

            # 3. Enrichissement avec ZH12
            df = traiter_historique_1an(
                df_ticket=df,
                df_merged=df_all_ticket,
                date_commencement=date_commencement,
                matricule=matricule
            )
            df["nb_key"] = df["key"].str.extract(r"-(\d+)", expand=False).astype(int)

            # 4. Prétraitement texte + TF-IDF
            df["fields.summary"] = df["fields.summary"].fillna("")
            df["summary"] = df["fields.summary"].apply(preprocess_libelle)
            df["fields.description"] = df["fields.description"].fillna("")
            df["description"] = df["fields.description"].apply(preprocess_libelle)
            
            # Colonnes catégorielles à encoder
            cat_cols = ["Matricule", "fields.project.key", "Type_Étendu"]

            df_prédict = df.copy()
            
            # Encodage
            df_prédict[cat_cols] = encoder.transform(df_prédict[cat_cols])

            # 5. Ajout des colonnes TF-IDF
            df_prédict = vectoriser_tfidf_merged_df(df_prédict)

            # 6. Sélection des colonnes d’entrée
            features = model.feature_names_in_
            df_input = df_prédict[features]

            # ✅ Nettoyage des NaN pour éviter erreur JSON
            df = df.fillna("")

            # 7. Prédiction
            df["predict"] = model.predict(df_input)

            now = datetime.now()
            sauvegarder_predictions(df, now)
            
            return jsonify(df.to_dict(orient="records"))

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Utilisez une requête POST avec un JSON contenant tickets, matricule et date_commencement."})

@app.route("/import", methods=["POST"])
def importer_excel():
    file = request.files.get("excel_file")
    if not file or file.filename == "":
        return "Aucun fichier sélectionné", 400

    try:
        filename = file.filename.lower()
        unique_name = f"import_{uuid.uuid4().hex}.csv"
        dest_path = os.path.join("../import", unique_name)

        if filename.endswith(".csv"):
            # 📄 Déjà un CSV → on copie directement
            file.save(dest_path)
        elif filename.endswith(".xlsx"):
            # 📄 XLSX → on convertit avec pandas
            df = pd.read_excel(file)
            df.to_csv(dest_path, index=False)
        else:
            return "❌ Format non supporté (.csv ou .xlsx uniquement)", 400

        return f"✅ Fichier prêt pour traitement Spark : {unique_name}<br><a href='/tickets'>Retour</a>"

    except Exception as e:
        return f"❌ Erreur pendant l'import : {e}", 500
    

@app.route("/api/import", methods=["POST"])
def importer():
    file = request.files.get("excel_file")
    if not file or file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    try:
        filename = file.filename.lower()
        unique_name = f"import_{uuid.uuid4().hex}.csv"
        dest_path = os.path.join("../import", unique_name)

        if filename.endswith(".csv"):
            # Déjà un CSV → on copie directement
            file.save(dest_path)
        elif filename.endswith(".xlsx"):
            # XLSX → on convertit avec pandas
            df = pd.read_excel(file)
            df.to_csv(dest_path, index=False)
        else:
            return jsonify({"error": "❌ Format non supporté (.csv ou .xlsx uniquement)"}), 400

        return jsonify({
            "message": "✅ Fichier prêt pour traitement Spark",
            "filename": unique_name
        })

    except Exception as e:
        return jsonify({"error": f"❌ Erreur pendant l'import : {str(e)}"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    # from waitress import serve  # ou autre serveur WSGI
    # serve(app, host="127.0.0.1", port=5000)
