import requests
import json
import os

def collect_jira_issues(projects, date_start, date_end, max_results=100, auth=None, output_dir="jira_exports", final_file="jira_merged_issues.json"):
    """
    Récupère tous les tickets JIRA (API REST) pour une ou plusieurs clés projet,
    paginés, puis fusionne toutes les issues dans un seul fichier JSON.
    """

    if auth is None:
        raise ValueError("auth=(username, token) est requis")

    os.makedirs(output_dir, exist_ok=True)
    all_issues = []

    for project in projects:
        print(f"🔍 Récupération pour projet : {project}")
        start_at = 0
        index = 0

        while True:
            jql = f"created >= {date_start} AND created <= {date_end} AND project={project}"
            url = "https://arvato-scs.atlassian.net/rest/api/latest/search"
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results
            }

            response = requests.get(url, params=params, auth=auth)
            if response.status_code != 200:
                print(f"⚠️ Erreur {response.status_code} pour {project} : {response.text}")
                break

            data = response.json()
            issues = data.get("issues", [])
            total = data.get("total", 0)

            if not issues:
                print(f"✅ Fin (aucun ticket) à startAt={start_at}")
                break

            # all_issues.extend(issues)  # ajouter dans la liste globale

            # Optionnel : sauvegarde temporaire par page (debug ou archives)
            filename = os.path.join(output_dir, f"{project}_{index:04d}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Page {index} sauvegardée : {filename}")

            start_at += max_results
            index += 1

            if start_at >= total:
                print(f"✅ Tous les tickets récupérés pour {project} ({total} au total).")
                break

    # # Sauvegarde finale fusionnée
    # merged_data = {"issues": all_issues}
    # merged_path = os.path.join(final_file)
    # with open(merged_path, "w", encoding="utf-8") as f:
    #     json.dump(merged_data, f, indent=2, ensure_ascii=False)

    # print(f"\n✅ Fichier fusionné créé : {merged_path} ({len(all_issues)} tickets au total)")


def merge_jira_pages(input_dir="jira_pages", output_file="jira_merged_issues.json"):
    all_issues = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".json"):
            path = os.path.join(input_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                issues = data.get("issues", [])
                all_issues.extend(issues)

    print(f"🧩 Total fusionné : {len(all_issues)} tickets")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"issues": all_issues}, f, indent=2, ensure_ascii=False)

    print(f"✅ Fichier fusionné : {output_file}")


collect_jira_issues(
    # projects=["ULIS", "SCMF"],
    projects=["SCMHDF", "CMH"],
    date_start="2017-01-01",
    date_end="2025-07-08",
    max_results=100,
    auth=("xxxxxxx.xxxxxxxx@xxxxxxxxx.com", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
)

merge_jira_pages(input_dir="jira_exports", output_file="jira_merged.json")
