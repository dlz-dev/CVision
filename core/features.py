# fichier qui lit les JSONs de data/processed et les transforme en tableau de chiffre pour sklearn

import json                 # pour lire les fichiers JSON
import pandas as pd         # pour créer un tableau de données (DataFrame)
from pathlib import Path    # pour gérer les chemins de fichiers


def load_features(processed_folder: str, labels_path: str) -> tuple:
    """
    Lit les JSONs extraits et les labels, et retourne X (features) et y (labels).
    """

    # charge le fichier labels.csv dans un DataFrame
    labels_df = pd.read_csv(labels_path)

    # crée un dictionnaire filename -> label pour accès rapide
    # ex: {"cv1.txt": "Inviter", "cv2.txt": "Rejeter"}
    labels_dict = dict(zip(labels_df["filename"], labels_df["label"]))

    rows = []  # liste qui contiendra une ligne de features par CV

    # parcourt tous les fichiers JSON dans data/processed/
    for json_file in Path(processed_folder).glob("*.json"):

        # charge le contenu du JSON
        with open(json_file, encoding="utf-8") as f:
            cv = json.load(f)

        # nom du fichier .txt correspondant (ex: cv1.json -> cv1.txt)
        txt_name = json_file.stem + ".txt"

        # ignore ce CV si on n'a pas de label pour lui
        if txt_name not in labels_dict:
            continue

        # extrait les features numériques du CV
        row = {
            "filename": txt_name,

            # âge du candidat
            "age": cv.get("age") or 0,

            # années d'expérience totales
            "total_experience_years": cv.get("total_experience_years") or 0,

            # nombre d'années depuis le diplôme
            "years_since_graduation": cv.get("education", {}).get("years_since_graduation") or 0,

            # nombre de compétences listées
            "nb_skills": len(cv.get("skills") or []),

            # nombre de langues parlées
            "nb_languages": len(cv.get("languages") or []),

            # nombre de certifications
            "nb_certifications": len(cv.get("certifications") or []),

            # nombre de postes occupés
            "nb_experiences": len(cv.get("experiences") or []),

            # nombre total de gaps (périodes de chômage)
            "nb_gaps": len(cv.get("experience_gaps_months") or []),

            # label : Inviter ou Rejeter
            "label": labels_dict[txt_name],
        }

        rows.append(row)  # ajoute la ligne à la liste

    # crée un DataFrame à partir de toutes les lignes
    df = pd.DataFrame(rows)

    # X = les colonnes de features (tout sauf filename et label)
    X = df.drop(columns=["filename", "label"])

    # y = la colonne cible (Inviter / Rejeter)
    y = df["label"]

    return X, y  # retourne les features et les labels séparément