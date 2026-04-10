# fichier qui lit les JSONs de data/processed et les transforme en tableau de chiffre pour sklearn

import json                 # pour lire les fichiers JSON
import pandas as pd         # pour créer un tableau de données (DataFrame)
from pathlib import Path    # pour gérer les chemins de fichiers


def cv_to_features(cv: dict) -> dict:
    """Extrait les features numériques d'un CV structuré."""
    return {
        "age":                    cv.get("age") or 0,
        "total_experience_years": cv.get("total_experience_years") or 0,
        "years_since_graduation": (
            cv.get("years_since_graduation")
            or cv.get("education", {}).get("years_since_graduation")
            or 0
        ),
        "nb_skills":         len(cv.get("skills") or []),
        "nb_languages":      len(cv.get("languages") or []),
        "nb_certifications": len(cv.get("certifications") or []),
        "nb_experiences":    len(cv.get("experiences") or []),
        "nb_gaps":           len(cv.get("experience_gaps_months") or []),
    }


def load_features(processed_folder: str, labels_path: str) -> tuple:
    """
    Lit les JSONs extraits et les labels, et retourne X (features) et y (labels).
    """

    labels_df = pd.read_csv(labels_path)
    labels_dict = dict(zip(labels_df["filename"], labels_df["label"]))

    rows = []

    for json_file in Path(processed_folder).glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            cv = json.load(f)

        txt_name = json_file.stem + ".txt"
        if txt_name not in labels_dict:
            continue

        row = {"filename": txt_name, **cv_to_features(cv), "label": labels_dict[txt_name]}
        rows.append(row)

    # crée un DataFrame à partir de toutes les lignes
    df = pd.DataFrame(rows)

    # X = les colonnes de features (tout sauf filename et label)
    X = df.drop(columns=["filename", "label"])

    # y = la colonne cible (Inviter / Rejeter)
    y = df["label"]

    return X, y  # retourne les features et les labels séparément