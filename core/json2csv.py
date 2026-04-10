import json
from pathlib import Path

import pandas as pd


def json2csv(input_path_json: str) -> pd.DataFrame:
    """Aplatit et compile tous les fichiers JSON d'un dossier vers un DataFrame pandas."""
    files = Path(input_path_json).glob("*.json")

    mappings = {
        "lang_fr": ["french", "français", "francais"],
        "lang_en": ["english", "anglais"],
        "lang_de": ["german", "allemand"],
        "lang_es": ["spanish", "espagnol"],
        "lang_it": ["italian", "italien"]
    }

    # Création d'une map inversée { "français": "lang_fr", ... } pour accès O(1)
    reverse_mapping = {
        syn: main_col for main_col, synonyms in mappings.items() for syn in synonyms
    }

    base_lang_dict = {col: 0 for col in mappings}
    base_lang_dict["lang_other_score_sum"] = 0

    data = []

    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            cv = json.load(file)

        meta = cv.get("meta", {})
        education = cv.get("education", {})
        gaps = cv.get("experience_gaps_months", [])

        row = {
            "cv_id": meta.get("cv_id"),
            "age": cv.get("age"),
            "distance_ville_haute_km": cv.get("distance_ville_haute_km"),
            "target_role": cv.get("target_role"),
            "total_experience_years": cv.get("total_experience_years"),
            "total_gap_months": sum(g.get("duration_months", 0) for g in gaps),
            "nb_gaps": len(gaps),
            "education_degree": education.get("degree"),
            "education_field": education.get("field"),
            "education_school": education.get("school"),
            "education_score": education.get("education_score"),
            "skills": ", ".join(cv.get("skills", [])),
            "certifications": ", ".join(c.get('name', '') for c in cv.get("certifications", [])),
            "number_of_experiences": len(cv.get("experiences", [])),
            **base_lang_dict.copy()
        }

        # Traitement des compétences linguistiques
        for lang in cv.get("languages", []):
            lang_name = lang.get("language", "").strip().lower()
            if not lang_name:
                continue

            score = lang.get("score") or 1
            col_name = reverse_mapping.get(lang_name)

            if col_name:
                row[col_name] = max(row[col_name], score)
            else:
                row["lang_other_score_sum"] += score

        data.append(row)

    df = pd.DataFrame(data)

    # Conversion de types optimisée en lot
    cols_to_int = list(mappings.keys()) + ["lang_other_score_sum", "total_gap_months", "nb_gaps"]
    df[cols_to_int] = df[cols_to_int].fillna(0).astype(int)

    print(f"DataFrame créé avec succès ({len(data)} entrées)")
    return df


if __name__ == "__main__":
    INPUT_FOLDER = "../data/extracted"
    OUTPUT_FILE = "../data/cv_dataset.csv"
    LABELS_FILE = "../data/student_labels.csv"

    print(f"Lancement de json2csv en mode standalone sur : {INPUT_FOLDER}...")

    if Path(INPUT_FOLDER).exists():
        df_result = json2csv(INPUT_FOLDER)

        labels_path = Path(LABELS_FILE)
        if labels_path.exists():
            print("Fusion avec les labels en cours...")
            # Chargement et filtrage optimisés en une ligne
            df_labels = pd.read_csv(labels_path, usecols=['filename', 'passed_next_stage']) \
                .rename(columns={'filename': 'cv_id'})
            df_result = pd.merge(df_result, df_labels, on="cv_id", how="left")
        else:
            print(f"Avertissement : Fichier '{LABELS_FILE}' introuvable. Fusion ignorée.")

        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\nAperçu:\n{df_result.head()}\nFichier sauvegardé : {OUTPUT_FILE}")
    else:
        print(f"\nErreur : Le dossier '{INPUT_FOLDER}' n'existe pas.")