import json
import pandas as pd
from pathlib import Path


def json2csv(input_path_json: str) -> pd.DataFrame:
    input_dir = Path(input_path_json)
    files = input_dir.glob("*.json")

    mappings = {
        "lang_fr": ["french", "français", "francais"],
        "lang_en": ["english", "anglais"],
        "lang_de": ["german", "allemand"],
        "lang_lu": ["luxembourgish", "luxembourgeois", "letzebuergesch"],
        "lang_es": ["spanish", "espagnol"],
        "lang_it": ["italian", "italien"]
    }

    reverse_mapping = {
        synonym: main_col
        for main_col, synonyms in mappings.items()
        for synonym in synonyms
    }

    lang_cols = list(mappings.keys())
    base_lang_dict = {col: 0 for col in lang_cols}
    base_lang_dict["lang_other_score_sum"] = 0

    data = []

    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            cv = json.load(file)

            meta = cv.get("meta", {})
            education = cv.get("education", {})

            row = {
                "cv_id": meta.get("cv_id"),
                "age": cv.get("age"),
                "distance_ville_haute_km": cv.get("distance_ville_haute_km"),
                "target_role": cv.get("target_role"),
                "total_experience_years": cv.get("total_experience_years"),
                "education_degree": education.get("degree"),
                "education_field": education.get("field"),
                "education_school": education.get("school"),
                "education_score": education.get("education_score"),
                "skills": ", ".join(cv.get("skills", [])),
                "certifications": ", ".join([c.get('name', '') for c in cv.get("certifications", [])]),
                "number_of_experiences": len(cv.get("experiences", []))
            }

            row.update(base_lang_dict)

            for lang in cv.get("languages", []):
                lang_name = lang.get("language", "")
                if not lang_name:
                    continue

                lang_name = lang_name.strip().lower()
                score = lang.get("score")
                score = score if score is not None else 1

                col_name = reverse_mapping.get(lang_name)

                if col_name:
                    if score > row[col_name]:
                        row[col_name] = score
                else:
                    row["lang_other_score_sum"] += score

            data.append(row)

    df = pd.DataFrame(data)

    cols_to_int = lang_cols + ["lang_other_score_sum"]
    df[cols_to_int] = df[cols_to_int].astype(int)

    print(f"DataFrame créé avec succès ({len(data)} entrées)")

    # On retourne le DataFrame au lieu de le sauvegarder
    return df