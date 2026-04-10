import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.analyzer import extract_cv
from core.json2csv import json2csv
from core.loader import load_cvs_from_folder
from core.preprocessor import clean_cv_text_for_llm, compute_experience_metrics, pre_process_cv, score_education

# Charge .env depuis la racine du projet (scripts/ → backend/ → racine/)
load_dotenv(Path(__file__).parents[2] / ".env")

_BASE = Path(__file__).parents[1]

RAW_DATA   = _BASE / "data" / "raw"
EXTRACTED  = _BASE / "data" / "extracted"
QUARANTINE = _BASE / "data" / "quarantaine"
LABELS     = _BASE / "data" / "student_labels.csv"
OUTPUT_CSV = _BASE / "data" / "cv_dataset.csv"

# Point de reprise (None pour traiter entièrement)
RESUME_FROM = None

EXTRACTED.mkdir(parents=True, exist_ok=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

all_cvs = load_cvs_from_folder(str(RAW_DATA))

if RESUME_FROM:
    keys = sorted(all_cvs.keys())
    resume_key = next((k for k in keys if k.startswith(RESUME_FROM)), None)
    if resume_key:
        idx = keys.index(resume_key)
        all_cvs = {k: all_cvs[k] for k in keys[idx:]}
        print(f"Reprise à partir de {resume_key} ({len(all_cvs)} fichiers restants)")
    else:
        print(f"Avertissement : '{RESUME_FROM}' introuvable, traitement complet.")

for filename, cv_text in all_cvs.items():
    print(f"Traitement de {filename}...")

    try:
        pre_processed_data = pre_process_cv(cv_text)
        optimized_cv_text = clean_cv_text_for_llm(cv_text)

        result_llm = extract_cv(optimized_cv_text)

        metrics = compute_experience_metrics(result_llm.get("experiences", []))
        education_data = result_llm.get("education") or {}

        if education_data:
            education_data.update({
                "graduation_year":        pre_processed_data.get("graduation_year"),
                "years_since_graduation": pre_processed_data.get("years_since_graduation"),
                "education_score":        score_education(education_data.get("degree"))
            })

        cv_id = filename.split(".")[0]
        final_json = {
            "meta": {
                "cv_id":        cv_id,
                "processed_at": datetime.now().strftime("%Y-%m-%d"),
            },
            "age":                     pre_processed_data.get("age"),
            "distance_ville_haute_km": pre_processed_data.get("distance_ville_haute_km"),
            "target_role":             pre_processed_data.get("target_role"),
            "education":               education_data,
            "experiences":             metrics["experiences"],
            "total_experience_years":  metrics["total_experience_years"],
            "experience_gaps_months":  metrics["experience_gaps_months"],
            "skills":                  pre_processed_data.get("skills"),
            "languages":               pre_processed_data.get("languages"),
            "certifications":          pre_processed_data.get("certifications"),
        }

        output_path = EXTRACTED / f"{cv_id}.json"
        output_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Succès -> {output_path} sauvegardé !\n")

    except Exception as e:
        print(f"Erreur inattendue pour {filename} : {e}. Déplacement en quarantaine.")
        (QUARANTINE / filename).write_text(cv_text, encoding="utf-8")
        continue

print("Conversion des JSON vers DataFrame...")
df_cvs = json2csv(input_path_json=str(EXTRACTED))

print("Fusion avec les labels étudiants...")
try:
    df_labels = pd.read_csv(LABELS, usecols=["filename", "passed_next_stage"]) \
        .rename(columns={"filename": "cv_id"})
    df_final = pd.merge(df_cvs, df_labels, on="cv_id", how="left")
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Pipeline achevé. Fichier final '{OUTPUT_CSV}' généré.")
except Exception as e:
    print(f"Erreur critique lors de la fusion finale : {e}")