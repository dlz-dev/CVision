import yaml
import json
from pathlib import Path
from datetime import datetime
from core.loader import load_cvs_from_folder
from core.analyzer import extract_cv
from core.preprocessor import pre_process_cv, clean_cv_text_for_llm, compute_experience_metrics, score_education
from core.json2csv import json2csv

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

all_cvs = load_cvs_from_folder("data/raw")
Path("data/extracted").mkdir(parents=True, exist_ok=True)

for filename, cv_text in all_cvs.items():
    print(f"Traitement de {filename}...")

    # 1. Extraction Python classique
    pre_processed_data = pre_process_cv(cv_text)

    # 2. Nettoyage du texte pour le LLM (économie de tokens)
    optimized_cv_text = clean_cv_text_for_llm(cv_text)

    # 3. Requête LLM allégée
    try:
        result_llm = extract_cv(optimized_cv_text, config)
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON pour {filename}, CV ignoré.")
        continue
    except Exception as e:
        print(f"Erreur inattendue pour {filename} : {e}")
        continue

    # 4. Calculs mathématiques basés sur le retour du LLM
    metrics = compute_experience_metrics(result_llm.get("experiences", []))

    # 5. Fusion de l'éducation (LLM + Python)
    education_data = result_llm.get("education") or {}
    if education_data:
        education_data["graduation_year"]        = pre_processed_data.get("graduation_year")
        education_data["years_since_graduation"] = pre_processed_data.get("years_since_graduation")
        education_data["education_score"]        = score_education(education_data.get("degree"))

    # 6. Assemblage final en JSON
    cv_id = filename.replace(".txt", "")
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

    output_path = Path("data/extracted") / f"{cv_id}.json"
    output_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fichier {output_path} sauvegardé !\n")

# 7. Transformation JSON → CSV
json2csv(input_path_json="data/extracted", output_path_csv="data/cv_dataset.csv")

print("Tous les CVs ont été traités !")