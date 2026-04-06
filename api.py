import yaml
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from core.analyzer import extract_cv

# ATTENTION CECI EST UN TEST AVEC FAST API -> TRANSFORMER LE CODE PYTHON EN UN SERV WEB POUR COMMUNIQUER AVEC N8N

# création de l'app FastAPI
app = FastAPI()

# chargement de la config comme d'habitude
with open("config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# chargement du modèle ML une seule fois au démarrage
model = joblib.load("models/model.pkl")

def predict(cv: dict) -> str:
    # transforme le CV en ligne de features (même logique que features.py)
    row = {
        "age": cv.get("age") or 0,
        "total_experience_years": cv.get("total_experience_years") or 0,
        "years_since_graduation": cv.get("education", {}).get("years_since_graduation") or 0,
        "nb_skills": len(cv.get("skills") or []),
        "nb_languages": len(cv.get("languages") or []),
        "nb_certifications": len(cv.get("certifications") or []),
        "nb_experiences": len(cv.get("experiences") or []),
        "nb_gaps": len(cv.get("experience_gaps_months") or []),
    }

    # crée un DataFrame avec une seule ligne
    X = pd.DataFrame([row])

    # retourne la prédiction : "Inviter" ou "Rejeter"
    return model.predict(X)[0]

# endpoint qui reçoit le fichier .txt et retourne un json extrait
@app.post("/process-cv")
async def process_cv(file: UploadFile = File(...)):

    # lit le contenu du fichier reçu
    cv_text = (await file.read()).decode("utf-8")

    # extrait les infos du CV via le LLM
    result = extract_cv(cv_text, config)

    # ajoute la décision du modèle ML au résultat
    result["decision"] = predict(result)

    # retourne le JSON
    return result

# uvicorn api:app --reload