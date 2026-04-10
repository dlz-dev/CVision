import yaml
import joblib
from fastapi import FastAPI, File, UploadFile
from core.analyzer import extract_cv
from core.features import cv_to_features
from core.preprocessor import pre_process_cv, compute_experience_metrics

app = FastAPI()

with open("config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model = joblib.load("models/model.pkl")


def predict(cv: dict) -> str:
    X = pd.DataFrame([cv_to_features(cv)])
    return model.predict(X)[0]


@app.post("/process-cv")
async def process_cv(file: UploadFile = File(...)):
    cv_text = (await file.read()).decode("utf-8")

    # LLM : extrait éducation + expériences avec les dates
    llm_data = extract_cv(cv_text, config)

    # Règles : extrait âge, compétences, langues, certifications, années depuis diplôme
    pre_data = pre_process_cv(cv_text)

    # Calcule total_experience_years et gaps depuis les dates retournées par le LLM
    exp_metrics = compute_experience_metrics(llm_data.get("experiences", []))

    result = {
        **pre_data,
        "education":              llm_data.get("education", {}),
        "experiences":            exp_metrics["experiences"],
        "total_experience_years": exp_metrics["total_experience_years"],
        "experience_gaps_months": exp_metrics["experience_gaps_months"],
    }

    result["decision"] = predict(result)
    return result

# uvicorn api:app --reload