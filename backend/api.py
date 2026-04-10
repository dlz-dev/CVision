import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.analyzer import extract_cv
from core.features import cv_to_features
from core.preprocessor import clean_cv_text_for_llm, compute_experience_metrics, pre_process_cv

_BASE = Path(__file__).parents[1]

MODEL_PATH = _BASE / "models" / "model.pkl"

app = FastAPI(title="CVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)


def predict(cv: dict) -> str:
    X = pd.DataFrame([cv_to_features(cv)])
    return model.predict(X)[0]


@app.post("/process-cv")
async def process_cv(file: UploadFile = File(...)):
    cv_text = (await file.read()).decode("utf-8")

    llm_data = extract_cv(cv_text)
    pre_data = pre_process_cv(cv_text)
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

# uvicorn api.api:app --reload