"""
app.py — FastAPI backend for RICAS-ML ICE Risk Prediction
──────────────────────────────────────────────────────────
Endpoints:
  GET  /              → serves index.html
  POST /api/predict   → returns risk probabilities + SHAP values for all models
  GET  /api/health    → health check

Run locally:
    uvicorn app:app --reload --port 8000

Deploy on Render / Railway:
    Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import json, pickle, warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RICAS-ML: Ischemic Cerebrovascular Event Risk Prediction",
    description=(
        "Predicts ischemic cerebrovascular event (ICE) risk in head & neck cancer "
        "patients undergoing radiotherapy, based on a 933-patient retrospective cohort."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
TEMPLATE    = BASE_DIR / "templates" / "index.html"

# ── Load artifacts at startup ──────────────────────────────────────────────────
with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
    META = json.load(f)

FEATURE_COLS = META["feature_cols"]
MODEL_FILES  = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
    "XGBoost":             "xgboost.pkl",
}
PIPELINES: Dict[str, object] = {}
for name, fname in MODEL_FILES.items():
    p = MODELS_DIR / fname
    if p.exists():
        with open(p, "rb") as f:
            PIPELINES[name] = pickle.load(f)
        print(f"  Loaded: {name}")
    else:
        print(f"  Skipped (not found): {name}")

print(f"Models loaded: {list(PIPELINES.keys())}")

# Pre-build SHAP TreeExplainers for tree-based models
EXPLAINERS: Dict[str, object] = {}
SHAP_MODELS = ["Random Forest", "Gradient Boosting", "XGBoost"]
for name in SHAP_MODELS:
    if name in PIPELINES:
        try:
            clf = PIPELINES[name]["clf"]
            EXPLAINERS[name] = shap.TreeExplainer(clf)
            print(f"  SHAP TreeExplainer ready: {name}")
        except Exception as e:
            print(f"  SHAP explainer skipped for {name}: {e}")

# Transformed feature names (after ColumnTransformer)
def get_transformed_feature_names(pipe):
    ct = pipe["prep"]
    names = []
    for trans_name, transformer, cols in ct.transformers_:
        if hasattr(cols, "__iter__") and not isinstance(cols, str):
            names.extend(cols)
    return names

TRANSFORMED_NAMES = {}
for name, pipe in PIPELINES.items():
    TRANSFORMED_NAMES[name] = get_transformed_feature_names(pipe)

# ── Input schema ──────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    # Demographics
    Age:   float = 60.0
    BMI:   float = 22.5
    sex_male: int = 1
    # Comorbidities
    htn:          int = 0
    dm:           int = 0
    dlp:          int = 0
    cad:          int = 0
    pad:          int = 0
    heart_disease: int = 0
    thyroid:      int = 0
    smoking:      int = 0
    # Medications
    antiplatelet:  int = 0
    anticoagulant: int = 0
    acei:          int = 0
    ccb:           int = 0
    statin:        int = 0
    # Cancer / RT
    hnc_stage:            float = 3.0
    rt_total_dose:        float = 66.0
    rt_dose_per_fraction: float = 2.0
    rt_fractions:         float = 33.0
    rt_definitive:        int   = 1
    rt_adjuvant:          int   = 0
    # Carotid imaging
    seen_carotid_pre_rt:  int   = 0
    sig_stenosis_pre_rt:  int   = 0
    max_stenosis_pre_rt:  float = 0.0
    total_plaque_pre_rt:  float = 0.0
    mean_imt_pre_rt:      float = 0.7
    seen_carotid_post_rt: int   = 0
    progression_post_rt:  int   = 0
    # Labs
    BUN:                float = 14.0
    Cr:                 float = 0.9
    GFR:                float = 85.0
    Total_cholesterol:  float = 180.0
    LDL:                float = 110.0
    Fasting_blood_sugar: float = 95.0

    class Config:
        populate_by_name = True

# ── Helper: build DataFrame from input ────────────────────────────────────────
def patient_to_df(data: PatientData) -> pd.DataFrame:
    row = {
        "Age":   data.Age,
        "BMI":   data.BMI,
        "sex_male": data.sex_male,
        "htn":   data.htn,
        "dm":    data.dm,
        "dlp":   data.dlp,
        "cad":   data.cad,
        "pad":   data.pad,
        "heart_disease": data.heart_disease,
        "thyroid":  data.thyroid,
        "smoking":  data.smoking,
        "antiplatelet":  data.antiplatelet,
        "anticoagulant": data.anticoagulant,
        "acei":   data.acei,
        "ccb":    data.ccb,
        "statin": data.statin,
        "hnc_stage":            data.hnc_stage,
        "rt_total_dose":        data.rt_total_dose,
        "rt_dose_per_fraction": data.rt_dose_per_fraction,
        "rt_fractions":         data.rt_fractions,
        "rt_definitive":        data.rt_definitive,
        "rt_adjuvant":          data.rt_adjuvant,
        "seen_carotid_pre_rt":  data.seen_carotid_pre_rt,
        "sig_stenosis_pre_rt":  data.sig_stenosis_pre_rt,
        "max_stenosis_pre_rt":  data.max_stenosis_pre_rt,
        "total_plaque_pre_rt":  data.total_plaque_pre_rt,
        "mean_imt_pre_rt":      data.mean_imt_pre_rt,
        "seen_carotid_post_rt": data.seen_carotid_post_rt,
        "progression_post_rt":  data.progression_post_rt,
        "BUN":   data.BUN,
        "Cr":    data.Cr,
        "GFR":   data.GFR,
        "Total cholesterol":   data.Total_cholesterol,
        "LDL":   data.LDL,
        "Fasting blood sugar": data.Fasting_blood_sugar,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(TEMPLATE)

@app.get("/api/health")
async def health():
    return {"status": "ok", "models": list(PIPELINES.keys())}

@app.post("/api/predict")
async def predict(data: PatientData):
    try:
        X = patient_to_df(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature construction error: {e}")

    results = {}
    shap_results = {}

    for name, pipe in PIPELINES.items():
        try:
            prob = float(pipe.predict_proba(X)[0, 1])
            results[name] = round(prob * 100, 2)   # percent
        except Exception as e:
            results[name] = None

        # SHAP for tree models
        if name in EXPLAINERS:
            try:
                X_t = pipe["prep"].transform(X)
                sv  = EXPLAINERS[name].shap_values(X_t)
                # sv may be (n, features) or list [(n, features), (n, features)]
                if isinstance(sv, list):
                    sv = sv[1]          # class=1 (ICE positive)
                if hasattr(sv, "values"):
                    sv = sv.values      # SHAP Explanation object
                sv = np.array(sv).flatten()
                feat_names = TRANSFORMED_NAMES.get(name, FEATURE_COLS)
                top_idx = np.argsort(np.abs(sv))[::-1][:15]
                shap_results[name] = {
                    "features": [feat_names[i] if i < len(feat_names) else f"f{i}"
                                 for i in top_idx],
                    "values":   [round(float(sv[i]), 5) for i in top_idx],
                }
            except Exception as e:
                shap_results[name] = {"error": str(e)}

    # Risk level based on average of available probabilities
    valid_probs = [v for v in results.values() if v is not None]
    avg_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0.0

    if avg_prob < 2:
        risk_level = "low"
        risk_en    = "Low Risk"
        risk_th    = "ความเสี่ยงต่ำ"
    elif avg_prob < 10:
        risk_level = "moderate"
        risk_en    = "Moderate Risk"
        risk_th    = "ความเสี่ยงปานกลาง"
    else:
        risk_level = "high"
        risk_en    = "High Risk"
        risk_th    = "ความเสี่ยงสูง"

    # Best SHAP model for display
    shap_display = None
    for preferred in ["Gradient Boosting", "Random Forest", "XGBoost"]:
        if preferred in shap_results and "features" in shap_results[preferred]:
            shap_display = {"model": preferred, **shap_results[preferred]}
            break

    return JSONResponse({
        "predictions":   results,
        "avg_prob_pct":  round(avg_prob, 2),
        "risk_level":    risk_level,
        "risk_en":       risk_en,
        "risk_th":       risk_th,
        "shap":          shap_display,
        "prevalence_pct": round(META["prevalence"] * 100, 2),
    })
