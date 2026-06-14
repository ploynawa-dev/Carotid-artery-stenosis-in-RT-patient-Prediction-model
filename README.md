# RICAS-ML Web Application
## ICE Risk Prediction · Head & Neck Cancer Post-Radiotherapy

Clinical decision support tool for predicting ischemic cerebrovascular event (ICE) risk in
head and neck cancer patients following radiotherapy. Built on a retrospective cohort of
**933 patients** (ICE prevalence = 1.4%) from the RICAS study.

---

## Architecture

```
web_app/
├── app.py                    # FastAPI backend (API + static serving)
├── train_save_models.py      # Train models locally → save to models/
├── templates/index.html      # Bilingual TH/EN clinical form (single-page)
├── models/                   # Serialized ML pipelines (commit these!)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   └── metadata.json
├── requirements.txt
├── Dockerfile
├── render.yaml               # Render deployment config
└── railway.toml              # Railway deployment config
```

## Models included

| Model | OOF AUC | Brier Score | Notes |
|---|---|---|---|
| Logistic Regression | 0.216 | 0.140 | Overcompensates for class imbalance |
| Random Forest | 0.544 | 0.049 | Best discrimination |
| Gradient Boosting | 0.540 | 0.020 | Best calibration ✓ |
| XGBoost | 0.504 | 0.023 | Moderate |

> Note: Low AUCs reflect extreme class imbalance (1.4% ICE rate). LR AUC < 0.5 indicates
> probability inversion due to class_weight='balanced'. GB is recommended for clinical use.

---

## Step 1 — Train models locally

Run this once with access to the original patient CSV:

```bash
python train_save_models.py \
  --data /path/to/CarotidArteryStenosi_DATA_LABELS_2023-09-17_0233.csv \
  --out models/
```

This creates `models/*.pkl` and `models/metadata.json`. **Commit the `models/` directory.**
The raw patient CSV must NOT be committed (patient privacy).

---

## Step 2 — Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open: http://localhost:8000

---

## Step 3 — Deploy to Render (free tier)

1. Push code + `models/` to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect GitHub repo
4. Render will auto-detect `render.yaml` → Deploy

Or use the manual settings:
- **Environment**: Docker
- **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Health check**: `/api/health`

---

## Step 4 — Deploy to Railway

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up
```

Or connect via Railway dashboard: New Project → Deploy from GitHub Repo

---

## API Reference

### `GET /api/health`
```json
{"status": "ok", "models": ["Logistic Regression", "Random Forest", ...]}
```

### `POST /api/predict`
Accepts JSON with 35 patient features. Returns predictions from all models + SHAP values.

**Example request:**
```json
{
  "Age": 65, "BMI": 24, "sex_male": 1,
  "htn": 1, "dm": 0, "dlp": 1, "smoking": 1,
  "statin": 1, "hnc_stage": 3,
  "rt_total_dose": 66, "rt_dose_per_fraction": 2, "rt_fractions": 33,
  "rt_definitive": 1, "rt_adjuvant": 0,
  "max_stenosis_pre_rt": 0, "mean_imt_pre_rt": 0.8,
  ...
}
```

**Response:**
```json
{
  "predictions": {
    "Gradient Boosting": 0.53,
    "Random Forest": 35.0,
    ...
  },
  "avg_prob_pct": 31.03,
  "risk_level": "high",
  "risk_en": "High Risk",
  "risk_th": "ความเสี่ยงสูง",
  "shap": {
    "model": "Gradient Boosting",
    "features": ["Fasting blood sugar", "rt_dose_per_fraction", ...],
    "values": [0.998, -0.756, ...]
  },
  "prevalence_pct": 1.39
}
```

---

## ⚠️ Disclaimer

This system is for **academic research and demonstration only**. It does not constitute
medical advice and must not replace clinical judgment. All predictions must be interpreted
by a qualified clinician in the context of the individual patient's circumstances.

**Study**: RICAS (Radiation-Induced Carotid Artery Stenosis), Ramathibodi Hospital
**PI**: [Ploy Nawaphantaengsakul], MD
