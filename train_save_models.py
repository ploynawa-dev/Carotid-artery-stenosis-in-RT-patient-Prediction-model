"""
train_save_models.py
────────────────────
Run this script ONCE locally with the original CSV to train all four models
and save them (+ metadata) to the models/ directory.

Usage:
    python train_save_models.py \
        --data /path/to/CarotidArteryStenosi_DATA_LABELS_2023-09-17_0233.csv

The models/ directory must then be committed to git before deploying to
Render / Railway.
"""

import argparse, json, pickle, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available; skipping XGBClassifier.")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=(
        "/sessions/sharp-peaceful-johnson/mnt/uploads/"
        "CarotidArteryStenosi_DATA_LABELS_2023-09-17_0233.csv"
    ),
    help="Path to the raw CSV file",
)
parser.add_argument(
    "--out", default=str(Path(__file__).parent / "models"),
    help="Directory to save model artifacts",
)
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ── Feature engineering (identical to stroke_prediction_ml.py) ────────────────
df = pd.read_csv(args.data, encoding="utf-8-sig")
df["ICE"] = (df["ischemic cerebrovascular event (ICE)"] == "Yes").astype(int)

demo_feats = ["Age", "BMI"]
df["sex_male"] = (df["Sex (choice=Male)"] == "Checked").astype(int)

comorbidity_map = {
    "htn":           "Underlying disease (choice=Hypertension)",
    "dm":            "Underlying disease (choice=DM)",
    "dlp":           "Underlying disease (choice=DLP)",
    "cad":           "Underlying disease (choice=CAD)",
    "pad":           "Underlying disease (choice=PAD)",
    "heart_disease": "Underlying disease (choice=Heart disease)",
    "thyroid":       "Underlying disease (choice=Thyroid disease)",
}
for feat, col in comorbidity_map.items():
    df[feat] = (df[col] == "Checked").astype(int)

df["smoking"] = (df["Smoking"] == "Yes").astype(int)

med_map = {
    "antiplatelet":  "Antiplatelet",
    "anticoagulant": "Anticoagulant",
    "acei":          "ACEI use",
    "ccb":           "CCB",
    "statin":        "Statin",
}
for feat, col in med_map.items():
    df[feat] = (df[col] == "Yes").astype(int)

stage_map = {"In situ": 0, "stage 1": 1, "stage 2": 2, "stage 3": 3, "stage 4": 4}
df["hnc_stage"] = df["Staging of HNC"].map(stage_map).fillna(-1)

df["rt_total_dose"]        = pd.to_numeric(df["ปริมาณการฉายแสงทั้งหมดในการรักษา"], errors="coerce")
df["rt_dose_per_fraction"] = pd.to_numeric(df["ปริมาณการฉายแสงต่อครั้ง"],           errors="coerce")
df["rt_fractions"]         = pd.to_numeric(df["จำนวนครั้ง (1st course)"],              errors="coerce")
df["rt_definitive"] = (df["Aim of radiation (choice=Definitive radiation)"] == "Checked").astype(int)
df["rt_adjuvant"]   = (df["Aim of radiation (choice=Adjuvant radiation)"]   == "Checked").astype(int)

df["seen_carotid_pre_rt"] = (df["Seen Carotid lesion before RT"] == "Yes").astype(int)
df["sig_stenosis_pre_rt"] = (df["Significant carotid stenosis (>50%) befrore RT"] == "Yes").astype(int)

stenosis_pre_cols = [
    "% stenosis of proximal CCA (Rt)", "% stenosis of proximal CCA (Lt)",
    "% stenosis of distal CCA (Rt)",   "% stenosis of distal CCA (Lt)",
    "% stenosis of carotid bulb (Rt)", "% stenosis of carotid bulb (Lt)",
    "% stenosis of ICA (Rt)",          "% stenosis of ICA (Lt)",
    "% stenosis of ECA (Rt)",          "% stenosis of ECA (Lt)",
    "NASCET % stenosis (Rt)",          "NASCET % stenosis (Lt)",
]
for c in stenosis_pre_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["max_stenosis_pre_rt"] = df[stenosis_pre_cols].max(axis=1)
df["total_plaque_pre_rt"] = (
    pd.to_numeric(df["Total plaque score (Rt)"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Total plaque score (Lt)"], errors="coerce").fillna(0)
)
imt_cols = [
    "intima-media thickness of CCA (Rt)",   "intima-media thickness of CCA (Lt)",
    "Intima-media thickness of bulb (Rt)",  "Intima-media thickness of bulb (Lt)",
    "Intimal-media thickness of ICA (Rt)",  "Intimal-media thickness of ICA (Lt)",
]
for c in imt_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["mean_imt_pre_rt"] = df[imt_cols].mean(axis=1)

df["seen_carotid_post_rt"] = (df["Seen carotid lesion after RT"] == "Yes").astype(int)
df["progression_post_rt"]  = (df["Progession to significant lesion after RT?"] == "Yes").astype(int)

lab_feats = ["BUN", "Cr", "GFR", "Total cholesterol", "LDL", "Fasting blood sugar"]
for col in lab_feats:
    df[col] = pd.to_numeric(df[col], errors="coerce")

FEATURE_COLS = (
    demo_feats + ["sex_male"] + list(comorbidity_map.keys()) + ["smoking"]
    + list(med_map.keys()) + ["hnc_stage"]
    + ["rt_total_dose", "rt_dose_per_fraction", "rt_fractions", "rt_definitive", "rt_adjuvant"]
    + ["seen_carotid_pre_rt", "sig_stenosis_pre_rt", "max_stenosis_pre_rt",
       "total_plaque_pre_rt", "mean_imt_pre_rt"]
    + ["seen_carotid_post_rt", "progression_post_rt"]
    + lab_feats
)

X = df[FEATURE_COLS].copy()
y = df["ICE"].values

# ── Preprocessor ──────────────────────────────────────────────────────────────
num_cols  = [c for c in FEATURE_COLS if X[c].dtype in [np.float64, np.int64, float, int]]
bool_cols = [c for c in FEATURE_COLS if c not in num_cols]

preprocessor = ColumnTransformer([
    ("num",  Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("scl", StandardScaler())]), num_cols),
    ("bool", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), bool_cols),
])

# ── Models ────────────────────────────────────────────────────────────────────
scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))

model_defs = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=42, C=0.1))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf",  RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         max_depth=4, min_samples_leaf=5, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("clf",  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                             max_depth=3, subsample=0.8, random_state=42))
    ]),
}
if HAS_XGB:
    model_defs["XGBoost"] = Pipeline([
        ("prep", preprocessor),
        ("clf",  XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                               scale_pos_weight=scale_pos, eval_metric="logloss",
                               random_state=42, use_label_encoder=False))
    ])

# ── Train full models + compute OOF AUC ───────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary = {}

print(f"\nTraining {len(model_defs)} models on {len(y)} patients "
      f"({y.sum()} ICE events = {y.mean()*100:.1f}%)...\n")

for name, pipe in model_defs.items():
    # OOF AUC
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    # Fit on full training set
    pipe.fit(X, y)
    # Save pipeline
    fname = name.lower().replace(" ", "_") + ".pkl"
    with open(OUT / fname, "wb") as f:
        pickle.dump(pipe, f)
    summary[name] = {"auc_oof": round(auc, 4), "file": fname}
    print(f"  {name:25s}  OOF AUC={auc:.4f}  →  {fname}")

# ── Save metadata ──────────────────────────────────────────────────────────────
# Compute median values for SHAP background / default form values
medians = {}
for col in FEATURE_COLS:
    v = X[col].median()
    medians[col] = float(v) if not np.isnan(v) else 0.0

# Feature display names (bilingual)
feature_labels = {
    "Age":                  {"en": "Age (years)", "th": "อายุ (ปี)"},
    "BMI":                  {"en": "BMI (kg/m²)", "th": "ดัชนีมวลกาย"},
    "sex_male":             {"en": "Sex (Male=1)", "th": "เพศชาย"},
    "htn":                  {"en": "Hypertension", "th": "ความดันโลหิตสูง"},
    "dm":                   {"en": "Diabetes", "th": "เบาหวาน"},
    "dlp":                  {"en": "Dyslipidemia", "th": "ไขมันในเลือดสูง"},
    "cad":                  {"en": "Coronary Artery Disease", "th": "โรคหลอดเลือดหัวใจ"},
    "pad":                  {"en": "Peripheral Artery Disease", "th": "โรคหลอดเลือดส่วนปลาย"},
    "heart_disease":        {"en": "Heart Disease", "th": "โรคหัวใจ"},
    "thyroid":              {"en": "Thyroid Disease", "th": "โรคไทรอยด์"},
    "smoking":              {"en": "Smoking", "th": "สูบบุหรี่"},
    "antiplatelet":         {"en": "Antiplatelet", "th": "ยาต้านเกล็ดเลือด"},
    "anticoagulant":        {"en": "Anticoagulant", "th": "ยาต้านการแข็งตัวของเลือด"},
    "acei":                 {"en": "ACEI", "th": "ยา ACEI"},
    "ccb":                  {"en": "CCB", "th": "ยา CCB"},
    "statin":               {"en": "Statin", "th": "ยาสแตติน"},
    "hnc_stage":            {"en": "HNC Stage (0=In situ, 4=Stage IV)", "th": "ระยะมะเร็งศีรษะ-คอ"},
    "rt_total_dose":        {"en": "RT Total Dose (Gy)", "th": "ปริมาณรังสีรวม (Gy)"},
    "rt_dose_per_fraction": {"en": "RT Dose/Fraction (Gy)", "th": "ปริมาณรังสีต่อครั้ง (Gy)"},
    "rt_fractions":         {"en": "Number of RT Fractions", "th": "จำนวนครั้งของการฉายแสง"},
    "rt_definitive":        {"en": "Definitive RT", "th": "การฉายแสงแบบ Definitive"},
    "rt_adjuvant":          {"en": "Adjuvant RT", "th": "การฉายแสงแบบ Adjuvant"},
    "seen_carotid_pre_rt":  {"en": "Carotid Lesion Seen Pre-RT", "th": "พบรอยโรคหลอดเลือดก่อนฉายแสง"},
    "sig_stenosis_pre_rt":  {"en": "Significant Stenosis Pre-RT (>50%)", "th": "หลอดเลือดตีบ >50% ก่อนฉายแสง"},
    "max_stenosis_pre_rt":  {"en": "Max Stenosis Pre-RT (%)", "th": "ความตีบสูงสุดก่อนฉายแสง (%)"},
    "total_plaque_pre_rt":  {"en": "Total Plaque Score Pre-RT", "th": "คะแนน Plaque รวมก่อนฉายแสง"},
    "mean_imt_pre_rt":      {"en": "Mean IMT Pre-RT (mm)", "th": "ความหนา Intima-Media เฉลี่ย (mm)"},
    "seen_carotid_post_rt": {"en": "Carotid Lesion Seen Post-RT", "th": "พบรอยโรคหลอดเลือดหลังฉายแสง"},
    "progression_post_rt":  {"en": "Progression to Significant Lesion Post-RT", "th": "หลอดเลือดแย่ลงหลังฉายแสง"},
    "BUN":                  {"en": "BUN (mg/dL)", "th": "ค่า BUN (mg/dL)"},
    "Cr":                   {"en": "Creatinine (mg/dL)", "th": "ครีเอตินีน (mg/dL)"},
    "GFR":                  {"en": "GFR (mL/min)", "th": "ค่า GFR (mL/min)"},
    "Total cholesterol":    {"en": "Total Cholesterol (mg/dL)", "th": "คอเลสเตอรอลรวม (mg/dL)"},
    "LDL":                  {"en": "LDL (mg/dL)", "th": "LDL (mg/dL)"},
    "Fasting blood sugar":  {"en": "Fasting Blood Sugar (mg/dL)", "th": "น้ำตาลกลูโคสขณะอดอาหาร (mg/dL)"},
}

meta = {
    "feature_cols":    FEATURE_COLS,
    "num_cols":        num_cols,
    "bool_cols":       bool_cols,
    "medians":         medians,
    "feature_labels":  feature_labels,
    "model_summary":   summary,
    "prevalence":      float(y.mean()),
    "n_patients":      int(len(y)),
    "n_ice":           int(y.sum()),
}
with open(OUT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved {len(model_defs)} model files + metadata.json to {OUT}")
print(f"\nModel summary:")
for name, info in summary.items():
    print(f"  {name:25s}  AUC={info['auc_oof']}")
print("\nNext: commit the models/ directory and deploy to Render/Railway.")
