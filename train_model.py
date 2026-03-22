"""
Train models for the asteroid competition:
1. mineral_value regression (on clean rows)
2. extraction_yield regression (on clean rows)
3. catastrophe probability classification (all rows)
4. extraction_delay regression (on clean rows)

Saves a single joblib dict to submission/model.joblib
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import os

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/training.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ── Define columns ───────────────────────────────────────────────────────────
TARGET_COLS = [
    "mineral_value", "extraction_yield", "extraction_delay",
    "catastrophe_type", "toxic_outgassing_impact",
]
META_COLS = ["asteroid_id", "time_period"]
CAT_COLS = ["spectral_class", "belt_region", "probe_type"]

FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + META_COLS]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

# ── Encode categoricals ─────────────────────────────────────────────────────
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X_all = df[FEATURE_COLS].copy()

# ── Clean subset (no catastrophe, no outgassing impact) ─────────────────────
clean_mask = (df["catastrophe_type"] == "none") & (df["toxic_outgassing_impact"] == 0)
X_clean = X_all[clean_mask]

# ── 1. Mineral Value Model ──────────────────────────────────────────────────
print("\n=== Training mineral_value model ===")
y_mv = df.loc[clean_mask, "mineral_value"]

lgb_mv = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

scores = cross_val_score(lgb_mv, X_clean, y_mv, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

lgb_mv.fit(X_clean, y_mv)

# Feature importance
imp = pd.Series(lgb_mv.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"  Top features: {list(imp.head(10).index)}")

# ── 2. Extraction Yield Model ───────────────────────────────────────────────
print("\n=== Training extraction_yield model ===")
y_ey = df.loc[clean_mask, "extraction_yield"]

lgb_ey = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,

)

scores = cross_val_score(lgb_ey, X_clean, y_ey, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.4f} ± {scores.std():.4f}")

lgb_ey.fit(X_clean, y_ey)

# ── 3. Catastrophe Classifier ───────────────────────────────────────────────
print("\n=== Training catastrophe model ===")
y_cat = (df["catastrophe_type"] != "none").astype(int)

lgb_cat = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,

)

scores = cross_val_score(lgb_cat, X_all, y_cat, cv=5, scoring="roc_auc")
print(f"  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

lgb_cat.fit(X_all, y_cat)

# ── 4. Toxic Outgassing Classifier ──────────────────────────────────────────
print("\n=== Training toxic outgassing model ===")
y_toxic = (df["catastrophe_type"] == "toxic_outgassing").astype(int)

lgb_toxic = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,

)

scores = cross_val_score(lgb_toxic, X_all, y_toxic, cv=5, scoring="roc_auc")
print(f"  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

lgb_toxic.fit(X_all, y_toxic)

# ── 5. Extraction Delay Model ───────────────────────────────────────────────
print("\n=== Training extraction_delay model ===")
y_delay = df.loc[clean_mask, "extraction_delay"]

lgb_delay = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,

)

scores = cross_val_score(lgb_delay, X_clean, y_delay, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.4f} ± {scores.std():.4f}")

lgb_delay.fit(X_clean, y_delay)

# ── Save ─────────────────────────────────────────────────────────────────────
model_bundle = {
    "mineral_value_model": lgb_mv,
    "extraction_yield_model": lgb_ey,
    "catastrophe_model": lgb_cat,
    "toxic_model": lgb_toxic,
    "delay_model": lgb_delay,
    "feature_cols": FEATURE_COLS,
    "cat_cols": CAT_COLS,
    "label_encoders": label_encoders,
}

out_path = os.path.join("submission", "model.joblib")
joblib.dump(model_bundle, out_path, compress=3)
file_size = os.path.getsize(out_path) / (1024 * 1024)
print(f"\nSaved model to {out_path} ({file_size:.1f} MB)")
print("Done!")
