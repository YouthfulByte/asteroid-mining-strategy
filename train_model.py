"""
Train improved models for the asteroid competition — V2.

Improvements over V1:
1. Feature engineering (interactions, ratios, AI residual modeling)
2. Separate catastrophe models per type (void_rock, structural_collapse, toxic_outgassing)
3. Direct recovered_value model (mineral_value * extraction_yield)
4. Stacked ensemble for value prediction
5. Better hyperparameters with early stopping via CV
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, roc_auc_score
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

BASE_FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + META_COLS]

# ── Feature engineering ──────────────────────────────────────────────────────
def engineer_features(data):
    """Add engineered features to a DataFrame. Works on both training and prediction."""
    d = data.copy()

    # Mineral signature * price * mass interactions
    for mineral in ["iron", "nickel", "cobalt", "platinum", "rare_earth"]:
        sig = f"mineral_signature_{mineral}"
        price = f"mineral_price_{mineral}"
        if sig in d.columns and price in d.columns:
            d[f"value_interaction_{mineral}"] = d[sig] * d[price] * d["mass"]
            d[f"sig_price_{mineral}"] = d[sig] * d[price]

    # Water value interaction
    d["water_value"] = d["water_ice_fraction"] * d["mineral_price_water"] * d["mass"]

    # Total mineral signature
    sig_cols = [c for c in d.columns if c.startswith("mineral_signature_")]
    d["total_mineral_signature"] = d[sig_cols].sum(axis=1)

    # Weighted mineral value estimate (signature * price)
    d["weighted_mineral_est"] = 0.0
    for mineral in ["iron", "nickel", "cobalt", "platinum", "rare_earth"]:
        sig = f"mineral_signature_{mineral}"
        price = f"mineral_price_{mineral}"
        if sig in d.columns and price in d.columns:
            d["weighted_mineral_est"] += d[sig] * d[price]
    d["weighted_mineral_est"] *= d["mass"]

    # Density-porosity interaction (void rock indicator)
    d["density_porosity"] = d["density"] * d["porosity"]
    d["low_density_high_porosity"] = ((d["density"] < 3.0) & (d["porosity"] > 0.3)).astype(float)

    # Structural risk score
    d["structural_risk"] = (1 - d["structural_integrity"]) * d["porosity"]

    # Toxic risk score
    d["toxic_risk"] = d["volatile_content"] * (1 - d["structural_integrity"])

    # AI estimate residual features (help model learn the bias)
    d["ai_est_per_mass"] = d["ai_valuation_estimate"] / (d["mass"] + 1)
    d["analyst_vs_ai"] = d["analyst_consensus_estimate"] - d["ai_valuation_estimate"]
    d["ai_analyst_avg"] = (d["ai_valuation_estimate"] + d["analyst_consensus_estimate"]) / 2

    # Survey quality composite
    d["survey_quality"] = (
        d["survey_confidence"] * d["data_completeness"] * d["spectral_resolution"]
    )

    # Accessibility vs difficulty ratio
    d["access_diff_ratio"] = d["accessibility_score"] / (d["extraction_difficulty"] + 0.01)

    # Location cost composite
    d["location_cost"] = d["delta_v"] * d["fuel_cost_per_unit"]

    # Economic adjusted estimates
    d["ai_est_cycle_adj"] = d["ai_valuation_estimate"] * d["economic_cycle_indicator"]

    # Equipment-yield proxy
    d["equip_drill_interaction"] = d["equipment_compatibility"] * d["drilling_feasibility"]
    d["equip_complete_interaction"] = d["equipment_compatibility"] * d["data_completeness"]

    # Cluster size proxy (not available directly, but cluster_id encodes it somewhat)
    # Risk-weighted value
    d["risk_adj_ai_est"] = d["ai_valuation_estimate"] * d["structural_integrity"]

    return d


df = engineer_features(df)

# Get all feature columns (base + engineered)
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + META_COLS]

print(f"Total features after engineering: {len(FEATURE_COLS)}")

# ── Encode categoricals ─────────────────────────────────────────────────────
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X_all = df[FEATURE_COLS].copy()

# ── Clean subset ─────────────────────────────────────────────────────────────
clean_mask = (df["catastrophe_type"] == "none") & (df["toxic_outgassing_impact"] == 0)
X_clean = X_all[clean_mask]

# ── 1. Mineral Value Model (primary) ────────────────────────────────────────
print("\n=== Training mineral_value model ===")
y_mv = df.loc[clean_mask, "mineral_value"]

lgb_mv = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=127,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
)

scores = cross_val_score(lgb_mv, X_clean, y_mv, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.2f} +/- {scores.std():.2f}")
lgb_mv.fit(X_clean, y_mv)

# Feature importance
imp = pd.Series(lgb_mv.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"  Top features: {list(imp.head(10).index)}")

# ── 2. Recovered Value Model (mineral_value * extraction_yield) ──────────────
print("\n=== Training recovered_value model ===")
y_rv = df.loc[clean_mask, "mineral_value"] * df.loc[clean_mask, "extraction_yield"]

lgb_rv = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=127,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
)

scores = cross_val_score(lgb_rv, X_clean, y_rv, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.2f} +/- {scores.std():.2f}")
lgb_rv.fit(X_clean, y_rv)

# ── 3. Extraction Yield Model ───────────────────────────────────────────────
print("\n=== Training extraction_yield model ===")
y_ey = df.loc[clean_mask, "extraction_yield"]

lgb_ey = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

scores = cross_val_score(lgb_ey, X_clean, y_ey, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.4f} +/- {scores.std():.4f}")
lgb_ey.fit(X_clean, y_ey)

# ── 4. Catastrophe Models (one per type + overall) ──────────────────────────
print("\n=== Training catastrophe models ===")

# Overall catastrophe
y_cat = (df["catastrophe_type"] != "none").astype(int)
lgb_cat = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
    is_unbalance=True,
)
scores = cross_val_score(lgb_cat, X_all, y_cat, cv=5, scoring="roc_auc")
print(f"  Overall catastrophe AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
lgb_cat.fit(X_all, y_cat)

# Void rock
y_void = (df["catastrophe_type"] == "void_rock").astype(int)
lgb_void = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
    is_unbalance=True,
)
scores = cross_val_score(lgb_void, X_all, y_void, cv=5, scoring="roc_auc")
print(f"  Void rock AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
lgb_void.fit(X_all, y_void)

# Structural collapse
y_collapse = (df["catastrophe_type"] == "structural_collapse").astype(int)
lgb_collapse = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
    is_unbalance=True,
)
scores = cross_val_score(lgb_collapse, X_all, y_collapse, cv=5, scoring="roc_auc")
print(f"  Structural collapse AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
lgb_collapse.fit(X_all, y_collapse)

# Toxic outgassing
y_toxic = (df["catastrophe_type"] == "toxic_outgassing").astype(int)
lgb_toxic = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
    is_unbalance=True,
)
scores = cross_val_score(lgb_toxic, X_all, y_toxic, cv=5, scoring="roc_auc")
print(f"  Toxic outgassing AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
lgb_toxic.fit(X_all, y_toxic)

# ── 5. Extraction Delay Model ───────────────────────────────────────────────
print("\n=== Training extraction_delay model ===")
y_delay = df.loc[clean_mask, "extraction_delay"]

lgb_delay = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
)

scores = cross_val_score(lgb_delay, X_clean, y_delay, cv=5, scoring="neg_mean_absolute_error")
print(f"  CV MAE: {-scores.mean():.4f} +/- {scores.std():.4f}")
lgb_delay.fit(X_clean, y_delay)

# ── 6. Negative value classifier (is mineral_value < 0?) ────────────────────
print("\n=== Training negative value classifier ===")
y_neg = (df.loc[clean_mask, "mineral_value"] < 0).astype(int)

lgb_neg = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
    is_unbalance=True,
)

scores = cross_val_score(lgb_neg, X_clean, y_neg, cv=5, scoring="roc_auc")
print(f"  Negative value AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
lgb_neg.fit(X_clean, y_neg)

# ── Compute training stats for bidding calibration ───────────────────────────
print("\n=== Computing calibration stats ===")
clean_df = df[clean_mask].copy()
clean_df["recovered"] = clean_df["mineral_value"] * clean_df["extraction_yield"]

# Percentiles of recovered value by economic cycle
calibration = {}
for cycle in [0.7, 1.0, 1.4]:
    subset = clean_df[clean_df["economic_cycle_indicator"] == cycle]
    calibration[cycle] = {
        "mean": float(subset["recovered"].mean()),
        "median": float(subset["recovered"].median()),
        "p25": float(subset["recovered"].quantile(0.25)),
        "p75": float(subset["recovered"].quantile(0.75)),
    }
    print(f"  Cycle {cycle}: mean={calibration[cycle]['mean']:.1f}, median={calibration[cycle]['median']:.1f}")

# Overall catastrophe rate
cat_rate = (df["catastrophe_type"] != "none").mean()
print(f"  Overall catastrophe rate: {cat_rate:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
model_bundle = {
    "mineral_value_model": lgb_mv,
    "recovered_value_model": lgb_rv,
    "extraction_yield_model": lgb_ey,
    "catastrophe_model": lgb_cat,
    "void_model": lgb_void,
    "collapse_model": lgb_collapse,
    "toxic_model": lgb_toxic,
    "delay_model": lgb_delay,
    "negative_value_model": lgb_neg,
    "feature_cols": FEATURE_COLS,
    "cat_cols": CAT_COLS,
    "label_encoders": label_encoders,
    "calibration": calibration,
    "base_catastrophe_rate": float(cat_rate),
}

out_path = os.path.join("submission", "model.joblib")
joblib.dump(model_bundle, out_path, compress=3)
file_size = os.path.getsize(out_path) / (1024 * 1024)
print(f"\nSaved model to {out_path} ({file_size:.1f} MB)")
print("Done!")
