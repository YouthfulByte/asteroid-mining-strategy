# Asteroid Mining Strategy — Deep Rock Mining Corp

A machine learning-powered bidding strategy for the [Asteroid Auction Challenge](https://github.com/pmanion0/asteroid-competition), a sealed-bid auction competition set in the Belt mining economy of Year 2247.

---

## Overview

This strategy uses **9 LightGBM models** trained on 10,000 historical asteroid records combined with **30 engineered features** and **game-theoretic bidding logic** designed to overcome the winner's curse in first-price sealed-bid auctions.

### Core Insight

In competitive auctions, you tend to win when competitors bid low — which usually means the asteroid is bad. The winning strategy is:
1. **Be extremely selective** — only bid on high-confidence, high-value asteroids
2. **Bid aggressively on the best ones** — to actually beat competitors for the good rocks
3. **Never touch marginal asteroids** — you'll only win the losers

---

## Models

| Model | Task | Target | CV Score |
|-------|------|--------|----------|
| Mineral Value | Regression | `mineral_value` | MAE 63.78 |
| Recovered Value | Regression | `mineral_value × extraction_yield` | MAE 62.72 |
| Extraction Yield | Regression | `extraction_yield` | MAE 0.017 |
| Catastrophe (overall) | Binary Classification | any catastrophe | AUC 0.644 |
| Void Rock | Binary Classification | `void_rock` | AUC 0.530 |
| Structural Collapse | Binary Classification | `structural_collapse` | AUC 0.644 |
| Toxic Outgassing | Binary Classification | `toxic_outgassing` | AUC 0.636 |
| Extraction Delay | Regression | `extraction_delay` | MAE 0.190 |
| Negative Value | Binary Classification | `mineral_value < 0` | AUC 0.962 |

### Simulated Tournament Performance

Tested across 10 randomized tournament simulations (100 rounds, 10 asteroids/round, 5 competitors each):

| Metric | Value |
|--------|-------|
| Mean return | **+182.1%** |
| Min return | +152.1% |
| Max return | +235.4% |
| Mean wins per tournament | 124.7 |
| Catastrophes hit | 0 across all simulations |

---

## Strategy Logic

### 1. Feature Engineering (30 new features)
- **Mineral value interactions**: `signature × price × mass` for each mineral
- **Risk scores**: `structural_risk`, `toxic_risk`, `density × porosity`
- **AI estimate features**: `analyst_vs_ai` residual, `ai_analyst_avg`, `risk_adj_ai_est`
- **Operational composites**: `survey_quality`, `access_diff_ratio`, `location_cost`
- **Equipment interactions**: `equipment × drilling`, `equipment × data_completeness`

### 2. Value Estimation (Ensemble)
Two independent models predict recovered value:
- **Direct model**: predicts `mineral_value × extraction_yield` end-to-end
- **Decomposed model**: predicts `mineral_value` and `extraction_yield` separately

Final estimate: weighted average (55% direct, 45% decomposed)

### 3. Risk Gating (3-Layer Filter)
Before computing any bid, each asteroid passes through three gates:
1. **Negative value gate**: skip if probability of negative mineral value > 50% (AUC 0.962 classifier)
2. **Catastrophe gate**: skip if catastrophe probability > 40%
3. **Minimum value gate**: skip if estimated recovered value < $100

This filtering eliminates the winner's curse by only competing for asteroids we're confident about.

### 4. Bid Sizing
- **Base fraction**: 50-65% of expected value, scaled by model confidence
- **Confidence**: measured by agreement between the two value models × (1 - catastrophe risk) × (1 - negative value risk)
- **Competition adjustment**: less aggressive with few competitors, more with many
- **Concentration**: max 3-8 bids per round (focus capital on best opportunities)

### 5. Capital Management
- **30-90% spend cap** per round (adapts to rounds remaining)
- Reduces exposure when many extractions are pending
- Increases spending when pending revenue is incoming

---

## Repository Structure

```
.
├── README.md                 # This file
├── train_model.py            # Model training script (V2, 9 models + feature engineering)
├── submission/               # Competition submission
│   ├── strategy.py           # Bidding function (V3 — selective + aggressive)
│   └── model.joblib          # Pre-trained model bundle (9.8 MB)
└── data/                     # Training data (not included — see competition repo)
    └── training.parquet
```

### Submission Files

The `submission/` directory is the competition deliverable:

| File | Purpose | Size |
|------|---------|------|
| `strategy.py` | Bidding logic with `load_model()` + `price_asteroids()` | ~7 KB |
| `model.joblib` | 9 LightGBM models + label encoders + calibration data | 9.8 MB |

---

## How It Works

### `load_model()` — Called Once at Tournament Start
Loads the pre-trained model bundle containing:
- 9 LightGBM models (value, yield, catastrophe × 4, delay, negative value)
- Feature column definitions and label encoders
- Economic cycle calibration data

### `price_asteroids(asteroids, capital, round_info, model)` — Called Each Round
1. Converts feature dicts to DataFrame and applies feature engineering
2. Encodes categoricals (`spectral_class`, `belt_region`, `probe_type`)
3. Runs all 9 models for predictions
4. Applies 3-layer risk gating (negative value → catastrophe → minimum value)
5. Computes ensemble recovered value (55/45 weighted average)
6. Sizes bids based on confidence and competition level
7. Selects top opportunities and applies capital constraints
8. Returns bid list

---

## Training

### Prerequisites

```bash
pip install numpy==1.26.4 pandas==2.2.0 pyarrow==15.0.0 scikit-learn==1.4.0 lightgbm joblib==1.3.2
```

### Reproduce the Models

```bash
# Clone the competition repo for training data
git clone https://github.com/pmanion0/asteroid-competition.git
cd asteroid-mining-strategy

# Copy training data
cp ../asteroid-competition/data/training.parquet data/

# Train all 9 models (outputs to submission/model.joblib)
python train_model.py
```

### Training Details

**Data splits:**
- Value/yield/delay models: trained on 7,180 clean rows (no catastrophes, no toxic impact)
- Catastrophe models: trained on all 10,000 rows
- Negative value classifier: trained on 7,180 clean rows

**Key features driving predictions:**
- **Value**: `ai_valuation_estimate` (r=0.97), `analyst_consensus_estimate` (r=0.87), `media_hype_score`, mineral interactions
- **Yield**: `equipment_compatibility` (R²=0.83 alone), `data_completeness`, `drilling_feasibility`
- **Catastrophe**: `structural_integrity` (collapse), `volatile_content` (toxic), `density`/`porosity` (void rock)
- **Negative value**: AUC 0.962 — reliably identifies money-losing asteroids

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 9 separate models | Each target has different data requirements (clean vs all rows) and feature importance |
| Ensemble value prediction | Direct `recovered_value` model + decomposed `mineral_value × yield` reduces prediction variance |
| 3-layer risk gating | Eliminates winner's curse by refusing to bid on uncertain asteroids |
| Aggressive on winners | Bid 50-65% of expected value on high-confidence picks (competitors bid ~20-55% of perceived value) |
| Max 8 bids per round | Concentrating capital beats spreading thin across many marginal bets |
| Feature engineering | 30 engineered features capture mineral value interactions, risk composites, and AI estimate residuals |
| Negative value classifier | AUC 0.962 — cheapest way to avoid the 19% of clean asteroids with negative recovered value |

---

## Version History

| Version | Key Change | Tournament Return |
|---------|-----------|-------------------|
| V1 | Basic LightGBM, 5 models, 40% bid fraction | -26.9% (winner's curse) |
| V2 | Feature engineering, 9 models, better filtering | Improved CV scores |
| **V3** | **Selective gating + aggressive bidding on best picks** | **+182.1% mean** |

---

## Dependencies

All packages match the competition sandbox versions:

| Package | Version |
|---------|---------|
| numpy | 1.26.4 |
| pandas | 2.2.0 |
| scikit-learn | 1.4.0 |
| lightgbm | 4.3.0 |
| joblib | 1.3.2 |

---

## Competition Reference

- [Competition Repository](https://github.com/pmanion0/asteroid-competition)
- [Data Dictionary](https://github.com/pmanion0/asteroid-competition/blob/main/DATA_DICTIONARY.md)
- [Submission Guide](https://github.com/pmanion0/asteroid-competition/blob/main/SUBMISSION_GUIDE.md)
