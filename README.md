# Asteroid Mining Strategy — Deep Rock Mining Corp

A machine learning-powered bidding strategy for the [Asteroid Auction Challenge](https://github.com/pmanion0/asteroid-competition), a sealed-bid auction competition set in the Belt mining economy of Year 2247.

---

## Overview

This strategy uses **5 LightGBM models** trained on 10,000 historical asteroid records to predict value, risk, and operational outcomes — then applies game-theoretic bidding logic to maximize profit while managing capital and catastrophe risk.

### Models

| Model | Task | Target | CV Score |
|-------|------|--------|----------|
| Mineral Value | Regression | `mineral_value` | MAE 65.74 |
| Extraction Yield | Regression | `extraction_yield` | MAE 0.017 |
| Catastrophe | Binary Classification | `catastrophe_type != "none"` | AUC 0.655 |
| Toxic Outgassing | Binary Classification | `catastrophe_type == "toxic_outgassing"` | AUC 0.632 |
| Extraction Delay | Regression | `extraction_delay` | MAE 0.222 |

### Backtest Performance

Tested on 2,000 random training samples (single-bidder simulation):

| Metric | Value |
|--------|-------|
| Bids placed | 1,250 / 2,000 |
| Profitable bids | 82.2% |
| Avg profit per bid | $164.73 |
| Catastrophes hit | 73 (5.8%) |

---

## Strategy Logic

### 1. Value Estimation
For each asteroid, the strategy predicts:
- **Mineral value** — what the rock is worth
- **Extraction yield** — fraction recovered during operations
- **Catastrophe probability** — chance of void rock, structural collapse, or toxic outgassing
- **Extraction delay** — rounds until revenue arrives

### 2. Expected Value Calculation
```
expected_value = mineral_value × extraction_yield × (1 - catastrophe_prob)
```
- Discounted for **time value of money** using the risk-free rate and predicted delay
- Reduced by **expected catastrophe penalties** ($100–$300+ depending on type)
- Reduced by **toxic outgassing cluster risk**

### 3. Bid Sizing
- Bids **40% of expected value** (adjusted for competition level)
- More conservative with fewer competitors (30%), more aggressive with many (45%)
- Extra discount applied for high-catastrophe-probability asteroids
- Bids under $10 are filtered out

### 4. Capital Management
- **Max 35% of capital** spent per round (prevents bankruptcy)
- Ramps to **60–80%** in final rounds (deploy remaining capital)
- Reduces exposure when many extractions are pending
- Bids scaled down proportionally if total exceeds budget

---

## Repository Structure

```
.
├── README.md                 # This file
├── submission/               # Competition submission (strategy + model)
│   ├── strategy.py           # Bidding function (ML interface)
│   └── model.joblib          # Pre-trained LightGBM model bundle (2.5 MB)
├── train_model.py            # Model training script
└── data/                     # Training data (not included — see competition repo)
    └── training.parquet
```

### Submission Files

The `submission/` directory is the competition deliverable:

| File | Purpose | Size |
|------|---------|------|
| `strategy.py` | Bidding logic with `load_model()` + `price_asteroids()` | 6.8 KB |
| `model.joblib` | Bundled LightGBM models + label encoders | 2.5 MB |

---

## How It Works

### `load_model()` — Called Once at Tournament Start
Loads the pre-trained model bundle from `model.joblib` containing all 5 models, feature column definitions, and label encoders for categorical variables.

### `price_asteroids(asteroids, capital, round_info, model)` — Called Each Round
1. Converts the asteroid feature dicts into a DataFrame
2. Encodes categorical features (`spectral_class`, `belt_region`, `probe_type`)
3. Runs all 5 models to get predictions
4. Computes risk-adjusted expected value for each asteroid
5. Applies bid fraction and capital management rules
6. Returns a list of bid amounts

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

# Train models (outputs to submission/model.joblib)
python train_model.py
```

### Training Details

- **mineral_value model**: 1000 trees, depth 7, trained on 7,180 clean rows (no catastrophes/outgassing impact)
- **extraction_yield model**: 500 trees, depth 5, trained on clean rows
- **catastrophe model**: 500 trees, depth 5, trained on all 10,000 rows
- **toxic outgassing model**: 500 trees, depth 5, trained on all rows
- **extraction delay model**: 300 trees, depth 5, trained on clean rows

Key features driving predictions:
- **Value**: `ai_valuation_estimate`, `analyst_consensus_estimate`, `mineral_signature_platinum`, `media_hype_score`
- **Yield**: `equipment_compatibility`, `data_completeness`, `drilling_feasibility`
- **Catastrophe risk**: `structural_integrity`, `volatile_content`, `porosity`, `density`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM over deep learning | Fast inference (<2s timeout), handles mixed feature types natively, small model size |
| Separate catastrophe model | Catastrophe rows have zeroed targets — joint modeling would bias value predictions |
| 40% bid fraction | Balances winner's curse (overpaying) against win rate in 5-player first-price auctions |
| 35% capital cap per round | Prevents overcommitment; preserves liquidity for future opportunities |
| Filter bids < $10 | Transaction costs make tiny wins unprofitable |

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
