"""
ML-based asteroid bidding strategy.

Uses LightGBM models for:
- mineral_value prediction
- extraction_yield prediction
- catastrophe probability
- toxic outgassing probability
- extraction delay prediction

Bidding logic:
1. Predict expected value = mineral_value * extraction_yield * (1 - catastrophe_prob)
2. Discount for delay (time value of money)
3. Subtract expected catastrophe penalties
4. Bid a fraction of expected profit to account for winner's curse
5. Capital management: limit total exposure per round
"""

STRATEGY_NAME = "Deep Rock Mining Corp"


def load_model():
    """Load pre-trained models. Called once at tournament start."""
    import joblib
    import os

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    return joblib.load(model_path)


def price_asteroids(asteroids, capital, round_info, model=None):
    """
    Bid on a batch of asteroids.

    Args:
        asteroids: list of feature dicts
        capital: current liquid capital
        round_info: round metadata
        model: pre-loaded model bundle

    Returns:
        list of bid amounts
    """
    import numpy as np
    import pandas as pd

    if model is None:
        return [0.0] * len(asteroids)

    n = len(asteroids)
    if n == 0:
        return []

    # ── Extract models and config ────────────────────────────────────────
    mv_model = model["mineral_value_model"]
    ey_model = model["extraction_yield_model"]
    cat_model = model["catastrophe_model"]
    toxic_model = model["toxic_model"]
    delay_model = model["delay_model"]
    feature_cols = model["feature_cols"]
    cat_cols = model["cat_cols"]
    label_encoders = model["label_encoders"]

    # ── Build feature DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(asteroids)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Encode categoricals
    for col in cat_cols:
        le = label_encoders[col]
        if col in df.columns:
            # Handle unseen categories by mapping to most common
            known = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )
        else:
            df[col] = 0

    X = df[feature_cols]

    # ── Predict ──────────────────────────────────────────────────────────
    pred_mv = mv_model.predict(X)
    pred_ey = ey_model.predict(X)
    pred_cat_prob = cat_model.predict_proba(X)[:, 1]
    pred_toxic_prob = toxic_model.predict_proba(X)[:, 1]
    pred_delay = delay_model.predict(X)

    # ── Round info ───────────────────────────────────────────────────────
    risk_free_rate = round_info.get("risk_free_rate", 0.002)
    round_number = round_info.get("round_number", 1)
    total_rounds = round_info.get("total_rounds", 100)
    num_competitors = round_info.get("num_active_competitors", 5)
    pending_revenue = round_info.get("pending_revenue", 0.0)
    num_pending = round_info.get("num_pending_extractions", 0)
    rounds_left = total_rounds - round_number

    # ── Compute expected values and bids ─────────────────────────────────
    bids = np.zeros(n)

    # Catastrophe penalties (expected)
    # void_rock: $100, structural_collapse: $200, toxic_outgassing: $300+
    avg_catastrophe_penalty = 180.0  # weighted average

    for i in range(n):
        mv = max(pred_mv[i], 0.0)
        ey = np.clip(pred_ey[i], 0.0, 1.15)
        cat_prob = np.clip(pred_cat_prob[i], 0.0, 1.0)
        toxic_prob = np.clip(pred_toxic_prob[i], 0.0, 1.0)
        delay = max(pred_delay[i], 1.0)

        # Expected recovered value (if no catastrophe)
        recovered = mv * ey

        # Discount for delay (time value of money)
        discount_factor = 1.0 / ((1.0 + risk_free_rate) ** delay)

        # Expected value accounting for catastrophe probability
        # If catastrophe: lose bid + penalty
        # If no catastrophe: get recovered value
        expected_revenue = recovered * (1.0 - cat_prob) * discount_factor

        # Expected catastrophe cost (penalty on top of lost bid)
        expected_penalty = cat_prob * avg_catastrophe_penalty

        # Toxic outgassing cluster risk: extra penalty for nearby asteroids
        # Being conservative here
        toxic_penalty = toxic_prob * 50.0

        # Net expected value
        expected_value = expected_revenue - expected_penalty - toxic_penalty

        if expected_value <= 0:
            bids[i] = 0.0
            continue

        # Bid a fraction of expected value
        # In a first-price auction with N competitors, optimal bid ≈ (N-1)/N * value
        # But we also want margin for error, so bid more conservatively
        # Lower fraction = more conservative = fewer wins but higher profit per win
        bid_fraction = 0.40  # Aggressive enough to win some, conservative enough to profit

        # Adjust bid fraction based on competition
        if num_competitors <= 2:
            bid_fraction = 0.30  # Less competition, bid lower
        elif num_competitors >= 6:
            bid_fraction = 0.45  # More competition, bid higher

        # Adjust for catastrophe risk - reduce bid more for risky asteroids
        safety_mult = 1.0 - 0.5 * cat_prob

        bid = expected_value * bid_fraction * safety_mult
        bids[i] = max(bid, 0.0)

    # ── Capital management ───────────────────────────────────────────────
    # Don't bid too much of capital in one round
    max_round_spend = capital * 0.35  # Spend at most 35% per round

    # Be more aggressive near the end
    if rounds_left <= 5:
        max_round_spend = capital * 0.60
    elif rounds_left <= 2:
        max_round_spend = capital * 0.80

    # Be more conservative if we have many pending extractions
    if num_pending > 5:
        max_round_spend *= 0.7

    # Rank asteroids by expected value / bid ratio and keep best ones
    total_bids = bids.sum()
    if total_bids > max_round_spend and total_bids > 0:
        # Scale down proportionally
        bids = bids * (max_round_spend / total_bids)

    # Filter out tiny bids (not worth the transaction)
    bids[bids < 10.0] = 0.0

    return bids.tolist()
