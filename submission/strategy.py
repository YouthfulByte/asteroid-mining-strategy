"""
Deep Rock Mining Corp — V3 Strategy

Key insight: In a competitive first-price auction, you mainly win when competitors
bid low — which usually means the asteroid is bad (winner's curse). To profit:

1. Be VERY selective — only bid on asteroids with high confidence of positive value
2. Bid aggressively on the best ones (to actually win them against competitors)
3. Never bid on marginal asteroids (you'll only win the losers)
4. Use model uncertainty to avoid overconfidence on noisy predictions
"""

STRATEGY_NAME = "Deep Rock Mining Corp"


def load_model():
    """Load pre-trained models. Called once at tournament start."""
    import joblib
    import os

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    return joblib.load(model_path)


def _engineer_features(df):
    """Add engineered features. Must match train_model_v2.py exactly."""
    d = df.copy()

    for mineral in ["iron", "nickel", "cobalt", "platinum", "rare_earth"]:
        sig = f"mineral_signature_{mineral}"
        price = f"mineral_price_{mineral}"
        if sig in d.columns and price in d.columns:
            d[f"value_interaction_{mineral}"] = d[sig] * d[price] * d["mass"]
            d[f"sig_price_{mineral}"] = d[sig] * d[price]

    d["water_value"] = d["water_ice_fraction"] * d["mineral_price_water"] * d["mass"]

    sig_cols = [c for c in d.columns if c.startswith("mineral_signature_")]
    d["total_mineral_signature"] = d[sig_cols].sum(axis=1)

    d["weighted_mineral_est"] = 0.0
    for mineral in ["iron", "nickel", "cobalt", "platinum", "rare_earth"]:
        sig = f"mineral_signature_{mineral}"
        price = f"mineral_price_{mineral}"
        if sig in d.columns and price in d.columns:
            d["weighted_mineral_est"] += d[sig] * d[price]
    d["weighted_mineral_est"] *= d["mass"]

    d["density_porosity"] = d["density"] * d["porosity"]
    d["low_density_high_porosity"] = ((d["density"] < 3.0) & (d["porosity"] > 0.3)).astype(float)
    d["structural_risk"] = (1 - d["structural_integrity"]) * d["porosity"]
    d["toxic_risk"] = d["volatile_content"] * (1 - d["structural_integrity"])
    d["ai_est_per_mass"] = d["ai_valuation_estimate"] / (d["mass"] + 1)
    d["analyst_vs_ai"] = d["analyst_consensus_estimate"] - d["ai_valuation_estimate"]
    d["ai_analyst_avg"] = (d["ai_valuation_estimate"] + d["analyst_consensus_estimate"]) / 2
    d["survey_quality"] = d["survey_confidence"] * d["data_completeness"] * d["spectral_resolution"]
    d["access_diff_ratio"] = d["accessibility_score"] / (d["extraction_difficulty"] + 0.01)
    d["location_cost"] = d["delta_v"] * d["fuel_cost_per_unit"]
    d["ai_est_cycle_adj"] = d["ai_valuation_estimate"] * d["economic_cycle_indicator"]
    d["equip_drill_interaction"] = d["equipment_compatibility"] * d["drilling_feasibility"]
    d["equip_complete_interaction"] = d["equipment_compatibility"] * d["data_completeness"]
    d["risk_adj_ai_est"] = d["ai_valuation_estimate"] * d["structural_integrity"]

    return d


def price_asteroids(asteroids, capital, round_info, model=None):
    """Bid on a batch of asteroids — selective and aggressive."""
    import numpy as np
    import pandas as pd

    if model is None:
        return [0.0] * len(asteroids)

    n = len(asteroids)
    if n == 0:
        return []

    # ── Extract models ───────────────────────────────────────────────────
    mv_model = model["mineral_value_model"]
    rv_model = model["recovered_value_model"]
    ey_model = model["extraction_yield_model"]
    cat_model = model["catastrophe_model"]
    void_model = model["void_model"]
    collapse_model = model["collapse_model"]
    toxic_model = model["toxic_model"]
    delay_model = model["delay_model"]
    neg_model = model["negative_value_model"]
    feature_cols = model["feature_cols"]
    cat_cols = model["cat_cols"]
    label_encoders = model["label_encoders"]

    # ── Build feature DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(asteroids)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = _engineer_features(df)

    for col in cat_cols:
        le = label_encoders[col]
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].apply(
                lambda x, k=known, l=le: l.transform([x])[0] if x in k else 0
            )
        else:
            df[col] = 0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols]

    # ── Predict ──────────────────────────────────────────────────────────
    pred_mv = mv_model.predict(X)
    pred_rv = rv_model.predict(X)
    pred_ey = ey_model.predict(X)
    pred_cat_prob = cat_model.predict_proba(X)[:, 1]
    pred_void_prob = void_model.predict_proba(X)[:, 1]
    pred_collapse_prob = collapse_model.predict_proba(X)[:, 1]
    pred_toxic_prob = toxic_model.predict_proba(X)[:, 1]
    pred_delay = delay_model.predict(X)
    pred_neg_prob = neg_model.predict_proba(X)[:, 1]

    # ── Round info ───────────────────────────────────────────────────────
    risk_free_rate = round_info.get("risk_free_rate", 0.002)
    round_number = round_info.get("round_number", 1)
    total_rounds = round_info.get("total_rounds", 100)
    num_competitors = round_info.get("num_active_competitors", 5)
    pending_revenue = round_info.get("pending_revenue", 0.0)
    num_pending = round_info.get("num_pending_extractions", 0)
    rounds_left = total_rounds - round_number

    # ── Compute expected values ──────────────────────────────────────────
    expected_values = np.zeros(n)
    confidence_scores = np.zeros(n)

    for i in range(n):
        neg_prob = np.clip(pred_neg_prob[i], 0, 1)
        cat_prob = np.clip(pred_cat_prob[i], 0, 0.95)
        void_prob = np.clip(pred_void_prob[i], 0, 0.95)
        collapse_prob = np.clip(pred_collapse_prob[i], 0, 0.95)
        toxic_prob = np.clip(pred_toxic_prob[i], 0, 0.95)

        # ── GATE 1: Skip likely negative or catastrophe-prone ────────
        if neg_prob > 0.50:
            continue
        if cat_prob > 0.40:
            continue

        # ── Ensemble recovered value ─────────────────────────────────
        rv1 = pred_rv[i]
        mv = pred_mv[i]
        ey = np.clip(pred_ey[i], 0.0, 1.15)
        rv2 = mv * ey
        recovered = 0.55 * rv1 + 0.45 * rv2

        # ── GATE 2: Skip low-value asteroids ─────────────────────────
        # Only bid on asteroids with substantial expected value
        # This is the key anti-winner's-curse measure
        if recovered < 100:
            continue

        # ── Discount and penalties ───────────────────────────────────
        delay = max(pred_delay[i], 1.0)
        discount = 1.0 / ((1.0 + risk_free_rate) ** delay)

        expected_penalty = (
            void_prob * 100.0
            + collapse_prob * 200.0
            + toxic_prob * 350.0
        )

        ev = recovered * (1.0 - cat_prob) * discount - expected_penalty
        ev *= (1.0 - 0.3 * neg_prob)

        if ev < 50:
            continue

        expected_values[i] = ev

        # Confidence: high when models agree, low catastrophe risk, low neg prob
        model_agreement = 1.0 - min(abs(rv1 - rv2) / (abs(recovered) + 1), 1.0)
        confidence_scores[i] = model_agreement * (1 - cat_prob) * (1 - neg_prob)

    # ── Bid computation ──────────────────────────────────────────────────
    # Strategy: bid aggressively on the BEST opportunities
    # In first-price auctions, you want to bid just above what competitors bid
    # For high-value asteroids, competitors also bid high, so we need to be aggressive
    bids = np.zeros(n)

    for i in range(n):
        if expected_values[i] <= 0:
            continue

        ev = expected_values[i]
        conf = confidence_scores[i]

        # Base bid: aggressive fraction of expected value
        # Higher confidence → bid closer to value (more likely to be real)
        # Lower confidence → bid less (protect against model error)
        base_frac = 0.50 + 0.15 * conf  # Range: 0.50 to 0.65

        # Adjust for competition
        if num_competitors <= 2:
            comp_adj = 0.80  # Less competition
        elif num_competitors <= 3:
            comp_adj = 0.90
        elif num_competitors >= 7:
            comp_adj = 1.10  # More competition
        else:
            comp_adj = 1.0

        bid = ev * base_frac * comp_adj
        bids[i] = max(bid, 0)

    # ── Select top opportunities ─────────────────────────────────────────
    # Focus on the best N asteroids to concentrate capital
    max_wins_per_round = max(3, min(8, n // 2))
    nonzero_mask = bids > 0

    if nonzero_mask.sum() > max_wins_per_round:
        nonzero_idx = np.where(nonzero_mask)[0]
        # Rank by expected_value (not bid) to pick best opportunities
        ev_at_idx = expected_values[nonzero_idx]
        sorted_order = np.argsort(-ev_at_idx)
        for rank, oi in enumerate(sorted_order):
            if rank >= max_wins_per_round:
                bids[nonzero_idx[oi]] = 0.0

    # ── Capital management ───────────────────────────────────────────────
    if rounds_left <= 1:
        max_spend_frac = 0.90
    elif rounds_left <= 3:
        max_spend_frac = 0.65
    elif rounds_left <= 8:
        max_spend_frac = 0.45
    else:
        max_spend_frac = 0.30

    # Adjust for pending extractions
    if num_pending > 8:
        max_spend_frac *= 0.5
    elif num_pending > 5:
        max_spend_frac *= 0.7

    # Allow more spending if pending revenue is coming
    if pending_revenue > capital * 0.5 and num_pending <= 4:
        max_spend_frac = min(max_spend_frac + 0.15, 0.90)

    max_spend = capital * max_spend_frac

    total_bids = bids.sum()
    if total_bids > max_spend and total_bids > 0:
        bids *= max_spend / total_bids

    # Filter small bids
    bids[bids < 15.0] = 0.0

    return bids.tolist()
