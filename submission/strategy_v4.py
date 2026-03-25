"""
Asteroid Mining Strategy V4 — Without Trap Features

Uses LightGBM ensemble for value estimation and risk assessment.
Heavy feature engineering compensates for removed trap features.
Conservative bidding with 3-layer risk gating.
"""

STRATEGY_NAME = "Stellar Prospectors V4"


def load_model():
    """Load pre-trained models (called once at tournament start, 30s timeout)."""
    import joblib
    import os
    model_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(model_dir, "model_v4.joblib"))


def price_asteroids(asteroids, capital, round_info, model=None):
    """
    Called each round with 2-second timeout.
    Returns list of bids (same length as asteroids). 0 = pass.
    """
    import numpy as np

    n = len(asteroids)
    if model is None or capital <= 0:
        return [0.0] * n

    feature_cols = model["feature_cols"]
    cat_cols = model["cat_cols"]
    label_encoders = model["label_encoders"]
    trap_cols = model["trap_cols"]

    # ── Build feature matrix ─────────────────────────────────────────────
    rows = []
    for ast in asteroids:
        row = dict(ast)
        rows.append(row)

    # Engineer features (same as training)
    import pandas as pd
    df = pd.DataFrame(rows)

    # Remove trap features if present
    for tc in trap_cols:
        if tc in df.columns:
            df.drop(columns=[tc], inplace=True)

    df = _engineer_features(df)

    # Encode categoricals
    for col in cat_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, _le=le: _safe_transform(_le, x)
            )

    # Align to training feature order
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].values

    # ── Model predictions ────────────────────────────────────────────────
    pred_mv = model["mineral_value_model"].predict(X)
    pred_rv = model["recovered_value_model"].predict(X)
    pred_yield = model["extraction_yield_model"].predict(X)
    pred_delay = model["delay_model"].predict(X)

    p_cat = model["catastrophe_model"].predict_proba(X)[:, 1]
    p_void = model["void_model"].predict_proba(X)[:, 1]
    p_collapse = model["collapse_model"].predict_proba(X)[:, 1]
    p_toxic = model["toxic_model"].predict_proba(X)[:, 1]
    p_neg = model["negative_value_model"].predict_proba(X)[:, 1]
    p_high = model["high_value_model"].predict_proba(X)[:, 1]

    # ── Compute expected values ──────────────────────────────────────────
    # Ensemble: average of direct recovered value and decomposed estimate
    decomposed_rv = pred_mv * np.clip(pred_yield, 0.0, 1.5)
    ensemble_rv = 0.5 * pred_rv + 0.5 * decomposed_rv

    # Expected catastrophe penalty
    cat_penalty = p_void * 100 + p_collapse * 200 + p_toxic * 300

    # Risk-free rate and delay discount
    rfr = round_info.get("risk_free_rate", 0.002)
    delay_discount = (1 + rfr) ** (-np.clip(pred_delay, 1, 20))

    # Expected value = P(no catastrophe) * recovered_value * delay_discount - P(catastrophe) * penalty
    ev = (1 - p_cat) * ensemble_rv * delay_discount - p_cat * cat_penalty

    # ── Round context ────────────────────────────────────────────────────
    round_num = round_info.get("round_number", 1)
    total_rounds = round_info.get("total_rounds", 100)
    progress = round_num / total_rounds
    n_competitors = round_info.get("num_active_competitors", 5)
    pending = round_info.get("pending_revenue", 0)
    n_pending = round_info.get("num_pending_extractions", 0)

    # Adaptive capital allocation
    if progress < 0.2:
        max_spend_frac = 0.25  # Conservative early
    elif progress < 0.7:
        max_spend_frac = 0.40  # Normal mid-game
    elif progress < 0.9:
        max_spend_frac = 0.60  # More aggressive late
    else:
        max_spend_frac = 0.85  # Deploy remaining capital

    # Reduce if lots of pending extractions (capital is locked)
    if n_pending > 5:
        max_spend_frac *= 0.7
    elif n_pending > 3:
        max_spend_frac *= 0.85

    max_budget = capital * max_spend_frac

    # ── 3-Layer Risk Gating + Bid Computation ────────────────────────────
    bid_candidates = []

    for i in range(n):
        bid = 0.0

        # Gate 1: Skip negative value asteroids
        if p_neg[i] > 0.55:
            bid_candidates.append((i, 0.0, -1000))
            continue

        # Gate 2: Skip high catastrophe risk
        if p_cat[i] > 0.35:
            bid_candidates.append((i, 0.0, -1000))
            continue

        # Gate 3: Minimum expected value
        if ev[i] < 50:
            bid_candidates.append((i, 0.0, -1000))
            continue

        # Bid fraction: more aggressive for high-confidence opportunities
        base_fraction = 0.50

        # Boost for high-value confidence
        if p_high[i] > 0.7:
            base_fraction = 0.60
        elif p_high[i] > 0.5:
            base_fraction = 0.55

        # Model agreement bonus (if both models agree, more confident)
        model_diff = abs(pred_rv[i] - decomposed_rv[i]) / (abs(ensemble_rv[i]) + 1)
        if model_diff < 0.2:
            base_fraction += 0.05  # Models agree → bid more
        elif model_diff > 0.5:
            base_fraction -= 0.10  # Models disagree → bid less

        # Low catastrophe risk bonus
        if p_cat[i] < 0.05:
            base_fraction += 0.03

        # Competitor adjustment: fewer competitors = can bid less
        if n_competitors <= 2:
            base_fraction *= 0.80
        elif n_competitors <= 3:
            base_fraction *= 0.90

        bid = max(0, ev[i] * np.clip(base_fraction, 0.30, 0.70))
        bid_candidates.append((i, bid, ev[i]))

    # ── Select top bids within budget ────────────────────────────────────
    # Sort by expected value, pick top ones
    bid_candidates.sort(key=lambda x: x[2], reverse=True)

    # Limit number of bids per round
    max_bids = min(8, max(2, int(n * 0.4)))
    if progress > 0.85:
        max_bids = min(10, n)

    bids = [0.0] * n
    total_bid = 0.0
    placed = 0

    for idx, bid_amt, ev_val in bid_candidates:
        if placed >= max_bids:
            break
        if bid_amt <= 0:
            continue
        if total_bid + bid_amt > max_budget:
            # Can we fit a smaller bid?
            remaining = max_budget - total_bid
            if remaining > 30 and ev_val > 80:
                bid_amt = min(bid_amt, remaining)
            else:
                break
        bids[idx] = bid_amt
        total_bid += bid_amt
        placed += 1

    return bids


def _engineer_features(d):
    """Replicate feature engineering from training."""
    import numpy as np

    minerals = ["iron", "nickel", "cobalt", "platinum", "rare_earth"]

    for m in minerals:
        sig, price = f"mineral_signature_{m}", f"mineral_price_{m}"
        if sig in d.columns and price in d.columns:
            d[f"value_int_{m}"] = d[sig] * d[price] * d["mass"]
            d[f"sig_price_{m}"] = d[sig] * d[price]
            d[f"sig_sq_{m}"] = d[sig] ** 2
            d[f"log_value_{m}"] = np.log1p(d[sig] * d[price]) * np.log1p(d["mass"])

    d["total_mineral_value"] = sum(
        d[f"mineral_signature_{m}"] * d[f"mineral_price_{m}"] * d["mass"]
        for m in minerals
    )
    d["total_sig_price"] = sum(
        d[f"mineral_signature_{m}"] * d[f"mineral_price_{m}"] for m in minerals
    )
    sig_cols = [f"mineral_signature_{m}" for m in minerals]
    d["total_signature"] = d[[c for c in sig_cols if c in d.columns]].sum(axis=1)
    d["water_value"] = d["water_ice_fraction"] * d["mineral_price_water"] * d["mass"]
    d["total_value_with_water"] = d["total_mineral_value"] + d["water_value"]
    d["cycle_adj_value"] = d["total_mineral_value"] * d["economic_cycle_indicator"]
    d["cycle_sq"] = d["economic_cycle_indicator"] ** 2
    d["log_mass"] = np.log1p(d["mass"])
    d["mass_density"] = d["mass"] * d["density"]
    d["mass_sq_root"] = np.sqrt(d["mass"].clip(lower=0))
    d["density_porosity"] = d["density"] * d["porosity"]
    d["density_integrity"] = d["density"] * d["structural_integrity"]
    d["volume_density"] = d["estimated_volume"] * d["density"]
    d["structural_risk"] = (1 - d["structural_integrity"]) * d["porosity"]
    d["toxic_risk"] = d["volatile_content"] * (1 - d["structural_integrity"])
    d["void_risk"] = (1 - d["density"] / 8.0) * d["porosity"]
    d["combined_risk"] = d["structural_risk"] + d["toxic_risk"] + d["void_risk"]
    d["integrity_volatile"] = d["structural_integrity"] * (1 - d["volatile_content"])
    d["survey_quality"] = d["survey_confidence"] * d["data_completeness"] * d["spectral_resolution"]
    d["survey_reliability"] = d["survey_confidence"] * d["surveyor_reputation"] * (1 - d["conflicting_results"])
    d["probe_confidence"] = d["survey_confidence"] * d["data_completeness"]
    d["access_diff_ratio"] = d["accessibility_score"] / (d["extraction_difficulty"] + 0.01)
    d["equip_drill"] = d["equipment_compatibility"] * d["drilling_feasibility"]
    d["equip_access"] = d["equipment_compatibility"] * d["accessibility_score"]
    d["operational_score"] = d["equipment_compatibility"] * d["drilling_feasibility"] * d["accessibility_score"]
    d["location_cost"] = d["delta_v"] * d["fuel_cost_per_unit"]
    d["total_cost_est"] = d["location_cost"] + d["estimated_extraction_cost"]
    d["net_value_est"] = d["total_mineral_value"] - d["total_cost_est"]
    d["value_per_cost"] = d["total_mineral_value"] / (d["total_cost_est"] + 1)
    d["station_access"] = 1.0 / (d["nearest_station_distance"] + 0.01)
    d["infra_station"] = d["infrastructure_proximity"] * d["station_access"]
    d["platinum_ratio"] = d["mineral_signature_platinum"] / (d["total_signature"] + 0.001)
    d["iron_ratio"] = d["mineral_signature_iron"] / (d["total_signature"] + 0.001)
    d["rare_earth_ratio"] = d["mineral_signature_rare_earth"] / (d["total_signature"] + 0.001)
    d["mass_x_escape_vel"] = d["mass"] * d["escape_velocity"]
    d["gravity_x_mass"] = d["surface_gravity"] * d["mass"]
    d["post_tax_value"] = d["total_mineral_value"] * (1 - d["tax_rate"])
    d["insurance_cost"] = d["total_mineral_value"] * d["insurance_rate"]
    d["net_after_costs"] = d["post_tax_value"] - d["insurance_cost"] - d["location_cost"]

    return d


def _safe_transform(le, val):
    """Handle unseen labels gracefully."""
    if val in le.classes_:
        return le.transform([val])[0]
    return 0  # Default for unseen
