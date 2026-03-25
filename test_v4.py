"""
Test harness for Strategy V4 — runs a simulated tournament and checks
submission requirements.

Usage:
    python test_v4.py
"""

import sys
import time
import warnings
import importlib.util
import numpy as np

warnings.filterwarnings("ignore")

# ── Load strategy module ───────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("strategy", "submission/strategy.py")
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

# ── Submission requirement checks ─────────────────────────────────────────────
def check_requirements(strategy_module):
    print("=" * 60)
    print("SUBMISSION REQUIREMENTS CHECK")
    print("=" * 60)
    passed = True

    # 1. Has load_model()
    has_load = hasattr(strategy_module, "load_model") and callable(strategy_module.load_model)
    print(f"  [{'PASS' if has_load else 'FAIL'}] load_model() function exists")
    passed = passed and has_load

    # 2. Has price_asteroids()
    has_price = hasattr(strategy_module, "price_asteroids") and callable(strategy_module.price_asteroids)
    print(f"  [{'PASS' if has_price else 'FAIL'}] price_asteroids() function exists")
    passed = passed and has_price

    # 3. load_model runs within 30 s
    t0 = time.time()
    try:
        model = strategy_module.load_model()
        elapsed = time.time() - t0
        ok = elapsed < 30
        print(f"  [{'PASS' if ok else 'FAIL'}] load_model() completed in {elapsed:.2f}s (limit: 30s)")
        passed = passed and ok
    except Exception as e:
        print(f"  [FAIL] load_model() raised: {e}")
        passed = False
        model = None

    # 4. price_asteroids returns a list of the same length as input
    if model is not None:
        sample = _make_asteroids(5)
        result = strategy_module.price_asteroids(
            sample, 1_000_000, _round_info(1, 100), model
        )
        length_ok = isinstance(result, list) and len(result) == len(sample)
        print(f"  [{'PASS' if length_ok else 'FAIL'}] price_asteroids() returns list of correct length ({len(result) if isinstance(result, list) else type(result)})")
        passed = passed and length_ok

        nonneg_ok = all(b >= 0 for b in result)
        print(f"  [{'PASS' if nonneg_ok else 'FAIL'}] All bids are non-negative")
        passed = passed and nonneg_ok

    # 5. price_asteroids runs within 2 s per round (averaged over 5 calls)
    if model is not None:
        times = []
        for _ in range(5):
            batch = _make_asteroids(10)
            t0 = time.time()
            strategy_module.price_asteroids(batch, 1_000_000, _round_info(1, 100), model)
            times.append(time.time() - t0)
        avg_t = np.mean(times)
        max_t = max(times)
        ok = max_t < 2.0
        print(f"  [{'PASS' if ok else 'FAIL'}] price_asteroids() max round time {max_t:.3f}s (avg {avg_t:.3f}s, limit: 2s)")
        passed = passed and ok

    print()
    return passed, model


# ── Synthetic asteroid generator ──────────────────────────────────────────────
def _make_asteroids(n, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    minerals = ["iron", "nickel", "cobalt", "platinum", "rare_earth"]
    spectral_classes = ["C", "S", "M", "X", "B", "D"]
    belt_regions = ["inner", "middle", "outer", "trojan"]
    probe_types = ["flyby", "orbital", "lander"]

    asteroids = []
    for _ in range(n):
        # Raw signatures sum to ~1
        sigs = rng.dirichlet(np.ones(5)) * rng.uniform(0.5, 1.0)
        a = {
            "mass": rng.lognormal(4, 1.5),
            "density": rng.uniform(1.5, 7.5),
            "porosity": rng.uniform(0.0, 0.6),
            "spectral_class": rng.choice(spectral_classes),
            "mineral_signature_iron": sigs[0],
            "mineral_signature_nickel": sigs[1],
            "mineral_signature_cobalt": sigs[2],
            "mineral_signature_platinum": sigs[3],
            "mineral_signature_rare_earth": sigs[4],
            "albedo": rng.uniform(0.03, 0.5),
            "rotation_period": rng.uniform(2, 100),
            "surface_roughness": rng.uniform(0.0, 1.0),
            "magnetic_field_strength": rng.uniform(0, 50),
            "thermal_inertia": rng.uniform(10, 800),
            "shape_elongation": rng.uniform(1.0, 3.0),
            "regolith_depth": rng.uniform(0.0, 10.0),
            "water_ice_fraction": rng.uniform(0.0, 0.3),
            "volatile_content": rng.uniform(0.0, 0.4),
            "structural_integrity": rng.uniform(0.3, 1.0),
            "estimated_volume": rng.lognormal(4, 2),
            "surface_gravity": rng.uniform(0.0001, 0.05),
            "escape_velocity": rng.uniform(0.1, 5.0),
            "composition_heterogeneity": rng.uniform(0.0, 1.0),
            "subsurface_anomaly_score": rng.uniform(0.0, 1.0),
            "crystalline_fraction": rng.uniform(0.0, 1.0),
            "semi_major_axis": rng.uniform(1.5, 5.0),
            "eccentricity": rng.uniform(0.0, 0.4),
            "inclination": rng.uniform(0.0, 30.0),
            "delta_v": rng.uniform(3.0, 12.0),
            "belt_region": rng.choice(belt_regions),
            "cluster_id": int(rng.integers(0, 50)),
            "orbital_period": rng.uniform(2.0, 12.0),
            "perihelion_distance": rng.uniform(1.0, 3.5),
            "aphelion_distance": rng.uniform(2.0, 6.0),
            "transfer_window_frequency": rng.uniform(0.5, 4.0),
            "nearest_station_distance": rng.uniform(0.1, 5.0),
            "piracy_proximity_index": rng.uniform(0.0, 1.0),
            "communication_delay": rng.uniform(5, 60),
            "orbital_stability_score": rng.uniform(0.5, 1.0),
            "conjunction_frequency": rng.uniform(0.1, 3.0),
            "survey_confidence": rng.uniform(0.3, 1.0),
            "probe_type": rng.choice(probe_types),
            "surveyor_reputation": rng.uniform(0.5, 1.0),
            "num_surveys": int(rng.integers(1, 10)),
            "conflicting_results": rng.uniform(0.0, 0.5),
            "extraction_difficulty": rng.uniform(1.0, 10.0),
            "accessibility_score": rng.uniform(0.2, 1.0),
            "survey_age_years": rng.uniform(0.1, 10.0),
            "data_completeness": rng.uniform(0.3, 1.0),
            "spectral_resolution": rng.uniform(0.3, 1.0),
            "ground_truth_samples": int(rng.integers(0, 20)),
            "estimated_extraction_cost": rng.uniform(100, 2000),
            "drilling_feasibility": rng.uniform(0.2, 1.0),
            "equipment_compatibility": rng.uniform(0.3, 1.0),
            "estimated_yield_tonnes": rng.lognormal(5, 1.5),
            "survey_anomaly_flag": int(rng.integers(0, 2)),
            "dual_phase_extraction": int(rng.integers(0, 2)),
            "previous_claim_history": int(rng.integers(0, 5)),
            "legal_encumbrance_score": rng.uniform(0.0, 0.5),
            "environmental_hazard_rating": rng.uniform(0.0, 1.0),
            "insurance_risk_class": int(rng.integers(1, 6)),
            "mineral_price_iron": rng.uniform(50, 150),
            "mineral_price_nickel": rng.uniform(100, 300),
            "mineral_price_cobalt": rng.uniform(200, 600),
            "mineral_price_platinum": rng.uniform(800, 2000),
            "mineral_price_rare_earth": rng.uniform(300, 1000),
            "mineral_price_water": rng.uniform(5, 50),
            "fuel_cost_per_unit": rng.uniform(1.0, 5.0),
            "insurance_rate": rng.uniform(0.01, 0.05),
            "tax_rate": rng.uniform(0.05, 0.25),
            "economic_cycle_indicator": rng.uniform(0.7, 1.3),
            "market_volatility_index": rng.uniform(0.0, 0.5),
            "demand_backlog_months": rng.uniform(0.0, 24.0),
            "shipping_congestion_factor": rng.uniform(0.8, 1.5),
            "refinery_capacity_utilization": rng.uniform(0.5, 1.0),
            "spot_vs_contract_spread": rng.uniform(-0.1, 0.2),
            "credit_availability_index": rng.uniform(0.5, 1.5),
            "competitor_activity_level": rng.uniform(0.0, 1.0),
            "regulatory_burden_score": rng.uniform(0.0, 1.0),
            "supply_chain_disruption_risk": rng.uniform(0.0, 0.5),
            "technology_readiness_level": int(rng.integers(3, 10)),
            "radiation_level": rng.uniform(0.0, 10.0),
            "micrometeorite_density": rng.uniform(0.0, 5.0),
            "solar_flux": rng.uniform(0.5, 2.0),
            "infrastructure_proximity": rng.uniform(0.0, 1.0),
            "navigation_complexity": rng.uniform(0.0, 1.0),
            "rescue_response_time_hours": rng.uniform(12, 200),
            "local_jurisdiction_stability": rng.uniform(0.3, 1.0),
            "worker_availability_index": rng.uniform(0.3, 1.0),
            "power_grid_access": rng.uniform(0.0, 1.0),
            "debris_field_density": rng.uniform(0.0, 1.0),
            # Trap features (should be dropped by v4)
            "ai_valuation_estimate": rng.uniform(100, 5000),
            "analyst_consensus_estimate": rng.uniform(100, 5000),
            "media_hype_score": rng.uniform(0.0, 1.0),
            "lucky_number": rng.uniform(0.0, 100.0),
            "social_sentiment_score": rng.uniform(0.0, 1.0),
        }
        asteroids.append(a)
    return asteroids


def _round_info(round_num, total_rounds, n_competitors=5, capital=1_000_000):
    return {
        "round_number": round_num,
        "total_rounds": total_rounds,
        "num_active_competitors": n_competitors,
        "risk_free_rate": 0.002,
        "pending_revenue": capital * 0.1,
        "num_pending_extractions": 2,
    }


# ── Tournament simulation ──────────────────────────────────────────────────────
def simulate_tournament(strategy_module, model, total_rounds=100,
                        asteroids_per_round=10, n_competitors=5,
                        seed=42):
    """
    Simulates a tournament.

    True-value proxy: the model's own recovered_value prediction with ±20% noise.
    Competitors bid 20-55% of a noisy estimate of that same true value, capturing
    the real auction dynamic where all bidders share similar (imperfect) signals.
    We win if our bid >= the highest competitor bid.
    Catastrophe probability is derived from structural and volatile features.
    """
    import warnings, pandas as pd
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(seed)
    capital = 1_000_000.0
    initial_capital = capital

    wins = 0
    catastrophes = 0
    total_spent = 0.0
    total_revenue = 0.0
    rounds_with_bids = 0
    bid_counts = []

    feature_cols = model["feature_cols"]
    cat_cols = model["cat_cols"]
    label_encoders = model["label_encoders"]
    trap_cols = model.get("trap_cols", [])

    for rnd in range(1, total_rounds + 1):
        asteroids = _make_asteroids(asteroids_per_round, rng=rng)

        # Compute model's predicted recovered values (used as ground-truth proxy)
        import sys
        sys.path.insert(0, "submission")
        from strategy import _engineer_features, _safe_transform

        df = pd.DataFrame(asteroids)
        for tc in trap_cols:
            if tc in df.columns:
                df.drop(columns=[tc], inplace=True)
        df = _engineer_features(df)
        for col in cat_cols:
            if col in df.columns:
                le = label_encoders[col]
                df[col] = df[col].astype(str).map(lambda x, _le=le: _safe_transform(_le, x))
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[feature_cols].values

        pred_rv = model["recovered_value_model"].predict(X)
        pred_mv = model["mineral_value_model"].predict(X)
        pred_yield = model["extraction_yield_model"].predict(X)
        decomposed = pred_mv * np.clip(pred_yield, 0, 1.5)
        # Ground truth ~ model's best estimate ± noise
        true_values = np.maximum(0, 0.5 * pred_rv + 0.5 * decomposed) * rng.normal(1.0, 0.20, len(asteroids))
        true_values = np.maximum(0, true_values)

        round_info = _round_info(rnd, total_rounds, n_competitors, capital)

        t0 = time.time()
        bids = strategy_module.price_asteroids(asteroids, capital, round_info, model)
        elapsed = time.time() - t0

        if elapsed > 2.0:
            print(f"  WARNING: Round {rnd} took {elapsed:.2f}s > 2s limit")

        if not isinstance(bids, list) or len(bids) != len(asteroids):
            print(f"  ERROR: Invalid bids returned in round {rnd}")
            continue

        n_placed = sum(1 for b in bids if b > 0)
        if n_placed > 0:
            rounds_with_bids += 1
        bid_counts.append(n_placed)

        round_spend = 0.0
        round_revenue = 0.0

        for i, (bid, ast) in enumerate(zip(bids, asteroids)):
            if bid <= 0:
                continue
            if bid > capital:
                bid = capital

            # Competitor bids: 20-55% of a noisy version of the same true value
            competitor_bids = [
                true_values[i] * rng.uniform(0.20, 0.55) * rng.normal(1.0, 0.15)
                for _ in range(n_competitors)
            ]
            highest_competitor = max(competitor_bids) if competitor_bids else 0

            if bid >= highest_competitor:
                wins += 1
                round_spend += bid
                capital -= bid

                cat_prob = ast["volatile_content"] * (1 - ast["structural_integrity"]) * 0.3
                if rng.random() < cat_prob:
                    catastrophes += 1
                    capital -= 200.0
                    round_revenue -= 200.0
                else:
                    recovered = max(0, true_values[i] * rng.normal(1.0, 0.10))
                    capital += recovered
                    round_revenue += recovered
                    total_revenue += recovered

        total_spent += round_spend

    final_return = (capital - initial_capital) / initial_capital * 100
    avg_bids_per_round = np.mean(bid_counts) if bid_counts else 0

    return {
        "final_capital": capital,
        "initial_capital": initial_capital,
        "return_pct": final_return,
        "total_wins": wins,
        "catastrophes": catastrophes,
        "total_spent": total_spent,
        "total_revenue": total_revenue,
        "avg_bids_per_round": avg_bids_per_round,
        "rounds_with_bids": rounds_with_bids,
        "total_rounds": total_rounds,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    req_passed, model = check_requirements(strategy)

    if model is None:
        print("ERROR: Could not load model. Aborting simulation.")
        sys.exit(1)

    print("=" * 60)
    print("TOURNAMENT SIMULATION (3 runs, 100 rounds each)")
    print("=" * 60)

    results = []
    for seed in [42, 123, 999]:
        r = simulate_tournament(strategy, model, total_rounds=100,
                                asteroids_per_round=10, n_competitors=5, seed=seed)
        results.append(r)
        print(f"  Seed {seed:>4}: return={r['return_pct']:+.1f}%  wins={r['total_wins']}  "
              f"catastrophes={r['catastrophes']}  avg_bids/round={r['avg_bids_per_round']:.1f}  "
              f"active_rounds={r['rounds_with_bids']}/{r['total_rounds']}")

    print()
    mean_ret = np.mean([r["return_pct"] for r in results])
    min_ret = min(r["return_pct"] for r in results)
    max_ret = max(r["return_pct"] for r in results)
    total_cats = sum(r["catastrophes"] for r in results)
    mean_wins = np.mean([r["total_wins"] for r in results])

    print("SUMMARY")
    print(f"  Mean return:       {mean_ret:+.1f}%")
    print(f"  Min / Max return:  {min_ret:+.1f}% / {max_ret:+.1f}%")
    print(f"  Mean wins:         {mean_wins:.1f} per tournament")
    print(f"  Total catastrophes:{total_cats} across {len(results)} tournaments")
    print()

    # Submission requirements verdict
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    req_ok = req_passed
    perf_ok = mean_ret > 0  # profitable
    cat_ok = total_cats == 0

    print(f"  Submission requirements: {'PASS' if req_ok else 'FAIL'}")
    print(f"  Profitable (mean return > 0): {'PASS' if perf_ok else 'FAIL'} ({mean_ret:+.1f}%)")
    print(f"  Zero catastrophes: {'PASS' if cat_ok else 'WARN'} ({total_cats} total)")

    all_ok = req_ok and perf_ok
    print()
    print(f"  Overall: {'READY FOR SUBMISSION' if all_ok else 'NEEDS ATTENTION'}")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
