import json
import glob
import os
import sys
import logging
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATA_DIR,
    PUMPFUN_FEE_PCT,
    BUY_SLIPPAGE_PCT,
    SELL_SLIPPAGE_PCT,
    STOP_LOSS_PCT,
    TRAILING_STOP_PCT,
    TIME_STOP_SEC,
    TIME_STOP_MIN_GAIN,
    BET_SIZE_USD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trainer")

FEATURES = [
    "log_buy_sol",
    "log_mcap",
    "buy_rate",
    "volume_rate",
    "buyer_rate",
    "sell_pressure",
    "momentum_15_30",
    "migrated",
]

COST_BUY = (1 - PUMPFUN_FEE_PCT) * (1 - BUY_SLIPPAGE_PCT)
COST_SELL = (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT)

ENTRY_AGE_MIN = 30
ENTRY_AGE_MAX = 120
MAX_HOLD_SEC = 900
SOL_PRICE_USD = 200.0


def simulate_trade_from_trades(token):
    trades = token.get("trades", [])
    if not trades or len(trades) < 3:
        return None

    created_ts = token.get("created_ts", 0)
    if not created_ts:
        created_ts = trades[0]["ts"]

    priced = [t for t in trades if t.get("price_sol", 0) > 0]
    if len(priced) < 3:
        return None

    entry_candidates = [
        t for t in priced
        if ENTRY_AGE_MIN <= (t["ts"] - created_ts) <= ENTRY_AGE_MAX
        and t["type"] == "buy"
    ]
    if not entry_candidates:
        entry_candidates = [
            t for t in priced
            if ENTRY_AGE_MIN <= (t["ts"] - created_ts) <= ENTRY_AGE_MAX
        ]
    if not entry_candidates:
        return None

    entry_trade = random.choice(entry_candidates)
    entry_ts = entry_trade["ts"]
    entry_age = entry_ts - created_ts
    entry_price_sol = entry_trade["price_sol"]

    if entry_price_sol <= 0:
        return None

    entry_cost_pct = (COST_BUY * COST_SELL - 1) * 100

    peak_gain = 0.0
    last_gain = entry_cost_pct
    exit_reason = "HOLD_END"

    post_entry = [t for t in priced if t["ts"] > entry_ts and t["ts"] <= entry_ts + MAX_HOLD_SEC]

    for t in post_entry:
        cur_price = t["price_sol"]
        if cur_price <= 0:
            continue

        elapsed = t["ts"] - entry_ts
        gross_change = (cur_price / entry_price_sol - 1) * 100
        gain = (COST_BUY * COST_SELL * (1 + gross_change / 100) - 1) * 100

        if gain > peak_gain:
            peak_gain = gain

        last_gain = gain

        if gain <= STOP_LOSS_PCT:
            return {
                "pnl": gain, "reason": "STOP_LOSS", "entry_age": entry_age,
                "hold_sec": elapsed, "peak_gain": peak_gain, "entry_ts": entry_ts,
            }

        if elapsed >= TIME_STOP_SEC and gain < TIME_STOP_MIN_GAIN:
            return {
                "pnl": gain, "reason": "TIME_STOP", "entry_age": entry_age,
                "hold_sec": elapsed, "peak_gain": peak_gain, "entry_ts": entry_ts,
            }

        if peak_gain >= 30.0 and (peak_gain - gain) >= TRAILING_STOP_PCT:
            return {
                "pnl": gain, "reason": "TRAILING_STOP", "entry_age": entry_age,
                "hold_sec": elapsed, "peak_gain": peak_gain, "entry_ts": entry_ts,
            }

        if peak_gain >= 15.0 and gain < 0:
            return {
                "pnl": gain, "reason": "PROFIT_GONE", "entry_age": entry_age,
                "hold_sec": elapsed, "peak_gain": peak_gain, "entry_ts": entry_ts,
            }

    return {
        "pnl": last_gain, "reason": exit_reason, "entry_age": entry_age,
        "hold_sec": post_entry[-1]["ts"] - entry_ts if post_entry else 0,
        "peak_gain": peak_gain, "entry_ts": entry_ts,
    }


def extract_features_from_trades(token, entry_ts):
    trades = token.get("trades", [])
    created_ts = token.get("created_ts", 0)
    if not created_ts:
        created_ts = trades[0]["ts"] if trades else 0

    entry_age = entry_ts - created_ts
    pre_entry = [t for t in trades if t["ts"] <= entry_ts]

    buys = [t for t in pre_entry if t["type"] == "buy"]
    sells = [t for t in pre_entry if t["type"] == "sell"]

    n_buys = len(buys)
    n_sells = len(sells)
    total = n_buys + n_sells
    buy_sol = sum(t.get("sol", 0) for t in buys)
    unique_buyers = len(set(t.get("trader", "") for t in buys))

    first_buy_sol = buys[0]["sol"] if buys else 0
    first_price = buys[0].get("price_sol", 0) if buys else 0
    initial_mcap = first_price * SOL_PRICE_USD * 1e9 if first_price > 0 else 0

    half_age = entry_age / 2
    mid_ts = created_ts + half_age
    prices_first = [t["price_sol"] for t in pre_entry if t.get("price_sol", 0) > 0 and t["ts"] <= mid_ts]
    prices_second = [t["price_sol"] for t in pre_entry if t.get("price_sol", 0) > 0 and t["ts"] > mid_ts]

    p_first = np.median(prices_first) if prices_first else 0
    p_second = np.median(prices_second) if prices_second else 0
    momentum = ((p_second / p_first) - 1) * 100 if p_first > 0 and p_second > 0 else 0.0

    age = max(1, entry_age)

    return {
        "log_buy_sol": np.log1p(first_buy_sol),
        "log_mcap": np.log1p(initial_mcap),
        "buy_rate": n_buys / age,
        "volume_rate": buy_sol / age,
        "buyer_rate": unique_buyers / age,
        "sell_pressure": n_sells / max(1, total) * 100,
        "momentum_15_30": momentum,
        "migrated": int(token.get("migrated", 0)),
    }


def pnl_to_label(pnl):
    if pnl >= 30:
        return "ROCKET"
    if pnl >= 10:
        return "winner"
    if pnl >= 0:
        return "good"
    return "trash"


def load_and_prepare(data_dir=None):
    directory = data_dir or DATA_DIR

    hist_files = sorted(glob.glob(os.path.join(directory, "historical_36h_*.json")))
    if not hist_files:
        hist_files = sorted(glob.glob(os.path.join(directory, "historical_*h_*.json")))

    collect_files = sorted(glob.glob(os.path.join(directory, "collect_*.json")))
    all_tokens = []

    for fp in hist_files:
        with open(fp) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
        has_trades = sum(1 for t in tokens if t.get("trades"))
        log.info("Loaded %d tokens (%d with trades) from %s", len(tokens), has_trades, os.path.basename(fp))
        all_tokens.extend(tokens)

    for fp in collect_files:
        with open(fp) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
        has_trades = sum(1 for t in tokens if t.get("trades"))
        log.info("Loaded %d tokens (%d with trades) from %s", len(tokens), has_trades, os.path.basename(fp))
        all_tokens.extend(tokens)

    log.info("Total raw tokens loaded: %d", len(all_tokens))

    seen_mints = set()
    unique_tokens = []
    for t in all_tokens:
        mint = t.get("mint", "")
        if mint and mint not in seen_mints:
            seen_mints.add(mint)
            unique_tokens.append(t)

    with_trades = [t for t in unique_tokens if t.get("trades") and len(t["trades"]) >= 3]
    log.info("Unique tokens: %d | With trades (>=3): %d", len(unique_tokens), len(with_trades))

    if not with_trades:
        log.error("No tokens with trades array! Run historical_collector.py --save-trades first.")
        return pd.DataFrame()

    rows = []
    sim_stats = {"total": 0, "no_entry": 0, "simulated": 0}
    exit_reasons = {}
    pnl_labels = {"ROCKET": 0, "winner": 0, "good": 0, "trash": 0}

    for token in with_trades:
        sim_stats["total"] += 1
        result = simulate_trade_from_trades(token)
        if result is None:
            sim_stats["no_entry"] += 1
            continue

        sim_stats["simulated"] += 1
        pnl = result["pnl"]
        reason = result["reason"]
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        outcome = pnl_to_label(pnl)
        pnl_labels[outcome] += 1

        features = extract_features_from_trades(token, result["entry_ts"])
        features["sim_pnl"] = pnl
        features["sim_exit_reason"] = reason
        features["sim_entry_age"] = result["entry_age"]
        features["sim_hold_sec"] = result["hold_sec"]
        features["sim_peak_gain"] = result["peak_gain"]
        features["outcome"] = outcome
        features["mint"] = token.get("mint", "")
        features["symbol"] = token.get("symbol", "")
        features["created_ts"] = token.get("created_ts", 0)
        rows.append(features)

    log.info("Simulation stats: %s", sim_stats)
    log.info("Exit reasons: %s", exit_reasons)
    log.info("P&L labels: %s", pnl_labels)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["label"] = (df["sim_pnl"] > 0).astype(int)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def analyze_patterns(df):
    log.info("=" * 60)
    log.info("PATTERN ANALYSIS (trade-by-trade simulation)")
    log.info("=" * 60)

    outcome_counts = df["outcome"].value_counts()
    log.info("Outcome distribution:\n%s", outcome_counts.to_string())

    rockets = df[df["outcome"] == "ROCKET"]
    trash = df[df["outcome"] == "trash"]

    log.info("\n--- ROCKET vs TRASH feature comparison ---")
    for col in FEATURES:
        r_med = rockets[col].median() if len(rockets) > 0 else 0
        t_med = trash[col].median() if len(trash) > 0 else 0
        log.info("  %s: ROCKET_med=%.4f  TRASH_med=%.4f", col.ljust(25), r_med, t_med)

    log.info("\n--- Simulated P&L stats ---")
    for grp_name in ["ROCKET", "winner", "good", "trash"]:
        grp = df[df["outcome"] == grp_name]
        if len(grp) > 0:
            pnl = grp["sim_pnl"]
            log.info(
                "  %s: count=%d mean_pnl=%+.1f%% median=%+.1f%% min=%+.1f%% max=%+.1f%%",
                grp_name.ljust(8), len(grp), pnl.mean(), pnl.median(), pnl.min(), pnl.max(),
            )

    if "sim_exit_reason" in df.columns:
        log.info("\nExit reasons by outcome:")
        for outcome in ["ROCKET", "winner", "good", "trash"]:
            grp = df[df["outcome"] == outcome]
            if len(grp) > 0:
                reasons = grp["sim_exit_reason"].value_counts()
                log.info("  %s: %s", outcome, dict(reasons))

    if "sim_entry_age" in df.columns:
        log.info("\nEntry age stats: mean=%.0fs median=%.0fs min=%.0fs max=%.0fs",
                 df["sim_entry_age"].mean(), df["sim_entry_age"].median(),
                 df["sim_entry_age"].min(), df["sim_entry_age"].max())

    if "sim_hold_sec" in df.columns:
        log.info("Hold time stats: mean=%.0fs median=%.0fs",
                 df["sim_hold_sec"].mean(), df["sim_hold_sec"].median())


def train(df, model_type="rf"):
    log.info("=" * 60)
    log.info("ML TRAINING (trade-by-trade simulation)")
    log.info("=" * 60)

    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].values
    y = df["label"].values

    log.info("Features (%d): %s", len(available_features), available_features)
    log.info("Samples: %d", len(X))
    log.info("Label distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    if len(np.unique(y)) < 2:
        log.warning("Not enough label variety to train.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42,
        )

    if len(X) >= 50:
        cv_scores = cross_val_score(
            model, X_scaled, y, cv=min(5, len(X) // 10), scoring="f1_weighted",
        )
        log.info("Cross-val F1 (weighted): %.3f (+/- %.3f)", cv_scores.mean(), cv_scores.std())

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) >= 2 else None,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    target_names = ["unprofitable", "profitable"]
    log.info("\nClassification Report:\n%s",
             classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    log.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    log.info("\nFeature Importances:")
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        log.info("  %s: %.4f", available_features[idx].ljust(25), importances[idx])

    model_path = os.path.join(DATA_DIR, "model.pkl")
    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    meta_path = os.path.join(DATA_DIR, "model_meta.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "n_samples": len(X),
        "n_features": len(available_features),
        "features": available_features,
        "binary_mode": True,
        "label_map": {"0": "unprofitable", "1": "profitable"},
        "sim_config": {
            "entry_age_range": [ENTRY_AGE_MIN, ENTRY_AGE_MAX],
            "cost_buy": round(COST_BUY, 4),
            "cost_sell": round(COST_SELL, 4),
            "stop_loss": STOP_LOSS_PCT,
            "trailing_stop": TRAILING_STOP_PCT,
            "time_stop_sec": TIME_STOP_SEC,
            "time_stop_min_gain": TIME_STOP_MIN_GAIN,
            "max_hold_sec": MAX_HOLD_SEC,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Model saved: %s", model_path)
    log.info("Scaler saved: %s", scaler_path)
    return model, scaler, available_features


def backtest(df, model, scaler, features):
    log.info("=" * 60)
    log.info("BACKTEST (time-based split, last 10%%)")
    log.info("=" * 60)

    if "created_ts" not in df.columns or df["created_ts"].max() == 0:
        log.info("No timestamp data for time-based backtest, skipping")
        return

    df_sorted = df.sort_values("created_ts").copy()
    split_idx = int(len(df_sorted) * 0.9)
    test_df = df_sorted.iloc[split_idx:]

    log.info("Train period: %d tokens | Test period: %d tokens (last 10%%)", split_idx, len(test_df))

    X_test = test_df[features].values
    X_scaled = scaler.transform(X_test)
    y_true = test_df["label"].values
    y_pred = model.predict(X_scaled)

    target_names = ["unprofitable", "profitable"]
    log.info("\nBacktest Classification Report:\n%s",
             classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    actual_pnls = test_df["sim_pnl"].values
    mask = y_pred == 1
    if mask.sum() > 0:
        pnls = actual_pnls[mask]
        wins = (pnls > 0).sum()
        total = len(pnls)
        avg_pnl = pnls.mean()
        total_pnl_usd = sum(BET_SIZE_USD * p / 100 for p in pnls)
        log.info(
            "BUY signals: %d | Win rate: %.1f%% (%d/%d) | Avg P&L: %+.1f%% | Total: $%+.2f",
            total, wins / total * 100, wins, total, avg_pnl, total_pnl_usd,
        )

        for threshold in [0.5, 0.6, 0.7, 0.8]:
            proba = model.predict_proba(X_scaled)[:, 1]
            high_conf = proba >= threshold
            if high_conf.sum() > 0:
                hc_pnls = actual_pnls[high_conf]
                hc_wins = (hc_pnls > 0).sum()
                hc_total = len(hc_pnls)
                hc_pnl_usd = sum(BET_SIZE_USD * p / 100 for p in hc_pnls)
                log.info(
                    "  conf>=%.0f%%: signals=%d win=%.1f%% avg=%+.1f%% total=$%+.2f",
                    threshold * 100, hc_total, hc_wins / hc_total * 100,
                    hc_pnls.mean(), hc_pnl_usd,
                )
    else:
        log.info("No BUY signals in test set")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    random.seed(42)
    np.random.seed(42)

    df = load_and_prepare(data_dir)
    if len(df) == 0:
        log.error("No valid training data!")
        sys.exit(1)

    analyze_patterns(df)
    model, scaler, features = train(df)
    if model is not None:
        backtest(df, model, scaler, features)
        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
