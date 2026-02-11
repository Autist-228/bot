import json
import glob
import os
import sys
import logging
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

BINARY_MODE = True

LABEL_MAP = {"ROCKET": 3, "winner": 2, "good": 1, "trash": 0}

COST_BUY = (1 - PUMPFUN_FEE_PCT) * (1 - BUY_SLIPPAGE_PCT)
COST_SELL = (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT)
COST_ROUNDTRIP = COST_BUY * COST_SELL

ENTRY_SNAPSHOT = "60s"
EXIT_SNAPSHOTS = ["120s", "300s", "600s", "900s"]
ENTRY_AGE = 60


def simulate_trade(token):
    snapshots = token.get("snapshots", {})
    entry_snap = snapshots.get(ENTRY_SNAPSHOT, {})
    entry_price = entry_snap.get("price_usd", 0)
    if entry_price <= 0:
        return -100.0, "NO_ENTRY_PRICE"
    peak_gain = 0.0
    last_gain = 0.0
    for snap_key in EXIT_SNAPSHOTS:
        snap = snapshots.get(snap_key, {})
        price = snap.get("price_usd", 0)
        if price <= 0:
            continue
        gross_change_pct = (price / entry_price - 1) * 100
        net_gain = (COST_ROUNDTRIP * (1 + gross_change_pct / 100) - 1) * 100
        last_gain = net_gain
        if net_gain > peak_gain:
            peak_gain = net_gain
        snap_age = snap.get("age_sec", 0)
        elapsed = snap_age - ENTRY_AGE
        if net_gain <= STOP_LOSS_PCT:
            return net_gain, "STOP_LOSS"
        if elapsed >= TIME_STOP_SEC and net_gain < TIME_STOP_MIN_GAIN:
            return net_gain, "TIME_STOP"
        if peak_gain >= 30.0 and (peak_gain - net_gain) >= TRAILING_STOP_PCT:
            return net_gain, "TRAILING_STOP"
        if peak_gain >= 15.0 and net_gain < 0:
            return net_gain, "PROFIT_GONE"
    return last_gain, "HOLD_END"


def pnl_to_label(pnl):
    if pnl >= 30:
        return "ROCKET"
    if pnl >= 10:
        return "winner"
    if pnl >= 0:
        return "good"
    return "trash"


def extract_early_features(token):
    snapshots = token.get("snapshots", {})
    p15 = snapshots.get("15s", {}).get("price_usd", 0)
    p30 = snapshots.get("30s", {}).get("price_usd", 0)
    mom_15_30 = ((p30 / p15) - 1) * 100 if p15 > 0 and p30 > 0 else 0.0
    snap_30 = snapshots.get("30s", {})
    buys_30 = snap_30.get("buys", 0)
    unique_buyers_30 = snap_30.get("unique_buyers", 0)
    buy_vol_30 = snap_30.get("buy_sol", 0)
    sells_30 = snap_30.get("sells", 0)
    total_trades = buys_30 + sells_30
    return {
        "log_buy_sol": np.log1p(token.get("initial_buy_sol", 0)),
        "log_mcap": np.log1p(token.get("initial_mcap_usd", 0)),
        "buy_rate": buys_30 / 30.0,
        "volume_rate": buy_vol_30 / 30.0,
        "buyer_rate": unique_buyers_30 / 30.0,
        "sell_pressure": sells_30 / max(1, total_trades) * 100,
        "momentum_15_30": mom_15_30,
        "migrated": int(token.get("migrated", 0)),
    }


def load_and_prepare(data_dir=None):
    directory = data_dir or DATA_DIR
    hist_files = sorted(glob.glob(os.path.join(directory, "historical_progress_*.json")))
    hist_files += sorted(glob.glob(os.path.join(directory, "historical_12h_*.json")))
    collect_files = sorted(glob.glob(os.path.join(directory, "collect_*.json")))
    jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    all_tokens = []
    for fp in collect_files:
        with open(fp) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
        log.info("Loaded %d tokens from %s", len(tokens), os.path.basename(fp))
        all_tokens.extend(tokens)
    for fp in jsonl_files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tokens = data.get("tokens", [])
                log.info("Loaded %d tokens from %s", len(tokens), os.path.basename(fp))
                all_tokens.extend(tokens)
    for fp in hist_files:
        with open(fp) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
        log.info("Loaded %d tokens from %s", len(tokens), os.path.basename(fp))
        all_tokens.extend(tokens)
    log.info("Total raw tokens loaded: %d", len(all_tokens))
    seen_mints = set()
    unique_tokens = []
    for t in all_tokens:
        mint = t.get("mint", "")
        if mint and mint not in seen_mints:
            seen_mints.add(mint)
            unique_tokens.append(t)
    log.info("Unique tokens (deduped): %d", len(unique_tokens))
    rows = []
    sim_stats = {"total": 0, "no_entry": 0, "simulated": 0}
    exit_reasons = {}
    pnl_labels = {"ROCKET": 0, "winner": 0, "good": 0, "trash": 0}
    for token in unique_tokens:
        snapshots = token.get("snapshots", {})
        if not snapshots or "60s" not in snapshots:
            sim_stats["no_entry"] += 1
            continue
        entry_price = snapshots["60s"].get("price_usd", 0)
        if entry_price <= 0:
            sim_stats["no_entry"] += 1
            continue
        snap_30 = snapshots.get("30s", {})
        if "buys" not in snap_30:
            sim_stats["no_entry"] += 1
            continue
        sim_stats["total"] += 1
        pnl, reason = simulate_trade(token)
        sim_stats["simulated"] += 1
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        outcome = pnl_to_label(pnl)
        pnl_labels[outcome] += 1
        features = extract_early_features(token)
        features["sim_pnl"] = pnl
        features["sim_exit_reason"] = reason
        features["outcome"] = outcome
        features["mint"] = token.get("mint", "")
        features["symbol"] = token.get("symbol", "")
        features["created_ts"] = token.get("created_ts", 0)
        rows.append(features)
    log.info("Simulation stats: %s", sim_stats)
    log.info("Exit reasons: %s", exit_reasons)
    log.info("P&L labels: %s", pnl_labels)
    df = pd.DataFrame(rows)
    if BINARY_MODE:
        df["label"] = (df["sim_pnl"] > 0).astype(int)
    else:
        df["label"] = df["outcome"].map(LABEL_MAP)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def analyze_patterns(df):
    log.info("=" * 60)
    log.info("PATTERN ANALYSIS (simulation-based labels)")
    log.info("=" * 60)
    outcome_counts = df["outcome"].value_counts()
    log.info("Outcome distribution:\n%s", outcome_counts.to_string())
    rockets = df[df["outcome"] == "ROCKET"]
    trash = df[df["outcome"] == "trash"]
    log.info("\n--- ROCKET vs TRASH feature comparison ---")
    compare_cols = [
        "log_buy_sol", "log_mcap", "buy_rate", "volume_rate",
        "buyer_rate", "sell_pressure",
        "momentum_15_30", "migrated",
    ]
    for col in compare_cols:
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


def train(df, model_type="rf"):
    log.info("=" * 60)
    log.info("ML TRAINING (simulation-based)")
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
    labels = sorted(np.unique(y))
    if BINARY_MODE:
        target_names = ["unprofitable", "profitable"]
    else:
        label_names = {0: "trash", 1: "good", 2: "winner", 3: "ROCKET"}
        target_names = [label_names.get(l, f"class_{l}") for l in labels]
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
        "binary_mode": BINARY_MODE,
        "label_map": {"0": "unprofitable", "1": "profitable"} if BINARY_MODE else {v: k for k, v in LABEL_MAP.items()},
        "sim_config": {
            "entry_snapshot": ENTRY_SNAPSHOT,
            "cost_roundtrip": round(COST_ROUNDTRIP, 4),
            "stop_loss": STOP_LOSS_PCT,
            "trailing_stop": TRAILING_STOP_PCT,
            "time_stop_sec": TIME_STOP_SEC,
            "time_stop_min_gain": TIME_STOP_MIN_GAIN,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Model saved: %s", model_path)
    log.info("Scaler saved: %s", scaler_path)
    return model, scaler, available_features


def backtest(df, model, scaler, features):
    log.info("=" * 60)
    log.info("BACKTEST (time-based split)")
    log.info("=" * 60)
    if "created_ts" not in df.columns or df["created_ts"].max() == 0:
        log.info("No timestamp data for time-based backtest, skipping")
        return
    df_sorted = df.sort_values("created_ts").copy()
    split_idx = int(len(df_sorted) * 0.9)
    test_df = df_sorted.iloc[split_idx:]
    log.info("Train: %d tokens | Test: %d tokens (last 10%%)", split_idx, len(test_df))
    X_test = test_df[features].values
    X_scaled = scaler.transform(X_test)
    y_true = test_df["label"].values
    y_pred = model.predict(X_scaled)
    labels = sorted(np.unique(y_true))
    if BINARY_MODE:
        target_names = ["unprofitable", "profitable"]
    else:
        label_names = {0: "trash", 1: "good", 2: "winner", 3: "ROCKET"}
        target_names = [label_names.get(l, f"class_{l}") for l in labels]
    log.info("\nBacktest Classification Report:\n%s",
             classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    actual_pnls = test_df["sim_pnl"].values
    if BINARY_MODE:
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
    else:
        for pred_label, pred_name in [(3, "ROCKET"), (2, "winner")]:
            mask = y_pred >= pred_label if pred_label == 2 else y_pred == pred_label
            if mask.sum() > 0:
                pnls = actual_pnls[mask]
                wins = (pnls > 0).sum()
                total = len(pnls)
                avg_pnl = pnls.mean()
                total_pnl_usd = sum(BET_SIZE_USD * p / 100 for p in pnls)
                name = pred_name if pred_label == 3 else "ROCKET+winner"
                log.info(
                    "%s signals: %d | Win rate: %.1f%% (%d/%d) | Avg P&L: %+.1f%% | Total: $%+.2f",
                    name, total, wins / total * 100, wins, total, avg_pnl, total_pnl_usd,
                )


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
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
    else:
        log.info("Not enough data to train!")


if __name__ == "__main__":
    main()
