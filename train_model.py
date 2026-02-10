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

from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trainer")

FEATURES = [
    "initial_buy_sol",
    "initial_price_usd",
    "initial_mcap_usd",
    "v_sol_in_bonding",
    "v_tokens_in_bonding",
    "total_buys",
    "total_sells",
    "total_buy_sol",
    "total_sell_sol",
    "buy_sell_ratio",
    "sell_pressure",
    "unique_buyers",
    "unique_sellers",
    "dex_liquidity_usd",
    "dex_volume_5m",
    "dex_volume_1h",
    "dex_buys_5m",
    "dex_sells_5m",
    "dex_buys_1h",
    "dex_sells_1h",
    "dex_market_cap",
    "dex_fdv",
    "has_website",
    "has_socials",
    "migrated",
]

LABEL_MAP = {
    "ROCKET": 3,
    "winner": 2,
    "good": 1,
    "flat": 0,
    "loser": 0,
    "dead": 0,
}


def load_data(data_dir: str | None = None) -> pd.DataFrame:
    directory = data_dir or DATA_DIR
    files = sorted(glob.glob(os.path.join(directory, "collect_*.json")))
    files += sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not files:
        log.error("No data files found in %s", directory)
        sys.exit(1)

    all_tokens = []
    for fp in files:
        with open(fp) as f:
            if fp.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    tokens = data.get("tokens", [])
                    log.info("Loaded %d tokens from %s", len(tokens), os.path.basename(fp))
                    all_tokens.extend(tokens)
            else:
                data = json.load(f)
                tokens = data.get("tokens", [])
                log.info("Loaded %d tokens from %s", len(tokens), os.path.basename(fp))
                all_tokens.extend(tokens)

    log.info("Total tokens loaded: %d", len(all_tokens))

    df = pd.DataFrame(all_tokens)

    if "best_change_pct" in df.columns:
        log.info("Re-deriving outcomes from best_change_pct with new thresholds...")
        df["outcome"] = df["best_change_pct"].apply(
            lambda x: "ROCKET" if x >= 500
            else "winner" if x >= 100
            else "good" if x >= 50
            else "flat" if x >= -15
            else "loser" if x >= -50
            else "dead"
        )

    df = df[df["outcome"].isin(LABEL_MAP.keys())].copy()
    log.info("Tokens with valid outcome: %d", len(df))

    if "total_buys" in df.columns:
        before = len(df)
        df = df[df["total_buys"] >= 1].copy()
        log.info("Tokens with at least 1 buy: %d / %d", len(df), before)

    df["label"] = df["outcome"].map(LABEL_MAP)

    for col in ["has_website", "has_socials", "migrated"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def analyze_patterns(df: pd.DataFrame):
    log.info("=" * 60)
    log.info("PATTERN ANALYSIS")
    log.info("=" * 60)

    outcome_counts = df["outcome"].value_counts()
    log.info("Outcome distribution:\n%s", outcome_counts.to_string())

    rockets = df[df["outcome"] == "ROCKET"]
    winners = df[df["outcome"] == "winner"]
    good = df[df["outcome"] == "good"]
    trash = df[df["outcome"].isin(["flat", "loser", "dead"])]

    log.info("\n--- ROCKET vs TRASH comparison ---")
    compare_cols = [
        "initial_buy_sol", "initial_mcap_usd", "total_buys", "total_sells",
        "buy_sell_ratio", "sell_pressure", "dex_liquidity_usd",
        "dex_volume_5m", "migrated",
    ]

    for col in compare_cols:
        r_mean = rockets[col].mean() if len(rockets) > 0 else 0
        t_mean = trash[col].mean() if len(trash) > 0 else 0
        log.info(
            "  %s: ROCKET=%.4f  TRASH=%.4f  (%.1fx)",
            col.ljust(25), r_mean, t_mean,
            r_mean / max(t_mean, 0.0001),
        )

    if len(rockets) > 0:
        log.info("\n--- ROCKET tokens details ---")
        for _, r in rockets.iterrows():
            log.info(
                "  %s: buys=%d sells=%d ratio=%.1f sell_pres=%.0f%% mcap=$%.0f change=%.0f%%",
                r["symbol"], r["total_buys"], r["total_sells"],
                r["buy_sell_ratio"], r["sell_pressure"],
                r["initial_mcap_usd"], r.get("best_change_pct", 0) or 0,
            )

    log.info("\n--- Quick rules analysis ---")
    if len(rockets) > 0 and len(trash) > 0:
        buy_threshold = rockets["total_buys"].quantile(0.25)
        sell_pressure_threshold = rockets["sell_pressure"].quantile(0.75)
        log.info("  Rule 1: total_buys >= %.0f (catches 75%% of rockets)", buy_threshold)
        log.info("  Rule 2: sell_pressure <= %.0f%% (catches 75%% of rockets)", sell_pressure_threshold)

        rule_hits = df[
            (df["total_buys"] >= buy_threshold) &
            (df["sell_pressure"] <= sell_pressure_threshold)
        ]
        rule_rockets = rule_hits[rule_hits["outcome"] == "ROCKET"]
        precision = len(rule_rockets) / max(1, len(rule_hits)) * 100
        recall = len(rule_rockets) / max(1, len(rockets)) * 100
        log.info("  Combined rule: precision=%.0f%% recall=%.0f%%", precision, recall)


def train(df: pd.DataFrame, model_type: str = "rf"):
    log.info("=" * 60)
    log.info("ML TRAINING")
    log.info("=" * 60)

    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].values
    y = df["label"].values

    log.info("Features: %d | Samples: %d", len(available_features), len(X))
    log.info("Label distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    if len(np.unique(y)) < 2:
        log.warning("Not enough label variety to train. Need at least 2 classes.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
        )

    if len(X) >= 20:
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // 4), scoring="f1_weighted")
        log.info("Cross-val F1 (weighted): %.3f (+/- %.3f)", cv_scores.mean(), cv_scores.std())

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) >= 2 else None,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(np.unique(y))
    label_names = {0: "trash", 1: "good", 2: "winner", 3: "ROCKET"}
    target_names = [label_names.get(l, f"class_{l}") for l in labels]

    log.info("\nClassification Report:\n%s", classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    log.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    log.info("\nTop 10 Feature Importances:")
    for i in range(min(10, len(sorted_idx))):
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
        "label_map": {v: k for k, v in LABEL_MAP.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Model saved: %s", model_path)
    log.info("Scaler saved: %s", scaler_path)

    return model, scaler, available_features


def predict_token(model, scaler, features: list[str], token_data: dict) -> str:
    values = [token_data.get(f, 0) for f in features]
    for i, f in enumerate(features):
        if f in ("has_website", "has_socials", "migrated"):
            values[i] = int(values[i])
    X = np.array([values])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    label = {0: "trash", 1: "good", 2: "winner", 3: "ROCKET"}.get(pred, "unknown")
    confidence = max(proba) * 100
    return f"{label} ({confidence:.0f}%)"


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(data_dir)

    analyze_patterns(df)

    model, scaler, features = train(df)

    if model is not None:
        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info("=" * 60)
    else:
        log.info("Not enough data to train. Collect more tokens!")


if __name__ == "__main__":
    main()
