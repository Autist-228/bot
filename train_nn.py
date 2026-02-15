import json
import os
import sys
import logging
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATA_DIR,
    PUMPFUN_FEE_PCT,
    BUY_SLIPPAGE_PCT,
    SELL_SLIPPAGE_PCT,
    BET_SIZE_USD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_nn")

ENTRY_FEATURES = [
    "log_buy_sol",
    "log_mcap",
    "buy_rate",
    "volume_rate",
    "buyer_rate",
    "sell_pressure",
    "momentum_15_30",
    "bonding_progress",
    "log_avg_buy_sol",
    "log_max_buy_sol",
    "buy_concentration",
    "sell_speed",
    "top5_holder_pct",
    "sniper_count",
    "num_holders",
    "log_volume_sol",
    "buy_sell_ratio",
    "whale_buy_pct",
    "dev_balance_pct",
    "price_velocity_norm",
    "log_net_sol_flow",
    "momentum_0_15",
    "large_buy_count",
    "token_age_norm",
    "bonding_curve_velocity",
    "seller_buyer_overlap",
    "sell_to_buy_sol_ratio",
    "holder_net_pct",
]

EXIT_POSITION_FEATURES = [
    "current_pnl",
    "peak_pnl",
    "time_held",
    "drop_from_peak",
    "pnl_velocity",
]

COST_BUY = (1 - PUMPFUN_FEE_PCT) * (1 - BUY_SLIPPAGE_PCT)
COST_SELL = (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT)

ENTRY_AGE_MIN = 15
ENTRY_AGE_MAX = 360
MAX_HOLD_SEC = 900
MAX_GAIN_PCT = 10000.0
SOL_PRICE_USD = 200.0

PROFITABLE_THRESHOLD = 30.0
EXIT_SAMPLE_INTERVAL = 5


BONDING_START_SOL = 30.0
BONDING_GRAD_SOL = 85.0


class EntryNet(nn.Module):
    def __init__(self, n_features=28):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.residual = nn.Linear(128, 32)
        self.head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_block(x)
        h1 = self.hidden1(x)
        h2 = self.hidden2(h1)
        h3 = self.hidden3(h2)
        res = self.residual(h1)
        return self.head(h3 + res).squeeze(-1)


class ExitNet(nn.Module):
    def __init__(self, n_features=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def calc_gain(entry_price, current_price):
    if entry_price <= 0 or current_price <= 0:
        return -100.0
    gross = (current_price / entry_price - 1) * 100
    gain = (COST_BUY * COST_SELL * (1 + gross / 100) - 1) * 100
    return float(np.clip(gain, -100.0, MAX_GAIN_PCT))


def extract_entry_features(token, entry_ts):
    trades = token.get("trades", [])
    created_ts = token.get("created_ts", 0)
    if not created_ts and trades:
        created_ts = trades[0]["ts"]

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
    momentum = max(-500.0, min(500.0, momentum))

    age = max(1, entry_age)

    sell_sol = sum(t.get("sol", 0) for t in sells)
    net_sol_in = buy_sol - sell_sol
    estimated_v_sol = BONDING_START_SOL + max(0, net_sol_in)
    bonding_progress = max(0.0, min(100.0, (estimated_v_sol - BONDING_START_SOL) / (BONDING_GRAD_SOL - BONDING_START_SOL) * 100))

    avg_buy_sol = buy_sol / max(1, n_buys)
    max_buy_sol = max((t.get("sol", 0) for t in buys), default=0)
    buy_conc = unique_buyers / max(1, n_buys)
    s_speed = n_sells / age

    snipers = set()
    for t in buys:
        if t["ts"] - created_ts <= 5:
            tr = t.get("trader", "")
            if tr:
                snipers.add(tr)
    sniper_count = len(snipers)

    buyer_totals = {}
    buyer_amounts = {}
    for t in pre_entry:
        tr = t.get("trader", "")
        if not tr:
            continue
        if t["type"] == "buy":
            buyer_totals[tr] = buyer_totals.get(tr, 0) + t.get("sol", 0)
            buyer_amounts[tr] = buyer_amounts.get(tr, 0) + t.get("sol", 0)
        else:
            buyer_totals[tr] = buyer_totals.get(tr, 0) - t.get("sol", 0)
    positive = {k: v for k, v in buyer_totals.items() if v > 0}
    total_pos = sum(positive.values())
    if total_pos > 0 and len(positive) >= 5:
        top5 = sorted(positive.values(), reverse=True)[:5]
        top5_pct = sum(top5) / total_pos * 100
    elif total_pos > 0:
        top5_pct = 100.0
    else:
        top5_pct = 0.0

    whale_buy_pct = max_buy_sol / max(0.01, buy_sol) * 100

    dev_addr = token.get("dev_address", "")
    if not dev_addr and buys:
        dev_addr = buys[0].get("trader", "")
    dev_bought = sum(t.get("sol", 0) for t in buys if t.get("trader") == dev_addr) if dev_addr else 0
    dev_sold = sum(t.get("sol", 0) for t in sells if t.get("trader") == dev_addr) if dev_addr else 0
    dev_balance_pct = (dev_bought - dev_sold) / max(0.01, buy_sol) * 100
    dev_balance_pct = max(-100.0, min(100.0, dev_balance_pct))

    cur_prices = [t.get("price_sol", 0) for t in pre_entry if t.get("price_sol", 0) > 0]
    cur_price = cur_prices[-1] if cur_prices else first_price
    price_velocity_norm = 0.0
    if first_price > 0 and cur_price > 0:
        price_velocity_norm = ((cur_price / first_price) - 1) / age * 100
        price_velocity_norm = max(-100.0, min(100.0, price_velocity_norm))

    prices_0_15 = [t["price_sol"] for t in pre_entry if t.get("price_sol", 0) > 0 and (t["ts"] - created_ts) <= 15]
    prices_15plus = [t["price_sol"] for t in pre_entry if t.get("price_sol", 0) > 0 and (t["ts"] - created_ts) > 15]
    p_0_15 = np.median(prices_0_15) if prices_0_15 else 0
    p_15plus = np.median(prices_15plus) if prices_15plus else p_0_15
    momentum_0_15 = ((p_15plus / p_0_15) - 1) * 100 if p_0_15 > 0 and p_15plus > 0 else 0.0
    momentum_0_15 = max(-500.0, min(500.0, momentum_0_15))

    large_buy_count = float(sum(1 for amt in buyer_amounts.values() if amt >= 1.0))

    buyer_set = set(t.get("trader", "") for t in buys if t.get("trader"))
    seller_set = set(t.get("trader", "") for t in sells if t.get("trader"))
    overlap = len(seller_set & buyer_set)
    seller_buyer_overlap = overlap / max(1, len(seller_set)) if seller_set else 0.0

    holder_net_pct = len(positive) / max(1, len(buyer_set)) * 100 if buyer_set else 0.0

    return {
        "log_buy_sol": np.log1p(first_buy_sol),
        "log_mcap": np.log1p(initial_mcap),
        "buy_rate": n_buys / age,
        "volume_rate": buy_sol / age,
        "buyer_rate": unique_buyers / age,
        "sell_pressure": n_sells / max(1, total) * 100,
        "momentum_15_30": momentum,
        "bonding_progress": bonding_progress,
        "log_avg_buy_sol": np.log1p(avg_buy_sol),
        "log_max_buy_sol": np.log1p(max_buy_sol),
        "buy_concentration": buy_conc,
        "sell_speed": s_speed,
        "top5_holder_pct": top5_pct,
        "sniper_count": float(sniper_count),
        "num_holders": float(unique_buyers),
        "log_volume_sol": np.log1p(buy_sol),
        "buy_sell_ratio": n_buys / max(1, n_sells),
        "whale_buy_pct": whale_buy_pct,
        "dev_balance_pct": dev_balance_pct,
        "price_velocity_norm": price_velocity_norm,
        "log_net_sol_flow": np.log1p(max(0, net_sol_in)),
        "momentum_0_15": momentum_0_15,
        "large_buy_count": large_buy_count,
        "token_age_norm": min(entry_age / 300.0, 1.0),
        "bonding_curve_velocity": bonding_progress / age * 60,
        "seller_buyer_overlap": seller_buyer_overlap,
        "sell_to_buy_sol_ratio": sell_sol / max(0.01, buy_sol),
        "holder_net_pct": holder_net_pct,
    }


def build_price_timeline(trades, entry_ts, entry_price, max_hold=MAX_HOLD_SEC):
    post = [t for t in trades if t["ts"] > entry_ts and t["ts"] <= entry_ts + max_hold and t.get("price_sol", 0) > 0]
    if not post:
        return []

    timeline = []
    seen_times = set()
    for t in sorted(post, key=lambda x: x["ts"]):
        elapsed = int(t["ts"] - entry_ts)
        if elapsed in seen_times:
            continue
        seen_times.add(elapsed)
        gain = calc_gain(entry_price, t["price_sol"])
        timeline.append({"elapsed": elapsed, "gain": gain, "price": t["price_sol"]})

    return timeline


def simulate_trade_with_exits(timeline):
    if not timeline:
        return None, 0.0, "NO_DATA"

    peak = 0.0
    for pt in timeline:
        g = pt["gain"]
        if g > peak:
            peak = g

        if g <= -15:
            return max(g, -15.0), peak, "STOP_LOSS"

        if pt["elapsed"] >= 60 and g < 10:
            return g, peak, "TIME_STOP"

        if peak >= 30 and (peak - g) >= 15:
            return g, peak, "TRAILING_STOP"

        if peak >= 15 and g < 0:
            return g, peak, "PROFIT_GONE"

    return timeline[-1]["gain"], peak, "HOLD_END"


def calculate_reward_weight(peak_gain, sim_pnl):
    if sim_pnl >= 200:
        return 10.0
    if sim_pnl >= 100:
        return 7.0
    if sim_pnl >= 50:
        return 5.0
    if sim_pnl >= 20:
        return 3.0
    if sim_pnl >= 5:
        return 2.0
    if sim_pnl <= -10:
        return 2.0
    return 1.0


def generate_exit_samples(timeline, entry_features_vec, entry_conf=0.7, entry_loss_norm=0.5, last_acc=0.5):
    if len(timeline) < 3:
        return [], []

    all_gains = [pt["gain"] for pt in timeline]
    overall_peak = max(all_gains)
    peak_idx = all_gains.index(overall_peak)

    features_list = []
    labels_list = []
    running_peak = 0.0
    prev_gain = timeline[0]["gain"]

    for i, pt in enumerate(timeline):
        if i > 0 and pt["elapsed"] % EXIT_SAMPLE_INTERVAL != 0 and i != peak_idx:
            if i % 3 != 0:
                continue

        gain = pt["gain"]
        if gain > running_peak:
            running_peak = gain

        drop_from_peak = running_peak - gain
        dt = pt["elapsed"] - timeline[max(0, i - 1)]["elapsed"] if i > 0 else 1
        velocity = (gain - prev_gain) / max(1, dt) if i > 0 else 0
        prev_gain = gain

        pos_features = [
            gain / 100.0,
            running_peak / 100.0,
            min(pt["elapsed"] / MAX_HOLD_SEC, 1.0),
            drop_from_peak / 100.0,
            float(np.clip(velocity, -1, 1)),
        ]

        full_features = list(entry_features_vec) + [entry_conf, entry_loss_norm, last_acc] + pos_features

        future_window = all_gains[i + 1 : i + 20]
        if not future_window:
            sell_label = 1.0
        elif i >= peak_idx and drop_from_peak >= 10:
            sell_label = 1.0
        elif running_peak >= 20 and drop_from_peak >= 12:
            sell_label = 0.9
        elif running_peak >= 15 and gain < 0:
            sell_label = 1.0
        elif max(future_window) > gain + 5:
            sell_label = 0.0
        elif max(future_window) < gain - 3:
            sell_label = 0.8
        elif gain >= overall_peak * 0.85 and gain > 10:
            sell_label = 0.7
        else:
            sell_label = 0.2

        features_list.append(full_features)
        labels_list.append(sell_label)

    return features_list, labels_list


def process_token(token):
    trades = token.get("trades", [])
    if not trades or len(trades) < 5:
        return None

    created_ts = token.get("created_ts", 0)
    if not created_ts:
        created_ts = trades[0]["ts"]

    priced = [t for t in trades if t.get("price_sol", 0) > 0]
    if len(priced) < 5:
        return None

    entry_candidates = [
        t
        for t in priced
        if ENTRY_AGE_MIN <= (t["ts"] - created_ts) <= ENTRY_AGE_MAX and t["type"] == "buy"
    ]
    if not entry_candidates:
        entry_candidates = [
            t for t in priced if ENTRY_AGE_MIN <= (t["ts"] - created_ts) <= ENTRY_AGE_MAX
        ]
    if not entry_candidates:
        return None

    entry_trade = random.choice(entry_candidates)
    entry_ts = entry_trade["ts"]
    entry_price = entry_trade["price_sol"]
    if entry_price <= 0:
        return None

    features = extract_entry_features(token, entry_ts)
    features_vec = [features[f] for f in ENTRY_FEATURES]

    timeline = build_price_timeline(trades, entry_ts, entry_price)
    sim_pnl, peak_gain, exit_reason = simulate_trade_with_exits(timeline)

    if sim_pnl is None:
        return None

    is_profitable = 1.0 if sim_pnl >= PROFITABLE_THRESHOLD else 0.0
    reward_weight = calculate_reward_weight(peak_gain, sim_pnl)

    exit_features, exit_labels = generate_exit_samples(timeline, features_vec)

    return {
        "entry_features": features_vec,
        "entry_label": is_profitable,
        "reward_weight": reward_weight,
        "sim_pnl": sim_pnl,
        "peak_gain": peak_gain,
        "exit_reason": exit_reason,
        "exit_features": exit_features,
        "exit_labels": exit_labels,
        "created_ts": created_ts,
        "symbol": token.get("symbol", ""),
        "mint": token.get("mint", ""),
        "entry_features_vec": features_vec,
    }


def load_data(data_file=None):
    if data_file and os.path.isfile(data_file):
        log.info("Loading %s...", data_file)
        with open(data_file) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
    else:
        path = os.path.join(DATA_DIR, "historical_12h_enriched.json")
        log.info("Loading %s...", path)
        with open(path) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])

    with_trades = [t for t in tokens if t.get("trades") and len(t["trades"]) >= 5]
    log.info("Total: %d tokens, %d with trades (>=5)", len(tokens), len(with_trades))
    return with_trades


def prepare_datasets(tokens, train_ratio=0.833):
    sorted_tokens = sorted(tokens, key=lambda t: t.get("created_ts", 0))
    split_idx = int(len(sorted_tokens) * train_ratio)
    train_tokens = sorted_tokens[:split_idx]
    test_tokens = sorted_tokens[split_idx:]
    log.info(
        "Split: %d train / %d test (%.0f%% / %.0f%%)",
        len(train_tokens),
        len(test_tokens),
        train_ratio * 100,
        (1 - train_ratio) * 100,
    )
    return train_tokens, test_tokens


def build_samples(tokens):
    entry_X, entry_y, entry_w = [], [], []
    exit_X, exit_y = [], []
    stats = {"total": 0, "profitable": 0, "peak50": 0, "peak20": 0, "peak5": 0}

    for token in tokens:
        result = process_token(token)
        if result is None:
            continue

        entry_X.append(result["entry_features"])
        entry_y.append(result["entry_label"])
        entry_w.append(result["reward_weight"])
        stats["total"] += 1

        if result["entry_label"] > 0:
            stats["profitable"] += 1
        pk = result["peak_gain"]
        if pk >= 50:
            stats["peak50"] += 1
        elif pk >= 20:
            stats["peak20"] += 1
        elif pk >= 5:
            stats["peak5"] += 1

        exit_X.extend(result["exit_features"])
        exit_y.extend(result["exit_labels"])

    log.info(
        "Samples: %d total, %d profitable (%.1f%%)",
        stats["total"],
        stats["profitable"],
        stats["profitable"] / max(1, stats["total"]) * 100,
    )
    log.info("Peaks: %d@50%%+ | %d@20%%+ | %d@5%%+", stats["peak50"], stats["peak20"], stats["peak5"])
    log.info("Exit samples: %d", len(exit_X))

    n_exit_features = len(ENTRY_FEATURES) + 3 + len(EXIT_POSITION_FEATURES)
    return (
        np.array(entry_X, dtype=np.float32),
        np.array(entry_y, dtype=np.float32),
        np.array(entry_w, dtype=np.float32),
        np.array(exit_X, dtype=np.float32) if exit_X else np.zeros((0, n_exit_features), dtype=np.float32),
        np.array(exit_y, dtype=np.float32) if exit_y else np.zeros(0, dtype=np.float32),
    )


def normalize_features(X_train, X_test=None):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std if X_test is not None else None
    return X_train_n, X_test_n, mean, std


def train_entry_model(X_train, y_train, w_train, epochs=100, lr=1e-3, batch_size=256):
    log.info("=" * 60)
    log.info("TRAINING ENTRY NETWORK (weighted binary classification)")
    log.info("=" * 60)

    pos = y_train.sum()
    neg = len(y_train) - pos
    log.info(
        "Labels: %d positive / %d negative (%.1f%% / %.1f%%)",
        int(pos),
        int(neg),
        pos / len(y_train) * 100,
        neg / len(y_train) * 100,
    )

    n_features = X_train.shape[1]
    model = EntryNet(n_features)

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)
    w_t = torch.from_numpy(w_train)

    dataset = TensorDataset(X_t, y_t, w_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pos_weight_val = neg / max(1, pos)
    log.info("Using pos_weight=%.2f to balance classes", pos_weight_val)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for X_b, y_b, w_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            bce = nn.functional.binary_cross_entropy(pred, y_b, reduction="none")
            class_w = torch.where(y_b > 0.5, pos_weight_val, 1.0)
            loss = (w_b * class_w * bce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            log.info(
                "Epoch %d/%d | Loss: %.6f | LR: %.6f",
                epoch + 1,
                epochs,
                avg_loss,
                scheduler.get_last_lr()[0],
            )

    model._final_loss = avg_loss
    return model, optimizer


def train_exit_model(X_train, y_train, epochs=80, lr=1e-3, batch_size=512):
    log.info("=" * 60)
    log.info("TRAINING EXIT NETWORK")
    log.info("=" * 60)

    if len(X_train) == 0:
        log.warning("No exit training data!")
        return None, None

    n_features = X_train.shape[1]
    model = ExitNet(n_features)

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)

    sell_count = int((y_train > 0.5).sum())
    hold_count = int((y_train <= 0.5).sum())
    log.info("Exit labels: %d sell / %d hold", sell_count, hold_count)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = nn.functional.binary_cross_entropy(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            log.info("Epoch %d/%d | Loss: %.6f", epoch + 1, epochs, avg_loss)

    model._final_loss = avg_loss
    return model, optimizer


def backtest(entry_model, test_tokens, entry_mean, entry_std):
    log.info("=" * 60)
    log.info("BACKTEST ON TEST SET")
    log.info("=" * 60)

    test_results = []
    test_X = []

    for token in test_tokens:
        result = process_token(token)
        if result is None:
            continue
        test_results.append(result)
        test_X.append(result["entry_features"])

    if not test_X:
        log.warning("No test data!")
        return None, None

    X = np.array(test_X, dtype=np.float32)
    X_n = (X - entry_mean) / entry_std

    entry_model.eval()
    with torch.no_grad():
        preds = entry_model(torch.from_numpy(X_n)).numpy()

    total_profitable = sum(1 for r in test_results if r["sim_pnl"] >= PROFITABLE_THRESHOLD)
    log.info(
        "Test samples: %d | Profitable: %d (%.1f%%)",
        len(preds),
        total_profitable,
        total_profitable / len(preds) * 100,
    )
    log.info("Pred range: [%.3f, %.3f], mean=%.3f", preds.min(), preds.max(), preds.mean())

    for conf_pct in [0, 50, 60, 70, 80, 90]:
        threshold = conf_pct / 100.0
        mask = preds >= threshold
        n_signals = int(mask.sum())
        if n_signals == 0:
            continue

        wins = 0
        total_pnl = 0.0
        total_invested = 0.0
        peak_sum = 0.0

        for i in range(len(preds)):
            if not mask[i]:
                continue
            r = test_results[i]
            pnl = r["sim_pnl"]
            if pnl >= PROFITABLE_THRESHOLD:
                wins += 1
            total_pnl += BET_SIZE_USD * pnl / 100
            total_invested += BET_SIZE_USD
            peak_sum += r["peak_gain"]

        win_rate = wins / n_signals * 100
        avg_peak = peak_sum / n_signals

        log.info(
            "conf>=%d%%: signals=%d | win=%.1f%% | P&L=$%+.2f (invested $%.0f) | avg_peak=%+.1f%%",
            conf_pct,
            n_signals,
            win_rate,
            total_pnl,
            total_invested,
            avg_peak,
        )

    return preds, test_results


def save_models(entry_model, exit_model, entry_opt, exit_opt, entry_mean, entry_std, exit_mean, exit_std, n_entry_samples=0, n_exit_samples=0):
    log.info("=" * 60)
    log.info("SAVING MODELS")
    log.info("=" * 60)

    entry_path = os.path.join(DATA_DIR, "entry_model.pt")
    exit_path = os.path.join(DATA_DIR, "exit_model.pt")
    meta_path = os.path.join(DATA_DIR, "nn_meta.json")

    entry_state = {
        "model": entry_model.state_dict(),
        "optimizer": entry_opt.state_dict(),
        "mean": entry_mean.tolist(),
        "std": entry_std.tolist(),
        "n_features": len(ENTRY_FEATURES),
        "model_info": {
            "cycles": 1,
            "loss": getattr(entry_model, "_final_loss", 0),
            "initial_loss": 0,
            "total_samples": n_entry_samples,
            "total_wins": 0,
            "rockets_found": 0, "rockets_missed": 0,
            "last_train_ts": time.time(),
            "last_save_time": datetime.now(timezone.utc).strftime("%H:%M"),
            "file_size_kb": 0,
        },
    }
    torch.save(entry_state, entry_path)
    log.info("Entry model saved: %s", entry_path)

    if exit_model is not None:
        exit_state = {
            "model": exit_model.state_dict(),
            "optimizer": exit_opt.state_dict(),
            "mean": exit_mean.tolist(),
            "std": exit_std.tolist(),
            "n_features": len(ENTRY_FEATURES) + 3 + len(EXIT_POSITION_FEATURES),
            "exit_model_info": {
                "cycles": 1,
                "loss": getattr(exit_model, "_final_loss", 0),
                "initial_loss": 0,
                "total_samples": n_exit_samples,
                "total_signals_used": 0,
                "last_train_ts": time.time(),
            },
        }
        torch.save(exit_state, exit_path)
        log.info("Exit model saved: %s", exit_path)

    meta = {
        "version": "v2_neural_net",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "entry_features": ENTRY_FEATURES,
        "exit_position_features": EXIT_POSITION_FEATURES,
        "profitable_threshold": PROFITABLE_THRESHOLD,
        "max_gain_pct": MAX_GAIN_PCT,
        "cost_buy": round(COST_BUY, 6),
        "cost_sell": round(COST_SELL, 6),
        "entry_age_range": [ENTRY_AGE_MIN, ENTRY_AGE_MAX],
        "max_hold_sec": MAX_HOLD_SEC,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Meta saved: %s", meta_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V2 Neural Network Training")
    parser.add_argument("--file", "-f", help="Data file path")
    parser.add_argument("--epochs-entry", type=int, default=100)
    parser.add_argument("--epochs-exit", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.833)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokens = load_data(data_file=args.file)
    if not tokens:
        log.error("No data!")
        sys.exit(1)

    train_tokens, test_tokens = prepare_datasets(tokens, train_ratio=args.train_ratio)

    log.info("Building training data from %d tokens...", len(train_tokens))
    processed_results = []
    entry_X, entry_y, entry_w = [], [], []
    for token in train_tokens:
        result = process_token(token)
        if result is None:
            continue
        processed_results.append(result)
        entry_X.append(result["entry_features"])
        entry_y.append(result["entry_label"])
        entry_w.append(result["reward_weight"])

    train_eX = np.array(entry_X, dtype=np.float32)
    train_ey = np.array(entry_y, dtype=np.float32)
    train_ew = np.array(entry_w, dtype=np.float32)
    log.info("Entry samples: %d", len(train_eX))

    train_eX_n, _, entry_mean, entry_std = normalize_features(train_eX)

    entry_model, entry_opt = train_entry_model(
        train_eX_n, train_ey, train_ew, epochs=args.epochs_entry, lr=args.lr
    )

    log.info("=" * 60)
    log.info("REBUILDING EXIT SAMPLES WITH REAL CONFIDENCE (23 features)")
    log.info("=" * 60)
    entry_model.eval()
    with torch.no_grad():
        all_confs = entry_model(torch.from_numpy(train_eX_n)).numpy()
    avg_acc = float((train_ey == (all_confs >= 0.5).astype(float)).mean())
    log.info("EntryNet avg accuracy on train: %.1f%%", avg_acc * 100)

    exit_X2, exit_y2 = [], []
    n_entry = len(ENTRY_FEATURES)
    for i, result in enumerate(processed_results):
        conf_val = float(all_confs[i])
        for xf_row, xl_row in zip(result["exit_features"], result["exit_labels"]):
            row = list(xf_row)
            row[n_entry] = conf_val
            row[n_entry + 1] = 0.5
            row[n_entry + 2] = avg_acc
            exit_X2.append(row)
            exit_y2.append(xl_row)

    log.info("Exit samples (23-feature): %d", len(exit_X2))
    n_exit = len(ENTRY_FEATURES) + 3 + len(EXIT_POSITION_FEATURES)
    train_xX = np.array(exit_X2, dtype=np.float32) if exit_X2 else np.zeros((0, n_exit), dtype=np.float32)
    train_xy = np.array(exit_y2, dtype=np.float32) if exit_y2 else np.zeros(0, dtype=np.float32)

    if len(train_xX) > 0:
        train_xX_n, _, exit_mean, exit_std = normalize_features(train_xX)
    else:
        exit_mean = np.zeros(n_exit, dtype=np.float32)
        exit_std = np.ones(n_exit, dtype=np.float32)
        train_xX_n = train_xX

    exit_model, exit_opt = train_exit_model(train_xX_n, train_xy, epochs=args.epochs_exit, lr=args.lr)

    backtest(entry_model, test_tokens, entry_mean, entry_std)

    save_models(
        entry_model, exit_model, entry_opt, exit_opt, entry_mean, entry_std, exit_mean, exit_std,
        n_entry_samples=len(train_eX), n_exit_samples=len(train_xX),
    )

    log.info("=" * 60)
    log.info("V2 TRAINING COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
