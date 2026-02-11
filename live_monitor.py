import asyncio
import json
import logging
import sys
import time
import os
from datetime import datetime, timezone

import httpx
import websockets
import numpy as np
import torch

from config import (
    PUMPPORTAL_WS_URL,
    DEXSCREENER_API,
    DATA_DIR,
    MIN_LIQUIDITY_USD,
    BET_SIZE_USD,
    MAX_SLOTS,
    PUMPFUN_FEE_PCT,
    BUY_SLIPPAGE_PCT,
    SELL_SLIPPAGE_PCT,
    STOP_LOSS_PCT,
    TRAILING_STOP_PCT,
    TIME_STOP_SEC,
    TIME_STOP_MIN_GAIN,
    MIN_CONFIDENCE_PCT,
    ROCKET_ONLY,
)

REAL_TRADING = "--real" in sys.argv
CONTINUOUS = "--continuous" in sys.argv
trader = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("live_monitor")

from train_nn import EntryNet, ExitNet, ENTRY_FEATURES, EXIT_POSITION_FEATURES

FEATURES = ENTRY_FEATURES

MIN_BUYS_FOR_SIGNAL = 5
MAX_BUYS_FOR_SIGNAL = 999
PRICE_POLL_INTERVAL = 0.5
MAX_SELLS_PER_TOKEN = 1

tokens: dict[str, dict] = {}
trade_counts: dict[str, dict] = {}
signals: list[dict] = []
stats = {"total": 0, "enriched": 0, "trades": 0, "signals": 0, "start": 0}

missed_tokens: list[dict] = []
missed_trained_mints: set[str] = set()
MISSED_CHECK_DELAY = 900
MISSED_PUMP_THRESHOLD = 50.0

SOL_PRICE_USD = 200.0
entry_model = None
exit_model = None
entry_mean = None
entry_std = None
exit_mean = None
exit_std = None
seen_mints: set[str] = set()


def bonding_curve_price_sol(v_sol: float, v_tokens: float) -> float:
    if v_tokens <= 0:
        return 0.0
    return v_sol / v_tokens


def bonding_curve_price_usd(v_sol: float, v_tokens: float) -> float:
    return bonding_curve_price_sol(v_sol, v_tokens) * SOL_PRICE_USD


def init_trader():
    global trader
    if not REAL_TRADING:
        return
    from real_trader import RealTrader
    trader = RealTrader()
    bal = trader.refresh_balance()
    trader.initial_balance = bal
    log.info(
        "REAL TRADING MODE | Wallet: %s | Balance: %.4f SOL ($%.2f)",
        trader.pubkey, bal, bal * SOL_PRICE_USD,
    )


def load_model():
    global entry_model, exit_model, entry_mean, entry_std, exit_mean, exit_std
    entry_path = os.path.join(DATA_DIR, "entry_model.pt")
    exit_path = os.path.join(DATA_DIR, "exit_model.pt")
    if not os.path.exists(entry_path):
        log.error("No entry model at %s", entry_path)
        return False
    state = torch.load(entry_path, map_location="cpu", weights_only=False)
    n_feat = state["n_features"]
    entry_model = EntryNet(n_feat)
    entry_model.load_state_dict(state["model"])
    entry_model.eval()
    entry_mean = np.array(state["mean"], dtype=np.float32)
    entry_std = np.array(state["std"], dtype=np.float32)
    log.info("Entry NN loaded (%d features)", n_feat)
    if os.path.exists(exit_path):
        xs = torch.load(exit_path, map_location="cpu", weights_only=False)
        n_xf = xs["n_features"]
        exit_model = ExitNet(n_xf)
        exit_model.load_state_dict(xs["model"])
        exit_model.eval()
        exit_mean = np.array(xs["mean"], dtype=np.float32)
        exit_std = np.array(xs["std"], dtype=np.float32)
        log.info("Exit NN loaded (%d features)", n_xf)
    return True


def predict_token(token_data: dict) -> tuple[str, float]:
    if entry_model is None:
        return "no_model", 0.0
    values = []
    for f in FEATURES:
        v = token_data.get(f, 0)
        values.append(float(v) if v else 0.0)
    X = np.array([values], dtype=np.float32)
    X_n = (X - entry_mean) / entry_std
    with torch.no_grad():
        prob = entry_model(torch.from_numpy(X_n)).item()
    confidence = prob * 100
    label = "ROCKET" if prob >= 0.5 else "trash"
    return label, confidence


async def fetch_sol_price(client: httpx.AsyncClient):
    global SOL_PRICE_USD
    try:
        r = await client.get(
            f"{DEXSCREENER_API}/tokens/v1/solana/So11111111111111111111111111111111111111112",
            timeout=10,
        )
        if r.status_code == 200:
            pairs = r.json()
            if pairs and isinstance(pairs, list):
                SOL_PRICE_USD = float(pairs[0].get("priceUsd") or 200)
                log.info("SOL price: $%.2f", SOL_PRICE_USD)
    except Exception as e:
        log.warning("Failed to fetch SOL price: %s", e)


async def enrich_token(client: httpx.AsyncClient, mint: str):
    try:
        r = await client.get(
            f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
            timeout=10,
        )
        if r.status_code != 200:
            return
        pairs = r.json()
        if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
            return
        p = pairs[0]
        token = tokens[mint]
        token["dex_liquidity_usd"] = float((p.get("liquidity") or {}).get("usd") or 0)
        token["dex_volume_5m"] = float((p.get("volume") or {}).get("m5") or 0)
        token["dex_volume_1h"] = float((p.get("volume") or {}).get("h1") or 0)
        token["dex_buys_5m"] = int((p.get("txns") or {}).get("m5", {}).get("buys") or 0)
        token["dex_sells_5m"] = int((p.get("txns") or {}).get("m5", {}).get("sells") or 0)
        token["dex_buys_1h"] = int((p.get("txns") or {}).get("h1", {}).get("buys") or 0)
        token["dex_sells_1h"] = int((p.get("txns") or {}).get("h1", {}).get("sells") or 0)
        token["dex_market_cap"] = float(p.get("marketCap") or 0)
        token["dex_fdv"] = float(p.get("fdv") or 0)
        info = p.get("info", {}) or {}
        token["has_website"] = len(info.get("websites", [])) > 0
        token["has_socials"] = len(info.get("socials", [])) > 0
        token["enriched"] = True
        stats["enriched"] += 1
    except Exception:
        pass


async def enrich_batch(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(15)
        now = time.time()
        to_enrich = []
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age >= 60 and not token.get("enriched") and not token.get("enrich_tried"):
                to_enrich.append(mint)
            if len(to_enrich) >= 5:
                break
        for mint in to_enrich:
            tokens[mint]["enrich_tried"] = True
            await enrich_token(client, mint)
            await asyncio.sleep(1.1)


def calc_bonding_curve_pnl(sig: dict, token_data: dict) -> float | None:
    sim_tokens = sig.get("sim_tokens_bought", 0)
    sim_sol_spent = sig.get("sim_sol_spent", 0)
    if sim_tokens <= 0 or sim_sol_spent <= 0:
        return None
    cur_v_sol = token_data.get("v_sol_in_bonding", 0)
    cur_v_tokens = token_data.get("v_tokens_in_bonding", 0)
    if cur_v_sol <= 0 or cur_v_tokens <= 0:
        return None
    k = cur_v_sol * cur_v_tokens
    new_v_tokens = cur_v_tokens + sim_tokens
    new_v_sol = k / new_v_tokens
    gross_sol_out = cur_v_sol - new_v_sol
    sol_after_fee = gross_sol_out * (1 - PUMPFUN_FEE_PCT)
    sol_after_slippage = sol_after_fee * (1 - SELL_SLIPPAGE_PCT)
    pnl_pct = ((sol_after_slippage / sim_sol_spent) - 1) * 100
    return pnl_pct


async def ml_scanner(client: httpx.AsyncClient):
    debug_count = [0]
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age < 30 or age > 120:
                continue
            if token.get("ml_checked"):
                continue

            tc = trade_counts.get(mint, {})
            buys = tc.get("buys", 0)
            sells = tc.get("sells", 0)
            if buys < MIN_BUYS_FOR_SIGNAL:
                continue
            if buys > MAX_BUYS_FOR_SIGNAL:
                token["ml_checked"] = True
                continue

            v_sol = token.get("v_sol_in_bonding", 0)
            v_tokens = token.get("v_tokens_in_bonding", 0)
            cur_price = bonding_curve_price_usd(v_sol, v_tokens)
            p15 = token.get("price_snap_15s", 0)
            p30 = token.get("price_snap_30s", 0)
            token["log_buy_sol"] = np.log1p(token.get("initial_buy_sol", 0))
            token["log_mcap"] = np.log1p(token.get("initial_mcap_usd", 0))
            token["buy_rate"] = buys / max(1, age)
            token["volume_rate"] = tc.get("buy_sol", 0) / max(1, age)
            token["buyer_rate"] = len(tc.get("buyers", set())) / max(1, age)
            total_trades = buys + sells
            token["sell_pressure"] = round(sells / max(1, total_trades) * 100, 1)
            raw_mom = ((p30 / p15) - 1) * 100 if p15 > 0 and p30 > 0 else 0.0
            token["momentum_15_30"] = max(-500.0, min(500.0, raw_mom))
            token["total_buys"] = buys
            token["total_sells"] = sells
            token["total_buy_sol"] = tc.get("buy_sol", 0)
            token["total_sell_sol"] = tc.get("sell_sol", 0)
            token["buy_sell_ratio"] = round(buys / max(1, sells), 2)
            token["sell_pressure"] = round(sells / max(1, buys + sells) * 100, 1)
            token["unique_buyers"] = len(tc.get("buyers", set()))
            token["unique_sellers"] = len(tc.get("sellers", set()))
            token["trade_count"] = buys + sells

            label, confidence = predict_token(token)
            token["ml_checked"] = True
            token["ml_label"] = label
            token["ml_confidence"] = confidence

            if label == "trash" and buys >= MIN_BUYS_FOR_SIGNAL:
                feat_snap = [float(token.get(f, 0) or 0) for f in FEATURES]
                missed_tokens.append({
                    "mint": mint,
                    "symbol": token["symbol"],
                    "features": feat_snap,
                    "eval_ts": now,
                    "entry_v_sol": v_sol,
                    "entry_v_tokens": v_tokens,
                })

            if confidence >= 25 or debug_count[0] < 10:
                debug_count[0] += 1
                log.info(
                    "ML %s: label=%s conf=%.1f%% age=%.0fs buys=%d buy_rate=%.2f vol_rate=%.3f buyer_rate=%.2f sell_p=%.0f%% mom=%.1f buy_sol=%.4f mcap=%.0f",
                    token["symbol"], label, confidence, age, buys,
                    token.get("buy_rate", 0), token.get("volume_rate", 0),
                    token.get("buyer_rate", 0), token.get("sell_pressure", 0),
                    token.get("momentum_15_30", 0), token.get("initial_buy_sol", 0),
                    token.get("initial_mcap_usd", 0),
                )

            if label in ("ROCKET", "winner"):
                if ROCKET_ONLY and label != "ROCKET":
                    continue
                if confidence < MIN_CONFIDENCE_PCT:
                    log.info(
                        "SKIP %s %s: conf=%.0f%% < %.0f%%",
                        label, token["symbol"], confidence, MIN_CONFIDENCE_PCT,
                    )
                    continue
                if mint in seen_mints:
                    continue
                seen_mints.add(mint)

                active_count = sum(1 for s in signals if s["status"] == "ACTIVE")
                if active_count >= MAX_SLOTS:
                    log.info(
                        "SKIP %s %s: %d/%d slots full",
                        label, token["symbol"], active_count, MAX_SLOTS,
                    )
                    continue

                liq = token.get("dex_liquidity_usd", 0)
                if liq < MIN_LIQUIDITY_USD and token.get("enriched"):
                    log.info(
                        "SKIP %s %s: liquidity=$%.0f < $%.0f",
                        label, token["symbol"], liq, MIN_LIQUIDITY_USD,
                    )
                    continue

                if not token.get("enriched"):
                    await enrich_token(client, mint)

                signal_time = time.time()
                initial_mcap = token.get("initial_mcap_usd", 0)

                sol_amount = round(BET_SIZE_USD / SOL_PRICE_USD, 4)
                v_sol = token.get("v_sol_in_bonding", 0)
                v_tokens = token.get("v_tokens_in_bonding", 0)
                sim_tokens = 0.0
                sim_sol_spent = 0.0
                if v_sol > 0 and v_tokens > 0:
                    sol_after_fee = sol_amount * (1 - PUMPFUN_FEE_PCT)
                    sol_effective = sol_after_fee * (1 - BUY_SLIPPAGE_PCT)
                    k = v_sol * v_tokens
                    new_v_sol = v_sol + sol_effective
                    new_v_tokens = k / new_v_sol
                    sim_tokens = v_tokens - new_v_tokens
                    sim_sol_spent = sol_amount

                entry_cost_pct = 0.0
                if sim_tokens > 0 and sim_sol_spent > 0 and v_sol > 0 and v_tokens > 0:
                    k_entry = v_sol * v_tokens
                    vt_after = v_tokens + sim_tokens
                    vs_after = k_entry / vt_after
                    gross_out = v_sol - vs_after
                    net_out = gross_out * (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT)
                    entry_cost_pct = ((net_out / sim_sol_spent) - 1) * 100

                entry_price_usd = bonding_curve_price_usd(v_sol, v_tokens)

                signal = {
                    "mint": mint,
                    "symbol": token["symbol"],
                    "name": token["name"],
                    "signal_time": signal_time,
                    "signal_at": datetime.now(timezone.utc).isoformat(),
                    "signal_age_sec": int(age),
                    "ml_label": label,
                    "ml_confidence": round(confidence, 1),
                    "initial_mcap_usd": initial_mcap,
                    "buys_at_signal": buys,
                    "sells_at_signal": sells,
                    "buy_ratio_at_signal": token["buy_sell_ratio"],
                    "sell_pressure_at_signal": token["sell_pressure"],
                    "entry_price_usd": entry_price_usd,
                    "entry_v_sol": v_sol,
                    "entry_v_tokens": v_tokens,
                    "current_price_usd": None,
                    "pnl_pct": None,
                    "pnl_usd": None,
                    "peak_pnl_pct": 0.0,
                    "gain_from_entry": 0.0,
                    "peak_gain": 0.0,
                    "entry_cost_pct": round(entry_cost_pct, 1),
                    "position_remaining_pct": 100.0,
                    "realized_pnl": 0.0,
                    "status": "ACTIVE",
                    "close_reason": None,
                    "close_pnl_pct": None,
                    "close_time": None,
                    "checked_at": None,
                    "sells_done": 0,
                    "sim_tokens_bought": sim_tokens,
                    "sim_sol_spent": sim_sol_spent,
                    "sim_v_sol_at_buy": v_sol,
                    "sim_v_tokens_at_buy": v_tokens,
                    "bet_usd": BET_SIZE_USD,
                }
                signals.append(signal)
                stats["signals"] += 1
                log.info(
                    "*** SIGNAL #%d: %s %s (%s) conf=%.0f%% buys=%d ratio=%.1f | "
                    "sim: %.4f SOL ($%.2f) -> %.0f tokens | entry_cost=%.1f%% ***",
                    stats["signals"], label, token["symbol"], mint[:8],
                    confidence, buys, token["buy_sell_ratio"],
                    sim_sol_spent, BET_SIZE_USD, sim_tokens, entry_cost_pct,
                )

                if REAL_TRADING and trader:
                    asyncio.create_task(execute_real_buy(signal, sol_amount))


NO_EXIT_RULES = "--no-exits" in sys.argv

def check_exit_rules(sig: dict, current_pnl: float) -> str | None:
    if sig["status"] != "ACTIVE":
        return None

    entry_cost = sig.get("entry_cost_pct", 0)
    gain = current_pnl - entry_cost
    sig["gain_from_entry"] = round(gain, 1)
    if gain > sig.get("peak_gain", 0):
        sig["peak_gain"] = round(gain, 1)

    if NO_EXIT_RULES:
        return None

    if gain <= STOP_LOSS_PCT:
        return (
            f"STOP_LOSS (gain={gain:+.1f}% <= {STOP_LOSS_PCT}%, "
            f"real_pnl={current_pnl:+.1f}%)"
        )

    age_sec = time.time() - sig["signal_time"]
    if age_sec >= TIME_STOP_SEC and gain < TIME_STOP_MIN_GAIN:
        return (
            f"TIME_STOP ({age_sec:.0f}s, gain={gain:+.1f}% < "
            f"+{TIME_STOP_MIN_GAIN}%, real_pnl={current_pnl:+.1f}%)"
        )

    peak_gain = sig.get("peak_gain", 0)
    if peak_gain >= 30.0:
        drop = peak_gain - gain
        if drop >= TRAILING_STOP_PCT:
            return (
                f"TRAILING_STOP (peak={peak_gain:+.1f}%, gain={gain:+.1f}%, "
                f"drop={drop:.1f}%, real_pnl={current_pnl:+.1f}%)"
            )

    if peak_gain >= 15.0 and gain < 0:
        return (
            f"PROFIT_GONE (peak={peak_gain:+.1f}%, gain={gain:+.1f}%, "
            f"real_pnl={current_pnl:+.1f}%)"
        )

    return None


def close_signal(sig: dict, reason: str, pnl: float):
    remaining = sig["position_remaining_pct"] / 100.0
    unrealized = remaining * (pnl / 100.0)
    total_pnl_per_dollar = sig["realized_pnl"] + unrealized
    sig["status"] = "CLOSED"
    sig["close_reason"] = reason
    sig["close_pnl_pct"] = round(total_pnl_per_dollar * 100, 1)
    sig["close_time"] = datetime.now(timezone.utc).isoformat()
    sig["pnl_usd"] = round(BET_SIZE_USD * total_pnl_per_dollar, 4)
    log.info(
        "CLOSED %s %s: %s | P&L: %+.1f%% ($%+.4f on $%.2f bet)",
        sig["ml_label"], sig["symbol"], reason,
        total_pnl_per_dollar * 100, sig["pnl_usd"], BET_SIZE_USD,
    )

    if REAL_TRADING and trader and sig.get("real_buy") and sig["position_remaining_pct"] > 0:
        sells_done = sig.get("sells_done", 0)
        if sells_done < MAX_SELLS_PER_TOKEN:
            sig["sells_done"] = sells_done + 1
            asyncio.create_task(execute_real_sell(sig, 100, reason))


async def execute_real_buy(sig: dict, sol_amount: float):
    try:
        result = await trader.buy_token(sig["mint"], sol_amount, sig["symbol"])
        sig["real_buy"] = result
        if result["success"]:
            sig["real_buy_tx"] = result["tx_hash"]
            sig["real_sol_spent"] = sol_amount
            sig["buy_confirmed"] = True
            log.info(
                "REAL BUY OK %s: %.4f SOL | tx=%s",
                sig["symbol"], sol_amount, result["tx_hash"][:20],
            )
        else:
            log.error("REAL BUY FAILED %s: %s", sig["symbol"], result["error"])
    except Exception as e:
        log.error("REAL BUY ERROR %s: %s", sig["symbol"], e)


async def execute_real_sell(sig: dict, sell_pct: int, reason: str):
    if REAL_TRADING and not sig.get("buy_confirmed"):
        log.warning("SELL SKIP %s: buy not confirmed, waiting...", sig["symbol"])
        for _ in range(60):
            await asyncio.sleep(0.5)
            if sig.get("buy_confirmed"):
                break
        if not sig.get("buy_confirmed"):
            log.error("SELL ABORT %s: buy never confirmed", sig["symbol"])
            return {"success": False, "error": "buy not confirmed"}
    try:
        result = await trader.sell_token(sig["mint"], sell_pct, sig["symbol"], reason)
        if result["success"]:
            log.info(
                "REAL SELL OK %s %d%% (%s): tx=%s",
                sig["symbol"], sell_pct, reason, result["tx_hash"][:20],
            )
        else:
            log.error(
                "REAL SELL FAILED %s (%s): %s",
                sig["symbol"], reason, result.get("error", ""),
            )
        return result
    except Exception as e:
        log.error("REAL SELL ERROR %s: %s", sig["symbol"], e)
        return {"success": False, "error": str(e)}


async def signal_price_updater():
    while True:
        await asyncio.sleep(PRICE_POLL_INTERVAL)
        active = [s for s in signals if s["status"] == "ACTIVE"]
        if not active:
            continue

        for sig in active:
            try:
                mint = sig["mint"]
                token_data = tokens.get(mint, {})
                current_pnl = calc_bonding_curve_pnl(sig, token_data)

                if current_pnl is not None:
                    sig["pnl_pct"] = round(current_pnl, 1)
                    if current_pnl > sig.get("peak_pnl_pct", 0):
                        sig["peak_pnl_pct"] = round(current_pnl, 1)
                    sig["checked_at"] = datetime.now(timezone.utc).isoformat()

                    v_sol = token_data.get("v_sol_in_bonding", 0)
                    v_tokens = token_data.get("v_tokens_in_bonding", 0)
                    sig["current_price_usd"] = bonding_curve_price_usd(v_sol, v_tokens)

                    reason = check_exit_rules(sig, current_pnl)
                    if reason:
                        close_signal(sig, reason, current_pnl)
                    else:
                        remaining = sig["position_remaining_pct"] / 100.0
                        unrealized = remaining * (current_pnl / 100.0)
                        total = sig["realized_pnl"] + unrealized
                        sig["pnl_usd"] = round(BET_SIZE_USD * total, 4)
            except Exception:
                pass


async def listen_pumpportal():
    ws = None
    while True:
        try:
            ws = await websockets.connect(PUMPPORTAL_WS_URL)
            log.info("Connected to PumpPortal WebSocket")
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            log.info("Subscribed to newToken events")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("txType") == "create":
                    mint = msg.get("mint", "")
                    if not mint or mint in tokens:
                        continue

                    now = time.time()
                    elapsed = int(now - stats["start"])
                    symbol = msg.get("symbol", "???")
                    name = msg.get("name", "")

                    v_sol = float(msg.get("vSolInBondingCurve") or 0)
                    v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
                    mcap_sol = float(msg.get("marketCapSol") or 0)
                    init_buy = (
                        float(msg.get("initialBuy") or 0) / 1e9
                        if msg.get("initialBuy")
                        else 0
                    )

                    price_sol = bonding_curve_price_sol(v_sol, v_tokens)
                    price_usd = price_sol * SOL_PRICE_USD
                    mcap_usd = mcap_sol * SOL_PRICE_USD

                    tokens[mint] = {
                        "mint": mint,
                        "symbol": symbol,
                        "name": name,
                        "created_ts": now,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_sec": elapsed,
                        "initial_buy_sol": init_buy,
                        "initial_price_sol": price_sol,
                        "initial_price_usd": price_usd,
                        "initial_mcap_usd": mcap_usd,
                        "market_cap_sol": mcap_sol,
                        "v_tokens_in_bonding": v_tokens,
                        "v_sol_in_bonding": v_sol,
                        "dev_address": msg.get("traderPublicKey", ""),
                        "migrated": False,
                        "enriched": False,
                        "enrich_tried": False,
                        "dex_liquidity_usd": 0,
                        "ml_checked": False,
                        "ml_label": None,
                        "ml_confidence": None,
                        "price_snap_15s": 0.0,
                        "price_snap_30s": 0.0,
                    }

                    trade_counts[mint] = {
                        "buys": 0, "sells": 0,
                        "buy_sol": 0, "sell_sol": 0,
                        "buyers": set(), "sellers": set(),
                    }
                    stats["total"] += 1

                    if stats["total"] % 20 == 0:
                        log.info(
                            "[%ds] tokens=%d signals=%d trades=%d",
                            elapsed, stats["total"], stats["signals"], stats["trades"],
                        )

                    await ws.send(json.dumps({
                        "method": "subscribeTokenTrade",
                        "keys": [mint],
                    }))

                elif msg.get("txType") in ("buy", "sell"):
                    mint = msg.get("mint", "")
                    if mint not in trade_counts:
                        continue

                    tx = msg["txType"]
                    sol_amount = float(msg.get("solAmount") or 0)
                    trader_key = msg.get("traderPublicKey", "")
                    tc = trade_counts[mint]

                    if tx == "buy":
                        tc["buys"] += 1
                        tc["buy_sol"] += sol_amount
                        if trader_key:
                            tc["buyers"].add(trader_key)
                    else:
                        tc["sells"] += 1
                        tc["sell_sol"] += sol_amount
                        if trader_key:
                            tc["sellers"].add(trader_key)

                    new_v_sol = float(msg.get("vSolInBondingCurve") or 0)
                    new_v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
                    new_mcap_sol = float(msg.get("marketCapSol") or 0)

                    if mint in tokens:
                        if new_mcap_sol > 0:
                            tokens[mint]["latest_mcap_sol"] = new_mcap_sol
                            tokens[mint]["latest_mcap_usd"] = new_mcap_sol * SOL_PRICE_USD
                        if new_v_sol > 0:
                            tokens[mint]["v_sol_in_bonding"] = new_v_sol
                        if new_v_tokens > 0:
                            tokens[mint]["v_tokens_in_bonding"] = new_v_tokens
                        token_age = time.time() - tokens[mint]["created_ts"]
                        cur_price = bonding_curve_price_usd(new_v_sol, new_v_tokens) if new_v_sol > 0 and new_v_tokens > 0 else 0
                        if cur_price > 0:
                            if token_age >= 15 and tokens[mint]["price_snap_15s"] == 0.0:
                                tokens[mint]["price_snap_15s"] = cur_price
                            if token_age >= 30 and tokens[mint]["price_snap_30s"] == 0.0:
                                tokens[mint]["price_snap_30s"] = cur_price

                    for sig in signals:
                        if sig["mint"] == mint and sig["status"] == "ACTIVE":
                            real_pnl = calc_bonding_curve_pnl(sig, tokens.get(mint, {}))
                            if real_pnl is not None:
                                sig["pnl_pct"] = round(real_pnl, 1)
                                if real_pnl > sig.get("peak_pnl_pct", 0):
                                    sig["peak_pnl_pct"] = round(real_pnl, 1)
                                sig["checked_at"] = datetime.now(timezone.utc).isoformat()

                                v_s = tokens.get(mint, {}).get("v_sol_in_bonding", 0)
                                v_t = tokens.get(mint, {}).get("v_tokens_in_bonding", 0)
                                sig["current_price_usd"] = bonding_curve_price_usd(v_s, v_t)

                                reason = check_exit_rules(sig, real_pnl)
                                if reason:
                                    close_signal(sig, reason, real_pnl)
                                else:
                                    remaining = sig["position_remaining_pct"] / 100.0
                                    unrealized = remaining * (real_pnl / 100.0)
                                    total = sig["realized_pnl"] + unrealized
                                    sig["pnl_usd"] = round(BET_SIZE_USD * total, 4)

                    stats["trades"] += 1

        except websockets.exceptions.ConnectionClosed:
            log.warning("Disconnected, reconnecting in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("Error: %s, reconnecting in 5s...", e)
            await asyncio.sleep(5)
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass


def print_signals_report():
    active = [s for s in signals if s["status"] == "ACTIVE"]
    closed = [s for s in signals if s["status"] == "CLOSED"]

    lines = []
    lines.append(
        f"=== SIGNAL REPORT | Active: {len(active)} | Closed: {len(closed)} "
        f"| Total: {len(signals)} ==="
    )

    total_pnl_usd = 0.0
    total_invested = 0

    if active:
        lines.append("--- ACTIVE POSITIONS ---")
        for sig in active:
            age = int(time.time() - sig["signal_time"])
            pnl = sig.get("pnl_pct")
            peak_gain = sig.get("peak_gain", 0)
            gain = sig.get("gain_from_entry", 0)

            if pnl is not None:
                pnl_usd = BET_SIZE_USD * (pnl / 100.0)
                pnl_str = f"pnl={pnl:+.1f}% (${pnl_usd:+.2f})"
                total_pnl_usd += pnl_usd
            else:
                pnl_str = "waiting..."

            total_invested += 1
            lines.append(
                f"  {sig['ml_label']} {sig['symbol']} | {pnl_str} | "
                f"peak={peak_gain:+.1f}% | {age}s ago | "
                f"buys={sig['buys_at_signal']}"
            )

    if closed:
        lines.append("--- CLOSED POSITIONS ---")
        wins = 0
        losses = 0
        for sig in closed:
            cpnl = sig.get("close_pnl_pct", 0)
            pnl_usd = sig.get("pnl_usd", BET_SIZE_USD * cpnl / 100)
            total_pnl_usd += pnl_usd
            total_invested += 1
            if cpnl > 0:
                wins += 1
            else:
                losses += 1
            lines.append(
                f"  {sig['symbol']} | {cpnl:+.1f}% (${pnl_usd:+.2f}) | "
                f"{sig['close_reason']}"
            )
        if wins + losses > 0:
            lines.append(
                f"  Stats: {wins}W / {losses}L "
                f"({wins/(wins+losses)*100:.0f}% win rate)"
            )

    if total_invested > 0:
        total_bet = total_invested * BET_SIZE_USD
        lines.append(
            f"  TOTAL: ${total_bet:.2f} invested -> "
            f"P&L: ${total_pnl_usd:+.2f} ({total_pnl_usd/total_bet*100:+.1f}%)"
        )

    if not signals:
        lines.append("  No signals yet...")

    return "\n".join(lines)


def save_session():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"live_session_{ts}.json")

    for mint, token in tokens.items():
        tc = trade_counts.get(mint, {})
        token["total_buys"] = tc.get("buys", 0)
        token["total_sells"] = tc.get("sells", 0)
        token["total_buy_sol"] = round(tc.get("buy_sol", 0), 4)
        token["total_sell_sol"] = round(tc.get("sell_sol", 0), 4)
        token["buy_sell_ratio"] = round(
            tc.get("buys", 0) / max(1, tc.get("sells", 1)), 2
        )
        token["sell_pressure"] = round(
            tc.get("sells", 0) / max(1, tc.get("buys", 1) + tc.get("sells", 0)) * 100, 1
        )

    active_count = sum(1 for s in signals if s["status"] == "ACTIVE")
    closed_count = sum(1 for s in signals if s["status"] == "CLOSED")
    closed_wins = sum(
        1 for s in signals
        if s["status"] == "CLOSED" and (s.get("close_pnl_pct", 0) or 0) > 0
    )

    output = {
        "meta": {
            "start_time": datetime.fromtimestamp(
                stats["start"], tz=timezone.utc
            ).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(time.time() - stats["start"]),
            "sol_price_usd": SOL_PRICE_USD,
            "total_tokens": stats["total"],
            "total_signals": stats["signals"],
            "total_trades": stats["trades"],
            "active_signals": active_count,
            "closed_signals": closed_count,
            "closed_wins": closed_wins,
            "price_source": "pumpfun_bonding_curve",
        },
        "config": {
            "bet_size_usd": BET_SIZE_USD,
            "max_slots": MAX_SLOTS,
            "min_buys": MIN_BUYS_FOR_SIGNAL,
            "min_confidence": MIN_CONFIDENCE_PCT,
            "min_liquidity": MIN_LIQUIDITY_USD,
            "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "time_stop_sec": TIME_STOP_SEC,
            "time_stop_min_gain": TIME_STOP_MIN_GAIN,
            "buy_slippage": BUY_SLIPPAGE_PCT,
            "sell_slippage": SELL_SLIPPAGE_PCT,
            "pumpfun_fee": PUMPFUN_FEE_PCT,
            "rocket_only": ROCKET_ONLY,
        },
        "signals": signals,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("Session saved: %s", filepath)
    return filepath


async def report_printer():
    while True:
        await asyncio.sleep(120)
        report = print_signals_report()
        log.info("\n%s", report)


INCREMENTAL_INTERVAL = 600
incremental_trained_mints: set[str] = set()
resolved_missed: list[dict] = []
MEMORY_CLEANUP_INTERVAL = 1800
TOKEN_MAX_AGE = 1800


def calc_missed_pnl(entry_v_sol, entry_v_tokens, cur_v_sol, cur_v_tokens):
    if entry_v_sol <= 0 or entry_v_tokens <= 0:
        return None
    if cur_v_sol <= 0 or cur_v_tokens <= 0:
        return None
    sol_amount = BET_SIZE_USD / SOL_PRICE_USD
    sol_after_fee = sol_amount * (1 - PUMPFUN_FEE_PCT)
    sol_effective = sol_after_fee * (1 - BUY_SLIPPAGE_PCT)
    k_buy = entry_v_sol * entry_v_tokens
    new_v_sol_buy = entry_v_sol + sol_effective
    new_v_tokens_buy = k_buy / new_v_sol_buy
    sim_tokens = entry_v_tokens - new_v_tokens_buy
    if sim_tokens <= 0:
        return None
    k_sell = cur_v_sol * cur_v_tokens
    vt_after_sell = cur_v_tokens + sim_tokens
    vs_after_sell = k_sell / vt_after_sell
    gross_out = cur_v_sol - vs_after_sell
    net_out = gross_out * (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT)
    return ((net_out / sol_amount) - 1) * 100


async def missed_token_checker():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        pending = [m for m in missed_tokens
                   if m["mint"] not in missed_trained_mints
                   and now - m["eval_ts"] >= MISSED_CHECK_DELAY]
        if not pending:
            continue

        checked = 0
        rockets = 0
        for mt in pending:
            missed_trained_mints.add(mt["mint"])
            token_data = tokens.get(mt["mint"])
            if not token_data:
                continue
            cur_v_sol = token_data.get("v_sol_in_bonding", 0)
            cur_v_tokens = token_data.get("v_tokens_in_bonding", 0)
            hyp_pnl = calc_missed_pnl(
                mt["entry_v_sol"], mt["entry_v_tokens"],
                cur_v_sol, cur_v_tokens,
            )
            if hyp_pnl is None:
                continue
            checked += 1
            is_profitable = 1.0 if hyp_pnl >= MISSED_PUMP_THRESHOLD else 0.0
            if hyp_pnl >= 100:
                weight = 5.0
            elif hyp_pnl >= MISSED_PUMP_THRESHOLD:
                weight = 4.0
            elif hyp_pnl <= -10:
                weight = 1.5
            else:
                weight = 1.0
            resolved_missed.append({
                "features": mt["features"],
                "label": is_profitable,
                "weight": weight,
                "pnl": hyp_pnl,
                "symbol": mt["symbol"],
            })
            if is_profitable > 0:
                rockets += 1
                log.info(
                    "MISSED ROCKET: %s hypothetical PNL=+%.0f%% (w=%.1f)",
                    mt["symbol"], hyp_pnl, weight,
                )
        if checked > 0:
            log.info(
                "MISSED CHECK: %d tokens resolved, %d were rockets, %d confirmed trash",
                checked, rockets, checked - rockets,
            )


async def incremental_learner():
    global entry_model, exit_model
    while True:
        await asyncio.sleep(INCREMENTAL_INTERVAL)
        closed = [s for s in signals if s["status"] == "CLOSED" and s["mint"] not in incremental_trained_mints]

        entry_X_list = []
        entry_y_list = []
        entry_w_list = []

        for sig in closed:
            incremental_trained_mints.add(sig["mint"])
            mint = sig["mint"]
            token_data = tokens.get(mint, {})
            values = []
            for f in FEATURES:
                v = token_data.get(f, 0)
                values.append(float(v) if v else 0.0)

            final_pnl = sig.get("close_pnl_pct", 0) or 0
            is_profitable = 1.0 if final_pnl >= 5.0 else 0.0

            if final_pnl >= 50:
                weight = 5.0
            elif final_pnl >= 20:
                weight = 4.0
            elif final_pnl >= 5:
                weight = 3.0
            elif final_pnl <= -10:
                weight = 2.0
            else:
                weight = 1.0

            entry_X_list.append(values)
            entry_y_list.append(is_profitable)
            entry_w_list.append(weight)

        while resolved_missed:
            rm = resolved_missed.pop(0)
            entry_X_list.append(rm["features"])
            entry_y_list.append(rm["label"])
            entry_w_list.append(rm["weight"])

        total = len(entry_X_list)
        if total < 3:
            log.info("INCREMENTAL: only %d samples (signals+missed), need >=3, skipping", total)
            continue

        n_signals = len(closed)
        n_missed = total - n_signals
        log.info("INCREMENTAL: feeding %d samples (%d signals + %d missed) to model...", total, n_signals, n_missed)

        if entry_model is None:
            continue

        X = np.array(entry_X_list, dtype=np.float32)
        y = np.array(entry_y_list, dtype=np.float32)
        w = np.array(entry_w_list, dtype=np.float32)
        X_n = (X - entry_mean) / entry_std

        X_t = torch.from_numpy(X_n)
        y_t = torch.from_numpy(y)
        w_t = torch.from_numpy(w)

        entry_model.train()
        optimizer = torch.optim.Adam(entry_model.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(10):
            optimizer.zero_grad()
            pred = entry_model(X_t)
            bce = torch.nn.functional.binary_cross_entropy(pred, y_t, reduction="none")
            loss = (w_t * bce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(entry_model.parameters(), 1.0)
            optimizer.step()
        entry_model.eval()

        wins = int(y.sum())
        log.info(
            "INCREMENTAL: trained on %d samples (%d wins, %.0f%%) | loss=%.4f",
            total, wins, wins / total * 100, loss.item(),
        )

        entry_path = os.path.join(DATA_DIR, "entry_model.pt")
        torch.save({
            "model": entry_model.state_dict(),
            "n_features": len(FEATURES),
            "mean": entry_mean.tolist(),
            "std": entry_std.tolist(),
        }, entry_path)
        log.info("INCREMENTAL: model saved to %s", entry_path)


async def memory_cleanup():
    while True:
        await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
        now = time.time()
        signal_mints = {s["mint"] for s in signals if s["status"] == "ACTIVE"}
        missed_pending = {m["mint"] for m in missed_tokens if m["mint"] not in missed_trained_mints}
        stale = []
        for mint, t in tokens.items():
            if mint in signal_mints or mint in missed_pending:
                continue
            last_trade = t.get("last_trade_time", t.get("created_ts", 0))
            if now - last_trade > TOKEN_MAX_AGE:
                stale.append(mint)
        for mint in stale:
            del tokens[mint]
            trade_counts.pop(mint, None)
        if stale:
            log.info("MEMORY CLEANUP: removed %d stale tokens, %d remain", len(stale), len(tokens))


AUTOSAVE_INTERVAL = 3600


async def auto_saver():
    while True:
        await asyncio.sleep(AUTOSAVE_INTERVAL)
        try:
            filepath = save_session()
            log.info("AUTOSAVE: session saved to %s", filepath)
        except Exception as e:
            log.error("AUTOSAVE failed: %s", e)


async def main():
    duration = int(os.getenv("MONITOR_DURATION", "1800"))
    log.info("=" * 60)
    mode_str = "REAL TRADING" if REAL_TRADING else "PAPER TRADING"
    run_mode = "CONTINUOUS" if CONTINUOUS else f"{duration}s"
    log.info("SOLANA SNIPER BOT V2 [%s] [%s]", mode_str, run_mode)
    log.info(
        "Bet: $%.2f | Max slots: %d | Price: bonding curve ONLY",
        BET_SIZE_USD, MAX_SLOTS,
    )
    rocket_str = "ROCKET only" if ROCKET_ONLY else "ROCKET + winner"
    log.info(
        "Filters: %s, conf >= %.0f%%, buys >= %d, liquidity >= $%.0f",
        rocket_str, MIN_CONFIDENCE_PCT, MIN_BUYS_FOR_SIGNAL, MIN_LIQUIDITY_USD,
    )
    log.info(
        "Stop-loss: %.0f%% | Time-stop: %ds (min +%.0f%%) | "
        "Trailing: -%.0f%% from peak (activates at +30%%)",
        STOP_LOSS_PCT, TIME_STOP_SEC, TIME_STOP_MIN_GAIN, TRAILING_STOP_PCT,
    )
    log.info(
        "Fees: buy slip %.0f%%, sell slip %.0f%%, PumpFun fee %.0f%%",
        BUY_SLIPPAGE_PCT * 100, SELL_SLIPPAGE_PCT * 100, PUMPFUN_FEE_PCT * 100,
    )
    log.info("=" * 60)

    if not load_model():
        log.error("Cannot start without ML model. Run train_nn.py first.")
        return

    if REAL_TRADING:
        init_trader()
        if not trader:
            log.error("Failed to init trader. Check .env")
            return

    stats["start"] = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        listener = asyncio.create_task(listen_pumpportal())
        scanner = asyncio.create_task(ml_scanner(client))
        price_updater = asyncio.create_task(signal_price_updater())
        enricher = asyncio.create_task(enrich_batch(client))
        reporter = asyncio.create_task(report_printer())
        learner = asyncio.create_task(incremental_learner())
        missed_checker = asyncio.create_task(missed_token_checker())
        cleaner = asyncio.create_task(memory_cleanup())
        saver = asyncio.create_task(auto_saver())

        if CONTINUOUS:
            log.info("CONTINUOUS MODE: bot will run until manually stopped (Ctrl+C)")
            try:
                while True:
                    await asyncio.sleep(3600)
                    log.info(
                        "HEARTBEAT: uptime=%.1fh tokens=%d signals=%d missed_tracked=%d",
                        (time.time() - stats["start"]) / 3600,
                        len(tokens), len(signals), len(missed_tokens),
                    )
            except asyncio.CancelledError:
                log.info("Continuous mode interrupted, shutting down...")
        else:
            await asyncio.sleep(duration)

        log.info(
            "Signal collection ended. Tracking active positions for 2 more minutes..."
        )
        listener.cancel()
        scanner.cancel()
        enricher.cancel()
        learner.cancel()
        missed_checker.cancel()
        cleaner.cancel()
        saver.cancel()

        track_extra = 120
        await asyncio.sleep(track_extra)

        log.info("Tracking period ended. Closing all active positions...")
        for sig in signals:
            if sig["status"] == "ACTIVE":
                final_pnl = sig.get("pnl_pct", 0) or 0
                close_signal(sig, "SESSION_END", final_pnl)

        price_updater.cancel()
        reporter.cancel()

    report = print_signals_report()
    log.info("\n%s", report)

    filepath = save_session()
    log.info("DONE! Session saved to: %s", filepath)

    if REAL_TRADING and trader:
        final_bal = trader.refresh_balance()
        pnl_sol = final_bal - trader.initial_balance
        pnl_usd = pnl_sol * SOL_PRICE_USD
        log.info(
            "REAL WALLET: %.4f -> %.4f SOL (%+.4f SOL / $%+.2f)",
            trader.initial_balance, final_bal, pnl_sol, pnl_usd,
        )


if __name__ == "__main__":
    asyncio.run(main())
