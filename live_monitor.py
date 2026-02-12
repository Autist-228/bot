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

BATCH_DURATION = 1800
current_batch_tokens: dict[str, dict] = {}
pending_batches: list[dict] = []

batch_state: dict = {
    "current_id": 1,
    "current_start": 0.0,
    "current_count": 0,
    "checking_id": 0,
    "checking_progress": 0,
    "checking_total": 0,
    "checking_samples": 0,
    "history": [],
    "total_tokens_fed": 0,
    "total_batches": 0,
}

SOL_PRICE_USD = 200.0
entry_model = None
exit_model = None
entry_mean = None
entry_std = None
exit_mean = None
exit_std = None
seen_mints: set[str] = set()

error_log: list[dict] = []
model_info: dict = {
    "cycles": 0, "loss": 0.0, "initial_loss": 0.0,
    "total_samples": 0, "total_wins": 0,
    "rockets_found": 0, "rockets_missed": 0,
    "last_train_ts": 0, "last_save_time": "---", "file_size_kb": 0,
}

exit_model_info: dict = {
    "cycles": 0, "loss": 0.0, "initial_loss": 0.0,
    "total_samples": 0, "total_signals_used": 0,
    "last_train_ts": 0,
}

telegram_state: dict = {
    "signals": signals,
    "tokens": tokens,
    "stats": stats,
    "errors": error_log,
    "model_info": model_info,
    "batch_state": batch_state,
    "bet_size": BET_SIZE_USD,
    "ws_connected": False,
    "ml_running": False,
    "learner_running": False,
    "exit_model_info": exit_model_info,
}


def log_error(msg: str):
    error_log.append({"ts": time.time(), "msg": msg})
    if len(error_log) > 500:
        error_log.pop(0)


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
    saved_info = state.get("model_info")
    if saved_info:
        for k, v in saved_info.items():
            model_info[k] = v
        log.info("Entry NN loaded (%d features) | restored %d cycles, loss=%.4f", n_feat, model_info["cycles"], model_info["loss"])
    else:
        log.info("Entry NN loaded (%d features)", n_feat)
    if os.path.exists(exit_path):
        xs = torch.load(exit_path, map_location="cpu", weights_only=False)
        n_xf = xs["n_features"]
        exit_model = ExitNet(n_xf)
        exit_model.load_state_dict(xs["model"])
        exit_model.eval()
        exit_mean = np.array(xs["mean"], dtype=np.float32)
        exit_std = np.array(xs["std"], dtype=np.float32)
        saved_exit_info = xs.get("exit_model_info")
        if saved_exit_info:
            for k, v in saved_exit_info.items():
                exit_model_info[k] = v
            log.info("Exit NN loaded (%d features) | restored %d cycles, loss=%.4f", n_xf, exit_model_info["cycles"], exit_model_info["loss"])
        else:
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
    except Exception as e:
        log_error(f"Enrich ошибка {mint[:8]}: {e}")


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
    telegram_state["ml_running"] = True
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age < 30 or age > 360:
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

            bonding_prog = max(0.0, min(100.0, (v_sol - 30.0) / (85.0 - 30.0) * 100))
            token["bonding_progress"] = bonding_prog
            token["log_avg_buy_sol"] = np.log1p(tc.get("buy_sol", 0) / max(1, buys))
            token["log_max_buy_sol"] = np.log1p(tc.get("max_buy_sol", 0))
            token["buy_concentration"] = len(tc.get("buyers", set())) / max(1, buys)
            token["sell_speed"] = sells / max(1, age)

            buyer_amts = tc.get("buyer_amounts", {})
            seller_amts = tc.get("seller_amounts", {})
            net_positions = {}
            for addr, amt in buyer_amts.items():
                net_positions[addr] = amt - seller_amts.get(addr, 0)
            positive = {k: v for k, v in net_positions.items() if v > 0}
            total_pos = sum(positive.values())
            if total_pos > 0 and len(positive) >= 5:
                top5 = sorted(positive.values(), reverse=True)[:5]
                top5_pct = sum(top5) / total_pos * 100
            elif total_pos > 0:
                top5_pct = 100.0
            else:
                top5_pct = 0.0
            token["top5_holder_pct"] = top5_pct
            token["sniper_count"] = float(len(tc.get("early_buyers", set())))
            token["num_holders"] = float(len(tc.get("buyers", set())))

            label, confidence = predict_token(token)
            token["ml_checked"] = True
            token["ml_label"] = label
            token["ml_confidence"] = confidence

            feat_snap = [float(token.get(f, 0) or 0) for f in FEATURES]
            current_batch_tokens[mint] = {
                "mint": mint,
                "symbol": token["symbol"],
                "features": feat_snap,
                "eval_ts": now,
                "entry_v_sol": v_sol,
                "entry_v_tokens": v_tokens,
                "ml_label": label,
                "confidence": confidence,
            }
            batch_state["current_count"] = len(current_batch_tokens)

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
                    "feature_snapshot": [float(token.get(f, 0) or 0) for f in FEATURES],
                    "price_timeline": [],
                }
                signals.append(signal)
                stats["signals"] += 1
                if label == "ROCKET":
                    model_info["rockets_found"] += 1
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
        log_error(f"BUY ошибка {sig['symbol']}: {e}")


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
        log_error(f"SELL ошибка {sig['symbol']}: {e}")
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

                    tl = sig.setdefault("price_timeline", [])
                    elapsed = time.time() - sig["signal_time"]
                    if not tl or elapsed - tl[-1]["t"] >= 5:
                        tl.append({
                            "t": round(elapsed, 1),
                            "pnl": round(current_pnl, 1),
                            "peak": round(sig.get("peak_pnl_pct", 0), 1),
                        })

                    reason = check_exit_rules(sig, current_pnl)
                    if reason:
                        close_signal(sig, reason, current_pnl)
                    else:
                        remaining = sig["position_remaining_pct"] / 100.0
                        unrealized = remaining * (current_pnl / 100.0)
                        total = sig["realized_pnl"] + unrealized
                        sig["pnl_usd"] = round(BET_SIZE_USD * total, 4)
            except Exception as exc:
                log.debug("Price update error for %s: %s", sig.get("symbol", "?"), exc)


async def listen_pumpportal():
    ws = None
    while True:
        try:
            ws = await websockets.connect(PUMPPORTAL_WS_URL)
            log.info("Connected to PumpPortal WebSocket")
            telegram_state["ws_connected"] = True
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
                        "max_buy_sol": 0,
                        "buyers": set(), "sellers": set(),
                        "early_buyers": set(),
                        "buyer_amounts": {},
                        "seller_amounts": {},
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
                        if sol_amount > tc.get("max_buy_sol", 0):
                            tc["max_buy_sol"] = sol_amount
                        if trader_key:
                            tc["buyers"].add(trader_key)
                            tc["buyer_amounts"][trader_key] = tc["buyer_amounts"].get(trader_key, 0) + sol_amount
                            if mint in tokens:
                                token_age = time.time() - tokens[mint]["created_ts"]
                                if token_age <= 5:
                                    tc["early_buyers"].add(trader_key)
                    else:
                        tc["sells"] += 1
                        tc["sell_sol"] += sol_amount
                        if trader_key:
                            tc["sellers"].add(trader_key)
                            tc["seller_amounts"][trader_key] = tc["seller_amounts"].get(trader_key, 0) + sol_amount

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
            telegram_state["ws_connected"] = False
            log.warning("Disconnected, reconnecting in 3s...")
            log_error("WebSocket отключился")
            await asyncio.sleep(3)
        except Exception as e:
            telegram_state["ws_connected"] = False
            log.error("Error: %s, reconnecting in 5s...", e)
            log_error(f"WebSocket ошибка: {e}")
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


MEMORY_CLEANUP_INTERVAL = 1800
TOKEN_MAX_AGE = 3600
HOURLY_STATS_FILE = os.path.join(DATA_DIR, "hourly_stats.json")
BATCH_STATE_FILE = os.path.join(DATA_DIR, "batch_state.json")
MODEL_SNAPSHOT_INTERVAL = 14400


def _save_batch_state():
    try:
        save_data = {
            "current_id": batch_state["current_id"],
            "history": batch_state["history"][-50:],
            "total_tokens_fed": batch_state["total_tokens_fed"],
            "total_batches": batch_state["total_batches"],
        }
        with open(BATCH_STATE_FILE, "w") as f:
            json.dump(save_data, f)
    except Exception as e:
        log.error("batch_state save error: %s", e)


def _load_batch_state():
    if not os.path.exists(BATCH_STATE_FILE):
        return
    try:
        with open(BATCH_STATE_FILE) as f:
            saved = json.load(f)
        batch_state["current_id"] = saved.get("current_id", 1)
        batch_state["history"] = saved.get("history", [])
        batch_state["total_tokens_fed"] = saved.get("total_tokens_fed", 0)
        batch_state["total_batches"] = saved.get("total_batches", 0)
        log.info("Batch state restored: %d batches, %d tokens fed, next=#%d",
                 batch_state["total_batches"], batch_state["total_tokens_fed"],
                 batch_state["current_id"])
    except Exception as e:
        log.error("batch_state load error: %s", e)


def _update_model_file_info():
    entry_path = os.path.join(DATA_DIR, "entry_model.pt")
    if os.path.exists(entry_path):
        model_info["file_size_kb"] = round(os.path.getsize(entry_path) / 1024)
        model_info["last_save_time"] = datetime.fromtimestamp(
            os.path.getmtime(entry_path), tz=timezone.utc
        ).strftime("%H:%M:%S UTC")


def get_reward_weight(pnl: float) -> float:
    if pnl >= 200:
        return 10.0
    if pnl >= 100:
        return 7.0
    if pnl >= 50:
        return 5.0
    if pnl >= 20:
        return 3.0
    if pnl >= 5:
        return 2.0
    if pnl <= -10:
        return 2.0
    return 1.0


def calc_token_pnl(entry_v_sol, entry_v_tokens, cur_v_sol, cur_v_tokens):
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


MAX_EXIT_HOLD_SEC = 900


def generate_live_exit_samples(sig: dict) -> tuple[list, list]:
    timeline = sig.get("price_timeline", [])
    if len(timeline) < 3:
        return [], []
    entry_features = sig.get("feature_snapshot", [])
    if len(entry_features) != len(FEATURES):
        return [], []
    entry_conf = sig.get("ml_confidence", 50.0) / 100.0
    entry_cost = sig.get("entry_cost_pct", 0)

    all_gains = [pt["pnl"] - entry_cost for pt in timeline]
    overall_peak = max(all_gains)
    peak_idx = all_gains.index(overall_peak)

    features_list = []
    labels_list = []
    running_peak = 0.0
    prev_gain = all_gains[0]

    for i, pt in enumerate(timeline):
        gain = all_gains[i]
        if gain > running_peak:
            running_peak = gain
        drop_from_peak = running_peak - gain
        dt = pt["t"] - timeline[max(0, i - 1)]["t"] if i > 0 else 1
        velocity = (gain - prev_gain) / max(1, dt) if i > 0 else 0
        prev_gain = gain

        if i > 0 and i != peak_idx and i % 3 != 0:
            continue

        pos_features = [
            gain / 100.0,
            running_peak / 100.0,
            min(pt["t"] / MAX_EXIT_HOLD_SEC, 1.0),
            drop_from_peak / 100.0,
            float(np.clip(velocity, -1, 1)),
        ]
        full_features = list(entry_features) + [entry_conf] + pos_features

        future_gains = all_gains[i + 1:i + 20]
        if not future_gains:
            sell_label = 1.0
        elif i >= peak_idx and drop_from_peak >= 10:
            sell_label = 1.0
        elif running_peak >= 20 and drop_from_peak >= 12:
            sell_label = 0.9
        elif running_peak >= 15 and gain < 0:
            sell_label = 1.0
        elif future_gains and max(future_gains) > gain + 5:
            sell_label = 0.0
        elif future_gains and max(future_gains) < gain - 3:
            sell_label = 0.8
        elif gain >= overall_peak * 0.85 and gain > 10:
            sell_label = 0.7
        else:
            sell_label = 0.2

        features_list.append(full_features)
        labels_list.append(sell_label)

    return features_list, labels_list


async def _process_batch(batch_id: int, batch_tokens: dict):
    global entry_model, exit_model, exit_mean, exit_std
    total = len(batch_tokens)
    batch_state["checking_id"] = batch_id
    batch_state["checking_total"] = total
    batch_state["checking_progress"] = 0
    batch_state["checking_samples"] = 0

    samples = []
    rockets_found = 0
    predicted_rockets = 0
    correct_rockets = 0
    correct_trash = 0

    unchecked = set(batch_tokens.keys())
    while unchecked:
        await asyncio.sleep(10)
        now = time.time()
        done_this_round = []
        for mint in list(unchecked):
            bt = batch_tokens[mint]
            age = now - bt["eval_ts"]
            if age < BATCH_DURATION:
                continue
            token_data = tokens.get(mint)
            if token_data:
                cur_v_sol = token_data.get("v_sol_in_bonding", 0)
                cur_v_tokens = token_data.get("v_tokens_in_bonding", 0)
                hyp_pnl = calc_token_pnl(
                    bt["entry_v_sol"], bt["entry_v_tokens"],
                    cur_v_sol, cur_v_tokens,
                )
                if hyp_pnl is not None:
                    is_profitable = 1.0 if hyp_pnl >= 5.0 else 0.0
                    weight = get_reward_weight(hyp_pnl)
                    samples.append({
                        "features": bt["features"],
                        "label": is_profitable,
                        "weight": weight,
                    })
                    actually_pumped = hyp_pnl >= 50.0
                    model_said_rocket = bt["ml_label"] in ("ROCKET", "winner")
                    if model_said_rocket:
                        predicted_rockets += 1
                    if actually_pumped:
                        rockets_found += 1
                    if model_said_rocket and actually_pumped:
                        correct_rockets += 1
                    if not model_said_rocket and not actually_pumped:
                        correct_trash += 1
            done_this_round.append(mint)

        for mint in done_this_round:
            unchecked.discard(mint)
            batch_state["checking_progress"] += 1
            batch_state["checking_samples"] = len(samples)

        if not unchecked:
            break
        earliest = min((batch_tokens[m]["eval_ts"] for m in unchecked), default=now)
        if now - earliest > BATCH_DURATION + 300:
            batch_state["checking_progress"] += len(unchecked)
            unchecked.clear()
            break

    fed_count = len(samples)
    final_loss = 0.0
    wins = 0
    accuracy = 0.0

    if fed_count >= 3 and entry_model is not None:
        X = np.array([s["features"] for s in samples], dtype=np.float32)
        y = np.array([s["label"] for s in samples], dtype=np.float32)
        w = np.array([s["weight"] for s in samples], dtype=np.float32)
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
        final_loss = loss.item()

        model_info["cycles"] += 1
        model_info["loss"] = final_loss
        if model_info["initial_loss"] == 0:
            model_info["initial_loss"] = final_loss
        model_info["total_samples"] += fed_count
        model_info["total_wins"] += wins
        model_info["last_train_ts"] = time.time()
        model_info["rockets_found"] += predicted_rockets
        model_info["rockets_missed"] += rockets_found

        entry_path = os.path.join(DATA_DIR, "entry_model.pt")
        torch.save({
            "model": entry_model.state_dict(),
            "n_features": len(FEATURES),
            "mean": entry_mean.tolist(),
            "std": entry_std.tolist(),
            "model_info": dict(model_info),
        }, entry_path)
        _update_model_file_info()

        total_correct = correct_rockets + correct_trash
        accuracy = total_correct / max(1, fed_count) * 100

        log.info(
            "BATCH #%d FED: %d/%d tokens | %d rockets | %d wins (%.0f%%) | "
            "loss=%.4f | accuracy=%.1f%%",
            batch_id, fed_count, total, rockets_found,
            wins, wins / max(1, fed_count) * 100, final_loss, accuracy,
        )

    else:
        log.info("BATCH #%d: %d valid samples (need >=3), skipping", batch_id, fed_count)

    exit_samples_X = []
    exit_samples_y = []
    exit_sigs_used = 0
    exit_loss_val = 0.0
    exit_fed = 0

    for sig in signals:
        if sig["status"] != "CLOSED":
            continue
        if sig.get("exit_trained"):
            continue
        tl = sig.get("price_timeline", [])
        if len(tl) < 3:
            continue
        xf, xl = generate_live_exit_samples(sig)
        if xf:
            exit_samples_X.extend(xf)
            exit_samples_y.extend(xl)
            exit_sigs_used += 1
            sig["exit_trained"] = True

    exit_fed = len(exit_samples_X)

    if exit_fed >= 5:
        eX = np.array(exit_samples_X, dtype=np.float32)
        ey = np.array(exit_samples_y, dtype=np.float32)
        n_exit_features = eX.shape[1]

        if exit_model is None or exit_mean is None or exit_mean.shape[0] != n_exit_features:
            exit_model = ExitNet(n_exit_features)
            exit_mean = eX.mean(axis=0)
            exit_std = eX.std(axis=0)
            exit_std[exit_std < 1e-6] = 1.0
            log.info("EXIT MODEL created fresh: %d features (15 entry + 1 conf + 5 position)", n_exit_features)

        eX_n = (eX - exit_mean) / exit_std
        eX_t = torch.from_numpy(eX_n)
        ey_t = torch.from_numpy(ey)

        exit_model.train()
        e_opt = torch.optim.Adam(exit_model.parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(10):
            e_opt.zero_grad()
            e_pred = exit_model(eX_t)
            e_loss = torch.nn.functional.binary_cross_entropy(e_pred, ey_t)
            e_loss.backward()
            torch.nn.utils.clip_grad_norm_(exit_model.parameters(), 1.0)
            e_opt.step()
        exit_model.eval()
        exit_loss_val = e_loss.item()

        new_mean = eX.mean(axis=0)
        new_std = eX.std(axis=0)
        new_std[new_std < 1e-6] = 1.0
        exit_mean = 0.9 * exit_mean + 0.1 * new_mean
        exit_std = 0.9 * exit_std + 0.1 * new_std

        exit_model_info["cycles"] += 1
        exit_model_info["loss"] = exit_loss_val
        if exit_model_info["initial_loss"] == 0:
            exit_model_info["initial_loss"] = exit_loss_val
        exit_model_info["total_samples"] += exit_fed
        exit_model_info["total_signals_used"] += exit_sigs_used
        exit_model_info["last_train_ts"] = time.time()

        exit_path = os.path.join(DATA_DIR, "exit_model.pt")
        torch.save({
            "model": exit_model.state_dict(),
            "n_features": n_exit_features,
            "mean": exit_mean.tolist(),
            "std": exit_std.tolist(),
            "exit_model_info": dict(exit_model_info),
        }, exit_path)

        log.info(
            "EXIT BATCH #%d: %d samples from %d signals | loss=%.4f",
            batch_id, exit_fed, exit_sigs_used, exit_loss_val,
        )

    elif exit_fed > 0:
        log.info("EXIT BATCH #%d: %d samples (need >=5), skipping", batch_id, exit_fed)
    else:
        log.info("EXIT BATCH #%d: no closed signals with timelines", batch_id)

    batch_record = {
        "id": batch_id,
        "tokens_total": total,
        "tokens_fed": fed_count,
        "rockets": rockets_found,
        "predicted_rockets": predicted_rockets,
        "correct_rockets": correct_rockets,
        "accuracy": round(accuracy, 1),
        "wins": wins,
        "win_pct": round(wins / max(1, fed_count) * 100, 1),
        "loss": round(final_loss, 4),
        "exit_samples": exit_fed,
        "exit_signals": exit_sigs_used,
        "exit_loss": round(exit_loss_val, 4),
        "fed_ts": time.time(),
    }
    batch_state["history"].append(batch_record)
    batch_state["total_tokens_fed"] += fed_count
    batch_state["total_batches"] += 1

    batch_state["checking_id"] = 0
    batch_state["checking_progress"] = 0
    batch_state["checking_total"] = 0
    batch_state["checking_samples"] = 0
    _save_batch_state()


async def batch_processor():
    batch_state["current_start"] = time.time()
    telegram_state["learner_running"] = True
    processing_task = None

    while True:
        await asyncio.sleep(30)
        now = time.time()

        elapsed = now - batch_state["current_start"]
        if elapsed >= BATCH_DURATION and len(current_batch_tokens) > 0:
            batch_id = batch_state["current_id"]
            closed_tokens = dict(current_batch_tokens)
            current_batch_tokens.clear()

            batch_state["current_id"] += 1
            batch_state["current_start"] = now
            batch_state["current_count"] = 0

            log.info(
                "BATCH #%d closed: %d tokens. Batch #%d collection started.",
                batch_id, len(closed_tokens), batch_state["current_id"],
            )
            pending_batches.append({"id": batch_id, "tokens": closed_tokens})

        if pending_batches and (processing_task is None or processing_task.done()):
            batch = pending_batches.pop(0)
            processing_task = asyncio.create_task(
                _process_batch(batch["id"], batch["tokens"])
            )

        batch_state["current_count"] = len(current_batch_tokens)


async def hourly_stats_saver():
    while True:
        await asyncio.sleep(3600)
        try:
            now = time.time()
            hour_ago = now - 3600
            hour_sigs = [s for s in signals if s.get("signal_time", 0) >= hour_ago]
            closed_hour = [s for s in hour_sigs if s["status"] == "CLOSED"]
            wins = sum(1 for s in closed_hour if (s.get("close_pnl_pct", 0) or 0) > 0)
            losses = len(closed_hour) - wins
            total_pnl = sum(s.get("pnl_usd", 0) or 0 for s in closed_hour)

            last_acc = batch_state["history"][-1]["accuracy"] if batch_state["history"] else 0
            snapshot = {
                "ts": now,
                "datetime": datetime.now(timezone.utc).isoformat(),
                "uptime_hours": round((now - stats["start"]) / 3600, 1),
                "tokens_scanned": stats["total"],
                "signals_total": stats["signals"],
                "hour_signals": len(hour_sigs),
                "hour_closed": len(closed_hour),
                "hour_wins": wins,
                "hour_losses": losses,
                "hour_win_rate": round(wins / max(1, wins + losses) * 100, 1),
                "hour_pnl_usd": round(total_pnl, 2),
                "batches_completed": batch_state["total_batches"],
                "tokens_fed_total": batch_state["total_tokens_fed"],
                "model_loss": round(model_info["loss"], 4),
                "model_cycles": model_info["cycles"],
                "model_accuracy": last_acc,
                "exit_loss": round(exit_model_info["loss"], 4),
                "exit_cycles": exit_model_info["cycles"],
                "exit_samples": exit_model_info["total_samples"],
            }

            existing = []
            if os.path.exists(HOURLY_STATS_FILE):
                try:
                    with open(HOURLY_STATS_FILE) as f:
                        existing = json.load(f)
                except Exception:
                    pass
            existing.append(snapshot)
            with open(HOURLY_STATS_FILE, "w") as f:
                json.dump(existing, f, indent=2)

            log.info(
                "HOURLY SNAPSHOT #%d: sigs=%d W=%d L=%d pnl=$%.2f loss=%.4f",
                len(existing), len(hour_sigs), wins, losses, total_pnl, model_info["loss"],
            )
        except Exception as e:
            log.error("HOURLY SNAPSHOT error: %s", e)
            log_error(f"Hourly snapshot: {e}")


async def model_snapshot_saver():
    while True:
        await asyncio.sleep(MODEL_SNAPSHOT_INTERVAL)
        try:
            import shutil
            hours = int((time.time() - stats["start"]) / 3600)
            last_acc = batch_state["history"][-1]["accuracy"] if batch_state["history"] else 0

            entry_path = os.path.join(DATA_DIR, "entry_model.pt")
            if os.path.exists(entry_path):
                shutil.copy2(entry_path, os.path.join(DATA_DIR, f"entry_model_h{hours}.pt"))
                log.info("ENTRY SNAPSHOT h%d: loss=%.4f, cycles=%d, samples=%d",
                         hours, model_info["loss"], model_info["cycles"], model_info["total_samples"])

            exit_path = os.path.join(DATA_DIR, "exit_model.pt")
            if os.path.exists(exit_path):
                shutil.copy2(exit_path, os.path.join(DATA_DIR, f"exit_model_h{hours}.pt"))
                log.info("EXIT SNAPSHOT h%d: loss=%.4f, cycles=%d, samples=%d",
                         hours, exit_model_info["loss"], exit_model_info["cycles"], exit_model_info["total_samples"])

            meta = {
                "hour": hours,
                "entry": {
                    "cycles": model_info["cycles"],
                    "loss": model_info["loss"],
                    "total_samples": model_info["total_samples"],
                    "total_wins": model_info["total_wins"],
                    "accuracy": last_acc,
                },
                "exit": {
                    "cycles": exit_model_info["cycles"],
                    "loss": exit_model_info["loss"],
                    "total_samples": exit_model_info["total_samples"],
                    "total_signals_used": exit_model_info["total_signals_used"],
                },
                "batches": batch_state["total_batches"],
                "tokens_fed": batch_state["total_tokens_fed"],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path = os.path.join(DATA_DIR, f"model_meta_h{hours}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            log.error("MODEL SNAPSHOT error: %s", e)
            log_error(f"Model snapshot: {e}")


async def memory_cleanup():
    while True:
        await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
        now = time.time()
        signal_mints = {s["mint"] for s in signals if s["status"] == "ACTIVE"}
        batch_mints = set(current_batch_tokens.keys())
        for pb in pending_batches:
            batch_mints.update(pb["tokens"].keys())
        stale = []
        for mint, t in tokens.items():
            if mint in signal_mints or mint in batch_mints:
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
            log_error(f"Autosave ошибка: {e}")


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

    _load_batch_state()

    if REAL_TRADING:
        init_trader()
        if not trader:
            log.error("Failed to init trader. Check .env")
            return

    stats["start"] = time.time()
    _update_model_file_info()

    tg_bot = None
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
    if tg_token and tg_chat:
        from telegram_bot import SniperTelegramBot
        tg_bot = SniperTelegramBot(tg_token, tg_chat, telegram_state)

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        listener = asyncio.create_task(listen_pumpportal())
        scanner = asyncio.create_task(ml_scanner(client))
        price_updater = asyncio.create_task(signal_price_updater())
        enricher = asyncio.create_task(enrich_batch(client))
        reporter = asyncio.create_task(report_printer())
        batch_proc = asyncio.create_task(batch_processor())
        cleaner = asyncio.create_task(memory_cleanup())
        saver = asyncio.create_task(auto_saver())
        hourly_saver = asyncio.create_task(hourly_stats_saver())
        model_snapper = asyncio.create_task(model_snapshot_saver())

        if tg_bot:
            await tg_bot.start()

        if CONTINUOUS:
            log.info("CONTINUOUS MODE: bot will run until manually stopped (Ctrl+C)")
            try:
                while True:
                    await asyncio.sleep(3600)
                    log.info(
                        "HEARTBEAT: uptime=%.1fh tokens=%d signals=%d missed_tracked=%d",
                        (time.time() - stats["start"]) / 3600,
                        len(tokens), len(signals), batch_state["total_batches"],
                    )
            except asyncio.CancelledError:
                log.info("Continuous mode interrupted, shutting down...")
        else:
            await asyncio.sleep(duration)

        if tg_bot:
            await tg_bot.stop()

        log.info(
            "Signal collection ended. Tracking active positions for 2 more minutes..."
        )
        listener.cancel()
        scanner.cancel()
        enricher.cancel()
        batch_proc.cancel()
        cleaner.cancel()
        saver.cancel()
        hourly_saver.cancel()
        model_snapper.cancel()

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
