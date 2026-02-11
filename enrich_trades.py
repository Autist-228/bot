#!/usr/bin/env python3
"""Enrich historical tokens with full trade arrays from blockchain.
Combo: Helius Enhanced API (fast) + public Solana RPC (fallback).
Hot-reloads keys from file. Saves progress. Never crashes.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timezone

import requests
from config import DATA_DIR

PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
KEYS_FILE = os.path.join(DATA_DIR, "helius_keys.txt")
BATCH_PARSE = 100
SAVE_EVERY = 10
MAX_SIG_PAGES = 10
MAX_PARSE_SIGS = 500
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("enrich")

_lock = threading.Lock()
_stats = {"done": 0, "trades": 0, "errors": 0, "helius_calls": 0, "public_calls": 0}


def load_keys():
    keys = set()
    env_keys = os.getenv("HELIUS_API_KEYS", "")
    for k in env_keys.split(","):
        k = k.strip()
        if k:
            keys.add(k)
    if not os.getenv("HELIUS_SINGLE_KEY") and os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            for line in f:
                k = line.strip()
                if k and not k.startswith("#"):
                    keys.add(k)
    return list(keys)


_tls = threading.local()

def _get_session():
    if not hasattr(_tls, 'session'):
        s = requests.Session()
        s.headers["Content-Type"] = "application/json"
        _tls.session = s
    return _tls.session


class RPCClient:
    def __init__(self):
        self._helius_keys = []
        self._dead_keys = set()
        self._key_idx = 0
        self._key_lock = threading.Lock()
        self._public_lock = threading.Lock()
        self._last_public = 0.0
        self._public_interval = 0.3
        self.reload_keys()

    def reload_keys(self):
        new_keys = load_keys()
        with self._key_lock:
            added = [k for k in new_keys if k not in set(self._helius_keys) and k not in self._dead_keys]
            if added:
                self._helius_keys.extend(added)
                log.info("Keys: %d active, %d dead, %d new",
                         len(self._helius_keys) - len(self._dead_keys), len(self._dead_keys), len(added))

    def _get_live_key(self):
        with self._key_lock:
            if not self._helius_keys:
                return None
            for _ in range(len(self._helius_keys)):
                k = self._helius_keys[self._key_idx % len(self._helius_keys)]
                self._key_idx += 1
                if k not in self._dead_keys:
                    return k
            return None

    def _mark_dead(self, key):
        with self._key_lock:
            self._dead_keys.add(key)
            alive = len(self._helius_keys) - len(self._dead_keys)
        log.warning("Key %s...%s DEAD. %d keys alive", key[:8], key[-4:], alive)

    def _wait_public(self):
        with self._public_lock:
            now = time.monotonic()
            wait = self._public_interval - (now - self._last_public)
            if wait > 0:
                time.sleep(wait)
            self._last_public = time.monotonic()

    def _public_rpc_call(self, method, params, timeout=20):
        self._wait_public()
        with _lock:
            _stats["public_calls"] += 1
        session = _get_session()
        for attempt in range(10):
            try:
                r = session.post(PUBLIC_RPC,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                    timeout=timeout)
                if r.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                d = r.json()
                if "error" in d:
                    err_msg = str(d["error"])
                    if "429" in err_msg or "Too many" in err_msg:
                        time.sleep(2 + attempt * 2)
                        continue
                return d
            except Exception as e:
                if attempt < 9:
                    time.sleep(1 + attempt)
                else:
                    log.warning("Public RPC fail after 10 tries: %s", e)
        return None

    def get_signatures(self, mint, limit=1000):
        all_sigs = []
        cursor = None
        for page in range(MAX_SIG_PAGES):
            params = [mint, {"limit": limit, "commitment": "confirmed"}]
            if cursor:
                params[1]["before"] = cursor
            result = None
            key = self._get_live_key()
            if key:
                with _lock:
                    _stats["helius_calls"] += 1
                session = _get_session()
                for attempt in range(3):
                    try:
                        r = session.post(
                            f"https://mainnet.helius-rpc.com/?api-key={key}",
                            json={"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress", "params": params},
                            timeout=20)
                        if r.status_code == 429:
                            time.sleep(1 + attempt)
                            continue
                        d = r.json()
                        if "error" in d:
                            err_str = str(d["error"])
                            if "429" in err_str or "limit" in err_str.lower() or "usage" in err_str.lower():
                                self._mark_dead(key)
                                break
                        result = d
                        break
                    except Exception:
                        time.sleep(1)
            if result is None:
                result = self._public_rpc_call("getSignaturesForAddress", params)
            if not result:
                break
            sigs = result.get("result", [])
            if not sigs:
                break
            all_sigs.extend(s["signature"] for s in sigs if not s.get("err"))
            if len(sigs) < limit:
                break
            cursor = sigs[-1]["signature"]
        return all_sigs

    def parse_helius(self, signatures):
        key = self._get_live_key()
        if not key:
            return None
        with _lock:
            _stats["helius_calls"] += 1
        session = _get_session()
        for attempt in range(3):
            try:
                r = session.post(
                    f"https://api-mainnet.helius-rpc.com/v0/transactions/?api-key={key}",
                    json={"transactions": signatures}, timeout=30)
                if r.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                if r.status_code == 200:
                    return r.json()
                err_text = r.text[:200]
                if "limit" in err_text.lower() or "usage" in err_text.lower():
                    self._mark_dead(key)
                    return None
                return None
            except Exception:
                time.sleep(1)
        return None

    def parse_public(self, sig):
        result = self._public_rpc_call("getTransaction",
            [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}], timeout=15)
        if result and "result" in result:
            return result["result"]
        return None

    @property
    def num_alive(self):
        with self._key_lock:
            return len(self._helius_keys) - len(self._dead_keys)


def extract_helius(txs, target_mint):
    trades = []
    for tx in txs:
        if tx.get("source") != "PUMP_FUN":
            continue
        ts = tx.get("timestamp", 0)
        if not ts:
            continue
        fp = tx.get("feePayer", "")
        mint = None
        tok_amt = 0.0
        for tt in tx.get("tokenTransfers", []):
            m = tt.get("mint", "")
            if m and m != "So11111111111111111111111111111111111111112":
                mint = m
                tok_amt = float(tt.get("tokenAmount", 0) or 0)
                break
        if mint != target_mint:
            continue
        sol_in = sol_out = 0.0
        for nt in tx.get("nativeTransfers", []):
            a = float(nt.get("amount", 0)) / 1e9
            if a < 0.0001:
                continue
            if nt.get("fromUserAccount") == fp:
                sol_in += a
            elif nt.get("toUserAccount") == fp:
                sol_out += a
        is_buy = sol_in > sol_out
        sol = sol_in if is_buy else sol_out
        price = sol / tok_amt if tok_amt > 0 and sol > 0 else 0.0
        trades.append({"ts": ts, "type": "buy" if is_buy else "sell",
                       "sol": round(sol, 6), "trader": fp, "price_sol": price})
    return trades


def extract_public(tx_data, target_mint):
    if not tx_data or not tx_data.get("meta"):
        return None
    meta = tx_data["meta"]
    if meta.get("err"):
        return None
    ts = tx_data.get("blockTime", 0)
    if not ts:
        return None
    msg = tx_data.get("transaction", {}).get("message", {})
    account_keys = []
    for ak in msg.get("accountKeys", []):
        if isinstance(ak, dict):
            account_keys.append(ak.get("pubkey", ""))
        else:
            account_keys.append(str(ak))
    if PUMPFUN_PROGRAM not in account_keys:
        return None
    fee_payer = account_keys[0] if account_keys else ""
    pre_balances = {}
    post_balances = {}
    for b in meta.get("preTokenBalances", []):
        m = b.get("mint", "")
        if m and m != "So11111111111111111111111111111111111111112":
            owner = b.get("owner", "")
            amt = float(b.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
            pre_balances[(m, owner)] = amt
    for b in meta.get("postTokenBalances", []):
        m = b.get("mint", "")
        if m and m != "So11111111111111111111111111111111111111112":
            owner = b.get("owner", "")
            amt = float(b.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
            post_balances[(m, owner)] = amt
    fp_pre = pre_balances.get((target_mint, fee_payer), 0)
    fp_post = post_balances.get((target_mint, fee_payer), 0)
    tok_change = fp_post - fp_pre
    pre_sol = meta.get("preBalances", [])
    post_sol = meta.get("postBalances", [])
    sol_change = 0.0
    if pre_sol and post_sol:
        sol_change = (post_sol[0] - pre_sol[0]) / 1e9
    if abs(tok_change) < 1:
        return None
    is_buy = tok_change > 0
    sol_amount = abs(sol_change)
    tok_amt = abs(tok_change)
    price = sol_amount / tok_amt if tok_amt > 0 and sol_amount > 0 else 0.0
    return {"ts": ts, "type": "buy" if is_buy else "sell",
            "sol": round(sol_amount, 6), "trader": fee_payer, "price_sol": price}


def enrich_token(rpc, token, idx=0, total=0):
    mint = token["mint"]
    name = token.get("name", mint[:12])
    t0 = time.time()
    sigs = rpc.get_signatures(mint)
    if not sigs:
        token["trades"] = []
        return 0
    total_sigs = len(sigs)
    if total_sigs > MAX_PARSE_SIGS:
        sigs = sigs[-MAX_PARSE_SIGS:]
    log.info("[%d/%d] %s: %d sigs (of %d), parsing...", idx, total, name, len(sigs), total_sigs)
    all_trades = []
    i = 0
    while i < len(sigs):
        batch = sigs[i:i + BATCH_PARSE]
        parsed = rpc.parse_helius(batch)
        if parsed is not None:
            all_trades.extend(extract_helius(parsed, mint))
            i += BATCH_PARSE
        else:
            for sig in batch:
                tx_data = rpc.parse_public(sig)
                if tx_data:
                    trade = extract_public(tx_data, mint)
                    if trade:
                        all_trades.append(trade)
            i += BATCH_PARSE
    all_trades.sort(key=lambda t: t["ts"])
    seen = set()
    uniq = []
    for t in all_trades:
        k = (t["ts"], t["type"], t["trader"], round(t["sol"], 4))
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    token["trades"] = uniq
    return len(uniq)


def save_data(data, path, tag):
    toks = data["tokens"]
    data["meta"]["enrichment"] = {
        "tag": tag, "ts": datetime.now(timezone.utc).isoformat(),
        "enriched": sum(1 for t in toks if t.get("trades") is not None),
        "with_trades": sum(1 for t in toks if t.get("trades") and len(t["trades"]) > 0),
        "total_trades": sum(len(t.get("trades", [])) for t in toks),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)
    sz = os.path.getsize(path) / 1e6
    log.info("SAVED [%s] %s (%.1fMB) enriched=%d with_trades=%d trades=%d",
             tag, os.path.basename(path), sz,
             data["meta"]["enrichment"]["enriched"],
             data["meta"]["enrichment"]["with_trades"],
             data["meta"]["enrichment"]["total_trades"])


def run_single(args):
    out = args.output or args.input.replace(".json", "_enriched.json")
    if args.resume and os.path.exists(out):
        log.info("RESUMING from %s", out)
        with open(out) as f:
            data = json.load(f)
    else:
        log.info("Loading %s", args.input)
        with open(args.input) as f:
            data = json.load(f)

    tokens = data["tokens"]
    need = [t for t in tokens if t.get("trades") is None]
    already = len(tokens) - len(need)
    rpc = RPCClient()

    log.info("=" * 60)
    log.info("ENRICHMENT: %d tokens, %d need (already=%d), keys=%d", len(tokens), len(need), already, rpc.num_alive)
    log.info("MAX_SIG_PAGES=%d MAX_PARSE_SIGS=%d SAVE_EVERY=%d", MAX_SIG_PAGES, MAX_PARSE_SIGS, SAVE_EVERY)
    log.info("Output: %s", out)
    log.info("=" * 60)

    _emergency = {"data": data, "out": out, "done": 0}

    def _save_on_signal(signum, frame):
        log.warning("SIGNAL %d received! Emergency save...", signum)
        save_data(_emergency["data"], _emergency["out"], f"emergency_{_emergency['done']}")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _save_on_signal)
    signal.signal(signal.SIGINT, _save_on_signal)

    start = time.time()
    last_key_reload = time.time()
    done = 0
    total_trades = 0
    errors = 0

    for idx, token in enumerate(need):
        if time.time() - last_key_reload > 30:
            rpc.reload_keys()
            last_key_reload = time.time()
        try:
            n = enrich_token(rpc, token, idx=already + idx + 1, total=len(tokens))
            done += 1
            total_trades += n
        except Exception as e:
            log.warning("Error %s: %s", token.get("mint", "?")[:16], e)
            token["trades"] = []
            done += 1
            errors += 1

        _emergency["done"] = already + done

        if done % 50 == 0 or done >= len(need):
            elapsed = time.time() - start
            rate = done / max(1, elapsed)
            eta_min = (len(need) - done) / max(0.01, rate) / 60
            pct = (already + done) / len(tokens) * 100
            log.info("[%5.0fs] %d/%d (%.1f%%) trades=%d %.1f/s ETA=%.0fm err=%d keys=%d",
                     elapsed, done, len(need), pct, total_trades, rate, eta_min, errors, rpc.num_alive)
        if done % SAVE_EVERY == 0:
            save_data(data, out, f"progress_{already + done}")

    elapsed = time.time() - start
    w = sum(1 for t in tokens if t.get("trades") and len(t["trades"]) > 0)
    log.info("=" * 60)
    log.info("DONE %.0fs (%.1fmin) | with_trades=%d/%d | trades=%d | err=%d",
             elapsed, elapsed / 60, w, len(tokens), total_trades, errors)
    log.info("=" * 60)
    save_data(data, out, "final")
    return out


def run_parallel(args):
    import subprocess
    import math

    log.info("Loading %s for splitting...", args.input)
    with open(args.input) as f:
        data = json.load(f)

    tokens = data["tokens"]
    keys = load_keys()
    n_workers = len(keys)
    if n_workers == 0:
        log.error("No Helius keys found!")
        return

    chunk_size = math.ceil(len(tokens) / n_workers)
    chunk_files = []
    out_files = []

    for i in range(n_workers):
        chunk = tokens[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            continue
        chunk_path = os.path.join(DATA_DIR, f"_chunk_{i}.json")
        out_path = os.path.join(DATA_DIR, f"_chunk_{i}_enriched.json")
        if os.path.exists(out_path):
            log.info("Chunk %d: RESUME from existing %s", i, os.path.basename(out_path))
        else:
            chunk_data = {"meta": dict(data["meta"]), "tokens": chunk}
            with open(chunk_path, "w") as f:
                json.dump(chunk_data, f, default=str)
            log.info("Chunk %d: %d tokens → %s (key=%s...%s)",
                     i, len(chunk), os.path.basename(chunk_path), keys[i][:8], keys[i][-4:])
        chunk_files.append(chunk_path)
        out_files.append(out_path)

    log.info("Launching %d parallel processes...", len(chunk_files))
    procs = []
    for i, (cf, of) in enumerate(zip(chunk_files, out_files)):
        env = dict(os.environ)
        env["HELIUS_API_KEYS"] = keys[i]
        env["HELIUS_SINGLE_KEY"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = ["python3", "enrich_trades.py", "--input", cf, "--output", of, "--mode", "single", "--resume"]
        p = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        procs.append((i, p))
        log.info("  Process %d started (PID %d)", i, p.pid)

    start = time.time()
    try:
        for i, p in procs:
            p.wait()
            elapsed = time.time() - start
            log.info("  Process %d finished (rc=%d, %.0fs)", i, p.returncode, elapsed)
    except KeyboardInterrupt:
        log.warning("INTERRUPTED! Sending SIGTERM to all children...")
        for i, p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except OSError:
                pass
        for i, p in procs:
            p.wait()
        log.warning("All children stopped. Data saved in chunk files. Re-run to resume.")
        return

    log.info("Merging %d chunks...", len(out_files))
    all_tokens = []
    total_trades = 0
    with_trades = 0
    for of in out_files:
        if not os.path.exists(of):
            log.warning("Missing output: %s", of)
            continue
        with open(of) as f:
            chunk_data = json.load(f)
        for t in chunk_data["tokens"]:
            all_tokens.append(t)
            trades = t.get("trades", [])
            total_trades += len(trades)
            if trades:
                with_trades += 1

    final_out = args.input.replace(".json", "_enriched.json")
    final_data = {"meta": data["meta"], "tokens": all_tokens}
    save_data(final_data, final_out, "merged_final")

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("ALL DONE %.0fs (%.1fmin) | %d tokens | with_trades=%d | trades=%d",
             elapsed, elapsed / 60, len(all_tokens), with_trades, total_trades)
    log.info("Output: %s", final_out)
    log.info("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "historical_12h_20260211_041106.json"))
    ap.add_argument("--output", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--mode", default="parallel", choices=["single", "parallel"])
    args = ap.parse_args()

    if args.mode == "parallel":
        run_parallel(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
