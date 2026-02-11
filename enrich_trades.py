#!/usr/bin/env python3
"""Enrich historical tokens with full trade arrays from blockchain.
Uses synchronous requests + ThreadPoolExecutor (1 thread per API key).
"""

import argparse
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from config import DATA_DIR

API_KEYS = [k.strip() for k in os.getenv("HELIUS_API_KEYS", "").split(",") if k.strip()]
BATCH_PARSE = 100
RPS_PER_KEY = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("enrich")


class KeyWorker:
    def __init__(self, key, rps=RPS_PER_KEY):
        self.key = key
        self.rpc = f"https://mainnet.helius-rpc.com/?api-key={key}"
        self.enh = f"https://api-mainnet.helius-rpc.com/v0/transactions/?api-key={key}"
        self.interval = 1.0 / rps
        self.last = 0.0
        self.session = requests.Session()

    def _wait(self):
        now = time.monotonic()
        wait = self.interval - (now - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()

    def get_sigs(self, mint):
        all_sigs = []
        cursor = None
        for _ in range(200):
            self._wait()
            params = [mint, {"limit": 1000, "commitment": "confirmed"}]
            if cursor:
                params[1]["before"] = cursor
            for attempt in range(5):
                try:
                    r = self.session.post(self.rpc, json={"jsonrpc":"2.0","id":1,"method":"getSignaturesForAddress","params":params}, timeout=20)
                    if r.status_code == 429:
                        time.sleep(3 + attempt * 3)
                        continue
                    d = r.json()
                    if "error" in d and "429" in str(d["error"]):
                        time.sleep(3 + attempt * 3)
                        continue
                    sigs = d.get("result", [])
                    break
                except Exception:
                    time.sleep(2 + attempt)
                    sigs = []
            else:
                break
            if not sigs:
                break
            all_sigs.extend(s["signature"] for s in sigs if not s.get("err"))
            if len(sigs) < 1000:
                break
            cursor = sigs[-1]["signature"]
        return all_sigs

    def parse_sigs(self, signatures):
        self._wait()
        for attempt in range(5):
            try:
                r = self.session.post(self.enh, json={"transactions": signatures}, timeout=30)
                if r.status_code == 429:
                    time.sleep(3 + attempt * 3)
                    continue
                return r.json() if r.status_code == 200 else []
            except Exception:
                time.sleep(2 + attempt)
        return []

    def enrich_token(self, token):
        mint = token["mint"]
        sigs = self.get_sigs(mint)
        if not sigs:
            token["trades"] = []
            return 0
        all_trades = []
        for i in range(0, len(sigs), BATCH_PARSE):
            parsed = self.parse_sigs(sigs[i:i+BATCH_PARSE])
            if parsed:
                all_trades.extend(_extract(parsed, mint))
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


def _extract(txs, target_mint):
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
        trades.append({"ts": ts, "type": "buy" if is_buy else "sell", "sol": round(sol, 6), "trader": fp, "price_sol": price})
    return trades


_progress_lock = threading.Lock()
_stats = {"done": 0, "trades": 0, "errors": 0}


def _process_token(worker, token):
    try:
        n = worker.enrich_token(token)
        with _progress_lock:
            _stats["done"] += 1
            _stats["trades"] += n
        return n
    except Exception as e:
        with _progress_lock:
            _stats["done"] += 1
            _stats["errors"] += 1
        log.warning("Error %s: %s", token["mint"][:16], e)
        token["trades"] = []
        return 0


def save(data, path, tag):
    toks = data["tokens"]
    data["meta"]["enrichment"] = {
        "tag": tag, "ts": datetime.now(timezone.utc).isoformat(),
        "enriched": sum(1 for t in toks if t.get("trades") is not None),
        "with_trades": sum(1 for t in toks if t.get("trades") and len(t["trades"]) > 0),
        "total_trades": sum(len(t.get("trades", [])) for t in toks),
    }
    with open(path, "w") as f:
        json.dump(data, f, default=str)
    log.info("Saved [%s] %s (%.1f MB)", tag, path, os.path.getsize(path)/1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "historical_12h_20260211_041106.json"))
    ap.add_argument("--output", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not API_KEYS:
        log.error("Set HELIUS_API_KEYS"); return

    out = args.output or args.input.replace(".json", "_enriched.json")
    if args.resume and os.path.exists(out):
        log.info("Resuming from %s", out)
        with open(out) as f: data = json.load(f)
    else:
        with open(args.input) as f: data = json.load(f)

    tokens = data["tokens"]
    if args.limit > 0:
        tokens = tokens[:args.limit]; data["tokens"] = tokens

    need = [t for t in tokens if t.get("trades") is None]
    log.info("=" * 50)
    log.info("ENRICHMENT: %d tokens, %d need enrichment", len(tokens), len(need))
    log.info("Keys=%d | RPS/key=%d | Total RPS=%d", len(API_KEYS), RPS_PER_KEY, len(API_KEYS)*RPS_PER_KEY)
    log.info("Output: %s", out)
    log.info("=" * 50)

    workers = [KeyWorker(k) for k in API_KEYS]
    start = time.time()

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {}
        for i, token in enumerate(need):
            w = workers[i % len(workers)]
            f = pool.submit(_process_token, w, token)
            futures[f] = i

        last_log = 0
        last_save = 0
        for f in as_completed(futures):
            f.result()
            d = _stats["done"]
            el = time.time() - start
            if d - last_log >= 50 or d >= len(need):
                rate = d / max(1, el)
                eta = (len(need) - d) / max(0.01, rate) / 60
                log.info("[%4.0fs] %5d/%d (%.1f%%) trades=%d %.1f/s ETA=%.0fm err=%d",
                         el, d, len(need), d/len(need)*100, _stats["trades"], rate, eta, _stats["errors"])
                last_log = d
            if d - last_save >= 500:
                save(data, out, "progress")
                last_save = d

    el = time.time() - start
    w = sum(1 for t in tokens if t.get("trades") and len(t["trades"]) > 0)
    log.info("=" * 50)
    log.info("DONE %.0fs (%.1fmin) | with_trades=%d/%d | trades=%d | err=%d",
             el, el/60, w, len(tokens), _stats["trades"], _stats["errors"])
    log.info("=" * 50)
    save(data, out, "final")


if __name__ == "__main__":
    main()
