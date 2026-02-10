import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import httpx
import base58
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

from config import HELIUS_RPC_URL, DATA_DIR

log = logging.getLogger("real_trader")

PUMPPORTAL_TRADE_URL = "https://pumpportal.fun/api/trade-local"
SOL_MINT = "So11111111111111111111111111111111111111112"
SLIPPAGE = 25
PRIORITY_FEE = 0.0001
BET_SIZE_USD = 5.0
TX_CONFIRM_TIMEOUT = 30
TX_CONFIRM_RETRIES = 2

trade_log: list[dict] = []


class RealTrader:
    def __init__(self):
        pk = os.getenv("SOLANA_PRIVATE_KEY", "")
        if not pk:
            raise ValueError("SOLANA_PRIVATE_KEY not set in .env")
        self.keypair = Keypair.from_base58_string(pk)
        self.pubkey = str(self.keypair.pubkey())
        self.rpc = Client(HELIUS_RPC_URL)
        self.sol_balance = 0.0
        self.initial_balance = 0.0
        self.positions: dict[str, dict] = {}
        log.info("Wallet: %s", self.pubkey)

    def refresh_balance(self) -> float:
        try:
            resp = self.rpc.get_balance(self.keypair.pubkey())
            lamports = resp.value
            self.sol_balance = lamports / 1e9
            return self.sol_balance
        except Exception as e:
            log.error("Failed to get balance: %s", e)
            return self.sol_balance

    def confirm_tx(self, tx_sig_str: str, timeout: int = TX_CONFIRM_TIMEOUT) -> bool:
        try:
            sig = Signature.from_string(tx_sig_str)
            start = time.time()
            while time.time() - start < timeout:
                resp = self.rpc.get_signature_statuses([sig])
                statuses = resp.value
                if statuses and statuses[0] is not None:
                    status = statuses[0]
                    if status.err is not None:
                        log.warning("TX %s FAILED on-chain: %s", tx_sig_str[:20], status.err)
                        return False
                    if status.confirmation_status is not None:
                        cs = str(status.confirmation_status)
                        if "confirmed" in cs.lower() or "finalized" in cs.lower():
                            log.info("TX %s CONFIRMED (%s)", tx_sig_str[:20], cs)
                            return True
                time.sleep(2)
            log.warning("TX %s TIMEOUT after %ds", tx_sig_str[:20], timeout)
            return False
        except Exception as e:
            log.warning("TX confirm error %s: %s", tx_sig_str[:20], e)
            return False

    async def _send_and_confirm(self, payload: dict, symbol: str, action: str) -> dict:
        result = {"success": False, "tx_hash": None, "error": None, "confirmed": False}
        sent_tx_hashes: list[str] = []
        for attempt in range(1, TX_CONFIRM_RETRIES + 1):
            try:
                if attempt > 1 and sent_tx_hashes:
                    prev_tx = sent_tx_hashes[-1]
                    log.info("%s %s: checking if prev tx %s landed before retry...", action, symbol, prev_tx[:20])
                    prev_confirmed = await asyncio.to_thread(self.confirm_tx, prev_tx, 5)
                    if prev_confirmed:
                        log.info("%s %s: prev tx %s WAS confirmed! No retry needed.", action, symbol, prev_tx[:20])
                        result["success"] = True
                        result["confirmed"] = True
                        result["tx_hash"] = prev_tx
                        self.refresh_balance()
                        return result
                async with httpx.AsyncClient() as client:
                    resp = await client.post(PUMPPORTAL_TRADE_URL, json=payload, timeout=15)
                if resp.status_code != 200:
                    result["error"] = f"PumpPortal {resp.status_code}: {resp.text[:200]}"
                    log.error("%s FAIL %s attempt %d: %s", action, symbol, attempt, result["error"])
                    if attempt < TX_CONFIRM_RETRIES:
                        await asyncio.sleep(1)
                    continue
                tx_bytes = resp.content
                tx = VersionedTransaction.from_bytes(tx_bytes)
                signed_tx = VersionedTransaction(tx.message, [self.keypair])
                tx_resp = self.rpc.send_raw_transaction(
                    bytes(signed_tx),
                    opts=TxOpts(skip_preflight=True, max_retries=3),
                )
                tx_hash = str(tx_resp.value)
                result["tx_hash"] = tx_hash
                sent_tx_hashes.append(tx_hash)
                log.info("%s %s sent tx=%s (attempt %d, confirming...)", action, symbol, tx_hash[:20], attempt)
                confirmed = await asyncio.to_thread(self.confirm_tx, tx_hash)
                if confirmed:
                    result["success"] = True
                    result["confirmed"] = True
                    self.refresh_balance()
                    return result
                else:
                    log.warning("%s %s NOT confirmed (attempt %d/%d)", action, symbol, attempt, TX_CONFIRM_RETRIES)
                    if attempt < TX_CONFIRM_RETRIES:
                        await asyncio.sleep(1)
            except Exception as e:
                result["error"] = str(e)
                log.error("%s ERROR %s attempt %d: %s", action, symbol, attempt, e)
                if attempt < TX_CONFIRM_RETRIES:
                    await asyncio.sleep(1)
        return result

    async def buy_token(self, mint: str, sol_amount: float, symbol: str = "") -> dict:
        result = {
            "action": "buy",
            "mint": mint,
            "symbol": symbol,
            "sol_amount": sol_amount,
            "success": False,
            "confirmed": False,
            "tx_hash": None,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.refresh_balance()
        if sol_amount > self.sol_balance * 0.9:
            result["error"] = f"Not enough SOL: need {sol_amount:.4f}, have {self.sol_balance:.4f}"
            log.error("BUY FAILED %s: %s", symbol, result["error"])
            trade_log.append(result)
            return result

        payload = {
            "publicKey": self.pubkey,
            "action": "buy",
            "mint": mint,
            "amount": sol_amount,
            "denominatedInSol": "true",
            "slippage": SLIPPAGE,
            "priorityFee": PRIORITY_FEE,
            "pool": "auto",
        }
        log.info("BUY %s: %.4f SOL (slippage=%d%%)...", symbol, sol_amount, SLIPPAGE)

        send_result = await self._send_and_confirm(payload, symbol, "BUY")
        result.update(send_result)

        if result["success"]:
            self.positions[mint] = {
                "mint": mint,
                "symbol": symbol,
                "buy_sol": sol_amount,
                "buy_time": time.time(),
                "buy_tx": result["tx_hash"],
                "sold_pct": 0,
                "sell_txs": [],
                "sells_count": 0,
            }
            log.info("BUY CONFIRMED %s: tx=%s | balance=%.4f SOL", symbol, result["tx_hash"][:20], self.sol_balance)
        else:
            log.error("BUY FAILED %s after %d retries", symbol, TX_CONFIRM_RETRIES)

        trade_log.append(result)
        return result

    async def sell_token(self, mint: str, sell_pct: int, symbol: str = "", reason: str = "") -> dict:
        result = {
            "action": "sell",
            "mint": mint,
            "symbol": symbol,
            "sell_pct": sell_pct,
            "reason": reason,
            "success": False,
            "confirmed": False,
            "tx_hash": None,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        balance_before = self.refresh_balance()

        payload = {
            "publicKey": self.pubkey,
            "action": "sell",
            "mint": mint,
            "amount": f"{sell_pct}%",
            "denominatedInSol": "false",
            "slippage": SLIPPAGE,
            "priorityFee": PRIORITY_FEE,
            "pool": "auto",
        }
        log.info("SELL %s: %d%% (%s) slippage=%d%%...", symbol, sell_pct, reason, SLIPPAGE)

        send_result = await self._send_and_confirm(payload, symbol, "SELL")
        result.update(send_result)

        if result["success"]:
            await asyncio.sleep(2)
            balance_after = self.refresh_balance()
            sol_received = balance_after - balance_before
            result["sol_received"] = sol_received

            if mint in self.positions:
                pos = self.positions[mint]
                remaining = 100 - pos["sold_pct"]
                actual_sold = min(sell_pct, remaining)
                pos["sold_pct"] += actual_sold
                pos["sells_count"] += 1
                pos["sell_txs"].append({
                    "pct": actual_sold,
                    "tx": result["tx_hash"],
                    "time": time.time(),
                    "sol_received": sol_received,
                    "reason": reason,
                })

            log.info(
                "SELL CONFIRMED %s %d%% (%s): tx=%s | got %.6f SOL | balance=%.4f SOL",
                symbol, sell_pct, reason, result["tx_hash"][:20], sol_received, self.sol_balance,
            )
        else:
            log.error("SELL FAILED %s after %d retries", symbol, TX_CONFIRM_RETRIES)

        trade_log.append(result)
        return result

    async def sell_all_remaining(self, mint: str, symbol: str = "", reason: str = "") -> dict:
        pos = self.positions.get(mint)
        if not pos:
            return await self.sell_token(mint, 100, symbol, reason)

        remaining = 100 - pos["sold_pct"]
        if remaining <= 0:
            log.info("SELL SKIP %s: already fully sold", symbol)
            return {"action": "sell", "success": True, "already_sold": True}

        return await self.sell_token(mint, 100, symbol, reason)

    def get_trade_summary(self) -> dict:
        buys = [t for t in trade_log if t["action"] == "buy"]
        sells = [t for t in trade_log if t["action"] == "sell"]
        confirmed_buys = [t for t in buys if t.get("confirmed")]
        confirmed_sells = [t for t in sells if t.get("confirmed")]
        return {
            "total_trades": len(trade_log),
            "buys": len(buys),
            "confirmed_buys": len(confirmed_buys),
            "failed_buys": len(buys) - len(confirmed_buys),
            "sells": len(sells),
            "confirmed_sells": len(confirmed_sells),
            "failed_sells": len(sells) - len(confirmed_sells),
            "total_sol_spent": sum(t.get("sol_amount", 0) for t in confirmed_buys),
            "total_sol_received": sum(t.get("sol_received", 0) for t in confirmed_sells),
            "positions": len(self.positions),
        }
