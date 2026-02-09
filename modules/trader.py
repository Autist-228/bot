import asyncio
import base64
import logging
import time

import base58
import httpx
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HELIUS_RPC_URL

logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"


class JupiterTrader:
    def __init__(self, private_key: str = "", dry_run: bool = True):
        self.dry_run = dry_run
        self.keypair = None
        self.public_key = None
        if private_key and not dry_run:
            try:
                key_bytes = base58.b58decode(private_key)
                self.keypair = Keypair.from_bytes(key_bytes)
                self.public_key = str(self.keypair.pubkey())
                logger.info("Wallet loaded: %s", self.public_key)
            except Exception as e:
                logger.error("Failed to load wallet: %s", e)
                self.dry_run = True

    async def get_sol_balance(self) -> float:
        if self.dry_run:
            return 0.0
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    HELIUS_RPC_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getBalance",
                        "params": [self.public_key],
                    },
                )
                data = resp.json()
                lamports = data.get("result", {}).get("value", 0)
                return lamports / 1e9
        except Exception as e:
            logger.error("Balance check error: %s", e)
            return 0.0

    async def get_quote(
        self, input_mint: str, output_mint: str, amount_lamports: int, slippage_bps: int = 300
    ) -> dict | None:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_lamports),
            "slippageBps": str(slippage_bps),
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(JUPITER_QUOTE_URL, params=params)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning("Jupiter quote error %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("Jupiter quote error: %s", e)
        return None

    async def execute_swap(self, quote_response: dict) -> str | None:
        if self.dry_run:
            out_amount = int(quote_response.get("outAmount", 0))
            in_amount = int(quote_response.get("inAmount", 0))
            logger.info(
                "DRY RUN swap: in=%d out=%d",
                in_amount,
                out_amount,
            )
            return f"dry_run_{int(time.time())}"

        if not self.keypair:
            logger.error("No wallet loaded")
            return None

        try:
            swap_body = {
                "quoteResponse": quote_response,
                "userPublicKey": self.public_key,
                "wrapAndUnwrapSol": True,
                "prioritizationFeeLamports": 50000,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    JUPITER_SWAP_URL,
                    json=swap_body,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code != 200:
                    logger.error("Jupiter swap error %d: %s", resp.status_code, resp.text[:300])
                    return None
                swap_data = resp.json()

            swap_tx_b64 = swap_data["swapTransaction"]
            raw_tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_b64))
            signature = self.keypair.sign_message(bytes(raw_tx.message))
            signed_tx = VersionedTransaction.populate(raw_tx.message, [signature])
            encoded_tx = base64.b64encode(bytes(signed_tx)).decode("utf-8")

            async with httpx.AsyncClient(timeout=30) as client:
                rpc_resp = await client.post(
                    HELIUS_RPC_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "sendTransaction",
                        "params": [
                            encoded_tx,
                            {
                                "skipPreflight": True,
                                "preflightCommitment": "confirmed",
                                "encoding": "base64",
                                "maxRetries": 3,
                            },
                        ],
                    },
                )
                result = rpc_resp.json()
                if "result" in result:
                    tx_sig = result["result"]
                    logger.info("TX sent: %s", tx_sig)
                    return tx_sig
                logger.error("TX send error: %s", result.get("error"))
                return None
        except Exception as e:
            logger.error("Swap execution error: %s", e)
            return None

    async def buy_token(
        self, token_mint: str, sol_amount: float, slippage_bps: int = 300,
        token_price_usd: float = 0.0, sol_price_usd: float = 0.0,
    ) -> dict | None:
        if self.dry_run:
            if token_price_usd > 0 and sol_price_usd > 0:
                usd_value = sol_amount * sol_price_usd
                tokens = int(usd_value / token_price_usd)
            else:
                tokens = int(sol_amount * 1e9)
            if tokens <= 0:
                logger.warning(
                    "DRY RUN SKIP: %s, %.4f SOL can't afford 1 token at $%.4f",
                    token_mint[:12], sol_amount, token_price_usd,
                )
                return None
            tx_sig = f"dry_run_buy_{int(time.time())}"
            logger.info(
                "DRY RUN BUY: %s, %.4f SOL -> %d tokens",
                token_mint[:12], sol_amount, tokens,
            )
            return {
                "tx_signature": tx_sig,
                "token_mint": token_mint,
                "sol_spent": sol_amount,
                "tokens_received": tokens,
                "price_impact": 0.0,
                "timestamp": time.time(),
            }

        amount_lamports = int(sol_amount * 1e9)
        quote = await self.get_quote(SOL_MINT, token_mint, amount_lamports, slippage_bps)
        if not quote:
            logger.warning("No quote for buying %s", token_mint)
            return None

        out_amount = int(quote.get("outAmount", 0))
        price_impact = float(quote.get("priceImpactPct", 0))

        if price_impact > 5.0:
            logger.warning("Price impact too high: %.2f%% for %s", price_impact, token_mint)
            return None

        tx_sig = await self.execute_swap(quote)
        if not tx_sig:
            return None

        return {
            "tx_signature": tx_sig,
            "token_mint": token_mint,
            "sol_spent": sol_amount,
            "tokens_received": out_amount,
            "price_impact": price_impact,
            "timestamp": time.time(),
        }

    async def sell_token(
        self, token_mint: str, token_amount: int, slippage_bps: int = 500,
        token_price_usd: float = 0.0, sol_price_usd: float = 0.0,
    ) -> dict | None:
        if self.dry_run:
            if token_price_usd > 0 and sol_price_usd > 0:
                usd_value = token_amount * token_price_usd
                sol_back = usd_value / sol_price_usd
            else:
                sol_back = token_amount / 1e9
            tx_sig = f"dry_run_sell_{int(time.time())}"
            logger.info(
                "DRY RUN SELL: %s, %d tokens -> %.6f SOL",
                token_mint[:12], token_amount, sol_back,
            )
            return {
                "tx_signature": tx_sig,
                "token_mint": token_mint,
                "tokens_sold": token_amount,
                "sol_received": sol_back,
                "timestamp": time.time(),
            }

        quote = await self.get_quote(token_mint, SOL_MINT, token_amount, slippage_bps)
        if not quote:
            logger.warning("No quote for selling %s", token_mint)
            return None

        out_amount = int(quote.get("outAmount", 0))
        tx_sig = await self.execute_swap(quote)
        if not tx_sig:
            return None

        return {
            "tx_signature": tx_sig,
            "token_mint": token_mint,
            "tokens_sold": token_amount,
            "sol_received": out_amount / 1e9,
            "timestamp": time.time(),
        }
