import logging
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.codex_tracker import get_token_price
from modules.telegram_bot import send_message
from utils.database import (
    get_active_tracked_tokens,
    update_tracked_price,
    mark_signal_inactive,
)

logger = logging.getLogger(__name__)

STALE_HOURS = 24
DUMP_THRESHOLD = -80.0


async def update_all_prices():
    tokens = await get_active_tracked_tokens()
    if not tokens:
        return

    logger.info("Updating prices for %d tracked tokens", len(tokens))

    for t in tokens:
        address = t["token_address"]
        symbol = t.get("token_symbol", "?")
        entry_price = t.get("price_at_signal", 0)

        price = await get_token_price(address)
        if price is None:
            continue

        await update_tracked_price(address, price)

        if entry_price and entry_price > 0:
            pnl_pct = ((price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0

        age_hours = (time.time() - t.get("created_at", time.time())) / 3600

        if pnl_pct <= DUMP_THRESHOLD:
            await mark_signal_inactive(address)
            await send_message(
                f"<b>${symbol}</b> dropped {pnl_pct:.1f}% — removed from tracking."
            )
            logger.info("%s dropped %.1f%%, marked inactive", symbol, pnl_pct)
            continue

        if age_hours > STALE_HOURS and pnl_pct < 0:
            await mark_signal_inactive(address)
            logger.info("%s stale (%dh) and negative, marked inactive",
                        symbol, int(age_hours))
