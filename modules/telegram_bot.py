import logging
import json
import re
import time
import httpx

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _escape_html(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


async def send_message(text: str, chat_id: str | None = None, parse_mode: str = "HTML"):
    target = chat_id or TELEGRAM_CHAT_ID
    url = f"{API_URL}/sendMessage"
    payload = {
        "chat_id": target,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 400:
                payload["text"] = _strip_html(text)
                del payload["parse_mode"]
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("Telegram send error: %s", e)
        return None


def format_signal_alert(signal: dict) -> str:
    token = signal["token"]
    details = signal["details"]
    social = details.get("social", {})
    onchain = details.get("onchain", {})
    safety = signal.get("safety", {})
    honeypot = signal.get("honeypot", {})
    total_score = signal["total_score"]

    stars = _score_stars(total_score)
    safety_icons = _safety_icons(details, honeypot)

    price = token.get("price_usd", 0)
    if price < 0.0001:
        price_str = f"${price:.8f}"
    elif price < 1:
        price_str = f"${price:.6f}"
    else:
        price_str = f"${price:.4f}"

    mcap = token.get("market_cap", 0)
    mcap_str = _format_number(mcap)

    liq = token.get("liquidity", 0)
    liq_str = _format_number(liq)

    vol = token.get("volume_24h", 0)
    vol_str = _format_number(vol)

    change_5m = onchain.get("change_5m", 0)
    change_1h = onchain.get("change_1h", 0)

    top_holders = details.get("top_holders_pct")
    holders_str = f"{top_holders:.1f}%" if top_holders else "N/A"

    tweet_count = social.get("tweet_count", 0)
    influencers = social.get("influencer_mentions", 0)
    views = social.get("total_views", 0)

    address = token.get("address", "")
    dexscreener = f"https://dexscreener.com/solana/{address}"
    birdeye = f"https://birdeye.so/token/{address}?chain=solana"

    lines = [
        f"{stars} <b>SIGNAL: ${token.get('symbol', '?')}</b> {stars}",
        f"<b>{token.get('name', '')}</b>",
        "",
        f"Score: <b>{total_score}/8</b> | {safety_icons}",
        "",
        f"Price: <b>{price_str}</b>",
        f"MCap: <b>{mcap_str}</b>",
        f"Liquidity: <b>{liq_str}</b>",
        f"Volume 24h: <b>{vol_str}</b>",
        "",
        f"5m: <b>{change_5m:+.1f}%</b> | 1h: <b>{change_1h:+.1f}%</b>",
        f"Buys 5m: <b>{onchain.get('buy_count_5m', 0)}</b> | Unique: <b>{onchain.get('unique_buys_5m', 0)}</b>",
        f"Holders: <b>{onchain.get('holders', 0)}</b>",
        f"Top 10 holders: <b>{holders_str}</b>",
        "",
        f"Tweets: <b>{tweet_count}</b> | KOLs: <b>{influencers}</b> | Views: <b>{_format_number(views)}</b>",
    ]

    top_tweet = social.get("top_tweet")
    if top_tweet:
        lines.append(
            f"\nTop tweet by @{top_tweet.get('author', '?')} "
            f"({_format_number(top_tweet.get('followers', 0))} followers):"
        )
        lines.append(f"<i>{top_tweet.get('text', '')[:150]}</i>")

    if safety.get("warnings"):
        lines.append(f"\nWarnings: {', '.join(safety['warnings'])}")

    lines.append(f"\n<a href='{dexscreener}'>DexScreener</a> | <a href='{birdeye}'>Birdeye</a>")
    lines.append(f"\n<code>{address}</code>")

    return "\n".join(lines)


def format_trade_open(pos_dict: dict) -> str:
    lines = [
        f"<b>BUY {pos_dict['symbol']}</b>",
        f"Price: <b>${pos_dict['entry_price']:.10f}</b>",
        f"Size: <b>{pos_dict['sol_spent']:.4f} SOL</b>",
        f"Score: <b>{pos_dict['score']}</b>",
        f"Tokens: <b>{pos_dict['tokens_held']:,}</b>",
        f"\n<code>{pos_dict['token_mint']}</code>",
    ]
    return "\n".join(lines)


def format_trade_close(pos_dict: dict) -> str:
    pnl = pos_dict["pnl_pct"]
    tag = "PROFIT" if pnl > 0 else "LOSS"
    lines = [
        f"<b>SELL {pos_dict['symbol']} — {tag}</b>",
        f"PnL: <b>{pnl:+.1f}%</b>",
        f"Peak: <b>+{pos_dict['peak_pnl_pct']:.1f}%</b>",
        f"Reason: <b>{pos_dict['exit_reason']}</b>",
        f"SOL back: <b>{pos_dict['sol_received']:.4f}</b>",
        f"Hold time: <b>{pos_dict['age_sec']}s</b>",
    ]
    return "\n".join(lines)


def format_portfolio(stats: dict) -> str:
    lines = [
        f"<b>Portfolio</b>",
        f"Bank: <b>{stats['bank_sol']:.4f} SOL</b>",
        f"PnL: <b>{stats['total_pnl_sol']:+.4f} SOL</b>",
        f"Open: <b>{stats['open_positions']}</b> | Closed: <b>{stats['closed_trades']}</b>",
        f"Wins: <b>{stats['wins']}</b> | Losses: <b>{stats['losses']}</b>",
        f"Win rate: <b>{stats['win_rate']}%</b>",
    ]
    for p in stats.get("positions", []):
        lines.append(
            f"\n{p['symbol']}: <b>{p['pnl_pct']:+.1f}%</b> "
            f"(peak +{p['peak_pnl_pct']:.1f}%) "
            f"{'LOCKED' if p['profit_locked'] else ''}"
        )
    return "\n".join(lines)


def format_stats_message(stats: dict) -> str:
    return (
        f"<b>Bot Statistics</b>\n\n"
        f"Total signals: <b>{stats['total']}</b>\n"
        f"Profitable (+10%): <b>{stats['profitable_10pct']}</b>\n"
        f"Doubled (2x): <b>{stats['doubled']}</b>\n"
        f"Rugged (-50%): <b>{stats['rugged']}</b>\n"
        f"Win rate: <b>{stats['win_rate']}%</b>"
    )


def format_tracked_tokens(tokens: list[dict]) -> str:
    if not tokens:
        return "No active tracked tokens."

    lines = ["<b>Active Tracked Tokens</b>\n"]
    for t in tokens[:15]:
        price_entry = t.get("price_at_signal", 0)
        price_now = t.get("price_current", 0)
        price_max = t.get("price_max", 0)

        if price_entry and price_entry > 0:
            pnl = ((price_now - price_entry) / price_entry) * 100
            max_pnl = ((price_max - price_entry) / price_entry) * 100
        else:
            pnl = 0
            max_pnl = 0

        emoji = "+" if pnl >= 0 else ""
        lines.append(
            f"${t.get('token_symbol', '?')} | "
            f"Now: <b>{emoji}{pnl:.1f}%</b> | "
            f"Max: <b>+{max_pnl:.1f}%</b> | "
            f"Score: {t.get('score', 0)}"
        )

    return "\n".join(lines)


def _score_stars(score: int) -> str:
    if score >= 7:
        return "!!!"
    elif score >= 5:
        return "!!"
    else:
        return "!"


def _safety_icons(details: dict, honeypot: dict) -> str:
    parts = []
    if honeypot.get("mint_disabled"):
        parts.append("Mint:OFF")
    else:
        parts.append("Mint:ON(!)")
    if honeypot.get("freeze_disabled"):
        parts.append("Freeze:OFF")
    else:
        parts.append("Freeze:ON(!)")
    return " | ".join(parts)


def _format_number(val) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (ValueError, TypeError):
        return str(val)
    if val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:.1f}K"
    else:
        return f"${val:.2f}"
