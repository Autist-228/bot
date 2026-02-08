# Solana Smart Money Tracker Bot

Telegram bot that monitors Solana tokens using on-chain data (Codex API), Twitter social signals (TwitterAPI.io), and safety filters to detect potential profitable entries.

## Features

- **On-chain monitoring**: Trending tokens, volume spikes, whale activity via Codex GraphQL API
- **Twitter social signals**: Tweet mentions, KOL/influencer detection, engagement metrics via TwitterAPI.io
- **Safety filters**: Top holder concentration, honeypot detection, mint/freeze authority checks
- **Multi-signal scoring**: Combines on-chain + social + safety into a single score (threshold: 3/8)
- **Telegram alerts**: Real-time alerts with full token info, links to DexScreener/Birdeye
- **Backtesting/tracking**: Records all signals, tracks price changes, shows win rate statistics

## Setup

1. Clone this repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```
4. Run the bot:
   ```
   python main.py
   ```

## Required API Keys

| Service | Purpose | Cost |
|---------|---------|------|
| Telegram Bot Token | Alerts | Free (via @BotFather) |
| Codex API | On-chain data | Free tier available |
| TwitterAPI.io | Social data | $10 for 1M credits |
| Helius | Solana RPC | Free tier available |

## Telegram Commands

- `/scan` - Run manual scan
- `/stats` - Show signal statistics (win rate, total signals, etc.)
- `/tracked` - Show currently tracked tokens with PnL
- `/help` - Help message

## Architecture

```
main.py              - Entry point, scheduler, command handler
config.py            - Configuration and environment variables
modules/
  codex_tracker.py   - Codex GraphQL API (trending tokens, prices, holders)
  twitter_monitor.py - TwitterAPI.io (tweet search, social scoring)
  safety_check.py    - Safety filters (holders, honeypot, mint authority)
  signal_engine.py   - Score aggregation, signal generation
  telegram_bot.py    - Telegram message formatting and sending
  price_tracker.py   - Price tracking for active signals
utils/
  database.py        - SQLite database for signal history and tracking
```
