import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CODEX_API_KEY = os.getenv("CODEX_API_KEY", "")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
HELIUS_RPC_URL = os.getenv("HELIUS_RPC_URL", "")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

CODEX_GRAPHQL_URL = "https://graph.codex.io/graphql"
TWITTER_API_BASE = "https://api.twitterapi.io"

SOLANA_NETWORK_ID = 1399811149

SIGNAL_THRESHOLD = 3

SCAN_INTERVAL_SECONDS = 60

MIN_LIQUIDITY_USD = 1000
MIN_VOLUME_24H_USD = 3000
MAX_TOP_HOLDER_PERCENT = 25.0
MIN_BUY_COUNT_5M = 3
MAX_TOKEN_AGE_HOURS = 6
MAX_HOLDERS_EARLY = 500
MIN_HOLDERS = 20
