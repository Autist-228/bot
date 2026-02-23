import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_RPC_URL = os.getenv("HELIUS_RPC_URL", "")

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data"
DEXSCREENER_API = "https://api.dexscreener.com"

COLLECT_DURATION = int(os.getenv("COLLECT_DURATION", "1800"))
COOLDOWN_DURATION = int(os.getenv("COOLDOWN_DURATION", "900"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MIN_LIQUIDITY_USD = float(os.getenv("MIN_LIQUIDITY_USD", "1500"))
BET_SIZE_USD = float(os.getenv("BET_SIZE_USD", "3.0"))
MAX_SLOTS = int(os.getenv("MAX_SLOTS", "10"))

PUMPFUN_FEE_PCT = 0.01
BUY_SLIPPAGE_PCT = 0.02
SELL_SLIPPAGE_PCT = 0.03

STOP_LOSS_PCT = -15.0
TRAILING_STOP_PCT = 15.0
TIME_STOP_SEC = 60
TIME_STOP_MIN_GAIN = 10.0

MIN_CONFIDENCE_PCT = 70.0
MIN_BUY_RATIO = 1.5
ROCKET_ONLY = True

PRICE_SNAPSHOT_INTERVALS = [15, 30, 60, 120, 300, 600, 900]
