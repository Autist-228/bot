import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data"

BET_SIZE_USD = float(os.getenv("BET_SIZE_USD", "1.0"))
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "50.0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "50"))

DEV_MIN_BUY_SOL = float(os.getenv("DEV_MIN_BUY_SOL", "0.05"))
DEV_MAX_BUY_SOL = float(os.getenv("DEV_MAX_BUY_SOL", "10.0"))

PHASE2_WAIT_SEC = int(os.getenv("PHASE2_WAIT_SEC", "5"))
PHASE2_MIN_BUYERS = int(os.getenv("PHASE2_MIN_BUYERS", "3"))

TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "15.0"))
TIME_STOP_SEC = int(os.getenv("TIME_STOP_SEC", "300"))
EMERGENCY_STOP_PCT = float(os.getenv("EMERGENCY_STOP_PCT", "-50.0"))

PUMPFUN_FEE_PCT = 0.01
BUY_SLIPPAGE_PCT = 0.01
SELL_SLIPPAGE_PCT = 0.02

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BLACKLIST_PATH = os.path.join(DATA_DIR, "dev_blacklist.json")
STATE_PATH = os.path.join(DATA_DIR, "sniper_state.json")
TRADES_LOG_PATH = os.path.join(DATA_DIR, "trades_log.jsonl")
