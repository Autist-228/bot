import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data"

BET_SIZE_USD = float(os.getenv("BET_SIZE_USD", "1.0"))
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "50.0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))

DEV_MIN_BUY_SOL = float(os.getenv("DEV_MIN_BUY_SOL", "0.05"))
DEV_MAX_BUY_SOL = float(os.getenv("DEV_MAX_BUY_SOL", "10.0"))

PHASE2_WAIT_SEC = int(os.getenv("PHASE2_WAIT_SEC", "3"))
PHASE2_MIN_BUYERS = int(os.getenv("PHASE2_MIN_BUYERS", "3"))

EMERGENCY_STOP_PCT = float(os.getenv("EMERGENCY_STOP_PCT", "-35.0"))

EPOCH_T0_END = int(os.getenv("EPOCH_T0_END", "10"))
EPOCH_T1_END = int(os.getenv("EPOCH_T1_END", "30"))
EPOCH_T2_END = int(os.getenv("EPOCH_T2_END", "120"))
EPOCH_T3_END = int(os.getenv("EPOCH_T3_END", "600"))

EPOCH_T0_TRAIL = float(os.getenv("EPOCH_T0_TRAIL", "30.0"))
EPOCH_T1_TRAIL = float(os.getenv("EPOCH_T1_TRAIL", "20.0"))
EPOCH_T2_TRAIL = float(os.getenv("EPOCH_T2_TRAIL", "15.0"))
EPOCH_T3_TRAIL = float(os.getenv("EPOCH_T3_TRAIL", "20.0"))

EPOCH_T0_RUG_PEAK = float(os.getenv("EPOCH_T0_RUG_PEAK", "3.0"))
EPOCH_T1_STALL_PEAK = float(os.getenv("EPOCH_T1_STALL_PEAK", "10.0"))

DEAD_TOKEN_SEC = int(os.getenv("DEAD_TOKEN_SEC", "15"))

PUMPFUN_FEE_PCT = 0.01
BUY_SLIPPAGE_PCT = 0.01
SELL_SLIPPAGE_PCT = 0.02

EXTRA_BUY_SLIPPAGE_PCT = float(os.getenv("EXTRA_BUY_SLIPPAGE_PCT", "0.02"))
EXTRA_SELL_SLIPPAGE_PCT = float(os.getenv("EXTRA_SELL_SLIPPAGE_PCT", "0.02"))

SOL_BASE_FEE = 0.000005
SOL_PRIORITY_FEE = float(os.getenv("SOL_PRIORITY_FEE", "0.0002"))
SOL_ATA_RENT = 0.00203928
SOL_TX_FEE_PER_TRADE = (SOL_BASE_FEE + SOL_PRIORITY_FEE) * 2 + SOL_ATA_RENT - SOL_ATA_RENT

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BLACKLIST_PATH = os.path.join(DATA_DIR, "dev_blacklist.json")
STATE_PATH = os.path.join(DATA_DIR, "sniper_state.json")
TRADES_LOG_PATH = os.path.join(DATA_DIR, "trades_log.jsonl")
