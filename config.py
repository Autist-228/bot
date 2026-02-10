import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_RPC_URL = os.getenv("HELIUS_RPC_URL", "")

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data"
DEXSCREENER_API = "https://api.dexscreener.com"

COLLECT_DURATION = int(os.getenv("COLLECT_DURATION", "1800"))
COLLECT_DURATION_6H = 21600
PRICE_CHECK_INTERVALS = [60, 300, 600, 1200, 1800]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
