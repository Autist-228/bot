import dotenv from "dotenv";
import { ExitStrategy, TradeConfidence } from "./types";

dotenv.config();

function requireEnv(key: string): string {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing required env variable: ${key}`);
  }
  return value;
}

function optionalEnv(key: string, defaultValue: string): string {
  return process.env[key] || defaultValue;
}

export const config = {
  helius: {
    rpcUrl: requireEnv("HELIUS_RPC_URL"),
    wsUrl: optionalEnv(
      "HELIUS_WS_URL",
      requireEnv("HELIUS_RPC_URL").replace("https://", "wss://")
    ),
    apiUrl: optionalEnv(
      "HELIUS_API_URL",
      "https://api-mainnet.helius-rpc.com/v0"
    ),
    apiKey: requireEnv("HELIUS_API_KEY"),
  },

  codex: {
    apiKey: requireEnv("CODEX_API_KEY"),
    apiUrl: "https://graph.codex.io/graphql",
  },

  twitter: {
    apiKey: requireEnv("TWITTER_API_KEY"),
    apiUrl: "https://api.twitterapi.io/twitter",
  },

  telegram: {
    botToken: optionalEnv("TELEGRAM_BOT_TOKEN", ""),
    chatId: optionalEnv("TELEGRAM_CHAT_ID", ""),
  },

  wallet: {
    privateKey: optionalEnv("WALLET_PRIVATE_KEY", ""),
  },

  trading: {
    bankSizeSol: parseFloat(optionalEnv("BANK_SIZE_SOL", "0.1")),
    maxPositionPct: parseFloat(optionalEnv("MAX_POSITION_PCT", "15")),
    maxOpenPositions: parseInt(optionalEnv("MAX_OPEN_POSITIONS", "10")),
    dailyLossLimitPct: parseFloat(optionalEnv("DAILY_LOSS_LIMIT_PCT", "20")),
    minRocketScore: parseInt(optionalEnv("MIN_ROCKET_SCORE", "40")),
    slippageBps: parseInt(optionalEnv("SLIPPAGE_BPS", "300")),
    priorityFeeLamports: parseInt(
      optionalEnv("PRIORITY_FEE_LAMPORTS", "100000")
    ),
  },

  crawler: {
    scanIntervalMs: 60_000,
    minTradesForScoring: 20,
    minWinRate: 0.4,
    topWalletsCount: 50,
    walletRefreshIntervalMs: 24 * 60 * 60 * 1000,
    blacklistCheckIntervalMs: 6 * 60 * 60 * 1000,
  },

  scoring: {
    weights: {
      winRate: 0.25,
      avgRoi: 0.2,
      activity: 0.1,
      maxDrawdown: 0.15,
      entrySpeed: 0.15,
      consistency: 0.15,
    },
    tierThresholds: {
      S: 80,
      A: 60,
      B: 40,
      C: 20,
    },
  },
};

export function getExitStrategy(confidence: TradeConfidence): ExitStrategy {
  switch (confidence) {
    case TradeConfidence.ROCKET:
      return {
        profitLadder: [
          {
            triggerPct: 50,
            sellPct: 20,
            description: "Recover initial stake",
          },
          { triggerPct: 150, sellPct: 20, description: "Lock profit" },
          { triggerPct: 300, sellPct: 20, description: "Major profit" },
        ],
        stopLoss: [
          { triggerPct: -15, sellPct: 50, description: "Partial stop loss" },
          { triggerPct: -25, sellPct: 100, description: "Full stop loss" },
        ],
        trailingStops: [
          { activateAtPct: 500, stopPct: 15 },
          { activateAtPct: 1000, stopPct: 10 },
        ],
        maxHoldTime: 24 * 60 * 60 * 1000,
        flatlineTimeout: 6 * 60 * 60 * 1000,
        minLiquidity: 5000,
      };

    case TradeConfidence.STRONG:
      return {
        profitLadder: [
          { triggerPct: 30, sellPct: 30, description: "Early profit" },
          { triggerPct: 80, sellPct: 30, description: "Mid profit" },
        ],
        stopLoss: [
          { triggerPct: -15, sellPct: 50, description: "Partial stop loss" },
          { triggerPct: -25, sellPct: 100, description: "Full stop loss" },
        ],
        trailingStops: [{ activateAtPct: 150, stopPct: 20 }],
        maxHoldTime: 12 * 60 * 60 * 1000,
        flatlineTimeout: 4 * 60 * 60 * 1000,
        minLiquidity: 5000,
      };

    case TradeConfidence.NORMAL:
      return {
        profitLadder: [
          { triggerPct: 30, sellPct: 50, description: "Quick profit" },
          { triggerPct: 60, sellPct: 50, description: "Full exit" },
        ],
        stopLoss: [
          { triggerPct: -10, sellPct: 50, description: "Partial stop loss" },
          { triggerPct: -20, sellPct: 100, description: "Full stop loss" },
        ],
        trailingStops: [],
        maxHoldTime: 6 * 60 * 60 * 1000,
        flatlineTimeout: 2 * 60 * 60 * 1000,
        minLiquidity: 10000,
      };

    default:
      return {
        profitLadder: [
          { triggerPct: 20, sellPct: 100, description: "Quick exit" },
        ],
        stopLoss: [
          { triggerPct: -10, sellPct: 100, description: "Full stop loss" },
        ],
        trailingStops: [],
        maxHoldTime: 2 * 60 * 60 * 1000,
        flatlineTimeout: 1 * 60 * 60 * 1000,
        minLiquidity: 10000,
      };
  }
}

export function getPositionSizePct(
  rocketScore: number,
  confidence: TradeConfidence
): number {
  const maxPct = config.trading.maxPositionPct;

  switch (confidence) {
    case TradeConfidence.ROCKET:
      return Math.min(maxPct, 10 + (rocketScore - 80) * 0.25);
    case TradeConfidence.STRONG:
      return Math.min(maxPct * 0.67, 5 + (rocketScore - 60) * 0.25);
    case TradeConfidence.NORMAL:
      return Math.min(maxPct * 0.33, 2 + (rocketScore - 40) * 0.15);
    default:
      return 1;
  }
}
