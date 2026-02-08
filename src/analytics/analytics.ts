import { TradeLog, BankState, TradeConfidence } from "../types";
import { createLogger } from "../utils/logger";
import * as fs from "fs";
import * as path from "path";

const log = createLogger("Analytics");

interface DailyStats {
  date: string;
  trades: number;
  wins: number;
  losses: number;
  pnlSol: number;
  bestTrade: number;
  worstTrade: number;
}

interface TokenStats {
  mint: string;
  symbol: string;
  trades: number;
  totalPnlSol: number;
  avgPnlPct: number;
  bestPnlPct: number;
}

export class Analytics {
  private tradeLogs: TradeLog[] = [];
  private dailyStats: Map<string, DailyStats> = new Map();
  private logFilePath: string;

  constructor(logDir: string = "./data") {
    this.logFilePath = path.join(logDir, "trades.json");
    this.loadLogs();
  }

  recordTrade(tradeLog: TradeLog): void {
    this.tradeLogs.push(tradeLog);
    this.updateDailyStats(tradeLog);
    this.saveLogs();

    log.info(
      `Trade recorded: ${tradeLog.tokenSymbol} ${tradeLog.action} | PnL: ${tradeLog.pnlPct.toFixed(1)}% (${tradeLog.pnlSol.toFixed(4)} SOL)`
    );
  }

  private updateDailyStats(tradeLog: TradeLog): void {
    const date = new Date(tradeLog.timestamp).toISOString().slice(0, 10);
    const existing = this.dailyStats.get(date) || {
      date,
      trades: 0,
      wins: 0,
      losses: 0,
      pnlSol: 0,
      bestTrade: -Infinity,
      worstTrade: Infinity,
    };

    existing.trades++;
    existing.pnlSol += tradeLog.pnlSol;

    if (tradeLog.pnlSol > 0) existing.wins++;
    else if (tradeLog.pnlSol < 0) existing.losses++;

    existing.bestTrade = Math.max(existing.bestTrade, tradeLog.pnlSol);
    existing.worstTrade = Math.min(existing.worstTrade, tradeLog.pnlSol);

    this.dailyStats.set(date, existing);
  }

  getOverallStats() {
    const sells = this.tradeLogs.filter(
      (t) => t.action === "sell" || t.action === "partial_sell"
    );

    const wins = sells.filter((t) => t.pnlSol > 0);
    const losses = sells.filter((t) => t.pnlSol < 0);

    const totalPnl = sells.reduce((s, t) => s + t.pnlSol, 0);
    const avgPnl = sells.length > 0 ? totalPnl / sells.length : 0;
    const winRate = sells.length > 0 ? wins.length / sells.length : 0;

    const avgWinSize = wins.length > 0 ? wins.reduce((s, t) => s + t.pnlSol, 0) / wins.length : 0;
    const avgLossSize = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + t.pnlSol, 0) / losses.length) : 0;
    const profitFactor = avgLossSize > 0 ? avgWinSize / avgLossSize : 0;

    return {
      totalTrades: sells.length,
      wins: wins.length,
      losses: losses.length,
      winRate,
      totalPnlSol: totalPnl,
      avgPnlSol: avgPnl,
      profitFactor,
      bestTradePnl: sells.length > 0 ? Math.max(...sells.map((t) => t.pnlSol)) : 0,
      worstTradePnl: sells.length > 0 ? Math.min(...sells.map((t) => t.pnlSol)) : 0,
    };
  }

  getStatsByConfidence(): Record<string, { trades: number; winRate: number; avgPnl: number }> {
    const byConfidence: Record<string, TradeLog[]> = {};

    for (const trade of this.tradeLogs) {
      const key = String(trade.rocketScore >= 80 ? "ROCKET" : trade.rocketScore >= 60 ? "STRONG" : trade.rocketScore >= 40 ? "NORMAL" : "WEAK");
      if (!byConfidence[key]) byConfidence[key] = [];
      byConfidence[key].push(trade);
    }

    const result: Record<string, { trades: number; winRate: number; avgPnl: number }> = {};

    for (const [confidence, trades] of Object.entries(byConfidence)) {
      const sells = trades.filter((t) => t.action === "sell" || t.action === "partial_sell");
      const wins = sells.filter((t) => t.pnlSol > 0);
      result[confidence] = {
        trades: sells.length,
        winRate: sells.length > 0 ? wins.length / sells.length : 0,
        avgPnl: sells.length > 0 ? sells.reduce((s, t) => s + t.pnlSol, 0) / sells.length : 0,
      };
    }

    return result;
  }

  getTopTokens(limit: number = 10): TokenStats[] {
    const tokenMap = new Map<string, { mint: string; symbol: string; trades: TradeLog[] }>();

    for (const trade of this.tradeLogs) {
      const existing = tokenMap.get(trade.tokenMint) || {
        mint: trade.tokenMint,
        symbol: trade.tokenSymbol,
        trades: [],
      };
      existing.trades.push(trade);
      tokenMap.set(trade.tokenMint, existing);
    }

    const stats: TokenStats[] = Array.from(tokenMap.values()).map((data) => {
      const sells = data.trades.filter((t) => t.action === "sell" || t.action === "partial_sell");
      return {
        mint: data.mint,
        symbol: data.symbol,
        trades: sells.length,
        totalPnlSol: sells.reduce((s, t) => s + t.pnlSol, 0),
        avgPnlPct: sells.length > 0 ? sells.reduce((s, t) => s + t.pnlPct, 0) / sells.length : 0,
        bestPnlPct: sells.length > 0 ? Math.max(...sells.map((t) => t.pnlPct)) : 0,
      };
    });

    return stats.sort((a, b) => b.totalPnlSol - a.totalPnlSol).slice(0, limit);
  }

  getDailyStats(): DailyStats[] {
    return Array.from(this.dailyStats.values()).sort((a, b) => a.date.localeCompare(b.date));
  }

  getRecentTrades(limit: number = 20): TradeLog[] {
    return this.tradeLogs.slice(-limit);
  }

  private saveLogs(): void {
    try {
      const dir = path.dirname(this.logFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(this.logFilePath, JSON.stringify(this.tradeLogs, null, 2));
    } catch (err) {
      log.error(`Failed to save trade logs: ${err}`);
    }
  }

  private loadLogs(): void {
    try {
      if (fs.existsSync(this.logFilePath)) {
        const data = fs.readFileSync(this.logFilePath, "utf-8");
        this.tradeLogs = JSON.parse(data);
        for (const trade of this.tradeLogs) {
          this.updateDailyStats(trade);
        }
        log.info(`Loaded ${this.tradeLogs.length} trade logs`);
      }
    } catch (err) {
      log.error(`Failed to load trade logs: ${err}`);
    }
  }

  formatReport(): string {
    const stats = this.getOverallStats();
    const daily = this.getDailyStats();
    const lastDay = daily[daily.length - 1];

    return [
      "=== SMART COPY-TRADE BOT REPORT ===",
      "",
      `Total Trades: ${stats.totalTrades}`,
      `Win Rate: ${(stats.winRate * 100).toFixed(1)}%`,
      `Total PnL: ${stats.totalPnlSol >= 0 ? "+" : ""}${stats.totalPnlSol.toFixed(4)} SOL`,
      `Avg PnL/Trade: ${stats.avgPnlSol >= 0 ? "+" : ""}${stats.avgPnlSol.toFixed(4)} SOL`,
      `Profit Factor: ${stats.profitFactor.toFixed(2)}`,
      `Best Trade: +${stats.bestTradePnl.toFixed(4)} SOL`,
      `Worst Trade: ${stats.worstTradePnl.toFixed(4)} SOL`,
      "",
      lastDay
        ? `Today: ${lastDay.trades} trades, ${lastDay.pnlSol >= 0 ? "+" : ""}${lastDay.pnlSol.toFixed(4)} SOL`
        : "No trades today",
    ].join("\n");
  }
}
