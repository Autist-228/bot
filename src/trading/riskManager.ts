import { BankState, TradeLog, TradeConfidence } from "../types";
import { config } from "../config";
import { getBalanceSol } from "../utils/solana";
import { createLogger } from "../utils/logger";

const log = createLogger("RiskManager");

export class RiskManager {
  private bank: BankState;
  private tradeLogs: TradeLog[] = [];
  private dailyResetInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.bank = {
      totalSol: config.trading.bankSizeSol,
      availableSol: config.trading.bankSizeSol,
      lockedInPositions: 0,
      dailyPnl: 0,
      weeklyPnl: 0,
      totalPnl: 0,
      dailyLossLimit: config.trading.bankSizeSol * (config.trading.dailyLossLimitPct / 100),
      isPaused: false,
      pauseUntil: 0,
      tradeCount: 0,
    };
  }

  async syncBalance(): Promise<void> {
    try {
      const balance = await getBalanceSol();
      this.bank.totalSol = balance;
      this.bank.availableSol = balance - this.bank.lockedInPositions;
      log.info(`Balance synced: ${balance.toFixed(4)} SOL (available: ${this.bank.availableSol.toFixed(4)} SOL)`);
    } catch (err) {
      log.error(`Balance sync failed: ${err}`);
    }
  }

  canTrade(requiredSol: number): { allowed: boolean; reason: string } {
    if (this.bank.isPaused) {
      if (Date.now() < this.bank.pauseUntil) {
        return { allowed: false, reason: `Trading paused until ${new Date(this.bank.pauseUntil).toISOString()}` };
      }
      this.bank.isPaused = false;
    }

    if (this.bank.dailyPnl < -this.bank.dailyLossLimit) {
      this.pauseTrading(60 * 60 * 1000, "Daily loss limit reached");
      return { allowed: false, reason: "Daily loss limit reached" };
    }

    if (this.bank.availableSol < requiredSol) {
      return { allowed: false, reason: `Insufficient funds: ${this.bank.availableSol.toFixed(4)} < ${requiredSol.toFixed(4)} SOL` };
    }

    if (requiredSol > this.bank.totalSol * (config.trading.maxPositionPct / 100)) {
      return { allowed: false, reason: "Position exceeds max size" };
    }

    return { allowed: true, reason: "" };
  }

  lockFunds(amount: number): void {
    this.bank.availableSol -= amount;
    this.bank.lockedInPositions += amount;
    this.bank.tradeCount++;
    log.info(`Locked ${amount.toFixed(4)} SOL | Available: ${this.bank.availableSol.toFixed(4)} SOL`);
  }

  releaseFunds(originalAmount: number, returnedAmount: number): void {
    this.bank.lockedInPositions -= originalAmount;
    this.bank.availableSol += returnedAmount;

    const pnl = returnedAmount - originalAmount;
    this.bank.dailyPnl += pnl;
    this.bank.weeklyPnl += pnl;
    this.bank.totalPnl += pnl;

    log.info(
      `Released ${returnedAmount.toFixed(4)} SOL (PnL: ${pnl >= 0 ? "+" : ""}${pnl.toFixed(4)} SOL) | Available: ${this.bank.availableSol.toFixed(4)} SOL`
    );
  }

  recordTrade(tradeLog: TradeLog): void {
    this.tradeLogs.push(tradeLog);

    if (tradeLog.action === "sell" || tradeLog.action === "partial_sell") {
      this.bank.dailyPnl += tradeLog.pnlSol;
      this.bank.weeklyPnl += tradeLog.pnlSol;
      this.bank.totalPnl += tradeLog.pnlSol;
    }

    if (this.bank.dailyPnl < -this.bank.dailyLossLimit) {
      this.pauseTrading(60 * 60 * 1000, "Daily loss limit reached");
    }
  }

  getMaxPositionSol(confidence: TradeConfidence): number {
    const maxPct = config.trading.maxPositionPct;
    let multiplier = 1;

    switch (confidence) {
      case TradeConfidence.ROCKET:
        multiplier = 1.0;
        break;
      case TradeConfidence.STRONG:
        multiplier = 0.67;
        break;
      case TradeConfidence.NORMAL:
        multiplier = 0.33;
        break;
      default:
        multiplier = 0.15;
    }

    return this.bank.availableSol * (maxPct / 100) * multiplier;
  }

  private pauseTrading(durationMs: number, reason: string): void {
    this.bank.isPaused = true;
    this.bank.pauseUntil = Date.now() + durationMs;
    log.warn(`TRADING PAUSED for ${durationMs / 60000}min: ${reason}`);
  }

  startDailyReset(): void {
    const msUntilMidnight = this.getMsUntilMidnightUTC();

    setTimeout(() => {
      this.resetDailyStats();
      this.dailyResetInterval = setInterval(
        () => this.resetDailyStats(),
        24 * 60 * 60 * 1000
      );
    }, msUntilMidnight);

    log.info(`Daily reset scheduled in ${(msUntilMidnight / 60000).toFixed(0)} minutes`);
  }

  stopDailyReset(): void {
    if (this.dailyResetInterval) {
      clearInterval(this.dailyResetInterval);
      this.dailyResetInterval = null;
    }
  }

  private resetDailyStats(): void {
    log.info(`Daily reset: PnL was ${this.bank.dailyPnl >= 0 ? "+" : ""}${this.bank.dailyPnl.toFixed(4)} SOL`);
    this.bank.dailyPnl = 0;
    this.bank.isPaused = false;
    this.bank.tradeCount = 0;
  }

  private getMsUntilMidnightUTC(): number {
    const now = new Date();
    const midnight = new Date(now);
    midnight.setUTCHours(24, 0, 0, 0);
    return midnight.getTime() - now.getTime();
  }

  getBankState(): BankState {
    return { ...this.bank };
  }

  getTradeLogs(): TradeLog[] {
    return [...this.tradeLogs];
  }

  getStats() {
    const wins = this.tradeLogs.filter((t) => t.pnlSol > 0).length;
    const losses = this.tradeLogs.filter((t) => t.pnlSol < 0).length;
    const totalTrades = wins + losses;

    return {
      bank: this.bank,
      winRate: totalTrades > 0 ? wins / totalTrades : 0,
      totalTrades,
      wins,
      losses,
      avgPnlSol: totalTrades > 0 ? this.bank.totalPnl / totalTrades : 0,
      bestTrade: this.tradeLogs.reduce((best, t) => (t.pnlSol > best.pnlSol ? t : best), this.tradeLogs[0]),
      worstTrade: this.tradeLogs.reduce((worst, t) => (t.pnlSol < worst.pnlSol ? t : worst), this.tradeLogs[0]),
    };
  }
}
