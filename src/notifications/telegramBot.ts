import TelegramBot from "node-telegram-bot-api";
import { config } from "../config";
import { RocketSignal, OpenPosition, TradeLog, BankState, TradeConfidence } from "../types";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";

const log = createLogger("Telegram");

export class TelegramNotifier {
  private bot: TelegramBot | null = null;
  private chatId: string;
  private enabled: boolean;

  constructor() {
    this.chatId = config.telegram.chatId;
    this.enabled = !!config.telegram.botToken && !!config.telegram.chatId;

    if (this.enabled) {
      this.bot = new TelegramBot(config.telegram.botToken, { polling: false });
      log.info("Telegram bot initialized");
    } else {
      log.warn("Telegram not configured, notifications disabled");
    }
  }

  async sendSignal(signal: RocketSignal): Promise<void> {
    if (!this.enabled) return;

    const emoji = this.getConfidenceEmoji(signal.confidence);
    const message = [
      `${emoji} <b>NEW SIGNAL: ${signal.tokenInfo.symbol}</b>`,
      ``,
      `Score: <b>${signal.rocketScore}/100</b> | ${signal.confidence}`,
      `Dev: ${signal.devScore} | SM: ${signal.smartMoneyScore} | TW: ${signal.twitterScore}`,
      `Token: ${signal.tokenScore} | Chain: ${signal.onChainScore}`,
      ``,
      `Liquidity: $${this.formatNumber(signal.tokenInfo.liquidity)}`,
      `Wallets buying: ${signal.buyingWallets.length}`,
      `Twitter mentions: ${signal.twitterMentions.length}`,
      ``,
      `Position: ${signal.suggestedPositionPct.toFixed(1)}%`,
      `Mint: <code>${signal.tokenMint}</code>`,
    ].join("\n");

    await this.send(message);
  }

  async sendBuyNotification(position: OpenPosition): Promise<void> {
    if (!this.enabled) return;

    const message = [
      `<b>BUY EXECUTED</b>`,
      ``,
      `Token: <b>${position.tokenSymbol}</b>`,
      `Amount: ${position.entryAmountSol.toFixed(4)} SOL`,
      `Score: ${position.rocketScore} | ${position.confidence}`,
      `Trigger: ${shortenAddress(position.triggerWallet)}`,
    ].join("\n");

    await this.send(message);
  }

  async sendSellNotification(tradeLog: TradeLog): Promise<void> {
    if (!this.enabled) return;

    const pnlEmoji = tradeLog.pnlSol >= 0 ? "+" : "";
    const message = [
      `<b>${tradeLog.action === "sell" ? "FULL SELL" : "PARTIAL SELL"}</b>`,
      ``,
      `Token: <b>${tradeLog.tokenSymbol}</b>`,
      `Amount: ${tradeLog.amountSol.toFixed(4)} SOL`,
      `PnL: ${pnlEmoji}${tradeLog.pnlPct.toFixed(1)}% (${pnlEmoji}${tradeLog.pnlSol.toFixed(4)} SOL)`,
      `Reason: ${tradeLog.reason}`,
    ].join("\n");

    await this.send(message);
  }

  async sendDailyReport(
    bankState: BankState,
    stats: { winRate: number; totalTrades: number; wins: number; losses: number }
  ): Promise<void> {
    if (!this.enabled) return;

    const pnlEmoji = bankState.dailyPnl >= 0 ? "+" : "";
    const message = [
      `<b>DAILY REPORT</b>`,
      ``,
      `Bank: ${bankState.totalSol.toFixed(4)} SOL`,
      `Available: ${bankState.availableSol.toFixed(4)} SOL`,
      `In positions: ${bankState.lockedInPositions.toFixed(4)} SOL`,
      ``,
      `Daily PnL: ${pnlEmoji}${bankState.dailyPnl.toFixed(4)} SOL`,
      `Weekly PnL: ${bankState.weeklyPnl >= 0 ? "+" : ""}${bankState.weeklyPnl.toFixed(4)} SOL`,
      `Total PnL: ${bankState.totalPnl >= 0 ? "+" : ""}${bankState.totalPnl.toFixed(4)} SOL`,
      ``,
      `Trades today: ${stats.totalTrades}`,
      `Win rate: ${(stats.winRate * 100).toFixed(1)}%`,
      `W/L: ${stats.wins}/${stats.losses}`,
    ].join("\n");

    await this.send(message);
  }

  async sendAlert(title: string, details: string): Promise<void> {
    if (!this.enabled) return;

    const message = `<b>${title}</b>\n\n${details}`;
    await this.send(message);
  }

  private async send(message: string): Promise<void> {
    if (!this.bot || !this.chatId) return;

    try {
      await this.bot.sendMessage(this.chatId, message, { parse_mode: "HTML" });
    } catch (err) {
      log.error(`Telegram send failed: ${err}`);
    }
  }

  private getConfidenceEmoji(confidence: TradeConfidence): string {
    switch (confidence) {
      case TradeConfidence.ROCKET: return "[ROCKET]";
      case TradeConfidence.STRONG: return "[STRONG]";
      case TradeConfidence.NORMAL: return "[NORMAL]";
      case TradeConfidence.WEAK: return "[WEAK]";
      default: return "[SKIP]";
    }
  }

  private formatNumber(num: number): string {
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(2)}K`;
    return num.toFixed(2);
  }
}
