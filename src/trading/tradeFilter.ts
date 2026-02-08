import {
  RocketSignal,
  TradeConfidence,
  WalletTrade,
  TrackedWallet,
  TokenInfo,
  TwitterMention,
} from "../types";
import { config } from "../config";
import { TokenAnalyzer } from "../analyzer/tokenAnalyzer";
import { RocketDetector } from "../scoring/rocketScore";
import { DevTracker } from "../crawler/devTracker";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";

const log = createLogger("TradeFilter");

interface FilterResult {
  shouldTrade: boolean;
  signal: RocketSignal | null;
  rejectReason: string;
}

export class TradeFilter {
  private tokenAnalyzer: TokenAnalyzer;
  private rocketDetector: RocketDetector;
  private devTracker: DevTracker;
  private recentSignals: Map<string, { timestamp: number; count: number }> = new Map();
  private twitterMentions: Map<string, TwitterMention[]> = new Map();
  private walletBuys: Map<string, TrackedWallet[]> = new Map();

  constructor(
    tokenAnalyzer: TokenAnalyzer,
    rocketDetector: RocketDetector,
    devTracker: DevTracker
  ) {
    this.tokenAnalyzer = tokenAnalyzer;
    this.rocketDetector = rocketDetector;
    this.devTracker = devTracker;
  }

  registerTwitterMention(tokenMint: string, mention: TwitterMention): void {
    const existing = this.twitterMentions.get(tokenMint) || [];
    existing.push(mention);
    this.twitterMentions.set(tokenMint, existing);
  }

  registerWalletBuy(tokenMint: string, wallet: TrackedWallet): void {
    const existing = this.walletBuys.get(tokenMint) || [];
    if (!existing.find((w) => w.address === wallet.address)) {
      existing.push(wallet);
      this.walletBuys.set(tokenMint, existing);
    }
  }

  async evaluateTrade(
    trade: WalletTrade,
    triggerWallet: TrackedWallet
  ): Promise<FilterResult> {
    if (trade.type !== "buy") {
      return { shouldTrade: false, signal: null, rejectReason: "Not a buy" };
    }

    const tokenMint = trade.tokenMint;

    if (this.isDuplicateSignal(tokenMint)) {
      return { shouldTrade: false, signal: null, rejectReason: "Duplicate signal (cooldown)" };
    }

    log.info(`Evaluating trade: ${shortenAddress(tokenMint)} from ${shortenAddress(trade.wallet)} (${triggerWallet.tier})`);

    const tokenInfo = await this.tokenAnalyzer.analyzeToken(tokenMint);
    if (!tokenInfo) {
      return { shouldTrade: false, signal: null, rejectReason: "Token analysis failed" };
    }

    const skipCheck = this.tokenAnalyzer.shouldSkipToken(tokenInfo);
    if (skipCheck.skip) {
      log.warn(`Skipping ${tokenInfo.symbol}: ${skipCheck.reason}`);
      return { shouldTrade: false, signal: null, rejectReason: skipCheck.reason };
    }

    if (tokenInfo.devAddress) {
      const devHistory = await this.devTracker.analyzeDevWallet(tokenInfo.devAddress);
      tokenInfo.devHistory = devHistory;

      if (this.devTracker.isKnownRugDev(tokenInfo.devAddress)) {
        return { shouldTrade: false, signal: null, rejectReason: "Known rug dev" };
      }
    }

    this.registerWalletBuy(tokenMint, triggerWallet);

    const buyingWallets = this.walletBuys.get(tokenMint) || [triggerWallet];
    const mentions = this.twitterMentions.get(tokenMint) || [];
    const safetyScore = this.tokenAnalyzer.getTokenSafetyScore(tokenInfo);

    const signal = this.rocketDetector.buildSignal(
      tokenInfo,
      buyingWallets,
      mentions,
      safetyScore
    );

    if (signal.confidence === TradeConfidence.SKIP) {
      return {
        shouldTrade: false,
        signal,
        rejectReason: `Score too low: ${signal.rocketScore}`,
      };
    }

    if (signal.rocketScore < config.trading.minRocketScore) {
      return {
        shouldTrade: false,
        signal,
        rejectReason: `Score ${signal.rocketScore} below minimum ${config.trading.minRocketScore}`,
      };
    }

    if (tokenInfo.liquidity < 5000 && signal.confidence !== TradeConfidence.ROCKET) {
      return {
        shouldTrade: false,
        signal,
        rejectReason: "Low liquidity for non-rocket signal",
      };
    }

    this.markSignalSeen(tokenMint);

    log.trade(
      `APPROVED: ${tokenInfo.symbol} | Score: ${signal.rocketScore} | ${signal.confidence} | Position: ${signal.suggestedPositionPct.toFixed(1)}%`
    );

    return { shouldTrade: true, signal, rejectReason: "" };
  }

  private isDuplicateSignal(tokenMint: string): boolean {
    const recent = this.recentSignals.get(tokenMint);
    if (!recent) return false;

    if (Date.now() - recent.timestamp < 5 * 60 * 1000) {
      return true;
    }

    this.recentSignals.delete(tokenMint);
    return false;
  }

  private markSignalSeen(tokenMint: string): void {
    const existing = this.recentSignals.get(tokenMint);
    this.recentSignals.set(tokenMint, {
      timestamp: Date.now(),
      count: (existing?.count || 0) + 1,
    });
  }

  cleanupOldData(): void {
    const now = Date.now();
    const maxAge = 30 * 60 * 1000;

    for (const [mint, data] of this.recentSignals) {
      if (now - data.timestamp > maxAge) {
        this.recentSignals.delete(mint);
      }
    }

    for (const [mint, mentions] of this.twitterMentions) {
      const fresh = mentions.filter((m) => now - m.timestamp < maxAge);
      if (fresh.length === 0) {
        this.twitterMentions.delete(mint);
      } else {
        this.twitterMentions.set(mint, fresh);
      }
    }

    for (const [mint, wallets] of this.walletBuys) {
      if (wallets.length === 0) {
        this.walletBuys.delete(mint);
      }
    }
  }
}
