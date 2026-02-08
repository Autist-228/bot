import axios from "axios";
import { config } from "../config";
import { TrackedWallet, WalletTier, HeliusTransaction } from "../types";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";

const log = createLogger("WalletCrawler");

const KNOWN_DEX_PROGRAMS = [
  "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
  "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
  "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
  "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",
];

interface WalletTradeHistory {
  address: string;
  trades: ParsedTrade[];
}

interface ParsedTrade {
  tokenMint: string;
  type: "buy" | "sell";
  solAmount: number;
  timestamp: number;
  signature: string;
}

interface WalletPnL {
  address: string;
  totalTrades: number;
  profitableTrades: number;
  winRate: number;
  totalPnlSol: number;
  avgRoiPct: number;
  maxDrawdownPct: number;
  consistency: number;
  avgEntrySpeed: number;
}

export class WalletCrawler {
  private trackedWallets: Map<string, TrackedWallet> = new Map();
  private blacklist: Set<string> = new Set();
  private scanInterval: ReturnType<typeof setInterval> | null = null;

  getTrackedWallets(): TrackedWallet[] {
    return Array.from(this.trackedWallets.values())
      .filter((w) => !w.blacklisted)
      .sort((a, b) => b.score - a.score);
  }

  getTopWallets(count: number = config.crawler.topWalletsCount): TrackedWallet[] {
    return this.getTrackedWallets().slice(0, count);
  }

  isBlacklisted(address: string): boolean {
    return this.blacklist.has(address);
  }

  blacklistWallet(address: string, reason: string): void {
    this.blacklist.add(address);
    const wallet = this.trackedWallets.get(address);
    if (wallet) {
      wallet.blacklisted = true;
    }
    log.warn(`Blacklisted wallet ${shortenAddress(address)}: ${reason}`);
  }

  async start(): Promise<void> {
    log.info("Starting Wallet Crawler...");
    await this.runFullScan();

    this.scanInterval = setInterval(
      () => this.runFullScan(),
      config.crawler.scanIntervalMs
    );

    log.info(
      `Crawler running, scan interval: ${config.crawler.scanIntervalMs / 1000}s`
    );
  }

  stop(): void {
    if (this.scanInterval) {
      clearInterval(this.scanInterval);
      this.scanInterval = null;
    }
    log.info("Wallet Crawler stopped");
  }

  async runFullScan(): Promise<void> {
    log.info("Running full wallet scan...");

    try {
      const dexWallets = await this.scanDexTransactions();
      log.info(`Found ${dexWallets.length} active wallets from DEX scan`);

      let analyzed = 0;
      for (const address of dexWallets) {
        if (this.blacklist.has(address)) continue;

        try {
          const pnl = await this.analyzeWalletPnL(address);
          if (pnl && this.meetsMinimumCriteria(pnl)) {
            const wallet = this.pnlToTrackedWallet(pnl);
            this.trackedWallets.set(address, wallet);
            analyzed++;
          }
        } catch {
          continue;
        }

        if (analyzed % 10 === 0 && analyzed > 0) {
          log.info(`Analyzed ${analyzed} wallets so far...`);
        }
      }

      this.pruneStaleWallets();

      log.info(
        `Scan complete. Tracking ${this.trackedWallets.size} wallets, ${this.blacklist.size} blacklisted`
      );
    } catch (err) {
      log.error(`Scan failed: ${err}`);
    }
  }

  private async scanDexTransactions(): Promise<string[]> {
    const walletSet = new Set<string>();

    try {
      const response = await axios.get(
        `${config.helius.apiUrl}/transactions/?api-key=${config.helius.apiKey}`,
        {
          params: {
            type: "SWAP",
            source: "JUPITER",
          },
          timeout: 15000,
        }
      );

      const transactions: HeliusTransaction[] = response.data || [];

      for (const tx of transactions) {
        if (tx.feePayer && !this.isKnownProgram(tx.feePayer)) {
          walletSet.add(tx.feePayer);
        }
      }
    } catch (err) {
      log.warn(`DEX scan via Helius failed, trying alternative: ${err}`);
    }

    try {
      const response = await axios.post(
        config.codex.apiUrl,
        {
          query: `{
            getLatestTokens(limit: 50, networkId: 1399811149) {
              items {
                address
                creatorAddress
              }
            }
          }`,
        },
        {
          headers: {
            Authorization: config.codex.apiKey,
            "Content-Type": "application/json",
          },
          timeout: 10000,
        }
      );

      const tokens = response.data?.data?.getLatestTokens?.items || [];
      for (const token of tokens) {
        if (token.creatorAddress) {
          walletSet.add(token.creatorAddress);
        }
      }
    } catch (err) {
      log.warn(`Codex token scan failed: ${err}`);
    }

    return Array.from(walletSet);
  }

  private async getWalletTransactions(address: string): Promise<HeliusTransaction[]> {
    try {
      const response = await axios.get(
        `${config.helius.apiUrl}/addresses/${address}/transactions/?api-key=${config.helius.apiKey}`,
        {
          params: { limit: 100 },
          timeout: 15000,
        }
      );
      return response.data || [];
    } catch {
      return [];
    }
  }

  private async analyzeWalletPnL(address: string): Promise<WalletPnL | null> {
    const transactions = await this.getWalletTransactions(address);
    if (transactions.length < config.crawler.minTradesForScoring) return null;

    const trades = this.parseSwapTransactions(address, transactions);
    if (trades.length < config.crawler.minTradesForScoring) return null;

    const tokenTrades = new Map<string, ParsedTrade[]>();
    for (const trade of trades) {
      const existing = tokenTrades.get(trade.tokenMint) || [];
      existing.push(trade);
      tokenTrades.set(trade.tokenMint, existing);
    }

    let profitableTrades = 0;
    let totalPnlSol = 0;
    let totalRoiPct = 0;
    let completedTrades = 0;
    let maxDrawdown = 0;
    let runningPnl = 0;
    let peakPnl = 0;
    const entryDelays: number[] = [];

    for (const [, tokenTradeList] of tokenTrades) {
      const buys = tokenTradeList.filter((t) => t.type === "buy");
      const sells = tokenTradeList.filter((t) => t.type === "sell");

      if (buys.length === 0 || sells.length === 0) continue;

      const totalBought = buys.reduce((s, t) => s + t.solAmount, 0);
      const totalSold = sells.reduce((s, t) => s + t.solAmount, 0);
      const pnl = totalSold - totalBought;
      const roi = totalBought > 0 ? (pnl / totalBought) * 100 : 0;

      completedTrades++;
      totalPnlSol += pnl;
      totalRoiPct += roi;

      if (pnl > 0) profitableTrades++;

      runningPnl += pnl;
      peakPnl = Math.max(peakPnl, runningPnl);
      const drawdown = peakPnl > 0 ? ((peakPnl - runningPnl) / peakPnl) * 100 : 0;
      maxDrawdown = Math.max(maxDrawdown, drawdown);

      if (buys[0]) {
        entryDelays.push(buys[0].timestamp);
      }
    }

    if (completedTrades < 5) return null;

    const winRate = profitableTrades / completedTrades;
    const avgRoi = totalRoiPct / completedTrades;
    const avgEntrySpeed = entryDelays.length > 0
      ? entryDelays.reduce((a, b) => a + b, 0) / entryDelays.length
      : 0;

    const roiValues = Array.from(tokenTrades.values()).map((tl) => {
      const b = tl.filter((t) => t.type === "buy").reduce((s, t) => s + t.solAmount, 0);
      const sl = tl.filter((t) => t.type === "sell").reduce((s, t) => s + t.solAmount, 0);
      return b > 0 ? ((sl - b) / b) * 100 : 0;
    });
    const avgRoiAll = roiValues.reduce((a, b) => a + b, 0) / roiValues.length;
    const variance = roiValues.reduce((s, r) => s + Math.pow(r - avgRoiAll, 2), 0) / roiValues.length;
    const stdDev = Math.sqrt(variance);
    const consistency = avgRoiAll > 0 ? Math.max(0, 100 - stdDev / avgRoiAll * 100) : 0;

    return {
      address,
      totalTrades: completedTrades,
      profitableTrades,
      winRate,
      totalPnlSol: totalPnlSol,
      avgRoiPct: avgRoi,
      maxDrawdownPct: maxDrawdown,
      consistency: Math.min(100, Math.max(0, consistency)),
      avgEntrySpeed,
    };
  }

  private parseSwapTransactions(
    walletAddress: string,
    transactions: HeliusTransaction[]
  ): ParsedTrade[] {
    const trades: ParsedTrade[] = [];

    for (const tx of transactions) {
      if (tx.type !== "SWAP" && tx.source !== "JUPITER") continue;

      for (const transfer of tx.tokenTransfers || []) {
        if (!transfer.mint) continue;

        const isBuy = transfer.toUserAccount === walletAddress;
        const isSell = transfer.fromUserAccount === walletAddress;

        if (!isBuy && !isSell) continue;

        const solTransfer = (tx.nativeTransfers || []).find(
          (nt) =>
            (isBuy && nt.fromUserAccount === walletAddress) ||
            (isSell && nt.toUserAccount === walletAddress)
        );

        const solAmount = solTransfer ? Math.abs(solTransfer.amount) / 1e9 : 0;

        if (solAmount > 0) {
          trades.push({
            tokenMint: transfer.mint,
            type: isBuy ? "buy" : "sell",
            solAmount,
            timestamp: tx.timestamp,
            signature: tx.signature,
          });
        }
      }
    }

    return trades;
  }

  private meetsMinimumCriteria(pnl: WalletPnL): boolean {
    return (
      pnl.winRate >= config.crawler.minWinRate &&
      pnl.totalTrades >= config.crawler.minTradesForScoring &&
      pnl.avgRoiPct > 0 &&
      pnl.totalPnlSol > 0
    );
  }

  private pnlToTrackedWallet(pnl: WalletPnL): TrackedWallet {
    const score = this.calculateWalletScore(pnl);
    const tier = this.scoreToTier(score);

    return {
      address: pnl.address,
      tier,
      winRate: pnl.winRate,
      avgRoi: pnl.avgRoiPct,
      totalTrades: pnl.totalTrades,
      profitableTrades: pnl.profitableTrades,
      maxDrawdown: pnl.maxDrawdownPct,
      avgEntrySpeed: pnl.avgEntrySpeed,
      consistency: pnl.consistency,
      score,
      lastActive: Date.now(),
      addedAt: Date.now(),
      linkedWallets: [],
      isDevWallet: false,
      blacklisted: false,
    };
  }

  private calculateWalletScore(pnl: WalletPnL): number {
    const w = config.scoring.weights;

    const winRateScore = Math.min(100, pnl.winRate * 100 * 1.3);
    const roiScore = Math.min(100, Math.max(0, pnl.avgRoiPct));
    const activityScore = Math.min(100, (pnl.totalTrades / 50) * 100);
    const drawdownScore = Math.max(0, 100 - pnl.maxDrawdownPct);
    const speedScore = 50;
    const consistencyScore = pnl.consistency;

    const totalScore =
      winRateScore * w.winRate +
      roiScore * w.avgRoi +
      activityScore * w.activity +
      drawdownScore * w.maxDrawdown +
      speedScore * w.entrySpeed +
      consistencyScore * w.consistency;

    return Math.min(100, Math.max(0, totalScore));
  }

  private scoreToTier(score: number): WalletTier {
    const t = config.scoring.tierThresholds;
    if (score >= t.S) return WalletTier.S;
    if (score >= t.A) return WalletTier.A;
    if (score >= t.B) return WalletTier.B;
    return WalletTier.C;
  }

  private pruneStaleWallets(): void {
    const now = Date.now();
    const staleThreshold = 7 * 24 * 60 * 60 * 1000;

    for (const [address, wallet] of this.trackedWallets) {
      if (now - wallet.lastActive > staleThreshold) {
        this.trackedWallets.delete(address);
        log.info(`Pruned stale wallet: ${shortenAddress(address)}`);
      }

      if (wallet.winRate < 0.35 && wallet.totalTrades > 30) {
        this.blacklistWallet(address, "Win rate dropped below 35%");
      }
    }
  }

  private isKnownProgram(address: string): boolean {
    return KNOWN_DEX_PROGRAMS.includes(address);
  }

  getStats() {
    const wallets = this.getTrackedWallets();
    return {
      walletsTracked: wallets.length,
      walletsBlacklisted: this.blacklist.size,
      tiers: {
        S: wallets.filter((w) => w.tier === WalletTier.S).length,
        A: wallets.filter((w) => w.tier === WalletTier.A).length,
        B: wallets.filter((w) => w.tier === WalletTier.B).length,
        C: wallets.filter((w) => w.tier === WalletTier.C).length,
      },
      topScore: wallets[0]?.score || 0,
    };
  }
}
