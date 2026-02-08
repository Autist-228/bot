import { WalletCrawler } from "./crawler/walletCrawler";
import { DevTracker } from "./crawler/devTracker";
import { RealtimeMonitor } from "./monitor/realtimeMonitor";
import { TwitterMonitor } from "./monitor/twitterMonitor";
import { TokenAnalyzer } from "./analyzer/tokenAnalyzer";
import { RocketDetector } from "./scoring/rocketScore";
import { TradeFilter } from "./trading/tradeFilter";
import { TradeExecutor } from "./trading/tradeExecutor";
import { PositionManager } from "./trading/positionManager";
import { RiskManager } from "./trading/riskManager";
import { TelegramNotifier } from "./notifications/telegramBot";
import { Analytics } from "./analytics/analytics";
import { config } from "./config";
import { createLogger } from "./utils/logger";
import { WalletTrade, TrackedWallet } from "./types";

const log = createLogger("Main");

class SmartCopyTradeBot {
  private crawler: WalletCrawler;
  private devTracker: DevTracker;
  private monitor: RealtimeMonitor;
  private twitter: TwitterMonitor;
  private tokenAnalyzer: TokenAnalyzer;
  private rocketDetector: RocketDetector;
  private tradeFilter: TradeFilter;
  private tradeExecutor: TradeExecutor;
  private positionManager: PositionManager;
  private riskManager: RiskManager;
  private telegram: TelegramNotifier;
  private analytics: Analytics;
  private isRunning = false;

  constructor() {
    this.crawler = new WalletCrawler();
    this.devTracker = new DevTracker();
    this.monitor = new RealtimeMonitor();
    this.twitter = new TwitterMonitor();
    this.tokenAnalyzer = new TokenAnalyzer();
    this.rocketDetector = new RocketDetector();
    this.tradeFilter = new TradeFilter(
      this.tokenAnalyzer,
      this.rocketDetector,
      this.devTracker
    );
    this.tradeExecutor = new TradeExecutor();
    this.positionManager = new PositionManager();
    this.riskManager = new RiskManager();
    this.telegram = new TelegramNotifier();
    this.analytics = new Analytics();
  }

  async start(): Promise<void> {
    log.info("=== SMART COPY-TRADE BOT STARTING ===");
    log.info(`Bank: ${config.trading.bankSizeSol} SOL`);
    log.info(`Max position: ${config.trading.maxPositionPct}%`);
    log.info(`Max open positions: ${config.trading.maxOpenPositions}`);
    log.info(`Min rocket score: ${config.trading.minRocketScore}`);

    this.isRunning = true;

    await this.riskManager.syncBalance();
    this.riskManager.startDailyReset();

    log.info("[1/5] Starting Wallet Crawler...");
    await this.crawler.start();

    log.info("[2/5] Starting Real-Time Monitor...");
    const topWallets = this.crawler.getTopWallets();
    if (topWallets.length > 0) {
      await this.monitor.start(topWallets);
    } else {
      log.warn("No wallets found yet, monitor will start when wallets are discovered");
    }

    log.info("[3/5] Starting Twitter Monitor...");
    await this.twitter.start();

    log.info("[4/5] Starting Position Manager...");
    await this.positionManager.startMonitoring();

    log.info("[5/5] Connecting event handlers...");
    this.connectEventHandlers();

    log.info("=== BOT IS LIVE ===");
    await this.telegram.sendAlert(
      "Bot Started",
      `Bank: ${config.trading.bankSizeSol} SOL\nTracking ${topWallets.length} wallets`
    );

    this.startPeriodicTasks();
  }

  async stop(): Promise<void> {
    log.info("=== SHUTTING DOWN ===");
    this.isRunning = false;

    this.crawler.stop();
    this.monitor.stop();
    this.twitter.stop();
    this.positionManager.stopMonitoring();
    this.riskManager.stopDailyReset();

    const report = this.analytics.formatReport();
    log.info(report);
    await this.telegram.sendAlert("Bot Stopped", report);

    log.info("=== SHUTDOWN COMPLETE ===");
  }

  private connectEventHandlers(): void {
    this.monitor.onTrade(async (trade: WalletTrade) => {
      await this.handleWalletTrade(trade);
    });

    this.twitter.onMention((mention, tokenMint) => {
      this.tradeFilter.registerTwitterMention(tokenMint, mention);
    });

    this.positionManager.onTradeLog((tradeLog) => {
      this.riskManager.recordTrade(tradeLog);
      this.analytics.recordTrade(tradeLog);
      this.telegram.sendSellNotification(tradeLog);
    });
  }

  private async handleWalletTrade(trade: WalletTrade): Promise<void> {
    if (!this.isRunning) return;

    const trackedWallets = this.crawler.getTrackedWallets();
    const triggerWallet = trackedWallets.find((w) => w.address === trade.wallet);
    if (!triggerWallet) return;

    if (trade.type === "sell") {
      const existingPosition = this.positionManager.getPositionByToken(trade.tokenMint);
      if (existingPosition) {
        log.trade(
          `Tracked wallet ${triggerWallet.tier} selling ${trade.tokenMint.slice(0, 8)}... — watching position`
        );
      }
      return;
    }

    if (this.positionManager.getPositionCount() >= config.trading.maxOpenPositions) {
      log.warn("Max open positions reached, skipping signal");
      return;
    }

    const filterResult = await this.tradeFilter.evaluateTrade(trade, triggerWallet);

    if (!filterResult.shouldTrade || !filterResult.signal) {
      if (filterResult.rejectReason) {
        log.info(`Signal rejected: ${filterResult.rejectReason}`);
      }
      return;
    }

    const signal = filterResult.signal;
    const bankState = this.riskManager.getBankState();

    const positionSol = (signal.suggestedPositionPct / 100) * bankState.availableSol;
    const canTrade = this.riskManager.canTrade(positionSol);

    if (!canTrade.allowed) {
      log.warn(`Trade blocked by risk manager: ${canTrade.reason}`);
      return;
    }

    await this.telegram.sendSignal(signal);

    this.riskManager.lockFunds(positionSol);

    const position = await this.tradeExecutor.executeBuy(signal, bankState.availableSol);

    if (position) {
      this.positionManager.addPosition(position);
      await this.telegram.sendBuyNotification(position);

      const buyLog = this.tradeExecutor.createTradeLog(
        position,
        "buy",
        position.entryAmountSol,
        "",
        `Score: ${signal.rocketScore} | ${signal.confidence}`
      );
      this.analytics.recordTrade(buyLog);
    } else {
      this.riskManager.releaseFunds(positionSol, positionSol);
    }
  }

  private startPeriodicTasks(): void {
    setInterval(() => {
      if (!this.isRunning) return;

      const topWallets = this.crawler.getTopWallets();
      for (const wallet of topWallets) {
        if (!this.monitor.getStatus().subscribedWallets) {
          this.monitor.addWallet(wallet);
        }
      }

      this.tradeFilter.cleanupOldData();
      this.tokenAnalyzer.clearCache();
    }, 5 * 60 * 1000);

    setInterval(async () => {
      if (!this.isRunning) return;

      await this.riskManager.syncBalance();
      const stats = this.riskManager.getStats();
      const bankState = this.riskManager.getBankState();
      const portfolio = this.positionManager.getPortfolioSummary();
      const crawlerStats = this.crawler.getStats();

      log.info(
        `[STATUS] Bank: ${bankState.availableSol.toFixed(4)} SOL | ` +
        `Positions: ${portfolio.openPositions} | ` +
        `PnL: ${bankState.totalPnl >= 0 ? "+" : ""}${bankState.totalPnl.toFixed(4)} SOL | ` +
        `Wallets: ${crawlerStats.walletsTracked} | ` +
        `Win rate: ${(stats.winRate * 100).toFixed(1)}%`
      );
    }, 60 * 1000);

    setInterval(async () => {
      if (!this.isRunning) return;

      const bankState = this.riskManager.getBankState();
      const stats = this.riskManager.getStats();
      await this.telegram.sendDailyReport(bankState, {
        winRate: stats.winRate,
        totalTrades: stats.totalTrades,
        wins: stats.wins,
        losses: stats.losses,
      });
    }, 6 * 60 * 60 * 1000);
  }
}

async function main(): Promise<void> {
  const bot = new SmartCopyTradeBot();

  process.on("SIGINT", async () => {
    log.info("Received SIGINT, shutting down...");
    await bot.stop();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    log.info("Received SIGTERM, shutting down...");
    await bot.stop();
    process.exit(0);
  });

  process.on("unhandledRejection", (reason) => {
    log.error(`Unhandled rejection: ${reason}`);
  });

  await bot.start();
}

main().catch((err) => {
  log.error(`Fatal error: ${err}`);
  process.exit(1);
});
