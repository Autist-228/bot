import { RocketSignal, OpenPosition, PositionStatus, TradeLog } from "../types";
import { config } from "../config";
import { buySol } from "../utils/jupiter";
import { getBalanceSol, shortenAddress } from "../utils/solana";
import { createLogger } from "../utils/logger";

const log = createLogger("TradeExecutor");

export class TradeExecutor {
  private pendingTrades: Map<string, RocketSignal> = new Map();
  private executedSignatures: Set<string> = new Set();

  async executeBuy(signal: RocketSignal, bankAvailableSol: number): Promise<OpenPosition | null> {
    const tokenMint = signal.tokenMint;

    if (this.pendingTrades.has(tokenMint)) {
      log.warn(`Already pending trade for ${signal.tokenInfo.symbol}`);
      return null;
    }

    const positionSol = (signal.suggestedPositionPct / 100) * bankAvailableSol;

    if (positionSol < 0.001) {
      log.warn(`Position too small: ${positionSol} SOL`);
      return null;
    }

    const balance = await getBalanceSol();
    if (balance < positionSol + 0.01) {
      log.warn(`Insufficient balance: ${balance} SOL, need ${positionSol + 0.01} SOL`);
      return null;
    }

    this.pendingTrades.set(tokenMint, signal);

    try {
      log.trade(
        `Executing BUY: ${signal.tokenInfo.symbol} | ${positionSol.toFixed(4)} SOL | Score: ${signal.rocketScore} | ${signal.confidence}`
      );

      const startTime = Date.now();
      const result = await buySol(tokenMint, positionSol);
      const execTime = Date.now() - startTime;

      log.trade(
        `BUY EXECUTED in ${execTime}ms: ${signal.tokenInfo.symbol} | ${positionSol.toFixed(4)} SOL | sig: ${result.signature.slice(0, 12)}...`
      );

      this.executedSignatures.add(result.signature);

      const position: OpenPosition = {
        id: `pos_${Date.now()}_${tokenMint.slice(0, 8)}`,
        tokenMint,
        tokenSymbol: signal.tokenInfo.symbol,
        entryPrice: result.inputAmount / result.outputAmount,
        currentPrice: result.inputAmount / result.outputAmount,
        entryAmountSol: positionSol,
        remainingTokens: result.outputAmount,
        initialTokens: result.outputAmount,
        totalSoldSol: 0,
        pnlPct: 0,
        pnlSol: 0,
        rocketScore: signal.rocketScore,
        confidence: signal.confidence,
        peakPrice: result.inputAmount / result.outputAmount,
        trailingStopActive: false,
        trailingStopPct: 0,
        partialSells: [],
        entryTimestamp: Date.now(),
        triggerWallet: signal.buyingWallets[0]?.address || "unknown",
        status: PositionStatus.ACTIVE,
      };

      return position;
    } catch (err) {
      log.error(`BUY FAILED for ${signal.tokenInfo.symbol}: ${err}`);
      return null;
    } finally {
      this.pendingTrades.delete(tokenMint);
    }
  }

  createTradeLog(
    position: OpenPosition,
    action: "buy" | "sell" | "partial_sell",
    amountSol: number,
    signature: string,
    reason: string
  ): TradeLog {
    return {
      id: `log_${Date.now()}_${position.tokenMint.slice(0, 8)}`,
      tokenMint: position.tokenMint,
      tokenSymbol: position.tokenSymbol,
      action,
      amountSol,
      price: position.currentPrice,
      pnlPct: position.pnlPct,
      pnlSol: position.pnlSol,
      rocketScore: position.rocketScore,
      triggerWallet: position.triggerWallet,
      reason,
      timestamp: Date.now(),
      signature,
    };
  }

  hasPendingTrade(tokenMint: string): boolean {
    return this.pendingTrades.has(tokenMint);
  }
}
