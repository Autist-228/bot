import WebSocket from "ws";
import { config } from "../config";
import { TrackedWallet, WalletTrade } from "../types";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";

const log = createLogger("RealtimeMonitor");

type TradeCallback = (trade: WalletTrade) => void;

export class RealtimeMonitor {
  private ws: WebSocket | null = null;
  private subscribedWallets: Map<string, TrackedWallet> = new Map();
  private onTradeCallbacks: TradeCallback[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 2000;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private isRunning = false;

  onTrade(callback: TradeCallback): void {
    this.onTradeCallbacks.push(callback);
  }

  async start(wallets: TrackedWallet[]): Promise<void> {
    this.isRunning = true;
    for (const wallet of wallets) {
      this.subscribedWallets.set(wallet.address, wallet);
    }

    log.info(`Starting real-time monitor for ${wallets.length} wallets`);
    await this.connect();
  }

  stop(): void {
    this.isRunning = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    log.info("Real-time monitor stopped");
  }

  addWallet(wallet: TrackedWallet): void {
    this.subscribedWallets.set(wallet.address, wallet);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.subscribeToWallet(wallet.address);
    }
    log.info(`Added wallet to monitor: ${shortenAddress(wallet.address)} (${wallet.tier})`);
  }

  removeWallet(address: string): void {
    this.subscribedWallets.delete(address);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.unsubscribeFromWallet(address);
    }
  }

  private async connect(): Promise<void> {
    const wsUrl = config.helius.wsUrl;
    log.info(`Connecting to WebSocket: ${wsUrl.slice(0, 40)}...`);

    this.ws = new WebSocket(wsUrl);

    this.ws.on("open", () => {
      log.info("WebSocket connected");
      this.reconnectAttempts = 0;

      for (const address of this.subscribedWallets.keys()) {
        this.subscribeToWallet(address);
      }

      this.startPingInterval();
    });

    this.ws.on("message", (data: WebSocket.Data) => {
      this.handleMessage(data);
    });

    this.ws.on("error", (err: Error) => {
      log.error(`WebSocket error: ${err.message}`);
    });

    this.ws.on("close", () => {
      log.warn("WebSocket disconnected");
      if (this.pingInterval) {
        clearInterval(this.pingInterval);
        this.pingInterval = null;
      }
      this.attemptReconnect();
    });
  }

  private subscribeToWallet(address: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const subscribeMsg = {
      jsonrpc: "2.0",
      id: 1,
      method: "accountSubscribe",
      params: [
        address,
        {
          encoding: "jsonParsed",
          commitment: "confirmed",
        },
      ],
    };

    this.ws.send(JSON.stringify(subscribeMsg));
    log.debug(`Subscribed to wallet: ${shortenAddress(address)}`);
  }

  private unsubscribeFromWallet(address: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const unsubMsg = {
      jsonrpc: "2.0",
      id: 2,
      method: "accountUnsubscribe",
      params: [address],
    };

    this.ws.send(JSON.stringify(unsubMsg));
  }

  private handleMessage(data: WebSocket.Data): void {
    try {
      const message = JSON.parse(data.toString());

      if (message.method === "accountNotification") {
        this.processAccountNotification(message);
      }
    } catch (err) {
      log.debug(`Failed to parse WS message: ${err}`);
    }
  }

  private async processAccountNotification(message: unknown): Promise<void> {
    const msg = message as {
      params?: {
        result?: {
          value?: {
            data?: unknown;
            lamports?: number;
            owner?: string;
          };
          context?: { slot: number };
        };
        subscription?: number;
      };
    };

    if (!msg.params?.result?.value) return;

    for (const [address, wallet] of this.subscribedWallets) {
      try {
        const recentTrades = await this.fetchRecentTrades(address);
        for (const trade of recentTrades) {
          log.trade(
            `[${wallet.tier}] ${shortenAddress(address)} ${trade.type.toUpperCase()} ${trade.amountSol.toFixed(4)} SOL of ${shortenAddress(trade.tokenMint)}`
          );

          for (const callback of this.onTradeCallbacks) {
            try {
              callback(trade);
            } catch (err) {
              log.error(`Trade callback error: ${err}`);
            }
          }
        }
      } catch {
        continue;
      }
    }
  }

  private async fetchRecentTrades(walletAddress: string): Promise<WalletTrade[]> {
    const trades: WalletTrade[] = [];

    try {
      const response = await fetch(
        `${config.helius.apiUrl}/addresses/${walletAddress}/transactions/?api-key=${config.helius.apiKey}&limit=5`,
        { signal: AbortSignal.timeout(10000) }
      );

      const transactions = (await response.json()) as Array<{
        type: string;
        source: string;
        timestamp: number;
        signature: string;
        tokenTransfers?: Array<{
          mint: string;
          toUserAccount: string;
          fromUserAccount: string;
          tokenAmount: number;
        }>;
        nativeTransfers?: Array<{
          fromUserAccount: string;
          toUserAccount: string;
          amount: number;
        }>;
      }>;

      for (const tx of transactions) {
        if (tx.type !== "SWAP") continue;

        for (const transfer of tx.tokenTransfers || []) {
          if (!transfer.mint) continue;

          const isBuy = transfer.toUserAccount === walletAddress;
          const isSell = transfer.fromUserAccount === walletAddress;

          if (!isBuy && !isSell) continue;

          const solTransfer = (tx.nativeTransfers || []).find(
            (nt: { fromUserAccount: string; toUserAccount: string; amount: number }) =>
              (isBuy && nt.fromUserAccount === walletAddress) ||
              (isSell && nt.toUserAccount === walletAddress)
          );

          const solAmount = solTransfer ? Math.abs(solTransfer.amount) / 1e9 : 0;

          if (solAmount > 0) {
            trades.push({
              wallet: walletAddress,
              tokenMint: transfer.mint,
              type: isBuy ? "buy" : "sell",
              amountSol: solAmount,
              amountToken: transfer.tokenAmount || 0,
              pricePerToken: solAmount / (transfer.tokenAmount || 1),
              timestamp: tx.timestamp * 1000,
              signature: tx.signature,
              dex: tx.source || "unknown",
            });
          }
        }
      }
    } catch (err) {
      log.warn(`Failed to fetch trades for ${shortenAddress(walletAddress)}: ${err}`);
    }

    return trades;
  }

  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.ping();
      }
    }, 30000);
  }

  private attemptReconnect(): void {
    if (!this.isRunning) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      log.error("Max reconnect attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts);
    log.info(`Reconnecting in ${delay / 1000}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => this.connect(), delay);
  }

  getStatus() {
    return {
      connected: this.ws?.readyState === WebSocket.OPEN,
      subscribedWallets: this.subscribedWallets.size,
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}
