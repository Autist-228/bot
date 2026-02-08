import axios from "axios";
import { config } from "../config";
import { DevHistory, PreviousToken, HeliusTransaction } from "../types";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";

const log = createLogger("DevTracker");

export class DevTracker {
  private devCache: Map<string, DevHistory> = new Map();
  private linkedWalletsCache: Map<string, string[]> = new Map();
  private rugDevs: Set<string> = new Set();

  async analyzeDevWallet(devAddress: string): Promise<DevHistory> {
    const cached = this.devCache.get(devAddress);
    if (cached && Date.now() - (cached as DevHistory & { cachedAt?: number }).cachedAt! < 3600_000) {
      return cached;
    }

    log.info(`Analyzing dev wallet: ${shortenAddress(devAddress)}`);

    const history = await this.getDevTokenHistory(devAddress);
    const devHistory: DevHistory = {
      address: devAddress,
      previousTokens: history,
      hasRugPull: history.some((t) => t.wasRug),
      hasSuccessfulProject: history.some((t) => t.maxMultiplier >= 5),
      bestMultiplier: Math.max(0, ...history.map((t) => t.maxMultiplier)),
    };

    if (devHistory.hasRugPull) {
      this.rugDevs.add(devAddress);
      log.warn(`Dev ${shortenAddress(devAddress)} has rug pull history!`);
    }

    this.devCache.set(devAddress, devHistory);
    return devHistory;
  }

  private async getDevTokenHistory(devAddress: string): Promise<PreviousToken[]> {
    const tokens: PreviousToken[] = [];

    try {
      const response = await axios.get(
        `${config.helius.apiUrl}/addresses/${devAddress}/transactions/?api-key=${config.helius.apiKey}`,
        {
          params: { limit: 100, type: "TOKEN_MINT" },
          timeout: 15000,
        }
      );

      const transactions: HeliusTransaction[] = response.data || [];

      for (const tx of transactions) {
        for (const transfer of tx.tokenTransfers || []) {
          if (transfer.fromUserAccount === devAddress && transfer.mint) {
            tokens.push({
              mint: transfer.mint,
              name: "",
              maxMultiplier: await this.estimateTokenMultiplier(transfer.mint),
              wasRug: await this.checkIfRug(transfer.mint),
              createdAt: tx.timestamp * 1000,
            });
          }
        }
      }
    } catch (err) {
      log.warn(`Failed to get dev history for ${shortenAddress(devAddress)}: ${err}`);
    }

    try {
      const response = await axios.post(
        config.codex.apiUrl,
        {
          query: `{
            filterTokens(
              filters: { creatorAddress: "${devAddress}", network: [1399811149] }
              limit: 20
            ) {
              results {
                token {
                  address
                  name
                  symbol
                  createdAt
                }
                priceUSD
                liquidity
                marketCap
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

      const results = response.data?.data?.filterTokens?.results || [];
      for (const result of results) {
        const existing = tokens.find((t) => t.mint === result.token.address);
        if (!existing) {
          tokens.push({
            mint: result.token.address,
            name: result.token.name || result.token.symbol || "",
            maxMultiplier: 0,
            wasRug: (result.liquidity || 0) < 100,
            createdAt: result.token.createdAt ? new Date(result.token.createdAt).getTime() : 0,
          });
        }
      }
    } catch (err) {
      log.warn(`Codex dev token query failed: ${err}`);
    }

    return tokens;
  }

  async findLinkedWallets(devAddress: string): Promise<string[]> {
    const cached = this.linkedWalletsCache.get(devAddress);
    if (cached) return cached;

    const linked: Set<string> = new Set();

    try {
      const response = await axios.get(
        `${config.helius.apiUrl}/addresses/${devAddress}/transactions/?api-key=${config.helius.apiKey}`,
        {
          params: { limit: 50 },
          timeout: 15000,
        }
      );

      const transactions: HeliusTransaction[] = response.data || [];

      for (const tx of transactions) {
        for (const transfer of tx.nativeTransfers || []) {
          if (transfer.fromUserAccount === devAddress && transfer.amount > 0.01 * 1e9) {
            linked.add(transfer.toUserAccount);
          }
          if (transfer.toUserAccount === devAddress && transfer.amount > 0.01 * 1e9) {
            linked.add(transfer.fromUserAccount);
          }
        }
      }
    } catch (err) {
      log.warn(`Failed to find linked wallets for ${shortenAddress(devAddress)}: ${err}`);
    }

    linked.delete(devAddress);
    const linkedArray = Array.from(linked);
    this.linkedWalletsCache.set(devAddress, linkedArray);

    log.info(`Found ${linkedArray.length} linked wallets for dev ${shortenAddress(devAddress)}`);
    return linkedArray;
  }

  private async estimateTokenMultiplier(tokenMint: string): Promise<number> {
    try {
      const response = await axios.post(
        config.codex.apiUrl,
        {
          query: `{
            token(input: { address: "${tokenMint}", networkId: 1399811149 }) {
              info { name symbol }
              explorerData { blueCheckmark }
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

      return response.data?.data?.token ? 1 : 0;
    } catch {
      return 0;
    }
  }

  private async checkIfRug(tokenMint: string): Promise<boolean> {
    try {
      const response = await axios.post(
        config.codex.apiUrl,
        {
          query: `{
            token(input: { address: "${tokenMint}", networkId: 1399811149 }) {
              info { name }
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

      return !response.data?.data?.token;
    } catch {
      return true;
    }
  }

  isKnownRugDev(address: string): boolean {
    return this.rugDevs.has(address);
  }

  getDevScore(devHistory: DevHistory): number {
    let score = 0;

    if (devHistory.hasRugPull) return -50;

    if (devHistory.hasSuccessfulProject) score += 25;

    if (devHistory.bestMultiplier >= 10) score += 15;
    else if (devHistory.bestMultiplier >= 5) score += 10;
    else if (devHistory.bestMultiplier >= 2) score += 5;

    if (devHistory.previousTokens.length === 0) score += 0;
    else if (devHistory.previousTokens.length <= 3) score += 5;

    const rugRate = devHistory.previousTokens.length > 0
      ? devHistory.previousTokens.filter((t) => t.wasRug).length / devHistory.previousTokens.length
      : 0;

    if (rugRate > 0.5) score -= 20;

    return Math.max(-50, Math.min(30, score));
  }
}
