import axios from "axios";
import { config } from "../config";
import { TokenInfo, DevHistory } from "../types";
import { createLogger } from "../utils/logger";
import { shortenAddress } from "../utils/solana";
import { simulateSell } from "../utils/jupiter";

const log = createLogger("TokenAnalyzer");

export class TokenAnalyzer {
  private tokenCache: Map<string, { info: TokenInfo; timestamp: number }> = new Map();
  private cacheTtlMs = 5 * 60 * 1000;

  async analyzeToken(tokenMint: string): Promise<TokenInfo | null> {
    const cached = this.tokenCache.get(tokenMint);
    if (cached && Date.now() - cached.timestamp < this.cacheTtlMs) {
      return cached.info;
    }

    log.info(`Analyzing token: ${shortenAddress(tokenMint)}`);

    try {
      const tokenInfo = await this.fetchTokenData(tokenMint);
      if (!tokenInfo) return null;

      const honeypotCheck = await this.checkHoneypot(tokenMint);
      tokenInfo.isHoneypot = honeypotCheck;

      this.tokenCache.set(tokenMint, { info: tokenInfo, timestamp: Date.now() });
      return tokenInfo;
    } catch (err) {
      log.error(`Token analysis failed for ${shortenAddress(tokenMint)}: ${err}`);
      return null;
    }
  }

  private async fetchTokenData(tokenMint: string): Promise<TokenInfo | null> {
    try {
      const response = await axios.post(
        config.codex.apiUrl,
        {
          query: `{
            token(input: { address: "${tokenMint}", networkId: 1399811149 }) {
              info {
                name
                symbol
                decimals
              }
              explorerData {
                blueCheckmark
              }
            }
            filterPairs(
              filters: { tokenAddress: "${tokenMint}", network: [1399811149] }
              limit: 1
            ) {
              results {
                liquidity
                volumeUSD24
                priceUSD
                pair { address }
                token0 { address name symbol }
                token1 { address name symbol }
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

      const tokenData = response.data?.data?.token;
      const pairData = response.data?.data?.filterPairs?.results?.[0];

      if (!tokenData?.info) return null;

      const tokenInfo: TokenInfo = {
        mint: tokenMint,
        symbol: tokenData.info.symbol || "UNKNOWN",
        name: tokenData.info.name || "Unknown Token",
        decimals: tokenData.info.decimals || 9,
        lpBurned: false,
        mintDisabled: false,
        buyTax: 0,
        sellTax: 0,
        topHoldersPct: 0,
        liquidity: pairData?.liquidity || 0,
        age: 0,
        isHoneypot: false,
        devAddress: "",
        devHistory: {
          address: "",
          previousTokens: [],
          hasRugPull: false,
          hasSuccessfulProject: false,
          bestMultiplier: 0,
        },
        uniqueBuyers1h: 0,
        buyToSellRatio: 0,
        volumeUsd1h: pairData?.volumeUSD24 ? pairData.volumeUSD24 / 24 : 0,
      };

      return tokenInfo;
    } catch (err) {
      log.warn(`Codex token fetch failed for ${shortenAddress(tokenMint)}: ${err}`);
      return null;
    }
  }

  private async checkHoneypot(tokenMint: string): Promise<boolean> {
    try {
      const result = await simulateSell(tokenMint, 1000000);
      if (!result.canSell) {
        log.warn(`Token ${shortenAddress(tokenMint)} is a honeypot (cannot sell)`);
        return true;
      }
      if (result.priceImpact > 50) {
        log.warn(`Token ${shortenAddress(tokenMint)} has extreme sell impact: ${result.priceImpact}%`);
        return true;
      }
      return false;
    } catch {
      return true;
    }
  }

  getTokenSafetyScore(tokenInfo: TokenInfo): number {
    let score = 50;

    if (tokenInfo.isHoneypot) return 0;

    if (tokenInfo.liquidity >= 50000) score += 15;
    else if (tokenInfo.liquidity >= 20000) score += 10;
    else if (tokenInfo.liquidity >= 5000) score += 5;
    else score -= 20;

    if (tokenInfo.lpBurned) score += 10;
    if (tokenInfo.mintDisabled) score += 10;

    if (tokenInfo.buyTax > 10 || tokenInfo.sellTax > 10) score -= 30;
    else if (tokenInfo.buyTax > 5 || tokenInfo.sellTax > 5) score -= 15;

    if (tokenInfo.topHoldersPct > 50) score -= 25;
    else if (tokenInfo.topHoldersPct > 30) score -= 10;

    if (tokenInfo.devHistory.hasRugPull) score -= 30;
    if (tokenInfo.devHistory.hasSuccessfulProject) score += 15;

    if (tokenInfo.uniqueBuyers1h > 100) score += 10;
    else if (tokenInfo.uniqueBuyers1h > 50) score += 5;

    if (tokenInfo.buyToSellRatio > 2) score += 10;
    else if (tokenInfo.buyToSellRatio > 1.5) score += 5;
    else if (tokenInfo.buyToSellRatio < 0.5) score -= 15;

    return Math.max(0, Math.min(100, score));
  }

  shouldSkipToken(tokenInfo: TokenInfo): { skip: boolean; reason: string } {
    if (tokenInfo.isHoneypot) return { skip: true, reason: "Honeypot detected" };
    if (tokenInfo.liquidity < 1000) return { skip: true, reason: "Liquidity too low" };
    if (tokenInfo.buyTax > 15) return { skip: true, reason: `Buy tax too high: ${tokenInfo.buyTax}%` };
    if (tokenInfo.sellTax > 15) return { skip: true, reason: `Sell tax too high: ${tokenInfo.sellTax}%` };
    if (tokenInfo.topHoldersPct > 70) return { skip: true, reason: "Top holders own >70%" };
    if (tokenInfo.devHistory.hasRugPull) return { skip: true, reason: "Dev has rug pull history" };

    return { skip: false, reason: "" };
  }

  clearCache(): void {
    const now = Date.now();
    for (const [mint, cached] of this.tokenCache) {
      if (now - cached.timestamp > this.cacheTtlMs * 2) {
        this.tokenCache.delete(mint);
      }
    }
  }
}
