import axios from "axios";
import { config } from "../config";
import { TwitterMention } from "../types";
import { createLogger } from "../utils/logger";

const log = createLogger("TwitterMonitor");

type MentionCallback = (mention: TwitterMention, tokenMint: string) => void;

const SOLANA_TOKEN_REGEX = /[1-9A-HJ-NP-Za-km-z]{32,44}/g;
const CASHTAG_REGEX = /\$([A-Za-z]{2,10})/g;

interface TwitterSearchResult {
  id: string;
  text: string;
  author: {
    userName: string;
    followerCount: number;
    isVerified: boolean;
  };
  createdAt: string;
}

export class TwitterMonitor {
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private onMentionCallbacks: MentionCallback[] = [];
  private seenTweetIds: Set<string> = new Set();
  private influencerAccounts: string[] = [];
  private searchQueries: string[] = [
    "solana memecoin 100x",
    "solana gem alpha",
    "$SOL pump",
    "just aped solana",
    "solana moonshot",
  ];

  onMention(callback: MentionCallback): void {
    this.onMentionCallbacks.push(callback);
  }

  setInfluencerAccounts(accounts: string[]): void {
    this.influencerAccounts = accounts;
    log.info(`Tracking ${accounts.length} influencer accounts`);
  }

  addSearchQuery(query: string): void {
    this.searchQueries.push(query);
  }

  async start(pollIntervalMs: number = 60_000): Promise<void> {
    log.info("Starting Twitter monitor...");

    await this.poll();

    this.pollInterval = setInterval(() => this.poll(), pollIntervalMs);
    log.info(`Twitter monitor running, poll interval: ${pollIntervalMs / 1000}s`);
  }

  stop(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    log.info("Twitter monitor stopped");
  }

  private async poll(): Promise<void> {
    try {
      for (const query of this.searchQueries) {
        await this.searchTweets(query);
        await new Promise((r) => setTimeout(r, 2000));
      }

      for (const account of this.influencerAccounts) {
        await this.checkInfluencerTweets(account);
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch (err) {
      log.error(`Twitter poll error: ${err}`);
    }
  }

  private async searchTweets(query: string): Promise<void> {
    try {
      const response = await axios.get(
        `${config.twitter.apiUrl}/tweet/advanced_search`,
        {
          params: {
            query,
            queryType: "Latest",
            cursor: "",
          },
          headers: {
            "X-API-Key": config.twitter.apiKey,
          },
          timeout: 10000,
        }
      );

      const tweets: TwitterSearchResult[] = response.data?.tweets || [];

      for (const tweet of tweets) {
        if (this.seenTweetIds.has(tweet.id)) continue;
        this.seenTweetIds.add(tweet.id);

        this.processTweet(tweet);
      }

      if (this.seenTweetIds.size > 10000) {
        const idsArray = Array.from(this.seenTweetIds);
        this.seenTweetIds = new Set(idsArray.slice(-5000));
      }
    } catch (err) {
      log.warn(`Twitter search failed for "${query}": ${err}`);
    }
  }

  private async checkInfluencerTweets(username: string): Promise<void> {
    try {
      const response = await axios.get(
        `${config.twitter.apiUrl}/user/last_tweets`,
        {
          params: {
            userName: username,
            cursor: "",
          },
          headers: {
            "X-API-Key": config.twitter.apiKey,
          },
          timeout: 10000,
        }
      );

      const tweets: TwitterSearchResult[] = response.data?.tweets || [];

      for (const tweet of tweets) {
        if (this.seenTweetIds.has(tweet.id)) continue;
        this.seenTweetIds.add(tweet.id);

        this.processTweet(tweet, true);
      }
    } catch (err) {
      log.warn(`Failed to check influencer ${username}: ${err}`);
    }
  }

  private processTweet(tweet: TwitterSearchResult, isInfluencer: boolean = false): void {
    const tokenAddresses = tweet.text.match(SOLANA_TOKEN_REGEX) || [];
    const cashtags = tweet.text.match(CASHTAG_REGEX) || [];

    if (tokenAddresses.length === 0 && cashtags.length === 0) return;

    const mention: TwitterMention = {
      username: tweet.author.userName,
      followers: tweet.author.followerCount,
      text: tweet.text,
      timestamp: new Date(tweet.createdAt).getTime(),
      isInfluencer: isInfluencer || tweet.author.followerCount > 50000,
    };

    for (const tokenMint of tokenAddresses) {
      if (tokenMint.length >= 32 && tokenMint.length <= 44) {
        log.info(
          `Twitter signal: @${mention.username} (${mention.followers} followers) mentioned token ${tokenMint.slice(0, 8)}...`
        );

        for (const callback of this.onMentionCallbacks) {
          try {
            callback(mention, tokenMint);
          } catch (err) {
            log.error(`Twitter mention callback error: ${err}`);
          }
        }
      }
    }
  }

  getTwitterScore(mentions: TwitterMention[]): number {
    if (mentions.length === 0) return 0;

    let score = 0;

    const influencerMentions = mentions.filter((m) => m.isInfluencer);
    score += influencerMentions.length * 10;

    const totalFollowers = mentions.reduce((sum, m) => sum + m.followers, 0);
    if (totalFollowers > 1_000_000) score += 15;
    else if (totalFollowers > 500_000) score += 10;
    else if (totalFollowers > 100_000) score += 5;

    if (mentions.length >= 5) score += 10;
    else if (mentions.length >= 3) score += 5;

    const now = Date.now();
    const recentMentions = mentions.filter((m) => now - m.timestamp < 30 * 60 * 1000);
    if (recentMentions.length >= 3) score += 10;

    return Math.min(25, score);
  }
}
