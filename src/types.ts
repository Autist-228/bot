export enum WalletTier {
  S = "S",
  A = "A",
  B = "B",
  C = "C",
}

export interface TrackedWallet {
  address: string;
  tier: WalletTier;
  winRate: number;
  avgRoi: number;
  totalTrades: number;
  profitableTrades: number;
  maxDrawdown: number;
  avgEntrySpeed: number;
  consistency: number;
  score: number;
  lastActive: number;
  addedAt: number;
  linkedWallets: string[];
  isDevWallet: boolean;
  blacklisted: boolean;
}

export interface TokenInfo {
  mint: string;
  symbol: string;
  name: string;
  decimals: number;
  lpBurned: boolean;
  mintDisabled: boolean;
  buyTax: number;
  sellTax: number;
  topHoldersPct: number;
  liquidity: number;
  age: number;
  isHoneypot: boolean;
  devAddress: string;
  devHistory: DevHistory;
  uniqueBuyers1h: number;
  buyToSellRatio: number;
  volumeUsd1h: number;
}

export interface DevHistory {
  address: string;
  previousTokens: PreviousToken[];
  hasRugPull: boolean;
  hasSuccessfulProject: boolean;
  bestMultiplier: number;
}

export interface PreviousToken {
  mint: string;
  name: string;
  maxMultiplier: number;
  wasRug: boolean;
  createdAt: number;
}

export interface WalletTrade {
  wallet: string;
  tokenMint: string;
  type: "buy" | "sell";
  amountSol: number;
  amountToken: number;
  pricePerToken: number;
  timestamp: number;
  signature: string;
  dex: string;
}

export interface RocketSignal {
  tokenMint: string;
  tokenInfo: TokenInfo;
  rocketScore: number;
  devScore: number;
  smartMoneyScore: number;
  twitterScore: number;
  tokenScore: number;
  onChainScore: number;
  buyingWallets: TrackedWallet[];
  twitterMentions: TwitterMention[];
  confidence: TradeConfidence;
  suggestedPositionPct: number;
  timestamp: number;
}

export enum TradeConfidence {
  ROCKET = "ROCKET",
  STRONG = "STRONG",
  NORMAL = "NORMAL",
  WEAK = "WEAK",
  SKIP = "SKIP",
}

export interface OpenPosition {
  id: string;
  tokenMint: string;
  tokenSymbol: string;
  entryPrice: number;
  currentPrice: number;
  entryAmountSol: number;
  remainingTokens: number;
  initialTokens: number;
  totalSoldSol: number;
  pnlPct: number;
  pnlSol: number;
  rocketScore: number;
  confidence: TradeConfidence;
  peakPrice: number;
  trailingStopActive: boolean;
  trailingStopPct: number;
  partialSells: PartialSell[];
  entryTimestamp: number;
  triggerWallet: string;
  status: PositionStatus;
}

export enum PositionStatus {
  ACTIVE = "ACTIVE",
  PARTIAL_EXIT = "PARTIAL_EXIT",
  TRAILING = "TRAILING",
  CLOSED = "CLOSED",
  EMERGENCY_EXIT = "EMERGENCY_EXIT",
}

export interface PartialSell {
  pctSold: number;
  priceSol: number;
  amountSol: number;
  pnlPct: number;
  timestamp: number;
  reason: string;
}

export interface ExitRule {
  triggerPct: number;
  sellPct: number;
  description: string;
}

export interface TrailingStopConfig {
  activateAtPct: number;
  stopPct: number;
}

export interface ExitStrategy {
  profitLadder: ExitRule[];
  stopLoss: ExitRule[];
  trailingStops: TrailingStopConfig[];
  maxHoldTime: number;
  flatlineTimeout: number;
  minLiquidity: number;
}

export interface TwitterMention {
  username: string;
  followers: number;
  text: string;
  timestamp: number;
  isInfluencer: boolean;
}

export interface BankState {
  totalSol: number;
  availableSol: number;
  lockedInPositions: number;
  dailyPnl: number;
  weeklyPnl: number;
  totalPnl: number;
  dailyLossLimit: number;
  isPaused: boolean;
  pauseUntil: number;
  tradeCount: number;
}

export interface TradeLog {
  id: string;
  tokenMint: string;
  tokenSymbol: string;
  action: "buy" | "sell" | "partial_sell";
  amountSol: number;
  price: number;
  pnlPct: number;
  pnlSol: number;
  rocketScore: number;
  triggerWallet: string;
  reason: string;
  timestamp: number;
  signature: string;
}

export interface CrawlerStats {
  walletsScanned: number;
  walletsTracked: number;
  walletsBlacklisted: number;
  lastScanTime: number;
  topWalletScore: number;
}

export interface HeliusTransaction {
  signature: string;
  type: string;
  timestamp: number;
  fee: number;
  feePayer: string;
  nativeTransfers: NativeTransfer[];
  tokenTransfers: TokenTransfer[];
  accountData: AccountData[];
  source: string;
  description: string;
}

export interface NativeTransfer {
  fromUserAccount: string;
  toUserAccount: string;
  amount: number;
}

export interface TokenTransfer {
  fromUserAccount: string;
  toUserAccount: string;
  fromTokenAccount: string;
  toTokenAccount: string;
  tokenAmount: number;
  mint: string;
  tokenStandard: string;
}

export interface AccountData {
  account: string;
  nativeBalanceChange: number;
  tokenBalanceChanges: TokenBalanceChange[];
}

export interface TokenBalanceChange {
  userAccount: string;
  tokenAccount: string;
  mint: string;
  rawTokenAmount: {
    tokenAmount: string;
    decimals: number;
  };
}
