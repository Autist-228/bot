import axios from "axios";
import { VersionedTransaction, Connection } from "@solana/web3.js";
import { config } from "../config";
import { getConnection, getWalletKeypair, getWalletPublicKey } from "./solana";
import { createLogger } from "./logger";

const log = createLogger("Jupiter");

const JUPITER_API = "https://quote-api.jup.ag/v6";
const SOL_MINT = "So11111111111111111111111111111111111111112";

interface JupiterQuote {
  inputMint: string;
  outputMint: string;
  inAmount: string;
  outAmount: string;
  priceImpactPct: string;
  routePlan: unknown[];
}

interface SwapResult {
  signature: string;
  inputAmount: number;
  outputAmount: number;
  priceImpact: number;
}

export async function getQuote(
  inputMint: string,
  outputMint: string,
  amount: number,
  slippageBps: number = config.trading.slippageBps
): Promise<JupiterQuote> {
  const response = await axios.get(`${JUPITER_API}/quote`, {
    params: {
      inputMint,
      outputMint,
      amount: Math.floor(amount).toString(),
      slippageBps,
      onlyDirectRoutes: false,
    },
    timeout: 10000,
  });

  return response.data;
}

export async function executeSwap(
  inputMint: string,
  outputMint: string,
  amount: number,
  slippageBps: number = config.trading.slippageBps
): Promise<SwapResult> {
  const quote = await getQuote(inputMint, outputMint, amount, slippageBps);

  log.info(
    `Quote: ${quote.inAmount} ${inputMint.slice(0, 6)} -> ${quote.outAmount} ${outputMint.slice(0, 6)}, impact: ${quote.priceImpactPct}%`
  );

  const walletPubkey = getWalletPublicKey();

  const swapResponse = await axios.post(
    `${JUPITER_API}/swap`,
    {
      quoteResponse: quote,
      userPublicKey: walletPubkey.toBase58(),
      wrapAndUnwrapSol: true,
      prioritizationFeeLamports: config.trading.priorityFeeLamports,
      dynamicComputeUnitLimit: true,
    },
    { timeout: 15000 }
  );

  const swapTransaction = swapResponse.data.swapTransaction;
  const transactionBuf = Buffer.from(swapTransaction, "base64");
  const tx = VersionedTransaction.deserialize(transactionBuf);

  const keypair = getWalletKeypair();
  tx.sign([keypair]);

  const conn = getConnection();
  const signature = await sendAndConfirmJupiterTx(conn, tx);

  const result: SwapResult = {
    signature,
    inputAmount: parseInt(quote.inAmount),
    outputAmount: parseInt(quote.outAmount),
    priceImpact: parseFloat(quote.priceImpactPct),
  };

  log.trade(`Swap executed: ${signature}`);
  return result;
}

async function sendAndConfirmJupiterTx(
  conn: Connection,
  tx: VersionedTransaction
): Promise<string> {
  const rawTx = tx.serialize();
  const latestBlockhash = await conn.getLatestBlockhash("confirmed");

  let retries = 3;
  while (retries > 0) {
    try {
      const signature = await conn.sendRawTransaction(rawTx, {
        skipPreflight: false,
        maxRetries: 2,
      });

      await conn.confirmTransaction(
        {
          signature,
          blockhash: latestBlockhash.blockhash,
          lastValidBlockHeight: latestBlockhash.lastValidBlockHeight,
        },
        "confirmed"
      );

      return signature;
    } catch (err) {
      retries--;
      if (retries === 0) throw err;
      log.warn(`Jupiter tx retry, ${retries} attempts left`);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  throw new Error("Jupiter swap failed after retries");
}

export async function buySol(
  tokenMint: string,
  amountSol: number,
  slippageBps?: number
): Promise<SwapResult> {
  const amountLamports = Math.floor(amountSol * 1e9);
  log.trade(`Buying ${tokenMint.slice(0, 8)}... for ${amountSol} SOL`);
  return executeSwap(SOL_MINT, tokenMint, amountLamports, slippageBps);
}

export async function sellToken(
  tokenMint: string,
  tokenAmount: number,
  slippageBps?: number
): Promise<SwapResult> {
  log.trade(`Selling ${tokenAmount} of ${tokenMint.slice(0, 8)}...`);
  return executeSwap(tokenMint, SOL_MINT, tokenAmount, slippageBps);
}

export async function getTokenPrice(tokenMint: string): Promise<number> {
  try {
    const response = await axios.get(
      `https://price.jup.ag/v6/price?ids=${tokenMint}`,
      { timeout: 5000 }
    );
    const data = response.data?.data?.[tokenMint];
    return data?.price || 0;
  } catch {
    return 0;
  }
}

export async function simulateSell(
  tokenMint: string,
  amount: number
): Promise<{ canSell: boolean; estimatedOutput: number; priceImpact: number }> {
  try {
    const quote = await getQuote(tokenMint, SOL_MINT, amount, 1000);
    return {
      canSell: true,
      estimatedOutput: parseInt(quote.outAmount) / 1e9,
      priceImpact: parseFloat(quote.priceImpactPct),
    };
  } catch {
    return { canSell: false, estimatedOutput: 0, priceImpact: 100 };
  }
}
