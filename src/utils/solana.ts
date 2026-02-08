import {
  Connection,
  Keypair,
  PublicKey,
  TransactionInstruction,
  TransactionMessage,
  VersionedTransaction,
  ComputeBudgetProgram,
  SendTransactionError,
} from "@solana/web3.js";
import bs58 from "bs58";
import { config } from "../config";
import { createLogger } from "./logger";

const log = createLogger("Solana");

let connection: Connection | null = null;
let walletKeypair: Keypair | null = null;

export function getConnection(): Connection {
  if (!connection) {
    connection = new Connection(config.helius.rpcUrl, {
      commitment: "confirmed",
      wsEndpoint: config.helius.wsUrl,
    });
    log.info(`Connected to Solana RPC: ${config.helius.rpcUrl.slice(0, 40)}...`);
  }
  return connection;
}

export function getWalletKeypair(): Keypair {
  if (!walletKeypair) {
    if (!config.wallet.privateKey) {
      throw new Error("WALLET_PRIVATE_KEY not set in .env");
    }
    walletKeypair = Keypair.fromSecretKey(bs58.decode(config.wallet.privateKey));
    log.info(`Wallet loaded: ${walletKeypair.publicKey.toBase58().slice(0, 8)}...`);
  }
  return walletKeypair;
}

export function getWalletPublicKey(): PublicKey {
  return getWalletKeypair().publicKey;
}

export async function getBalanceSol(): Promise<number> {
  const conn = getConnection();
  const balance = await conn.getBalance(getWalletPublicKey());
  return balance / 1e9;
}

export async function sendTransaction(
  instructions: TransactionInstruction[],
  priorityFeeLamports: number = config.trading.priorityFeeLamports
): Promise<string> {
  const conn = getConnection();
  const keypair = getWalletKeypair();

  const computeBudgetIx = ComputeBudgetProgram.setComputeUnitPrice({
    microLamports: priorityFeeLamports,
  });

  const allInstructions = [computeBudgetIx, ...instructions];

  const latestBlockhash = await conn.getLatestBlockhash("confirmed");

  const messageV0 = new TransactionMessage({
    payerKey: keypair.publicKey,
    recentBlockhash: latestBlockhash.blockhash,
    instructions: allInstructions,
  }).compileToV0Message();

  const tx = new VersionedTransaction(messageV0);
  tx.sign([keypair]);

  let retries = 3;
  while (retries > 0) {
    try {
      const signature = await conn.sendRawTransaction(tx.serialize(), {
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

      log.trade(`Transaction confirmed: ${signature}`);
      return signature;
    } catch (err) {
      retries--;
      if (retries === 0) {
        const message = err instanceof SendTransactionError ? err.message : String(err);
        log.error(`Transaction failed after retries: ${message}`);
        throw err;
      }
      log.warn(`Transaction retry, ${retries} attempts left`);
      await sleep(1000);
    }
  }

  throw new Error("Transaction failed after all retries");
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function lamportsToSol(lamports: number): number {
  return lamports / 1e9;
}

export function solToLamports(sol: number): number {
  return Math.floor(sol * 1e9);
}

export function shortenAddress(address: string, chars = 4): string {
  return `${address.slice(0, chars)}...${address.slice(-chars)}`;
}
