export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  TRADE = 4,
}

const LOG_COLORS: Record<LogLevel, string> = {
  [LogLevel.DEBUG]: "\x1b[90m",
  [LogLevel.INFO]: "\x1b[36m",
  [LogLevel.WARN]: "\x1b[33m",
  [LogLevel.ERROR]: "\x1b[31m",
  [LogLevel.TRADE]: "\x1b[32m",
};

const RESET = "\x1b[0m";

let currentLogLevel = LogLevel.INFO;

export function setLogLevel(level: LogLevel): void {
  currentLogLevel = level;
}

function formatTimestamp(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 23);
}

function log(level: LogLevel, module: string, message: string, data?: unknown): void {
  if (level < currentLogLevel) return;

  const color = LOG_COLORS[level];
  const levelName = LogLevel[level].padEnd(5);
  const prefix = `${color}[${formatTimestamp()}] [${levelName}] [${module}]${RESET}`;

  if (data !== undefined) {
    console.log(`${prefix} ${message}`, typeof data === "object" ? JSON.stringify(data, null, 2) : data);
  } else {
    console.log(`${prefix} ${message}`);
  }
}

export function createLogger(module: string) {
  return {
    debug: (msg: string, data?: unknown) => log(LogLevel.DEBUG, module, msg, data),
    info: (msg: string, data?: unknown) => log(LogLevel.INFO, module, msg, data),
    warn: (msg: string, data?: unknown) => log(LogLevel.WARN, module, msg, data),
    error: (msg: string, data?: unknown) => log(LogLevel.ERROR, module, msg, data),
    trade: (msg: string, data?: unknown) => log(LogLevel.TRADE, module, msg, data),
  };
}
