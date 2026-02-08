import aiosqlite
import os
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT NOT NULL,
                token_symbol TEXT,
                token_name TEXT,
                network_id INTEGER,
                price_at_signal REAL,
                liquidity_usd REAL,
                volume_24h REAL,
                score INTEGER,
                signal_details TEXT,
                created_at INTEGER NOT NULL,
                price_max REAL,
                price_max_at INTEGER,
                price_after_1h REAL,
                price_after_6h REAL,
                price_after_24h REAL,
                result_pct REAL,
                status TEXT DEFAULT 'active'
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tracked_tokens (
                token_address TEXT PRIMARY KEY,
                signal_id INTEGER,
                price_at_signal REAL,
                price_current REAL,
                price_max REAL,
                last_updated INTEGER,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        """)
        await db.commit()


async def save_signal(token_address: str, token_symbol: str, token_name: str,
                      network_id: int, price: float, liquidity: float,
                      volume: float, score: int, details: str) -> int:
    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO signals
               (token_address, token_symbol, token_name, network_id,
                price_at_signal, liquidity_usd, volume_24h, score,
                signal_details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (token_address, token_symbol, token_name, network_id,
             price, liquidity, volume, score, details, now)
        )
        signal_id = cursor.lastrowid
        await db.execute(
            """INSERT OR REPLACE INTO tracked_tokens
               (token_address, signal_id, price_at_signal, price_current,
                price_max, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (token_address, signal_id, price, price, price, now)
        )
        await db.commit()
        return signal_id


async def update_tracked_price(token_address: str, current_price: float):
    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        row = await db.execute_fetchall(
            "SELECT price_max FROM tracked_tokens WHERE token_address = ?",
            (token_address,)
        )
        if not row:
            return
        old_max = row[0][0] or 0.0
        new_max = max(old_max, current_price)
        await db.execute(
            """UPDATE tracked_tokens
               SET price_current = ?, price_max = ?, last_updated = ?
               WHERE token_address = ?""",
            (current_price, new_max, now, token_address)
        )
        if new_max > old_max:
            await db.execute(
                """UPDATE signals SET price_max = ?, price_max_at = ?
                   WHERE id = (SELECT signal_id FROM tracked_tokens
                               WHERE token_address = ?)""",
                (new_max, now, token_address)
            )
        await db.commit()


async def get_active_tracked_tokens() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """SELECT t.token_address, t.signal_id, t.price_at_signal,
                      t.price_current, t.price_max, s.token_symbol,
                      s.token_name, s.score, s.created_at
               FROM tracked_tokens t
               JOIN signals s ON s.id = t.signal_id
               WHERE s.status = 'active'
               ORDER BY s.created_at DESC"""
        )
        return [dict(r) for r in rows]


async def get_signal_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        total = await db.execute_fetchall("SELECT COUNT(*) FROM signals")
        total_count = total[0][0]

        profitable = await db.execute_fetchall(
            "SELECT COUNT(*) FROM signals WHERE price_max > price_at_signal * 1.1"
        )
        profitable_count = profitable[0][0]

        big_wins = await db.execute_fetchall(
            "SELECT COUNT(*) FROM signals WHERE price_max > price_at_signal * 2.0"
        )
        big_win_count = big_wins[0][0]

        rugs = await db.execute_fetchall(
            """SELECT COUNT(*) FROM tracked_tokens
               WHERE price_current < price_at_signal * 0.5"""
        )
        rug_count = rugs[0][0]

        return {
            "total": total_count,
            "profitable_10pct": profitable_count,
            "doubled": big_win_count,
            "rugged": rug_count,
            "win_rate": round(profitable_count / total_count * 100, 1) if total_count > 0 else 0,
        }


async def mark_signal_inactive(token_address: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE signals SET status = 'inactive'
               WHERE id = (SELECT signal_id FROM tracked_tokens
                           WHERE token_address = ?)""",
            (token_address,)
        )
        await db.execute(
            "DELETE FROM tracked_tokens WHERE token_address = ?",
            (token_address,)
        )
        await db.commit()


async def already_signaled(token_address: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall(
            "SELECT 1 FROM signals WHERE token_address = ? AND created_at > ? LIMIT 1",
            (token_address, int(time.time()) - 3600 * 6)
        )
        return len(rows) > 0
