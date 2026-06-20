from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .key import CacheKey


@dataclass
class CachedKernel:
    cache_key: CacheKey
    params: dict[str, object]
    kernel_code: str
    benchmark_json: dict[str, object]
    created_at: datetime


class KernelRepository:
    def __init__(self, path: str | Path = "~/.forge/cache.db") -> None:
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS kernels (
                cache_key_hash  TEXT PRIMARY KEY,
                cache_key_json  TEXT NOT NULL,
                params_json     TEXT NOT NULL,
                kernel_code     TEXT NOT NULL,
                benchmark_json  TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key_hash  TEXT NOT NULL,
                params_json     TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                benchmark_json  TEXT,
                is_best         INTEGER NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_key
                ON experiments(cache_key_hash);
        """)
        self.conn.commit()

    def get(self, key: CacheKey) -> CachedKernel | None:
        row = self.conn.execute(
            "SELECT cache_key_json, params_json, kernel_code, benchmark_json, created_at "
            "FROM kernels WHERE cache_key_hash = ?",
            (key.digest(),),
        ).fetchone()
        if row is None:
            return None
        return CachedKernel(
            cache_key=key,
            params=json.loads(row[1]),
            kernel_code=row[2],
            benchmark_json=json.loads(row[3]),
            created_at=datetime.fromisoformat(row[4]),
        )

    def put(self, key: CacheKey, kernel: CachedKernel) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO kernels VALUES (?, ?, ?, ?, ?, ?)",
            (
                key.digest(),
                json.dumps(key.__dict__, default=list),
                json.dumps(kernel.params),
                kernel.kernel_code,
                json.dumps(kernel.benchmark_json),
                datetime.now(UTC).isoformat(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
