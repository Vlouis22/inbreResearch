"""
SQLite-backed cache for literature API responses and normalized papers.

The cache is intentionally local and serverless. It avoids repeated API calls
without requiring PostgreSQL, vector databases, or any external service.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_cache_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / "literature.sqlite3"


class LiteratureCache:
    """Small SQLite cache used by source adapters and the pipeline."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._connection.close()

    def get_api_response(self, provider: str, cache_key: str) -> str | None:
        row = self._connection.execute(
            "SELECT payload FROM api_cache WHERE provider = ? AND cache_key = ?",
            (provider, cache_key),
        ).fetchone()
        return str(row["payload"]) if row else None

    def set_api_response(self, provider: str, cache_key: str, payload: str) -> None:
        self._connection.execute(
            """
            INSERT INTO api_cache(provider, cache_key, payload, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, cache_key)
            DO UPDATE SET payload = excluded.payload, created_at = excluded.created_at
            """,
            (provider, cache_key, payload, _now()),
        )
        self._connection.commit()

    def get_paper(self, paper_key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload FROM paper_cache WHERE paper_key = ?",
            (paper_key,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(str(row["payload"]))
        except json.JSONDecodeError:
            return None

    def set_paper(self, paper_key: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            """
            INSERT INTO paper_cache(paper_key, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(paper_key)
            DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
            """,
            (paper_key, json.dumps(payload, sort_keys=True), _now()),
        )
        self._connection.commit()

    def _ensure_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_cache (
                provider TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(provider, cache_key)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_cache (
                paper_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> "LiteratureCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
