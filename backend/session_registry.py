from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time


class SessionRegistry:
    """
    Tracks active sessions and their capability-token secrets in a small
    SQLite table (same file the LangGraph SqliteSaver checkpointer uses, by
    convention — see backend/main.py).

    The app is intentionally accountless and single-use (snap a photo, chat,
    leave), so there's no login/user model here. Instead, `thread_id` (a
    high-entropy client-generated UUID) is paired with a server-generated
    bearer secret handed back exactly once, when the session is created.
    Every later request for that thread_id must present the secret. This
    makes the thread_id alone insufficient to read or evict someone else's
    session, without adding accounts.

    `create()` is a single atomic INSERT (thread_id is the primary key), so
    "is this the first request for this thread_id" and "register it" happen
    as one DB operation — no separate exists-then-insert race if a client's
    first request is retried concurrently.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL is a database-file-level setting (persists across connections,
            # not per-connection) — set it here explicitly rather than relying on
            # the LangGraph SqliteSaver on the same file (backend/main.py) having
            # already enabled it first. Without WAL, two separate connections
            # writing to the same file can hit "database is locked" under
            # concurrent access.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "thread_id TEXT PRIMARY KEY, secret_hash TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.commit()

    @staticmethod
    def _hash(secret: str) -> str:
        return hashlib.sha256(secret.encode()).hexdigest()

    def create(self, thread_id: str) -> str | None:
        """
        Atomically register a brand-new session and return its plaintext
        secret. Returns None if thread_id is already registered — the caller
        must then verify() an existing secret instead.
        """
        secret = secrets.token_urlsafe(32)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO sessions (thread_id, secret_hash, created_at) VALUES (?, ?, ?)",
                    (thread_id, self._hash(secret), time.time()),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                return None
        return secret

    def verify(self, thread_id: str, secret: str | None) -> bool:
        if not secret:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT secret_hash FROM sessions WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        if row is None:
            return False
        return hmac.compare_digest(row[0], self._hash(secret))

    def exists(self, thread_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return row is not None

    def evict(self, thread_id: str) -> None:
        """Raise KeyError if thread_id is not registered."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(thread_id)
