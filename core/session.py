"""Session persistence via SQLite — stores conversation history across restarts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Session:
    id: int
    workspace: str
    created_at: str
    updated_at: str
    messages: list[dict]
    summary: str = ""


class SessionStore:
    """Manages conversation sessions in SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_workspace
                ON sessions(workspace, updated_at DESC)
            """)
            conn.commit()

    def create_session(self, workspace: str) -> Session:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (workspace, created_at, updated_at, messages) VALUES (?, ?, ?, ?)",
                (workspace, now, now, "[]"),
            )
            session_id = cursor.lastrowid
            conn.commit()
        return Session(
            id=session_id,
            workspace=workspace,
            created_at=now,
            updated_at=now,
            messages=[],
        )

    def load_session(self, session_id: int) -> Optional[Session]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, workspace, created_at, updated_at, messages, summary FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return Session(
            id=row[0],
            workspace=row[1],
            created_at=row[2],
            updated_at=row[3],
            messages=json.loads(row[4]),
            summary=row[5],
        )

    def save_session(self, session: Session) -> None:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, messages = ?, summary = ? WHERE id = ?",
                (now, json.dumps(session.messages), session.summary, session.id),
            )
            conn.commit()
        session.updated_at = now

    def list_sessions(self, workspace: str, limit: int = 5) -> list[Session]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, workspace, created_at, updated_at, messages, summary FROM sessions "
                "WHERE workspace = ? ORDER BY updated_at DESC LIMIT ?",
                (workspace, limit),
            ).fetchall()
        return [
            Session(
                id=row[0],
                workspace=row[1],
                created_at=row[2],
                updated_at=row[3],
                messages=json.loads(row[4]),
                summary=row[5],
            )
            for row in rows
        ]

    def get_last_session(self, workspace: str) -> Optional[Session]:
        sessions = self.list_sessions(workspace, limit=1)
        return sessions[0] if sessions else None

    def delete_session(self, session_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
