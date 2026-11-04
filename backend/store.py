"""SQLite store for SOPProof users, sessions, inspections and invitations.

Stdlib sqlite3 only. The database is an ownership/auth index over the
existing file artifacts — report and upload files stay exactly where the
pipelines already write them.
"""
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = PROJECT_ROOT / "data" / "sopproof.db"


def db_path() -> Path:
    raw = os.environ.get("SOPPROOF_DB_PATH", "").strip()
    return Path(raw) if raw else _DEFAULT_DB


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    owner_id INTEGER NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS guest_sessions (
    token_hash TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    inspections_used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS guest_inspections (
    inspection_id TEXT PRIMARY KEY,
    guest_token_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    inspection_id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS invitations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT NOT NULL UNIQUE,
    email TEXT,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    used_at REAL,
    revoked_at REAL
);
"""


def init_db() -> None:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA)
        # Migration for databases created before guest inspection credits:
        # existing guest_sessions rows gain the counter at the default 0.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(guest_sessions)")}
        if "inspections_used" not in cols:
            conn.execute(
                "ALTER TABLE guest_sessions ADD COLUMN"
                " inspections_used INTEGER NOT NULL DEFAULT 0"
            )


@contextmanager
def connect():
    init_db()
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- users ----------

def create_user(email: str, password_hash: str, is_admin: bool = False):
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, is_admin, created_at)"
            " VALUES (?, ?, ?, ?)",
            (email, password_hash, 1 if is_admin else 0, time.time()),
        )
        return cur.lastrowid


def delete_user(user_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def get_user_by_email(email: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def count_users() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def first_admin():
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


# ---------- sessions ----------

def create_session(token_hash: str, user_id: int, expires_at: float) -> None:
    now = time.time()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen)"
            " VALUES (?, ?, ?, ?, ?)",
            (token_hash, user_id, now, expires_at, now),
        )


def get_session(token_hash: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        return dict(row) if row else None


def touch_session(token_hash: str, expires_at: float) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET last_seen = ?, expires_at = ? WHERE token_hash = ?",
            (time.time(), expires_at, token_hash),
        )


def delete_session(token_hash: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def cleanup_expired_sessions() -> int:
    with connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        cur2 = conn.execute(
            "DELETE FROM guest_sessions WHERE expires_at < ?", (time.time(),))
        return cur.rowcount + cur2.rowcount


# ---------- guest sessions (no user account is created) ----------

def create_guest_session(token_hash: str, expires_at: float) -> None:
    now = time.time()
    with connect() as conn:
        conn.execute(
            "INSERT INTO guest_sessions (token_hash, created_at, expires_at, last_seen)"
            " VALUES (?, ?, ?, ?)",
            (token_hash, now, expires_at, now),
        )


def get_guest_session(token_hash: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM guest_sessions WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        return dict(row) if row else None


def delete_guest_session(token_hash: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM guest_sessions WHERE token_hash = ?", (token_hash,))


# ---------- guest inspection credits + ownership ----------

GUEST_INSPECTION_LIMIT = 2


def guest_credits_used(token_hash: str):
    """Credits consumed by a guest session, or None if it does not exist."""
    with connect() as conn:
        row = conn.execute(
            "SELECT inspections_used FROM guest_sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        return row[0] if row else None


def consume_guest_credit(token_hash: str) -> bool:
    """Atomically consume one guest inspection credit.

    A single conditional UPDATE under BEGIN IMMEDIATE: two simultaneous
    requests cannot both take the final credit. Returns False when the
    session is missing, expired, or already at the limit.
    """
    init_db()
    conn = sqlite3.connect(str(db_path()), timeout=30)
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "UPDATE guest_sessions SET inspections_used = inspections_used + 1,"
            " last_seen = ? WHERE token_hash = ? AND inspections_used < ?"
            " AND expires_at > ?",
            (time.time(), token_hash, GUEST_INSPECTION_LIMIT, time.time()),
        )
        if cur.rowcount == 1:
            conn.execute("COMMIT")
            return True
        conn.execute("ROLLBACK")
        return False
    finally:
        conn.close()


def record_guest_inspection(inspection_id: str, guest_token_hash: str, kind: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO guest_inspections"
            " (inspection_id, guest_token_hash, kind, created_at)"
            " VALUES (?, ?, ?, ?)",
            (inspection_id, guest_token_hash, kind, time.time()),
        )


def guest_session_of(inspection_id: str):
    """Token hash of the guest session that owns an inspection, or None."""
    with connect() as conn:
        row = conn.execute(
            "SELECT guest_token_hash FROM guest_inspections"
            " WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        return row[0] if row else None


def create_job(inspection_id: str, owner_key: str, kind: str) -> None:
    now = time.time()
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (inspection_id, owner_key, kind, status,"
            " created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (inspection_id, owner_key, kind, now, now),
        )


def update_job(inspection_id: str, status: str, result_json: str = None,
               error: str = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, result_json = ?, error = ?,"
            " updated_at = ? WHERE inspection_id = ?",
            (status, result_json, error, time.time(), inspection_id),
        )


def get_job(inspection_id: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE inspection_id = ?", (inspection_id,)
        ).fetchone()
        return dict(row) if row else None


def fail_orphaned_jobs() -> int:
    """Mark jobs left pending/processing by a previous process as failed."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed',"
            " error = 'Inspection was interrupted by a server restart.',"
            " updated_at = ? WHERE status IN ('pending', 'processing')",
            (time.time(),),
        )
        return cur.rowcount


def guest_inspection_ids(guest_token_hash: str) -> set:
    with connect() as conn:
        rows = conn.execute(
            "SELECT inspection_id FROM guest_inspections"
            " WHERE guest_token_hash = ?",
            (guest_token_hash,),
        ).fetchall()
        return {r[0] for r in rows}


# ---------- inspections (ownership registry) ----------

def record_inspection(inspection_id: str, owner_id: int, kind: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO inspections (inspection_id, owner_id, kind, created_at)"
            " VALUES (?, ?, ?, ?)",
            (inspection_id, owner_id, kind, time.time()),
        )


def owner_of(inspection_id: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT owner_id FROM inspections WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        return row[0] if row else None


def inspection_ids_for(owner_id: int) -> set:
    with connect() as conn:
        rows = conn.execute(
            "SELECT inspection_id FROM inspections WHERE owner_id = ?", (owner_id,)
        ).fetchall()
        return {r[0] for r in rows}


# ---------- invitations ----------

def create_invitation(token_hash: str, email, created_by: int, expires_at: float) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO invitations (token_hash, email, created_by, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (token_hash, email, created_by, time.time(), expires_at),
        )
        return cur.lastrowid


def get_invitation_by_hash(token_hash: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM invitations WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        return dict(row) if row else None


def mark_invitation_used(invitation_id: int) -> bool:
    """Atomically claim an invitation; False if already used/revoked/expired."""
    now = time.time()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE invitations SET used_at = ? WHERE id = ?"
            " AND used_at IS NULL AND revoked_at IS NULL AND expires_at > ?",
            (now, invitation_id, now),
        )
        return cur.rowcount == 1


def revoke_invitation(invitation_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE invitations SET revoked_at = ? WHERE id = ? AND used_at IS NULL",
            (time.time(), invitation_id),
        )
        return cur.rowcount == 1


def list_invitations():
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, email, created_by, created_at, expires_at, used_at, revoked_at"
            " FROM invitations ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]
