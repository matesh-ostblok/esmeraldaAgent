import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration from environment (agent loads .env beforehand)
USE_AGENTS_SESSION = os.environ.get("USE_AGENTS_SESSION", "1").strip() not in ("", "0", "false", "False")
AGENTS_SQLITE_PATH = os.environ.get(
    "AGENTS_SQLITE_PATH",
    str((Path(__file__).resolve().parent.parent / "agents_memory.sqlite3").resolve()),
)
MEMORY_LIMIT = int(os.environ.get("MEMORY_LIMIT", "10") or 10)
MEMORY_MAX_ROWS = int(os.environ.get("MEMORY_MAX_ROWS", "200") or 200)
VACUUM_INTERVAL_HOURS = int(os.environ.get("VACUUM_INTERVAL_HOURS", "24") or 24)
TABLE_NAME = "conversation_memory"


def use_memory() -> bool:
    return USE_AGENTS_SESSION


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _mem_init(conn: sqlite3.Connection) -> None:
    # Unified conversation table: stores both roles
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_uid_created ON {TABLE_NAME}(uid, created_at);"
    )
    # One-time migration from previous split tables if present and target is empty
    try:
        cur = conn.execute(f"SELECT COUNT(1) FROM {TABLE_NAME}")
        count = int(cur.fetchone()[0])
        if count == 0:
            exist_old = {
                row[0]: True
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('assistant_memory','user_memory')"
                )
            }
            if exist_old.get('assistant_memory'):
                conn.execute(
                    f"INSERT INTO {TABLE_NAME}(uid, role, content, created_at) "
                    "SELECT uid, 'assistant', content, created_at FROM assistant_memory"
                )
            if exist_old.get('user_memory'):
                conn.execute(
                    f"INSERT INTO {TABLE_NAME}(uid, role, content, created_at) "
                    "SELECT uid, 'user', content, created_at FROM user_memory"
                )
            conn.commit()
    except Exception as e:
        print(f"[Memory] migration skipped: {e}", file=sys.stderr)


def _mem_open() -> sqlite3.Connection:
    db_path = Path(AGENTS_SQLITE_PATH)
    _ensure_parent(db_path)
    conn = sqlite3.connect(str(db_path))
    _mem_init(conn)
    return conn


def _mem_fetch(uid: str, limit: int = MEMORY_LIMIT) -> List[Dict[str, Any]]:
    try:
        with _mem_open() as conn:
            cur = conn.execute(
                f"SELECT role, content, created_at FROM {TABLE_NAME} WHERE uid = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (uid, int(limit)),
            )
            rows = [{"role": r[0], "content": r[1], "created_at": r[2]} for r in cur.fetchall()]
            rows.reverse()  # chronological
            return rows
    except Exception as e:
        print(f"[Memory] fetch failed: {e}", file=sys.stderr)
        return []


def _mem_append(uid: str, content: str, role: str) -> None:
    if not content:
        return
    try:
        with _mem_open() as conn:
            conn.execute(
                f"INSERT INTO {TABLE_NAME}(uid, role, content) VALUES(?, ?, ?)",
                (uid, role, content),
            )
            conn.commit()
    except Exception as e:
        print(f"[Memory] append failed: {e}", file=sys.stderr)


def _mem_prune(uid: str, keep: int) -> None:
    if keep <= 0:
        return
    try:
        with _mem_open() as conn:
            conn.execute(
                f"""
                DELETE FROM {TABLE_NAME}
                WHERE uid = ? AND id NOT IN (
                    SELECT id FROM {TABLE_NAME}
                    WHERE uid = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (uid, uid, int(keep)),
            )
            conn.commit()
    except Exception as e:
        print(f"[Memory] prune failed: {e}", file=sys.stderr)


def _maybe_vacuum() -> None:
    try:
        interval = max(1, int(VACUUM_INTERVAL_HOURS)) * 3600
    except Exception:
        interval = 24 * 3600
    marker = Path(str(AGENTS_SQLITE_PATH) + ".vacuum_ts")
    now = int(time.time())
    try:
        last = int(marker.stat().st_mtime)
    except Exception:
        last = 0
    if now - last < interval:
        return
    try:
        db_path = Path(AGENTS_SQLITE_PATH)
        _ensure_parent(db_path)
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            with open(marker, "wb") as f:
                f.write(b"1")
        except Exception:
            pass
        print("[Memory] VACUUM completed", file=sys.stderr)
    except Exception as e:
        print(f"[Memory] VACUUM failed: {e}", file=sys.stderr)


def build_input(uid: str, prompt: str, fallback_history: Optional[List[Dict[str, Any]]] = None) -> str:
    if use_memory():
        prev_msgs = _mem_fetch(uid, limit=MEMORY_LIMIT)
        if prev_msgs:
            lines = []
            for m in prev_msgs:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if role and content:
                    lines.append(f"{role}: {content}")
            memory_block = "\n".join(lines)
            return f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
        return prompt
    # Fallback to website-supplied history
    rows = fallback_history or []
    if rows:
        memory_block = "\n".join([f"{r['role']}: {r['content']}" for r in rows if 'role' in r and 'content' in r])
        return f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
    return prompt


def record_user(uid: str, content: str) -> None:
    if not use_memory():
        return
    _mem_append(uid, content, role="user")
    _mem_prune(uid, MEMORY_MAX_ROWS)


def record_assistant(uid: str, content: str) -> None:
    if not use_memory():
        return
    _mem_append(uid, content, role="assistant")
    _mem_prune(uid, MEMORY_MAX_ROWS)
    _maybe_vacuum()

