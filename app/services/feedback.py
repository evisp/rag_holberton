import sqlite3
import time
from pathlib import Path
from app.config import BASE_DIR

DB_PATH = BASE_DIR / "evaluation" / "feedback.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the feedback table if it does not exist.
    Called once at app startup.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                question    TEXT    NOT NULL,
                answer      TEXT    NOT NULL,
                vote        TEXT    NOT NULL CHECK(vote IN ('up', 'down')),
                timestamp   REAL    NOT NULL
            )
        """)
        conn.commit()
    print(f"Feedback DB ready at {DB_PATH}")


def save_feedback(session_id: str, question: str, answer: str, vote: str):
    """
    Saves a thumbs up or down vote for a specific answer.
    vote must be 'up' or 'down'.
    """
    if vote not in ("up", "down"):
        raise ValueError(f"Invalid vote value: {vote}. Must be 'up' or 'down'.")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (session_id, question, answer, vote, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, question, answer, vote, time.time()),
        )
        conn.commit()


def get_stats() -> dict:
    """
    Returns aggregated feedback statistics.
    Used by /health and the evaluation dashboard.
    """
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as n FROM feedback"
        ).fetchone()["n"]

        up = conn.execute(
            "SELECT COUNT(*) as n FROM feedback WHERE vote = 'up'"
        ).fetchone()["n"]

        down = conn.execute(
            "SELECT COUNT(*) as n FROM feedback WHERE vote = 'down'"
        ).fetchone()["n"]

        recent = conn.execute(
            """
            SELECT question, vote, timestamp
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT 5
            """
        ).fetchall()

    satisfaction = round((up / total * 100), 1) if total > 0 else 0

    return {
        "total_votes":    total,
        "thumbs_up":      up,
        "thumbs_down":    down,
        "satisfaction":   f"{satisfaction}%",
        "recent":         [dict(r) for r in recent],
    }


def get_all_feedback() -> list[dict]:
    """
    Returns all feedback rows.
    Used by the evaluation dashboard.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, question, answer, vote, timestamp
            FROM feedback
            ORDER BY timestamp DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]