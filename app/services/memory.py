import time
import uuid
from app.config import MAX_HISTORY, SESSION_TTL


class SessionMemory:
    """
    In-memory conversation store.
    Each session holds a list of {question, answer} turns
    plus a timestamp for TTL expiry.

    Structure:
        _store = {
            "session-uuid": {
                "turns": [
                    {"question": "...", "answer": "..."},
                    ...
                ],
                "last_active": 1234567890.0
            }
        }
    """

    def __init__(self):
        self._store: dict = {}

    def get_or_create(self, session_id: str) -> dict:
        """Return existing session or create a new one."""
        if session_id not in self._store:
            self._store[session_id] = {
                "turns": [],
                "last_active": time.time(),
            }
        return self._store[session_id]

    def add_turn(self, session_id: str, question: str, answer: str):
        """Append a question/answer turn to the session history."""
        session = self.get_or_create(session_id)
        session["turns"].append({
            "question": question,
            "answer":   answer,
        })
        session["last_active"] = time.time()
        self._evict_expired()

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return the last MAX_HISTORY turns for this session.
        Returns empty list if session does not exist.
        """
        if session_id not in self._store:
            return []
        session = self._store[session_id]
        session["last_active"] = time.time()
        return session["turns"][-MAX_HISTORY:]

    def clear_session(self, session_id: str):
        """Manually clear a session — used by the clear chat button."""
        if session_id in self._store:
            del self._store[session_id]

    def new_session_id(self) -> str:
        """Generate a fresh session ID."""
        return str(uuid.uuid4())

    def _evict_expired(self):
        """
        Remove sessions that have been inactive longer than SESSION_TTL.
        Called automatically after every add_turn.
        """
        now = time.time()
        expired = [
            sid for sid, data in self._store.items()
            if now - data["last_active"] > SESSION_TTL
        ]
        for sid in expired:
            del self._store[sid]

    def stats(self) -> dict:
        """Useful for the /health endpoint."""
        return {
            "active_sessions": len(self._store),
            "max_history":     MAX_HISTORY,
            "session_ttl":     SESSION_TTL,
        }


_memory_instance = None


def get_memory() -> SessionMemory:
    """
    Singleton — one memory store shared across all requests.
    Same pattern as get_retriever().
    """
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = SessionMemory()
    return _memory_instance