"""In-memory KratosSessionStore implementation for deterministic unit testing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.kratos_sync._types import KratosSession, KratosSessionWrite, KratosStoreError


def _normalize_session(item: KratosSession | Mapping[str, Any]) -> KratosSession:
    if isinstance(item, KratosSession):
        return item
    return KratosSession.from_api_dict(item)


class InMemoryKratosSessionStore:
    """Fake KratosSessionStore over normalized KratosSession records."""

    def __init__(
        self,
        initial: Iterable[KratosSession | Mapping[str, Any]] = (),
        *,
        direct_only: Iterable[KratosSession | Mapping[str, Any]] = (),
        fail_delete_ids: Iterable[str] | None = None,
    ) -> None:
        self._sessions: dict[str, KratosSession] = {}
        for item in initial:
            session = _normalize_session(item)
            self._sessions[session.id] = session

        self._direct_only: dict[str, KratosSession] = {}
        for item in direct_only:
            session = _normalize_session(item)
            self._direct_only[session.id] = session

        self._fail_delete_ids = set(fail_delete_ids or ())
        self.upserts: list[KratosSessionWrite] = []
        self.deleted_ids: list[str] = []

    def list_sessions(
        self,
        *,
        limit: int = 200,
        include_deleted: bool = True,
    ) -> tuple[KratosSession, ...]:
        """Returns listed sessions up to limit, optionally excluding soft-deleted rows."""
        sessions = (
            self._sessions.values()
            if include_deleted
            else (s for s in self._sessions.values() if s.deleted_at is None)
        )
        return tuple(list(sessions)[:limit])

    def get_session(self, session_id: str) -> KratosSession | None:
        """Returns a session by ID from listed or direct-only storage."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._direct_only.get(session_id)

    def upsert_session(self, session: KratosSessionWrite) -> None:
        """Records an upsert and updates the in-memory store snapshot."""
        self.upserts.append(session)
        stored = session.to_session()
        if session.id in self._direct_only and session.id not in self._sessions:
            self._direct_only[session.id] = stored
            return
        self._sessions[session.id] = stored

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session or raises KratosStoreError if configured in fail_delete_ids."""
        if session_id in self._fail_delete_ids:
            raise KratosStoreError(
                f"Simulated delete failure for session {session_id}",
                operation="delete_session",
                status_code=500,
            )
        self.deleted_ids.append(session_id)
        self._sessions.pop(session_id, None)
        self._direct_only.pop(session_id, None)
        return True
