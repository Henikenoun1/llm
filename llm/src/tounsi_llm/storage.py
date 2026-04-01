"""
Optional database persistence layer.

The runtime still works without a database, but when SQLAlchemy and a database
URL are available, conversation state and review events are mirrored into a DB.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from .config import CFG, logger

try:
    from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine, select
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
    from sqlalchemy.types import JSON
except Exception:  # pragma: no cover - optional dependency
    Boolean = Float = Integer = String = Text = JSON = None
    create_engine = select = sessionmaker = None
    Session = None
    DeclarativeBase = object  # type: ignore[assignment]
    Mapped = mapped_column = None  # type: ignore[assignment]


if sessionmaker is not None:

    class Base(DeclarativeBase):
        pass


    class SessionStateRow(Base):
        __tablename__ = "session_states"

        session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
        updated_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())


    class ConversationRow(Base):
        __tablename__ = "conversations"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(128), index=True)
        created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())
        model_variant: Mapped[str] = mapped_column(String(64), default="prod")
        user_text: Mapped[str] = mapped_column(Text, default="")
        assistant_text: Mapped[str] = mapped_column(Text, default="")
        metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


    class LearningExampleRow(Base):
        __tablename__ = "learning_examples"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
        created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())
        status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
        source: Mapped[str] = mapped_column(String(64), default="runtime_candidate")
        payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


    class FeedbackRow(Base):
        __tablename__ = "feedback_events"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(128), index=True)
        created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())
        reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        approve_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
        payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


    class RatingRow(Base):
        __tablename__ = "ratings"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(128), index=True)
        created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())
        reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        verdict: Mapped[str] = mapped_column(String(16), index=True)
        notes: Mapped[str] = mapped_column(Text, default="")
        payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


    class AdminCorrectionRow(Base):
        __tablename__ = "admin_corrections"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())
        pattern_text: Mapped[str] = mapped_column(Text, default="")
        normalized_pattern: Mapped[str] = mapped_column(Text, default="", index=True)
        intent: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
        runtime_mode: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
        action: Mapped[str] = mapped_column(String(32), default="replace")
        corrected_response: Mapped[str] = mapped_column(Text, default="")
        reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        notes: Mapped[str] = mapped_column(Text, default="")
        slots_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

else:
    Base = object  # type: ignore[assignment]
    SessionStateRow = ConversationRow = LearningExampleRow = FeedbackRow = RatingRow = AdminCorrectionRow = None


class DatabaseBackend:
    def __init__(self) -> None:
        self.url = CFG.database_url
        self.enabled = bool(self.url and sessionmaker is not None)
        self.error: str | None = None
        self._engine = None
        self._session_factory = None

        if not self.url:
            self.error = "database disabled: no url configured"
            return
        if sessionmaker is None:
            self.error = "database disabled: SQLAlchemy not installed"
            return

        try:
            connect_args = {"check_same_thread": False} if str(self.url).startswith("sqlite") else {}
            self._engine = create_engine(self.url, future=True, echo=CFG.database_echo, connect_args=connect_args)
            self._session_factory = sessionmaker(self._engine, expire_on_commit=False, future=True)
            Base.metadata.create_all(self._engine)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            self.enabled = False
            self.error = str(exc)
            logger.warning("Database backend disabled: %s", exc)

    def _session(self) -> Session:
        if not self.enabled or self._session_factory is None:
            raise RuntimeError("database backend is not enabled")
        return self._session_factory()

    def health(self) -> dict[str, Any]:
        if not self.url:
            return {"enabled": False, "reason": "no_url"}
        if not self.enabled:
            return {"enabled": False, "reason": self.error or "init_failed"}
        try:
            with self._session() as session:
                session.execute(select(SessionStateRow.session_id).limit(1))
            return {"enabled": True, "url": self._safe_url()}
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            return {"enabled": False, "reason": str(exc)}

    def _safe_url(self) -> str:
        if not self.url:
            return ""
        if "@" not in self.url or "://" not in self.url:
            return self.url
        prefix, suffix = self.url.split("://", 1)
        creds, tail = suffix.split("@", 1)
        if ":" in creds:
            username = creds.split(":", 1)[0]
            return f"{prefix}://{username}:***@{tail}"
        return f"{prefix}://***@{tail}"

    def load_session_states(self) -> dict[str, dict[str, Any]]:
        if not self.enabled:
            return {}
        payload: dict[str, dict[str, Any]] = {}
        with self._session() as session:
            rows = session.scalars(select(SessionStateRow)).all()
            for row in rows:
                payload[row.session_id] = row.payload or {}
        return payload

    def save_session_states(self, session_states: dict[str, dict[str, Any]]) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._session() as session:
            for session_id, payload in session_states.items():
                row = session.get(SessionStateRow, session_id)
                if row is None:
                    row = SessionStateRow(session_id=session_id, payload=payload, updated_at=now)
                    session.add(row)
                else:
                    row.payload = payload
                    row.updated_at = now
            session.commit()

    def record_conversation(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._session() as session:
            session.add(
                ConversationRow(
                    session_id=str(payload.get("session_id", "")),
                    created_at=float(payload.get("timestamp", time.time())),
                    model_variant=str(payload.get("model_variant", "prod")),
                    user_text=str(payload.get("user", "")),
                    assistant_text=str(payload.get("assistant", "")),
                    metadata_json=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
                )
            )
            session.commit()

    def record_learning_example(self, payload: dict[str, Any], *, status: str, source: str) -> None:
        if not self.enabled:
            return
        with self._session() as session:
            session.add(
                LearningExampleRow(
                    session_id=str(payload.get("session_id")) if payload.get("session_id") else None,
                    created_at=float(payload.get("timestamp", time.time())),
                    status=status,
                    source=source,
                    payload=payload,
                )
            )
            session.commit()

    def record_feedback(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._session() as session:
            session.add(
                FeedbackRow(
                    session_id=str(payload.get("session_id", "")),
                    created_at=float(payload.get("timestamp", time.time())),
                    reviewer_id=payload.get("reviewer_id"),
                    approve_for_training=bool(payload.get("approve_for_training", False)),
                    payload=payload,
                )
            )
            session.commit()

    def record_rating(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._session() as session:
            session.add(
                RatingRow(
                    session_id=str(payload.get("session_id", "")),
                    created_at=float(payload.get("timestamp", time.time())),
                    reviewer_id=payload.get("reviewer_id"),
                    verdict=str(payload.get("verdict", "")),
                    notes=str(payload.get("notes", "")),
                    payload=payload,
                )
            )
            session.commit()

    def record_admin_correction(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._session() as session:
            session.add(
                AdminCorrectionRow(
                    created_at=float(payload.get("timestamp", time.time())),
                    pattern_text=str(payload.get("pattern_text", "")),
                    normalized_pattern=str(payload.get("normalized_pattern", "")),
                    intent=payload.get("intent"),
                    runtime_mode=payload.get("runtime_mode"),
                    action=str(payload.get("action", "replace")),
                    corrected_response=str(payload.get("corrected_response", "")),
                    reviewer_id=payload.get("reviewer_id"),
                    notes=str(payload.get("notes", "")),
                    slots_json=payload.get("slots", {}) if isinstance(payload.get("slots"), dict) else {},
                )
            )
            session.commit()

    def load_admin_corrections(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._session() as session:
            rows = session.scalars(select(AdminCorrectionRow)).all()
            result: list[dict[str, Any]] = []
            for row in rows:
                result.append(
                    {
                        "timestamp": row.created_at,
                        "pattern_text": row.pattern_text,
                        "normalized_pattern": row.normalized_pattern,
                        "intent": row.intent,
                        "slots": row.slots_json or {},
                        "runtime_mode": row.runtime_mode,
                        "corrected_response": row.corrected_response,
                        "action": row.action,
                        "reviewer_id": row.reviewer_id,
                        "notes": row.notes,
                    }
                )
            return result

    def counts(self) -> dict[str, int]:
        if not self.enabled:
            return {}
        with self._session() as session:
            return {
                "session_states": len(session.scalars(select(SessionStateRow.session_id)).all()),
                "conversations": len(session.scalars(select(ConversationRow.id)).all()),
                "learning_examples": len(session.scalars(select(LearningExampleRow.id)).all()),
                "feedback_events": len(session.scalars(select(FeedbackRow.id)).all()),
                "ratings": len(session.scalars(select(RatingRow.id)).all()),
                "admin_corrections": len(session.scalars(select(AdminCorrectionRow.id)).all()),
            }

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()


@lru_cache(maxsize=1)
def get_database_backend() -> DatabaseBackend:
    return DatabaseBackend()
