"""
Persistent conversation memory, multi-turn task state, and supervised learning logs.

The runtime keeps short-term task state for follow-up turns. Training data from
production is only promoted to the approved learning buffer after an explicit
human correction or approval step.
"""
from __future__ import annotations

import copy
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import CFG, DOMAIN_CFG, HISTORY_DIR, resolve_project_path, logger
from .domain_utils import canonicalize_intent, canonicalize_slots
from .storage import get_database_backend


PHONE_RE = re.compile(r"\+?216[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}")
ORDER_RE = re.compile(r"ORD-[A-Z0-9]{5,}")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _token_score(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), len(b_tokens))


def _sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = PHONE_RE.sub("<PHONE>", text)
    text = ORDER_RE.sub("<ORDER_ID>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return _sanitize_text(payload)
    if isinstance(payload, list):
        return [_sanitize_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _sanitize_payload(value) for key, value in payload.items()}
    return payload


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _merge_slots(base: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(canonicalize_slots(base))
    for key, value in canonicalize_slots(updates).items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def _session_template() -> dict[str, Any]:
    return {
        "history": [],
        "last_active": time.time(),
        "state": {
            "active_intent": None,
            "current_intent": None,
            "slots": {},
            "missing_slots": [],
            "open_form": False,
            "last_tool_call": None,
            "last_tool_result": None,
            "customer_context": {},
            "last_exchange": None,
            "last_review_status": None,
            "review_required": False,
            "turn_index": 0,
        },
    }


class ConversationMemoryStore:
    def __init__(self) -> None:
        memory_cfg = DOMAIN_CFG.get("memory", {})
        self.state_path = resolve_project_path(
            memory_cfg.get("history_state_path", HISTORY_DIR / "session_state.json")
        )
        self.log_path = resolve_project_path(
            memory_cfg.get("conversation_log_path", HISTORY_DIR / "conversations.jsonl")
        )
        self.pending_learning_path = resolve_project_path(
            memory_cfg.get("pending_learning_path", HISTORY_DIR / "learning_pending.jsonl")
        )
        self.approved_learning_path = resolve_project_path(
            memory_cfg.get("learning_buffer_path", HISTORY_DIR / "learning_buffer.jsonl")
        )
        self.feedback_log_path = resolve_project_path(
            memory_cfg.get("feedback_log_path", HISTORY_DIR / "feedback_log.jsonl")
        )
        self.feedback_dpo_path = resolve_project_path(
            memory_cfg.get("approved_dpo_feedback_path", HISTORY_DIR / "feedback_dpo.jsonl")
        )
        self.ratings_log_path = resolve_project_path(
            memory_cfg.get("ratings_log_path", HISTORY_DIR / "ratings_log.jsonl")
        )

        for path in [
            self.state_path,
            self.log_path,
            self.pending_learning_path,
            self.approved_learning_path,
            self.feedback_log_path,
            self.feedback_dpo_path,
            self.ratings_log_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.db = get_database_backend()
        self._sessions: dict[str, dict[str, Any]] = defaultdict(_session_template)
        self._load_state()

    def _touch_session(self, session_id: str) -> dict[str, Any]:
        self._cleanup()
        session = self._sessions[session_id]
        session["last_active"] = time.time()
        session.setdefault("history", [])
        session.setdefault("state", copy.deepcopy(_session_template()["state"]))
        return session

    def _load_state(self) -> None:
        loaded_any = False
        if not self.state_path.exists():
            payload = {}
        else:
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load session state: %s", exc)
                payload = {}

        if isinstance(payload, dict):
            for session_id, data in payload.items():
                session = _session_template()
                if isinstance(data, dict):
                    session["history"] = data.get("history", [])
                    session["last_active"] = data.get("last_active", time.time())
                    session["state"] = _merge_slots(session["state"], data.get("state", {}))
                    session["state"]["slots"] = _merge_slots(
                        {},
                        data.get("state", {}).get("slots", {}) if isinstance(data.get("state"), dict) else {},
                    )
                self._sessions[session_id] = session
                loaded_any = True

        db_payload = self.db.load_session_states()
        for session_id, data in db_payload.items():
            if not isinstance(data, dict):
                continue
            session = _session_template()
            session["history"] = data.get("history", [])
            session["last_active"] = data.get("last_active", time.time())
            session["state"] = _merge_slots(session["state"], data.get("state", {}))
            session["state"]["slots"] = _merge_slots(
                {},
                data.get("state", {}).get("slots", {}) if isinstance(data.get("state"), dict) else {},
            )
            self._sessions[session_id] = session
            loaded_any = True

        if loaded_any:
            self._save_state()

    def _save_state(self) -> None:
        payload = {}
        for session_id, data in self._sessions.items():
            history = data.get("history", [])[-CFG.max_history_messages :]
            payload[session_id] = {
                "history": _sanitize_payload(history),
                "last_active": data.get("last_active", time.time()),
                "state": _sanitize_payload(data.get("state", {})),
            }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db.save_session_states(payload)

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, data in self._sessions.items()
            if now - float(data.get("last_active", now)) > CFG.session_ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        session = self._touch_session(session_id)
        return list(session.get("history", []))

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        session = self._touch_session(session_id)
        return copy.deepcopy(session.get("state", {}))

    def update_session_state(
        self,
        session_id: str,
        *,
        intent: str,
        slots: dict[str, Any],
        missing_slots: list[str],
        review_required: bool = False,
        tool_call: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._touch_session(session_id)
        state = session["state"]
        intent = canonicalize_intent(intent)
        slots = canonicalize_slots(slots)
        normalized_missing = [str(item) for item in missing_slots if item]
        tool_status = str((tool_result or {}).get("status", "")).lower() if isinstance(tool_result, dict) else ""
        draft_statuses = {"draft", "collecting", "needs_confirmation", "pending_confirmation"}
        completed_statuses = {"ok", "created", "confirmed", "completed"}

        state["current_intent"] = intent
        if intent not in {"unknown", "greeting", "thanks"}:
            state["active_intent"] = intent
        state["slots"] = _merge_slots(state.get("slots", {}), slots)
        state["missing_slots"] = list(dict.fromkeys(normalized_missing))
        state["review_required"] = review_required
        state["open_form"] = bool(state["missing_slots"]) or review_required or tool_status in draft_statuses
        if tool_call is not None:
            state["last_tool_call"] = tool_call
        if tool_result is not None:
            state["last_tool_result"] = tool_result
            if isinstance(tool_result, dict) and tool_status in completed_statuses:
                state["customer_context"] = _merge_slots(
                    state.get("customer_context", {}),
                    {
                        key: value
                        for key, value in tool_result.items()
                        if key not in {"status", "message"} and value not in (None, "", [])
                    },
                )
                if tool_call is not None and not missing_slots and not review_required:
                    state["open_form"] = False
                    state["missing_slots"] = []
            elif isinstance(tool_result, dict) and tool_status in draft_statuses:
                extra_missing = [str(item) for item in tool_result.get("missing_fields", []) if item]
                if extra_missing:
                    state["missing_slots"] = list(dict.fromkeys([*state["missing_slots"], *extra_missing]))
                state["open_form"] = True
        self._save_state()
        return copy.deepcopy(state)

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        model_variant: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        session = self._touch_session(session_id)
        state = session["state"]
        history = session.setdefault("history", [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        if len(history) > CFG.max_history_messages:
            session["history"] = history[-CFG.max_history_messages :]

        state["turn_index"] = int(state.get("turn_index", 0)) + 1
        state["last_exchange"] = {
            "timestamp": now,
            "user": user_text,
            "assistant": assistant_text,
            "model_variant": model_variant,
            "metadata": metadata or {},
        }

        record = {
            "session_id": session_id,
            "timestamp": now,
            "model_variant": model_variant,
            "user": _sanitize_text(user_text),
            "assistant": _sanitize_text(assistant_text),
            "metadata": _sanitize_payload(metadata or {}),
        }
        _append_jsonl(self.log_path, record)
        self.db.record_conversation(record)

        learning_candidate = {
            "session_id": session_id,
            "timestamp": now,
            "review_status": "pending",
            "messages": [
                {"role": "user", "content": _sanitize_text(user_text)},
                {"role": "assistant", "content": _sanitize_text(assistant_text)},
            ],
            "model_variant": model_variant,
            "metadata": _sanitize_payload(metadata or {}),
            "source": "runtime_candidate",
        }
        _append_jsonl(self.pending_learning_path, learning_candidate)
        self.db.record_learning_example(learning_candidate, status="pending", source="runtime_candidate")

        self._save_state()

    def capture_feedback(
        self,
        session_id: str,
        *,
        reviewer_id: str | None = None,
        corrected_intent: str | None = None,
        corrected_slots: dict[str, Any] | None = None,
        corrected_response: str | None = None,
        corrected_tool_call: dict[str, Any] | None = None,
        corrected_tool_result: dict[str, Any] | None = None,
        approve_for_training: bool = True,
        notes: str | None = None,
    ) -> dict[str, Any]:
        session = self._touch_session(session_id)
        state = session["state"]
        last_exchange = state.get("last_exchange") or {}
        draft_user = str(last_exchange.get("user", ""))
        draft_response = str(last_exchange.get("assistant", ""))
        metadata = dict(last_exchange.get("metadata", {}) or {})

        final_intent = canonicalize_intent(
            corrected_intent or state.get("current_intent") or state.get("active_intent") or "unknown"
        )
        final_slots = _merge_slots(state.get("slots", {}), corrected_slots or {})
        final_response = corrected_response or draft_response

        feedback_record = {
            "session_id": session_id,
            "timestamp": time.time(),
            "reviewer_id": reviewer_id,
            "approve_for_training": approve_for_training,
            "draft": {
                "user": _sanitize_text(draft_user),
                "assistant": _sanitize_text(draft_response),
                "intent": metadata.get("intent"),
                "slots": _sanitize_payload(metadata.get("slots", {})),
                "tool_call": _sanitize_payload(metadata.get("tool_call")),
                "tool_result": _sanitize_payload(metadata.get("tool_result")),
            },
            "corrected": {
                "intent": final_intent,
                "slots": _sanitize_payload(final_slots),
                "response": _sanitize_text(final_response),
                "tool_call": _sanitize_payload(corrected_tool_call),
                "tool_result": _sanitize_payload(corrected_tool_result),
            },
            "notes": _sanitize_text(notes or ""),
        }
        _append_jsonl(self.feedback_log_path, feedback_record)
        self.db.record_feedback(feedback_record)

        state["current_intent"] = final_intent
        if final_intent not in {"unknown", "greeting", "thanks"}:
            state["active_intent"] = final_intent
        state["slots"] = final_slots
        if corrected_tool_call is not None:
            state["last_tool_call"] = corrected_tool_call
        if corrected_tool_result is not None:
            state["last_tool_result"] = corrected_tool_result
            if isinstance(corrected_tool_result, dict):
                state["customer_context"] = _merge_slots(
                    state.get("customer_context", {}),
                    {
                        key: value
                        for key, value in corrected_tool_result.items()
                        if key not in {"status", "message"} and value not in (None, "", [])
                    },
                )
        state["last_review_status"] = "approved" if approve_for_training else "logged"
        state["open_form"] = False
        state["missing_slots"] = []
        state["review_required"] = False

        approved_sft = False
        approved_dpo = False
        if approve_for_training and draft_user and final_response:
            approved_example = {
                "messages": [
                    {"role": "user", "content": _sanitize_text(draft_user)},
                    {"role": "assistant", "content": _sanitize_text(final_response)},
                ],
                "source": "human_feedback",
                "reviewer_id": reviewer_id,
            }
            _append_jsonl(self.approved_learning_path, approved_example)
            self.db.record_learning_example(approved_example, status="approved", source="human_feedback")
            approved_sft = True

            if draft_response and _sanitize_text(draft_response) != _sanitize_text(final_response):
                approved_pair = {
                    "prompt": _sanitize_text(draft_user),
                    "chosen": _sanitize_text(final_response),
                    "rejected": _sanitize_text(draft_response),
                    "source": "human_feedback",
                    "reviewer_id": reviewer_id,
                }
                _append_jsonl(self.feedback_dpo_path, approved_pair)
                self.db.record_learning_example(approved_pair, status="approved", source="human_feedback_dpo")
                approved_dpo = True

        self._save_state()
        return {
            "status": "feedback_recorded",
            "session_id": session_id,
            "approved_sft": approved_sft,
            "approved_dpo": approved_dpo,
            "state": copy.deepcopy(state),
        }

    def record_rating(
        self,
        session_id: str,
        *,
        verdict: str,
        reviewer_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        session = self._touch_session(session_id)
        state = session["state"]
        last_exchange = state.get("last_exchange") or {}
        user_text = str(last_exchange.get("user", ""))
        assistant_text = str(last_exchange.get("assistant", ""))

        verdict = verdict.lower().strip()
        if verdict not in {"good", "bad"}:
            raise ValueError("Rating verdict must be 'good' or 'bad'.")

        rating_record = {
            "session_id": session_id,
            "timestamp": time.time(),
            "reviewer_id": reviewer_id,
            "verdict": verdict,
            "notes": _sanitize_text(notes or ""),
            "user": _sanitize_text(user_text),
            "assistant": _sanitize_text(assistant_text),
            "metadata": _sanitize_payload(last_exchange.get("metadata", {})),
        }
        _append_jsonl(self.ratings_log_path, rating_record)
        self.db.record_rating(rating_record)

        approved_sft = False
        if verdict == "good" and user_text and assistant_text:
            approved_example = {
                "messages": [
                    {"role": "user", "content": _sanitize_text(user_text)},
                    {"role": "assistant", "content": _sanitize_text(assistant_text)},
                ],
                "source": "positive_rating",
                "reviewer_id": reviewer_id,
            }
            _append_jsonl(self.approved_learning_path, approved_example)
            self.db.record_learning_example(approved_example, status="approved", source="positive_rating")
            approved_sft = True

        state["last_review_status"] = verdict
        self._save_state()
        return {
            "status": "rating_recorded",
            "session_id": session_id,
            "verdict": verdict,
            "approved_sft": approved_sft,
            "state": copy.deepcopy(state),
        }

    def retrieve_relevant(self, session_id: str, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or CFG.memory_top_k
        history = self.get_session_history(session_id)
        if not history:
            return []

        pairs: list[dict[str, Any]] = []
        for idx in range(0, len(history) - 1, 2):
            user_turn = history[idx]
            assistant_turn = history[idx + 1] if idx + 1 < len(history) else {"content": ""}
            pairs.append(
                {
                    "user": user_turn.get("content", ""),
                    "assistant": assistant_turn.get("content", ""),
                }
            )

        scored = []
        for pair in pairs:
            score = _token_score(query, pair["user"])
            if score > 0:
                scored.append((score, pair))
        scored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, pair in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 4),
                    "text": f"Client avant: {pair['user']}\nAgent avant: {pair['assistant']}",
                }
            )
        return results

    def learning_stats(self) -> dict[str, int]:
        stats = {
            "pending_candidates": _count_lines(self.pending_learning_path),
            "approved_sft": _count_lines(self.approved_learning_path),
            "approved_dpo": _count_lines(self.feedback_dpo_path),
            "feedback_events": _count_lines(self.feedback_log_path),
            "ratings": _count_lines(self.ratings_log_path),
        }
        db_counts = self.db.counts()
        for key, value in db_counts.items():
            stats[f"db_{key}"] = value
        return stats

    def reset(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            self._sessions.pop(session_id, None)
        else:
            self._sessions.clear()
        self._save_state()
        return {"status": "reset", "session_id": session_id}
