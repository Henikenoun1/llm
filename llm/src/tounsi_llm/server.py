"""
FastAPI server for the production chatbot.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .corrections import LiveCorrectionStore
from .config import CFG, DOMAIN_CFG, RUNTIME, logger
from .inference import load_llm, production_infer
from .memory import ConversationMemoryStore
from .rag import VectorRAGRetriever
from .storage import get_database_backend
from .tools import ToolRegistry, get_tool_registry


app = FastAPI(title="Configurable Call Center LLM", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: VectorRAGRetriever | None = None
_tool_registry: ToolRegistry | None = None
_memory_store: ConversationMemoryStore | None = None
_correction_store: LiveCorrectionStore | None = None


def _authorize(x_api_key: str | None = Header(default=None)) -> None:
    if not CFG.require_api_key:
        return
    if not CFG.api_key:
        raise HTTPException(status_code=500, detail="API key auth is enabled but no API key is configured.")
    if x_api_key != CFG.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.on_event("startup")
def startup() -> None:
    global _retriever, _tool_registry, _memory_store, _correction_store
    _tool_registry = get_tool_registry()
    _retriever = VectorRAGRetriever(refresh=True)
    _memory_store = ConversationMemoryStore()
    _correction_store = LiveCorrectionStore()

    try:
        if CFG.preload_models_on_startup and CFG.resolve_adapter_dir("prod") is not None:
            load_llm("prod")
    except Exception as exc:
        logger.warning("Could not preload prod model: %s", exc)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_variant: str = "prod"
    runtime_mode: str | None = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    language_style: str
    slots: dict
    missing_slots: list[str] = []
    session_id: str
    latency_ms: float
    tool_call: dict | None = None
    tool_result: dict | None = None
    rag_results: list[dict] = []
    memory_hits: list[dict] = []
    session_state: dict = {}
    needs_human_review: bool = False
    model_variant: str = "prod"
    runtime_mode: str = "speak"
    correction_applied: bool = False
    routing_reason: str = ""

class FeedbackRequest(BaseModel):
    session_id: str
    reviewer_id: str | None = None
    corrected_intent: str | None = None
    corrected_slots: dict = {}
    corrected_response: str | None = None
    corrected_tool_name: str | None = None
    corrected_tool_args: dict = {}
    approve_for_training: bool = True
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    session_id: str
    approved_sft: bool = False
    approved_dpo: bool = False
    corrected_tool_result: dict | None = None
    state: dict = {}


class RatingRequest(BaseModel):
    session_id: str
    verdict: str
    reviewer_id: str | None = None
    notes: str | None = None


class RatingResponse(BaseModel):
    status: str
    session_id: str
    verdict: str
    approved_sft: bool = False
    state: dict = {}


class AdminCorrectionRequest(BaseModel):
    session_id: str | None = None
    pattern_text: str | None = None
    corrected_response: str
    intent: str | None = None
    slots: dict = {}
    runtime_mode: str | None = None
    reviewer_id: str | None = None
    notes: str | None = None
    approve_for_training: bool = True


class AdminCorrectionResponse(BaseModel):
    status: str
    pattern_text: str
    runtime_mode: str | None = None
    approved_sft: bool = False
    approved_dpo: bool = False


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: None = Depends(_authorize)) -> ChatResponse:
    assert _retriever is not None
    assert _memory_store is not None
    assert _tool_registry is not None
    assert _correction_store is not None

    history = _memory_store.get_session_history(req.session_id)
    result = production_infer(
        user_text=req.message,
        retriever=_retriever,
        history=history,
        session_id=req.session_id,
        model_variant=req.model_variant,
        runtime_mode=req.runtime_mode,
        memory_store=_memory_store,
        tool_registry=_tool_registry,
        correction_store=_correction_store,
    )

    _memory_store.append_exchange(
        req.session_id,
        req.message,
        result["response"],
        model_variant=req.model_variant,
        metadata={
            "intent": result["intent"],
            "slots": result["slots"],
            "missing_slots": result.get("missing_slots", []),
            "tool_call": result.get("tool_call"),
            "tool_result": result.get("tool_result"),
            "session_state": result.get("session_state", {}),
            "needs_human_review": result.get("needs_human_review", False),
            "runtime_mode": result.get("runtime_mode", req.runtime_mode or "speak"),
            "correction_applied": result.get("correction_applied", False),
        },
    )

    tool_call = result.get("tool_call")
    if tool_call and "args" in tool_call:
        tool_call = {"name": tool_call["name"], "arguments": tool_call["args"]}

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        language_style=DOMAIN_CFG.get("default_language", "tounsi"),
        slots=result["slots"],
        missing_slots=result.get("missing_slots", []),
        session_id=req.session_id,
        latency_ms=result["latency_ms"],
        tool_call=tool_call,
        tool_result=result.get("tool_result"),
        rag_results=result.get("rag_results", []),
        memory_hits=result.get("memory_hits", []),
        session_state=result.get("session_state", {}),
        needs_human_review=result.get("needs_human_review", False),
        model_variant=result.get("model_variant", req.model_variant),
        runtime_mode=result.get("runtime_mode", req.runtime_mode or "speak"),
        correction_applied=result.get("correction_applied", False),
        routing_reason=result["intent"],
    )


@app.get("/health")
async def health(_: None = Depends(_authorize)) -> dict:
    retriever_info = None
    if _retriever is not None:
        retriever_info = {
            "chunks": len(_retriever.docs),
            "embedding_backend": _retriever.embedding_backend.backend_name,
            "faiss_enabled": bool(_retriever._index is not None),
        }

    return {
        "status": "ok",
        "version": "3.0",
        "runtime": RUNTIME,
        "domain": {
            "name": DOMAIN_CFG.get("domain_name"),
            "organization": DOMAIN_CFG.get("organization"),
            "assistant_name": DOMAIN_CFG.get("assistant_name"),
            "languages": DOMAIN_CFG.get("supported_languages", []),
        },
        "models": CFG.available_variants(),
        "rag": retriever_info,
        "learning": _memory_store.learning_stats() if _memory_store is not None else {},
        "database": get_database_backend().health(),
    }


@app.get("/models")
async def models(_: None = Depends(_authorize)) -> dict:
    return {
        "active_variants": CFG.available_variants(),
        "default_variant": "prod",
    }


@app.get("/tools")
async def tools(_: None = Depends(_authorize)) -> dict:
    assert _tool_registry is not None
    return {"tools": _tool_registry.list_tools()}


@app.post("/reset")
async def reset(session_id: str | None = None, _: None = Depends(_authorize)) -> dict:
    assert _memory_store is not None
    return _memory_store.reset(session_id=session_id)


@app.get("/session/{session_id}")
async def session_state(session_id: str, _: None = Depends(_authorize)) -> dict:
    assert _memory_store is not None
    return {
        "session_id": session_id,
        "state": _memory_store.get_session_state(session_id),
        "history": _memory_store.get_session_history(session_id),
    }


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest, _: None = Depends(_authorize)) -> FeedbackResponse:
    assert _memory_store is not None
    assert _tool_registry is not None

    corrected_tool_call = None
    corrected_tool_result = None
    if req.corrected_tool_name:
        corrected_tool_call = {
            "name": req.corrected_tool_name,
            "args": req.corrected_tool_args or {},
        }
        corrected_tool_result = _tool_registry.execute(req.corrected_tool_name, req.corrected_tool_args or {})

    result = _memory_store.capture_feedback(
        req.session_id,
        reviewer_id=req.reviewer_id,
        corrected_intent=req.corrected_intent,
        corrected_slots=req.corrected_slots or {},
        corrected_response=req.corrected_response,
        corrected_tool_call=corrected_tool_call,
        corrected_tool_result=corrected_tool_result,
        approve_for_training=req.approve_for_training,
        notes=req.notes,
    )
    return FeedbackResponse(
        status=result["status"],
        session_id=req.session_id,
        approved_sft=result.get("approved_sft", False),
        approved_dpo=result.get("approved_dpo", False),
        corrected_tool_result=corrected_tool_result,
        state=result.get("state", {}),
    )


@app.post("/rating", response_model=RatingResponse)
async def rating(req: RatingRequest, _: None = Depends(_authorize)) -> RatingResponse:
    assert _memory_store is not None
    result = _memory_store.record_rating(
        req.session_id,
        verdict=req.verdict,
        reviewer_id=req.reviewer_id,
        notes=req.notes,
    )
    return RatingResponse(
        status=result["status"],
        session_id=req.session_id,
        verdict=result["verdict"],
        approved_sft=result.get("approved_sft", False),
        state=result.get("state", {}),
    )


@app.post("/admin/corrections", response_model=AdminCorrectionResponse)
async def admin_corrections(req: AdminCorrectionRequest, _: None = Depends(_authorize)) -> AdminCorrectionResponse:
    assert _correction_store is not None
    assert _memory_store is not None

    pattern_text = req.pattern_text
    approved_sft = False
    approved_dpo = False

    if not pattern_text and req.session_id:
        last_state = _memory_store.get_session_state(req.session_id)
        last_exchange = last_state.get("last_exchange") if isinstance(last_state, dict) else None
        if isinstance(last_exchange, dict):
            pattern_text = last_exchange.get("user")

    if not pattern_text:
        raise HTTPException(status_code=400, detail="pattern_text or session_id with a last exchange is required.")

    _correction_store.add_correction(
        pattern_text=pattern_text,
        corrected_response=req.corrected_response,
        intent=req.intent,
        slots=req.slots or {},
        runtime_mode=req.runtime_mode,
        reviewer_id=req.reviewer_id,
        notes=req.notes,
    )

    if req.session_id:
        feedback_result = _memory_store.capture_feedback(
            req.session_id,
            reviewer_id=req.reviewer_id,
            corrected_intent=req.intent,
            corrected_slots=req.slots or {},
            corrected_response=req.corrected_response,
            approve_for_training=req.approve_for_training,
            notes=req.notes,
        )
        approved_sft = feedback_result.get("approved_sft", False)
        approved_dpo = feedback_result.get("approved_dpo", False)

    return AdminCorrectionResponse(
        status="admin_correction_recorded",
        pattern_text=pattern_text,
        runtime_mode=req.runtime_mode,
        approved_sft=approved_sft,
        approved_dpo=approved_dpo,
    )
