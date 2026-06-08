"""
FastAPI Route — Chat Endpoints
POST /chat/query    — Standard JSON response
GET  /chat/stream   — Server-Sent Events (SSE) streaming response
POST /chat/feedback — Customer satisfaction feedback
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas import ChatRequest, ChatResponse, FeedbackRequest, StreamChunk, EvaluationResult

router = APIRouter(prefix="/chat", tags=["Chat"])


# ─── Money-matter keywords that trigger admin HITL review ───────────────────────
_MONEY_KEYWORDS = {
    "refund", "return", "returned", "money", "payment", "charge", "charged",
    "overcharged", "billing", "invoice", "reimburs", "credit", "damaged",
    "broken", "defective", "not initiated", "not processed", "dispute",
    "chargeback", "compensation", "reimburse", "fee", "penalty"
}
_MONEY_CATEGORIES = {"Billing", "Payment", "Refund"}


def _is_money_matter(query: str, category: str, escalation_needed: bool) -> bool:
    """Return True if this conversation involves money and needs admin approval."""
    if escalation_needed:
        return True
    if category in _MONEY_CATEGORIES:
        return True
    q_lower = query.lower()
    return any(kw in q_lower for kw in _MONEY_KEYWORDS)


def _maybe_queue_admin_ticket(
    cache,
    query: str,
    category: str,
    urgency: str,
    user_id: str,
    session_id: str,
    trace_id: str,
    ai_response: str,
    escalation_needed: bool = False,
    escalation_report: dict = None,
) -> None:
    """
    If this conversation is a money matter, write a pending admin ticket
    to Redis so the admin dashboard can show it for human review.
    """
    if not _is_money_matter(query, category, escalation_needed):
        return

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id":          ticket_id,
        "status":             "pending",
        "user_id":            user_id,
        "session_id":         session_id,
        "trace_id":           trace_id,
        "query":              query,
        "ai_response":        ai_response,
        "category":           category,
        "urgency":            urgency,
        "escalation_needed":  escalation_needed,
        "escalation_report":  escalation_report or {},
        "ai_recommendation":  (
            escalation_report.get("handoff_notes", "")
            if escalation_report else "Requires manual review — money matter detected"
        ),
        "timestamp":          datetime.now().isoformat(),
    }

    import json
    ticket_json = json.dumps(ticket)

    if cache and cache._available:
        key = f"admin:pending:{ticket_id}"
        cache._r.setex(key, 86400 * 7, ticket_json)  # keep 7 days
    else:
        # In-memory fallback (module-level dict so it persists across requests)
        if not hasattr(cache, "_pending_store") or cache is None:
            import importlib
            import sys
            # Store on module level as fallback
            mod = sys.modules[__name__]
            if not hasattr(mod, "_in_memory_pending"):
                mod._in_memory_pending = {}
            mod._in_memory_pending[ticket_id] = ticket

    import logging
    logging.getLogger(__name__).info(
        f"🚨 Admin ticket queued: {ticket_id} | {category} | {urgency} | user={user_id}"
    )


# ─── Shared pipeline runner ───────────────────────────────────────────────────
async def _run_pipeline(
    request: Request,
    chat_req: ChatRequest,
    trace_id: str,
) -> dict:
    """Core logic: guardrails → cache check → LangGraph → evaluation."""
    app_state = request.app.state

    guard = app_state.guardrails
    cache = app_state.cache_service
    graph = app_state.support_graph
    fallback_engine = app_state.fallback_engine
    evaluator = app_state.evaluator

    user_id = chat_req.user_id or f"user_{uuid.uuid4().hex[:8]}"
    session_id = chat_req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    query = chat_req.query

    # ── 1. Rate limit ────────────────────────────────────────────────────────
    rate_info = cache.check_rate_limit(user_id) if cache else {"allowed": True}
    if not rate_info["allowed"]:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Reset in {rate_info.get('reset_in', 60)}s",
        )

    # ── 2. Guardrails (input) ────────────────────────────────────────────────
    guard_result = guard.validate_input(query, user_id)
    if not guard_result["valid"]:
        raise HTTPException(status_code=400, detail=guard_result["block_reason"])
    query = guard_result["sanitized_query"]

    # ── 3. Complaint tracking + cache key ────────────────────────────────────
    fingerprint = cache.query_fingerprint(query) if cache else uuid.uuid4().hex[:8]
    # We don't know category yet, so check generic cache first
    cache_key = f"response:{fingerprint}"
    cached = cache.get_cached_response(cache_key) if cache else None

    if cached:
        # Track complaint even on cache hit
        if cache:
            cache.track_complaint(cached.get("category", "General"))
        return {
            **cached,
            "user_id": user_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "cache_hit": True,
            "fallback_used": False,
        }

    # ── 4. LangGraph pipeline ────────────────────────────────────────────────
    from ...observability.traces import TraceManager
    TraceManager.start_trace(trace_id, query, user_id)

    try:
        final_state = await graph.ainvoke(
            query=query,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
        )
    except Exception as e:
        # Fallback engine handles all pipeline failures
        result = fallback_engine.handle_failure(
            query=query,
            error=e,
            category="General",
            cache_service=cache,
        )
        result.update({"user_id": user_id, "session_id": session_id, "trace_id": trace_id})
        return result

    # ── 5. Guardrails (output) ───────────────────────────────────────────────
    raw_response = final_state.get("final_response", "")
    out_guard = guard.validate_output(raw_response)
    clean_response = out_guard["sanitized_response"]

    # ── 6. Evaluation (FCR) ──────────────────────────────────────────────────
    evaluation = evaluator.evaluate_response(
        customer_query=query,
        final_response=clean_response,
        solution_package=final_state.get("solution_package"),
        qa_score=final_state.get("qa_score", 7),
    )

    # ── 7. Complaint pattern tracking ────────────────────────────────────────
    category = final_state.get("issue_category", "General")
    urgency = final_state.get("urgency_level", "Medium")
    if cache:
        cache.track_complaint(category, urgency)
        # Cache this response for future identical queries
        full_cache_key = cache.make_cache_key(
            category, urgency, fingerprint
        )
        cache.cache_response(full_cache_key, {
            "final_response": clean_response,
            "category": category,
            "urgency": urgency,
            "qa_score": final_state.get("qa_score", 7),
        })
        # Also update the generic key
        cache.cache_response(cache_key, {
            "final_response": clean_response,
            "category": category,
            "urgency": urgency,
            "qa_score": final_state.get("qa_score", 7),
        })

    # ── 8. Finish trace ──────────────────────────────────────────────────────
    TraceManager.finish_trace(
        trace_id=trace_id,
        final_response=clean_response,
        qa_score=final_state.get("qa_score", 7),
        processing_time=final_state.get("processing_time", 0),
    )

    # ── 9. Money-matter detection → Admin HITL queue ─────────────────────────
    _maybe_queue_admin_ticket(
        cache=cache,
        query=query,
        category=category,
        urgency=final_state.get("urgency_level", "Medium"),
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        ai_response=clean_response,
        escalation_needed=final_state.get("escalation_needed", False),
        escalation_report=final_state.get("escalation_report"),
    )

    return {
        "response": clean_response,
        "user_id": user_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "urgency_level": category and final_state.get("urgency_level"),
        "issue_category": category,
        "sentiment": final_state.get("sentiment"),
        "complexity": final_state.get("complexity"),
        "processing_time_seconds": final_state.get("processing_time", 0),
        "cache_hit": False,
        "fallback_used": False,
        "qa_score": final_state.get("qa_score"),
        "qa_attempts": final_state.get("qa_attempts"),
        "escalation_needed": final_state.get("escalation_needed", False),
        "escalation_report": final_state.get("escalation_report"),
        "node_timings": final_state.get("node_timings", {}),
        "errors": final_state.get("errors", []) + out_guard.get("flags", []),
        "evaluation": evaluation,
        "parallel_phases_used": ["retrieval", "post_persist"],
    }


# ─── POST /chat/query ─────────────────────────────────────────────────────────
@router.post("/query", response_model=ChatResponse)
async def chat_query(chat_req: ChatRequest, request: Request) -> ChatResponse:
    """
    Standard synchronous chat endpoint.
    Runs the full LangGraph pipeline and returns a complete JSON response.
    """
    trace_id = str(uuid.uuid4())
    result = await _run_pipeline(request, chat_req, trace_id)

    eval_data = result.get("evaluation")
    eval_result = None
    if eval_data and isinstance(eval_data, dict):
        eval_result = EvaluationResult(
            need_fulfillment_score=eval_data.get("need_fulfillment_score", 7),
            completeness_score=eval_data.get("completeness_score", 7),
            actionability_score=eval_data.get("actionability_score", 7),
            overall_fcr_score=eval_data.get("overall_fcr_score", 7),
            verdict=eval_data.get("verdict", "Partially Resolved"),
            gaps=eval_data.get("gaps", []),
            recommendation=eval_data.get("recommendation"),
            would_customer_reply=eval_data.get("would_customer_reply", False),
        )

    return ChatResponse(
        response=result.get("response") or result.get("final_response", ""),
        session_id=result.get("session_id", ""),
        user_id=result.get("user_id", ""),
        trace_id=result.get("trace_id", trace_id),
        urgency_level=result.get("urgency_level"),
        issue_category=result.get("issue_category"),
        sentiment=result.get("sentiment"),
        complexity=result.get("complexity"),
        processing_time_seconds=result.get("processing_time_seconds", 0),
        cache_hit=result.get("cache_hit", False),
        fallback_used=result.get("fallback_used", False),
        fallback_tier=result.get("fallback_tier"),
        qa_score=result.get("qa_score"),
        qa_attempts=result.get("qa_attempts"),
        evaluation=eval_result,
        escalation_needed=result.get("escalation_needed", False),
        escalation_report=result.get("escalation_report"),
        node_timings=result.get("node_timings"),
        errors=result.get("errors", []),
        parallel_phases_used=result.get("parallel_phases_used", []),
    )


# ─── GET /chat/stream ─────────────────────────────────────────────────────────
@router.post("/stream")
async def chat_stream(chat_req: ChatRequest, request: Request) -> StreamingResponse:
    """
    SSE streaming endpoint — yields node-by-node progress as text/event-stream.
    Frontend displays live pipeline execution status.
    """
    trace_id = str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        app_state = request.app.state
        guard = app_state.guardrails
        cache = app_state.cache_service
        graph = app_state.support_graph
        fallback_engine = app_state.fallback_engine
        evaluator = app_state.evaluator

        user_id = chat_req.user_id or f"user_{uuid.uuid4().hex[:8]}"
        session_id = chat_req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
        query = chat_req.query

        yield sse({"node": "start", "status": "running",
                   "message": "🚀 Starting support pipeline...", "trace_id": trace_id})

        # Rate limit
        rate_info = cache.check_rate_limit(user_id) if cache else {"allowed": True}
        if not rate_info["allowed"]:
            yield sse({"node": "error", "status": "error",
                       "message": f"Rate limit exceeded. Retry in {rate_info.get('reset_in')}s"})
            return

        # Guardrails
        yield sse({"node": "guardrails", "status": "running", "message": "🛡️ Validating input..."})
        guard_result = guard.validate_input(query, user_id)
        if not guard_result["valid"]:
            yield sse({"node": "guardrails", "status": "error",
                       "message": f"❌ Blocked: {guard_result['block_reason']}"})
            return
        query = guard_result["sanitized_query"]
        yield sse({"node": "guardrails", "status": "completed", "message": "✅ Input validated"})

        # Cache check
        fingerprint = cache.query_fingerprint(query) if cache else uuid.uuid4().hex[:8]
        cache_key = f"response:{fingerprint}"
        cached = cache.get_cached_response(cache_key) if cache else None
        if cached:
            yield sse({"node": "cache", "status": "completed",
                       "message": "⚡ Cache hit! Returning cached response",
                       "partial_data": {"cache_hit": True}})
            yield sse({
                "node": "final",
                "status": "completed",
                "message": "✅ Response ready",
                "partial_data": {
                    "response": cached.get("final_response", ""),
                    "cache_hit": True,
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "processing_time_seconds": 0.05,
                }
            })
            return

        # Stream LangGraph nodes
        from ...observability.traces import TraceManager
        TraceManager.start_trace(trace_id, query, user_id)

        node_messages = {
            "classify_ticket":        "🏷️  Classifying ticket (urgency, category, sentiment)...",
            "manage_session":         "👤 Loading customer session & context...",
            "parallel_retrieval":     "⚡ PARALLEL: KB + Policy + Web search running simultaneously...",
            "fuse_information":       "🔗 Fusing information from all sources...",
            "generate_solution":      "💡 Generating solution package...",
            "personalize_response":   "✍️  Personalizing response for customer...",
            "qa_review":              "⭐ QA review & compliance check...",
            "persist_conversation":   "💾 Persisting to ChromaDB + Redis cache...",
            "escalation_coordinator": "🚨 Checking escalation requirements...",
            "cx_optimizer":           "📊 Running CX optimization analysis...",
        }

        try:
            final_state = None
            async for chunk in graph.astream(query, user_id, session_id, trace_id):
                for node_name, node_output in chunk.items():
                    if node_name == "__end__":
                        continue
                    msg = node_messages.get(node_name, f"🔄 Running {node_name}...")
                    summary = {}
                    if isinstance(node_output, dict):
                        if "urgency_level" in node_output:
                            summary["urgency"] = node_output["urgency_level"]
                            summary["category"] = node_output.get("issue_category")
                        if "qa_score" in node_output:
                            summary["qa_score"] = node_output["qa_score"]
                            summary["qa_attempts"] = node_output.get("qa_attempts")
                        if "escalation_needed" in node_output:
                            summary["escalation_needed"] = node_output["escalation_needed"]
                        if "processing_time" in node_output:
                            summary["processing_time"] = node_output["processing_time"]
                    yield sse({
                        "node": node_name,
                        "status": "completed",
                        "message": msg.replace("...", " ✅"),
                        "partial_data": summary,
                    })
                    final_state = node_output if node_output else final_state

            # Get final state from last ainvoke
            complete_state = await graph.ainvoke(query, user_id, session_id, trace_id)
            raw_response = complete_state.get("final_response", "")
            out_guard = guard.validate_output(raw_response)
            clean_response = out_guard["sanitized_response"]

            # Evaluation
            yield sse({"node": "evaluation", "status": "running", "message": "🔍 Evaluating response quality..."})
            evaluation = evaluator.evaluate_response(
                customer_query=query,
                final_response=clean_response,
                solution_package=complete_state.get("solution_package"),
                qa_score=complete_state.get("qa_score", 7),
            )
            yield sse({"node": "evaluation", "status": "completed",
                       "message": f"📊 FCR Score: {evaluation.get('overall_fcr_score', 0):.1f}/10 — {evaluation.get('verdict', '')}",
                       "partial_data": {"fcr_score": evaluation.get("overall_fcr_score")}})

            # Complaint tracking
            category = complete_state.get("issue_category", "General")
            if cache:
                cache.track_complaint(category, complete_state.get("urgency_level", "Medium"))
                cache.cache_response(cache_key, {
                    "final_response": clean_response,
                    "category": category,
                    "urgency": complete_state.get("urgency_level"),
                    "qa_score": complete_state.get("qa_score", 7),
                })

            TraceManager.finish_trace(trace_id, clean_response,
                                      complete_state.get("qa_score", 7),
                                      complete_state.get("processing_time", 0))

            yield sse({
                "node": "final",
                "status": "completed",
                "message": "✅ Response ready",
                "partial_data": {
                    "response": clean_response,
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "urgency_level": complete_state.get("urgency_level"),
                    "issue_category": category,
                    "sentiment": complete_state.get("sentiment"),
                    "processing_time_seconds": complete_state.get("processing_time", 0),
                    "qa_score": complete_state.get("qa_score"),
                    "escalation_needed": complete_state.get("escalation_needed", False),
                    "node_timings": complete_state.get("node_timings", {}),
                    "evaluation": evaluation,
                    "cache_hit": False,
                    "errors": complete_state.get("errors", []),
                }
            })

        except Exception as e:
            yield sse({"node": "error", "status": "error", "message": f"⚠️ Pipeline error: {str(e)}"})
            fallback = fallback_engine.handle_failure(query, e, cache_service=cache)
            yield sse({
                "node": "final",
                "status": "completed",
                "message": "✅ Fallback response ready",
                "partial_data": {
                    "response": fallback.get("final_response", ""),
                    "fallback_used": True,
                    "fallback_tier": fallback.get("fallback_tier"),
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "session_id": session_id,
                }
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── POST /chat/feedback ──────────────────────────────────────────────────────
@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, request: Request) -> dict:
    """Record customer satisfaction feedback."""
    from datetime import datetime
    # Store in cache for analytics
    cache = request.app.state.cache_service
    if cache and cache._available:
        key = f"feedback:{feedback.session_id}"
        import json
        cache._r.setex(key, 86400 * 7, json.dumps(feedback.dict()))

    return {
        "status": "recorded",
        "session_id": feedback.session_id,
        "rating": feedback.rating,
        "timestamp": datetime.now().isoformat(),
    }
