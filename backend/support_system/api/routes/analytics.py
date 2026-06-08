"""
FastAPI Routes — Analytics & Sessions
GET /analytics/trends
GET /analytics/complaints
GET /analytics/traces
GET /sessions/{user_id}
DELETE /sessions/{session_id}
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from ..schemas import AnalyticsResponse, SessionResponse, TrendingComplaint

analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])
sessions_router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@analytics_router.get("/trends", response_model=AnalyticsResponse)
async def get_complaint_trends(request: Request) -> AnalyticsResponse:
    """
    Return trending complaint categories from Redis counters.
    Used by the frontend dashboard.
    """
    cache = request.app.state.cache_service

    trending_raw = cache.get_trending_categories(top_n=8) if cache else []
    hourly = cache.get_hourly_stats() if cache else {}

    total = sum(t["count"] for t in trending_raw)
    trending = [
        TrendingComplaint(
            category=t["category"],
            count_24h=t["count"],
            is_trending=t["count"] >= 5,
            percentage=round(t["count"] / total * 100, 1) if total > 0 else 0,
        )
        for t in trending_raw
    ]

    return AnalyticsResponse(
        trending_categories=trending,
        hourly_stats=hourly,
        total_queries_24h=total,
    )


@analytics_router.get("/complaints")
async def get_complaint_stats(request: Request) -> Dict[str, Any]:
    """Detailed complaint breakdown with hourly distribution."""
    cache = request.app.state.cache_service
    if not cache:
        return {"error": "Cache service unavailable"}

    trending = cache.get_trending_categories(top_n=10)
    hourly = cache.get_hourly_stats()

    return {
        "summary": {
            "total_24h": sum(t["count"] for t in trending),
            "trending_count": sum(1 for t in trending if t["count"] >= 5),
            "top_category": trending[0]["category"] if trending else "none",
        },
        "categories": trending,
        "hourly_distribution": hourly,
        "timestamp": datetime.now().isoformat(),
    }


@analytics_router.get("/traces")
async def get_traces(request: Request, limit: int = 20) -> Dict[str, Any]:
    """Return recent LangGraph execution traces for observability dashboard."""
    from ...observability.traces import TraceManager

    traces = TraceManager.get_recent_traces(limit=min(limit, 50))
    completed = [t for t in traces if t.get("status") == "completed"]

    avg_time = (
        sum(t.get("processing_time_s", 0) for t in completed) / len(completed)
        if completed else 0
    )
    avg_qa = (
        sum(t.get("qa_score", 0) for t in completed) / len(completed)
        if completed else 0
    )

    return {
        "traces": traces,
        "summary": {
            "total": len(traces),
            "completed": len(completed),
            "running": len([t for t in traces if t.get("status") == "running"]),
            "avg_processing_time_s": round(avg_time, 2),
            "avg_qa_score": round(avg_qa, 1),
        },
        "timestamp": datetime.now().isoformat(),
    }


@analytics_router.get("/performance")
async def get_performance_metrics(request: Request) -> Dict[str, Any]:
    """Node-level performance breakdown across recent traces."""
    from ...observability.traces import TraceManager

    traces = TraceManager.get_recent_traces(limit=50)
    completed = [t for t in traces if t.get("status") == "completed"]

    node_times: Dict[str, List[float]] = {}
    for trace in completed:
        for span in trace.get("spans", []):
            name = span.get("name", "unknown")
            ms = span.get("duration_ms", 0)
            node_times.setdefault(name, []).append(ms)

    node_stats = {
        name: {
            "avg_ms": round(sum(times) / len(times), 1),
            "max_ms": round(max(times), 1),
            "min_ms": round(min(times), 1),
            "count": len(times),
        }
        for name, times in node_times.items()
    }

    return {
        "node_performance": node_stats,
        "total_traces_analyzed": len(completed),
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@sessions_router.get("/{user_id}", response_model=SessionResponse)
async def get_user_session(user_id: str, request: Request) -> SessionResponse:
    """Retrieve conversation history and profile for a user."""
    cache = request.app.state.cache_service
    kb = request.app.state.kb_service

    # Try Redis session cache first
    session_data = cache.get_session_context(f"user_{user_id}") if cache else None

    # Fall back to ChromaDB conversation history
    conv_history: List[Dict] = []
    if kb:
        try:
            raw = kb.get_conversation_context(
                query="",  # empty query = get recent convos
                user_id=user_id,
                n_results=10,
            )
            conv_history = [
                {
                    "content": r.get("content", ""),
                    "timestamp": r.get("metadata", {}).get("timestamp", ""),
                    "session_id": r.get("metadata", {}).get("session_id", ""),
                }
                for r in raw
            ]
        except Exception:
            pass

    return SessionResponse(
        user_id=user_id,
        session_id=session_data.get("session_id", f"sess_{user_id}") if session_data else f"sess_{user_id}",
        conversation_history=conv_history,
        total_interactions=len(conv_history),
        account_status=session_data.get("account_status") if session_data else None,
        preferences=session_data.get("personalization") if session_data else None,
    )


@sessions_router.delete("/{session_id}")
async def clear_session(session_id: str, request: Request) -> Dict[str, Any]:
    """Clear a specific session from Redis cache."""
    cache = request.app.state.cache_service
    if cache:
        cache.invalidate(f"session:{session_id}")
    return {
        "status": "cleared",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }


@sessions_router.post("/cache/invalidate/{category}")
async def invalidate_category_cache(category: str, request: Request) -> Dict[str, Any]:
    """
    Invalidate all cached responses for a complaint category.
    Use when product/policy changes affect that category's responses.
    """
    cache = request.app.state.cache_service
    count = cache.invalidate_category(category) if cache else 0
    return {
        "status": "invalidated",
        "category": category,
        "entries_removed": count,
        "timestamp": datetime.now().isoformat(),
    }
