"""
FastAPI Route — Health & System Status
GET /health
GET /system-status
GET /graph
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from ..schemas import HealthResponse
from ...langgraph_system.graph import get_mermaid_diagram, get_graph_json

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic liveness check — returns service health."""
    app = request.app
    state = app.state

    services: Dict[str, Any] = {}

    # KB / ChromaDB
    if hasattr(state, "kb_service") and state.kb_service:
        services["chromadb"] = state.kb_service.health()
    else:
        services["chromadb"] = {"status": "not_initialized"}

    # Redis
    if hasattr(state, "cache_service") and state.cache_service:
        services["redis"] = state.cache_service.health()
    else:
        services["redis"] = {"status": "not_initialized"}

    # Policy
    if hasattr(state, "policy_service") and state.policy_service:
        services["policy"] = state.policy_service.health()
    else:
        services["policy"] = {"status": "not_initialized"}

    # LangGraph graph
    services["langgraph"] = {
        "status": "compiled" if hasattr(state, "support_graph") and state.support_graph else "not_initialized",
        "nodes": 10,
        "parallel_phases": 2,
    }

    overall = (
        "healthy"
        if all(s.get("status") not in ("unavailable", "error") for s in services.values())
        else "degraded"
    )

    from datetime import datetime
    return HealthResponse(
        status=overall,
        timestamp=datetime.now().isoformat(),
        services=services,
    )


@router.get("/graph")
async def get_graph_topology() -> Dict[str, Any]:
    """Return graph topology JSON — used by the frontend pipeline visualizer."""
    return {
        "graph": get_graph_json(),
        "mermaid": get_mermaid_diagram(),
    }


@router.get("/system-status")
async def system_status(request: Request) -> Dict[str, Any]:
    """Detailed system status including observability metrics."""
    from ...observability.traces import TraceManager
    from datetime import datetime

    recent_traces = TraceManager.get_recent_traces(limit=10)
    completed = [t for t in recent_traces if t.get("status") == "completed"]
    avg_time = (
        sum(t.get("processing_time_s", 0) for t in completed) / len(completed)
        if completed else 0
    )

    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "langgraph": {
            "nodes": 10,
            "parallel_phases": ["retrieval (KB+Policy+Web)", "post_persist (Escalation+CX)"],
            "conditional_edges": ["qa_review → generate_solution (retry)", "qa_review → persist"],
        },
        "observability": {
            "traces_in_memory": len(recent_traces),
            "avg_processing_time_s": round(avg_time, 2),
            "langsmith_enabled": bool(__import__("os").getenv("LANGCHAIN_API_KEY")),
        },
        "version": "2.0.0-langgraph",
    }
