"""
Pydantic request/response schemas for the FastAPI backend.
All API endpoints use these models for validation and serialization.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ─── REQUEST MODELS ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Customer's support query")
    user_id: Optional[str] = Field(None, description="Customer identifier (generated if omitted)")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("query")
    def clean_query(cls, v):
        return v.strip()


class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    rating: int = Field(..., ge=1, le=5, description="Customer satisfaction rating 1–5")
    helpful: bool = Field(True)
    comment: Optional[str] = Field(None, max_length=500)


class IngestRequest(BaseModel):
    documents: List[Dict[str, str]] = Field(..., description="List of {id, content, metadata} dicts")
    source: str = Field("manual", description="Source identifier")


# ─── RESPONSE MODELS ──────────────────────────────────────────────────────────

class NodeTiming(BaseModel):
    node: str
    duration_seconds: float


class EvaluationResult(BaseModel):
    need_fulfillment_score: float
    completeness_score: float
    actionability_score: float
    overall_fcr_score: float
    verdict: str   # Fully Resolved | Partially Resolved | Not Resolved
    gaps: List[str] = []
    recommendation: Optional[str] = None
    would_customer_reply: bool = False


class ChatResponse(BaseModel):
    # Core response
    response: str = Field(..., description="Final customer-facing response")
    session_id: str
    user_id: str
    trace_id: str

    # Classification
    urgency_level: Optional[str] = None
    issue_category: Optional[str] = None
    sentiment: Optional[str] = None
    complexity: Optional[str] = None

    # Performance
    processing_time_seconds: float
    cache_hit: bool = False
    fallback_used: bool = False
    fallback_tier: Optional[str] = None

    # Quality
    qa_score: Optional[int] = None
    qa_attempts: Optional[int] = None
    evaluation: Optional[EvaluationResult] = None

    # Escalation
    escalation_needed: bool = False
    escalation_report: Optional[Dict[str, Any]] = None

    # Observability
    node_timings: Optional[Dict[str, float]] = None
    errors: List[str] = []
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Graph info
    parallel_phases_used: List[str] = ["retrieval", "post_persist"]
    
    # Auto Video Trigger
    trigger_video: bool = False

    # Resolution Detection
    resolution_detected: bool = False


class StreamChunk(BaseModel):
    """Single SSE chunk for streaming responses."""
    node: str                           # which node just completed
    status: str                         # running | completed | error
    partial_data: Optional[Dict] = None # node output summary
    message: Optional[str] = None       # human-readable status
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    graph_topology: Optional[Dict] = None


class TrendingComplaint(BaseModel):
    category: str
    count_24h: int
    is_trending: bool
    percentage: Optional[float] = None


class AnalyticsResponse(BaseModel):
    trending_categories: List[TrendingComplaint]
    hourly_stats: Dict[str, int]
    cache_hit_rate: Optional[float] = None
    avg_processing_time: Optional[float] = None
    total_queries_24h: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionResponse(BaseModel):
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    total_interactions: int
    account_status: Optional[str] = None
    preferences: Optional[Dict] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    trace_id: Optional[str] = None
