"""
LangGraph State Schema for Customer Support System
Replaces CrewAI's task context passing with a shared TypedDict state.
All nodes read from and write to this single state object.
"""
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class SupportState(TypedDict, total=False):
    # ─── INPUT ───────────────────────────────────────────────────────────────
    customer_query: str
    user_id: str
    session_id: str
    current_year: str
    timestamp: str
    image_base64: str

    # ─── IMAGE VALIDATION ────────────────────────────────────────────────────
    image_validation_result: Dict[str, Any]

    # ─── PHASE 1 · TICKET CLASSIFICATION ─────────────────────────────────────
    urgency_level: str          # Critical / High / Medium / Low
    issue_category: str         # Technical / Billing / Account / Product / General
    sentiment: str              # Satisfied / Neutral / Frustrated / Angry
    complexity: str             # Simple / Moderate / Complex / Expert
    special_flags: List[str]    # VIP, Legal, Compliance, Escalation
    ticket_classification: Dict[str, Any]  # Full classification report

    # ─── PHASE 2 · SESSION MANAGEMENT ────────────────────────────────────────
    session_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    customer_profile: Dict[str, Any]
    personalization: Dict[str, Any]   # comm style, expertise level, etc.

    # ─── PHASE 3 · PARALLEL RETRIEVAL ────────────────────────────────────────
    kb_results: List[Dict[str, Any]]
    policy_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]

    # ─── PHASE 4 · INFORMATION FUSION ────────────────────────────────────────
    fused_information: Dict[str, Any]
    information_quality: float          # 0.0 – 1.0 confidence
    knowledge_gaps: List[str]

    # ─── PHASE 5 · SOLUTION GENERATION ───────────────────────────────────────
    solution_package: Dict[str, Any]
    primary_solution: str
    alternative_solutions: List[str]

    # ─── PHASE 6 · DYNAMIC RESPONSE ──────────────────────────────────────────
    personalized_response: str

    # ─── PHASE 7 · QA REVIEW ─────────────────────────────────────────────────
    qa_result: Dict[str, Any]
    qa_score: int               # 1–10
    qa_attempts: int            # max 2 retries

    # ─── PHASE 8 · PERSISTENCE ───────────────────────────────────────────────
    conversation_persisted: bool
    escalation_needed: bool
    escalation_report: Dict[str, Any]
    cx_optimization: Dict[str, Any]

    # ─── FINAL OUTPUT ─────────────────────────────────────────────────────────
    final_response: str
    processing_time: float

    # ─── CONTRADICTION & DEFECT DETECTION ─────────────────────────────────────
    contradiction_detected: bool
    contradiction_type: str
    contradiction_message: str
    defect_language_detected: bool

    # ─── RESOLUTION DETECTION ─────────────────────────────────────────────────
    resolution_detected: bool

    # ─── OBSERVABILITY ────────────────────────────────────────────────────────
    trace_id: str
    node_timings: Dict[str, float]   # node_name → seconds
    errors: List[str]

    # ─── CACHE ────────────────────────────────────────────────────────────────
    cache_hit: bool
    cache_key: str


def make_initial_state(
    customer_query: str,
    user_id: str,
    session_id: str,
    trace_id: str,
    image_base64: str = "",
) -> SupportState:
    """Factory: create a fresh state from an incoming request."""
    return SupportState(
        customer_query=customer_query,
        user_id=user_id,
        session_id=session_id,
        current_year=str(datetime.now().year),
        timestamp=datetime.now().isoformat(),
        trace_id=trace_id,
        image_base64=image_base64,
        image_validation_result={},
        node_timings={},
        errors=[],
        qa_attempts=0,
        cache_hit=False,
        cache_key="",
        conversation_history=[],
        special_flags=[],
        kb_results=[],
        policy_results=[],
        web_results=[],
        knowledge_gaps=[],
        alternative_solutions=[],
        conversation_persisted=False,
        escalation_needed=False,
        contradiction_detected=False,
        contradiction_type="",
        contradiction_message="",
        defect_language_detected=False,
        resolution_detected=False,
    )
