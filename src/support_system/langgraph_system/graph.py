"""
LangGraph Graph Definition — Customer Support System
Compiles the StateGraph with all nodes, edges, and conditional routing.
Also generates a Mermaid diagram for visual verification.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END, START

from .state import SupportState, make_initial_state
from .agents import (
    classify_ticket_node,
    manage_session_node,
    parallel_retrieval_node,
    fuse_information_node,
    generate_solution_node,
    personalize_response_node,
    qa_review_node,
    persist_conversation_node,
    escalation_coordinator_node,
    cx_optimizer_node,
)

logger = logging.getLogger(__name__)


# ─── CONDITIONAL EDGE ROUTER: QA Review ───────────────────────────────────────
def route_after_qa(state: SupportState) -> Literal["generate_solution", "persist_conversation"]:
    """
    After QA review:
      - score >= 7  → persist and deliver
      - score <  7 and attempts <= 2 → retry solution generation
      - score <  7 and attempts >  2 → force persist (avoid infinite loop)
    """
    qa_score = state.get("qa_score", 7)
    qa_attempts = state.get("qa_attempts", 0)

    if qa_score >= 7 or qa_attempts >= 2:
        logger.info(f"QA passed (score={qa_score}, attempts={qa_attempts}) → persisting")
        return "persist_conversation"
    else:
        logger.info(f"QA retry #{qa_attempts} (score={qa_score}) → regenerating solution")
        return "generate_solution"


# ─── GRAPH FACTORY ────────────────────────────────────────────────────────────
def create_support_graph(
    kb_service=None,
    policy_service=None,
    cache_service=None,
):
    """
    Build and compile the LangGraph StateGraph.

    Graph topology:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ START → classify_ticket → manage_session → parallel_retrieval       │
    │       → fuse_information → generate_solution → personalize_response │
    │       → qa_review ──(score<7, attempt≤2)──→ generate_solution      │
    │                   ──(score≥7 or attempt>2)──→ persist_conversation  │
    │       → [escalation_coordinator + cx_optimizer (parallel)] → END   │
    └─────────────────────────────────────────────────────────────────────┘
    """
    workflow = StateGraph(SupportState)

    # ── Bind services to nodes that need them ─────────────────────────────────
    parallel_retrieval_with_services = functools.partial(
        _run_async_node,
        async_fn=functools.partial(parallel_retrieval_node, kb_service=kb_service, policy_service=policy_service),
    )
    persist_with_services = functools.partial(
        persist_conversation_node,
        kb_service=kb_service,
        cache_service=cache_service,
    )

    # ── Add nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("classify_ticket",        classify_ticket_node)
    workflow.add_node("manage_session",         manage_session_node)
    workflow.add_node("parallel_retrieval",     parallel_retrieval_with_services)
    workflow.add_node("fuse_information",       fuse_information_node)
    workflow.add_node("generate_solution",      generate_solution_node)
    workflow.add_node("personalize_response",   personalize_response_node)
    workflow.add_node("qa_review",              qa_review_node)
    workflow.add_node("persist_conversation",   persist_with_services)
    workflow.add_node("escalation_coordinator", escalation_coordinator_node)
    workflow.add_node("cx_optimizer",           cx_optimizer_node)

    # ── Add edges ──────────────────────────────────────────────────────────────
    workflow.add_edge(START,                    "classify_ticket")
    workflow.add_edge("classify_ticket",        "manage_session")
    workflow.add_edge("manage_session",         "parallel_retrieval")
    workflow.add_edge("parallel_retrieval",     "fuse_information")
    workflow.add_edge("fuse_information",       "generate_solution")
    workflow.add_edge("generate_solution",      "personalize_response")
    workflow.add_edge("personalize_response",   "qa_review")

    # Conditional QA retry
    workflow.add_conditional_edges(
        "qa_review",
        route_after_qa,
        {
            "generate_solution":    "generate_solution",
            "persist_conversation": "persist_conversation",
        },
    )

    # Post-persist: run escalation + CX optimizer in parallel
    # LangGraph handles fan-out automatically when both nodes have the same source
    workflow.add_edge("persist_conversation",   "escalation_coordinator")
    workflow.add_edge("persist_conversation",   "cx_optimizer")
    workflow.add_edge("escalation_coordinator", END)
    workflow.add_edge("cx_optimizer",           END)

    compiled = workflow.compile()
    logger.info("✅ LangGraph support graph compiled successfully")
    return compiled


def _run_async_node(state: SupportState, async_fn) -> Dict[str, Any]:
    """Adapter: run an async node function from a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, async_fn(state))
                return future.result()
        else:
            return loop.run_until_complete(async_fn(state))
    except RuntimeError:
        return asyncio.run(async_fn(state))


# ─── GRAPH VISUALIZATION ──────────────────────────────────────────────────────
def get_mermaid_diagram() -> str:
    """Return Mermaid diagram string for frontend rendering."""
    return """
graph TD
    START([🚀 START]) --> A[🏷️ Classify Ticket<br/>urgency · category · sentiment]
    A --> B[👤 Manage Session<br/>history · profile · personalization]
    B --> C[⚡ PARALLEL RETRIEVAL]
    C --> D1[📚 KB Retriever<br/>ChromaDB search]
    C --> D2[📋 Policy Retriever<br/>Semantic policy search]
    C --> D3[🌐 Web Search<br/>External info coordinator]
    D1 --> E[🔗 Fuse Information<br/>Merge · resolve conflicts]
    D2 --> E
    D3 --> E
    E --> F[💡 Generate Solution<br/>Steps · alternatives · risks]
    F --> G[✍️ Personalize Response<br/>Tone · style · empathy]
    G --> H{⭐ QA Review<br/>Score 1–10}
    H -- score < 7 AND attempts ≤ 2 --> F
    H -- score ≥ 7 OR attempts > 2 --> I[💾 Persist Conversation<br/>ChromaDB + Redis cache]
    I --> J[🚨 Escalation Coordinator]
    I --> K[📊 CX Optimizer]
    J --> END_NODE([✅ END])
    K --> END_NODE

    style START fill:#6366f1,color:#fff,stroke:none
    style END_NODE fill:#10b981,color:#fff,stroke:none
    style C fill:#f59e0b,color:#fff,stroke:none
    style D1 fill:#3b82f6,color:#fff,stroke:none
    style D2 fill:#3b82f6,color:#fff,stroke:none
    style D3 fill:#3b82f6,color:#fff,stroke:none
    style H fill:#ef4444,color:#fff,stroke:none
    style I fill:#8b5cf6,color:#fff,stroke:none
"""


def get_graph_json() -> Dict[str, Any]:
    """Return graph topology as JSON for API responses."""
    return {
        "nodes": [
            {"id": "classify_ticket",        "label": "Ticket Classifier",        "phase": 1, "parallel": False},
            {"id": "manage_session",          "label": "Session Manager",           "phase": 2, "parallel": False},
            {"id": "kb_retriever",            "label": "KB Retriever",              "phase": 3, "parallel": True},
            {"id": "policy_retriever",        "label": "Policy Retriever",          "phase": 3, "parallel": True},
            {"id": "web_search",              "label": "Web Search Coordinator",    "phase": 3, "parallel": True},
            {"id": "fuse_information",        "label": "Information Fusion",        "phase": 4, "parallel": False},
            {"id": "generate_solution",       "label": "Solution Generator",        "phase": 5, "parallel": False},
            {"id": "personalize_response",    "label": "Dynamic Responder",         "phase": 6, "parallel": False},
            {"id": "qa_review",               "label": "QA Reviewer",               "phase": 7, "parallel": False, "conditional": True},
            {"id": "persist_conversation",    "label": "Conversation Persister",    "phase": 8, "parallel": False},
            {"id": "escalation_coordinator",  "label": "Escalation Coordinator",    "phase": 9, "parallel": True},
            {"id": "cx_optimizer",            "label": "CX Optimizer",              "phase": 9, "parallel": True},
        ],
        "edges": [
            {"from": "START",                 "to": "classify_ticket"},
            {"from": "classify_ticket",       "to": "manage_session"},
            {"from": "manage_session",        "to": "parallel_retrieval"},
            {"from": "parallel_retrieval",    "to": "kb_retriever",           "type": "parallel"},
            {"from": "parallel_retrieval",    "to": "policy_retriever",       "type": "parallel"},
            {"from": "parallel_retrieval",    "to": "web_search",             "type": "parallel"},
            {"from": "kb_retriever",          "to": "fuse_information"},
            {"from": "policy_retriever",      "to": "fuse_information"},
            {"from": "web_search",            "to": "fuse_information"},
            {"from": "fuse_information",      "to": "generate_solution"},
            {"from": "generate_solution",     "to": "personalize_response"},
            {"from": "personalize_response",  "to": "qa_review"},
            {"from": "qa_review",             "to": "generate_solution",      "type": "conditional", "label": "score<7 & attempts≤2"},
            {"from": "qa_review",             "to": "persist_conversation",   "type": "conditional", "label": "score≥7"},
            {"from": "persist_conversation",  "to": "escalation_coordinator", "type": "parallel"},
            {"from": "persist_conversation",  "to": "cx_optimizer",           "type": "parallel"},
            {"from": "escalation_coordinator","to": "END"},
            {"from": "cx_optimizer",          "to": "END"},
        ],
        "parallel_phases": [
            {"phase": 3, "nodes": ["kb_retriever", "policy_retriever", "web_search"]},
            {"phase": 9, "nodes": ["escalation_coordinator", "cx_optimizer"]},
        ],
    }


# ─── PUBLIC INTERFACE ─────────────────────────────────────────────────────────
class SupportGraph:
    """High-level interface wrapping the compiled LangGraph."""

    def __init__(self, kb_service=None, policy_service=None, cache_service=None):
        self._kb = kb_service
        self._policy = policy_service
        self._cache = cache_service
        self._graph = create_support_graph(kb_service, policy_service, cache_service)

    def invoke(self, query: str, user_id: str, session_id: str, trace_id: str) -> SupportState:
        """Synchronous invocation — returns final state."""
        import uuid
        state = make_initial_state(
            customer_query=query,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id or str(uuid.uuid4()),
        )
        return self._graph.invoke(state)

    async def ainvoke(self, query: str, user_id: str, session_id: str, trace_id: str) -> SupportState:
        """Async invocation — returns final state."""
        import uuid
        state = make_initial_state(
            customer_query=query,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id or str(uuid.uuid4()),
        )
        return await self._graph.ainvoke(state)

    async def astream(self, query: str, user_id: str, session_id: str, trace_id: str):
        """Async streaming — yields node outputs as they complete."""
        import uuid
        state = make_initial_state(
            customer_query=query,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id or str(uuid.uuid4()),
        )
        async for chunk in self._graph.astream(state):
            yield chunk

    @staticmethod
    def get_mermaid() -> str:
        return get_mermaid_diagram()

    @staticmethod
    def get_graph_json() -> Dict[str, Any]:
        return get_graph_json()
