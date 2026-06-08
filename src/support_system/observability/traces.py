"""
AI Observability — Tracing
LangSmith integration + custom span tracking for every LangGraph node.
"""
from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger(__name__)


# ─── LangSmith setup ──────────────────────────────────────────────────────────
def _setup_langsmith() -> bool:
    """Enable LangSmith tracing if configured."""
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "customer-support-system")
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        logger.info("✅ LangSmith tracing enabled")
        return True
    logger.info("ℹ️  LangSmith not configured (LANGSMITH_API_KEY not set) — custom tracing active")
    return False


LANGSMITH_ENABLED = _setup_langsmith()


# ─── Span / Trace models ──────────────────────────────────────────────────────
class Span:
    """Represents a single node execution."""

    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.span_id = str(uuid.uuid4())[:8]
        self.name = name
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status: str = "running"
        self.metadata: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def finish(self, status: str = "success", metadata: Dict = None, error: str = None):
        self.end_time = time.time()
        self.status = status
        self.metadata = metadata or {}
        self.error = error

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return round((end - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "error": self.error,
        }


class TraceManager:
    """Manages traces across a support interaction."""

    _traces: Dict[str, Dict] = {}   # trace_id → {spans, metadata}

    @classmethod
    def start_trace(cls, trace_id: str, query: str, user_id: str) -> None:
        cls._traces[trace_id] = {
            "trace_id": trace_id,
            "query": query,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "spans": [],
            "status": "running",
        }

    @classmethod
    def add_span(cls, trace_id: str, span: Span) -> None:
        if trace_id in cls._traces:
            cls._traces[trace_id]["spans"].append(span.to_dict())

    @classmethod
    def finish_trace(
        cls,
        trace_id: str,
        final_response: str,
        qa_score: int,
        processing_time: float,
    ) -> None:
        if trace_id in cls._traces:
            cls._traces[trace_id].update({
                "status": "completed",
                "finished_at": datetime.now().isoformat(),
                "final_response_length": len(final_response),
                "qa_score": qa_score,
                "processing_time_s": processing_time,
            })

    @classmethod
    def get_trace(cls, trace_id: str) -> Optional[Dict]:
        return cls._traces.get(trace_id)

    @classmethod
    def get_recent_traces(cls, limit: int = 20) -> list:
        traces = list(cls._traces.values())
        return sorted(traces, key=lambda t: t.get("started_at", ""), reverse=True)[:limit]

    @classmethod
    def clear_old_traces(cls, max_count: int = 500) -> None:
        if len(cls._traces) > max_count:
            oldest_keys = sorted(cls._traces.keys())[: len(cls._traces) - max_count]
            for k in oldest_keys:
                del cls._traces[k]


@contextmanager
def trace_span(
    name: str, trace_id: str, metadata: Dict = None
) -> Generator[Span, None, None]:
    """Context manager for tracing a code block."""
    span = Span(name=name, trace_id=trace_id)
    try:
        yield span
        span.finish(status="success", metadata=metadata or {})
    except Exception as e:
        span.finish(status="error", error=str(e))
        raise
    finally:
        TraceManager.add_span(trace_id, span)


def trace_node(node_name: str):
    """Decorator to automatically trace a LangGraph node."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            trace_id = state.get("trace_id", "no-trace")
            with trace_span(node_name, trace_id) as span:
                result = fn(state, *args, **kwargs)
                span.metadata.update({
                    "input_keys": list(state.keys()),
                    "output_keys": list(result.keys()) if isinstance(result, dict) else [],
                })
                return result
        return wrapper
    return decorator
