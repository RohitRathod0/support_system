"""
AI Observability — Fallback Engine
Multi-tier fallback strategies when LLM nodes fail.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─── Fallback response templates ──────────────────────────────────────────────
_FALLBACK_TEMPLATES: Dict[str, str] = {
    "Technical": (
        "Thank you for reaching out about the technical issue you're experiencing.\n\n"
        "While our automated system is temporarily unavailable, here are immediate steps to try:\n"
        "1. Clear your browser cache and cookies, then retry\n"
        "2. Check our Status Page at status.yourcompany.com for known outages\n"
        "3. Try a different browser or device\n"
        "4. Restart the application\n\n"
        "If the issue persists, our technical team is available 24/7. "
        "Please contact us at support@yourcompany.com with your error details."
    ),
    "Billing": (
        "Thank you for contacting us about your billing concern.\n\n"
        "Our billing team is available to assist you directly:\n"
        "📧 billing@yourcompany.com\n"
        "📞 1-800-SUPPORT (Mon–Fri, 9am–6pm)\n\n"
        "For immediate reference:\n"
        "• Refund requests are processed within 5-7 business days\n"
        "• You can view your billing history in Account → Billing\n"
        "• Payment disputes can be filed within 30 days\n\n"
        "We apologize for any inconvenience and will resolve this promptly."
    ),
    "Account": (
        "Thank you for reaching out about your account.\n\n"
        "For account-related issues, you can:\n"
        "1. Use 'Forgot Password' on the login page for access issues\n"
        "2. Visit account.yourcompany.com for self-service options\n"
        "3. Contact account@yourcompany.com for account-specific help\n\n"
        "Our team will review your case within 24 hours. "
        "Your account security is our top priority."
    ),
    "General": (
        "Thank you for contacting our support team.\n\n"
        "Our AI system is momentarily busy, but a human support agent will "
        "review your request within the next few minutes.\n\n"
        "In the meantime:\n"
        "• Visit our Help Center: help.yourcompany.com\n"
        "• Check FAQs for common questions\n"
        "• Email us: support@yourcompany.com\n\n"
        "We appreciate your patience and will get back to you shortly."
    ),
}


class FallbackEngine:
    """Provides graceful degradation when the LangGraph pipeline fails."""

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: Cached response fallback
    # ─────────────────────────────────────────────────────────────────────────
    def get_cached_fallback(
        self,
        cache_service,
        query: str,
        category: str = "General",
    ) -> Optional[Dict[str, Any]]:
        """Try to serve a cached response from Redis."""
        if not cache_service:
            return None
        try:
            fingerprint = cache_service.query_fingerprint(query)
            cache_key = cache_service.make_cache_key(category, "fallback", fingerprint)
            return cache_service.get_cached_response(cache_key)
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2: Template-based fallback
    # ─────────────────────────────────────────────────────────────────────────
    def get_template_fallback(
        self,
        category: str = "General",
        query: str = "",
        urgency: str = "Medium",
    ) -> Dict[str, Any]:
        """Return a category-appropriate pre-written response."""
        template = _FALLBACK_TEMPLATES.get(category, _FALLBACK_TEMPLATES["General"])

        return {
            "final_response": template,
            "fallback_tier": "template",
            "category": category,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False,
            "processing_time": 0.1,
            "qa_score": 6,
            "node_timings": {"fallback": 0.1},
            "errors": [f"LangGraph pipeline failed — template fallback for category: {category}"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 3: Minimal LLM fallback (single-shot, no graph)
    # ─────────────────────────────────────────────────────────────────────────
    def get_llm_fallback(self, query: str, category: str = "General") -> Dict[str, Any]:
        """
        Last-resort: single LLM call bypassing the full graph.
        Much faster than the full pipeline, but less thorough.
        """
        try:
            from langchain_mistralai import ChatMistralAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatMistralAI(
                model="mistral-small-latest",   # cheaper/faster model for fallback
                api_key=os.getenv("MISTRAL_API_KEY"),
                temperature=0.2,
            )
            system = (
                "You are a helpful customer support agent. "
                "Provide a concise, professional, empathetic response to the customer's issue. "
                "Keep it under 200 words."
            )
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=f"Customer issue ({category}): {query}"),
            ])
            return {
                "final_response": response.content.strip(),
                "fallback_tier": "llm_single_shot",
                "category": category,
                "processing_time": 2.0,
                "qa_score": 6,
                "cache_hit": False,
                "errors": ["Full graph failed — single-shot LLM fallback used"],
                "node_timings": {"llm_fallback": 2.0},
            }
        except Exception as e:
            logger.error(f"LLM fallback also failed: {e}")
            return self.get_template_fallback(category, query)

    # ─────────────────────────────────────────────────────────────────────────
    # ORCHESTRATOR
    # ─────────────────────────────────────────────────────────────────────────
    def handle_failure(
        self,
        query: str,
        error: Exception,
        category: str = "General",
        urgency: str = "Medium",
        cache_service=None,
    ) -> Dict[str, Any]:
        """
        Cascading fallback strategy:
          1. Try Redis cached response
          2. Try single-shot LLM (fast)
          3. Return template response (always works)
        """
        logger.error(f"Pipeline failure for query '{query[:50]}': {error}")

        # Tier 1: Cache
        cached = self.get_cached_fallback(cache_service, query, category)
        if cached:
            cached["fallback_tier"] = "cache"
            cached["errors"] = [f"Pipeline error: {error}"]
            return cached

        # Tier 2: Single LLM
        if os.getenv("MISTRAL_API_KEY"):
            try:
                result = self.get_llm_fallback(query, category)
                result["errors"] = [f"Pipeline error: {error}", "Used single-shot LLM fallback"]
                return result
            except Exception:
                pass

        # Tier 3: Template
        result = self.get_template_fallback(category, query, urgency)
        result["errors"] = [f"Pipeline error: {error}", "Used template fallback"]
        return result
