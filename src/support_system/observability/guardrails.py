"""
AI Observability — Guardrails
Input validation, PII detection, content safety, and output sanitization.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── PII Patterns ─────────────────────────────────────────────────────────────
_PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn":         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone":       re.compile(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "password":    re.compile(r"(?i)(password\s*[:=]\s*\S+)"),
}

# ─── Unsafe content keywords ──────────────────────────────────────────────────
_UNSAFE_KEYWORDS = [
    "hack", "exploit", "malware", "ransomware", "phishing",
    "sql injection", "xss attack", "ddos",
]

# ─── Prompt injection patterns ────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+a"),
    re.compile(r"(?i)act\s+as\s+(a\s+)?(?:different|evil|unrestricted)"),
    re.compile(r"(?i)system\s*prompt"),
    re.compile(r"(?i)jailbreak"),
]

# ─── Max length limits ────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 3


class GuardrailsEngine:
    """Validates inputs and outputs for safety and compliance."""

    # ─────────────────────────────────────────────────────────────────────────
    # INPUT GUARDRAILS
    # ─────────────────────────────────────────────────────────────────────────

    def validate_input(self, query: str, user_id: str = "") -> Dict[str, Any]:
        """
        Run all input checks. Returns:
        {
          "valid": bool,
          "sanitized_query": str,
          "flags": List[str],
          "block_reason": Optional[str]
        }
        """
        flags: List[str] = []
        block_reason: Optional[str] = None

        # 1. Length check
        if len(query.strip()) < MIN_QUERY_LENGTH:
            return {
                "valid": False,
                "sanitized_query": query,
                "flags": ["too_short"],
                "block_reason": "Query too short",
            }
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
            flags.append("truncated")

        # 2. Prompt injection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                flags.append("prompt_injection_detected")
                block_reason = "Potential prompt injection detected"
                break

        # 3. Unsafe content
        q_lower = query.lower()
        for kw in _UNSAFE_KEYWORDS:
            if kw in q_lower:
                flags.append(f"unsafe_keyword:{kw}")
                # Don't block — log and continue (customer might have legitimate concern)

        # 4. PII detection (log but don't block — customer may need to share)
        pii_found = self._detect_pii(query)
        if pii_found:
            flags.extend([f"pii:{k}" for k in pii_found])
            logger.warning(f"PII detected in query from {user_id}: {pii_found}")

        return {
            "valid": block_reason is None,
            "sanitized_query": query,
            "flags": flags,
            "block_reason": block_reason,
            "pii_types": pii_found,
        }

    def _detect_pii(self, text: str) -> List[str]:
        found = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                found.append(pii_type)
        return found

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT GUARDRAILS
    # ─────────────────────────────────────────────────────────────────────────

    def validate_output(self, response: str) -> Dict[str, Any]:
        """
        Validate LLM output before sending to customer.
        Returns sanitized response + any flags.
        """
        flags: List[str] = []
        sanitized = response

        # 1. Mask any PII that leaked into response
        sanitized, pii_masked = self._mask_pii(sanitized)
        if pii_masked:
            flags.extend([f"pii_masked:{p}" for p in pii_masked])

        # 2. Remove JSON artifacts if they leaked
        if sanitized.strip().startswith("{") and sanitized.strip().endswith("}"):
            flags.append("json_leaked")
            # Try to extract meaningful text
            import json
            try:
                data = json.loads(sanitized)
                # Try common response fields
                for field in ["personalized_response", "response", "message", "content"]:
                    if field in data:
                        sanitized = str(data[field])
                        break
            except Exception:
                pass

        # 3. Minimum response length
        if len(sanitized.strip()) < 50:
            flags.append("response_too_short")
            sanitized = sanitized + "\n\nPlease don't hesitate to contact us if you need further assistance."

        # 4. Hallucination pattern: response references things not in query
        if "[LLM ERROR" in sanitized:
            flags.append("llm_error_in_response")
            sanitized = "I apologize for the technical difficulty. Our team is here to help. Please describe your issue and a support agent will assist you shortly."

        return {
            "sanitized_response": sanitized,
            "flags": flags,
            "is_safe": len([f for f in flags if "error" in f or "json_leaked" in f]) == 0,
        }

    def _mask_pii(self, text: str) -> Tuple[str, List[str]]:
        masked_types = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if pii_type == "email":
                continue  # Allow emails in responses (support contact info)
            if pattern.search(text):
                text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
                masked_types.append(pii_type)
        return text, masked_types

    # ─────────────────────────────────────────────────────────────────────────
    # CONTENT SAFETY
    # ─────────────────────────────────────────────────────────────────────────

    def is_support_relevant(self, query: str) -> Dict[str, Any]:
        """Heuristic check: is this a genuine support query?"""
        support_keywords = [
            "help", "issue", "problem", "error", "can't", "cannot", "won't",
            "broken", "fix", "failed", "billing", "refund", "account", "password",
            "login", "payment", "cancel", "subscription", "crash", "not working",
        ]
        q_lower = query.lower()
        match_count = sum(1 for kw in support_keywords if kw in q_lower)

        return {
            "is_relevant": match_count > 0 or len(query) > 20,
            "confidence": min(1.0, match_count / 3),
            "match_count": match_count,
        }
