"""
AI Observability — Evaluation Engine
LLM-as-judge for response quality and need-fulfillment scoring.
Answers: "Did we fully resolve the customer's issue in one go?"
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluates support responses using LLM-as-judge pattern."""

    # ─────────────────────────────────────────────────────────────────────────
    # CORE: LLM-AS-JUDGE
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_response(
        self,
        customer_query: str,
        final_response: str,
        solution_package: Optional[Dict] = None,
        qa_score: int = 7,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to judge whether the response fully satisfies the customer's query.

        Returns:
          - need_fulfillment_score: 1–10 (10 = all needs fully met)
          - completeness_score: 1–10
          - actionability_score: 1–10
          - gaps: list of unaddressed questions/needs
          - verdict: 'Fully Resolved' | 'Partially Resolved' | 'Not Resolved'
          - recommendation: what to improve
        """
        start = time.time()

        # Quick check: if response is a fallback/error, score low
        error_indicators = ["technical difficulty", "unavailable", "template fallback", "[LLM ERROR"]
        if any(ind in final_response for ind in error_indicators):
            return self._low_score_result(
                reason="Fallback response detected — not LLM-generated",
                duration=time.time() - start,
            )

        try:
            from langchain_mistralai import ChatMistralAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatMistralAI(
                model="mistral-small-latest",
                api_key=os.getenv("MISTRAL_API_KEY"),
                temperature=0.1,
            )

            system = """You are an expert Customer Support Quality Evaluator.
Your job: Assess whether the support response FULLY resolves the customer's needs in ONE interaction.
This is the "First Contact Resolution" (FCR) metric.

Evaluate and return ONLY valid JSON:
{
  "need_fulfillment_score": 8,
  "completeness_score": 9,
  "actionability_score": 8,
  "clarity_score": 9,
  "empathy_score": 8,
  "overall_fcr_score": 8.4,
  "verdict": "Fully Resolved|Partially Resolved|Not Resolved",
  "unaddressed_needs": ["need not covered 1", "need not covered 2"],
  "gaps": ["specific gap 1"],
  "strengths": ["what worked well"],
  "recommendation": "specific improvement suggestion",
  "would_customer_reply": false,
  "confidence": 0.85
}

Scoring guide:
- 9-10: Exceptional, customer won't need to contact us again
- 7-8: Good, most needs met, minor gaps
- 5-6: Partial, customer likely has follow-up questions
- 1-4: Poor, core issue not addressed"""

            human = f"""CUSTOMER QUERY:
{customer_query}

SUPPORT RESPONSE:
{final_response[:1500]}

QA Score from internal review: {qa_score}/10
Solution steps provided: {len(solution_package.get('primary_steps', [])) if solution_package else 'unknown'}"""

            raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            result = json.loads(raw.content.strip())
            result["evaluation_duration_ms"] = round((time.time() - start) * 1000, 2)
            result["evaluated_at"] = datetime.now().isoformat()
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Evaluation JSON parse error: {e}")
            return self._heuristic_evaluation(customer_query, final_response, qa_score, time.time() - start)
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e} — using heuristic")
            return self._heuristic_evaluation(customer_query, final_response, qa_score, time.time() - start)

    # ─────────────────────────────────────────────────────────────────────────
    # HEURISTIC FALLBACK (no LLM needed)
    # ─────────────────────────────────────────────────────────────────────────

    def _heuristic_evaluation(
        self,
        query: str,
        response: str,
        qa_score: int,
        duration: float,
    ) -> Dict[str, Any]:
        """Rule-based evaluation when LLM judge is unavailable."""
        resp_lower = response.lower()
        query_words = set(query.lower().split())

        # Heuristics
        has_steps = any(w in resp_lower for w in ["step", "1.", "first,", "2.", "second,"])
        has_alternatives = any(w in resp_lower for w in ["alternatively", "another option", "if that doesn't"])
        has_empathy = any(w in resp_lower for w in ["understand", "apologize", "sorry", "appreciate"])
        has_contact = any(w in resp_lower for w in ["contact", "reach out", "email", "call"])
        response_length = len(response.split())
        coverage = len(query_words.intersection(set(resp_lower.split()))) / max(len(query_words), 1)

        base_score = qa_score
        if has_steps:         base_score = min(10, base_score + 0.5)
        if has_alternatives:  base_score = min(10, base_score + 0.3)
        if has_empathy:       base_score = min(10, base_score + 0.2)
        if response_length < 50: base_score = max(1, base_score - 2)

        verdict = "Fully Resolved" if base_score >= 7.5 else (
            "Partially Resolved" if base_score >= 5 else "Not Resolved"
        )

        return {
            "need_fulfillment_score": round(base_score, 1),
            "completeness_score": round(base_score, 1),
            "actionability_score": 7 if has_steps else 5,
            "clarity_score": 7,
            "empathy_score": 8 if has_empathy else 5,
            "overall_fcr_score": round(base_score, 1),
            "verdict": verdict,
            "unaddressed_needs": [],
            "gaps": [] if base_score >= 7 else ["Response may not fully address all aspects"],
            "strengths": [s for s, c in [
                ("Provides actionable steps", has_steps),
                ("Offers alternatives", has_alternatives),
                ("Empathetic tone", has_empathy),
                ("Provides contact info", has_contact),
            ] if c],
            "recommendation": "Good response" if base_score >= 7 else "Add more specific steps",
            "would_customer_reply": base_score < 6,
            "confidence": 0.6,
            "method": "heuristic",
            "evaluation_duration_ms": round(duration * 1000, 2),
            "evaluated_at": datetime.now().isoformat(),
        }

    def _low_score_result(self, reason: str, duration: float) -> Dict[str, Any]:
        return {
            "need_fulfillment_score": 4,
            "completeness_score": 3,
            "actionability_score": 3,
            "clarity_score": 5,
            "empathy_score": 5,
            "overall_fcr_score": 4.0,
            "verdict": "Not Resolved",
            "unaddressed_needs": ["Core issue not addressed by AI system"],
            "gaps": [reason],
            "strengths": [],
            "recommendation": "Full pipeline required for proper resolution",
            "would_customer_reply": True,
            "confidence": 0.9,
            "evaluation_duration_ms": round(duration * 1000, 2),
            "evaluated_at": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────

    def compute_aggregate_metrics(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate metrics across multiple evaluations."""
        if not evaluations:
            return {"error": "No evaluations provided"}

        scores = [e.get("overall_fcr_score", 0) for e in evaluations]
        verdicts = [e.get("verdict", "Unknown") for e in evaluations]

        return {
            "total_evaluations": len(evaluations),
            "avg_fcr_score": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "verdicts": {
                "Fully Resolved":   verdicts.count("Fully Resolved"),
                "Partially Resolved": verdicts.count("Partially Resolved"),
                "Not Resolved":     verdicts.count("Not Resolved"),
            },
            "fcr_rate": round(
                verdicts.count("Fully Resolved") / len(verdicts) * 100, 1
            ),
            "needs_improvement": [
                e for e in evaluations
                if e.get("overall_fcr_score", 10) < 6
            ],
        }
