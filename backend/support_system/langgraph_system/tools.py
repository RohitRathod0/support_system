"""
LangGraph Tools — wrappers around KB and Policy services
that can be injected as LangChain tools if needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class SupportTools:
    """Thin wrapper making KB + Policy services available as callable tools."""

    def __init__(self, kb_service=None, policy_service=None, cache_service=None):
        self.kb = kb_service
        self.policy = policy_service
        self.cache = cache_service

    def search_knowledge_base(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        if self.kb:
            return self.kb.search(query, n_results=n)
        return [{"content": f"KB lookup for: {query}", "score": 0.5, "source": "fallback"}]

    def search_policies(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        if self.policy:
            return self.policy.search(query, n_results=n)
        return [{"title": "General Policy", "rules": "Standard policies apply", "score": 0.5}]

    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if self.cache:
            return self.cache.get_cached_response(cache_key)
        return None

    def store_conversation(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        if self.kb:
            self.kb.store_conversation(user_id, session_id, query, response, metadata)
