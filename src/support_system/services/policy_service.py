"""
Policy Service — Microservice for company policy retrieval.
Loads policies from CSV, stores in ChromaDB for semantic search.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# ─── Default sample policies ─────────────────────────────────────────────────
SAMPLE_POLICIES = [
    {"policy_id": "POL001", "category": "billing",   "title": "Payment Processing",
     "description": "Handle payment issues and disputes",
     "rules": "Investigate payment issues within 24 hours. Refunds over $100 need manager approval.",
     "authorization_level": "agent"},
    {"policy_id": "POL002", "category": "account",   "title": "Account Access",
     "description": "Account login and password reset procedures",
     "rules": "Password resets require email verification. Account lockouts need manual review after 5 failed attempts.",
     "authorization_level": "agent"},
    {"policy_id": "POL003", "category": "technical", "title": "Technical Support SLA",
     "description": "Handle technical issues and bugs with SLA targets",
     "rules": "Critical issues: 1h response. High: 4h. Medium: 24h. Low: 72h.",
     "authorization_level": "agent"},
    {"policy_id": "POL004", "category": "billing",   "title": "Refund Policy",
     "description": "Customer refund guidelines",
     "rules": "Full refunds within 30 days. Partial refunds (50%) for service disruptions. Refunds >$500 require supervisor approval.",
     "authorization_level": "manager"},
    {"policy_id": "POL005", "category": "general",   "title": "Customer Communication Standards",
     "description": "Professional communication standards for all channels",
     "rules": "All communications must be helpful and professional. Response time SLA: 24 hours.",
     "authorization_level": "agent"},
    {"policy_id": "POL006", "category": "account",   "title": "Data Privacy & GDPR",
     "description": "Customer data protection and privacy compliance",
     "rules": "Customer data requires explicit consent for sharing. All data access must be logged. GDPR deletion requests fulfilled within 30 days.",
     "authorization_level": "agent"},
    {"policy_id": "POL007", "category": "product",   "title": "Return & Exchange Policy",
     "description": "Product return and exchange procedures",
     "rules": "Products returnable within 30 days in original condition. Digital products non-refundable after download.",
     "authorization_level": "agent"},
    {"policy_id": "POL008", "category": "general",   "title": "Escalation Guidelines",
     "description": "When and how to escalate customer issues",
     "rules": "Escalate: angry customers after 2 failed resolution attempts, requests >$500 value, legal threats, VIP accounts.",
     "authorization_level": "supervisor"},
    {"policy_id": "POL009", "category": "technical", "title": "Known Issues Registry",
     "description": "Track and communicate known product issues",
     "rules": "Known issues must be documented. Customers affected by known issues get priority handling and proactive updates.",
     "authorization_level": "agent"},
    {"policy_id": "POL010", "category": "billing",   "title": "Subscription Management",
     "description": "Subscription upgrades, downgrades, and cancellations",
     "rules": "Cancellations processed immediately. Pro-rated refunds for unused period. 7-day cooling-off period for new subscriptions.",
     "authorization_level": "agent"},
]


class PolicyService:
    """Semantic policy search using ChromaDB."""

    COLLECTION_NAME = "company_policies"

    def __init__(
        self,
        data_dir: str = "./data",
        vector_db_path: str = "./vector_db",
    ):
        self.data_dir = Path(data_dir)
        self.policies: Dict[str, Dict] = {}
        self._collection = None
        self._load_policies()
        self._init_vector_store(vector_db_path)

    # ─────────────────────────────────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def _load_policies(self) -> None:
        """Load policies from CSV files or use defaults."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            self._create_sample_csv()
            csv_files = list(self.data_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    pid = row.get("policy_id", f"POL{len(self.policies):03d}")
                    self.policies[pid] = row.to_dict()
                logger.info(f"Loaded {len(df)} policies from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

        if not self.policies:
            for p in SAMPLE_POLICIES:
                self.policies[p["policy_id"]] = p
            logger.info("Using built-in sample policies")

    def _create_sample_csv(self) -> None:
        df = pd.DataFrame(SAMPLE_POLICIES)
        path = self.data_dir / "company_policies.csv"
        df.to_csv(path, index=False)
        logger.info(f"Created sample policy CSV at {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # VECTOR STORE
    # ─────────────────────────────────────────────────────────────────────────

    def _init_vector_store(self, vector_db_path: str) -> None:
        """Index policies into ChromaDB for semantic search."""
        try:
            client = chromadb.PersistentClient(path=vector_db_path)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME, embedding_function=ef
            )

            # Upsert policies (idempotent)
            ids, docs, metas = [], [], []
            for pid, p in self.policies.items():
                ids.append(pid)
                docs.append(
                    f"{p.get('title', '')}. {p.get('description', '')}. {p.get('rules', '')}"
                )
                metas.append({
                    "policy_id": pid,
                    "category": str(p.get("category", "general")),
                    "title": str(p.get("title", "")),
                    "authorization_level": str(p.get("authorization_level", "agent")),
                })
            if ids:
                self._collection.upsert(documents=docs, metadatas=metas, ids=ids)
            logger.info(f"✅ Policy vector store ready ({len(ids)} policies)")
        except Exception as e:
            logger.warning(f"Policy vector store init failed: {e}")
            self._collection = None

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Semantic policy search. Falls back to keyword matching."""
        if self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, len(self.policies)),
                    include=["documents", "metadatas", "distances"],
                )
                output = []
                if results["documents"] and results["documents"][0]:
                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ):
                        pid = meta.get("policy_id", "")
                        full = self.policies.get(pid, {})
                        output.append({
                            "policy_id": pid,
                            "title": meta.get("title", ""),
                            "category": meta.get("category", ""),
                            "rules": full.get("rules", doc),
                            "description": full.get("description", ""),
                            "authorization_level": meta.get("authorization_level", "agent"),
                            "score": round(1 - dist, 4),
                        })
                return output
            except Exception as e:
                logger.warning(f"Semantic policy search failed: {e}")

        # Keyword fallback
        return self._keyword_search(query, n_results)

    def _keyword_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        q = query.lower()
        scored = []
        for pid, p in self.policies.items():
            score = 0
            for field in ["title", "description", "rules", "category"]:
                if q in str(p.get(field, "")).lower():
                    score += 2
            for word in q.split():
                for field in ["title", "description", "rules"]:
                    if word in str(p.get(field, "")).lower():
                        score += 1
            if score > 0:
                scored.append({**p, "score": score / 10.0})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n]

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        return self.policies.get(policy_id)

    def get_all_categories(self) -> List[str]:
        return list({p.get("category", "general") for p in self.policies.values()})

    def health(self) -> Dict[str, Any]:
        return {
            "status": "operational",
            "policies_loaded": len(self.policies),
            "vector_store": "connected" if self._collection else "fallback_keyword",
            "categories": self.get_all_categories(),
        }
