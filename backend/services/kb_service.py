"""
Knowledge Base Service — ChromaDB-backed microservice.
Wraps EnhancedRAGManager with a clean service interface.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """ChromaDB vector store with hybrid dense+sparse search."""

    COLLECTION_NAME = "support_knowledge_base"
    CONVERSATION_COLLECTION = "support_conversations"

    def __init__(
        self,
        vector_db_path: str = "./vector_db",
        embedding_model: str = "sentence_transformer",
    ):
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self._client: Optional[chromadb.PersistentClient] = None
        self._knowledge_col = None
        self._conversation_col = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            self._client = chromadb.PersistentClient(path=self.vector_db_path)

            if self.embedding_model == "openai" and os.getenv("OPENAI_API_KEY"):
                ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small",
                )
            else:
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )

            self._knowledge_col = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._conversation_col = self._client.get_or_create_collection(
                name=self.CONVERSATION_COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"✅ ChromaDB initialized at {self.vector_db_path}")
            logger.info(f"   📚 Knowledge docs: {self._knowledge_col.count()}")
            logger.info(f"   💬 Conversations: {self._conversation_col.count()}")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            self._client = None

    # ─────────────────────────────────────────────────────────────────────────
    # INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_documents(
        self,
        documents: List[Dict[str, str]],
        source: str = "manual",
    ) -> int:
        """Ingest list of {id, content, metadata} dicts. Returns count inserted."""
        if not self._knowledge_col:
            return 0
        ids, docs, metas = [], [], []
        for doc in documents:
            ids.append(doc.get("id", f"{source}_{len(ids)}"))
            docs.append(doc["content"])
            metas.append({**doc.get("metadata", {}), "source": source})
        try:
            self._knowledge_col.upsert(documents=docs, metadatas=metas, ids=ids)
            logger.info(f"Ingested {len(ids)} docs from '{source}'")
            return len(ids)
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return 0

    def ingest_langchain_docs(
        self, lc_docs: List[Document], source: str = "langchain"
    ) -> int:
        chunks = [
            {
                "id": f"{source}_{i}",
                "content": d.page_content,
                "metadata": d.metadata,
            }
            for i, d in enumerate(lc_docs)
        ]
        return self.ingest_documents(chunks, source)

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense similarity search. Returns ranked list with scores."""
        if not self._knowledge_col:
            return self._fallback_result(query)
        try:
            kwargs: Dict[str, Any] = dict(
                query_texts=[query],
                n_results=min(n_results, max(1, self._knowledge_col.count())),
                include=["documents", "metadatas", "distances"],
            )
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self._knowledge_col.query(**kwargs)

            output = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    output.append({
                        "content": doc,
                        "metadata": meta,
                        "score": round(1 - dist, 4),
                        "source": meta.get("source", "kb"),
                    })
            return output
        except Exception as e:
            logger.error(f"KB search failed: {e}")
            return self._fallback_result(query)

    def _fallback_result(self, query: str) -> List[Dict[str, Any]]:
        return [{
            "content": f"General support guidance for: {query}",
            "metadata": {"source": "fallback", "confidence": "low"},
            "score": 0.3,
            "source": "fallback",
        }]

    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSATION CONTEXT
    # ─────────────────────────────────────────────────────────────────────────

    def store_conversation(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store Q/A turn in conversation collection for future context retrieval."""
        if not self._conversation_col:
            return
        try:
            import hashlib
            from datetime import datetime
            doc_id = f"conv_{user_id}_{hashlib.md5((query+response).encode()).hexdigest()[:8]}"
            self._conversation_col.upsert(
                documents=[f"User: {query}\nSupport: {response}"],
                metadatas=[{
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                }],
                ids=[doc_id],
            )
        except Exception as e:
            logger.error(f"Conversation store failed: {e}")

    def get_conversation_context(
        self,
        query: str,
        user_id: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past conversations for a user."""
        if not self._conversation_col:
            return []
        try:
            count = self._conversation_col.count()
            if count == 0:
                return []
            results = self._conversation_col.query(
                query_texts=[query],
                n_results=min(n_results, count),
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
            output = []
            if results["documents"] and results["documents"][0]:
                for doc, meta in zip(
                    results["documents"][0], results["metadatas"][0]
                ):
                    output.append({"content": doc, "metadata": meta})
            return output
        except Exception as e:
            logger.warning(f"Conversation context fetch failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        if not self._client:
            return {"status": "unavailable"}
        try:
            return {
                "status": "connected",
                "knowledge_docs": self._knowledge_col.count() if self._knowledge_col else 0,
                "conversations": self._conversation_col.count() if self._conversation_col else 0,
                "path": self.vector_db_path,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}
