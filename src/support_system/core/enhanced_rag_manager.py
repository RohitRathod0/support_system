import os
import json
import logging
import numpy as np
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    chunk_id: str

@dataclass
class EnhancedContext:
    compressed_history: List[Dict]
    persistent_entities: List[str]
    predicted_intent: str
    topic_evolution: List[str]
    context_confidence: float

class ContextualUnderstandingEngine:
    """Advanced context management across conversation turns"""
    
    def __init__(self):
        self.conversation_memory = {}
        self.entity_tracker = {}
        self.intent_patterns = {}
        
    def maintain_conversation_context(self, user_id: str, current_query: str, conversation_history: List[Dict]) -> EnhancedContext:
        """Maintain rich context across conversation turns"""
        
        # Analyze conversation flow
        conversation_flow = self._analyze_conversation_flow(conversation_history)
        
        # Extract persistent entities
        persistent_entities = self._extract_persistent_entities(conversation_history)
        
        # Track topic evolution
        evolving_topics = self._track_topic_evolution(conversation_history)
        
        # Predict user intent
        predicted_intent = self._predict_next_intent(conversation_flow, current_query, persistent_entities)
        
        # Compress context for token efficiency
        compressed_context = self._compress_context(conversation_history, current_query, max_tokens=2000, preserve_entities=persistent_entities)
        
        return EnhancedContext(
            compressed_history=compressed_context,
            persistent_entities=persistent_entities,
            predicted_intent=predicted_intent,
            topic_evolution=evolving_topics,
            context_confidence=self._calculate_context_confidence(conversation_flow)
        )
    
    def _analyze_conversation_flow(self, history: List[Dict]) -> Dict:
        """Analyze patterns in conversation flow"""
        if not history:
            return {"pattern": "new_conversation", "turns": 0}
        
        return {
            "pattern": "ongoing" if len(history) > 3 else "short",
            "turns": len(history),
            "last_topic": history[-1].get("topic", "unknown") if history else None,
            "escalation_indicators": sum(1 for h in history if "frustrated" in h.get("sentiment", "").lower())
        }
    
    def _extract_persistent_entities(self, history: List[Dict]) -> List[str]:
        """Extract entities that persist across conversation"""
        entities = set()
        for turn in history[-5:]:  # Last 5 turns
            content = turn.get("query", "") + " " + turn.get("response", "")
            # Simple entity extraction (in production, use NER)
            words = content.split()
            entities.update([word for word in words if word.startswith("#") or word.isupper()])
        return list(entities)[:10]  # Top 10 entities
    
    def _track_topic_evolution(self, history: List[Dict]) -> List[str]:
        """Track how conversation topics evolve"""
        topics = []
        for turn in history[-3:]:  # Last 3 turns
            topic = turn.get("topic") or turn.get("category") or "general"
            if topic not in topics:
                topics.append(topic)
        return topics
    
    def _predict_next_intent(self, flow: Dict, query: str, entities: List[str]) -> str:
        """Predict user's likely intent"""
        query_lower = query.lower()
        
        if "help" in query_lower or "how" in query_lower:
            return "help_seeking"
        elif "problem" in query_lower or "issue" in query_lower:
            return "problem_solving"
        elif "thank" in query_lower:
            return "conversation_closing"
        elif flow["escalation_indicators"] > 2:
            return "escalation_needed"
        else:
            return "information_request"
    
    def _compress_context(self, history: List[Dict], current_query: str, max_tokens: int, preserve_entities: List[str]) -> List[Dict]:
        """Compress conversation history to fit token budget"""
        compressed = []
        token_count = 0
        encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Always include current query
        current_tokens = len(encoding.encode(current_query))
        token_count += current_tokens
        
        # Add history from most recent backwards
        for turn in reversed(history):
            turn_text = turn.get("query", "") + " " + turn.get("response", "")
            turn_tokens = len(encoding.encode(turn_text))
            
            if token_count + turn_tokens <= max_tokens:
                compressed.insert(0, turn)
                token_count += turn_tokens
            else:
                break
        
        return compressed
    
    def _calculate_context_confidence(self, flow: Dict) -> float:
        """Calculate confidence in context understanding"""
        base_confidence = 0.5
        
        # More turns = higher confidence
        if flow["turns"] > 3:
            base_confidence += 0.2
        
        # Clear pattern = higher confidence  
        if flow["pattern"] == "ongoing":
            base_confidence += 0.2
        
        # Low escalation = higher confidence
        if flow["escalation_indicators"] == 0:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

class EnhancedRAGManager:
    """Advanced RAG system with hybrid search, semantic chunking, and context optimization"""
    
    def __init__(self, vector_db_path: str = "./vector_db", embedding_model: str = "openai", enable_hybrid_search: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self._setup_logging()
        self._initialize_vector_db()
        self._initialize_embedding_models()
        self._initialize_text_splitters()
        self._initialize_sparse_retrieval()
        
        # Initialize contextual understanding engine
        self.context_engine = ContextualUnderstandingEngine()
        
        # Initialize encoding for token counting
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        logging.info("Enhanced RAG Manager initialized successfully")
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB with optimized settings"""
        try:
            self.client = chromadb.PersistentClient(path=self.vector_db_path)
            
            # Setup embedding function
            if self.embedding_model_name == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=openai_api_key,
                        model_name="text-embedding-3-large"
                    )
                else:
                    self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            else:
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            
            # Create collections
            self.knowledge_collection = self._get_or_create_collection(
                "enhanced_knowledge_base",
                metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
            )
            
            self.conversation_collection = self._get_or_create_collection(
                "enhanced_conversations",
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            self.logger.error(f"Vector DB initialization failed: {e}")
            self.client = None
    
    def _get_or_create_collection(self, name: str, metadata: Dict = None):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata=metadata or {}
        )
    
    def _initialize_embedding_models(self):
        """Initialize multiple embedding models"""
        self.local_embedding_model = None
        if self.embedding_model_name == "sentence_transformer":
            try:
                self.local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Local embedding model loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load local embedding model: {e}")
    
    def _initialize_text_splitters(self):
        """Initialize text splitting strategies"""
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def _initialize_sparse_retrieval(self):
        """Initialize TF-IDF for sparse retrieval"""
        if self.enable_hybrid_search:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.sparse_index = {}
            self.document_texts = []
    
    def ingest_documents(self, documents: List[Document], source: str = "unknown", batch_size: int = 100) -> bool:
        """Ingest documents with advanced chunking"""
        try:
            processed_chunks = []
            
            for doc in documents:
                chunks = self._smart_chunk_document(doc)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{source}_{hash(doc.page_content)}_{i}"
                    
                    enhanced_metadata = {
                        "source": source,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "token_count": len(self.encoding.encode(chunk.page_content)),
                        **chunk.metadata
                    }
                    
                    processed_chunks.append({
                        "id": chunk_id,
                        "content": chunk.page_content,
                        "metadata": enhanced_metadata
                    })
            
            # Batch insert
            self._batch_insert_chunks(processed_chunks, batch_size)
            
            # Update sparse index
            if self.enable_hybrid_search:
                self._update_sparse_index(processed_chunks)
            
            self.logger.info(f"Successfully ingested {len(processed_chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Document ingestion failed: {e}")
            return False
    
    def _smart_chunk_document(self, document: Document) -> List[Document]:
        """Smart chunking based on document characteristics"""
        doc_length = len(document.page_content)
        
        if doc_length < 1000:
            return [document]
        elif doc_length < 5000:
            return self.recursive_splitter.split_documents([document])
        else:
            primary_chunks = self.recursive_splitter.split_documents([document])
            final_chunks = []
            for chunk in primary_chunks:
                if len(chunk.page_content) > self.chunk_size * 1.5:
                    sub_chunks = self.token_splitter.split_documents([chunk])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            return final_chunks
    
    def _batch_insert_chunks(self, chunks: List[Dict], batch_size: int):
        """Insert chunks in batches"""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.knowledge_collection.add(
                documents=[chunk["content"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                ids=[chunk["id"] for chunk in batch]
            )
    
    def _update_sparse_index(self, chunks: List[Dict]):
        """Update TF-IDF index"""
        texts = [chunk["content"] for chunk in chunks]
        self.document_texts.extend(texts)
        if len(self.document_texts) > 0:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
    
    def retrieve_with_query_enhancement(self, query: str, k: int = 10, enable_reranking: bool = True, context_window_tokens: int = 4000) -> List[RetrievalResult]:
        """Advanced retrieval with query enhancement and reranking"""
        try:
            # Query enhancement
            enhanced_queries = self._enhance_query(query)
            
            # Multi-strategy retrieval
            all_results = []
            for enhanced_query in enhanced_queries:
                # Dense retrieval
                dense_results = self._dense_retrieval(enhanced_query, k)
                all_results.extend(dense_results)
                
                # Sparse retrieval
                if self.enable_hybrid_search and hasattr(self, 'tfidf_matrix'):
                    sparse_results = self._sparse_retrieval(enhanced_query, k)
                    all_results.extend(sparse_results)
            
            # Fusion and deduplication
            fused_results = self._fuse_and_deduplicate(all_results)
            
            # Reranking
            if enable_reranking:
                reranked_results = self._rerank_results(query, fused_results)
            else:
                reranked_results = fused_results[:k]
            
            # Context optimization
            optimized_results = self._optimize_context(reranked_results, context_window_tokens)
            
            return optimized_results
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []
    
    def _enhance_query(self, query: str) -> List[str]:
        """Generate enhanced query versions"""
        enhanced_queries = [query]
        if len(query.split()) > 2:
            words = query.split()
            key_terms = [word for word in words if len(word) > 3][:3]
            if key_terms:
                enhanced_queries.append(" ".join(key_terms))
        return enhanced_queries
    
    def _dense_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """Dense vector retrieval"""
        if not self.client:
            return []
        
        try:
            results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieval_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    score = 1 - distance
                    retrieval_results.append(RetrievalResult(
                        content=doc,
                        metadata=metadata,
                        score=score,
                        source="dense",
                        chunk_id=metadata.get("chunk_id", f"dense_{i}")
                    ))
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _sparse_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """Sparse TF-IDF retrieval"""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            
            retrieval_results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:
                    retrieval_results.append(RetrievalResult(
                        content=self.document_texts[idx],
                        metadata={"sparse_rank": i, "tfidf_score": similarities[idx]},
                        score=similarities[idx],
                        source="sparse",
                        chunk_id=f"sparse_{idx}"
                    ))
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _fuse_and_deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fuse and deduplicate results"""
        unique_results = []
        seen_contents = set()
        
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            content_hash = hash(result.content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using query term overlap"""
        for result in results:
            query_terms = set(query.lower().split())
            content_terms = set(result.content.lower().split())
            overlap_score = len(query_terms.intersection(content_terms)) / len(query_terms)
            result.score = (result.score + overlap_score) / 2
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _optimize_context(self, results: List[RetrievalResult], max_tokens: int) -> List[RetrievalResult]:
        """Optimize context for token budget"""
        optimized_results = []
        current_tokens = 0
        
        for result in results:
            result_tokens = len(self.encoding.encode(result.content))
            
            if current_tokens + result_tokens <= max_tokens:
                optimized_results.append(result)
                current_tokens += result_tokens
            else:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:
                    truncated_content = self._truncate_to_tokens(result.content, remaining_tokens)
                    truncated_result = RetrievalResult(
                        content=truncated_content,
                        metadata={**result.metadata, "truncated": True},
                        score=result.score,
                        source=result.source,
                        chunk_id=result.chunk_id
                    )
                    optimized_results.append(truncated_result)
                break
        
        return optimized_results
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def add_conversation_context(self, user_id: str, query: str, response: str, metadata: Dict = None):
        """Add conversation context with enhanced metadata"""
        try:
            conv_text = f"User: {query}\nAssistant: {response}"
            conv_id = f"conv_{user_id}_{datetime.now().timestamp()}"
            
            enhanced_metadata = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "token_count": len(self.encoding.encode(conv_text)),
                "conversation_type": "support",
                **(metadata or {})
            }
            
            self.conversation_collection.add(
                documents=[conv_text],
                metadatas=[enhanced_metadata],
                ids=[conv_id]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation context: {e}")
    
    def get_relevant_context(self, query: str, user_id: str = None, include_conversations: bool = True, max_results: int = 5, conversation_history: List[Dict] = None, user_preferences: Dict = None, user_expertise_level: str = "intermediate") -> Dict[str, Any]:
        """Get comprehensive relevant context with enhanced understanding"""
        
        # Use contextual understanding engine for conversation history
        enhanced_context_info = None
        if conversation_history and user_id:
            enhanced_context_info = self.context_engine.maintain_conversation_context(
                user_id=user_id,
                current_query=query,
                conversation_history=conversation_history
            )
        
        context = {
            "knowledge_results": [],
            "conversation_context": [],
            "enhanced_understanding": enhanced_context_info,
            "total_tokens": 0
        }
        
        try:
            # Get knowledge base results
            knowledge_results = self.retrieve_with_query_enhancement(query, k=max_results)
            context["knowledge_results"] = [
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "source": result.source
                }
                for result in knowledge_results
            ]
            
            # Get conversation context
            if include_conversations and user_id:
                conv_results = self.conversation_collection.query(
                    query_texts=[query],
                    where={"user_id": user_id} if user_id else {},
                    n_results=3
                )
                
                if conv_results['documents'] and conv_results['documents'][0]:
                    context["conversation_context"] = [
                        {"content": doc, "metadata": metadata}
                        for doc, metadata in zip(
                            conv_results['documents'][0],
                            conv_results['metadatas'][0]
                        )
                    ]
            
            # Calculate total tokens
            all_text = ""
            for result in context["knowledge_results"]:
                all_text += result["content"] + "\n"
            for conv in context["conversation_context"]:
                all_text += conv["content"] + "\n"
            
            context["total_tokens"] = len(self.encoding.encode(all_text))
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get relevant context: {e}")
            return context
