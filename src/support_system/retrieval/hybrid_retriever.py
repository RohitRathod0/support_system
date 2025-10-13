import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

@dataclass
class HybridSearchResult:
    content: str
    metadata: Dict[str, Any]
    dense_score: float
    sparse_score: float
    hybrid_score: float
    source: str
    chunk_id: str
    rank: int

class ReciprocalRankFusion:
    """Implement Reciprocal Rank Fusion for combining search results"""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse_results(self, 
                    dense_results: List[Dict], 
                    sparse_results: List[Dict],
                    dense_weight: float = 0.7,
                    sparse_weight: float = 0.3) -> List[HybridSearchResult]:
        """Fuse dense and sparse results using RRF"""
        
        # Create unified result dictionary
        all_results = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            result_id = result.get('chunk_id', f"dense_{rank}")
            rrf_score = dense_weight / (self.k + rank + 1)
            
            all_results[result_id] = {
                'content': result.get('content', ''),
                'metadata': result.get('metadata', {}),
                'dense_score': result.get('score', 0.0),
                'sparse_score': 0.0,
                'dense_rank': rank,
                'sparse_rank': float('inf'),
                'rrf_score': rrf_score,
                'source': 'dense'
            }
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            result_id = result.get('chunk_id', f"sparse_{rank}")
            rrf_score = sparse_weight / (self.k + rank + 1)
            
            if result_id in all_results:
                # Update existing result
                all_results[result_id]['sparse_score'] = result.get('score', 0.0)
                all_results[result_id]['sparse_rank'] = rank
                all_results[result_id]['rrf_score'] += rrf_score
                all_results[result_id]['source'] = 'hybrid'
            else:
                # Add new result
                all_results[result_id] = {
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'dense_score': 0.0,
                    'sparse_score': result.get('score', 0.0),
                    'dense_rank': float('inf'),
                    'sparse_rank': rank,
                    'rrf_score': rrf_score,
                    'source': 'sparse'
                }
        
        # Convert to HybridSearchResult objects and sort by RRF score
        hybrid_results = []
        for rank, (result_id, result_data) in enumerate(
            sorted(all_results.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
        ):
            hybrid_result = HybridSearchResult(
                content=result_data['content'],
                metadata=result_data['metadata'],
                dense_score=result_data['dense_score'],
                sparse_score=result_data['sparse_score'],
                hybrid_score=result_data['rrf_score'],
                source=result_data['source'],
                chunk_id=result_id,
                rank=rank
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results

class DenseRetriever:
    """Dense retrieval using vector embeddings"""
    
    def __init__(self, vector_db_path: str, embedding_model: str = "openai"):
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self._initialize_vector_db()
        
    def _initialize_vector_db(self):
        """Initialize ChromaDB connection"""
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
            
            self.collection = self.client.get_or_create_collection(
                name="hybrid_knowledge_base",
                embedding_function=self.embedding_fn
            )
            
        except Exception as e:
            logging.error(f"Dense retriever initialization failed: {e}")
            self.client = None
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform dense vector search"""
        
        if not self.client:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    score = 1 - distance  # Convert distance to similarity
                    search_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': score,
                        'chunk_id': metadata.get('chunk_id', f'dense_{i}'),
                        'source': 'dense'
                    })
            
            return search_results
            
        except Exception as e:
            logging.error(f"Dense search failed: {e}")
            return []

class SparseRetriever:
    """Sparse retrieval using TF-IDF"""
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
        self.document_texts = []
        self.document_metadata = []
        self.tfidf_matrix = None
        self.is_fitted = False
        
    def index_documents(self, documents: List[Dict]):
        """Index documents for sparse retrieval"""
        
        self.document_texts = [doc['content'] for doc in documents]
        self.document_metadata = [doc.get('metadata', {}) for doc in documents]
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)
            self.is_fitted = True
            logging.info(f"Sparse retriever indexed {len(documents)} documents")
        except Exception as e:
            logging.error(f"Sparse indexing failed: {e}")
            self.is_fitted = False
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform sparse TF-IDF search"""
        
        if not self.is_fitted:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            search_results = []
            for rank, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include relevant results
                    search_results.append({
                        'content': self.document_texts[idx],
                        'metadata': self.document_metadata[idx],
                        'score': similarities[idx],
                        'chunk_id': self.document_metadata[idx].get('chunk_id', f'sparse_{idx}'),
                        'source': 'sparse'
                    })
            
            return search_results
            
        except Exception as e:
            logging.error(f"Sparse search failed: {e}")
            return []
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        
        if not self.is_fitted:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        return {
            'vocabulary_size': len(feature_names),
            'total_documents': len(self.document_texts),
            'avg_document_length': np.mean([len(doc.split()) for doc in self.document_texts]),
            'sparsity': (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100
        }

class HybridRetriever:
    """Main hybrid retriever combining dense and sparse search"""
    
    def __init__(self, 
                 vector_db_path: str = "./vector_db",
                 embedding_model: str = "openai",
                 max_sparse_features: int = 10000):
        
        self.dense_retriever = DenseRetriever(vector_db_path, embedding_model)
        self.sparse_retriever = SparseRetriever(max_sparse_features)
        self.rrf_fusion = ReciprocalRankFusion()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'dense_searches': 0,
            'sparse_searches': 0,
            'hybrid_searches': 0,
            'avg_dense_time': 0.0,
            'avg_sparse_time': 0.0
        }
        
        logging.info("Hybrid Retriever initialized")
    
    def index_documents(self, documents: List[Dict]):
        """Index documents for both dense and sparse retrieval"""
        
        # Index for sparse retrieval
        self.sparse_retriever.index_documents(documents)
        
        # Dense indexing happens through ChromaDB collection.add()
        # This should be called separately for the dense retriever
        
        logging.info(f"Hybrid indexing completed for {len(documents)} documents")
    
    def search(self, 
              query: str, 
              k: int = 10,
              search_mode: str = "hybrid",
              dense_weight: float = 0.7,
              sparse_weight: float = 0.3,
              min_sparse_score: float = 0.1) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            k: Number of results to return
            search_mode: "hybrid", "dense", "sparse", or "adaptive"
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            min_sparse_score: Minimum score threshold for sparse results
        """
        
        import time
        start_time = time.time()
        
        self.search_stats['total_searches'] += 1
        
        if search_mode == "dense":
            return self._dense_only_search(query, k)
        elif search_mode == "sparse":
            return self._sparse_only_search(query, k)
        elif search_mode == "adaptive":
            return self._adaptive_search(query, k)
        else:  # hybrid
            return self._hybrid_search(query, k, dense_weight, sparse_weight, min_sparse_score)
    
    def _hybrid_search(self, 
                      query: str, 
                      k: int,
                      dense_weight: float,
                      sparse_weight: float,
                      min_sparse_score: float) -> List[HybridSearchResult]:
        """Perform full hybrid search"""
        
        import time
        
        # Dense search
        dense_start = time.time()
        dense_results = self.dense_retriever.search(query, k * 2)  # Get more for fusion
        dense_time = time.time() - dense_start
        
        # Sparse search
        sparse_start = time.time()
        sparse_results = self.sparse_retriever.search(query, k * 2)
        sparse_time = time.time() - sparse_start
        
        # Filter sparse results by minimum score
        sparse_results = [r for r in sparse_results if r['score'] >= min_sparse_score]
        
        # Fuse results
        hybrid_results = self.rrf_fusion.fuse_results(
            dense_results, sparse_results, dense_weight, sparse_weight
        )
        
        # Update stats
        self.search_stats['hybrid_searches'] += 1
        self.search_stats['avg_dense_time'] = (
            (self.search_stats['avg_dense_time'] * (self.search_stats['hybrid_searches'] - 1) + dense_time) 
            / self.search_stats['hybrid_searches']
        )
        self.search_stats['avg_sparse_time'] = (
            (self.search_stats['avg_sparse_time'] * (self.search_stats['hybrid_searches'] - 1) + sparse_time) 
            / self.search_stats['hybrid_searches']
        )
        
        return hybrid_results[:k]
    
    def _dense_only_search(self, query: str, k: int) -> List[HybridSearchResult]:
        """Perform dense-only search"""
        
        dense_results = self.dense_retriever.search(query, k)
        self.search_stats['dense_searches'] += 1
        
        hybrid_results = []
        for rank, result in enumerate(dense_results):
            hybrid_results.append(HybridSearchResult(
                content=result['content'],
                metadata=result['metadata'],
                dense_score=result['score'],
                sparse_score=0.0,
                hybrid_score=result['score'],
                source='dense_only',
                chunk_id=result['chunk_id'],
                rank=rank
            ))
        
        return hybrid_results
    
    def _sparse_only_search(self, query: str, k: int) -> List[HybridSearchResult]:
        """Perform sparse-only search"""
        
        sparse_results = self.sparse_retriever.search(query, k)
        self.search_stats['sparse_searches'] += 1
        
        hybrid_results = []
        for rank, result in enumerate(sparse_results):
            hybrid_results.append(HybridSearchResult(
                content=result['content'],
                metadata=result['metadata'],
                dense_score=0.0,
                sparse_score=result['score'],
                hybrid_score=result['score'],
                source='sparse_only',
                chunk_id=result['chunk_id'],
                rank=rank
            ))
        
        return hybrid_results
    
    def _adaptive_search(self, query: str, k: int) -> List[HybridSearchResult]:
        """Adaptive search that chooses best strategy based on query characteristics"""
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(query)
        
        if query_analysis['is_keyword_heavy']:
            # Use sparse for keyword-heavy queries
            return self._sparse_only_search(query, k)
        elif query_analysis['is_semantic_heavy']:
            # Use dense for semantic queries
            return self._dense_only_search(query, k)
        else:
            # Use hybrid for balanced queries
            return self._hybrid_search(query, k, 0.6, 0.4, 0.05)
    
    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query characteristics for adaptive search"""
        
        words = query.lower().split()
        
        # Check for keyword indicators
        keyword_indicators = ['error', 'code', 'number', 'id', 'name', 'type']
        keyword_score = sum(1 for word in words if word in keyword_indicators) / len(words)
        
        # Check for semantic indicators
        semantic_indicators = ['how', 'why', 'what', 'when', 'where', 'explain', 'understand']
        semantic_score = sum(1 for word in words if word in semantic_indicators) / len(words)
        
        return {
            'is_keyword_heavy': keyword_score > 0.3,
            'is_semantic_heavy': semantic_score > 0.2,
            'query_length': len(words),
            'keyword_score': keyword_score,
            'semantic_score': semantic_score
        }
    
    def optimize_weights(self, evaluation_queries: List[Dict]) -> Tuple[float, float]:
        """Optimize dense/sparse weights based on evaluation queries"""
        
        best_weights = (0.7, 0.3)
        best_score = 0.0
        
        # Test different weight combinations
        weight_combinations = [
            (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)
        ]
        
        for dense_w, sparse_w in weight_combinations:
            total_score = 0.0
            
            for eval_query in evaluation_queries:
                results = self._hybrid_search(
                    eval_query['query'], 
                    k=10, 
                    dense_weight=dense_w,
                    sparse_weight=sparse_w,
                    min_sparse_score=0.1
                )
                
                # Calculate relevance score (simplified)
                relevance_score = self._calculate_relevance_score(
                    results, eval_query.get('expected_results', [])
                )
                total_score += relevance_score
            
            avg_score = total_score / len(evaluation_queries)
            if avg_score > best_score:
                best_score = avg_score
                best_weights = (dense_w, sparse_w)
        
        logging.info(f"Optimized weights: dense={best_weights[0]}, sparse={best_weights[1]}, score={best_score}")
        return best_weights
    
    def _calculate_relevance_score(self, results: List[HybridSearchResult], expected: List[str]) -> float:
        """Calculate relevance score for evaluation"""
        
        if not expected:
            return 0.0
        
        # Simple relevance calculation based on content overlap
        relevant_count = 0
        for result in results[:5]:  # Top 5 results
            for expected_content in expected:
                if expected_content.lower() in result.content.lower():
                    relevant_count += 1
                    break
        
        return relevant_count / min(5, len(expected))
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        
        stats = self.search_stats.copy()
        
        # Add sparse retriever stats
        if self.sparse_retriever.is_fitted:
            stats['sparse_vocab_stats'] = self.sparse_retriever.get_vocabulary_stats()
        
        # Add performance metrics
        if stats['total_searches'] > 0:
            stats['hybrid_percentage'] = (stats['hybrid_searches'] / stats['total_searches']) * 100
            stats['dense_percentage'] = (stats['dense_searches'] / stats['total_searches']) * 100
            stats['sparse_percentage'] = (stats['sparse_searches'] / stats['total_searches']) * 100
        
        return stats
    
    def benchmark_search_modes(self, test_queries: List[str], k: int = 10) -> Dict[str, Any]:
        """Benchmark different search modes"""
        
        import time
        
        benchmark_results = {
            'dense': {'times': [], 'result_counts': []},
            'sparse': {'times': [], 'result_counts': []},
            'hybrid': {'times': [], 'result_counts': []}
        }
        
        for query in test_queries:
            # Test dense search
            start_time = time.time()
            dense_results = self._dense_only_search(query, k)
            dense_time = time.time() - start_time
            benchmark_results['dense']['times'].append(dense_time)
            benchmark_results['dense']['result_counts'].append(len(dense_results))
            
            # Test sparse search
            start_time = time.time()
            sparse_results = self._sparse_only_search(query, k)
            sparse_time = time.time() - start_time
            benchmark_results['sparse']['times'].append(sparse_time)
            benchmark_results['sparse']['result_counts'].append(len(sparse_results))
            
            # Test hybrid search
            start_time = time.time()
            hybrid_results = self._hybrid_search(query, k, 0.7, 0.3, 0.1)
            hybrid_time = time.time() - start_time
            benchmark_results['hybrid']['times'].append(hybrid_time)
            benchmark_results['hybrid']['result_counts'].append(len(hybrid_results))
        
        # Calculate averages
        summary = {}
        for mode, results in benchmark_results.items():
            summary[mode] = {
                'avg_time': np.mean(results['times']),
                'avg_results': np.mean(results['result_counts']),
                'min_time': np.min(results['times']),
                'max_time': np.max(results['times'])
            }
        
        return {
            'summary': summary,
            'detailed_results': benchmark_results,
            'test_query_count': len(test_queries)
        }
