import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
from langchain.schema import Document
import tiktoken

@dataclass
class SemanticChunk:
    content: str
    chunk_type: str
    metadata: Dict[str, Any]
    embedding: np.ndarray
    semantic_boundaries: List[int]
    coherence_score: float

class SemanticChunker:
    """Advanced semantic-aware text chunking that preserves meaning and context"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.75,
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Load spacy model for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found, using basic sentence splitting")
            self.nlp = None
        
        # Token encoder for chunk size management
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        logging.info(f"Semantic Chunker initialized with model: {embedding_model}")
    
    def chunk_document(self, 
                      document: Document, 
                      chunk_strategy: str = "semantic_similarity") -> List[SemanticChunk]:
        """
        Chunk document using semantic understanding
        
        Args:
            document: Input document to chunk
            chunk_strategy: Strategy to use ('semantic_similarity', 'topic_coherence', 'hybrid')
        """
        
        if chunk_strategy == "semantic_similarity":
            return self._chunk_by_semantic_similarity(document)
        elif chunk_strategy == "topic_coherence":
            return self._chunk_by_topic_coherence(document)
        elif chunk_strategy == "hybrid":
            return self._chunk_hybrid_approach(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {chunk_strategy}")
    
    def _chunk_by_semantic_similarity(self, document: Document) -> List[SemanticChunk]:
        """Chunk based on semantic similarity between sentences"""
        
        # Extract sentences
        sentences = self._extract_sentences(document.page_content)
        if len(sentences) < 2:
            return [self._create_single_chunk(document)]
        
        # Generate embeddings for sentences
        sentence_embeddings = self.embedding_model.encode(sentences)
        
        # Group semantically similar sentences
        chunks = []
        current_chunk_sentences = []
        current_chunk_embedding = None
        current_token_count = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_chunk_embedding is None:
                # Start first chunk
                current_chunk_sentences.append(sentence)
                current_chunk_embedding = embedding
                current_token_count = sentence_tokens
            else:
                # Check semantic similarity
                similarity = cosine_similarity([current_chunk_embedding], [embedding])[0][0]
                
                # Check if adding sentence would exceed size limit
                would_exceed_limit = (current_token_count + sentence_tokens) > self.max_chunk_size
                
                if similarity >= self.similarity_threshold and not would_exceed_limit:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_token_count += sentence_tokens
                    
                    # Update chunk embedding (weighted average)
                    weight = len(current_chunk_sentences)
                    current_chunk_embedding = (
                        (current_chunk_embedding * (weight - 1) + embedding) / weight
                    )
                else:
                    # Finalize current chunk and start new one
                    if current_token_count >= self.min_chunk_size:
                        chunk = self._create_semantic_chunk(
                            current_chunk_sentences,
                            current_chunk_embedding,
                            document.metadata,
                            i
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk_sentences = [sentence]
                    current_chunk_embedding = embedding
                    current_token_count = sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences and current_token_count >= self.min_chunk_size:
            chunk = self._create_semantic_chunk(
                current_chunk_sentences,
                current_chunk_embedding,
                document.metadata,
                len(sentences)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_topic_coherence(self, document: Document) -> List[SemanticChunk]:
        """Chunk based on topic coherence using clustering"""
        
        sentences = self._extract_sentences(document.page_content)
        if len(sentences) < 3:
            return [self._create_single_chunk(document)]
        
        # Generate embeddings
        sentence_embeddings = self.embedding_model.encode(sentences)
        
        # Determine optimal number of clusters
        optimal_clusters = min(max(2, len(sentences) // 5), 10)  # 2-10 clusters
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(sentence_embeddings)
        
        # Group sentences by cluster
        clusters = {}
        for i, (sentence, label) in enumerate(zip(sentences, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, sentence))
        
        # Create chunks from clusters
        chunks = []
        for cluster_id, cluster_sentences in clusters.items():
            # Sort sentences by original order
            cluster_sentences.sort(key=lambda x: x[0])
            sentences_text = [s[1] for s in cluster_sentences]
            
            # Check if cluster is large enough
            cluster_text = " ".join(sentences_text)
            if len(self.encoding.encode(cluster_text)) >= self.min_chunk_size:
                
                # Calculate cluster centroid
                cluster_indices = [s[0] for s in cluster_sentences]
                cluster_embeddings = sentence_embeddings[cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                chunk = SemanticChunk(
                    content=cluster_text,
                    chunk_type="topic_coherent",
                    metadata={
                        **document.metadata,
                        "cluster_id": cluster_id,
                        "sentence_count": len(sentences_text),
                        "coherence_method": "clustering"
                    },
                    embedding=centroid,
                    semantic_boundaries=cluster_indices,
                    coherence_score=self._calculate_cluster_coherence(cluster_embeddings)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid_approach(self, document: Document) -> List[SemanticChunk]:
        """Hybrid approach combining similarity and topic coherence"""
        
        # First, try semantic similarity chunking
        semantic_chunks = self._chunk_by_semantic_similarity(document)
        
        # If chunks are too large or too few, try topic coherence
        if (len(semantic_chunks) < 2 or 
            any(len(self.encoding.encode(chunk.content)) > self.max_chunk_size * 1.2 
                for chunk in semantic_chunks)):
            
            topic_chunks = self._chunk_by_topic_coherence(document)
            
            # Choose better chunking based on coherence scores
            semantic_avg_coherence = np.mean([chunk.coherence_score for chunk in semantic_chunks])
            topic_avg_coherence = np.mean([chunk.coherence_score for chunk in topic_chunks])
            
            if topic_avg_coherence > semantic_avg_coherence:
                return topic_chunks
        
        return semantic_chunks
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using spaCy or basic splitting"""
        
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            # Basic sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _create_semantic_chunk(self, 
                              sentences: List[str], 
                              embedding: np.ndarray, 
                              base_metadata: Dict,
                              boundary_index: int) -> SemanticChunk:
        """Create a semantic chunk from sentences"""
        
        content = " ".join(sentences)
        
        # Calculate coherence score
        if len(sentences) > 1:
            sentence_embeddings = self.embedding_model.encode(sentences)
            coherence_score = self._calculate_coherence_score(sentence_embeddings)
        else:
            coherence_score = 1.0
        
        return SemanticChunk(
            content=content,
            chunk_type="semantic_similarity",
            metadata={
                **base_metadata,
                "sentence_count": len(sentences),
                "token_count": len(self.encoding.encode(content)),
                "boundary_index": boundary_index
            },
            embedding=embedding,
            semantic_boundaries=[boundary_index],
            coherence_score=coherence_score
        )
    
    def _create_single_chunk(self, document: Document) -> List[SemanticChunk]:
        """Create single chunk when document is too small to split"""
        
        embedding = self.embedding_model.encode([document.page_content])[0]
        
        chunk = SemanticChunk(
            content=document.page_content,
            chunk_type="single_document",
            metadata={
                **document.metadata,
                "sentence_count": 1,
                "token_count": len(self.encoding.encode(document.page_content))
            },
            embedding=embedding,
            semantic_boundaries=[0],
            coherence_score=1.0
        )
        
        return [chunk]
    
    def _calculate_coherence_score(self, embeddings: np.ndarray) -> float:
        """Calculate coherence score for a group of embeddings"""
        
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarities)
        mask = np.ones(similarities.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Return average similarity
        return np.mean(similarities[mask])
    
    def _calculate_cluster_coherence(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate coherence score for a cluster"""
        
        if len(cluster_embeddings) < 2:
            return 1.0
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate average distance to centroid
        distances = [cosine_similarity([embedding], [centroid])[0][0] 
                    for embedding in cluster_embeddings]
        
        return np.mean(distances)
    
    def optimize_chunk_boundaries(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Optimize chunk boundaries for better coherence"""
        
        if len(chunks) < 2:
            return chunks
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check if chunk should be merged with next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Calculate potential merged chunk properties
                merged_content = chunk.content + " " + next_chunk.content
                merged_tokens = len(self.encoding.encode(merged_content))
                
                # Check if merging improves coherence and stays within size limits
                if (merged_tokens <= self.max_chunk_size and 
                    self._should_merge_chunks(chunk, next_chunk)):
                    
                    # Merge chunks
                    merged_embedding = (chunk.embedding + next_chunk.embedding) / 2
                    merged_chunk = SemanticChunk(
                        content=merged_content,
                        chunk_type="optimized_merge",
                        metadata={
                            **chunk.metadata,
                            "merged_from": [chunk.metadata.get("boundary_index"), 
                                          next_chunk.metadata.get("boundary_index")],
                            "token_count": merged_tokens
                        },
                        embedding=merged_embedding,
                        semantic_boundaries=chunk.semantic_boundaries + next_chunk.semantic_boundaries,
                        coherence_score=(chunk.coherence_score + next_chunk.coherence_score) / 2
                    )
                    
                    optimized_chunks.append(merged_chunk)
                    chunks[i + 1] = None  # Mark next chunk as merged
                else:
                    optimized_chunks.append(chunk)
            elif chunk is not None:  # Not merged
                optimized_chunks.append(chunk)
        
        return [chunk for chunk in optimized_chunks if chunk is not None]
    
    def _should_merge_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """Determine if two chunks should be merged"""
        
        # Calculate similarity between chunks
        similarity = cosine_similarity([chunk1.embedding], [chunk2.embedding])[0][0]
        
        # Merge if highly similar and both have good coherence
        return (similarity > self.similarity_threshold and 
                chunk1.coherence_score > 0.6 and 
                chunk2.coherence_score > 0.6)
    
    def get_chunk_statistics(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get statistics about generated chunks"""
        
        if not chunks:
            return {}
        
        token_counts = [chunk.metadata.get("token_count", 0) for chunk in chunks]
        coherence_scores = [chunk.coherence_score for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": np.mean(token_counts),
            "min_chunk_size": np.min(token_counts),
            "max_chunk_size": np.max(token_counts),
            "avg_coherence": np.mean(coherence_scores),
            "chunk_types": {chunk.chunk_type: 1 for chunk in chunks},
            "total_tokens": sum(token_counts)
        }
