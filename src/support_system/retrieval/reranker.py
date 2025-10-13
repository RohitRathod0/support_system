import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from collections import defaultdict

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

@dataclass
class RerankedResult:
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    final_score: float
    rerank_factors: Dict[str, float]
    rank_change: int
    chunk_id: str

class CrossEncoderReranker:
    """Cross-encoder based reranking for high accuracy"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.cross_encoder = CrossEncoder(model_name)
            self.available = True
            logging.info(f"CrossEncoder loaded: {model_name}")
        except Exception as e:
            logging.warning(f"CrossEncoder not available: {e}")
            self.available = False
    
    def rerank(self, query: str, results: List[Dict]) -> List[RerankedResult]:
        """Rerank results using cross-encoder"""
        
        if not self.available or not results:
            return self._fallback_rerank(query, results)
        
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for result in results:
                pairs.append([query, result.get('content', '')])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Create reranked results
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, scores)):
                reranked_result = RerankedResult(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    original_score=result.get('score', 0.0),
                    rerank_score=float(score),
                    final_score=float(score),
                    rerank_factors={'cross_encoder': float(score)},
                    rank_change=0,  # Will be calculated after sorting
                    chunk_id=result.get('chunk_id', f'rerank_{i}')
                )
                reranked_results.append(reranked_result)
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Calculate rank changes
            for new_rank, result in enumerate(reranked_results):
                original_rank = next(i for i, r in enumerate(results) 
                                   if r.get('chunk_id') == result.chunk_id)
                result.rank_change = original_rank - new_rank
            
            return reranked_results
            
        except Exception as e:
            logging.error(f"Cross-encoder reranking failed: {e}")
            return self._fallback_rerank(query, results)
    
    def _fallback_rerank(self, query: str, results: List[Dict]) -> List[RerankedResult]:
        """Fallback reranking when cross-encoder is not available"""
        
        fallback_reranker = FallbackReranker()
        return fallback_reranker.rerank(query, results)

class FallbackReranker:
    """Fallback reranker using traditional methods"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def rerank(self, query: str, results: List[Dict]) -> List[RerankedResult]:
        """Rerank using multiple fallback methods"""
        
        reranked_results = []
        query_terms = set(query.lower().split())
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            
            # Calculate various scoring factors
            factors = {}
            
            # Term overlap score
            content_terms = set(content.lower().split())
            factors['term_overlap'] = len(query_terms.intersection(content_terms)) / len(query_terms)
            
            # Position in content score (earlier mentions get higher scores)
            factors['position_score'] = self._calculate_position_score(query, content)
            
            # Length normalization (prefer comprehensive but not too long)
            content_length = len(self.encoding.encode(content))
            factors['length_score'] = self._calculate_length_score(content_length)
            
            # Completeness score (check if content seems complete)
            factors['completeness'] = self._calculate_completeness_score(content)
            
            # Combine scores
            rerank_score = (
                factors['term_overlap'] * 0.4 +
                factors['position_score'] * 0.2 +
                factors['length_score'] * 0.2 +
                factors['completeness'] * 0.2
            )
            
            reranked_result = RerankedResult(
                content=content,
                metadata=result.get('metadata', {}),
                original_score=result.get('score', 0.0),
                rerank_score=rerank_score,
                final_score=rerank_score,
                rerank_factors=factors,
                rank_change=0,
                chunk_id=result.get('chunk_id', f'fallback_{i}')
            )
            reranked_results.append(reranked_result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Calculate rank changes
        for new_rank, result in enumerate(reranked_results):
            original_rank = next(i for i, r in enumerate(results) 
                               if r.get('chunk_id') == result.chunk_id)
            result.rank_change = original_rank - new_rank
        
        return reranked_results
    
    def _calculate_position_score(self, query: str, content: str) -> float:
        """Calculate score based on query term positions in content"""
        
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        if not query_terms:
            return 0.0
        
        total_score = 0.0
        content_length = len(content)
        
        for term in query_terms:
            position = content_lower.find(term)
            if position != -1:
                # Earlier positions get higher scores
                position_score = 1.0 - (position / content_length)
                total_score += position_score
        
        return total_score / len(query_terms)
    
    def _calculate_length_score(self, content_length: int) -> float:
        """Calculate score based on content length (optimal range)"""
        
        # Optimal range: 200-800 tokens
        if 200 <= content_length <= 800:
            return 1.0
        elif content_length < 200:
            return content_length / 200  # Penalty for too short
        else:
            return max(0.3, 800 / content_length)  # Penalty for too long
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate completeness score based on content structure"""
        
        score = 0.5  # Base score
        
        # Check for complete sentences
        if content.strip().endswith(('.', '!', '?')):
            score += 0.2
        
        # Check for structured content
        if any(indicator in content.lower() for indicator in ['step', 'first', 'then', 'finally']):
            score += 0.2
        
        # Check for explanatory content
        if any(indicator in content.lower() for indicator in ['because', 'therefore', 'however', 'additionally']):
            score += 0.1
        
        return min(1.0, score)

class DomainSpecificReranker:
    """Domain-specific reranking for support queries"""
    
    def __init__(self):
        self.domain_weights = {
            'policy': {
                'authority_keywords': ['must', 'shall', 'required', 'mandatory', 'policy'],
                'weight': 0.3
            },
            'technical': {
                'authority_keywords': ['error', 'code', 'system', 'configuration', 'api'],
                'weight': 0.25
            },
            'procedural': {
                'authority_keywords': ['step', 'process', 'procedure', 'how to', 'instructions'],
                'weight': 0.2
            }
        }
    
    def rerank_by_domain(self, query: str, results: List[RerankedResult], domain: str = None) -> List[RerankedResult]:
        """Apply domain-specific reranking"""
        
        if domain and domain in self.domain_weights:
            domain_config = self.domain_weights[domain]
            authority_keywords = domain_config['authority_keywords']
            weight = domain_config['weight']
            
            for result in results:
                content_lower = result.content.lower()
                
                # Calculate domain relevance
                keyword_matches = sum(1 for keyword in authority_keywords 
                                    if keyword in content_lower)
                domain_relevance = keyword_matches / len(authority_keywords)
                
                # Adjust final score
                result.final_score = (
                    result.rerank_score * (1 - weight) + 
                    domain_relevance * weight
                )
                result.rerank_factors['domain_relevance'] = domain_relevance
            
            # Re-sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results

class UserPersonalizedReranker:
    """Personalized reranking based on user preferences and history"""
    
    def __init__(self):
        self.user_preferences = {}
        self.interaction_history = defaultdict(list)
    
    def update_user_preferences(self, user_id: str, feedback_data: Dict[str, Any]):
        """Update user preferences based on feedback"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'preferred_detail_level': 'medium',  # low, medium, high
                'preferred_content_types': [],
                'successful_patterns': [],
                'feedback_history': []
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update detail level preference
        if 'detail_feedback' in feedback_data:
            if feedback_data['detail_feedback'] == 'too_detailed':
                prefs['preferred_detail_level'] = 'low' if prefs['preferred_detail_level'] == 'medium' else 'medium'
            elif feedback_data['detail_feedback'] == 'not_detailed_enough':
                prefs['preferred_detail_level'] = 'high' if prefs['preferred_detail_level'] == 'medium' else 'medium'
        
        # Track successful patterns
        if feedback_data.get('satisfaction_score', 0) > 0.7:
            content_pattern = {
                'length': len(feedback_data.get('content', '')),
                'content_type': feedback_data.get('content_type', 'general'),
                'query_type': feedback_data.get('query_type', 'general')
            }
            prefs['successful_patterns'].append(content_pattern)
            
            # Keep only recent patterns
            prefs['successful_patterns'] = prefs['successful_patterns'][-20:]
        
        prefs['feedback_history'].append({
            'timestamp': datetime.now().isoformat(),
            **feedback_data
        })
    
    def personalized_rerank(self, user_id: str, query: str, results: List[RerankedResult]) -> List[RerankedResult]:
        """Apply personalized reranking"""
        
        if user_id not in self.user_preferences:
            return results
        
        prefs = self.user_preferences[user_id]
        
        for result in results:
            personalization_score = 0.0
            
            # Detail level preference
            content_length = len(result.content.split())
            detail_score = self._calculate_detail_preference_score(
                content_length, prefs['preferred_detail_level']
            )
            personalization_score += detail_score * 0.4
            
            # Successful pattern matching
            pattern_score = self._calculate_pattern_score(result, prefs['successful_patterns'])
            personalization_score += pattern_score * 0.3
            
            # Content type preference
            content_type_score = self._calculate_content_type_score(result, prefs)
            personalization_score += content_type_score * 0.3
            
            # Combine with existing score
            result.final_score = (
                result.final_score * 0.7 + 
                personalization_score * 0.3
            )
            result.rerank_factors['personalization'] = personalization_score
        
        # Re-sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def _calculate_detail_preference_score(self, content_length: int, preferred_level: str) -> float:
        """Calculate score based on detail level preference"""
        
        if preferred_level == 'low':
            # Prefer shorter content
            return 1.0 if content_length < 100 else max(0.3, 100 / content_length)
        elif preferred_level == 'high':
            # Prefer longer content
            return 1.0 if content_length > 200 else content_length / 200
        else:  # medium
            # Prefer moderate length
            if 50 <= content_length <= 200:
                return 1.0
            elif content_length < 50:
                return content_length / 50
            else:
                return max(0.5, 200 / content_length)
    
    def _calculate_pattern_score(self, result: RerankedResult, successful_patterns: List[Dict]) -> float:
        """Calculate score based on successful interaction patterns"""
        
        if not successful_patterns:
            return 0.5
        
        result_length = len(result.content)
        result_type = result.metadata.get('content_type', 'general')
        
        similarity_scores = []
        
        for pattern in successful_patterns:
            # Length similarity
            length_sim = 1.0 - abs(result_length - pattern['length']) / max(result_length, pattern['length'])
            
            # Type similarity
            type_sim = 1.0 if result_type == pattern['content_type'] else 0.3
            
            # Combined similarity
            similarity_scores.append((length_sim + type_sim) / 2)
        
        return np.mean(similarity_scores) if similarity_scores else 0.5
    
    def _calculate_content_type_score(self, result: RerankedResult, prefs: Dict) -> float:
        """Calculate score based on preferred content types"""
        
        preferred_types = prefs.get('preferred_content_types', [])
        if not preferred_types:
            return 0.5
        
        result_type = result.metadata.get('content_type', 'general')
        return 1.0 if result_type in preferred_types else 0.3

class AdvancedReranker:
    """Main reranking engine combining multiple strategies"""
    
    def __init__(self, use_cross_encoder: bool = True):
        self.cross_encoder_reranker = CrossEncoderReranker() if use_cross_encoder else None
        self.fallback_reranker = FallbackReranker()
        self.domain_reranker = DomainSpecificReranker()
        self.personalized_reranker = UserPersonalizedReranker()
        
        # Reranking statistics
        self.rerank_stats = {
            'total_reranks': 0,
            'cross_encoder_used': 0,
            'fallback_used': 0,
            'avg_rank_changes': 0.0,
            'domain_reranks': defaultdict(int)
        }
        
        logging.info("Advanced Reranker initialized")
    
    def rerank_results(self, 
                      query: str, 
                      results: List[Dict],
                      user_id: str = None,
                      domain: str = None,
                      rerank_strategy: str = "adaptive") -> List[RerankedResult]:
        """
        Comprehensive result reranking
        
        Args:
            query: Original search query
            results: List of search results to rerank
            user_id: User identifier for personalization
            domain: Domain context for domain-specific reranking
            rerank_strategy: "cross_encoder", "fallback", "adaptive"
        """
        
        if not results:
            return []
        
        self.rerank_stats['total_reranks'] += 1
        
        # Step 1: Initial reranking
        if rerank_strategy == "cross_encoder" and self.cross_encoder_reranker:
            reranked_results = self.cross_encoder_reranker.rerank(query, results)
            self.rerank_stats['cross_encoder_used'] += 1
        elif rerank_strategy == "fallback":
            reranked_results = self.fallback_reranker.rerank(query, results)
            self.rerank_stats['fallback_used'] += 1
        else:  # adaptive
            if self.cross_encoder_reranker and self.cross_encoder_reranker.available and len(results) <= 20:
                reranked_results = self.cross_encoder_reranker.rerank(query, results)
                self.rerank_stats['cross_encoder_used'] += 1
            else:
                reranked_results = self.fallback_reranker.rerank(query, results)
                self.rerank_stats['fallback_used'] += 1
        
        # Step 2: Domain-specific reranking
        if domain:
            reranked_results = self.domain_reranker.rerank_by_domain(query, reranked_results, domain)
            self.rerank_stats['domain_reranks'][domain] += 1
        
        # Step 3: User personalization
        if user_id:
            reranked_results = self.personalized_reranker.personalized_rerank(user_id, query, reranked_results)
        
        # Update statistics
        rank_changes = [abs(result.rank_change) for result in reranked_results]
        self.rerank_stats['avg_rank_changes'] = (
            (self.rerank_stats['avg_rank_changes'] * (self.rerank_stats['total_reranks'] - 1) + 
             np.mean(rank_changes)) / self.rerank_stats['total_reranks']
        )
        
        return reranked_results
    
    def update_user_feedback(self, user_id: str, query: str, results: List[RerankedResult], feedback: Dict[str, Any]):
        """Update reranker with user feedback"""
        
        # Update personalized reranker
        self.personalized_reranker.update_user_preferences(user_id, feedback)
        
        # Log feedback for analysis
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'query': query,
            'result_count': len(results),
            'feedback': feedback
        }
        
        # Store feedback (in production, this would go to a database)
        logging.info(f"Reranker feedback: {json.dumps(feedback_entry)}")
    
    def analyze_reranking_effectiveness(self, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Analyze reranking effectiveness using evaluation data"""
        
        effectiveness_metrics = {
            'precision_improvements': [],
            'ndcg_improvements': [],
            'rank_changes_beneficial': 0,
            'rank_changes_detrimental': 0
        }
        
        for eval_item in evaluation_data:
            query = eval_item['query']
            original_results = eval_item['original_results']
            expected_relevant = eval_item.get('relevant_ids', [])
            
            # Rerank results
            reranked_results = self.rerank_results(
                query, original_results, 
                rerank_strategy="adaptive"
            )
            
            # Calculate improvements
            original_precision = self._calculate_precision_at_k(original_results, expected_relevant, k=5)
            reranked_precision = self._calculate_precision_at_k(reranked_results, expected_relevant, k=5)
            
            effectiveness_metrics['precision_improvements'].append(reranked_precision - original_precision)
            
            # Analyze rank changes
            for result in reranked_results:
                if result.chunk_id in expected_relevant:
                    if result.rank_change > 0:  # Moved up
                        effectiveness_metrics['rank_changes_beneficial'] += 1
                    elif result.rank_change < 0:  # Moved down
                        effectiveness_metrics['rank_changes_detrimental'] += 1
        
        # Calculate summary statistics
        summary = {
            'avg_precision_improvement': np.mean(effectiveness_metrics['precision_improvements']),
            'precision_improvement_std': np.std(effectiveness_metrics['precision_improvements']),
            'beneficial_changes': effectiveness_metrics['rank_changes_beneficial'],
            'detrimental_changes': effectiveness_metrics['rank_changes_detrimental'],
            'net_benefit_ratio': (
                effectiveness_metrics['rank_changes_beneficial'] / 
                max(1, effectiveness_metrics['rank_changes_beneficial'] + effectiveness_metrics['rank_changes_detrimental'])
            ),
            'rerank_stats': self.rerank_stats
        }
        
        return summary
    
    def _calculate_precision_at_k(self, results: List[Any], relevant_ids: List[str], k: int) -> float:
        """Calculate precision@k metric"""
        
        if not results or not relevant_ids:
            return 0.0
        
        top_k_results = results[:k]
        relevant_in_top_k = 0
        
        for result in top_k_results:
            result_id = getattr(result, 'chunk_id', None) or result.get('chunk_id')
            if result_id in relevant_ids:
                relevant_in_top_k += 1
        
        return relevant_in_top_k / min(k, len(results))
    
    def get_reranking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reranking statistics"""
        
        stats = self.rerank_stats.copy()
        
        # Add performance ratios
        if stats['total_reranks'] > 0:
            stats['cross_encoder_ratio'] = stats['cross_encoder_used'] / stats['total_reranks']
            stats['fallback_ratio'] = stats['fallback_used'] / stats['total_reranks']
        
        # Add user personalization stats
        stats['users_with_preferences'] = len(self.personalized_reranker.user_preferences)
        
        return stats
    
    def optimize_reranking_weights(self, training_data: List[Dict]) -> Dict[str, float]:
        """Optimize reranking weights using training data"""
        
        # This is a simplified optimization - in production, use more sophisticated methods
        best_weights = {'cross_encoder': 0.7, 'domain': 0.2, 'personalization': 0.1}
        best_score = 0.0
        
        weight_combinations = [
            (0.8, 0.15, 0.05),
            (0.7, 0.2, 0.1),
            (0.6, 0.25, 0.15),
            (0.5, 0.3, 0.2)
        ]
        
        for ce_weight, domain_weight, pers_weight in weight_combinations:
            total_score = 0.0
            
            for data_item in training_data:
                # Simulate reranking with these weights
                # This would need actual implementation based on your specific needs
                score = self._evaluate_weight_combination(data_item, ce_weight, domain_weight, pers_weight)
                total_score += score
            
            avg_score = total_score / len(training_data)
            if avg_score > best_score:
                best_score = avg_score
                best_weights = {
                    'cross_encoder': ce_weight,
                    'domain': domain_weight,
                    'personalization': pers_weight
                }
        
        logging.info(f"Optimized reranking weights: {best_weights}, score: {best_score}")
        return best_weights
    
    def _evaluate_weight_combination(self, data_item: Dict, ce_weight: float, domain_weight: float, pers_weight: float) -> float:
        """Evaluate a specific weight combination"""
        
        # Simplified evaluation - implement based on your specific metrics
        query = data_item['query']
        results = data_item['results']
        expected_relevant = data_item.get('relevant_ids', [])
        
        reranked_results = self.rerank_results(query, results)
        precision = self._calculate_precision_at_k(reranked_results, expected_relevant, k=5)
        
        return precision
