import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
from sklearn.cluster import KMeans

@dataclass
class QueryPattern:
    query_type: str
    frequency: int
    avg_response_time: float
    avg_satisfaction: float
    common_failures: List[str]
    successful_strategies: List[str]

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    query_type: str
    response_time: float
    retrieval_time: float
    satisfaction_score: float
    context_tokens_used: int
    results_relevance: float
    user_expertise: str

class RealTimePerformanceOptimizer:
    """Continuous system optimization based on usage patterns"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.auto_tuner = AutoTuner()
        self.cache_manager = IntelligentCacheManager()
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.optimization_log = []
        
        # Performance thresholds
        self.response_time_threshold = 3.0  # seconds
        self.satisfaction_threshold = 0.75
        self.relevance_threshold = 0.70
        
        logging.info("Performance Optimizer initialized")
    
    def record_interaction_metrics(self, metrics: PerformanceMetrics):
        """Record interaction metrics for analysis"""
        self.metrics_history.append(metrics)
        
        # Real-time optimization triggers
        if len(self.metrics_history) % 100 == 0:  # Every 100 interactions
            self._trigger_optimization_analysis()
    
    def _trigger_optimization_analysis(self):
        """Trigger real-time optimization based on recent metrics"""
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 interactions
        
        # Analyze performance patterns
        performance_analysis = self._analyze_performance_patterns(recent_metrics)
        
        # Apply optimizations
        optimizations_applied = []
        
        if performance_analysis['avg_response_time'] > self.response_time_threshold:
            optimizations_applied.extend(self._optimize_response_time(recent_metrics))
        
        if performance_analysis['avg_satisfaction'] < self.satisfaction_threshold:
            optimizations_applied.extend(self._optimize_satisfaction(recent_metrics))
        
        if performance_analysis['avg_relevance'] < self.relevance_threshold:
            optimizations_applied.extend(self._optimize_relevance(recent_metrics))
        
        # Log optimizations
        if optimizations_applied:
            self.optimization_log.append({
                'timestamp': datetime.now().isoformat(),
                'optimizations': optimizations_applied,
                'performance_before': performance_analysis
            })
            logging.info(f"Applied optimizations: {optimizations_applied}")
    
    def _analyze_performance_patterns(self, metrics: List[PerformanceMetrics]) -> Dict:
        """Analyze performance patterns in recent metrics"""
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        satisfaction_scores = [m.satisfaction_score for m in metrics if m.satisfaction_score > 0]
        relevance_scores = [m.results_relevance for m in metrics if m.results_relevance > 0]
        
        # Query type analysis
        query_type_performance = defaultdict(list)
        for m in metrics:
            query_type_performance[m.query_type].append(m)
        
        return {
            'avg_response_time': np.mean(response_times),
            'avg_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0,
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'total_interactions': len(metrics),
            'query_type_breakdown': {
                qtype: {
                    'count': len(qmetrics),
                    'avg_response_time': np.mean([m.response_time for m in qmetrics]),
                    'avg_satisfaction': np.mean([m.satisfaction_score for m in qmetrics if m.satisfaction_score > 0])
                }
                for qtype, qmetrics in query_type_performance.items()
            }
        }
    
    def _optimize_response_time(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Optimize system for faster response times"""
        optimizations = []
        
        # Identify slow query types
        slow_queries = {}
        for metric in metrics:
            if metric.response_time > self.response_time_threshold:
                if metric.query_type not in slow_queries:
                    slow_queries[metric.query_type] = []
                slow_queries[metric.query_type].append(metric)
        
        for query_type, slow_metrics in slow_queries.items():
            # Analyze bottlenecks
            avg_retrieval_time = np.mean([m.retrieval_time for m in slow_metrics])
            avg_context_tokens = np.mean([m.context_tokens_used for m in slow_metrics])
            
            # Apply specific optimizations
            if avg_retrieval_time > 1.0:  # Retrieval is slow
                self.auto_tuner.reduce_retrieval_k_for_query_type(query_type)
                optimizations.append(f"reduced_retrieval_k_{query_type}")
            
            if avg_context_tokens > 3000:  # Too much context
                self.auto_tuner.reduce_context_window_for_query_type(query_type)
                optimizations.append(f"reduced_context_window_{query_type}")
        
        return optimizations
    
    def _optimize_satisfaction(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Optimize for better user satisfaction"""
        optimizations = []
        
        # Analyze satisfaction by user expertise
        expertise_satisfaction = defaultdict(list)
        for metric in metrics:
            if metric.satisfaction_score > 0:
                expertise_satisfaction[metric.user_expertise].append(metric.satisfaction_score)
        
        for expertise, scores in expertise_satisfaction.items():
            avg_satisfaction = np.mean(scores)
            if avg_satisfaction < self.satisfaction_threshold:
                # Adjust response style for this expertise level
                if expertise == 'beginner' and avg_satisfaction < 0.7:
                    self.auto_tuner.increase_explanation_detail_for_beginners()
                    optimizations.append("increased_detail_for_beginners")
                elif expertise == 'expert' and avg_satisfaction < 0.7:
                    self.auto_tuner.reduce_explanation_verbosity_for_experts()
                    optimizations.append("reduced_verbosity_for_experts")
        
        return optimizations
    
    def _optimize_relevance(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Optimize retrieval relevance"""
        optimizations = []
        
        # Find low relevance patterns
        low_relevance_metrics = [m for m in metrics if m.results_relevance < self.relevance_threshold]
        
        if len(low_relevance_metrics) > len(metrics) * 0.3:  # More than 30% low relevance
            # Increase retrieval diversity
            self.auto_tuner.increase_retrieval_diversity()
            optimizations.append("increased_retrieval_diversity")
            
            # Enable hybrid search if not already enabled
            self.auto_tuner.enable_hybrid_search()
            optimizations.append("enabled_hybrid_search")
        
        return optimizations
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations"""
        if len(self.metrics_history) < 50:
            return {"message": "Insufficient data for recommendations"}
        
        recent_metrics = list(self.metrics_history)[-200:]  # Last 200 interactions
        analysis = self._analyze_performance_patterns(recent_metrics)
        
        recommendations = []
        
        # Response time recommendations
        if analysis['avg_response_time'] > self.response_time_threshold:
            recommendations.append({
                'type': 'response_time',
                'issue': f"Average response time is {analysis['avg_response_time']:.2f}s",
                'recommendation': "Consider reducing retrieval scope or context window size"
            })
        
        # Satisfaction recommendations
        if analysis['avg_satisfaction'] < self.satisfaction_threshold:
            recommendations.append({
                'type': 'satisfaction',
                'issue': f"Average satisfaction is {analysis['avg_satisfaction']:.2f}",
                'recommendation': "Review response quality and personalization settings"
            })
        
        # Query type specific recommendations
        for qtype, qdata in analysis['query_type_breakdown'].items():
            if qdata['avg_response_time'] > self.response_time_threshold * 1.5:
                recommendations.append({
                    'type': 'query_specific',
                    'issue': f"Query type '{qtype}' is consistently slow",
                    'recommendation': f"Optimize retrieval strategy for {qtype} queries"
                })
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'recent_optimizations': self.optimization_log[-5:] if self.optimization_log else []
        }

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.active_queries = {}
        self.performance_cache = {}
    
    def start_query_timing(self, query_id: str) -> str:
        """Start timing a query"""
        self.active_queries[query_id] = {
            'start_time': time.time(),
            'stages': {}
        }
        return query_id
    
    def mark_stage(self, query_id: str, stage_name: str):
        """Mark completion of a processing stage"""
        if query_id in self.active_queries:
            self.active_queries[query_id]['stages'][stage_name] = time.time()
    
    def end_query_timing(self, query_id: str) -> Dict[str, float]:
        """End timing and return performance metrics"""
        if query_id not in self.active_queries:
            return {}
        
        query_data = self.active_queries[query_id]
        end_time = time.time()
        total_time = end_time - query_data['start_time']
        
        # Calculate stage durations
        stage_durations = {}
        prev_time = query_data['start_time']
        
        for stage, stage_time in query_data['stages'].items():
            stage_durations[stage] = stage_time - prev_time
            prev_time = stage_time
        
        # Clean up
        del self.active_queries[query_id]
        
        return {
            'total_time': total_time,
            'stage_durations': stage_durations
        }

class AutoTuner:
    """Automatic parameter tuning based on performance"""
    
    def __init__(self):
        self.current_settings = {
            'retrieval_k': 10,
            'context_window_tokens': 4000,
            'hybrid_search_enabled': True,
            'reranking_enabled': True,
            'retrieval_diversity': 0.5
        }
        self.query_type_settings = {}
    
    def reduce_retrieval_k_for_query_type(self, query_type: str):
        """Reduce retrieval results for specific query type"""
        if query_type not in self.query_type_settings:
            self.query_type_settings[query_type] = self.current_settings.copy()
        
        self.query_type_settings[query_type]['retrieval_k'] = max(5, 
            self.query_type_settings[query_type]['retrieval_k'] - 2)
        
        logging.info(f"Reduced retrieval_k for {query_type} to {self.query_type_settings[query_type]['retrieval_k']}")
    
    def reduce_context_window_for_query_type(self, query_type: str):
        """Reduce context window for specific query type"""
        if query_type not in self.query_type_settings:
            self.query_type_settings[query_type] = self.current_settings.copy()
        
        self.query_type_settings[query_type]['context_window_tokens'] = max(2000,
            self.query_type_settings[query_type]['context_window_tokens'] - 500)
        
        logging.info(f"Reduced context window for {query_type} to {self.query_type_settings[query_type]['context_window_tokens']}")
    
    def increase_explanation_detail_for_beginners(self):
        """Increase detail for beginner users"""
        if 'beginner' not in self.query_type_settings:
            self.query_type_settings['beginner'] = self.current_settings.copy()
        
        self.query_type_settings['beginner']['explanation_detail'] = 'high'
        logging.info("Increased explanation detail for beginner users")
    
    def reduce_explanation_verbosity_for_experts(self):
        """Reduce verbosity for expert users"""
        if 'expert' not in self.query_type_settings:
            self.query_type_settings['expert'] = self.current_settings.copy()
        
        self.query_type_settings['expert']['explanation_detail'] = 'concise'
        logging.info("Reduced explanation verbosity for expert users")
    
    def increase_retrieval_diversity(self):
        """Increase diversity in retrieval results"""
        self.current_settings['retrieval_diversity'] = min(1.0,
            self.current_settings['retrieval_diversity'] + 0.1)
        logging.info(f"Increased retrieval diversity to {self.current_settings['retrieval_diversity']}")
    
    def enable_hybrid_search(self):
        """Enable hybrid search if not already enabled"""
        if not self.current_settings['hybrid_search_enabled']:
            self.current_settings['hybrid_search_enabled'] = True
            logging.info("Enabled hybrid search")
    
    def get_settings_for_query(self, query_type: str, user_expertise: str) -> Dict:
        """Get optimized settings for specific query and user"""
        base_settings = self.current_settings.copy()
        
        # Apply query type specific settings
        if query_type in self.query_type_settings:
            base_settings.update(self.query_type_settings[query_type])
        
        # Apply user expertise specific settings
        if user_expertise in self.query_type_settings:
            base_settings.update(self.query_type_settings[user_expertise])
        
        return base_settings

class IntelligentCacheManager:
    """Intelligent caching based on query patterns"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.cache_stats = defaultdict(int)
        self.max_cache_size = max_cache_size
        self.access_times = {}
    
    def should_cache(self, query: str, results: List[Any]) -> bool:
        """Determine if query results should be cached"""
        cache_score = self._calculate_cache_worthiness(query, results)
        return cache_score > 0.7
    
    def _calculate_cache_worthiness(self, query: str, results: List[Any]) -> float:
        """Calculate how worthy a query is of caching"""
        score = 0.0
        
        # High-quality results are more worthy
        if results and hasattr(results[0], 'score'):
            avg_score = np.mean([r.score for r in results])
            score += avg_score * 0.4
        
        # Common query patterns are more worthy
        query_words = set(query.lower().split())
        common_words = {'help', 'how', 'what', 'when', 'where', 'policy', 'procedure'}
        if query_words.intersection(common_words):
            score += 0.3
        
        # Short queries are more likely to be repeated
        if len(query.split()) <= 5:
            score += 0.2
        
        # Frequently accessed queries
        query_hash = hash(query)
        if query_hash in self.cache_stats:
            score += min(0.3, self.cache_stats[query_hash] * 0.05)
        
        return score
    
    def cache_results(self, query_hash: str, results: List[Any], ttl: int = 3600, tags: List[str] = None):
        """Cache results with intelligent expiry"""
        if len(self.cache) >= self.max_cache_size:
            self._evict_least_useful()
        
        self.cache[query_hash] = {
            'results': results,
            'cached_at': datetime.now(),
            'ttl': ttl,
            'tags': tags or [],
            'access_count': 0
        }
        
        self.access_times[query_hash] = datetime.now()
    
    def get_cached_results(self, query_hash: str) -> Optional[List[Any]]:
        """Get cached results if still valid"""
        if query_hash not in self.cache:
            return None
        
        cache_entry = self.cache[query_hash]
        age = (datetime.now() - cache_entry['cached_at']).total_seconds()
        
        if age > cache_entry['ttl']:
            del self.cache[query_hash]
            return None
        
        # Update access stats
        cache_entry['access_count'] += 1
        self.access_times[query_hash] = datetime.now()
        self.cache_stats[query_hash] += 1
        
        return cache_entry['results']
    
    def _evict_least_useful(self):
        """Evict least useful cache entries"""
        if not self.cache:
            return
        
        # Score each cache entry by usefulness
        entry_scores = {}
        for key, entry in self.cache.items():
            age = (datetime.now() - entry['cached_at']).total_seconds()
            access_frequency = entry['access_count'] / (age / 3600)  # accesses per hour
            
            # Higher score = more useful
            score = access_frequency * 0.7 + (1 / (age / 3600)) * 0.3
            entry_scores[key] = score
        
        # Remove lowest scoring entries
        sorted_entries = sorted(entry_scores.items(), key=lambda x: x[1])
        to_remove = sorted_entries[:len(sorted_entries) // 4]  # Remove bottom 25%
        
        for key, _ in to_remove:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_accesses = sum(self.cache_stats.values())
        unique_queries = len(self.cache_stats)
        
        return {
            'cache_size': len(self.cache),
            'total_accesses': total_accesses,
            'unique_queries': unique_queries,
            'hit_rate': len(self.cache) / unique_queries if unique_queries > 0 else 0,
            'avg_accesses_per_query': total_accesses / unique_queries if unique_queries > 0 else 0
        }
