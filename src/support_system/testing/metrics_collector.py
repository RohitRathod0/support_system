import json
import logging
import sqlite3
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import threading
from contextlib import contextmanager

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

@dataclass
class Metric:
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any] = None

@dataclass
class MetricSummary:
    name: str
    metric_type: MetricType
    count: int
    sum_value: float
    avg_value: float
    min_value: float
    max_value: float
    percentiles: Dict[str, float]
    start_time: datetime
    end_time: datetime

class MetricsDatabase:
    """Database layer for metrics storage"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for metrics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT,
                metadata TEXT
            )
        ''')
        
        # Metric summaries table for aggregated data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                count INTEGER,
                sum_value REAL,
                avg_value REAL,
                min_value REAL,
                max_value REAL,
                percentiles TEXT,
                start_time TEXT,
                end_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics (name, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_summaries_name ON metric_summaries (name)')
        
        conn.commit()
        conn.close()
    
    def insert_metric(self, metric: Metric):
        """Insert single metric"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (name, metric_type, value, timestamp, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric.name,
            metric.metric_type.value,
            metric.value,
            metric.timestamp.isoformat(),
            json.dumps(metric.tags),
            json.dumps(metric.metadata) if metric.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def insert_metrics_batch(self, metrics: List[Metric]):
        """Insert multiple metrics efficiently"""
        
        if not metrics:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = [
            (
                m.name,
                m.metric_type.value,
                m.value,
                m.timestamp.isoformat(),
                json.dumps(m.tags),
                json.dumps(m.metadata) if m.metadata else None
            )
            for m in metrics
        ]
        
        cursor.executemany('''
            INSERT INTO metrics (name, metric_type, value, timestamp, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        conn.close()
    
    def query_metrics(self, 
                     metric_name: str = None,
                     start_time: datetime = None,
                     end_time: datetime = None,
                     tags: Dict[str, str] = None,
                     limit: int = None) -> List[Metric]:
        """Query metrics with filters"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND name = ?"
            params.append(metric_name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in rows:
            metric = Metric(
                name=row[1],
                metric_type=MetricType(row[2]),
                value=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                tags=json.loads(row[5]) if row[5] else {},
                metadata=json.loads(row[6]) if row[6] else None
            )
            
            # Filter by tags if specified
            if tags:
                if all(metric.tags.get(k) == v for k, v in tags.items()):
                    metrics.append(metric)
            else:
                metrics.append(metric)
        
        return metrics
    
    def save_summary(self, summary: MetricSummary):
        """Save metric summary"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metric_summaries 
            (name, metric_type, count, sum_value, avg_value, min_value, max_value, 
             percentiles, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary.name,
            summary.metric_type.value,
            summary.count,
            summary.sum_value,
            summary.avg_value,
            summary.min_value,
            summary.max_value,
            json.dumps(summary.percentiles),
            summary.start_time.isoformat(),
            summary.end_time.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up old metrics to manage database size"""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_date.isoformat(),))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logging.info(f"Cleaned up {deleted_count} old metrics")
        return deleted_count

class MetricsAggregator:
    """Aggregates metrics for analysis and reporting"""
    
    def __init__(self):
        self.aggregation_functions = {
            'count': len,
            'sum': sum,
            'avg': lambda x: np.mean(x) if x else 0,
            'min': lambda x: min(x) if x else 0,
            'max': lambda x: max(x) if x else 0,
            'median': lambda x: np.median(x) if x else 0,
            'std': lambda x: np.std(x) if x else 0,
            'percentile_50': lambda x: np.percentile(x, 50) if x else 0,
            'percentile_75': lambda x: np.percentile(x, 75) if x else 0,
            'percentile_90': lambda x: np.percentile(x, 90) if x else 0,
            'percentile_95': lambda x: np.percentile(x, 95) if x else 0,
            'percentile_99': lambda x: np.percentile(x, 99) if x else 0
        }
    
    def aggregate_metrics(self, metrics: List[Metric], group_by: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics by specified dimensions"""
        
        if not metrics:
            return {}
        
        # Group metrics
        if group_by:
            groups = defaultdict(list)
            for metric in metrics:
                group_key = tuple(metric.tags.get(key, 'unknown') for key in group_by)
                groups[group_key].append(metric.value)
        else:
            groups = {'all': [m.value for m in metrics]}
        
        # Calculate aggregations for each group
        results = {}
        for group_key, values in groups.items():
            group_name = '_'.join(group_key) if isinstance(group_key, tuple) else str(group_key)
            
            results[group_name] = {}
            for agg_name, agg_func in self.aggregation_functions.items():
                try:
                    results[group_name][agg_name] = float(agg_func(values))
                except Exception as e:
                    logging.warning(f"Aggregation {agg_name} failed for group {group_name}: {e}")
                    results[group_name][agg_name] = 0.0
        
        return results
    
    def create_summary(self, metrics: List[Metric]) -> MetricSummary:
        """Create metric summary from list of metrics"""
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        percentiles = {
            'p50': np.percentile(values, 50),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
        return MetricSummary(
            name=metrics[0].name,
            metric_type=metrics[0].metric_type,
            count=len(metrics),
            sum_value=sum(values),
            avg_value=np.mean(values),
            min_value=min(values),
            max_value=max(values),
            percentiles=percentiles,
            start_time=min(m.timestamp for m in metrics),
            end_time=max(m.timestamp for m in metrics)
        )
    
    def time_series_analysis(self, metrics: List[Metric], interval_minutes: int = 5) -> Dict[str, List[float]]:
        """Analyze metrics as time series with specified intervals"""
        
        if not metrics:
            return {}
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.timestamp)
        
        # Create time buckets
        start_time = metrics[0].timestamp
        end_time = metrics[-1].timestamp
        
        buckets = defaultdict(list)
        
        for metric in metrics:
            # Calculate which bucket this metric belongs to
            minutes_since_start = (metric.timestamp - start_time).total_seconds() / 60
            bucket_index = int(minutes_since_start // interval_minutes)
            buckets[bucket_index].append(metric.value)
        
        # Aggregate each bucket
        time_series = {
            'timestamps': [],
            'values': [],
            'counts': []
        }
        
        max_bucket = max(buckets.keys()) if buckets else 0
        
        for i in range(max_bucket + 1):
            bucket_time = start_time + timedelta(minutes=i * interval_minutes)
            time_series['timestamps'].append(bucket_time.isoformat())
            
            if i in buckets:
                bucket_values = buckets[i]
                time_series['values'].append(np.mean(bucket_values))
                time_series['counts'].append(len(bucket_values))
            else:
                time_series['values'].append(0.0)
                time_series['counts'].append(0)
        
        return time_series

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector, metric_name: str, tags: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.timer(self.metric_name, duration, self.tags)

class MetricsCollector:
    """Main metrics collection engine"""
    
    def __init__(self, db_path: str = "metrics.db", buffer_size: int = 1000, flush_interval: int = 30):
        self.db = MetricsDatabase(db_path)
        self.aggregator = MetricsAggregator()
        
        # Buffering for performance
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.last_flush = time.time()
        self.buffer_lock = threading.Lock()
        
        # In-memory metrics for real-time access
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.timers = defaultdict(list)
        self.rates = defaultdict(list)
        
        # Collection statistics
        self.collection_stats = {
            'total_metrics_collected': 0,
            'metrics_flushed': 0,
            'buffer_flushes': 0,
            'collection_errors': 0
        }
        
        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        
        logging.info("Metrics Collector initialized")
    
    def _add_metric(self, name: str, metric_type: MetricType, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Internal method to add metric"""
        
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            self.collection_stats['total_metrics_collected'] += 1
        
        # Auto-flush if buffer is full or enough time has passed
        if (len(self.metrics_buffer) >= self.buffer_size or 
            time.time() - self.last_flush >= self.flush_interval):
            self._flush_buffer()
    
    def counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record counter metric"""
        self.counters[name] += value
        self._add_metric(name, MetricType.COUNTER, value, tags, metadata)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record gauge metric"""
        self.gauges[name] = value
        self._add_metric(name, MetricType.GAUGE, value, tags, metadata)
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record histogram metric"""
        self._add_metric(name, MetricType.HISTOGRAM, value, tags, metadata)
    
    def timer(self, name: str, duration: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record timer metric"""
        self.timers[name].append(duration)
        # Keep only recent timer values
        self.timers[name] = self.timers[name][-1000:]
        self._add_metric(name, MetricType.TIMER, duration, tags, metadata)
    
    def rate(self, name: str, value: float = 1.0, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record rate metric"""
        self.rates[name].append((time.time(), value))
        # Keep only recent rate values (last hour)
        cutoff_time = time.time() - 3600
        self.rates[name] = [(t, v) for t, v in self.rates[name] if t > cutoff_time]
        self._add_metric(name, MetricType.RATE, value, tags, metadata)
    
    @contextmanager
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        timer = PerformanceTimer(self, name, tags)
        with timer:
            yield timer
    
    def track_user_interaction(self, user_id: str, interaction_type: str, response_time: float = None, 
                              satisfaction_score: float = None, query_type: str = None):
        """Track user interaction metrics"""
        
        base_tags = {
            'user_id': user_id,
            'interaction_type': interaction_type
        }
        
        if query_type:
            base_tags['query_type'] = query_type
        
        # Track interaction count
        self.counter('user_interactions_total', 1.0, base_tags)
        
        # Track response time if provided
        if response_time is not None:
            self.timer('response_time_seconds', response_time, base_tags)
        
        # Track satisfaction if provided
        if satisfaction_score is not None:
            self.gauge('user_satisfaction_score', satisfaction_score, base_tags)
            
            # Track satisfaction categories
            if satisfaction_score >= 0.8:
                self.counter('high_satisfaction_interactions', 1.0, base_tags)
            elif satisfaction_score <= 0.3:
                self.counter('low_satisfaction_interactions', 1.0, base_tags)
    
    def track_system_performance(self, cpu_usage: float = None, memory_usage: float = None, 
                                disk_usage: float = None, active_connections: int = None):
        """Track system performance metrics"""
        
        system_tags = {'component': 'system'}
        
        if cpu_usage is not None:
            self.gauge('cpu_usage_percent', cpu_usage, system_tags)
        
        if memory_usage is not None:
            self.gauge('memory_usage_percent', memory_usage, system_tags)
        
        if disk_usage is not None:
            self.gauge('disk_usage_percent', disk_usage, system_tags)
        
        if active_connections is not None:
            self.gauge('active_connections', active_connections, system_tags)
    
    def track_ai_model_metrics(self, model_name: str, inference_time: float = None, 
                              token_usage: int = None, cost: float = None, accuracy: float = None):
        """Track AI model performance metrics"""
        
        model_tags = {
            'model_name': model_name,
            'component': 'ai_model'
        }
        
        if inference_time is not None:
            self.timer('ai_inference_time_seconds', inference_time, model_tags)
        
        if token_usage is not None:
            self.counter('ai_tokens_used_total', token_usage, model_tags)
        
        if cost is not None:
            self.counter('ai_cost_total', cost, model_tags)
        
        if accuracy is not None:
            self.gauge('ai_model_accuracy', accuracy, model_tags)
    
    def track_retrieval_metrics(self, query: str, results_count: int, retrieval_time: float, 
                               relevance_score: float = None, search_type: str = "hybrid"):
        """Track RAG retrieval performance"""
        
        retrieval_tags = {
            'search_type': search_type,
            'component': 'retrieval'
        }
        
        # Track retrieval performance
        self.counter('retrieval_queries_total', 1.0, retrieval_tags)
        self.gauge('retrieval_results_count', results_count, retrieval_tags)
        self.timer('retrieval_time_seconds', retrieval_time, retrieval_tags)
        
        if relevance_score is not None:
            self.gauge('retrieval_relevance_score', relevance_score, retrieval_tags)
        
        # Track query characteristics
        query_length = len(query.split())
        self.histogram('query_length_words', query_length, retrieval_tags)
    
    def _flush_buffer(self):
        """Flush metrics buffer to database"""
        
        with self.buffer_lock:
            if not self.metrics_buffer:
                return
            
            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        
        try:
            self.db.insert_metrics_batch(metrics_to_flush)
            self.collection_stats['metrics_flushed'] += len(metrics_to_flush)
            self.collection_stats['buffer_flushes'] += 1
            self.last_flush = time.time()
        except Exception as e:
            logging.error(f"Failed to flush metrics: {e}")
            self.collection_stats['collection_errors'] += 1
    
    def _flush_worker(self):
        """Background worker for periodic buffer flushing"""
        
        while True:
            try:
                time.sleep(self.flush_interval)
                if time.time() - self.last_flush >= self.flush_interval:
                    self._flush_buffer()
            except Exception as e:
                logging.error(f"Flush worker error: {e}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current in-memory metrics for real-time monitoring"""
        
        current_time = time.time()
        
        # Calculate rates for last minute
        recent_rates = {}
        for name, rate_data in self.rates.items():
            recent_data = [(t, v) for t, v in rate_data if current_time - t <= 60]
            if recent_data:
                total_value = sum(v for t, v in recent_data)
                recent_rates[name] = total_value / 60  # Per second
            else:
                recent_rates[name] = 0.0
        
        # Calculate timer statistics
        timer_stats = {}
        for name, timer_data in self.timers.items():
            if timer_data:
                timer_stats[name] = {
                    'count': len(timer_data),
                    'avg': np.mean(timer_data),
                    'p95': np.percentile(timer_data, 95),
                    'p99': np.percentile(timer_data, 99)
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'rates_per_second': recent_rates,
            'timer_stats': timer_stats,
            'collection_stats': self.collection_stats
        }
    
    def get_historical_metrics(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get historical metrics for analysis"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.db.query_metrics(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return {}
        
        # Create aggregations
        aggregations = self.aggregator.aggregate_metrics(metrics)
        
        # Create time series
        time_series = self.aggregator.time_series_analysis(metrics, interval_minutes=5)
        
        # Create summary
        summary = self.aggregator.create_summary(metrics)
        
        return {
            'metric_name': metric_name,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'summary': asdict(summary) if summary else None,
            'aggregations': aggregations,
            'time_series': time_series,
            'total_data_points': len(metrics)
        }
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create comprehensive dashboard data"""
        
        real_time = self.get_real_time_metrics()
        
        # Get key metrics for last hour
        key_metrics = [
            'user_interactions_total',
            'response_time_seconds',
            'user_satisfaction_score',
            'retrieval_time_seconds',
            'ai_inference_time_seconds'
        ]
        
        dashboard = {
            'real_time': real_time,
            'historical_metrics': {},
            'alerts': [],
            'system_health': {}
        }
        
        for metric_name in key_metrics:
            historical = self.get_historical_metrics(metric_name, hours=1)
            if historical:
                dashboard['historical_metrics'][metric_name] = historical
        
        # Generate alerts based on thresholds
        alerts = self._generate_alerts(real_time)
        dashboard['alerts'] = alerts
        
        # System health summary
        dashboard['system_health'] = self._calculate_system_health(real_time)
        
        return dashboard
    
    def _generate_alerts(self, real_time_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts based on metric thresholds"""
        
        alerts = []
        
        # Check response time
        timer_stats = real_time_metrics.get('timer_stats', {})
        response_time_stats = timer_stats.get('response_time_seconds', {})
        
        if response_time_stats and response_time_stats.get('p95', 0) > 5.0:
            alerts.append({
                'level': 'warning',
                'metric': 'response_time_seconds',
                'message': f"95th percentile response time is {response_time_stats['p95']:.2f}s (threshold: 5.0s)"
            })
        
        # Check satisfaction score
        gauges = real_time_metrics.get('gauges', {})
        satisfaction = gauges.get('user_satisfaction_score', 1.0)
        
        if satisfaction < 0.7:
            alerts.append({
                'level': 'critical',
                'metric': 'user_satisfaction_score',
                'message': f"User satisfaction score is {satisfaction:.2f} (threshold: 0.7)"
            })
        
        # Check error rates
        collection_stats = real_time_metrics.get('collection_stats', {})
        error_rate = collection_stats.get('collection_errors', 0) / max(collection_stats.get('total_metrics_collected', 1), 1)
        
        if error_rate > 0.01:  # 1% error rate
            alerts.append({
                'level': 'warning',
                'metric': 'collection_errors',
                'message': f"High error rate: {error_rate:.2%} (threshold: 1%)"
            })
        
        return alerts
    
    def _calculate_system_health(self, real_time_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score"""
        
        health_factors = {}
        
        # Response time health
        timer_stats = real_time_metrics.get('timer_stats', {})
        response_time_stats = timer_stats.get('response_time_seconds', {})
        
        if response_time_stats:
            avg_response_time = response_time_stats.get('avg', 0)
            response_health = max(0, 1 - (avg_response_time / 10))  # 10s = 0 health
            health_factors['response_time'] = response_health
        
        # Satisfaction health
        gauges = real_time_metrics.get('gauges', {})
        satisfaction = gauges.get('user_satisfaction_score', 0.8)
        health_factors['satisfaction'] = satisfaction
        
        # Error rate health
        collection_stats = real_time_metrics.get('collection_stats', {})
        total_metrics = collection_stats.get('total_metrics_collected', 1)
        error_rate = collection_stats.get('collection_errors', 0) / total_metrics
        error_health = max(0, 1 - (error_rate * 100))  # 1% errors = 0 health
        health_factors['error_rate'] = error_health
        
        # Calculate overall health
        if health_factors:
            overall_health = np.mean(list(health_factors.values()))
        else:
            overall_health = 0.8  # Default health
        
        return {
            'overall_health_score': overall_health,
            'health_factors': health_factors,
            'status': 'healthy' if overall_health > 0.8 else 'degraded' if overall_health > 0.6 else 'unhealthy'
        }
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old metrics data"""
        
        deleted_count = self.db.cleanup_old_metrics(retention_days)
        
        # Clear old in-memory data
        current_time = time.time()
        cutoff_time = current_time - (retention_days * 24 * 3600)
        
        for name in list(self.rates.keys()):
            self.rates[name] = [(t, v) for t, v in self.rates[name] if t > cutoff_time]
        
        return deleted_count
    
    def export_metrics(self, metric_name: str = None, start_time: datetime = None, 
                      end_time: datetime = None, format: str = "json") -> str:
        """Export metrics in specified format"""
        
        metrics = self.db.query_metrics(metric_name, start_time, end_time)
        
        if format == "json":
            return json.dumps([asdict(m) for m in metrics], default=str, indent=2)
        elif format == "csv":
            # Simple CSV export
            lines = ["name,type,value,timestamp,tags"]
            for metric in metrics:
                tags_str = ";".join(f"{k}={v}" for k, v in metric.tags.items())
                lines.append(f"{metric.name},{metric.metric_type.value},{metric.value},{metric.timestamp},{tags_str}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        
        return {
            **self.collection_stats,
            'buffer_size': len(self.metrics_buffer),
            'max_buffer_size': self.buffer_size,
            'active_counters': len(self.counters),
            'active_gauges': len(self.gauges),
            'active_timers': len(self.timers),
            'active_rates': len(self.rates),
            'timestamp': datetime.now().isoformat()
        }
