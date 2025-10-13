import json
import uuid
import hashlib
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
from collections import defaultdict

class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

@dataclass
class ExperimentVariant:
    variant_id: str
    variant_name: str
    variant_type: VariantType
    traffic_percentage: float
    configuration: Dict[str, Any]
    description: str

@dataclass
class ExperimentMetric:
    metric_name: str
    metric_type: str  # "conversion", "continuous", "count"
    primary: bool
    target_improvement: Optional[float] = None
    min_sample_size: Optional[int] = None

@dataclass
class Experiment:
    experiment_id: str
    experiment_name: str
    description: str
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    start_date: datetime
    end_date: Optional[datetime]
    min_runtime_days: int
    confidence_level: float
    power: float
    created_by: str
    metadata: Dict[str, Any]

@dataclass
class ExperimentAssignment:
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: Optional[str] = None

@dataclass
class ExperimentEvent:
    event_id: str
    user_id: str
    experiment_id: str
    variant_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class ExperimentDatabase:
    """Database layer for A/B testing framework"""
    
    def __init__(self, db_path: str = "ab_tests.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for experiments"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                variants TEXT NOT NULL,
                metrics TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                min_runtime_days INTEGER,
                confidence_level REAL,
                power REAL,
                created_by TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User assignments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_assignments (
                user_id TEXT,
                experiment_id TEXT,
                variant_id TEXT,
                assigned_at TEXT,
                session_id TEXT,
                PRIMARY KEY (user_id, experiment_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT,
                experiment_id TEXT,
                variant_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT,
                session_id TEXT,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_experiment_metric ON experiment_events (experiment_id, metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON experiment_events (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_experiment ON user_assignments (experiment_id)')
        
        conn.commit()
        conn.close()
    
    def save_experiment(self, experiment: Experiment):
        """Save experiment to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment.experiment_id,
            experiment.experiment_name,
            experiment.description,
            experiment.status.value,
            json.dumps([asdict(v) for v in experiment.variants]),
            json.dumps([asdict(m) for m in experiment.metrics]),
            experiment.start_date.isoformat(),
            experiment.end_date.isoformat() if experiment.end_date else None,
            experiment.min_runtime_days,
            experiment.confidence_level,
            experiment.power,
            experiment.created_by,
            json.dumps(experiment.metadata),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments WHERE experiment_id = ?', (experiment_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Parse the row data
        variants_data = json.loads(row[4])
        variants = [ExperimentVariant(**v) for v in variants_data]
        
        metrics_data = json.loads(row[5])
        metrics = [ExperimentMetric(**m) for m in metrics_data]
        
        return Experiment(
            experiment_id=row[0],
            experiment_name=row[1],
            description=row[2],
            status=ExperimentStatus(row[3]),
            variants=variants,
            metrics=metrics,
            start_date=datetime.fromisoformat(row[6]),
            end_date=datetime.fromisoformat(row[7]) if row[7] else None,
            min_runtime_days=row[8],
            confidence_level=row[9],
            power=row[10],
            created_by=row[11],
            metadata=json.loads(row[12]) if row[12] else {}
        )
    
    def save_assignment(self, assignment: ExperimentAssignment):
        """Save user assignment"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_assignments VALUES (?, ?, ?, ?, ?)
        ''', (
            assignment.user_id,
            assignment.experiment_id,
            assignment.variant_id,
            assignment.assigned_at.isoformat(),
            assignment.session_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_assignment(self, user_id: str, experiment_id: str) -> Optional[ExperimentAssignment]:
        """Get user assignment for experiment"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_assignments 
            WHERE user_id = ? AND experiment_id = ?
        ''', (user_id, experiment_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return ExperimentAssignment(
            user_id=row[0],
            experiment_id=row[1],
            variant_id=row[2],
            assigned_at=datetime.fromisoformat(row[3]),
            session_id=row[4]
        )
    
    def save_event(self, event: ExperimentEvent):
        """Save experiment event"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiment_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.user_id,
            event.experiment_id,
            event.variant_id,
            event.metric_name,
            event.metric_value,
            event.timestamp.isoformat(),
            event.session_id,
            json.dumps(event.metadata) if event.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment_events(self, experiment_id: str, metric_name: str = None) -> List[ExperimentEvent]:
        """Get events for experiment"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if metric_name:
            cursor.execute('''
                SELECT * FROM experiment_events 
                WHERE experiment_id = ? AND metric_name = ?
                ORDER BY timestamp
            ''', (experiment_id, metric_name))
        else:
            cursor.execute('''
                SELECT * FROM experiment_events 
                WHERE experiment_id = ?
                ORDER BY timestamp
            ''', (experiment_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append(ExperimentEvent(
                event_id=row[0],
                user_id=row[1],
                experiment_id=row[2],
                variant_id=row[3],
                metric_name=row[4],
                metric_value=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                session_id=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            ))
        
        return events

class UserAssignmentEngine:
    """Engine for assigning users to experiment variants"""
    
    def __init__(self):
        self.assignment_cache = {}
    
    def assign_user_to_variant(self, user_id: str, experiment: Experiment) -> str:
        """Assign user to experiment variant using consistent hashing"""
        
        # Check cache first
        cache_key = f"{user_id}_{experiment.experiment_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Use deterministic hash for consistent assignment
        hash_input = f"{user_id}_{experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Convert to percentage (0-99.99...)
        percentage = (hash_value % 10000) / 100.0
        
        # Find variant based on traffic allocation
        cumulative_percentage = 0.0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                self.assignment_cache[cache_key] = variant.variant_id
                return variant.variant_id
        
        # Fallback to control if something goes wrong
        control_variant = next((v for v in experiment.variants if v.variant_type == VariantType.CONTROL), experiment.variants[0])
        self.assignment_cache[cache_key] = control_variant.variant_id
        return control_variant.variant_id
    
    def is_user_eligible(self, user_id: str, experiment: Experiment, user_metadata: Dict[str, Any] = None) -> bool:
        """Check if user is eligible for experiment"""
        
        # Basic eligibility checks
        if experiment.status != ExperimentStatus.ACTIVE:
            return False
        
        if datetime.now() < experiment.start_date:
            return False
        
        if experiment.end_date and datetime.now() > experiment.end_date:
            return False
        
        # Custom eligibility rules from experiment metadata
        eligibility_rules = experiment.metadata.get('eligibility_rules', {})
        
        if user_metadata and eligibility_rules:
            # Check user segment filters
            required_segments = eligibility_rules.get('required_segments', [])
            user_segments = user_metadata.get('segments', [])
            
            if required_segments and not any(seg in user_segments for seg in required_segments):
                return False
            
            # Check excluded segments
            excluded_segments = eligibility_rules.get('excluded_segments', [])
            if excluded_segments and any(seg in user_segments for seg in excluded_segments):
                return False
            
            # Check user properties
            required_properties = eligibility_rules.get('required_properties', {})
            for prop, expected_value in required_properties.items():
                if user_metadata.get(prop) != expected_value:
                    return False
        
        return True

class StatisticalAnalyzer:
    """Statistical analysis for A/B test results"""
    
    def __init__(self):
        self.confidence_levels = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
    
    def analyze_experiment_results(self, experiment: Experiment, events: List[ExperimentEvent]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        results = {
            'experiment_id': experiment.experiment_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'metrics': {},
            'overall_summary': {},
            'recommendations': []
        }
        
        # Group events by metric and variant
        metric_data = defaultdict(lambda: defaultdict(list))
        for event in events:
            metric_data[event.metric_name][event.variant_id].append(event.metric_value)
        
        # Analyze each metric
        for metric in experiment.metrics:
            metric_name = metric.metric_name
            
            if metric_name not in metric_data:
                continue
            
            metric_results = self._analyze_metric(
                metric, metric_data[metric_name], experiment.variants, experiment.confidence_level
            )
            results['metrics'][metric_name] = metric_results
        
        # Generate overall summary
        results['overall_summary'] = self._generate_overall_summary(results['metrics'], experiment)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['metrics'], experiment)
        
        return results
    
    def _analyze_metric(self, metric: ExperimentMetric, variant_data: Dict[str, List[float]], 
                       variants: List[ExperimentVariant], confidence_level: float) -> Dict[str, Any]:
        """Analyze a specific metric across variants"""
        
        analysis = {
            'metric_name': metric.metric_name,
            'metric_type': metric.metric_type,
            'primary': metric.primary,
            'variants': {},
            'comparisons': {},
            'significance_test': None
        }
        
        # Calculate descriptive statistics for each variant
        for variant in variants:
            variant_id = variant.variant_id
            
            if variant_id not in variant_data:
                continue
            
            data = variant_data[variant_id]
            
            if not data:
                continue
            
            analysis['variants'][variant_id] = {
                'variant_name': variant.variant_name,
                'sample_size': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'confidence_interval': self._calculate_confidence_interval(data, confidence_level)
            }
        
        # Perform pairwise comparisons
        control_variant = next((v for v in variants if v.variant_type == VariantType.CONTROL), variants[0])
        control_id = control_variant.variant_id
        
        if control_id in variant_data:
            control_data = variant_data[control_id]
            
            for variant in variants:
                if variant.variant_id == control_id or variant.variant_id not in variant_data:
                    continue
                
                treatment_data = variant_data[variant.variant_id]
                
                # Perform statistical test
                comparison_result = self._perform_statistical_test(
                    control_data, treatment_data, metric.metric_type, confidence_level
                )
                
                analysis['comparisons'][variant.variant_id] = comparison_result
        
        return analysis
    
    def _calculate_confidence_interval(self, data: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std_error = stats.sem(data)
        z_score = self.confidence_levels.get(confidence_level, 1.96)
        
        margin_error = z_score * std_error
        
        return (mean - margin_error, mean + margin_error)
    
    def _perform_statistical_test(self, control_data: List[float], treatment_data: List[float], 
                                 metric_type: str, confidence_level: float) -> Dict[str, Any]:
        """Perform appropriate statistical test"""
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                'test_type': 'insufficient_data',
                'p_value': None,
                'statistically_significant': False,
                'effect_size': 0.0,
                'power': 0.0
            }
        
        # Choose appropriate test based on metric type
        if metric_type == 'conversion':
            # Use chi-square test for conversion rates
            control_conversions = sum(control_data)
            control_total = len(control_data)
            treatment_conversions = sum(treatment_data)
            treatment_total = len(treatment_data)
            
            # Create contingency table
            contingency_table = [
                [control_conversions, control_total - control_conversions],
                [treatment_conversions, treatment_total - treatment_conversions]
            ]
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            # Calculate effect size (relative difference)
            control_rate = control_conversions / control_total if control_total > 0 else 0
            treatment_rate = treatment_conversions / treatment_total if treatment_total > 0 else 0
            effect_size = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
            
            test_result = {
                'test_type': 'chi_square',
                'chi2_statistic': chi2,
                'p_value': p_value,
                'control_rate': control_rate,
                'treatment_rate': treatment_rate,
                'effect_size': effect_size
            }
        
        else:
            # Use t-test for continuous metrics
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data) + 
                                (len(treatment_data) - 1) * np.var(treatment_data)) / 
                               (len(control_data) + len(treatment_data) - 2))
            
            effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std if pooled_std > 0 else 0
            
            test_result = {
                'test_type': 't_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'control_mean': np.mean(control_data),
                'treatment_mean': np.mean(treatment_data),
                'effect_size': effect_size
            }
        
        # Determine statistical significance
        alpha = 1 - confidence_level
        test_result['statistically_significant'] = p_value < alpha if p_value is not None else False
        
        # Calculate statistical power (simplified)
        test_result['power'] = self._calculate_statistical_power(
            len(control_data), len(treatment_data), effect_size, alpha
        )
        
        return test_result
    
    def _calculate_statistical_power(self, n1: int, n2: int, effect_size: float, alpha: float) -> float:
        """Calculate statistical power (simplified calculation)"""
        
        # This is a simplified power calculation
        # In production, you'd use more sophisticated methods
        
        if n1 < 10 or n2 < 10:
            return 0.0
        
        # Approximate power calculation
        n_harmonic = 2 / (1/n1 + 1/n2)
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = abs(effect_size) * np.sqrt(n_harmonic / 4) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    def _generate_overall_summary(self, metric_results: Dict[str, Any], experiment: Experiment) -> Dict[str, Any]:
        """Generate overall experiment summary"""
        
        primary_metrics = [m for m in experiment.metrics if m.primary]
        significant_results = 0
        total_comparisons = 0
        
        for metric_name, results in metric_results.items():
            for variant_id, comparison in results.get('comparisons', {}).items():
                total_comparisons += 1
                if comparison.get('statistically_significant', False):
                    significant_results += 1
        
        # Calculate experiment duration
        start_date = experiment.start_date
        current_date = datetime.now()
        days_running = (current_date - start_date).days
        
        # Determine experiment readiness
        min_runtime_met = days_running >= experiment.min_runtime_days
        sufficient_power = all(
            any(comp.get('power', 0) > 0.8 for comp in results.get('comparisons', {}).values())
            for results in metric_results.values()
        )
        
        return {
            'days_running': days_running,
            'min_runtime_met': min_runtime_met,
            'total_comparisons': total_comparisons,
            'significant_results': significant_results,
            'significance_rate': significant_results / total_comparisons if total_comparisons > 0 else 0,
            'sufficient_power': sufficient_power,
            'ready_for_conclusion': min_runtime_met and sufficient_power,
            'primary_metrics_count': len(primary_metrics)
        }
    
    def _generate_recommendations(self, metric_results: Dict[str, Any], experiment: Experiment) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Check for winning variants
        for metric_name, results in metric_results.items():
            for variant_id, comparison in results.get('comparisons', {}).items():
                if comparison.get('statistically_significant', False):
                    effect_size = comparison.get('effect_size', 0)
                    if effect_size > 0:
                        variant_name = next(v.variant_name for v in experiment.variants if v.variant_id == variant_id)
                        recommendations.append(
                            f"Consider implementing variant '{variant_name}' - shows significant improvement in {metric_name}"
                        )
                    elif effect_size < 0:
                        variant_name = next(v.variant_name for v in experiment.variants if v.variant_id == variant_id)
                        recommendations.append(
                            f"Avoid variant '{variant_name}' - shows significant decrease in {metric_name}"
                        )
        
        # Check for insufficient power
        low_power_metrics = []
        for metric_name, results in metric_results.items():
            for comparison in results.get('comparisons', {}).values():
                if comparison.get('power', 0) < 0.8:
                    low_power_metrics.append(metric_name)
                    break
        
        if low_power_metrics:
            recommendations.append(
                f"Consider extending experiment duration for metrics: {', '.join(low_power_metrics)} (insufficient statistical power)"
            )
        
        # Check for no significant results
        if not any(
            any(comp.get('statistically_significant', False) for comp in results.get('comparisons', {}).values())
            for results in metric_results.values()
        ):
            recommendations.append(
                "No significant differences detected. Consider larger effect sizes or longer experiment duration."
            )
        
        return recommendations

class ABTestFramework:
    """Main A/B testing framework"""
    
    def __init__(self, db_path: str = "ab_tests.db"):
        self.db = ExperimentDatabase(db_path)
        self.assignment_engine = UserAssignmentEngine()
        self.analyzer = StatisticalAnalyzer()
        
        # Framework statistics
        self.framework_stats = {
            'total_experiments': 0,
            'active_experiments': 0,
            'total_assignments': 0,
            'total_events': 0
        }
        
        logging.info("A/B Test Framework initialized")
    
    def create_experiment(self, 
                         experiment_name: str,
                         description: str,
                         variants: List[Dict[str, Any]],
                         metrics: List[Dict[str, Any]],
                         start_date: datetime,
                         min_runtime_days: int = 14,
                         confidence_level: float = 0.95,
                         power: float = 0.8,
                         created_by: str = "system",
                         metadata: Dict[str, Any] = None) -> str:
        """Create new A/B test experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        # Create variant objects
        variant_objects = []
        for variant_data in variants:
            variant_objects.append(ExperimentVariant(
                variant_id=str(uuid.uuid4()),
                variant_name=variant_data['name'],
                variant_type=VariantType(variant_data.get('type', 'treatment')),
                traffic_percentage=variant_data['traffic_percentage'],
                configuration=variant_data.get('configuration', {}),
                description=variant_data.get('description', '')
            ))
        
        # Create metric objects
        metric_objects = []
        for metric_data in metrics:
            metric_objects.append(ExperimentMetric(
                metric_name=metric_data['name'],
                metric_type=metric_data.get('type', 'continuous'),
                primary=metric_data.get('primary', False),
                target_improvement=metric_data.get('target_improvement'),
                min_sample_size=metric_data.get('min_sample_size')
            ))
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=variant_objects,
            metrics=metric_objects,
            start_date=start_date,
            end_date=None,
            min_runtime_days=min_runtime_days,
            confidence_level=confidence_level,
            power=power,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        # Save to database
        self.db.save_experiment(experiment)
        self.framework_stats['total_experiments'] += 1
        
        logging.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        self.db.save_experiment(experiment)
        self.framework_stats['active_experiments'] += 1
        
        logging.info(f"Started experiment: {experiment.experiment_name}")
        return True
    
    def assign_user_to_experiment(self, user_id: str, experiment_id: str, 
                                 user_metadata: Dict[str, Any] = None, session_id: str = None) -> Optional[str]:
        """Assign user to experiment variant"""
        
        # Check if user already assigned
        existing_assignment = self.db.get_user_assignment(user_id, experiment_id)
        if existing_assignment:
            return existing_assignment.variant_id
        
        # Get experiment
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            return None
        
        # Check eligibility
        if not self.assignment_engine.is_user_eligible(user_id, experiment, user_metadata):
            return None
        
        # Assign to variant
        variant_id = self.assignment_engine.assign_user_to_variant(user_id, experiment)
        
        # Save assignment
        assignment = ExperimentAssignment(
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            assigned_at=datetime.now(),
            session_id=session_id
        )
        
        self.db.save_assignment(assignment)
        self.framework_stats['total_assignments'] += 1
        
        return variant_id
    
    def track_event(self, user_id: str, experiment_id: str, metric_name: str, 
                   metric_value: float, session_id: str = None, metadata: Dict[str, Any] = None):
        """Track experiment event/metric"""
        
        # Get user assignment
        assignment = self.db.get_user_assignment(user_id, experiment_id)
        if not assignment:
            return
        
        # Create event
        event = ExperimentEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=assignment.variant_id,
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata
        )
        
        # Save event
        self.db.save_event(event)
        self.framework_stats['total_events'] += 1
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results"""
        
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            return {}
        
        # Get all events for this experiment
        events = self.db.get_experiment_events(experiment_id)
        
        # Perform statistical analysis
        results = self.analyzer.analyze_experiment_results(experiment, events)
        
        # Add experiment metadata
        results['experiment_info'] = {
            'name': experiment.experiment_name,
            'description': experiment.description,
            'status': experiment.status.value,
            'start_date': experiment.start_date.isoformat(),
            'days_running': (datetime.now() - experiment.start_date).days,
            'variants': [
                {
                    'id': v.variant_id,
                    'name': v.variant_name,
                    'type': v.variant_type.value,
                    'traffic': v.traffic_percentage
                }
                for v in experiment.variants
            ]
        }
        
        return results
    
    def get_user_variant(self, user_id: str, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get user's assigned variant configuration"""
        
        assignment = self.db.get_user_assignment(user_id, experiment_id)
        if not assignment:
            return None
        
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            return None
        
        variant = next((v for v in experiment.variants if v.variant_id == assignment.variant_id), None)
        if not variant:
            return None
        
        return {
            'variant_id': variant.variant_id,
            'variant_name': variant.variant_name,
            'variant_type': variant.variant_type.value,
            'configuration': variant.configuration
        }
    
    def stop_experiment(self, experiment_id: str, reason: str = "completed") -> bool:
        """Stop an experiment"""
        
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED if reason == "completed" else ExperimentStatus.CANCELLED
        experiment.end_date = datetime.now()
        
        self.db.save_experiment(experiment)
        
        if experiment.status == ExperimentStatus.ACTIVE:
            self.framework_stats['active_experiments'] -= 1
        
        logging.info(f"Stopped experiment: {experiment.experiment_name} (Reason: {reason})")
        return True
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get A/B testing framework statistics"""
        
        return {
            **self.framework_stats,
            'timestamp': datetime.now().isoformat()
        }
