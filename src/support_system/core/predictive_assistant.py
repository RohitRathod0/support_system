import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class Suggestion:
    type: str  # 'followup_question', 'related_resource', 'proactive_help'
    content: str
    confidence: float
    reason: str
    link: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class SatisfactionPrediction:
    predicted_score: float
    confidence: float
    factors: Dict[str, float]
    improvements: List[str]

class UserBehaviorModel:
    """Model user behavior patterns for prediction"""
    
    def __init__(self):
        self.behavior_patterns = {}
        self.interaction_clusters = {}
    
    def analyze_patterns(self, user_history: List[Dict]) -> Dict:
        """Analyze user behavior patterns"""
        if not user_history:
            return self._default_patterns()
        
        patterns = {
            'tends_to_ask_followups': self._analyze_followup_tendency(user_history),
            'explores_related_topics': self._analyze_exploration_tendency(user_history),
            'prefers_detailed_responses': self._analyze_detail_preference(user_history),
            'typical_session_length': self._analyze_session_length(user_history),
            'common_query_types': self._analyze_common_queries(user_history),
            'satisfaction_trend': self._analyze_satisfaction_trend(user_history),
            'escalation_triggers': self._identify_escalation_triggers(user_history)
        }
        
        return patterns
    
    def _analyze_followup_tendency(self, history: List[Dict]) -> bool:
        """Analyze if user tends to ask follow-up questions"""
        followup_count = 0
        session_count = 0
        
        current_session_turns = []
        for interaction in history:
            if interaction.get('session_start', False) or not current_session_turns:
                if current_session_turns and len(current_session_turns) > 1:
                    session_count += 1
                current_session_turns = [interaction]
            else:
                current_session_turns.append(interaction)
                if len(current_session_turns) > 1:
                    followup_count += 1
        
        return followup_count > session_count * 0.3 if session_count > 0 else False
    
    def _analyze_exploration_tendency(self, history: List[Dict]) -> bool:
        """Analyze if user explores related topics"""
        unique_topics = set()
        for interaction in history:
            topic = interaction.get('topic') or interaction.get('category', 'general')
            unique_topics.add(topic)
        
        return len(unique_topics) > len(history) * 0.3
    
    def _analyze_detail_preference(self, history: List[Dict]) -> str:
        """Analyze user's preference for response detail"""
        detail_indicators = []
        
        for interaction in history:
            feedback = interaction.get('feedback', '')
            if 'more detail' in feedback.lower() or 'explain more' in feedback.lower():
                detail_indicators.append('high')
            elif 'too long' in feedback.lower() or 'be brief' in feedback.lower():
                detail_indicators.append('low')
            
            # Analyze response length vs satisfaction
            response_length = interaction.get('response_length', 0)
            satisfaction = interaction.get('satisfaction_score', 0)
            
            if response_length > 200 and satisfaction > 0.8:
                detail_indicators.append('high')
            elif response_length < 100 and satisfaction > 0.8:
                detail_indicators.append('low')
        
        if not detail_indicators:
            return 'medium'
        
        high_count = detail_indicators.count('high')
        low_count = detail_indicators.count('low')
        
        if high_count > low_count:
            return 'high'
        elif low_count > high_count:
            return 'low'
        else:
            return 'medium'
    
    def _analyze_session_length(self, history: List[Dict]) -> int:
        """Analyze typical session length"""
        session_lengths = []
        current_session_length = 0
        
        for interaction in history:
            if interaction.get('session_start', False):
                if current_session_length > 0:
                    session_lengths.append(current_session_length)
                current_session_length = 1
            else:
                current_session_length += 1
        
        if current_session_length > 0:
            session_lengths.append(current_session_length)
        
        return int(np.mean(session_lengths)) if session_lengths else 1
    
    def _analyze_common_queries(self, history: List[Dict]) -> List[str]:
        """Analyze most common query types"""
        query_types = {}
        for interaction in history:
            qtype = interaction.get('query_type', 'general')
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        sorted_types = sorted(query_types.items(), key=lambda x: x[1], reverse=True)
        return [qtype for qtype, count in sorted_types[:3]]
    
    def _analyze_satisfaction_trend(self, history: List[Dict]) -> str:
        """Analyze satisfaction trend over time"""
        satisfaction_scores = []
        for interaction in history:
            score = interaction.get('satisfaction_score')
            if score and score > 0:
                satisfaction_scores.append(score)
        
        if len(satisfaction_scores) < 3:
            return 'stable'
        
        recent_avg = np.mean(satisfaction_scores[-3:])
        earlier_avg = np.mean(satisfaction_scores[:-3])
        
        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_escalation_triggers(self, history: List[Dict]) -> List[str]:
        """Identify what triggers user escalation"""
        triggers = []
        
        for interaction in history:
            if interaction.get('escalated', False):
                # Look for patterns before escalation
                query_type = interaction.get('query_type', 'unknown')
                response_time = interaction.get('response_time', 0)
                
                if response_time > 5.0:
                    triggers.append('slow_response')
                if query_type in ['technical', 'billing']:
                    triggers.append(f'complex_{query_type}')
                
                feedback = interaction.get('feedback', '').lower()
                if 'frustrated' in feedback or 'angry' in feedback:
                    triggers.append('emotional_state')
        
        return list(set(triggers))
    
    def _default_patterns(self) -> Dict:
        """Default behavior patterns for new users"""
        return {
            'tends_to_ask_followups': False,
            'explores_related_topics': False,
            'prefers_detailed_responses': 'medium',
            'typical_session_length': 2,
            'common_query_types': ['general'],
            'satisfaction_trend': 'stable',
            'escalation_triggers': []
        }

class SatisfactionPredictor:
    """Predict user satisfaction using machine learning"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'response_length', 'technical_complexity', 'completeness_score',
            'user_expertise_match', 'response_tone_match', 'resolution_indicators',
            'response_time', 'context_relevance', 'followup_needed'
        ]
    
    def extract_features(self, response: str, user_profile: Dict, query_context: Dict) -> Dict[str, float]:
        """Extract features for satisfaction prediction"""
        features = {}
        
        # Response characteristics
        features['response_length'] = len(response.split())
        features['technical_complexity'] = self._assess_technical_complexity(response)
        features['completeness_score'] = self._assess_completeness(response, query_context)
        
        # User match features
        features['user_expertise_match'] = self._match_user_expertise(response, user_profile)
        features['response_tone_match'] = self._match_communication_tone(response, user_profile)
        
        # Resolution indicators
        features['resolution_indicators'] = self._detect_resolution_indicators(response)
        
        # Context features
        features['response_time'] = query_context.get('response_time', 0)
        features['context_relevance'] = query_context.get('context_relevance', 0.5)
        features['followup_needed'] = self._predict_followup_needed(response)
        
        return features
    
    def _assess_technical_complexity(self, response: str) -> float:
        """Assess technical complexity of response"""
        technical_terms = ['API', 'database', 'server', 'configuration', 'protocol', 
                          'authentication', 'encryption', 'firewall', 'bandwidth']
        
        response_lower = response.lower()
        tech_term_count = sum(1 for term in technical_terms if term.lower() in response_lower)
        
        # Normalize by response length
        complexity = tech_term_count / (len(response.split()) / 50)  # per 50 words
        return min(1.0, complexity)
    
    def _assess_completeness(self, response: str, query_context: Dict) -> float:
        """Assess how complete the response is"""
        completeness_indicators = [
            'step', 'process', 'procedure', 'first', 'then', 'finally',
            'example', 'specifically', 'detailed', 'complete'
        ]
        
        response_lower = response.lower()
        indicator_count = sum(1 for indicator in completeness_indicators 
                            if indicator in response_lower)
        
        # Consider query complexity
        query = query_context.get('query', '')
        query_complexity = len(query.split()) / 10  # Rough complexity measure
        
        base_score = min(1.0, indicator_count / 5)  # Up to 5 indicators
        adjusted_score = base_score * (1 + query_complexity * 0.2)
        
        return min(1.0, adjusted_score)
    
    def _match_user_expertise(self, response: str, user_profile: Dict) -> float:
        """Calculate how well response matches user expertise level"""
        user_expertise = user_profile.get('expertise_level', 'intermediate')
        response_complexity = self._assess_technical_complexity(response)
        
        if user_expertise == 'beginner':
            # Beginners prefer simpler explanations
            return 1.0 - response_complexity
        elif user_expertise == 'expert':
            # Experts can handle complex explanations
            return response_complexity
        else:  # intermediate
            # Intermediate users prefer moderate complexity
            ideal_complexity = 0.5
            return 1.0 - abs(response_complexity - ideal_complexity)
    
    def _match_communication_tone(self, response: str, user_profile: Dict) -> float:
        """Calculate how well response tone matches user preference"""
        user_style = user_profile.get('communication_style', 'professional')
        
        # Simple tone analysis
        formal_indicators = ['please', 'kindly', 'would', 'could', 'may']
        casual_indicators = ['hey', 'sure', 'okay', 'great', 'awesome']
        
        response_lower = response.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in response_lower)
        casual_count = sum(1 for indicator in casual_indicators if indicator in response_lower)
        
        response_formality = formal_count / (formal_count + casual_count + 1)
        
        if user_style == 'professional':
            return response_formality
        elif user_style == 'casual':
            return 1.0 - response_formality
        else:  # friendly
            return 1.0 - abs(response_formality - 0.5)
    
    def _detect_resolution_indicators(self, response: str) -> float:
        """Detect indicators that the response resolves the issue"""
        resolution_indicators = [
            'solved', 'resolved', 'fixed', 'completed', 'done',
            'should work', 'this will', 'you can now', 'problem is'
        ]
        
        response_lower = response.lower()
        indicator_count = sum(1 for indicator in resolution_indicators 
                            if indicator in response_lower)
        
        return min(1.0, indicator_count / 3)
    
    def _predict_followup_needed(self, response: str) -> float:
        """Predict if user will need follow-up questions"""
        followup_indicators = [
            'if you need', 'let me know', 'any questions', 'need help',
            'contact us', 'additional support', 'further assistance'
        ]
        
        response_lower = response.lower()
        indicator_count = sum(1 for indicator in followup_indicators 
                            if indicator in response_lower)
        
        return min(1.0, indicator_count / 2)
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Predict satisfaction score and confidence"""
        if not self.is_trained:
            # Use heuristic prediction if model not trained
            return self._heuristic_prediction(features)
        
        # Prepare features for model
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        feature_vector = self.scaler.transform([feature_vector])
        
        # Predict
        prediction = self.model.predict(feature_vector)[0]
        
        # Calculate confidence based on feature consistency
        confidence = self._calculate_prediction_confidence(features)
        
        return max(0.0, min(1.0, prediction)), confidence
    
    def _heuristic_prediction(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Heuristic satisfaction prediction when model not trained"""
        score = 0.5  # Base score
        
        # Positive factors
        score += features.get('completeness_score', 0) * 0.2
        score += features.get('user_expertise_match', 0) * 0.15
        score += features.get('response_tone_match', 0) * 0.1
        score += features.get('resolution_indicators', 0) * 0.25
        
        # Negative factors
        response_time = features.get('response_time', 0)
        if response_time > 3.0:
            score -= (response_time - 3.0) * 0.05
        
        if features.get('followup_needed', 0) > 0.7:
            score -= 0.1
        
        return max(0.0, min(1.0, score)), 0.6  # Medium confidence for heuristics
    
    def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in prediction"""
        confidence = 0.7  # Base confidence
        
        # Higher confidence if features are well-defined
        feature_completeness = sum(1 for v in features.values() if v > 0) / len(features)
        confidence *= feature_completeness
        
        # Higher confidence if user expertise match is good
        if features.get('user_expertise_match', 0) > 0.7:
            confidence += 0.1
        
        # Lower confidence if response time was very high
        if features.get('response_time', 0) > 5.0:
            confidence -= 0.2
        
        return max(0.3, min(1.0, confidence))
    
    def train_model(self, training_data: List[Dict]):
        """Train the satisfaction prediction model"""
        if len(training_data) < 50:
            logging.warning("Insufficient training data for satisfaction model")
            return
        
        X = []
        y = []
        
        for data_point in training_data:
            features = data_point['features']
            satisfaction = data_point['satisfaction_score']
            
            if satisfaction > 0:  # Valid satisfaction score
                feature_vector = [features.get(name, 0) for name in self.feature_names]
                X.append(feature_vector)
                y.append(satisfaction)
        
        if len(X) < 10:
            logging.warning("Too few valid training samples")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logging.info(f"Satisfaction prediction model trained on {len(X)} samples")

class ProactiveSuggester:
    """Generate proactive suggestions for users"""
    
    def __init__(self):
        self.suggestion_patterns = {}
        self.resource_database = {}
        self._initialize_resource_database()
    
    def _initialize_resource_database(self):
        """Initialize database of helpful resources"""
        self.resource_database = {
            'password_reset': {
                'title': 'Password Reset Guide',
                'description': 'Step-by-step guide to reset your password',
                'url': '/help/password-reset',
                'relevance_keywords': ['password', 'login', 'access', 'account']
            },
            'billing_questions': {
                'title': 'Billing FAQ',
                'description': 'Common questions about billing and payments',
                'url': '/help/billing-faq',
                'relevance_keywords': ['billing', 'payment', 'invoice', 'charge']
            },
            'technical_support': {
                'title': 'Technical Troubleshooting',
                'description': 'Troubleshoot common technical issues',
                'url': '/help/technical-support',
                'relevance_keywords': ['error', 'bug', 'technical', 'issue']
            }
        }
    
    def generate_suggestions(self, current_interaction: Dict, user_behavior: Dict) -> List[Suggestion]:
        """Generate proactive suggestions"""
        suggestions = []
        
        # Predict follow-up questions
        if user_behavior.get('tends_to_ask_followups', False):
            followup_suggestions = self._predict_followup_questions(current_interaction, user_behavior)
            suggestions.extend(followup_suggestions)
        
        # Suggest related resources
        if user_behavior.get('explores_related_topics', False):
            resource_suggestions = self._suggest_related_resources(current_interaction)
            suggestions.extend(resource_suggestions)
        
        # Proactive help based on patterns
        proactive_suggestions = self._generate_proactive_help(current_interaction, user_behavior)
        suggestions.extend(proactive_suggestions)
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:5]  # Top 5 suggestions
    
    def _predict_followup_questions(self, interaction: Dict, behavior: Dict) -> List[Suggestion]:
        """Predict likely follow-up questions"""
        suggestions = []
        query_type = interaction.get('query_type', 'general')
        
        followup_patterns = {
            'password_reset': [
                "What if I don't receive the reset email?",
                "How long does the reset link stay active?",
                "Can I reset my password from the mobile app?"
            ],
            'billing': [
                "When will I be charged next?",
                "How do I update my payment method?",
                "Can I get a refund?"
            ],
            'technical': [
                "What if this doesn't work?",
                "Are there alternative solutions?",
                "How do I prevent this issue in the future?"
            ]
        }
        
        if query_type in followup_patterns:
            for question in followup_patterns[query_type]:
                suggestions.append(Suggestion(
                    type='followup_question',
                    content=question,
                    confidence=0.7,
                    reason="Based on similar queries from other users"
                ))
        
        return suggestions
    
    def _suggest_related_resources(self, interaction: Dict) -> List[Suggestion]:
        """Suggest related helpful resources"""
        suggestions = []
        query = interaction.get('query', '').lower()
        
        for resource_id, resource in self.resource_database.items():
            relevance = self._calculate_resource_relevance(query, resource['relevance_keywords'])
            
            if relevance > 0.3:
                suggestions.append(Suggestion(
                    type='related_resource',
                    content=resource['description'],
                    link=resource['url'],
                    confidence=relevance,
                    reason="You might find this resource helpful",
                    metadata={'title': resource['title']}
                ))
        
        return suggestions
    
    def _calculate_resource_relevance(self, query: str, keywords: List[str]) -> float:
        """Calculate relevance of resource to query"""
        query_words = set(query.lower().split())
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query_words)
        
        return keyword_matches / len(keywords) if keywords else 0.0
    
    def _generate_proactive_help(self, interaction: Dict, behavior: Dict) -> List[Suggestion]:
        """Generate proactive help suggestions"""
        suggestions = []
        
        # Check if user might be frustrated
        escalation_triggers = behavior.get('escalation_triggers', [])
        if 'slow_response' in escalation_triggers:
            suggestions.append(Suggestion(
                type='proactive_help',
                content="I notice you prefer faster responses. Let me prioritize getting you a quick answer.",
                confidence=0.8,
                reason="Based on your preference for quick responses"
            ))
        
        # Check satisfaction trend
        if behavior.get('satisfaction_trend') == 'declining':
            suggestions.append(Suggestion(
                type='proactive_help',
                content="I want to make sure I'm providing the help you need. Would you like me to connect you with a human agent?",
                confidence=0.7,
                reason="To ensure you get the best possible support"
            ))
        
        # Suggest based on common query types
        common_queries = behavior.get('common_query_types', [])
        if 'technical' in common_queries:
            suggestions.append(Suggestion(
                type='proactive_help',
                content="Since you often have technical questions, would you like me to provide more detailed technical explanations?",
                confidence=0.6,
                reason="Based on your interest in technical topics"
            ))
        
        return suggestions

class PredictiveUserAssistant:
    """Main predictive assistant combining all prediction capabilities"""
    
    def __init__(self):
        self.user_behavior_model = UserBehaviorModel()
        self.satisfaction_predictor = SatisfactionPredictor()
        self.proactive_suggester = ProactiveSuggester()
        
        logging.info("Predictive User Assistant initialized")
    
    def predict_user_satisfaction(self, response: str, user_profile: Dict, query_context: Dict) -> SatisfactionPrediction:
        """Predict user satisfaction with response"""
        features = self.satisfaction_predictor.extract_features(response, user_profile, query_context)
        predicted_score, confidence = self.satisfaction_predictor.predict(features)
        
        # Generate improvement suggestions if satisfaction is predicted to be low
        improvements = []
        if predicted_score < 0.7:
            improvements = self._suggest_response_improvements(features, user_profile)
        
        return SatisfactionPrediction(
            predicted_score=predicted_score,
            confidence=confidence,
            factors=features,
            improvements=improvements
        )
    
    def _suggest_response_improvements(self, features: Dict, user_profile: Dict) -> List[str]:
        """Suggest improvements for low-satisfaction responses"""
        improvements = []
        
        if features.get('completeness_score', 0) < 0.5:
            improvements.append("Add more detailed steps or examples")
        
        if features.get('user_expertise_match', 0) < 0.5:
            user_expertise = user_profile.get('expertise_level', 'intermediate')
            if user_expertise == 'beginner':
                improvements.append("Simplify technical language and add more explanations")
            else:
                improvements.append("Provide more technical detail and advanced options")
        
        if features.get('resolution_indicators', 0) < 0.3:
            improvements.append("Include clearer resolution steps and confirmation")
        
        if features.get('response_time', 0) > 3.0:
            improvements.append("Optimize response time by reducing context or using cache")
        
        return improvements
    
    def generate_proactive_suggestions(self, user_id: str, current_interaction: Dict, user_history: List[Dict]) -> List[Suggestion]:
        """Generate proactive suggestions for user"""
        behavior_patterns = self.user_behavior_model.analyze_patterns(user_history)
        suggestions = self.proactive_suggester.generate_suggestions(current_interaction, behavior_patterns)
        
        return suggestions
    
    def should_escalate_to_human(self, user_satisfaction_history: List[float], current_interaction: Dict) -> Tuple[bool, str]:
        """Predict if interaction should be escalated to human agent"""
        
        # Check recent satisfaction trend
        if len(user_satisfaction_history) >= 3:
            recent_satisfaction = np.mean(user_satisfaction_history[-3:])
            if recent_satisfaction < 0.5:
                return True, "Low recent satisfaction scores indicate need for human assistance"
        
        # Check for explicit escalation requests
        query = current_interaction.get('query', '').lower()
        escalation_keywords = ['human', 'agent', 'person', 'speak to someone', 'manager']
        if any(keyword in query for keyword in escalation_keywords):
            return True, "User explicitly requested human assistance"
        
        # Check for repeated similar queries
        query_type = current_interaction.get('query_type')
        if current_interaction.get('repeat_query_count', 0) > 2:
            return True, f"User has repeated {query_type} queries multiple times"
        
        # Check for high complexity with low confidence
        if (current_interaction.get('query_complexity', 0) > 0.8 and 
            current_interaction.get('response_confidence', 1.0) < 0.6):
            return True, "High complexity query with low AI confidence"
        
        return False, ""
    
    def optimize_response_for_user(self, base_response: str, user_profile: Dict, prediction: SatisfactionPrediction) -> str:
        """Optimize response based on user profile and satisfaction prediction"""
        
        if prediction.predicted_score > 0.8:
            return base_response  # Response is already good
        
        optimized_response = base_response
        
        # Apply improvements based on prediction
        for improvement in prediction.improvements:
            if "simplify technical language" in improvement:
                optimized_response = self._simplify_technical_language(optimized_response)
            elif "add more detailed steps" in improvement:
                optimized_response = self._add_detail_markers(optimized_response)
            elif "clearer resolution steps" in improvement:
                optimized_response = self._add_resolution_markers(optimized_response)
        
        # Adjust based on user communication style
        communication_style = user_profile.get('communication_style', 'professional')
        if communication_style == 'casual':
            optimized_response = self._make_more_casual(optimized_response)
        elif communication_style == 'professional':
            optimized_response = self._make_more_formal(optimized_response)
        
        return optimized_response
    
    def _simplify_technical_language(self, response: str) -> str:
        """Simplify technical language in response"""
        # Simple replacements for common technical terms
        replacements = {
            'API': 'interface',
            'database': 'data storage',
            'server': 'computer system',
            'authentication': 'login verification',
            'configuration': 'settings'
        }
        
        for tech_term, simple_term in replacements.items():
            response = response.replace(tech_term, f"{simple_term} ({tech_term})")
        
        return response
    
    def _add_detail_markers(self, response: str) -> str:
        """Add markers to encourage more detailed explanations"""
        if "Here's how:" not in response and "Steps:" not in response:
            response = response + "\n\nHere's a detailed breakdown of the process:"
        
        return response
    
    def _add_resolution_markers(self, response: str) -> str:
        """Add clear resolution indicators"""
        if not any(indicator in response.lower() for indicator in ['this should resolve', 'you should now be able to', 'problem solved']):
            response = response + "\n\nThis should resolve your issue. Let me know if you need any clarification!"
        
        return response
    
    def _make_more_casual(self, response: str) -> str:
        """Make response more casual"""
        response = response.replace("Please", "")
        response = response.replace("Thank you", "Thanks")
        return response
    
    def _make_more_formal(self, response: str) -> str:
        """Make response more formal"""
        if not response.startswith("Thank you"):
            response = "Thank you for your question. " + response
        
        return response
