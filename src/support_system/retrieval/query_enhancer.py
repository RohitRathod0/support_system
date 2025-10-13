import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EnhancedQuery:
    original_query: str
    enhanced_queries: List[str]
    query_type: str
    intent: str
    entities: List[Dict[str, str]]
    synonyms: List[str]
    expansion_keywords: List[str]
    confidence_score: float

class IntentClassifier:
    """Classify user intent from queries"""
    
    def __init__(self):
        self.intent_patterns = {
            'question': [
                r'\b(what|how|why|when|where|which|who)\b',
                r'\?\s*$',
                r'\b(explain|tell me|show me)\b'
            ],
            'problem': [
                r'\b(error|issue|problem|bug|broken|not working|failed)\b',
                r'\b(can\'t|cannot|unable|won\'t|doesn\'t work)\b',
                r'\b(help|fix|solve|resolve)\b'
            ],
            'request': [
                r'\b(please|can you|could you|would you)\b',
                r'\b(need|want|require|looking for)\b',
                r'\b(get|provide|give|send)\b'
            ],
            'information': [
                r'\b(about|regarding|concerning|info|information)\b',
                r'\b(policy|procedure|rule|guideline)\b',
                r'\b(status|update|details)\b'
            ],
            'complaint': [
                r'\b(frustrated|angry|disappointed|upset)\b',
                r'\b(terrible|awful|horrible|worst)\b',
                r'\b(complain|complaint)\b'
            ]
        }
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify query intent with confidence score"""
        
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            
            # Normalize by query length
            intent_scores[intent] = score / len(query.split())
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return 'general', 0.5
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        return best_intent, min(1.0, confidence * 2)  # Scale confidence

class EntityExtractor:
    """Extract entities from queries using NLP"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found, using basic entity extraction")
            self.nlp = None
        
        # Custom entity patterns for support domain
        self.custom_patterns = {
            'error_code': r'\b[A-Z]{1,3}\d{3,6}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ticket_id': r'\b(ticket|case|ref)[\s#:]*(\d+)\b',
            'product_name': r'\b(app|application|software|system|platform|service)\b'
        }
    
    def extract_entities(self, query: str) -> List[Dict[str, str]]:
        """Extract entities from query"""
        
        entities = []
        
        # Custom pattern extraction
        for entity_type, pattern in self.custom_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(0),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9  # High confidence for pattern matches
                })
        
        # spaCy NER if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                # Filter relevant entity types
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'MONEY', 'DATE', 'TIME']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_.lower(),
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8
                    })
        
        return entities

class SynonymExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self):
        # Domain-specific synonym dictionary
        self.synonyms = {
            'error': ['issue', 'problem', 'bug', 'failure', 'fault'],
            'fix': ['solve', 'resolve', 'repair', 'correct', 'address'],
            'help': ['assist', 'support', 'aid', 'guidance'],
            'account': ['profile', 'user account', 'login', 'credentials'],
            'password': ['passcode', 'login credentials', 'authentication'],
            'payment': ['billing', 'charge', 'transaction', 'invoice'],
            'cancel': ['terminate', 'end', 'stop', 'discontinue'],
            'refund': ['reimbursement', 'money back', 'return payment'],
            'update': ['modify', 'change', 'edit', 'revise'],
            'delete': ['remove', 'erase', 'eliminate', 'clear']
        }
        
        # Technical synonyms
        self.technical_synonyms = {
            'app': ['application', 'software', 'program'],
            'website': ['site', 'web page', 'portal', 'platform'],
            'download': ['install', 'get', 'fetch'],
            'upload': ['submit', 'send', 'transfer'],
            'login': ['sign in', 'log in', 'access'],
            'logout': ['sign out', 'log out', 'exit']
        }
        
        self.synonyms.update(self.technical_synonyms)
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        word_lower = word.lower()
        
        # Direct lookup
        if word_lower in self.synonyms:
            return self.synonyms[word_lower]
        
        # Reverse lookup
        for key, syns in self.synonyms.items():
            if word_lower in syns:
                return [key] + [s for s in syns if s != word_lower]
        
        return []
    
    def expand_query_with_synonyms(self, query: str, max_synonyms: int = 3) -> List[str]:
        """Generate query variations using synonyms"""
        
        words = query.split()
        expanded_queries = [query]  # Include original
        
        for i, word in enumerate(words):
            synonyms = self.get_synonyms(word)
            if synonyms:
                # Create new queries with synonyms
                for synonym in synonyms[:max_synonyms]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    expanded_queries.append(' '.join(new_words))
        
        return list(set(expanded_queries))  # Remove duplicates

class QueryExpander:
    """Expand queries using various techniques"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.expansion_terms = {}
        self.query_log = []  # Store successful queries for learning
        
    def learn_from_query_log(self, query_log: List[Dict]):
        """Learn expansion terms from successful query patterns"""
        
        # Group queries by topic/category
        topic_queries = defaultdict(list)
        
        for log_entry in query_log:
            if log_entry.get('success', False):  # Only learn from successful queries
                category = log_entry.get('category', 'general')
                topic_queries[category].append(log_entry['query'])
        
        # Extract common terms for each topic
        for category, queries in topic_queries.items():
            common_terms = self._extract_common_terms(queries)
            self.expansion_terms[category] = common_terms
        
        logging.info(f"Learned expansion terms for {len(topic_queries)} categories")
    
    def _extract_common_terms(self, queries: List[str]) -> List[str]:
        """Extract common terms from a list of queries"""
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(queries)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms
            top_indices = np.argsort(mean_scores)[::-1][:10]
            return [feature_names[i] for i in top_indices]
            
        except Exception:
            return []
    
    def expand_query_semantically(self, query: str, category: str = None) -> List[str]:
        """Expand query using semantic similarity"""
        
        expanded_queries = [query]
        
        # Add category-specific terms if available
        if category and category in self.expansion_terms:
            expansion_terms = self.expansion_terms[category]
            
            # Create variations with expansion terms
            for term in expansion_terms[:3]:  # Top 3 terms
                expanded_queries.append(f"{query} {term}")
                
                # Also try replacing key terms
                words = query.split()
                if len(words) > 1:
                    # Replace last word with expansion term
                    new_query = ' '.join(words[:-1]) + f" {term}"
                    expanded_queries.append(new_query)
        
        return list(set(expanded_queries))
    
    def generate_question_variations(self, query: str) -> List[str]:
        """Generate different question formulations"""
        
        variations = [query]
        query_lower = query.lower()
        
        # Convert statements to questions
        if not query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'which', 'who')):
            variations.extend([
                f"How do I {query}",
                f"What is {query}",
                f"How to {query}",
                f"Why does {query}"
            ])
        
        # Convert questions to statements
        if query.endswith('?'):
            statement = query[:-1]
            variations.extend([
                f"I need help with {statement}",
                f"Issue with {statement}",
                f"Problem: {statement}"
            ])
        
        return variations

class QueryEnhancer:
    """Main query enhancement engine"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.synonym_expander = SynonymExpander()
        self.query_expander = QueryExpander(embedding_model)
        
        # Query processing statistics
        self.enhancement_stats = {
            'total_queries': 0,
            'intent_classifications': defaultdict(int),
            'entity_extractions': defaultdict(int),
            'synonym_expansions': 0,
            'semantic_expansions': 0
        }
        
        logging.info("Query Enhancer initialized")
    
    def enhance_query(self, 
                     query: str, 
                     context: Dict[str, Any] = None,
                     enhancement_level: str = "standard") -> EnhancedQuery:
        """
        Enhance query with multiple techniques
        
        Args:
            query: Original query
            context: Additional context (user history, category, etc.)
            enhancement_level: "minimal", "standard", "aggressive"
        """
        
        self.enhancement_stats['total_queries'] += 1
        
        # Step 1: Intent classification
        intent, intent_confidence = self.intent_classifier.classify_intent(query)
        self.enhancement_stats['intent_classifications'][intent] += 1
        
        # Step 2: Entity extraction
        entities = self.entity_extractor.extract_entities(query)
        for entity in entities:
            self.enhancement_stats['entity_extractions'][entity['label']] += 1
        
        # Step 3: Query type determination
        query_type = self._determine_query_type(query, intent, entities)
        
        # Step 4: Generate enhanced queries based on level
        enhanced_queries = [query]  # Always include original
        
        if enhancement_level in ["standard", "aggressive"]:
            # Synonym expansion
            synonym_queries = self.synonym_expander.expand_query_with_synonyms(query)
            enhanced_queries.extend(synonym_queries[1:])  # Exclude original
            if len(synonym_queries) > 1:
                self.enhancement_stats['synonym_expansions'] += 1
            
            # Question variations
            question_vars = self.query_expander.generate_question_variations(query)
            enhanced_queries.extend(question_vars[1:])  # Exclude original
        
        if enhancement_level == "aggressive":
            # Semantic expansion
            category = context.get('category') if context else None
            semantic_queries = self.query_expander.expand_query_semantically(query, category)
            enhanced_queries.extend(semantic_queries[1:])  # Exclude original
            if len(semantic_queries) > 1:
                self.enhancement_stats['semantic_expansions'] += 1
            
            # Context-based enhancement
            if context:
                context_queries = self._enhance_with_context(query, context)
                enhanced_queries.extend(context_queries)
        
        # Remove duplicates and limit number
        enhanced_queries = list(set(enhanced_queries))
        max_queries = {"minimal": 2, "standard": 5, "aggressive": 8}[enhancement_level]
        enhanced_queries = enhanced_queries[:max_queries]
        
        # Step 5: Extract expansion keywords
        expansion_keywords = self._extract_expansion_keywords(query, enhanced_queries)
        
        # Step 6: Get synonyms for key terms
        synonyms = self._extract_key_synonyms(query)
        
        # Calculate overall confidence
        confidence_score = self._calculate_enhancement_confidence(
            intent_confidence, entities, len(enhanced_queries)
        )
        
        return EnhancedQuery(
            original_query=query,
            enhanced_queries=enhanced_queries,
            query_type=query_type,
            intent=intent,
            entities=entities,
            synonyms=synonyms,
            expansion_keywords=expansion_keywords,
            confidence_score=confidence_score
        )
    
    def _determine_query_type(self, query: str, intent: str, entities: List[Dict]) -> str:
        """Determine the type of query for better processing"""
        
        query_lower = query.lower()
        
        # Technical query indicators
        if any(term in query_lower for term in ['error', 'code', 'api', 'system', 'bug']):
            return 'technical'
        
        # Policy query indicators
        if any(term in query_lower for term in ['policy', 'rule', 'allowed', 'procedure']):
            return 'policy'
        
        # Account query indicators
        if any(term in query_lower for term in ['account', 'profile', 'login', 'password']):
            return 'account'
        
        # Billing query indicators
        if any(term in query_lower for term in ['payment', 'billing', 'charge', 'refund']):
            return 'billing'
        
        # Based on intent
        if intent == 'problem':
            return 'troubleshooting'
        elif intent == 'question':
            return 'informational'
        elif intent == 'request':
            return 'service_request'
        
        return 'general'
    
    def _enhance_with_context(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Enhance query using context information"""
        
        enhanced = []
        
        # User history context
        if 'recent_queries' in context:
            recent_queries = context['recent_queries']
            if recent_queries:
                # Find common themes in recent queries
                for recent_query in recent_queries[-3:]:  # Last 3 queries
                    common_words = set(query.lower().split()) & set(recent_query.lower().split())
                    if common_words:
                        enhanced.append(f"{query} {' '.join(common_words)}")
        
        # Category context
        if 'category' in context:
            category = context['category']
            enhanced.append(f"{query} {category}")
        
        # User role context
        if 'user_role' in context:
            role = context['user_role']
            enhanced.append(f"{role} {query}")
        
        return enhanced[:3]  # Limit context expansions
    
    def _extract_expansion_keywords(self, original_query: str, enhanced_queries: List[str]) -> List[str]:
        """Extract keywords that were added during expansion"""
        
        original_words = set(original_query.lower().split())
        expansion_keywords = set()
        
        for enhanced_query in enhanced_queries:
            enhanced_words = set(enhanced_query.lower().split())
            new_words = enhanced_words - original_words
            expansion_keywords.update(new_words)
        
        return list(expansion_keywords)[:10]  # Top 10 expansion keywords
    
    def _extract_key_synonyms(self, query: str) -> List[str]:
        """Extract synonyms for key terms in the query"""
        
        words = query.split()
        all_synonyms = []
        
        for word in words:
            synonyms = self.synonym_expander.get_synonyms(word)
            all_synonyms.extend(synonyms[:2])  # Top 2 synonyms per word
        
        return all_synonyms[:8]  # Limit total synonyms
    
    def _calculate_enhancement_confidence(self, intent_confidence: float, 
                                        entities: List[Dict], 
                                        enhanced_count: int) -> float:
        """Calculate overall enhancement confidence"""
        
        confidence = 0.5  # Base confidence
        
        # Intent classification confidence
        confidence += intent_confidence * 0.3
        
        # Entity extraction confidence
        if entities:
            entity_confidence = np.mean([e.get('confidence', 0.5) for e in entities])
            confidence += entity_confidence * 0.2
        
        # Enhancement diversity (more variations = higher confidence)
        diversity_score = min(1.0, enhanced_count / 5)
        confidence += diversity_score * 0.2
        
        return min(1.0, confidence)
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get query enhancement statistics"""
        
        stats = dict(self.enhancement_stats)
        
        # Convert defaultdict to regular dict
        stats['intent_classifications'] = dict(stats['intent_classifications'])
        stats['entity_extractions'] = dict(stats['entity_extractions'])
        
        # Calculate rates
        if stats['total_queries'] > 0:
            stats['synonym_expansion_rate'] = stats['synonym_expansions'] / stats['total_queries']
            stats['semantic_expansion_rate'] = stats['semantic_expansions'] / stats['total_queries']
        
        return stats
    
    def reset_statistics(self):
        """Reset enhancement statistics"""
        
        self.enhancement_stats = {
            'total_queries': 0,
            'intent_classifications': defaultdict(int),
            'entity_extractions': defaultdict(int),
            'synonym_expansions': 0,
            'semantic_expansions': 0
        }
