import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.schema import Document
import spacy
from sentence_transformers import SentenceTransformer
import tiktoken

@dataclass
class PolicyChunk:
    content: str
    policy_section: str
    rule_category: str
    authority_level: str
    applies_to: List[str]
    related_policies: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[any] = None

class PolicyAwareChunker:
    """Specialized chunker for company policy documents that preserves rule structure"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Load spacy for NER and structure detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found, using basic parsing")
            self.nlp = None
        
        # Policy structure patterns
        self.section_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^.]+)',  # Numbered sections
            r'^([A-Z][A-Z\s]+):',  # ALL CAPS sections
            r'^(Section|Article|Chapter)\s+(\w+)',  # Named sections
            r'^([A-Z][a-z]+ Policy)',  # Policy titles
        ]
        
        # Authority level indicators
        self.authority_indicators = {
            'employee': ['employee', 'staff', 'worker', 'personnel'],
            'supervisor': ['supervisor', 'manager', 'team lead'],
            'hr': ['human resources', 'hr department', 'hr'],
            'executive': ['executive', 'director', 'ceo', 'president'],
            'legal': ['legal', 'compliance', 'attorney']
        }
        
        # Rule category patterns
        self.category_patterns = {
            'attendance': ['attendance', 'punctuality', 'time off', 'vacation', 'sick leave'],
            'conduct': ['conduct', 'behavior', 'harassment', 'discrimination'],
            'security': ['security', 'confidential', 'data protection', 'access'],
            'compensation': ['salary', 'wage', 'bonus', 'benefits', 'payment'],
            'disciplinary': ['disciplinary', 'violation', 'termination', 'warning'],
            'safety': ['safety', 'health', 'emergency', 'accident']
        }
        
        logging.info("Policy-Aware Chunker initialized")
    
    def chunk_policy_document(self, document: Document) -> List[PolicyChunk]:
        """Chunk policy document maintaining rule coherence"""
        
        text = document.page_content
        
        # Identify document structure
        sections = self._identify_policy_sections(text)
        
        chunks = []
        for section in sections:
            # Extract rules from each section
            rules = self._extract_policy_rules(section)
            
            for rule in rules:
                chunk = self._create_policy_chunk(rule, section, document.metadata)
                chunks.append(chunk)
        
        # Post-process to ensure optimal chunking
        optimized_chunks = self._optimize_policy_chunks(chunks)
        
        return optimized_chunks
    
    def _identify_policy_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify major sections in policy document"""
        
        sections = []
        lines = text.split('\n')
        
        current_section = {
            'title': 'Introduction',
            'content': '',
            'start_line': 0,
            'level': 0
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_match = self._is_section_header(line)
            
            if section_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': section_match['title'],
                    'content': '',
                    'start_line': i,
                    'level': section_match['level'],
                    'section_number': section_match.get('number', '')
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, line: str) -> Optional[Dict[str, Any]]:
        """Check if line is a section header"""
        
        for pattern in self.section_patterns:
            match = re.match(pattern, line)
            if match:
                return {
                    'title': match.group(2) if len(match.groups()) > 1 else match.group(1),
                    'number': match.group(1) if len(match.groups()) > 1 else '',
                    'level': self._determine_section_level(match.group(1) if len(match.groups()) > 1 else match.group(1))
                }
        
        return None
    
    def _determine_section_level(self, section_indicator: str) -> int:
        """Determine hierarchical level of section"""
        
        if re.match(r'^\d+$', section_indicator):
            return 1
        elif re.match(r'^\d+\.\d+$', section_indicator):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+$', section_indicator):
            return 3
        else:
            return 1
    
    def _extract_policy_rules(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual rules from a policy section"""
        
        content = section['content']
        rules = []
        
        # Split by common rule delimiters
        potential_rules = re.split(r'\n(?=\d+\.|\w+\.|\-|\*)', content)
        
        for rule_text in potential_rules:
            rule_text = rule_text.strip()
            if len(rule_text) < 20:  # Skip very short fragments
                continue
            
            # Analyze rule content
            rule_analysis = self._analyze_rule_content(rule_text)
            
            rule = {
                'text': rule_text,
                'section_title': section['title'],
                'category': rule_analysis['category'],
                'authority_level': rule_analysis['authority_level'],
                'applies_to': rule_analysis['applies_to'],
                'enforcement_level': rule_analysis['enforcement_level'],
                'related_concepts': rule_analysis['related_concepts'],
                'is_actionable': rule_analysis['is_actionable']
            }
            
            rules.append(rule)
        
        return rules
    
    def _analyze_rule_content(self, rule_text: str) -> Dict[str, Any]:
        """Analyze rule content to extract metadata"""
        
        text_lower = rule_text.lower()
        
        # Determine category
        category = 'general'
        for cat, keywords in self.category_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                category = cat
                break
        
        # Determine authority level
        authority_level = 'employee'
        for level, keywords in self.authority_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                authority_level = level
                break
        
        # Extract who the rule applies to
        applies_to = self._extract_applies_to(rule_text)
        
        # Determine enforcement level
        enforcement_level = self._determine_enforcement_level(rule_text)
        
        # Extract related concepts using NER
        related_concepts = self._extract_related_concepts(rule_text)
        
        # Check if rule is actionable (contains imperatives)
        is_actionable = self._is_actionable_rule(rule_text)
        
        return {
            'category': category,
            'authority_level': authority_level,
            'applies_to': applies_to,
            'enforcement_level': enforcement_level,
            'related_concepts': related_concepts,
            'is_actionable': is_actionable
        }
    
    def _extract_applies_to(self, rule_text: str) -> List[str]:
        """Extract who the rule applies to"""
        
        applies_to = []
        text_lower = rule_text.lower()
        
        # Look for explicit mentions
        if 'all employees' in text_lower or 'all staff' in text_lower:
            applies_to.append('all_employees')
        elif 'managers' in text_lower or 'supervisors' in text_lower:
            applies_to.append('managers')
        elif 'new employees' in text_lower or 'new hires' in text_lower:
            applies_to.append('new_employees')
        elif 'contractors' in text_lower:
            applies_to.append('contractors')
        else:
            applies_to.append('all_employees')  # Default
        
        return applies_to
    
    def _determine_enforcement_level(self, rule_text: str) -> str:
        """Determine how strictly the rule is enforced"""
        
        text_lower = rule_text.lower()
        
        if any(word in text_lower for word in ['must', 'required', 'mandatory', 'shall']):
            return 'mandatory'
        elif any(word in text_lower for word in ['should', 'recommended', 'encouraged']):
            return 'recommended'
        elif any(word in text_lower for word in ['may', 'can', 'optional']):
            return 'optional'
        else:
            return 'standard'
    
    def _extract_related_concepts(self, rule_text: str) -> List[str]:
        """Extract related concepts using NLP"""
        
        concepts = []
        
        if self.nlp:
            doc = self.nlp(rule_text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'EVENT']:
                    concepts.append(ent.text.lower())
            
            # Extract important nouns
            for token in doc:
                if (token.pos_ == 'NOUN' and 
                    token.text.lower() not in ['policy', 'rule', 'employee'] and
                    len(token.text) > 3):
                    concepts.append(token.text.lower())
        
        return list(set(concepts))[:5]  # Top 5 unique concepts
    
    def _is_actionable_rule(self, rule_text: str) -> bool:
        """Check if rule contains actionable directives"""
        
        text_lower = rule_text.lower()
        action_verbs = ['must', 'shall', 'will', 'should', 'report', 'contact', 
                       'submit', 'complete', 'attend', 'notify']
        
        return any(verb in text_lower for verb in action_verbs)
    
    def _create_policy_chunk(self, 
                            rule: Dict[str, Any], 
                            section: Dict[str, Any], 
                            base_metadata: Dict) -> PolicyChunk:
        """Create a policy chunk from rule analysis"""
        
        # Generate embedding
        embedding = self.embedding_model.encode([rule['text']])[0]
        
        # Create comprehensive metadata
        metadata = {
            **base_metadata,
            'token_count': len(self.encoding.encode(rule['text'])),
            'section_level': section['level'],
            'section_number': section.get('section_number', ''),
            'enforcement_level': rule['enforcement_level'],
            'is_actionable': rule['is_actionable'],
            'related_concepts': rule['related_concepts'],
            'chunk_type': 'policy_rule'
        }
        
        return PolicyChunk(
            content=rule['text'],
            policy_section=section['title'],
            rule_category=rule['category'],
            authority_level=rule['authority_level'],
            applies_to=rule['applies_to'],
            related_policies=[],  # Will be populated by cross-reference analysis
            metadata=metadata,
            embedding=embedding
        )
    
    def _optimize_policy_chunks(self, chunks: List[PolicyChunk]) -> List[PolicyChunk]:
        """Optimize policy chunks for better retrieval"""
        
        optimized_chunks = []
        
        # Group related chunks
        related_groups = self._find_related_policy_chunks(chunks)
        
        for group in related_groups:
            if len(group) == 1:
                optimized_chunks.append(group[0])
            else:
                # Consider merging closely related chunks
                merged_chunk = self._merge_related_policy_chunks(group)
                if merged_chunk:
                    optimized_chunks.append(merged_chunk)
                else:
                    optimized_chunks.extend(group)
        
        # Add cross-references
        self._add_cross_references(optimized_chunks)
        
        return optimized_chunks
    
    def _find_related_policy_chunks(self, chunks: List[PolicyChunk]) -> List[List[PolicyChunk]]:
        """Find groups of related policy chunks"""
        
        groups = []
        unprocessed = chunks.copy()
        
        while unprocessed:
            current_chunk = unprocessed.pop(0)
            related_group = [current_chunk]
            
            # Find chunks with same category and authority level
            remaining = []
            for chunk in unprocessed:
                if (chunk.rule_category == current_chunk.rule_category and
                    chunk.authority_level == current_chunk.authority_level and
                    chunk.policy_section == current_chunk.policy_section):
                    related_group.append(chunk)
                else:
                    remaining.append(chunk)
            
            unprocessed = remaining
            groups.append(related_group)
        
        return groups
    
    def _merge_related_policy_chunks(self, chunks: List[PolicyChunk]) -> Optional[PolicyChunk]:
        """Merge related policy chunks if beneficial"""
        
        # Only merge if chunks are small and very related
        total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in chunks)
        
        if total_tokens > 800 or len(chunks) > 3:
            return None  # Don't merge large or many chunks
        
        # Merge content
        merged_content = "\n\n".join(chunk.content for chunk in chunks)
        
        # Combine metadata
        all_concepts = []
        for chunk in chunks:
            all_concepts.extend(chunk.metadata.get('related_concepts', []))
        
        merged_metadata = {
            **chunks[0].metadata,
            'token_count': len(self.encoding.encode(merged_content)),
            'merged_from': len(chunks),
            'related_concepts': list(set(all_concepts)),
            'chunk_type': 'merged_policy_rules'
        }
        
        # Generate new embedding
        embedding = self.embedding_model.encode([merged_content])[0]
        
        return PolicyChunk(
            content=merged_content,
            policy_section=chunks[0].policy_section,
            rule_category=chunks[0].rule_category,
            authority_level=chunks[0].authority_level,
            applies_to=list(set(sum([chunk.applies_to for chunk in chunks], []))),
            related_policies=[],
            metadata=merged_metadata,
            embedding=embedding
        )
    
    def _add_cross_references(self, chunks: List[PolicyChunk]):
        """Add cross-references between related policy chunks"""
        
        for i, chunk in enumerate(chunks):
            related_policies = []
            
            for j, other_chunk in enumerate(chunks):
                if i != j:
                    # Check for conceptual overlap
                    chunk_concepts = set(chunk.metadata.get('related_concepts', []))
                    other_concepts = set(other_chunk.metadata.get('related_concepts', []))
                    
                    overlap = len(chunk_concepts.intersection(other_concepts))
                    if overlap > 0:
                        related_policies.append({
                            'section': other_chunk.policy_section,
                            'category': other_chunk.rule_category,
                            'overlap_score': overlap / len(chunk_concepts.union(other_concepts))
                        })
            
            # Keep top 3 most related policies
            related_policies.sort(key=lambda x: x['overlap_score'], reverse=True)
            chunk.related_policies = related_policies[:3]
    
    def get_policy_statistics(self, chunks: List[PolicyChunk]) -> Dict[str, Any]:
        """Get statistics about policy chunks"""
        
        if not chunks:
            return {}
        
        # Category distribution
        categories = {}
        authority_levels = {}
        enforcement_levels = {}
        
        for chunk in chunks:
            categories[chunk.rule_category] = categories.get(chunk.rule_category, 0) + 1
            authority_levels[chunk.authority_level] = authority_levels.get(chunk.authority_level, 0) + 1
            enforcement_level = chunk.metadata.get('enforcement_level', 'unknown')
            enforcement_levels[enforcement_level] = enforcement_levels.get(enforcement_level, 0) + 1
        
        total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in chunks)
        actionable_rules = sum(1 for chunk in chunks if chunk.metadata.get('is_actionable', False))
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_chunk_size': total_tokens / len(chunks),
            'actionable_rules': actionable_rules,
            'actionable_percentage': (actionable_rules / len(chunks)) * 100,
            'category_distribution': categories,
            'authority_distribution': authority_levels,
            'enforcement_distribution': enforcement_levels,
            'cross_references': sum(len(chunk.related_policies) for chunk in chunks)
        }
