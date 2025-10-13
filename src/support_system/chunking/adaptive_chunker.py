import re
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import tiktoken

class ContentType(Enum):
    POLICY = "policy"
    FAQ = "faq"
    TECHNICAL = "technical"
    PROCEDURE = "procedure"
    NARRATIVE = "narrative"
    STRUCTURED = "structured"
    CONVERSATIONAL = "conversational"

@dataclass
class AdaptiveChunk:
    content: str
    content_type: ContentType
    chunk_strategy: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    confidence_score: float = 0.0

class ContentTypeClassifier:
    """Classify content type for adaptive chunking"""
    
    def __init__(self):
        self.patterns = {
            ContentType.POLICY: [
                r'policy|procedure|guideline|regulation',
                r'must|shall|required|mandatory',
                r'violation|compliance|disciplinary',
                r'section \d+|article \d+|clause \d+'
            ],
            ContentType.FAQ: [
                r'q:|question:|a:|answer:',
                r'frequently asked|common questions',
                r'how to|what is|why does|when should'
            ],
            ContentType.TECHNICAL: [
                r'api|database|server|configuration',
                r'error code|troubleshoot|diagnostic',
                r'install|setup|configure|debug',
                r'system|network|software|hardware'
            ],
            ContentType.PROCEDURE: [
                r'step \d+|first|then|next|finally',
                r'process|workflow|instructions',
                r'complete|finish|submit|approve'
            ],
            ContentType.STRUCTURED: [
                r'table|list|bullet|numbered',
                r'\d+\.|•|\*|\-',
                r'column|row|field|entry'
            ],
            ContentType.CONVERSATIONAL: [
                r'hi|hello|thanks|please|sorry',
                r'i am|you are|we can|let me',
                r'customer|user|client|support'
            ]
        }
    
    def classify_content(self, text: str) -> ContentType:
        """Classify content type based on patterns"""
        
        text_lower = text.lower()
        scores = {}
        
        for content_type, pattern_list in self.patterns.items():
            score = 0
            for pattern in pattern_list:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            # Normalize by text length
            scores[content_type] = score / (len(text.split()) / 100)
        
        # Return type with highest score, default to NARRATIVE
        if not scores or max(scores.values()) == 0:
            return ContentType.NARRATIVE
        
        return max(scores, key=scores.get)

class AdaptiveChunker:
    """Adaptive chunker that selects optimal strategy based on content type"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.content_classifier = ContentTypeClassifier()
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Strategy configurations
        self.strategies = {
            ContentType.POLICY: {
                'max_chunk_size': 800,
                'overlap': 100,
                'preserve_structure': True,
                'split_on_sections': True
            },
            ContentType.FAQ: {
                'max_chunk_size': 400,
                'overlap': 50,
                'preserve_qa_pairs': True,
                'split_on_sections': False
            },
            ContentType.TECHNICAL: {
                'max_chunk_size': 1000,
                'overlap': 150,
                'preserve_code_blocks': True,
                'split_on_sections': True
            },
            ContentType.PROCEDURE: {
                'max_chunk_size': 600,
                'overlap': 100,
                'preserve_steps': True,
                'split_on_sections': True
            },
            ContentType.NARRATIVE: {
                'max_chunk_size': 1200,
                'overlap': 200,
                'preserve_paragraphs': True,
                'split_on_sections': False
            },
            ContentType.STRUCTURED: {
                'max_chunk_size': 500,
                'overlap': 50,
                'preserve_lists': True,
                'split_on_sections': True
            },
            ContentType.CONVERSATIONAL: {
                'max_chunk_size': 300,
                'overlap': 50,
                'preserve_exchanges': True,
                'split_on_sections': False
            }
        }
        
        logging.info("Adaptive Chunker initialized")
    
    def chunk_document(self, document: Document) -> List[AdaptiveChunk]:
        """Adaptively chunk document based on content type"""
        
        # Classify content type
        content_type = self.content_classifier.classify_content(document.page_content)
        
        # Get strategy for content type
        strategy = self.strategies[content_type]
        
        # Apply appropriate chunking method
        if content_type == ContentType.POLICY:
            chunks = self._chunk_policy_content(document, strategy)
        elif content_type == ContentType.FAQ:
            chunks = self._chunk_faq_content(document, strategy)
        elif content_type == ContentType.TECHNICAL:
            chunks = self._chunk_technical_content(document, strategy)
        elif content_type == ContentType.PROCEDURE:
            chunks = self._chunk_procedure_content(document, strategy)
        elif content_type == ContentType.STRUCTURED:
            chunks = self._chunk_structured_content(document, strategy)
        elif content_type == ContentType.CONVERSATIONAL:
            chunks = self._chunk_conversational_content(document, strategy)
        else:  # NARRATIVE
            chunks = self._chunk_narrative_content(document, strategy)
        
        # Add embeddings and confidence scores
        self._add_embeddings_and_scores(chunks)
        
        return chunks
    
    def _chunk_policy_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk policy content preserving rule structure"""
        
        text = document.page_content
        chunks = []
        
        # Split on policy sections
        sections = re.split(r'\n(?=\d+\.|\w+\.)', text)
        
        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue
            
            # Further split if section is too large
            if len(self.encoding.encode(section)) > strategy['max_chunk_size']:
                subsections = self._split_by_sentences(section, strategy['max_chunk_size'], strategy['overlap'])
                for subsection in subsections:
                    chunks.append(AdaptiveChunk(
                        content=subsection,
                        content_type=ContentType.POLICY,
                        chunk_strategy="policy_section_split",
                        metadata={**document.metadata, "token_count": len(self.encoding.encode(subsection))}
                    ))
            else:
                chunks.append(AdaptiveChunk(
                    content=section,
                    content_type=ContentType.POLICY,
                    chunk_strategy="policy_section",
                    metadata={**document.metadata, "token_count": len(self.encoding.encode(section))}
                ))
        
        return chunks
    
    def _chunk_faq_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk FAQ content preserving Q&A pairs"""
        
        text = document.page_content
        chunks = []
        
        # Find Q&A pairs
        qa_pattern = r'(q:|question:)(.*?)(?=q:|question:|$)'
        qa_matches = re.finditer(qa_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in qa_matches:
            qa_content = match.group(0).strip()
            if len(qa_content) > 20:
                chunks.append(AdaptiveChunk(
                    content=qa_content,
                    content_type=ContentType.FAQ,
                    chunk_strategy="qa_pair",
                    metadata={**document.metadata, "token_count": len(self.encoding.encode(qa_content))}
                ))
        
        # If no Q&A patterns found, fall back to sentence splitting
        if not chunks:
            sentences = self._split_by_sentences(text, strategy['max_chunk_size'], strategy['overlap'])
            for sentence_group in sentences:
                chunks.append(AdaptiveChunk(
                    content=sentence_group,
                    content_type=ContentType.FAQ,
                    chunk_strategy="sentence_split",
                    metadata={**document.metadata, "token_count": len(self.encoding.encode(sentence_group))}
                ))
        
        return chunks
    
    def _chunk_technical_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk technical content preserving code blocks and procedures"""
        
        text = document.page_content
        chunks = []
        
        # Preserve code blocks
        code_pattern = r'``````|`[^`]+`'
        parts = re.split(f'({code_pattern})', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for part in parts:
            part_tokens = len(self.encoding.encode(part))
            
            if re.match(code_pattern, part):
                # This is a code block - try to keep it intact
                if current_tokens + part_tokens <= strategy['max_chunk_size']:
                    current_chunk += part
                    current_tokens += part_tokens
                else:
                    # Finalize current chunk
                    if current_chunk.strip():
                        chunks.append(AdaptiveChunk(
                            content=current_chunk.strip(),
                            content_type=ContentType.TECHNICAL,
                            chunk_strategy="technical_with_code",
                            metadata={**document.metadata, "token_count": current_tokens}
                        ))
                    
                    # Start new chunk with code block
                    current_chunk = part
                    current_tokens = part_tokens
            else:
                # Regular text - split by sentences if needed
                if current_tokens + part_tokens <= strategy['max_chunk_size']:
                    current_chunk += part
                    current_tokens += part_tokens
                else:
                    # Finalize current chunk
                    if current_chunk.strip():
                        chunks.append(AdaptiveChunk(
                            content=current_chunk.strip(),
                            content_type=ContentType.TECHNICAL,
                            chunk_strategy="technical_split",
                            metadata={**document.metadata, "token_count": current_tokens}
                        ))
                    
                    # Split remaining part
                    remaining_chunks = self._split_by_sentences(part, strategy['max_chunk_size'], strategy['overlap'])
                    for chunk_text in remaining_chunks:
                        chunks.append(AdaptiveChunk(
                            content=chunk_text,
                            content_type=ContentType.TECHNICAL,
                            chunk_strategy="technical_sentence_split",
                            metadata={**document.metadata, "token_count": len(self.encoding.encode(chunk_text))}
                        ))
                    
                    current_chunk = ""
                    current_tokens = 0
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(AdaptiveChunk(
                content=current_chunk.strip(),
                content_type=ContentType.TECHNICAL,
                chunk_strategy="technical_final",
                metadata={**document.metadata, "token_count": current_tokens}
            ))
        
        return chunks
    
    def _chunk_procedure_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk procedural content preserving step sequences"""
        
        text = document.page_content
        chunks = []
        
        # Find step sequences
        step_pattern = r'(step \d+|^\d+\.|\d+\))'
        lines = text.split('\n')
        
        current_procedure = []
        current_tokens = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_tokens = len(self.encoding.encode(line))
            
            if re.match(step_pattern, line.lower()) or current_tokens + line_tokens > strategy['max_chunk_size']:
                # Start new procedure chunk
                if current_procedure:
                    chunk_content = '\n'.join(current_procedure)
                    chunks.append(AdaptiveChunk(
                        content=chunk_content,
                        content_type=ContentType.PROCEDURE,
                        chunk_strategy="procedure_steps",
                        metadata={**document.metadata, "token_count": current_tokens}
                    ))
                
                current_procedure = [line]
                current_tokens = line_tokens
            else:
                current_procedure.append(line)
                current_tokens += line_tokens
        
        # Add final procedure
        if current_procedure:
            chunk_content = '\n'.join(current_procedure)
            chunks.append(AdaptiveChunk(
                content=chunk_content,
                content_type=ContentType.PROCEDURE,
                chunk_strategy="procedure_final",
                metadata={**document.metadata, "token_count": current_tokens}
            ))
        
        return chunks
    
    def _chunk_structured_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk structured content preserving lists and tables"""
        
        text = document.page_content
        chunks = []
        
        # Split on major structural elements
        structural_breaks = r'\n(?=•|\*|\d+\.|\-\s)'
        parts = re.split(structural_breaks, text)
        
        current_chunk = ""
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = len(self.encoding.encode(part))
            
            if current_tokens + part_tokens <= strategy['max_chunk_size']:
                current_chunk += '\n' + part if current_chunk else part
                current_tokens += part_tokens
            else:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(AdaptiveChunk(
                        content=current_chunk,
                        content_type=ContentType.STRUCTURED,
                        chunk_strategy="structured_list",
                        metadata={**document.metadata, "token_count": current_tokens}
                    ))
                
                # Start new chunk
                current_chunk = part
                current_tokens = part_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(AdaptiveChunk(
                content=current_chunk,
                content_type=ContentType.STRUCTURED,
                chunk_strategy="structured_final",
                metadata={**document.metadata, "token_count": current_tokens}
            ))
        
        return chunks
    
    def _chunk_conversational_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk conversational content preserving exchange flows"""
        
        text = document.page_content
        chunks = []
        
        # Split on conversation turns
        conversation_breaks = r'\n(?=user:|customer:|agent:|assistant:)'
        exchanges = re.split(conversation_breaks, text, flags=re.IGNORECASE)
        
        current_exchange = ""
        current_tokens = 0
        
        for exchange in exchanges:
            exchange = exchange.strip()
            if not exchange:
                continue
            
            exchange_tokens = len(self.encoding.encode(exchange))
            
            if current_tokens + exchange_tokens <= strategy['max_chunk_size']:
                current_exchange += '\n' + exchange if current_exchange else exchange
                current_tokens += exchange_tokens
            else:
                # Finalize current exchange
                if current_exchange:
                    chunks.append(AdaptiveChunk(
                        content=current_exchange,
                        content_type=ContentType.CONVERSATIONAL,
                        chunk_strategy="conversation_exchange",
                        metadata={**document.metadata, "token_count": current_tokens}
                    ))
                
                # Start new exchange
                current_exchange = exchange
                current_tokens = exchange_tokens
        
        # Add final exchange
        if current_exchange:
            chunks.append(AdaptiveChunk(
                content=current_exchange,
                content_type=ContentType.CONVERSATIONAL,
                chunk_strategy="conversation_final",
                metadata={**document.metadata, "token_count": current_tokens}
            ))
        
        return chunks
    
    def _chunk_narrative_content(self, document: Document, strategy: Dict) -> List[AdaptiveChunk]:
        """Chunk narrative content preserving paragraph flow"""
        
        text = document.page_content
        chunks = []
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(self.encoding.encode(paragraph))
            
            if current_tokens + paragraph_tokens <= strategy['max_chunk_size']:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens
            else:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(AdaptiveChunk(
                        content=current_chunk,
                        content_type=ContentType.NARRATIVE,
                        chunk_strategy="narrative_paragraph",
                        metadata={**document.metadata, "token_count": current_tokens}
                    ))
                
                # Handle large paragraphs
                if paragraph_tokens > strategy['max_chunk_size']:
                    # Split large paragraph by sentences
                    sentence_chunks = self._split_by_sentences(paragraph, strategy['max_chunk_size'], strategy['overlap'])
                    for sentence_chunk in sentence_chunks:
                        chunks.append(AdaptiveChunk(
                            content=sentence_chunk,
                            content_type=ContentType.NARRATIVE,
                            chunk_strategy="narrative_sentence_split",
                            metadata={**document.metadata, "token_count": len(self.encoding.encode(sentence_chunk))}
                        ))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(AdaptiveChunk(
                content=current_chunk,
                content_type=ContentType.NARRATIVE,
                chunk_strategy="narrative_final",
                metadata={**document.metadata, "token_count": current_tokens}
            ))
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Split text by sentences with token awareness"""
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence + '. '
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + sentence + '. '
                    current_tokens = len(self.encoding.encode(current_chunk))
                else:
                    current_chunk = sentence + '. '
                    current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text for chunk continuity"""
        
        words = text.split()
        overlap_words = []
        current_tokens = 0
        
        for word in reversed(words):
            word_tokens = len(self.encoding.encode(word))
            if current_tokens + word_tokens <= overlap_tokens:
                overlap_words.insert(0, word)
                current_tokens += word_tokens
            else:
                break
        
        return ' '.join(overlap_words) + ' ' if overlap_words else ''
    
    def _add_embeddings_and_scores(self, chunks: List[AdaptiveChunk]):
        """Add embeddings and confidence scores to chunks"""
        
        for chunk in chunks:
            # Generate embedding
            chunk.embedding = self.embedding_model.encode([chunk.content])[0]
            
            # Calculate confidence score based on chunk quality
            chunk.confidence_score = self._calculate_chunk_confidence(chunk)
    
    def _calculate_chunk_confidence(self, chunk: AdaptiveChunk) -> float:
        """Calculate confidence score for chunk quality"""
        
        confidence = 0.5  # Base confidence
        
        # Size appropriateness
        token_count = chunk.metadata.get('token_count', 0)
        if 100 <= token_count <= 800:
            confidence += 0.2
        elif token_count < 50:
            confidence -= 0.2
        
        # Content type alignment
        if chunk.chunk_strategy.startswith(chunk.content_type.value):
            confidence += 0.15
        
        # Structure preservation
        if any(indicator in chunk.chunk_strategy for indicator in ['section', 'pair', 'steps', 'exchange']):
            confidence += 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def get_chunking_statistics(self, chunks: List[AdaptiveChunk]) -> Dict[str, Any]:
        """Get statistics about adaptive chunking results"""
        
        if not chunks:
            return {}
        
        # Group by content type
        type_stats = {}
        for chunk in chunks:
            content_type = chunk.content_type.value
            if content_type not in type_stats:
                type_stats[content_type] = {'count': 0, 'total_tokens': 0, 'strategies': set()}
            
            type_stats[content_type]['count'] += 1
            type_stats[content_type]['total_tokens'] += chunk.metadata.get('token_count', 0)
            type_stats[content_type]['strategies'].add(chunk.chunk_strategy)
        
        # Convert sets to lists for JSON serialization
        for stats in type_stats.values():
            stats['strategies'] = list(stats['strategies'])
        
        total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in chunks)
        avg_confidence = np.mean([chunk.confidence_score for chunk in chunks])
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_chunk_size': total_tokens / len(chunks),
            'avg_confidence': avg_confidence,
            'content_type_distribution': type_stats,
            'strategy_usage': {chunk.chunk_strategy: 1 for chunk in chunks}
        }
