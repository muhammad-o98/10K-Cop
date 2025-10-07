"""
Text Chunker Module
Creates RAG-ready chunks with metadata for semantic search and QA
"""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tiktoken
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    chunk_id: str
    document_id: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 0
    word_count: int = 0
    token_count: int = 0
    
    # Filing metadata
    ticker: Optional[str] = None
    cik: Optional[str] = None
    fiscal_year: Optional[int] = None
    filing_date: Optional[str] = None
    form_type: Optional[str] = None
    accession_number: Optional[str] = None
    
    # Chunk relationships
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    overlap_with_prev: int = 0
    overlap_with_next: int = 0
    
    # Semantic metadata
    has_financial_data: bool = False
    has_risk_mention: bool = False
    has_forward_looking: bool = False
    entities_mentioned: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary"""
        return asdict(self)


class TextChunker:
    """
    Creates overlapping text chunks optimized for RAG applications
    """
    
    # Common section headers to preserve
    SECTION_HEADERS = [
        'Business Overview', 'Risk Factors', 'Financial Performance',
        'Management Discussion', 'Market Risk', 'Competition',
        'Revenue Recognition', 'Critical Accounting', 'Segment Information'
    ]
    
    # Financial keywords for metadata tagging
    FINANCIAL_KEYWORDS = [
        'revenue', 'income', 'expense', 'profit', 'loss', 'margin',
        'cash flow', 'assets', 'liabilities', 'equity', 'debt',
        'earnings', 'ebitda', 'roi', 'roe', 'eps'
    ]
    
    # Risk-related keywords
    RISK_KEYWORDS = [
        'risk', 'uncertainty', 'volatility', 'exposure', 'threat',
        'challenge', 'adverse', 'negative', 'decline', 'loss'
    ]
    
    # Forward-looking statement keywords
    FORWARD_KEYWORDS = [
        'expect', 'anticipate', 'believe', 'intend', 'plan',
        'forecast', 'project', 'estimate', 'will', 'may', 'could',
        'should', 'potential', 'future', 'outlook'
    ]
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 use_token_chunking: bool = True,
                 model_name: str = "gpt-3.5-turbo",
                 preserve_sentences: bool = True,
                 preserve_paragraphs: bool = False):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size for chunks (tokens or characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum size for a valid chunk
            use_token_chunking: Use token-based chunking vs character-based
            model_name: Model name for tokenizer
            preserve_sentences: Try to preserve sentence boundaries
            preserve_paragraphs: Try to preserve paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_token_chunking = use_token_chunking
        self.preserve_sentences = preserve_sentences
        self.preserve_paragraphs = preserve_paragraphs
        
        # Initialize tokenizer if using token-based chunking
        if self.use_token_chunking:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except:
                # Fallback to cl100k_base encoding
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self,
                      text: str,
                      document_id: str,
                      metadata: Dict = None,
                      section_name: Optional[str] = None) -> List[Chunk]:
        """
        Chunk a document into overlapping segments
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            metadata: Document metadata to include in chunks
            section_name: Name of the section being chunked
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        if self.preserve_paragraphs:
            raw_chunks = self._chunk_by_paragraphs(text)
        elif self.preserve_sentences:
            raw_chunks = self._chunk_by_sentences(text)
        else:
            raw_chunks = self._chunk_by_tokens(text) if self.use_token_chunking else self._chunk_by_characters(text)
        
        # Create Chunk objects with metadata
        chunks = []
        total_chunks = len(raw_chunks)
        
        for i, (chunk_text, overlap_prev, overlap_next) in enumerate(raw_chunks):
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(document_id, i, chunk_text)
            
            # Count tokens/words
            token_count = len(self.tokenizer.encode(chunk_text)) if self.use_token_chunking else len(chunk_text)
            word_count = len(chunk_text.split())
            
            # Detect metadata flags
            has_financial = self._contains_financial_data(chunk_text)
            has_risk = self._contains_risk_mention(chunk_text)
            has_forward = self._contains_forward_looking(chunk_text)
            entities = self._extract_entities(chunk_text)
            
            # Create chunk
            chunk = Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                section=section_name,
                chunk_index=i,
                total_chunks=total_chunks,
                word_count=word_count,
                token_count=token_count,
                overlap_with_prev=overlap_prev,
                overlap_with_next=overlap_next,
                has_financial_data=has_financial,
                has_risk_mention=has_risk,
                has_forward_looking=has_forward,
                entities_mentioned=entities
            )
            
            # Add document metadata
            if metadata:
                chunk.ticker = metadata.get('ticker')
                chunk.cik = metadata.get('cik')
                chunk.fiscal_year = metadata.get('fiscal_year')
                chunk.filing_date = metadata.get('filing_date')
                chunk.form_type = metadata.get('form_type')
                chunk.accession_number = metadata.get('accession_number')
            
            # Link chunks
            if i > 0:
                chunk.prev_chunk_id = chunks[i-1].chunk_id
                chunks[i-1].next_chunk_id = chunk_id
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text before chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        # Preserve paragraph breaks
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()
    
    def _chunk_by_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by sentences with overlap"""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return [(text, 0, 0)]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = self._get_size(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk with overlap
                chunk_text = ' '.join(current_chunk)
                
                # Calculate overlap for next chunk
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    sent_size = self._get_size(sent)
                    if overlap_size + sent_size <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += sent_size
                    else:
                        break
                
                chunks.append((chunk_text, 0, len(' '.join(overlap_sentences))))
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append((' '.join(current_chunk), 0, 0))
        
        # Update overlap values
        for i in range(len(chunks) - 1):
            text, prev_overlap, _ = chunks[i]
            chunks[i] = (text, prev_overlap, chunks[i+1][1] if i+1 < len(chunks) else 0)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by paragraphs with overlap"""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [(text, 0, 0)]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = self._get_size(paragraph)
            
            # If single paragraph exceeds chunk size, split it
            if para_size > self.chunk_size:
                if current_chunk:
                    chunks.append(('\n\n'.join(current_chunk), 0, 0))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                para_chunks = self._chunk_by_sentences(paragraph)
                chunks.extend(para_chunks)
            elif current_size + para_size > self.chunk_size and current_chunk:
                # Create chunk
                chunks.append(('\n\n'.join(current_chunk), 0, 0))
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n\n'.join(current_chunk), 0, 0))
        
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by token count with overlap"""
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        chunks = []
        start = 0
        
        while start < total_tokens:
            # Calculate end position
            end = min(start + self.chunk_size, total_tokens)
            
            # Extract chunk tokens
            chunk_tokens = tokens[start:end]
            
            # Calculate overlap start for next chunk
            overlap_start = max(start, end - self.chunk_overlap)
            
            # Decode tokens to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate actual overlap sizes
            prev_overlap = 0 if start == 0 else self.chunk_overlap
            next_overlap = 0 if end >= total_tokens else min(self.chunk_overlap, end - overlap_start)
            
            chunks.append((chunk_text, prev_overlap, next_overlap))
            
            # Move to next chunk position
            start = overlap_start if overlap_start < end else end
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by character count with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary
            if self.preserve_sentences and end < text_length:
                # Look for sentence end near chunk boundary
                search_start = max(start, end - 100)
                search_text = text[search_start:end + 100]
                
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_text)]
                if sentence_ends:
                    # Find closest sentence end to target size
                    best_end = search_start + min(sentence_ends, key=lambda x: abs(x - (self.chunk_size - (search_start - start))))
                    end = min(best_end, text_length)
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Calculate overlap
            overlap_start = max(start, end - self.chunk_overlap)
            prev_overlap = 0 if start == 0 else min(self.chunk_overlap, start - (start - self.chunk_overlap))
            next_overlap = 0 if end >= text_length else min(self.chunk_overlap, end - overlap_start)
            
            chunks.append((chunk_text, prev_overlap, next_overlap))
            
            # Move to next chunk
            start = overlap_start if overlap_start < end else end
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        # In production, use more sophisticated NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Merge very short sentences
        merged = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) < 50:  # Merge if combined length < 50
                current = current + " " + sentence if current else sentence
            else:
                if current:
                    merged.append(current)
                current = sentence
        
        if current:
            merged.append(current)
        
        return merged
    
    def _get_size(self, text: str) -> int:
        """Get size of text based on chunking method"""
        if self.use_token_chunking:
            return len(self.tokenizer.encode(text))
        else:
            return len(text)
    
    def _generate_chunk_id(self, document_id: str, index: int, text: str) -> str:
        """Generate unique chunk ID"""
        # Create hash from document ID, index, and text preview
        content = f"{document_id}_{index}_{text[:100]}"
        hash_obj = hashlib.md5(content.encode())
        return f"chunk_{hash_obj.hexdigest()[:12]}"
    
    def _contains_financial_data(self, text: str) -> bool:
        """Check if text contains financial data"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.FINANCIAL_KEYWORDS)
    
    def _contains_risk_mention(self, text: str) -> bool:
        """Check if text contains risk mentions"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.RISK_KEYWORDS)
    
    def _contains_forward_looking(self, text: str) -> bool:
        """Check if text contains forward-looking statements"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.FORWARD_KEYWORDS)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        entities = []
        
        # Extract dollar amounts
        dollar_pattern = r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?'
        dollar_matches = re.findall(dollar_pattern, text, re.IGNORECASE)
        entities.extend(dollar_matches[:5])  # Limit to 5
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        percent_matches = re.findall(percent_pattern, text)
        entities.extend(percent_matches[:5])
        
        # Extract years
        year_pattern = r'\b20\d{2}\b'
        year_matches = re.findall(year_pattern, text)
        entities.extend(list(set(year_matches))[:3])
        
        return entities
    
    def create_sliding_window_chunks(self,
                                    text: str,
                                    window_size: int = 3,
                                    stride: int = 1) -> List[str]:
        """
        Create chunks using sliding window over sentences
        
        Args:
            text: Text to chunk
            window_size: Number of sentences per chunk
            stride: Number of sentences to slide
            
        Returns:
            List of chunk texts
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = ' '.join(sentences[i:i + window_size])
            chunks.append(chunk)
        
        # Add final chunk if there are remaining sentences
        if len(sentences) % stride != 0:
            chunk = ' '.join(sentences[-window_size:])
            if chunk not in chunks:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_for_qa(self,
                    sections: Dict[str, str],
                    document_id: str,
                    metadata: Dict = None) -> List[Chunk]:
        """
        Create chunks optimized for Q&A from document sections
        
        Args:
            sections: Dictionary of section_name -> content
            document_id: Document identifier
            metadata: Document metadata
            
        Returns:
            List of chunks optimized for Q&A
        """
        all_chunks = []
        
        for section_name, content in sections.items():
            if not content:
                continue
            
            # Use smaller chunks for Q&A
            original_size = self.chunk_size
            self.chunk_size = min(500, self.chunk_size)  # Smaller chunks for Q&A
            
            # Create chunks for this section
            section_chunks = self.chunk_document(
                content,
                document_id=f"{document_id}_{section_name}",
                metadata=metadata,
                section_name=section_name
            )
            
            all_chunks.extend(section_chunks)
            
            # Restore original size
            self.chunk_size = original_size
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> pd.DataFrame:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            DataFrame with chunk statistics
        """
        stats = []
        
        for chunk in chunks:
            stats.append({
                'chunk_id': chunk.chunk_id,
                'section': chunk.section,
                'word_count': chunk.word_count,
                'token_count': chunk.token_count,
                'has_financial': chunk.has_financial_data,
                'has_risk': chunk.has_risk_mention,
                'has_forward_looking': chunk.has_forward_looking,
                'num_entities': len(chunk.entities_mentioned) if chunk.entities_mentioned else 0,
                'overlap_prev': chunk.overlap_with_prev,
                'overlap_next': chunk.overlap_with_next
            })
        
        df = pd.DataFrame(stats)
        
        # Add summary statistics
        if not df.empty:
            logger.info(f"Chunk Statistics:")
            logger.info(f"  Total chunks: {len(df)}")
            logger.info(f"  Avg words per chunk: {df['word_count'].mean():.1f}")
            logger.info(f"  Avg tokens per chunk: {df['token_count'].mean():.1f}")
            logger.info(f"  Chunks with financial data: {df['has_financial'].sum()}")
            logger.info(f"  Chunks with risk mentions: {df['has_risk'].sum()}")
        
        return df