processing_init = """
from .document_parser import DocumentParser, Section
from .text_chunker import TextChunker, Chunk

__all__ = [
    'DocumentParser',
    'Section',
    'TextChunker', 
    'Chunk'
]
"""