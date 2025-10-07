storage_init = """
from .duckdb_manager import DuckDBManager
from .vector_store import VectorStore

__all__ = [
    'DuckDBManager',
    'VectorStore'
]
"""