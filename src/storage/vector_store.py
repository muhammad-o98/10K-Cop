"""
Vector Store Module
Manages embeddings and similarity search using ChromaDB
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dataclasses import asdict

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings for semantic search using ChromaDB
    """
    
    def __init__(self,
                 persist_dir: str = "./data/vector_store",
                 collection_name: str = "10k_documents",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 use_gpu: bool = False,
                 batch_size: int = 32):
        """
        Initialize vector store
        
        Args:
            persist_dir: Directory to persist vector database
            collection_name: Name of the collection
            embedding_model: Model name for embeddings
            use_gpu: Whether to use GPU for embeddings
            batch_size: Batch size for embedding generation
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        device = 'cuda' if use_gpu else 'cpu'
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Get or create collection
        self._init_collection()
        
        logger.info(f"VectorStore initialized with collection '{collection_name}' at {persist_dir}")
    
    def _init_collection(self):
        """Initialize or get collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
    
    def _get_embedding_function(self):
        """Get embedding function for ChromaDB"""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
    
    def add_documents(self,
                     documents: List[str],
                     metadatas: List[Dict],
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(doc) for doc in documents]
        
        # Clean metadata (ensure JSON serializable)
        cleaned_metadatas = []
        for metadata in metadatas:
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    cleaned_metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists to JSON strings
                    cleaned_metadata[key] = json.dumps(value)
                elif isinstance(value, datetime):
                    cleaned_metadata[key] = value.isoformat()
                else:
                    cleaned_metadata[key] = str(value)
            cleaned_metadatas.append(cleaned_metadata)
        
        # Add in batches for better performance
        total_added = 0
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_metas = cleaned_metadatas[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                total_added += len(batch_docs)
                logger.debug(f"Added batch {i//self.batch_size + 1}, total: {total_added}")
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
        
        logger.info(f"Added {total_added} documents to vector store")
        return ids[:total_added]
    
    def add_chunks(self, chunks: List) -> List[str]:
        """
        Add text chunks to vector store
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk.content)
            
            # Convert chunk to metadata dict
            metadata = {
                'document_id': chunk.document_id,
                'section': chunk.section or '',
                'chunk_index': chunk.chunk_index,
                'ticker': chunk.ticker or '',
                'cik': chunk.cik or '',
                'fiscal_year': chunk.fiscal_year or 0,
                'filing_date': chunk.filing_date or '',
                'form_type': chunk.form_type or '',
                'accession_number': chunk.accession_number or '',
                'has_financial_data': chunk.has_financial_data,
                'has_risk_mention': chunk.has_risk_mention,
                'has_forward_looking': chunk.has_forward_looking,
                'word_count': chunk.word_count,
                'token_count': chunk.token_count
            }
            
            metadatas.append(metadata)
            ids.append(chunk.chunk_id)
        
        return self.add_documents(documents, metadatas, ids)
    
    def similarity_search(self,
                         query: str,
                         n_results: int = 10,
                         filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Perform similarity search
        
        Args:
            query: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Build where clause for filtering
            where = None
            if filter_metadata:
                where = {}
                for key, value in filter_metadata.items():
                    if value is not None:
                        where[key] = value
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def hybrid_search(self,
                     query: str,
                     documents: List[str],
                     n_results: int = 10,
                     alpha: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and BM25
        
        Args:
            query: Query text
            documents: List of documents to search
            n_results: Number of results
            alpha: Weight for vector search (1-alpha for BM25)
            
        Returns:
            List of search results
        """
        # Get vector search results
        vector_results = self.similarity_search(query, n_results * 2)
        
        # BM25 search (simplified implementation)
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Normalize scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = [s / max_bm25 for s in bm25_scores]
        
        # Combine scores
        combined_scores = {}
        
        # Add vector search scores
        for result in vector_results:
            doc_id = result['metadata'].get('document_id', '')
            combined_scores[doc_id] = alpha * result['similarity_score']
        
        # Add BM25 scores
        for i, score in enumerate(bm25_scores_norm):
            doc_id = f"doc_{i}"  # Simplified ID
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * score
            else:
                combined_scores[doc_id] = (1 - alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc_id, score in sorted_results[:n_results]:
            # Find corresponding document
            for result in vector_results:
                if result['metadata'].get('document_id', '') == doc_id:
                    result['hybrid_score'] = score
                    results.append(result)
                    break
        
        return results
    
    def search_by_section(self,
                         query: str,
                         section_type: str,
                         ticker: Optional[str] = None,
                         fiscal_year: Optional[int] = None,
                         n_results: int = 10) -> List[Dict]:
        """
        Search within specific document sections
        
        Args:
            query: Query text
            section_type: Type of section (e.g., 'item_1a', 'item_7')
            ticker: Optional ticker filter
            fiscal_year: Optional year filter
            n_results: Number of results
            
        Returns:
            List of search results
        """
        filter_metadata = {'section': section_type}
        
        if ticker:
            filter_metadata['ticker'] = ticker
        if fiscal_year:
            filter_metadata['fiscal_year'] = fiscal_year
        
        return self.similarity_search(query, n_results, filter_metadata)
    
    def find_similar_chunks(self,
                           chunk_id: str,
                           n_results: int = 5) -> List[Dict]:
        """
        Find chunks similar to a given chunk
        
        Args:
            chunk_id: ID of the reference chunk
            n_results: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        try:
            # Get the chunk
            result = self.collection.get(ids=[chunk_id], include=['documents'])
            
            if not result or not result['documents']:
                logger.warning(f"Chunk {chunk_id} not found")
                return []
            
            # Search for similar chunks
            document = result['documents'][0]
            similar = self.similarity_search(document, n_results + 1)  # +1 to exclude self
            
            # Filter out the original chunk
            similar = [s for s in similar if s['metadata'].get('chunk_id') != chunk_id]
            
            return similar[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    def update_document(self,
                       document_id: str,
                       document: Optional[str] = None,
                       metadata: Optional[Dict] = None):
        """
        Update a document in the vector store
        
        Args:
            document_id: ID of the document to update
            document: New document text
            metadata: New metadata
        """
        try:
            if document:
                self.collection.update(
                    ids=[document_id],
                    documents=[document],
                    metadatas=[metadata] if metadata else None
                )
                logger.info(f"Updated document {document_id}")
        except Exception as e:
            logger.error(f"Error updating document: {e}")
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents from vector store"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand the data
            sample = self.collection.get(limit=100, include=['metadatas'])
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name
            }
            
            if sample and sample['metadatas']:
                # Analyze metadata
                df = pd.DataFrame(sample['metadatas'])
                
                if 'ticker' in df.columns:
                    stats['unique_tickers'] = df['ticker'].nunique()
                    stats['tickers'] = df['ticker'].unique().tolist()[:10]
                
                if 'fiscal_year' in df.columns:
                    stats['year_range'] = f"{df['fiscal_year'].min()}-{df['fiscal_year'].max()}"
                
                if 'section' in df.columns:
                    stats['unique_sections'] = df['section'].nunique()
                    stats['sections'] = df['section'].value_counts().head(5).to_dict()
                
                if 'has_financial_data' in df.columns:
                    stats['pct_with_financial'] = df['has_financial_data'].mean() * 100
                
                if 'has_risk_mention' in df.columns:
                    stats['pct_with_risk'] = df['has_risk_mention'].mean() * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def create_index(self):
        """Create or update the HNSW index for faster search"""
        try:
            # ChromaDB automatically maintains the index
            # This method is here for compatibility
            logger.info("ChromaDB automatically maintains the HNSW index")
        except Exception as e:
            logger.error(f"Error with index: {e}")
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for a document"""
        hash_obj = hashlib.md5(text.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def export_embeddings(self, output_path: str):
        """Export embeddings to file for analysis"""
        try:
            # Get all documents with embeddings
            results = self.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            # Create DataFrame
            data = []
            for i in range(len(results['documents'])):
                data.append({
                    'document': results['documents'][i],
                    'metadata': json.dumps(results['metadatas'][i]),
                    'embedding': results['embeddings'][i] if results.get('embeddings') else None
                })
            
            df = pd.DataFrame(data)
            df.to_parquet(output_path, index=False)
            logger.info(f"Exported {len(df)} embeddings to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
    
    def batch_embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def close(self):
        """Close vector store connection"""
        # ChromaDB handles cleanup automatically
        logger.info("Vector store closed")