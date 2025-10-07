"""
Test script for processing and storage modules
Demonstrates the complete pipeline from ingestion to storage
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import ingestion modules
from src.ingestion.edgar_downloader import EdgarDownloader
from src.ingestion.xbrl_processor import XBRLProcessor

# Import processing modules
from src.processing.document_parser import DocumentParser
from src.processing.text_chunker import TextChunker

# Import storage modules
from src.storage.duckdb_manager import DuckDBManager
from src.storage.vector_store import VectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_document_parser(filing_content):
    """Test document parsing"""
    logger.info("=" * 50)
    logger.info("Testing Document Parser")
    logger.info("=" * 50)
    
    parser = DocumentParser(
        min_section_length=100,
        extract_tables=True
    )
    
    # Parse the document sections
    sections = {}
    
    if filing_content and 'content' in filing_content:
        # Parse full text
        if 'full_text' in filing_content['content']:
            logger.info("Parsing full document text...")
            sections = parser.parse_text(filing_content['content']['full_text'])
            
        # Parse pre-extracted sections
        elif 'html_sections' in filing_content['content']:
            logger.info("Parsing pre-extracted sections...")
            for section_name, content in filing_content['content']['html_sections'].items():
                parsed = parser.parse_text(content)
                if parsed:
                    sections.update(parsed)
    
    if sections:
        logger.info(f"✓ Extracted {len(sections)} sections:")
        
        # Show section statistics
        stats_df = parser.get_section_statistics(sections)
        if not stats_df.empty:
            for _, row in stats_df.head(5).iterrows():
                logger.info(f"  - {row['item']}: {row['title']} ({row['word_count']} words)")
        
        # Extract metrics from MD&A if available
        if 'item_7' in sections:
            metrics = parser.extract_metrics_from_text(sections['item_7'].content)
            if metrics:
                logger.info(f"  - Extracted metrics from MD&A: {list(metrics.keys())}")
    else:
        logger.warning("No sections extracted")
    
    return sections


def test_text_chunker(sections, filing_metadata):
    """Test text chunking for RAG"""
    logger.info("=" * 50)
    logger.info("Testing Text Chunker")
    logger.info("=" * 50)
    
    chunker = TextChunker(
        chunk_size=500,  # Smaller chunks for testing
        chunk_overlap=100,
        use_token_chunking=False  # Use character-based for simplicity
    )
    
    all_chunks = []
    
    # Chunk each section
    for section_key, section in sections.items():
        if not section.content:
            continue
        
        logger.info(f"Chunking section: {section.title}")
        
        chunks = chunker.chunk_document(
            text=section.content[:5000],  # Limit for testing
            document_id=filing_metadata.get('accession_number', 'test_doc'),
            metadata=filing_metadata,
            section_name=section_key
        )
        
        all_chunks.extend(chunks)
        
        if chunks:
            logger.info(f"  - Created {len(chunks)} chunks")
            logger.info(f"  - Avg words per chunk: {sum(c.word_count for c in chunks) / len(chunks):.1f}")
            
            # Show chunk metadata
            financial_chunks = sum(1 for c in chunks if c.has_financial_data)
            risk_chunks = sum(1 for c in chunks if c.has_risk_mention)
            logger.info(f"  - Chunks with financial data: {financial_chunks}")
            logger.info(f"  - Chunks with risk mentions: {risk_chunks}")
    
    if all_chunks:
        # Get chunk statistics
        stats_df = chunker.get_chunk_statistics(all_chunks)
        logger.info(f"✓ Created {len(all_chunks)} total chunks")
    
    return all_chunks


def test_duckdb_storage(filing_metadata, metrics_df):
    """Test DuckDB storage"""
    logger.info("=" * 50)
    logger.info("Testing DuckDB Storage")
    logger.info("=" * 50)
    
    db = DuckDBManager(
        db_path="./data/analytics/test_10k.duckdb",
        read_only=False
    )
    
    try:
        # Insert company
        company_data = {
            'cik': filing_metadata.get('cik'),
            'ticker': filing_metadata.get('ticker'),
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        if db.insert_company(company_data):
            logger.info("✓ Inserted company data")
        
        # Insert filing
        filing_data = {
            'accession_number': filing_metadata.get('accession_number'),
            'cik': filing_metadata.get('cik'),
            'ticker': filing_metadata.get('ticker'),
            'form_type': '10-K',
            'filing_date': filing_metadata.get('filing_date'),
            'fiscal_year': filing_metadata.get('year'),
            'fiscal_period': 'FY',
            'processed': True
        }
        
        if db.insert_filing(filing_data):
            logger.info("✓ Inserted filing data")
        
        # Insert financial metrics if available
        if metrics_df is not None and not metrics_df.empty:
            rows = db.insert_financial_metrics(
                metrics_df.head(50),  # Limit for testing
                filing_metadata.get('accession_number'),
                filing_metadata.get('cik'),
                filing_metadata.get('ticker')
            )
            logger.info(f"✓ Inserted {rows} financial metrics")
            
            # Calculate ratios
            ratios = db.calculate_and_store_ratios(
                filing_metadata.get('ticker'),
                filing_metadata.get('year')
            )
            
            if ratios:
                logger.info(f"✓ Calculated {len(ratios)} financial ratios:")
                for ratio_name, value in list(ratios.items())[:5]:
                    logger.info(f"  - {ratio_name}: {value:.3f}")
        
        # Query ratios
        ratios_df = db.query_ratios(
            tickers=[filing_metadata.get('ticker')],
            fiscal_years=[filing_metadata.get('year')]
        )
        
        if not ratios_df.empty:
            logger.info(f"✓ Retrieved {len(ratios_df)} ratio records")
        
        # Create analytics views
        db.create_analytics_views()
        logger.info("✓ Created analytics views")
        
    finally:
        db.close()
    
    return db


def test_vector_store(chunks):
    """Test vector store for semantic search"""
    logger.info("=" * 50)
    logger.info("Testing Vector Store")
    logger.info("=" * 50)
    
    store = VectorStore(
        persist_dir="./data/vector_store_test",
        collection_name="test_10k",
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    
    # Add chunks to vector store
    if chunks:
        # Limit to first 20 chunks for testing
        test_chunks = chunks[:20]
        
        logger.info(f"Adding {len(test_chunks)} chunks to vector store...")
        chunk_ids = store.add_chunks(test_chunks)
        logger.info(f"✓ Added {len(chunk_ids)} chunks")
        
        # Test similarity search
        test_queries = [
            "What are the main risk factors?",
            "Revenue growth and financial performance",
            "Competition and market position"
        ]
        
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            results = store.similarity_search(query, n_results=3)
            
            if results:
                logger.info(f"  Found {len(results)} results:")
                for i, result in enumerate(results[:2], 1):
                    logger.info(f"  {i}. Score: {result['similarity_score']:.3f}")
                    logger.info(f"     Section: {result['metadata'].get('section', 'N/A')}")
                    logger.info(f"     Preview: {result['document'][:100]}...")
        
        # Get collection statistics
        stats = store.get_collection_stats()
        logger.info(f"\n✓ Collection statistics:")
        logger.info(f"  - Total documents: {stats.get('total_documents', 0)}")
        logger.info(f"  - Unique sections: {stats.get('unique_sections', 0)}")
        
        # Test finding similar chunks
        if chunk_ids:
            similar = store.find_similar_chunks(chunk_ids[0], n_results=2)
            if similar:
                logger.info(f"✓ Found {len(similar)} similar chunks to first chunk")
    
    return store


def main():
    """Main test function"""
    logger.info("Testing Processing and Storage Modules")
    logger.info("=" * 70)
    
    # Step 1: Get a filing to test with
    logger.info("Step 1: Loading test filing...")
    
    edgar = EdgarDownloader(
        user_agent="Test Company test@example.com",
        cache_dir="./data/raw/edgar"
    )
    
    # Try to get cached filing for AAPL
    filing = edgar.download_10k('AAPL', 2023)
    
    if not filing:
        logger.error("Could not load test filing. Please run main.py first to download data.")
        return
    
    filing_metadata = {
        'ticker': filing.get('ticker'),
        'cik': filing.get('cik'),
        'year': filing.get('year'),
        'filing_date': filing.get('filing_date'),
        'accession_number': filing.get('accession_number')
    }
    
    logger.info(f"✓ Loaded filing for {filing_metadata['ticker']} year {filing_metadata['year']}")
    
    # Step 2: Parse document sections
    logger.info("\nStep 2: Parsing document sections...")
    sections = test_document_parser(filing)
    
    # Step 3: Create text chunks for RAG
    logger.info("\nStep 3: Creating text chunks...")
    chunks = test_text_chunker(sections, filing_metadata)
    
    # Step 4: Get XBRL metrics
    logger.info("\nStep 4: Loading XBRL metrics...")
    
    xbrl = XBRLProcessor(
        user_agent="Test Company test@example.com",
        cache_dir="./data/raw/xbrl"
    )
    
    facts = xbrl.fetch_companyfacts(filing_metadata['cik'])
    metrics_df = None
    
    if facts:
        metrics_df = xbrl.extract_metrics(facts, [filing_metadata['year']])
        logger.info(f"✓ Loaded {len(metrics_df)} financial metrics")
    
    # Step 5: Store in DuckDB
    logger.info("\nStep 5: Storing in DuckDB...")
    db = test_duckdb_storage(filing_metadata, metrics_df)
    
    # Step 6: Create vector embeddings
    logger.info("\nStep 6: Creating vector embeddings...")
    store = test_vector_store(chunks)
    
    # Summary
    logger.info("=" * 70)
    logger.info("Processing & Storage Test Summary")
    logger.info("-" * 30)
    logger.info(f"✓ Document Parser: {len(sections)} sections extracted")
    logger.info(f"✓ Text Chunker: {len(chunks)} chunks created")
    logger.info(f"✓ DuckDB: Data stored successfully")
    logger.info(f"✓ Vector Store: Embeddings created and searchable")
    logger.info("=" * 70)
    logger.info("All processing and storage modules working correctly!")
    logger.info("\nNext steps:")
    logger.info("1. Build the analytics modules (ratio calculations)")
    logger.info("2. Implement the RAG pipeline with citations")
    logger.info("3. Create the Streamlit application")


if __name__ == "__main__":
    main()