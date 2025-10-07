"""
Hybrid retriever for 10-K sections + analytics context.

- Loads processed 10-K section JSONs from data/processed/{ticker}/*.json
  Each JSON should be a dict of {section_key: text}.
- Optionally augments corpus with analytics context (e.g., ratios) from DuckDB
  at data/analytics/sec.duckdb.
- Builds a hybrid retriever: dense (Chroma + BGE embeddings) + BM25 lexical.
- Returns a LangChain EnsembleRetriever ready for use by a QA chain.

Environment variables (all optional):
  DATA_DIR: base data directory (default: ./data)
  CHROMA_DIR: Chroma persistence directory (default: ./data/chroma)
  EMBEDDING_MODEL: HF model id (default: BAAI/bge-small-en-v1.5)
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Tuple

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Optional DuckDB analytics context
try:
    import duckdb  # noqa
    _HAS_DUCKDB = True
except Exception:
    _HAS_DUCKDB = False

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def _chunk(text: str, max_len: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks, i = [], 0
    n = max_len
    step = max(1, n - overlap)
    while i < len(text):
        chunks.append(text[i : i + n])
        i += step
    return chunks


def _load_10k_documents(ticker: str) -> List[Document]:
    processed_dir = DATA_DIR / "processed" / ticker
    docs: List[Document] = []
    if not processed_dir.exists():
        return docs

    for f in sorted(processed_dir.glob("*.json")):
        fy = f.stem
        try:
            sections: Dict[str, str] = json.loads(f.read_text())
        except Exception:
            continue
        for section_key, section_text in sections.items():
            for j, chunk in enumerate(_chunk(section_text)):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "ticker": ticker,
                            "fy": fy,
                            "section": section_key,
                            "source": f.name,  # store filename; upstream can map to accession if available
                            "kind": "10K",
                            "chunk": j,
                        },
                    )
                )
    return docs


def _load_analytics_documents(ticker: str) -> List[Document]:
    """
    Convert analytics (e.g., ratios) into small textual facts to help the LLM
    interpret numbers and cite them. Safe no-op if DuckDB not present.
    """
    if not _HAS_DUCKDB:
        return []
    db = DATA_DIR / "analytics" / "sec.duckdb"
    if not db.exists():
        return []
    try:
        con = duckdb.connect(str(db), read_only=True)  # type: ignore
        # Ratios table schema assumed: (ticker, fy, metric, value)
        ratios = con.execute(
            "SELECT fy, metric, value FROM ratios WHERE ticker = ? ORDER BY fy, metric",
            [ticker],
        ).df()
        con.close()
    except Exception:
        return []

    docs: List[Document] = []
    # Group by year; create compact bullet-like summaries so BM25 has hooks.
    if not ratios.empty:
        for fy, grp in ratios.groupby("fy"):
            lines = [f"{m}: {v}" for m, v in zip(grp["metric"], grp["value"])]
            text = "Financial ratios for FY {fy}:\n".format(fy=fy) + "\n".join(lines)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "ticker": ticker,
                        "fy": str(int(fy)),
                        "section": "ANALYTICS",
                        "source": "ratios",
                        "kind": "ANALYTICS",
                    },
                )
            )
    return docs


def build_hybrid_retriever(
    ticker: str,
    k: int = 6,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
):
    """
    Returns a LangChain EnsembleRetriever combining:
      - Dense retriever over Chroma
      - BM25 lexical retriever over the same documents

    Persisted vector store lives in CHROMA_DIR under collection name {ticker}_10k.
    """
    base_docs = _load_10k_documents(ticker)
    analytics_docs = _load_analytics_documents(ticker)
    all_docs = base_docs + analytics_docs
    if not all_docs:
        # Return an empty BM25 retriever to fail gracefully
        bm25 = BM25Retriever.from_texts([""], metadatas=[{}])
        bm25.k = 0
        return bm25

    # Dense retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=f"{ticker}_10k",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    # Idempotent-ish: Chroma has no native unique constraint via LangChain wrapper.
    # Add in small batches; duplicates won’t break retrieval but can be cleaned later.
    vectordb.add_documents(all_docs)

    dense = vectordb.as_retriever(search_kwargs={"k": k})

    # BM25
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k

    # Hybrid
    hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[dense_weight, bm25_weight])
    return hybrid