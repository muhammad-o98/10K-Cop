import os
import json
import tempfile
from pathlib import Path
import pytest

import pytest

# Optional deps used by retriever; skip tests if missing
try:
    import chromadb  # noqa: F401
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False

try:
    import sentence_transformers  # noqa: F401
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import duckdb  # noqa: F401
    _HAS_DUCKDB = True
except Exception:
    _HAS_DUCKDB = False


@pytest.fixture(scope="module")
def tmp_data_env():
    """
    Creates a temporary DATA_DIR and CHROMA_DIR with:
      - data/processed/TEST/2023.json containing sections ("7", "1A")
      - data/analytics/sec.duckdb with a minimal ratios table for ticker TEST
    Also sets environment variables so the RAG code uses this temp area.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        chroma_dir = data_dir / "chroma"
        processed_dir = data_dir / "processed" / "TEST"
        analytics_dir = data_dir / "analytics"
        processed_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        analytics_dir.mkdir(parents=True, exist_ok=True)

        # Minimal processed 10-K sections
        sample = {
            "7": "Management’s Discussion and Analysis. Liquidity improved due to higher operating cash flows.",
            "1A": "Risk Factors include supply chain disruptions and macroeconomic volatility.",
        }
        (processed_dir / "2023.json").write_text(json.dumps(sample), encoding="utf-8")

        # Minimal analytics DB
        if _HAS_DUCKDB:
            import duckdb
            con = duckdb.connect(str(analytics_dir / "sec.duckdb"))
            con.execute("CREATE TABLE IF NOT EXISTS ratios (ticker STRING, fy INTEGER, metric STRING, value DOUBLE)")
            con.execute(
                "INSERT INTO ratios VALUES ('TEST', 2023, 'current_ratio', 1.5), ('TEST', 2023, 'debt_to_equity', 0.8)"
            )
            con.close()

        # Point the code to our temp dirs and use a very small embedding model if available
        os.environ["DATA_DIR"] = str(data_dir)
        os.environ["CHROMA_DIR"] = str(chroma_dir)
        # Keep default embedding id if your project sets one; otherwise pick a tiny one to reduce download time
        if "EMBEDDING_MODEL" not in os.environ:
            os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-MiniLM-L3-v2"

        yield {"data_dir": data_dir, "chroma_dir": chroma_dir, "ticker": "TEST"}


@pytest.mark.skipif(not _HAS_CHROMA or not _HAS_ST, reason="RAG dependencies (Chroma/sentence-transformers) not installed")
def test_build_hybrid_retriever_returns_docs(tmp_data_env):
    # Import here so env vars are already set
    from src.rag.retriever import build_hybrid_retriever

    retriever = build_hybrid_retriever(ticker=tmp_data_env["ticker"], k=4)
    docs = retriever.get_relevant_documents("How did liquidity change?")
    assert isinstance(docs, list)
    assert len(docs) >= 1
    # Basic metadata/citation presence
    md = docs[0].metadata
    assert "fy" in md
    assert "section" in md
    assert "source" in md
    assert md.get("ticker", "") in ("TEST", "")  # some retrievers may omit ticker


class FakeLLM:
    """Simple LLM stub returning a deterministic JSON string."""
    def invoke(self, _prompt: str):
        return json.dumps({
            "answer": "Liquidity improved, supported by higher operating cash flows.",
            "citations": [
                {"fy": "2023", "section": "7", "source": "2023.json", "snippet": "Liquidity improved due to higher operating cash flows."}
            ],
            "derived_metrics": [
                {"name": "current_ratio", "value": 1.5, "formula": "current_assets/current_liabilities", "source": "ANALYTICS"}
            ],
            "viz_suggestions": [
                {"type": "line", "title": "Current ratio trend", "data_refs": ["current_ratio"]}
            ]
        })


@pytest.mark.skipif(not _HAS_CHROMA or not _HAS_ST, reason="RAG dependencies (Chroma/sentence-transformers) not installed")
def test_answer_with_citations_json_and_viz(tmp_data_env, monkeypatch):
    # Monkeypatch the LLM client to avoid calling Ollama or any external model
    import src.rag.llm_client as llm_client
    monkeypatch.setattr(llm_client, "get_llm", lambda: FakeLLM())

    from src.rag.citation_generator import answer_with_citations

    result = answer_with_citations(tmp_data_env["ticker"], "Summarize liquidity changes and support with citations.", k=4)

    # Structural checks
    assert isinstance(result, dict)
    for key in ("answer", "citations", "derived_metrics", "viz_suggestions"):
        assert key in result

    # Content sanity
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert isinstance(result["citations"], list) and len(result["citations"]) >= 1
    c0 = result["citations"][0]
    for k in ("fy", "section", "source", "snippet"):
        assert k in c0
    # Derived metrics present and numeric
    assert result["derived_metrics"][0]["name"] == "current_ratio"
    assert isinstance(result["derived_metrics"][0]["value"], (int, float))
    # Viz suggestion shape
    vz = result["viz_suggestions"][0]
    assert all(k in vz for k in ("type", "title", "data_refs"))