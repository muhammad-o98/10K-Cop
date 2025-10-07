from typing import List
from src.utils.helpers import load_yaml
from src.ingestion.edgar_downloader import EdgarDownloader, get_cik_for_ticker
from src.processing.document_parser import DocumentParser
from src.processing.text_chunker import TextChunker
from src.storage.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger("orchestrator")

def build_index_for_ticker_years(ticker: str, years: List[int]):
    cfg = load_yaml("config/config.yaml")
    ed = EdgarDownloader(user_agent=cfg["ingestion"]["user_agent"])
    dp = DocumentParser()
    tk = TextChunker()
    vs = VectorStore(persist_dir=cfg["storage"]["chroma_dir"], embedding_model=cfg["rag"]["embedding_model"])

    cik = get_cik_for_ticker(ticker) or ticker
    for y in years:
        path = ed.download_10k(ticker, y)
        if not path:
            logger.warning(f"Skipping {ticker} {y}")
            continue
        html = open(path, "r", encoding="utf-8").read()
        text = dp.clean_text(dp.parse_html(html))
        sections = dp.extract_sections(text)
        docs = []
        for section, sect_text in sections.items():
            meta = {"ticker": ticker, "cik": cik, "fy": y, "section": section}
            docs.extend(tk.chunk_document(sect_text, chunk_size=cfg["features"]["chunk_size"], overlap=cfg["features"]["overlap"], metadata=meta))
        if docs:
            vs.add_documents(docs)
            logger.info(f"Added {len(docs)} docs for {ticker} {y}")