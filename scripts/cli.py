#!/usr/bin/env python3
import os
import click
from typing import List
from src.utils.helpers import load_yaml
from src.ingestion.edgar_downloader import EdgarDownloader, get_cik_for_ticker
from src.processing.document_parser import DocumentParser
from src.processing.text_chunker import TextChunker
from src.storage.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger("cli")

def _cfg():
    return load_yaml("config/config.yaml")

def _ensure_dirs():
    for d in ["data/raw/edgar", "data/processed", "data/analytics"]:
        os.makedirs(d, exist_ok=True)

def _index_files_for_ticker_years(ticker: str, years: List[int], cfg):
    ed = EdgarDownloader(user_agent=cfg["ingestion"]["user_agent"] or os.getenv("USER_AGENT"))
    dp = DocumentParser()
    tk = TextChunker()
    vs = VectorStore(persist_dir=cfg["storage"]["chroma_dir"], embedding_model=cfg["rag"]["embedding_model"])

    cik = get_cik_for_ticker(ticker) or ticker
    for y in years:
        path = os.path.join(ed.cache_dir, f"{cik}_{y}.html")
        if not os.path.exists(path):
            logger.info(f"10-K for {ticker} {y} not found locally. Downloading...")
            downloaded = ed.download_10k(ticker, y)
            if not downloaded:
                logger.warning(f"Skipping year {y}, no file.")
                continue
            path = downloaded
        logger.info(f"Parsing {path}")
        html = open(path, "r", encoding="utf-8").read()
        text = dp.clean_text(dp.parse_html(html))
        sections = dp.extract_sections(text)
        docs = []
        for section, sect_text in sections.items():
            meta = {"ticker": ticker, "cik": cik, "fy": y, "section": section}
            docs.extend(tk.chunk_document(sect_text, chunk_size=cfg["features"]["chunk_size"], overlap=cfg["features"]["overlap"], metadata=meta))
        if docs:
            vs.add_documents(docs)
            logger.info(f"Indexed {len(docs)} chunks for {ticker} {y}")

@click.group()
def cli():
    _ensure_dirs()

@cli.command("ingest-10k")
@click.option("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
@click.option("--years", required=True, multiple=True, type=int, help="Years to ingest (e.g., 2022 2023)")
def ingest_10k_cmd(ticker, years):
    cfg = _cfg()
    ed = EdgarDownloader(user_agent=cfg["ingestion"]["user_agent"] or os.getenv("USER_AGENT"))
    for y in years:
        ed.download_10k(ticker, y)
    click.echo("Ingestion complete.")

@cli.command("build-index")
@click.option("--ticker", required=True)
@click.option("--years", required=True, multiple=True, type=int)
def build_index_cmd(ticker, years):
    cfg = _cfg()
    _index_files_for_ticker_years(ticker, list(years), cfg)
    click.echo("Index build complete.")

@cli.command("seed-demo")
@click.option("--ticker", required=True)
@click.option("--years", required=True, multiple=True, type=int)
def seed_demo_cmd(ticker, years):
    cfg = _cfg()
    ed = EdgarDownloader(user_agent=cfg["ingestion"]["user_agent"] or os.getenv("USER_AGENT"))
    for y in years:
        ed.download_10k(ticker, y)
    _index_files_for_ticker_years(ticker, list(years), cfg)
    click.echo("Demo seed complete.")

if __name__ == "__main__":
    cli()