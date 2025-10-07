"""
Main script to demonstrate the 10K Cop ingestion modules
"""

import os
import sys
import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.edgar_downloader import EdgarDownloader
from src.ingestion.xbrl_processor import XBRLProcessor
from src.ingestion.price_fetcher import PriceFetcher
from src.ingestion.macro_fetcher import MacroFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file or create default"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'edgar': {
                'user_agent': 'Test Company test@example.com',
                'rate_limit_delay': 0.1,
                'max_retries': 3
            },
            'fred': {
                'api_key': os.environ.get('FRED_API_KEY', 'YOUR_FRED_API_KEY_HERE')
            },
            'cache': {
                'edgar_dir': './data/raw/edgar',
                'xbrl_dir': './data/raw/xbrl',
                'prices_dir': './data/raw/prices',
                'macro_dir': './data/raw/macro',
                'expiry_hours': 24
            },
            'tickers': ['AAPL', 'MSFT', 'JNJ'],
            'years': [2021, 2022, 2023]
        }


def test_edgar_downloader(config, ticker='AAPL', year=2023):
    """Test EDGAR downloader functionality"""
    logger.info("=" * 50)
    logger.info("Testing EDGAR Downloader")
    logger.info("=" * 50)
    
    edgar = EdgarDownloader(
        user_agent=config['edgar']['user_agent'],
        cache_dir=config['cache']['edgar_dir'],
        rate_limit_delay=config['edgar']['rate_limit_delay']
    )
    
    # Test ticker to CIK conversion
    cik = edgar.ticker_to_cik(ticker)
    logger.info(f"Ticker {ticker} -> CIK: {cik}")
    
    if not cik:
        logger.error(f"Could not find CIK for ticker {ticker}")
        return None
    
    # Try to download 10-K
    logger.info(f"Downloading 10-K for {ticker} fiscal year {year}...")
    filing = edgar.download_10k(ticker, year)
    
    if filing:
        logger.info(f"✓ Successfully downloaded 10-K")
        logger.info(f"  - Filing date: {filing['filing_date']}")
        logger.info(f"  - Accession number: {filing['accession_number']}")
        logger.info(f"  - Sections found: {list(filing['content']['html_sections'].keys())}")
        
        # Show snippet of MD&A section if available
        if 'mda' in filing['content']['html_sections']:
            mda_text = filing['content']['html_sections']['mda'][:500]
            logger.info(f"  - MD&A snippet: {mda_text[:200]}...")
    else:
        logger.warning(f"✗ Could not download 10-K for {ticker} year {year}")
        
        # Try previous year
        logger.info(f"Trying year {year-1}...")
        filing = edgar.download_10k(ticker, year-1)
        if filing:
            logger.info(f"✓ Found filing for year {year-1}")
    
    return filing


def test_xbrl_processor(config, ticker='AAPL', years=[2022, 2023]):
    """Test XBRL processor functionality"""
    logger.info("=" * 50)
    logger.info("Testing XBRL Processor")
    logger.info("=" * 50)
    
    edgar = EdgarDownloader(
        user_agent=config['edgar']['user_agent'],
        cache_dir=config['cache']['edgar_dir']
    )
    
    xbrl = XBRLProcessor(
        user_agent=config['edgar']['user_agent'],
        cache_dir=config['cache']['xbrl_dir']
    )
    
    # Get CIK
    cik = edgar.ticker_to_cik(ticker)
    if not cik:
        logger.error(f"Could not find CIK for ticker {ticker}")
        return None
    
    logger.info(f"Fetching XBRL data for {ticker} (CIK: {cik})...")
    
    # Fetch company facts
    facts = xbrl.fetch_companyfacts(cik)
    
    if facts:
        logger.info(f"✓ Successfully fetched company facts")
        
        # Extract metrics
        metrics_df = xbrl.extract_metrics(facts, years)
        
        if not metrics_df.empty:
            logger.info(f"✓ Extracted {len(metrics_df)} metric records")
            
            # Show available metrics
            unique_metrics = metrics_df['metric'].unique()
            logger.info(f"  - Available metrics: {list(unique_metrics[:10])}")
            
            # Get financial statements
            statements = xbrl.get_financial_statements(cik, years)
            
            for stmt_type, stmt_df in statements.items():
                if not stmt_df.empty:
                    logger.info(f"  - {stmt_type} statement: {stmt_df.shape[0]} items × {stmt_df.shape[1]} years")
                    
                    # Show key metrics
                    if stmt_type == 'IS' and 'Revenue' in stmt_df.index:
                        revenue_row = stmt_df.loc['Revenue']
                        logger.info(f"    Revenue: {revenue_row.to_dict()}")
                    elif stmt_type == 'BS' and 'Assets' in stmt_df.index:
                        assets_row = stmt_df.loc['Assets']
                        logger.info(f"    Assets: {assets_row.to_dict()}")
            
            # Validate data quality
            validation = xbrl.validate_data_quality(metrics_df)
            logger.info(f"  - Data quality: {validation['total_records']} records")
            if validation['consistency_checks']:
                logger.warning(f"    Found {len(validation['consistency_checks'])} consistency issues")
        else:
            logger.warning("✗ No metrics extracted")
    else:
        logger.warning(f"✗ Could not fetch company facts for {ticker}")
    
    return facts


def test_price_fetcher(config, ticker='AAPL'):
    """Test price fetcher functionality"""
    logger.info("=" * 50)
    logger.info("Testing Price Fetcher")
    logger.info("=" * 50)
    
    prices = PriceFetcher(cache_dir=config['cache']['prices_dir'])
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Fetching prices for {ticker}...")
    
    # Fetch prices
    df = prices.fetch_prices(
        ticker=ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if not df.empty:
        logger.info(f"✓ Successfully fetched {len(df)} days of price data")
        logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  - Latest close: ${df.iloc[-1]['close']:.2f}")
        logger.info(f"  - Average volume: {df['volume'].mean():.0f}")
        
        # Calculate indicators
        logger.info("Calculating technical indicators...")
        df = prices.calculate_indicators(df, indicators=['sma', 'rsi', 'macd'])
        
        # Show latest indicators
        latest = df.iloc[-1]
        if 'sma_20' in df.columns and not pd.isna(latest['sma_20']):
            logger.info(f"  - SMA(20): ${latest['sma_20']:.2f}")
        if 'rsi_14' in df.columns and not pd.isna(latest['rsi_14']):
            logger.info(f"  - RSI(14): {latest['rsi_14']:.2f}")
        if 'macd' in df.columns and not pd.isna(latest['macd']):
            logger.info(f"  - MACD: {latest['macd']:.4f}")
        
        # Get company info
        logger.info("Fetching company info...")
        info = prices.get_company_info(ticker)
        if info and 'error' not in info:
            logger.info(f"  - Company: {info.get('longName', ticker)}")
            logger.info(f"  - Sector: {info.get('sector', 'N/A')}")
            logger.info(f"  - Market Cap: ${info.get('marketCap', 0):,.0f}")
    else:
        logger.warning(f"✗ Could not fetch prices for {ticker}")
    
    return df


def test_macro_fetcher(config):
    """Test macro fetcher functionality"""
    logger.info("=" * 50)
    logger.info("Testing Macro Fetcher")
    logger.info("=" * 50)
    
    # Check for FRED API key
    if config['fred']['api_key'] == 'YOUR_FRED_API_KEY_HERE':
        logger.warning("✗ FRED API key not configured. Skipping macro tests.")
        logger.info("  Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    macro = MacroFetcher(
        api_key=config['fred']['api_key'],
        cache_dir=config['cache']['macro_dir']
    )
    
    # Test fetching a single series
    logger.info("Fetching Federal Funds Rate...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    fed_funds = macro.fetch_series(
        'FEDFUNDS',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if not fed_funds.empty:
        logger.info(f"✓ Successfully fetched {len(fed_funds)} data points")
        latest_value = fed_funds.iloc[-1]['value']
        latest_date = fed_funds.iloc[-1]['date']
        logger.info(f"  - Latest Fed Funds Rate: {latest_value:.2f}% (as of {latest_date})")
    
    # Test fetching multiple categories
    logger.info("Fetching macro indicators for multiple categories...")
    
    macro_df = macro.get_macro_features(
        categories=['rates', 'inflation'],
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if not macro_df.empty:
        logger.info(f"✓ Successfully fetched {len(macro_df.columns)-1} indicators")
        logger.info(f"  - Indicators: {list(macro_df.columns[:5])}")
        
        # Show latest values for key indicators
        latest = macro_df.iloc[-1]
        if 'fed_funds_rate' in macro_df.columns:
            logger.info(f"  - Fed Funds Rate: {latest['fed_funds_rate']:.2f}%")
        if 'treasury_10y' in macro_df.columns:
            logger.info(f"  - 10Y Treasury: {latest['treasury_10y']:.2f}%")
        if 'cpi_all' in macro_df.columns:
            logger.info(f"  - CPI: {latest['cpi_all']:.2f}")
    
    # Test recession indicators
    logger.info("Fetching recession indicators...")
    recession_df = macro.get_recession_indicators(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if not recession_df.empty:
        logger.info(f"✓ Got recession indicators")
        latest_recession = recession_df.iloc[-1]
        if 'recession_score' in recession_df.columns:
            logger.info(f"  - Recession Score: {latest_recession['recession_score']:.3f}")
    
    return macro_df


def main():
    """Main execution function"""
    logger.info("Starting 10K Cop Ingestion Module Tests")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Create data directories
    for key, value in config['cache'].items():
        if key != 'expiry_hours' and isinstance(value, str):
            Path(value).mkdir(parents=True, exist_ok=True)
    
    # Test each module
    results = {}
    
    # Test EDGAR Downloader
    try:
        filing = test_edgar_downloader(config, ticker='AAPL', year=2023)
        results['edgar'] = '✓' if filing else '✗'
    except Exception as e:
        logger.error(f"EDGAR test failed: {e}")
        results['edgar'] = '✗'
    
    # Test XBRL Processor
    try:
        facts = test_xbrl_processor(config, ticker='AAPL', years=[2022, 2023])
        results['xbrl'] = '✓' if facts else '✗'
    except Exception as e:
        logger.error(f"XBRL test failed: {e}")
        results['xbrl'] = '✗'
    
    # Test Price Fetcher
    try:
        prices = test_price_fetcher(config, ticker='AAPL')
        results['prices'] = '✓' if prices is not None and not prices.empty else '✗'
    except Exception as e:
        logger.error(f"Price fetcher test failed: {e}")
        results['prices'] = '✗'
    
    # Test Macro Fetcher
    try:
        macro = test_macro_fetcher(config)
        results['macro'] = '✓' if macro is not None else '△'  # Triangle for skipped
    except Exception as e:
        logger.error(f"Macro fetcher test failed: {e}")
        results['macro'] = '✗'
    
    # Summary
    logger.info("=" * 70)
    logger.info("Test Results Summary:")
    logger.info("-" * 30)
    for module, status in results.items():
        status_text = {
            '✓': 'SUCCESS',
            '✗': 'FAILED',
            '△': 'SKIPPED'
        }.get(status, status)
        logger.info(f"  {module.upper():15} {status} {status_text}")
    
    # Next steps
    logger.info("=" * 70)
    logger.info("Next Steps:")
    logger.info("1. Set up FRED API key for macro data")
    logger.info("2. Configure your company info in config.yaml")
    logger.info("3. Run full data pipeline for your target companies")
    logger.info("4. Proceed to build processing and storage modules")

if __name__ == "__main__":
    main()