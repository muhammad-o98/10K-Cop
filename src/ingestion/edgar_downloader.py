"""
SEC EDGAR 10-K Filing Downloader Module
Handles downloading and caching of 10-K filings from SEC EDGAR database
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EdgarDownloader:
    """
    Downloads 10-K filings from SEC EDGAR with rate limiting and caching
    """
    
    BASE_URL = "https://www.sec.gov"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
    
    def __init__(self, 
                 user_agent: str,
                 cache_dir: str = "./data/raw/edgar",
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3):
        """
        Initialize EDGAR downloader with configuration
        
        Args:
            user_agent: Required user agent string for SEC compliance
            cache_dir: Directory to cache downloaded filings
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        
        # Load company tickers mapping
        self.tickers_to_cik = self._load_ticker_mapping()
        
        logger.info(f"EdgarDownloader initialized with cache at {self.cache_dir}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, cik: str, year: int, form_type: str = "10-K") -> Path:
        """Generate cache file path for a filing"""
        cache_key = f"{cik}_{year}_{form_type.replace('-', '_')}"
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_ticker_mapping(self) -> Dict[str, str]:
        """Load ticker to CIK mapping from SEC"""
        cache_file = self.cache_dir / "company_tickers.json"
        
        # Use cached mapping if recent (less than 1 day old)
        if cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(days=1):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return {v['ticker']: str(v['cik_str']).zfill(10) 
                       for v in data.values()}
        
        # Download fresh mapping
        try:
            self._rate_limit()
            response = self.session.get(self.COMPANY_TICKERS_URL)
            response.raise_for_status()
            
            data = response.json()
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            mapping = {v['ticker']: str(v['cik_str']).zfill(10) 
                      for v in data.values()}
            logger.info(f"Loaded {len(mapping)} ticker mappings")
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to load ticker mapping: {e}")
            return {}
    
    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        """Convert ticker symbol to CIK"""
        ticker = ticker.upper()
        cik = self.tickers_to_cik.get(ticker)
        if not cik:
            logger.warning(f"Ticker {ticker} not found in mapping")
        return cik
    
    def download_10k(self, 
                     identifier: str, 
                     year: int,
                     get_exhibits: bool = False) -> Optional[Dict]:
        """
        Download a single 10-K filing
        
        Args:
            identifier: Company ticker or CIK
            year: Fiscal year to download
            get_exhibits: Whether to download exhibits
            
        Returns:
            Dictionary containing filing data and metadata
        """
        # Convert ticker to CIK if needed
        if not identifier.isdigit():
            cik = self.ticker_to_cik(identifier)
            if not cik:
                return None
        else:
            cik = identifier.zfill(10)
        
        # Check cache first
        cache_path = self._get_cache_path(cik, year)
        if cache_path.exists() and not get_exhibits:
            logger.info(f"Loading cached filing for {cik} year {year}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        try:
            # Get company submissions
            submissions = self._get_company_submissions(cik)
            if not submissions:
                return None
            
            # Find 10-K filing for the specified year
            filing_data = self._find_10k_filing(submissions, year)
            if not filing_data:
                logger.warning(f"No 10-K found for {cik} year {year}")
                return None
            
            # Download filing content
            content = self._download_filing_content(
                cik, 
                filing_data['accessionNumber'],
                get_exhibits
            )
            
            if content:
                result = {
                    'cik': cik,
                    'ticker': identifier if not identifier.isdigit() else 
                             self._get_ticker_from_cik(cik),
                    'year': year,
                    'filing_date': filing_data['filingDate'],
                    'accession_number': filing_data['accessionNumber'],
                    'form': filing_data['form'],
                    'content': content,
                    'metadata': {
                        'downloaded_at': datetime.now().isoformat(),
                        'file_size': filing_data.get('size', 0),
                        'primary_document': filing_data.get('primaryDocument'),
                    }
                }
                
                # Cache the result
                with open(cache_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Successfully downloaded 10-K for {cik} year {year}")
                return result
                
        except Exception as e:
            logger.error(f"Error downloading 10-K for {cik} year {year}: {e}")
            return None
    
    def _get_company_submissions(self, cik: str) -> Optional[Dict]:
        """Get company submission history"""
        try:
            self._rate_limit()
            url = self.SUBMISSIONS_URL.format(cik)
            
            # Need different headers for data.sec.gov
            headers = {
                'User-Agent': self.user_agent,
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get submissions for CIK {cik}: {e}")
            return None
    
    def _find_10k_filing(self, submissions: Dict, year: int) -> Optional[Dict]:
        """Find 10-K filing for specified year in submissions"""
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        primary_documents = recent_filings.get('primaryDocument', [])
        
        for i, (form, date_str) in enumerate(zip(forms, filing_dates)):
            if form == '10-K':
                filing_year = int(date_str[:4])
                # 10-K is usually filed for previous fiscal year
                if filing_year == year or filing_year == year + 1:
                    return {
                        'form': form,
                        'filingDate': date_str,
                        'accessionNumber': accession_numbers[i].replace('-', ''),
                        'primaryDocument': primary_documents[i] if i < len(primary_documents) else None
                    }
        
        return None
    
    def _download_filing_content(self, 
                                cik: str, 
                                accession: str,
                                get_exhibits: bool = False) -> Optional[Dict]:
        """Download the actual filing content"""
        try:
            # Format accession number with hyphens if not already formatted
            if '-' not in accession:
                # Format: xxxxxxxxxx-xx-xxxxxx
                accession_formatted = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
            else:
                accession_formatted = accession
            
            # Remove leading zeros from CIK for URL
            cik_no_zeros = str(int(cik))
            
            # Construct filing URL
            filing_url = f"{self.ARCHIVES_URL}/{cik_no_zeros}/{accession}/{accession_formatted}.txt"
            
            self._rate_limit()
            response = self.session.get(filing_url)
            response.raise_for_status()
            
            content = {
                'full_text': response.text,
                'html_sections': self._extract_html_sections(response.text),
                'exhibits': []
            }
            
            # Download exhibits if requested
            if get_exhibits:
                exhibits_index = f"{self.ARCHIVES_URL}/{cik}/{accession}/{accession}-index.html"
                content['exhibits'] = self._download_exhibits(exhibits_index)
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to download filing content: {e}")
            return None
    
    def _extract_html_sections(self, full_text: str) -> Dict[str, str]:
        """Extract HTML sections from filing text"""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            ('business', r'Item 1\.?\s*Business'),
            ('risk_factors', r'Item 1A\.?\s*Risk Factors'),
            ('properties', r'Item 2\.?\s*Properties'),
            ('legal_proceedings', r'Item 3\.?\s*Legal Proceedings'),
            ('mda', r'Item 7\.?\s*Management.*Discussion'),
            ('financial_statements', r'Item 8\.?\s*Financial Statements'),
        ]
        
        # Parse HTML if present
        if '<html>' in full_text.lower():
            soup = BeautifulSoup(full_text, 'html.parser')
            text_content = soup.get_text()
        else:
            text_content = full_text
        
        # Extract sections (simplified - in production use more robust parsing)
        import re
        for section_name, pattern in section_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                start = match.start()
                # Find next section or end
                end = len(text_content)
                for _, next_pattern in section_patterns:
                    if next_pattern != pattern:
                        next_match = re.search(next_pattern, text_content[start:], re.IGNORECASE)
                        if next_match:
                            end = min(end, start + next_match.start())
                
                sections[section_name] = text_content[start:end][:50000]  # Limit size
        
        return sections
    
    def _download_exhibits(self, index_url: str) -> List[Dict]:
        """Download exhibit information"""
        # Simplified - would parse exhibit index and download key exhibits
        return []
    
    def _get_ticker_from_cik(self, cik: str) -> Optional[str]:
        """Reverse lookup ticker from CIK"""
        for ticker, cik_val in self.tickers_to_cik.items():
            if cik_val == cik:
                return ticker
        return None
    
    def bulk_download(self, 
                      tickers: List[str], 
                      years: List[int],
                      parallel: bool = False) -> List[Dict]:
        """
        Download multiple 10-K filings
        
        Args:
            tickers: List of ticker symbols
            years: List of years to download
            parallel: Whether to download in parallel (be careful with rate limits)
            
        Returns:
            List of filing dictionaries
        """
        results = []
        total = len(tickers) * len(years)
        completed = 0
        
        for ticker in tickers:
            for year in years:
                logger.info(f"Downloading {ticker} for year {year} ({completed+1}/{total})")
                
                filing = self.download_10k(ticker, year)
                if filing:
                    results.append(filing)
                
                completed += 1
                
                # Extra delay between companies
                if completed < total:
                    time.sleep(self.rate_limit_delay * 2)
        
        logger.info(f"Downloaded {len(results)}/{total} filings")
        return results
    
    def get_filing_metadata(self, identifier: str, years: List[int]) -> pd.DataFrame:
        """
        Get metadata for filings without downloading full content
        
        Returns:
            DataFrame with filing metadata
        """
        if not identifier.isdigit():
            cik = self.ticker_to_cik(identifier)
            if not cik:
                return pd.DataFrame()
        else:
            cik = identifier.zfill(10)
        
        submissions = self._get_company_submissions(cik)
        if not submissions:
            return pd.DataFrame()
        
        recent = submissions.get('filings', {}).get('recent', {})
        
        # Convert to DataFrame
        df = pd.DataFrame(recent)
        
        # Filter for 10-K filings in specified years
        df = df[df['form'] == '10-K']
        df['filing_year'] = pd.to_datetime(df['filingDate']).dt.year
        df = df[df['filing_year'].isin(years) | df['filing_year'].isin([y+1 for y in years])]
        
        return df[['form', 'filingDate', 'accessionNumber', 'primaryDocument', 'filing_year']]