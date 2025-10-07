"""
Document Parser Module
Parses HTML/PDF documents and extracts structured sections from 10-K filings
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup, NavigableString
import html2text
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section"""
    title: str
    content: str
    section_type: str
    item_number: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    word_count: int = 0
    char_count: int = 0
    tables: List[pd.DataFrame] = None
    subsections: List['Section'] = None


class DocumentParser:
    """
    Parses 10-K documents and extracts structured sections
    """
    
    # Standard 10-K sections mapping
    SECTION_PATTERNS = {
        'part_i': {
            'item_1': {
                'pattern': r'(?i)item\s*1\.?\s*(?:and\s*1a\.?\s*)?(?:business|description\s*of\s*business)',
                'title': 'Business',
                'alt_patterns': [r'(?i)^business$', r'(?i)description\s*of\s*business']
            },
            'item_1a': {
                'pattern': r'(?i)item\s*1a\.?\s*risk\s*factors',
                'title': 'Risk Factors',
                'alt_patterns': [r'(?i)^risk\s*factors$']
            },
            'item_1b': {
                'pattern': r'(?i)item\s*1b\.?\s*unresolved\s*staff\s*comments',
                'title': 'Unresolved Staff Comments'
            },
            'item_2': {
                'pattern': r'(?i)item\s*2\.?\s*properties',
                'title': 'Properties'
            },
            'item_3': {
                'pattern': r'(?i)item\s*3\.?\s*legal\s*proceedings',
                'title': 'Legal Proceedings'
            },
            'item_4': {
                'pattern': r'(?i)item\s*4\.?\s*mine\s*safety\s*disclosures',
                'title': 'Mine Safety Disclosures'
            }
        },
        'part_ii': {
            'item_5': {
                'pattern': r'(?i)item\s*5\.?\s*market\s*for\s*registrant',
                'title': 'Market for Registrant\'s Common Equity'
            },
            'item_6': {
                'pattern': r'(?i)item\s*6\.?\s*(?:selected|consolidated)\s*financial\s*data',
                'title': 'Selected Financial Data'
            },
            'item_7': {
                'pattern': r'(?i)item\s*7\.?\s*management[\'s]?\s*discussion\s*and\s*analysis',
                'title': 'Management\'s Discussion and Analysis (MD&A)',
                'alt_patterns': [r'(?i)md\&a', r'(?i)management.*discussion.*analysis']
            },
            'item_7a': {
                'pattern': r'(?i)item\s*7a\.?\s*quantitative\s*and\s*qualitative\s*disclosures',
                'title': 'Quantitative and Qualitative Disclosures About Market Risk'
            },
            'item_8': {
                'pattern': r'(?i)item\s*8\.?\s*financial\s*statements',
                'title': 'Financial Statements and Supplementary Data'
            },
            'item_9': {
                'pattern': r'(?i)item\s*9\.?\s*changes\s*in\s*and\s*disagreements',
                'title': 'Changes in and Disagreements with Accountants'
            },
            'item_9a': {
                'pattern': r'(?i)item\s*9a\.?\s*controls\s*and\s*procedures',
                'title': 'Controls and Procedures'
            },
            'item_9b': {
                'pattern': r'(?i)item\s*9b\.?\s*other\s*information',
                'title': 'Other Information'
            }
        },
        'part_iii': {
            'item_10': {
                'pattern': r'(?i)item\s*10\.?\s*directors.*executive\s*officers',
                'title': 'Directors, Executive Officers and Corporate Governance'
            },
            'item_11': {
                'pattern': r'(?i)item\s*11\.?\s*executive\s*compensation',
                'title': 'Executive Compensation'
            },
            'item_12': {
                'pattern': r'(?i)item\s*12\.?\s*security\s*ownership',
                'title': 'Security Ownership'
            },
            'item_13': {
                'pattern': r'(?i)item\s*13\.?\s*certain\s*relationships',
                'title': 'Certain Relationships and Related Transactions'
            },
            'item_14': {
                'pattern': r'(?i)item\s*14\.?\s*principal\s*account.*fees',
                'title': 'Principal Accountant Fees and Services'
            }
        },
        'part_iv': {
            'item_15': {
                'pattern': r'(?i)item\s*15\.?\s*exhibits.*financial\s*statement\s*schedules',
                'title': 'Exhibits and Financial Statement Schedules'
            },
            'item_16': {
                'pattern': r'(?i)item\s*16\.?\s*form\s*10-?k\s*summary',
                'title': 'Form 10-K Summary'
            }
        }
    }
    
    def __init__(self, 
                 min_section_length: int = 100,
                 max_section_length: int = 500000,
                 extract_tables: bool = True,
                 clean_text: bool = True):
        """
        Initialize document parser
        
        Args:
            min_section_length: Minimum characters for valid section
            max_section_length: Maximum characters per section
            extract_tables: Whether to extract tables as DataFrames
            clean_text: Whether to clean extracted text
        """
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length
        self.extract_tables = extract_tables
        self.clean_text = clean_text
        
        # Initialize HTML to text converter
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0  # Don't wrap text
        
        logger.info("DocumentParser initialized")
    
    def parse_html(self, html_content: str, filing_metadata: Dict = None) -> Dict[str, Section]:
        """
        Parse HTML content and extract sections
        
        Args:
            html_content: Raw HTML content
            filing_metadata: Optional metadata about the filing
            
        Returns:
            Dictionary of sections by item number
        """
        if not html_content:
            logger.warning("Empty HTML content provided")
            return {}
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text_content = soup.get_text(separator='\n')
        
        # Extract sections
        sections = self._extract_sections_from_text(text_content)
        
        # Extract tables if requested
        if self.extract_tables:
            tables = self._extract_tables_from_html(soup)
            self._associate_tables_with_sections(sections, tables)
        
        # Add metadata to sections
        if filing_metadata:
            for section in sections.values():
                section.metadata = filing_metadata
        
        logger.info(f"Parsed HTML document, extracted {len(sections)} sections")
        
        return sections
    
    def parse_text(self, text_content: str, filing_metadata: Dict = None) -> Dict[str, Section]:
        """
        Parse plain text content and extract sections
        
        Args:
            text_content: Plain text content
            filing_metadata: Optional metadata
            
        Returns:
            Dictionary of sections
        """
        sections = self._extract_sections_from_text(text_content)
        
        if filing_metadata:
            for section in sections.values():
                section.metadata = filing_metadata
        
        return sections
    
    def _extract_sections_from_text(self, text: str) -> Dict[str, Section]:
        """Extract sections based on patterns"""
        sections = {}
        text_lower = text.lower()
        
        # Find all section matches
        matches = []
        
        for part_name, part_items in self.SECTION_PATTERNS.items():
            for item_key, item_info in part_items.items():
                # Try main pattern
                pattern = item_info['pattern']
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    matches.append({
                        'item': item_key,
                        'title': item_info['title'],
                        'start': match.start(),
                        'end': match.end(),
                        'part': part_name
                    })
                
                # Try alternative patterns
                if 'alt_patterns' in item_info:
                    for alt_pattern in item_info['alt_patterns']:
                        for match in re.finditer(alt_pattern, text, re.IGNORECASE | re.MULTILINE):
                            matches.append({
                                'item': item_key,
                                'title': item_info['title'],
                                'start': match.start(),
                                'end': match.end(),
                                'part': part_name
                            })
        
        # Sort matches by position
        matches.sort(key=lambda x: x['start'])
        
        # Extract content between sections
        for i, match in enumerate(matches):
            start_pos = match['start']
            
            # Find end position (start of next section or end of document)
            if i < len(matches) - 1:
                end_pos = matches[i + 1]['start']
            else:
                end_pos = len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos]
            
            # Clean content if requested
            if self.clean_text:
                section_content = self._clean_text(section_content)
            
            # Skip if too short or too long
            if len(section_content) < self.min_section_length:
                continue
            if len(section_content) > self.max_section_length:
                section_content = section_content[:self.max_section_length]
            
            # Create section object
            section = Section(
                title=match['title'],
                content=section_content,
                section_type=match['part'],
                item_number=match['item'],
                word_count=len(section_content.split()),
                char_count=len(section_content)
            )
            
            sections[match['item']] = section
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)
        
        # Remove table of contents artifacts
        text = re.sub(r'\.{5,}', '', text)  # Remove dots used in TOC
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def _extract_tables_from_html(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables from HTML as DataFrames"""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            try:
                # Extract headers
                headers = []
                header_rows = table.find_all('tr')[:2]  # Check first 2 rows for headers
                
                for row in header_rows:
                    cols = row.find_all(['th', 'td'])
                    if cols and all(col.get_text().strip() for col in cols[:3]):  # At least 3 non-empty
                        headers = [col.get_text().strip() for col in cols]
                        break
                
                # Extract data rows
                data_rows = []
                for row in table.find_all('tr'):
                    cols = row.find_all('td')
                    if cols:
                        data_rows.append([col.get_text().strip() for col in cols])
                
                # Create DataFrame if we have data
                if data_rows:
                    if headers and len(headers) == len(data_rows[0]):
                        df = pd.DataFrame(data_rows, columns=headers)
                    else:
                        df = pd.DataFrame(data_rows)
                    
                    # Clean numeric columns
                    df = self._clean_table_data(df)
                    
                    tables.append({
                        'index': i,
                        'dataframe': df,
                        'position': str(table.sourceline) if hasattr(table, 'sourceline') else i
                    })
                    
            except Exception as e:
                logger.debug(f"Could not parse table {i}: {e}")
        
        logger.info(f"Extracted {len(tables)} tables from HTML")
        return tables
    
    def _clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert table data"""
        for col in df.columns:
            # Try to convert to numeric
            try:
                # Remove common symbols and convert
                df[col] = df[col].str.replace('$', '', regex=False)
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = df[col].str.replace('(', '-', regex=False)
                df[col] = df[col].str.replace(')', '', regex=False)
                
                # Try conversion
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df
    
    def _associate_tables_with_sections(self, sections: Dict[str, Section], tables: List[Dict]):
        """Associate extracted tables with their respective sections"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated table-to-section matching
        
        for section in sections.values():
            section.tables = []
        
        # For now, associate tables with Item 8 (Financial Statements)
        if 'item_8' in sections and tables:
            sections['item_8'].tables = [t['dataframe'] for t in tables[:10]]  # First 10 tables
    
    def extract_key_sections(self, filing_content: Dict) -> Dict[str, str]:
        """
        Extract key sections for analysis
        
        Args:
            filing_content: Dictionary with filing content
            
        Returns:
            Dictionary of key sections
        """
        key_sections = {}
        
        # If we have pre-extracted sections from EDGAR
        if 'html_sections' in filing_content:
            html_sections = filing_content['html_sections']
            
            # Map EDGAR sections to our standard sections
            mapping = {
                'business': 'item_1',
                'risk_factors': 'item_1a',
                'mda': 'item_7',
                'financial_statements': 'item_8'
            }
            
            for edgar_key, our_key in mapping.items():
                if edgar_key in html_sections:
                    key_sections[our_key] = html_sections[edgar_key]
        
        # Parse full text if available
        if 'full_text' in filing_content:
            parsed_sections = self.parse_text(filing_content['full_text'])
            key_sections.update(parsed_sections)
        
        return key_sections
    
    def extract_financial_tables(self, section_content: str) -> List[pd.DataFrame]:
        """
        Extract financial tables from a section
        
        Args:
            section_content: Section text content
            
        Returns:
            List of DataFrames containing financial tables
        """
        tables = []
        
        # Parse as HTML if it contains HTML tags
        if '<table' in section_content.lower():
            soup = BeautifulSoup(section_content, 'html.parser')
            tables = self._extract_tables_from_html(soup)
            return [t['dataframe'] for t in tables]
        
        # Otherwise try to extract structured data from text
        # This is a simplified implementation
        lines = section_content.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            # Simple heuristic: lines with multiple numbers are likely table rows
            numbers = re.findall(r'\d+[\d,]*\.?\d*', line)
            if len(numbers) >= 2:
                in_table = True
                current_table.append(line)
            elif in_table and not numbers:
                # End of table
                if current_table:
                    # Try to parse as DataFrame
                    try:
                        df = self._parse_text_table(current_table)
                        if df is not None and not df.empty:
                            tables.append(df)
                    except:
                        pass
                current_table = []
                in_table = False
        
        return tables
    
    def _parse_text_table(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Parse text lines into a DataFrame"""
        if not lines:
            return None
        
        # Split each line by multiple spaces or tabs
        data = []
        for line in lines:
            # Split by 2+ spaces or tabs
            parts = re.split(r'\s{2,}|\t+', line.strip())
            if parts:
                data.append(parts)
        
        if data:
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Try to identify header row
            if not df.iloc[0].str.match(r'^\d').any():
                # First row doesn't start with numbers, likely headers
                df.columns = df.iloc[0]
                df = df[1:]
            
            return df
        
        return None
    
    def extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract key metrics mentioned in text
        
        Args:
            text: Text content
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Revenue patterns
        revenue_patterns = [
            r'(?:total\s+)?(?:net\s+)?revenue[s]?\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion)?',
            r'(?:total\s+)?(?:net\s+)?sales\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion)?',
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(',', '')
                try:
                    metrics['revenue'] = float(value)
                    if 'billion' in match.group(0).lower():
                        metrics['revenue'] *= 1_000_000_000
                    elif 'million' in match.group(0).lower():
                        metrics['revenue'] *= 1_000_000
                    break
                except:
                    pass
        
        # Net income patterns
        income_patterns = [
            r'net\s+income\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion)?',
            r'net\s+(?:earnings|profit)\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion)?',
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(',', '')
                try:
                    metrics['net_income'] = float(value)
                    if 'billion' in match.group(0).lower():
                        metrics['net_income'] *= 1_000_000_000
                    elif 'million' in match.group(0).lower():
                        metrics['net_income'] *= 1_000_000
                    break
                except:
                    pass
        
        # EPS patterns
        eps_pattern = r'(?:diluted\s+)?earnings\s+per\s+share\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)'
        match = re.search(eps_pattern, text, re.IGNORECASE)
        if match:
            try:
                metrics['eps'] = float(match.group(1).replace(',', ''))
            except:
                pass
        
        # Gross margin patterns
        margin_pattern = r'gross\s+margin\s+(?:of\s+)?([\d,]+\.?\d*)%'
        match = re.search(margin_pattern, text, re.IGNORECASE)
        if match:
            try:
                metrics['gross_margin'] = float(match.group(1).replace(',', ''))
            except:
                pass
        
        return metrics
    
    def create_section_summary(self, section: Section, max_length: int = 500) -> str:
        """
        Create a summary of a section
        
        Args:
            section: Section object
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if not section.content:
            return ""
        
        # For now, return first N characters
        # In production, you'd use more sophisticated summarization
        summary = section.content[:max_length]
        
        # Try to end at a sentence
        last_period = summary.rfind('.')
        if last_period > max_length * 0.7:
            summary = summary[:last_period + 1]
        
        return summary.strip()
    
    def get_section_statistics(self, sections: Dict[str, Section]) -> pd.DataFrame:
        """
        Get statistics about extracted sections
        
        Args:
            sections: Dictionary of sections
            
        Returns:
            DataFrame with section statistics
        """
        stats = []
        
        for item_key, section in sections.items():
            stats.append({
                'item': item_key,
                'title': section.title,
                'type': section.section_type,
                'word_count': section.word_count,
                'char_count': section.char_count,
                'num_tables': len(section.tables) if section.tables else 0,
                'has_content': len(section.content) > 0
            })
        
        df = pd.DataFrame(stats)
        
        # Sort by item number
        df['item_order'] = df['item'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values('item_order').drop('item_order', axis=1)
        
        return df