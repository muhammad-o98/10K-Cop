"""
DuckDB Manager Module
Manages structured financial data storage and analytical queries using DuckDB
"""

import duckdb
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    Manages DuckDB database for financial data storage and analytics
    """
    
    def __init__(self,
                 db_path: str = "./data/analytics/10k_cop.duckdb",
                 read_only: bool = False,
                 memory_limit: str = "4GB"):
        """
        Initialize DuckDB connection and create tables
        
        Args:
            db_path: Path to DuckDB database file
            read_only: Open database in read-only mode
            memory_limit: Memory limit for DuckDB operations
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        
        # Connect to DuckDB
        self.conn = duckdb.connect(
            str(self.db_path),
            read_only=read_only,
            config={'memory_limit': memory_limit}
        )
        
        # Create tables if not read-only
        if not read_only:
            self._create_tables()
        
        logger.info(f"DuckDBManager initialized with database at {self.db_path}")
    
    def _create_tables(self):
        """Create database schema"""
        
        # Companies table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                cik VARCHAR PRIMARY KEY,
                ticker VARCHAR UNIQUE,
                name VARCHAR,
                sic_code VARCHAR,
                industry VARCHAR,
                sector VARCHAR,
                fiscal_year_end VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Filings table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                accession_number VARCHAR PRIMARY KEY,
                cik VARCHAR REFERENCES companies(cik),
                ticker VARCHAR,
                form_type VARCHAR,
                filing_date DATE,
                fiscal_year INTEGER,
                fiscal_period VARCHAR,
                file_path VARCHAR,
                content_hash VARCHAR,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Financial metrics table (XBRL data)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY,
                accession_number VARCHAR REFERENCES filings(accession_number),
                cik VARCHAR REFERENCES companies(cik),
                ticker VARCHAR,
                metric_name VARCHAR,
                gaap_field VARCHAR,
                value DECIMAL(20, 4),
                unit VARCHAR,
                fiscal_year INTEGER,
                fiscal_period VARCHAR,
                start_date DATE,
                end_date DATE,
                instant_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_ticker_year 
            ON financial_metrics(ticker, fiscal_year)
        """)
        
        # Calculated ratios table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_ratios (
                id INTEGER PRIMARY KEY,
                cik VARCHAR REFERENCES companies(cik),
                ticker VARCHAR,
                fiscal_year INTEGER,
                fiscal_period VARCHAR,
                
                -- Liquidity Ratios
                current_ratio DECIMAL(10, 4),
                quick_ratio DECIMAL(10, 4),
                cash_ratio DECIMAL(10, 4),
                
                -- Leverage Ratios
                debt_to_equity DECIMAL(10, 4),
                debt_to_assets DECIMAL(10, 4),
                interest_coverage DECIMAL(10, 4),
                
                -- Profitability Ratios
                gross_margin DECIMAL(10, 4),
                operating_margin DECIMAL(10, 4),
                net_margin DECIMAL(10, 4),
                roa DECIMAL(10, 4),  -- Return on Assets
                roe DECIMAL(10, 4),  -- Return on Equity
                roic DECIMAL(10, 4), -- Return on Invested Capital
                
                -- Efficiency Ratios
                asset_turnover DECIMAL(10, 4),
                inventory_turnover DECIMAL(10, 4),
                receivables_turnover DECIMAL(10, 4),
                days_inventory DECIMAL(10, 2),
                days_receivables DECIMAL(10, 2),
                cash_conversion_cycle DECIMAL(10, 2),
                
                -- Valuation Ratios (if stock price available)
                pe_ratio DECIMAL(10, 4),
                pb_ratio DECIMAL(10, 4),
                ps_ratio DECIMAL(10, 4),
                ev_to_ebitda DECIMAL(10, 4),
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Stock prices table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY,
                ticker VARCHAR,
                date DATE,
                open DECIMAL(10, 2),
                high DECIMAL(10, 2),
                low DECIMAL(10, 2),
                close DECIMAL(10, 2),
                adjusted_close DECIMAL(10, 2),
                volume BIGINT,
                
                -- Returns
                daily_return DECIMAL(10, 6),
                log_return DECIMAL(10, 6),
                
                -- Moving averages
                sma_20 DECIMAL(10, 2),
                sma_50 DECIMAL(10, 2),
                sma_200 DECIMAL(10, 2),
                
                -- Technical indicators
                rsi_14 DECIMAL(10, 2),
                macd DECIMAL(10, 4),
                macd_signal DECIMAL(10, 4),
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # Macro indicators table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_indicators (
                id INTEGER PRIMARY KEY,
                series_id VARCHAR,
                series_name VARCHAR,
                date DATE,
                value DECIMAL(20, 4),
                unit VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(series_id, date)
            )
        """)
        
        # Document sections table (for parsed 10-K sections)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS document_sections (
                id INTEGER PRIMARY KEY,
                accession_number VARCHAR REFERENCES filings(accession_number),
                section_type VARCHAR,
                section_title VARCHAR,
                content TEXT,
                word_count INTEGER,
                char_count INTEGER,
                has_tables BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Text chunks table (for RAG)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                accession_number VARCHAR REFERENCES filings(accession_number),
                ticker VARCHAR,
                section VARCHAR,
                chunk_index INTEGER,
                content TEXT,
                word_count INTEGER,
                token_count INTEGER,
                has_financial_data BOOLEAN,
                has_risk_mention BOOLEAN,
                embedding_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Peer groups table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS peer_groups (
                id INTEGER PRIMARY KEY,
                ticker VARCHAR,
                peer_ticker VARCHAR,
                peer_type VARCHAR,  -- 'sic', 'market_cap', 'custom'
                similarity_score DECIMAL(5, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info("Database tables created successfully")
    
    def insert_company(self, company_data: Dict) -> bool:
        """Insert or update company information"""
        try:
            # First try to insert
            self.conn.execute("""
                INSERT OR REPLACE INTO companies (cik, ticker, name, sic_code, industry, sector, fiscal_year_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                company_data.get('cik'),
                company_data.get('ticker'),
                company_data.get('name'),
                company_data.get('sic_code'),
                company_data.get('industry'),
                company_data.get('sector'),
                company_data.get('fiscal_year_end')
            ])
            return True
        except Exception as e:
            logger.error(f"Error inserting company: {e}")
            return False
    
    def insert_filing(self, filing_data: Dict) -> bool:
        """Insert filing information"""
        try:
            # Ensure company exists first
            if filing_data.get('cik'):
                self.conn.execute("""
                    INSERT OR IGNORE INTO companies (cik, ticker)
                    VALUES (?, ?)
                """, [filing_data.get('cik'), filing_data.get('ticker')])
            
            self.conn.execute("""
                INSERT OR REPLACE INTO filings (
                    accession_number, cik, ticker, form_type, filing_date,
                    fiscal_year, fiscal_period, file_path, content_hash, processed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                filing_data.get('accession_number'),
                filing_data.get('cik'),
                filing_data.get('ticker'),
                filing_data.get('form_type'),
                filing_data.get('filing_date'),
                filing_data.get('fiscal_year'),
                filing_data.get('fiscal_period'),
                filing_data.get('file_path'),
                filing_data.get('content_hash'),
                filing_data.get('processed', False)
            ])
            return True
        except Exception as e:
            logger.error(f"Error inserting filing: {e}")
            return False
    
    def insert_financial_metrics(self, metrics_df: pd.DataFrame, 
                                accession_number: str,
                                cik: str,
                                ticker: str) -> int:
        """
        Insert financial metrics from XBRL data
        
        Returns:
            Number of metrics inserted
        """
        if metrics_df.empty:
            return 0
        
        try:
            # Prepare data for insertion
            metrics_df = metrics_df.copy()
            
            # Map columns properly
            for _, row in metrics_df.iterrows():
                self.conn.execute("""
                    INSERT INTO financial_metrics (
                        accession_number, cik, ticker, metric_name, gaap_field,
                        value, unit, fiscal_year, fiscal_period, 
                        start_date, end_date, instant_date
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    accession_number,
                    cik,
                    ticker,
                    row.get('metric'),
                    row.get('gaap_field'),
                    row.get('value'),
                    row.get('unit'),
                    row.get('fiscal_year'),
                    row.get('fiscal_period'),
                    row.get('start_date'),
                    row.get('end_date'),
                    row.get('instant')
                ])
            
            rows_inserted = len(metrics_df)
            logger.info(f"Inserted {rows_inserted} financial metrics")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error inserting financial metrics: {e}")
            return 0
    
    def calculate_and_store_ratios(self, ticker: str, fiscal_year: int) -> Dict:
        """
        Calculate financial ratios and store in database
        
        Returns:
            Dictionary of calculated ratios
        """
        try:
            # Query metrics for the given ticker and year
            metrics_df = self.conn.execute("""
                SELECT metric_name, value
                FROM financial_metrics
                WHERE ticker = ? AND fiscal_year = ?
                AND fiscal_period = 'FY'
            """, [ticker, fiscal_year]).fetchdf()
            
            if metrics_df.empty:
                logger.warning(f"No metrics found for {ticker} year {fiscal_year}")
                return {}
            
            # Pivot to get metrics as columns
            metrics = metrics_df.set_index('metric_name')['value'].to_dict()
            
            # Calculate ratios
            ratios = {}
            
            # Liquidity Ratios
            if 'CurrentAssets' in metrics and 'CurrentLiabilities' in metrics:
                ratios['current_ratio'] = metrics['CurrentAssets'] / metrics['CurrentLiabilities']
            
            if 'Cash' in metrics and 'CurrentLiabilities' in metrics:
                ratios['cash_ratio'] = metrics['Cash'] / metrics['CurrentLiabilities']
            
            # Leverage Ratios
            if 'LongTermDebt' in metrics and 'Equity' in metrics:
                ratios['debt_to_equity'] = metrics['LongTermDebt'] / metrics['Equity']
            
            if 'Liabilities' in metrics and 'Assets' in metrics:
                ratios['debt_to_assets'] = metrics['Liabilities'] / metrics['Assets']
            
            # Profitability Ratios
            if 'GrossProfit' in metrics and 'Revenue' in metrics:
                ratios['gross_margin'] = metrics['GrossProfit'] / metrics['Revenue']
            
            if 'OperatingIncome' in metrics and 'Revenue' in metrics:
                ratios['operating_margin'] = metrics['OperatingIncome'] / metrics['Revenue']
            
            if 'NetIncome' in metrics and 'Revenue' in metrics:
                ratios['net_margin'] = metrics['NetIncome'] / metrics['Revenue']
            
            if 'NetIncome' in metrics and 'Assets' in metrics:
                ratios['roa'] = metrics['NetIncome'] / metrics['Assets']
            
            if 'NetIncome' in metrics and 'Equity' in metrics:
                ratios['roe'] = metrics['NetIncome'] / metrics['Equity']
            
            # Efficiency Ratios
            if 'Revenue' in metrics and 'Assets' in metrics:
                ratios['asset_turnover'] = metrics['Revenue'] / metrics['Assets']
            
            if 'CostOfRevenue' in metrics and 'Inventory' in metrics and metrics['Inventory'] > 0:
                ratios['inventory_turnover'] = metrics['CostOfRevenue'] / metrics['Inventory']
                ratios['days_inventory'] = 365 / ratios['inventory_turnover']
            
            if 'Revenue' in metrics and 'AccountsReceivable' in metrics and metrics['AccountsReceivable'] > 0:
                ratios['receivables_turnover'] = metrics['Revenue'] / metrics['AccountsReceivable']
                ratios['days_receivables'] = 365 / ratios['receivables_turnover']
            
            # Cash Conversion Cycle
            if 'days_inventory' in ratios and 'days_receivables' in ratios:
                ratios['cash_conversion_cycle'] = ratios['days_inventory'] + ratios['days_receivables']
            
            # Get CIK for the ticker
            cik = self.conn.execute(
                "SELECT cik FROM companies WHERE ticker = ?", 
                [ticker]
            ).fetchone()[0]
            
            # Insert ratios into database
            if ratios:
                columns = ['cik', 'ticker', 'fiscal_year', 'fiscal_period'] + list(ratios.keys())
                values = [cik, ticker, fiscal_year, 'FY'] + list(ratios.values())
                
                placeholders = ','.join(['?' for _ in values])
                column_names = ','.join(columns)
                
                self.conn.execute(f"""
                    INSERT INTO financial_ratios ({column_names})
                    VALUES ({placeholders})
                """, values)
                
                logger.info(f"Calculated and stored {len(ratios)} ratios for {ticker} year {fiscal_year}")
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return {}
    
    def insert_stock_prices(self, prices_df: pd.DataFrame, ticker: str) -> int:
        """Insert stock price data"""
        if prices_df.empty:
            return 0
        
        try:
            prices_df = prices_df.copy()
            prices_df['ticker'] = ticker
            
            # Select relevant columns
            columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 
                      'volume', 'daily_return', 'sma_20', 'sma_50', 
                      'rsi_14', 'macd', 'macd_signal']
            
            # Filter columns that exist
            available_columns = [col for col in columns if col in prices_df.columns]
            prices_to_insert = prices_df[available_columns]
            
            # Insert using ON CONFLICT to handle duplicates
            for _, row in prices_to_insert.iterrows():
                values = [row[col] if pd.notna(row[col]) else None for col in available_columns]
                placeholders = ','.join(['?' for _ in values])
                columns_str = ','.join(available_columns)
                
                self.conn.execute(f"""
                    INSERT INTO stock_prices ({columns_str})
                    VALUES ({placeholders})
                    ON CONFLICT (ticker, date) DO NOTHING
                """, values)
            
            rows_inserted = len(prices_to_insert)
            logger.info(f"Inserted {rows_inserted} stock price records for {ticker}")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error inserting stock prices: {e}")
            return 0
    
    def insert_macro_indicators(self, macro_df: pd.DataFrame) -> int:
        """Insert macroeconomic indicators"""
        if macro_df.empty:
            return 0
        
        try:
            count = 0
            for _, row in macro_df.iterrows():
                self.conn.execute("""
                    INSERT INTO macro_indicators (series_id, series_name, date, value, unit)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (series_id, date) DO UPDATE SET
                        value = EXCLUDED.value
                """, [
                    row.get('series_id'),
                    row.get('series_name'),
                    row.get('date'),
                    row.get('value'),
                    row.get('unit')
                ])
                count += 1
            
            logger.info(f"Inserted {count} macro indicator records")
            return count
            
        except Exception as e:
            logger.error(f"Error inserting macro indicators: {e}")
            return 0
    
    def query_ratios(self, 
                    tickers: Optional[List[str]] = None,
                    fiscal_years: Optional[List[int]] = None,
                    ratio_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query financial ratios
        
        Args:
            tickers: List of tickers to query
            fiscal_years: List of fiscal years
            ratio_names: Specific ratio names to retrieve
            
        Returns:
            DataFrame with ratios
        """
        query = "SELECT * FROM financial_ratios WHERE 1=1"
        params = []
        
        if tickers:
            placeholders = ','.join(['?' for _ in tickers])
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        
        if fiscal_years:
            placeholders = ','.join(['?' for _ in fiscal_years])
            query += f" AND fiscal_year IN ({placeholders})"
            params.extend(fiscal_years)
        
        query += " ORDER BY ticker, fiscal_year"
        
        df = self.conn.execute(query, params).fetchdf()
        
        # Filter columns if specific ratios requested
        if ratio_names:
            base_columns = ['ticker', 'fiscal_year', 'fiscal_period']
            columns_to_keep = base_columns + [col for col in ratio_names if col in df.columns]
            df = df[columns_to_keep]
        
        return df
    
    def get_peer_comparison(self, ticker: str, fiscal_year: int) -> pd.DataFrame:
        """Get peer comparison data"""
        # Get peer tickers
        peers_df = self.conn.execute("""
            SELECT DISTINCT peer_ticker
            FROM peer_groups
            WHERE ticker = ?
            AND peer_type = 'sic'
            LIMIT 10
        """, [ticker]).fetchdf()
        
        if peers_df.empty:
            # Fallback to companies in same sector
            sector = self.conn.execute("""
                SELECT sector FROM companies WHERE ticker = ?
            """, [ticker]).fetchone()
            
            if sector:
                peers_df = self.conn.execute("""
                    SELECT ticker as peer_ticker
                    FROM companies
                    WHERE sector = ? AND ticker != ?
                    LIMIT 10
                """, [sector[0], ticker]).fetchdf()
        
        if peers_df.empty:
            return pd.DataFrame()
        
        # Get ratios for ticker and peers
        all_tickers = [ticker] + peers_df['peer_ticker'].tolist()
        ratios_df = self.query_ratios(all_tickers, [fiscal_year])
        
        # Calculate percentiles
        if not ratios_df.empty:
            numeric_columns = ratios_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col not in ['fiscal_year', 'id']]
            
            for col in numeric_columns:
                ratios_df[f'{col}_percentile'] = ratios_df[col].rank(pct=True) * 100
        
        return ratios_df
    
    def export_to_parquet(self, table_name: str, output_path: str):
        """Export table to Parquet file"""
        df = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {table_name} to {output_path}")
    
    def create_analytics_views(self):
        """Create analytical views for easier querying"""
        
        # Year-over-year metrics view
        self.conn.execute("""
            CREATE OR REPLACE VIEW yoy_metrics AS
            SELECT 
                a.ticker,
                a.fiscal_year,
                a.metric_name,
                a.value as current_value,
                b.value as prior_value,
                (a.value - b.value) as absolute_change,
                ((a.value - b.value) / NULLIF(b.value, 0)) * 100 as percent_change
            FROM financial_metrics a
            LEFT JOIN financial_metrics b
                ON a.ticker = b.ticker 
                AND a.metric_name = b.metric_name
                AND b.fiscal_year = a.fiscal_year - 1
            WHERE a.fiscal_period = 'FY' AND b.fiscal_period = 'FY'
        """)
        
        # Trend analysis view
        self.conn.execute("""
            CREATE OR REPLACE VIEW ratio_trends AS
            SELECT 
                ticker,
                fiscal_year,
                current_ratio,
                LAG(current_ratio) OVER (PARTITION BY ticker ORDER BY fiscal_year) as prev_current_ratio,
                gross_margin,
                LAG(gross_margin) OVER (PARTITION BY ticker ORDER BY fiscal_year) as prev_gross_margin,
                roe,
                LAG(roe) OVER (PARTITION BY ticker ORDER BY fiscal_year) as prev_roe
            FROM financial_ratios
            ORDER BY ticker, fiscal_year
        """)
        
        logger.info("Created analytics views")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")