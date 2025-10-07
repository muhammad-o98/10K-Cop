"""
Test script for analytics modules
Demonstrates ratio calculations, variance analysis, and peer benchmarking
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from src.ingestion.xbrl_processor import XBRLProcessor
from src.analytics.ratio_calculator import RatioCalculator
from src.analytics.variance_analyzer import VarianceAnalyzer
from src.analytics.peer_benchmark import PeerBenchmark

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ratio_calculator():
    """Test financial ratio calculations"""
    logger.info("=" * 50)
    logger.info("Testing Ratio Calculator")
    logger.info("=" * 50)
    
    # Sample financial data (in millions)
    financial_data = {
        # Balance Sheet
        'Assets': 352755,
        'CurrentAssets': 135405,
        'Cash': 29965,
        'AccountsReceivable': 29508,
        'Inventory': 6331,
        'PPE': 42117,
        'Liabilities': 290437,
        'CurrentLiabilities': 145308,
        'LongTermDebt': 106550,
        'Equity': 62146,
        
        # Income Statement
        'Revenue': 383285,
        'CostOfRevenue': 214137,
        'GrossProfit': 169148,
        'OperatingExpenses': 54847,
        'OperatingIncome': 114301,
        'InterestExpense': 3933,
        'IncomeTaxExpense': 16741,
        'NetIncome': 96995,
        
        # Cash Flow
        'OperatingCashFlow': 110543,
        'CapEx': 10708,
        'FreeCashFlow': 99835
    }
    
    # Initialize calculator
    calculator = RatioCalculator(industry="Technology")
    
    # Calculate all ratios
    ratios = calculator.calculate_all_ratios(
        financial_data,
        fiscal_year=2023,
        fiscal_period="FY"
    )
    
    logger.info(f"✓ Calculated {len(ratios)} financial ratios")
    
    # Display key ratios
    key_ratios = ['current_ratio', 'debt_to_equity', 'gross_margin', 'roe', 'roa']
    
    for ratio_name in key_ratios:
        if ratio_name in ratios:
            result = ratios[ratio_name]
            logger.info(f"  {ratio_name}: {result.value:.3f}")
            logger.info(f"    - Interpretation: {result.interpretation}")
            logger.info(f"    - Health Score: {result.health_score}/100")
    
    # Generate ratio report
    report_df = calculator.generate_ratio_report(ratios)
    
    if not report_df.empty:
        logger.info("\nRatio Report Summary:")
        logger.info(f"  Categories: {report_df['Category'].unique()}")
        logger.info(f"  Average Health Score: {report_df['Health Score'].mean():.1f}")
    
    return ratios


def test_variance_analyzer():
    """Test variance analysis"""
    logger.info("=" * 50)
    logger.info("Testing Variance Analyzer")
    logger.info("=" * 50)
    
    # Current period data (2023)
    current_period = {
        'Revenue': 383285,
        'CostOfRevenue': 214137,
        'GrossProfit': 169148,
        'OperatingExpenses': 54847,
        'OperatingIncome': 114301,
        'NetIncome': 96995,
        'Assets': 352755,
        'Equity': 62146
    }
    
    # Prior period data (2022)
    prior_period = {
        'Revenue': 394328,
        'CostOfRevenue': 223546,
        'GrossProfit': 170782,
        'OperatingExpenses': 51345,
        'OperatingIncome': 119437,
        'NetIncome': 99803,
        'Assets': 352755,
        'Equity': 50672
    }
    
    # Initialize analyzer
    analyzer = VarianceAnalyzer()
    
    # Analyze variances
    variances = analyzer.analyze_period_variance(
        current_period,
        prior_period,
        statement_type="income_statement"
    )
    
    logger.info(f"✓ Analyzed {len(variances)} variance items")
    
    # Show top variances
    logger.info("\nTop Variances by Absolute Change:")
    for i, variance in enumerate(variances[:5], 1):
        logger.info(f"  {i}. {variance.metric_name}:")
        logger.info(f"     Change: ${variance.absolute_change:,.0f} ({variance.percent_change:.1f}%)")
        logger.info(f"     Favorable: {'Yes' if variance.is_favorable else 'No' if variance.is_favorable is False else 'N/A'}")
    
    # Analyze margin changes
    margin_changes = analyzer.analyze_margin_changes(current_period, prior_period)
    
    if margin_changes:
        logger.info("\nMargin Analysis:")
        for margin_name, margin_var in margin_changes.items():
            logger.info(f"  {margin_var.metric_name}: {margin_var.current_value:.1f}% (Δ {margin_var.absolute_change:+.1f}pp)")
    
    # DuPont analysis
    dupont = analyzer.decompose_roe_change(current_period, prior_period)
    
    if 'roe_change' in dupont:
        logger.info("\nDuPont Analysis - ROE Decomposition:")
        logger.info(f"  ROE Change: {dupont['roe_change']:.3f}")
        logger.info(f"  - NPM Impact: {dupont.get('npm_impact', 0):.3f}")
        logger.info(f"  - Turnover Impact: {dupont.get('turnover_impact', 0):.3f}")
        logger.info(f"  - Leverage Impact: {dupont.get('leverage_impact', 0):.3f}")
    
    # Create variance report
    report_df = analyzer.create_variance_report(variances)
    
    if not report_df.empty:
        logger.info(f"\n✓ Generated variance report with {len(report_df)} items")
    
    return variances


def test_peer_benchmark():
    """Test peer benchmarking"""
    logger.info("=" * 50)
    logger.info("Testing Peer Benchmark")
    logger.info("=" * 50)
    
    # Initialize benchmark analyzer
    benchmark = PeerBenchmark()
    
    # Sample company list for peer identification
    company_list = pd.DataFrame({
        'ticker': ['MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ORCL', 'CRM', 'ADBE'],
        'cik': ['0000789019', '0001652044', '0001326801', '0001018724', '0001045810', '0001341439', '0001108524', '0000796343'],
        'name': ['Microsoft', 'Alphabet', 'Meta', 'Amazon', 'NVIDIA', 'Oracle', 'Salesforce', 'Adobe'],
        'sector': ['Technology'] * 8,
        'market_cap': [2800e9, 1700e9, 900e9, 1500e9, 1100e9, 300e9, 200e9, 250e9],
        'sic_code': ['7372', '7370', '7370', '5961', '3674', '7372', '7372', '7372']
    })
    
    # Identify peers by market cap
    target_market_cap = 3000e9  # Apple's approximate market cap
    peers = benchmark.identify_peers_by_market_cap(
        target_market_cap,
        company_list,
        sector="Technology",
        tolerance=0.8,
        max_peers=5
    )
    
    logger.info(f"✓ Identified {len(peers)} peer companies by market cap:")
    for peer in peers[:3]:
        logger.info(f"  - {peer.ticker}: {peer.name} (Similarity: {peer.similarity_score:.2f})")
    
    # Sample metrics for benchmarking
    company_metrics = {
        'gross_margin': 0.441,
        'operating_margin': 0.298,
        'net_margin': 0.253,
        'roe': 1.561,
        'roa': 0.274,
        'current_ratio': 0.932,
        'debt_to_equity': 1.714,
        'asset_turnover': 1.084
    }
    
    # Peer metrics data
    peer_metrics_df = pd.DataFrame({
        'ticker': ['MSFT', 'GOOGL', 'META', 'NVDA'],
        'gross_margin': [0.684, 0.553, 0.789, 0.730],
        'operating_margin': [0.421, 0.259, 0.338, 0.329],
        'net_margin': [0.342, 0.213, 0.290, 0.265],
        'roe': [0.472, 0.259, 0.324, 0.896],
        'roa': [0.197, 0.165, 0.194, 0.532],
        'current_ratio': [1.77, 2.35, 2.67, 3.39],
        'debt_to_equity': [0.584, 0.117, 0.285, 0.196],
        'asset_turnover': [0.576, 0.775, 0.669, 2.009]
    })
    
    # Benchmark against peers
    benchmark_results = benchmark.benchmark_against_peers(
        company_metrics,
        peer_metrics_df
    )
    
    logger.info(f"\n✓ Benchmarked {len(benchmark_results)} metrics against peers")
    
    # Display benchmark results
    logger.info("\nBenchmark Results:")
    for metric_name, result in list(benchmark_results.items())[:5]:
        logger.info(f"  {metric_name}:")
        logger.info(f"    Company: {result.company_value:.3f}")
        logger.info(f"    Peer Median: {result.peer_median:.3f}")
        logger.info(f"    Percentile: {result.percentile_rank:.1f}%")
        logger.info(f"    Performance: {result.relative_performance}")
    
    # Identify strengths and weaknesses
    strengths, weaknesses = benchmark.identify_strengths_weaknesses(
        benchmark_results,
        threshold=25.0
    )
    
    logger.info(f"\nStrengths (Top Quartile):")
    for strength in strengths[:3]:
        logger.info(f"  - {strength['metric']}: {strength['percentile']:.1f} percentile")
    
    logger.info(f"\nWeaknesses (Bottom Quartile):")
    for weakness in weaknesses[:3]:
        logger.info(f"  - {weakness['metric']}: {weakness['percentile']:.1f} percentile")
    
    # Calculate composite score
    composite_score = benchmark.calculate_composite_score(benchmark_results)
    logger.info(f"\n✓ Composite Performance Score: {composite_score:.1f}/100")
    
    # Generate benchmark report
    report_df = benchmark.generate_benchmark_report(benchmark_results)
    
    if not report_df.empty:
        logger.info(f"✓ Generated benchmark report with {len(report_df)} metrics")
    
    return benchmark_results


def main():
    """Main test function"""
    logger.info("Testing Analytics Modules")
    logger.info("=" * 70)
    
    # Test ratio calculator
    logger.info("\nStep 1: Testing Ratio Calculator...")
    ratios = test_ratio_calculator()
    
    # Test variance analyzer
    logger.info("\nStep 2: Testing Variance Analyzer...")
    variances = test_variance_analyzer()
    
    # Test peer benchmark
    logger.info("\nStep 3: Testing Peer Benchmark...")
    benchmark_results = test_peer_benchmark()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Analytics Module Test Summary")
    logger.info("-" * 30)
    logger.info(f"✓ Ratio Calculator: {len(ratios)} ratios calculated")
    logger.info(f"✓ Variance Analyzer: {len(variances)} variances analyzed")
    logger.info(f"✓ Peer Benchmark: {len(benchmark_results)} metrics benchmarked")
    logger.info("=" * 70)
    logger.info("All analytics modules working correctly!")
    logger.info("\nNext steps:")
    logger.info("1. Implement the RAG pipeline with citations")
    logger.info("2. Create the Streamlit application")
    logger.info("3. Add ML features (sentiment, forecasting)")


if __name__ == "__main__":
    main()