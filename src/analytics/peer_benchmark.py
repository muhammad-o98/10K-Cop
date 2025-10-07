"""
Peer Benchmark Module
Performs peer comparison and industry benchmarking analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


@dataclass
class PeerCompany:
    """Container for peer company information"""
    ticker: str
    cik: str
    name: str
    sector: str
    industry: str
    market_cap: float
    sic_code: Optional[str] = None
    similarity_score: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Container for benchmark comparison results"""
    metric_name: str
    company_value: float
    peer_median: float
    peer_mean: float
    peer_min: float
    peer_max: float
    peer_q1: float
    peer_q3: float
    percentile_rank: float
    z_score: float
    relative_performance: str
    peer_count: int


class PeerBenchmark:
    """
    Performs comprehensive peer comparison and industry benchmarking
    """
    
    # Define SIC code ranges for industries
    SIC_INDUSTRY_MAPPING = {
        'Technology': [(3570, 3579), (3670, 3679), (7370, 7379)],
        'Healthcare': [(2830, 2839), (3840, 3849), (8000, 8099)],
        'Financial': [(6000, 6799)],
        'Consumer': [(2000, 2099), (5200, 5999)],
        'Industrial': [(3400, 3499), (3500, 3599)],
        'Energy': [(1300, 1399), (2900, 2999)],
        'Materials': [(1000, 1099), (2800, 2899), (3300, 3399)],
        'Utilities': [(4900, 4999)],
        'Real Estate': [(6500, 6599)],
        'Telecom': [(4800, 4899)]
    }
    
    # Key metrics for peer comparison
    COMPARISON_METRICS = {
        'Valuation': ['pe_ratio', 'price_to_book', 'ev_to_ebitda', 'price_to_sales'],
        'Profitability': ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa', 'roic'],
        'Liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
        'Leverage': ['debt_to_equity', 'debt_to_assets', 'interest_coverage'],
        'Efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover'],
        'Growth': ['revenue_growth', 'earnings_growth', 'fcf_growth']
    }
    
    def __init__(self, db_connection=None):
        """
        Initialize peer benchmark analyzer
        
        Args:
            db_connection: Database connection for fetching peer data
        """
        self.db_connection = db_connection
        logger.info("PeerBenchmark initialized")
    
    def identify_peers_by_sic(self, 
                             target_sic: str,
                             company_list: pd.DataFrame,
                             max_peers: int = 20) -> List[PeerCompany]:
        """
        Identify peer companies by SIC code
        
        Args:
            target_sic: Target company's SIC code
            company_list: DataFrame with company information
            max_peers: Maximum number of peers to return
            
        Returns:
            List of peer companies
        """
        peers = []
        
        if not target_sic or company_list.empty:
            return peers
        
        # Find companies with same 4-digit SIC
        same_sic = company_list[company_list['sic_code'] == target_sic]
        
        # If not enough, expand to 3-digit SIC
        if len(same_sic) < 5:
            sic_prefix = target_sic[:3]
            same_sic = company_list[company_list['sic_code'].str.startswith(sic_prefix)]
        
        # If still not enough, expand to 2-digit SIC
        if len(same_sic) < 5:
            sic_prefix = target_sic[:2]
            same_sic = company_list[company_list['sic_code'].str.startswith(sic_prefix)]
        
        # Convert to PeerCompany objects
        for _, row in same_sic.head(max_peers).iterrows():
            peer = PeerCompany(
                ticker=row['ticker'],
                cik=row['cik'],
                name=row.get('name', ''),
                sector=row.get('sector', ''),
                industry=row.get('industry', ''),
                market_cap=row.get('market_cap', 0),
                sic_code=row['sic_code']
            )
            peers.append(peer)
        
        logger.info(f"Identified {len(peers)} peers by SIC code")
        return peers
    
    def identify_peers_by_market_cap(self,
                                    target_market_cap: float,
                                    company_list: pd.DataFrame,
                                    sector: Optional[str] = None,
                                    tolerance: float = 0.5,
                                    max_peers: int = 20) -> List[PeerCompany]:
        """
        Identify peer companies by market capitalization
        
        Args:
            target_market_cap: Target company's market cap
            company_list: DataFrame with company information
            sector: Optional sector filter
            tolerance: Tolerance range (0.5 = 50% to 150% of target)
            max_peers: Maximum number of peers
            
        Returns:
            List of peer companies
        """
        peers = []
        
        # Filter by sector if specified
        if sector:
            company_list = company_list[company_list['sector'] == sector]
        
        # Define market cap range
        min_cap = target_market_cap * (1 - tolerance)
        max_cap = target_market_cap * (1 + tolerance)
        
        # Filter by market cap range
        similar_companies = company_list[
            (company_list['market_cap'] >= min_cap) &
            (company_list['market_cap'] <= max_cap)
        ]
        
        # Calculate similarity score based on market cap distance
        similar_companies['similarity'] = 1 - abs(
            similar_companies['market_cap'] - target_market_cap
        ) / target_market_cap
        
        # Sort by similarity
        similar_companies = similar_companies.sort_values('similarity', ascending=False)
        
        # Convert to PeerCompany objects
        for _, row in similar_companies.head(max_peers).iterrows():
            peer = PeerCompany(
                ticker=row['ticker'],
                cik=row['cik'],
                name=row.get('name', ''),
                sector=row.get('sector', ''),
                industry=row.get('industry', ''),
                market_cap=row['market_cap'],
                sic_code=row.get('sic_code'),
                similarity_score=row['similarity']
            )
            peers.append(peer)
        
        logger.info(f"Identified {len(peers)} peers by market cap")
        return peers
    
    def identify_peers_by_fundamentals(self,
                                      target_metrics: Dict[str, float],
                                      company_metrics: pd.DataFrame,
                                      weights: Optional[Dict[str, float]] = None,
                                      max_peers: int = 20) -> List[PeerCompany]:
        """
        Identify peers based on fundamental metrics similarity
        
        Args:
            target_metrics: Target company's metrics
            company_metrics: DataFrame with metrics for all companies
            weights: Optional weights for different metrics
            max_peers: Maximum number of peers
            
        Returns:
            List of peer companies
        """
        if company_metrics.empty:
            return []
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'revenue': 0.25,
                'market_cap': 0.25,
                'gross_margin': 0.15,
                'roe': 0.15,
                'debt_to_equity': 0.10,
                'asset_turnover': 0.10
            }
        
        # Normalize metrics for comparison
        normalized_metrics = company_metrics.copy()
        
        for metric in weights.keys():
            if metric in normalized_metrics.columns and metric in target_metrics:
                # Z-score normalization
                mean_val = normalized_metrics[metric].mean()
                std_val = normalized_metrics[metric].std()
                
                if std_val > 0:
                    normalized_metrics[f'{metric}_norm'] = (normalized_metrics[metric] - mean_val) / std_val
                    target_norm = (target_metrics[metric] - mean_val) / std_val
                else:
                    normalized_metrics[f'{metric}_norm'] = 0
                    target_norm = 0
                
                # Calculate distance for this metric
                normalized_metrics[f'{metric}_distance'] = abs(
                    normalized_metrics[f'{metric}_norm'] - target_norm
                )
        
        # Calculate weighted similarity score
        normalized_metrics['similarity_score'] = 0
        
        for metric, weight in weights.items():
            if f'{metric}_distance' in normalized_metrics.columns:
                # Convert distance to similarity (1 = identical, 0 = very different)
                normalized_metrics['similarity_score'] += weight * (
                    1 / (1 + normalized_metrics[f'{metric}_distance'])
                )
        
        # Sort by similarity score
        normalized_metrics = normalized_metrics.sort_values('similarity_score', ascending=False)
        
        # Convert to PeerCompany objects
        peers = []
        for _, row in normalized_metrics.head(max_peers).iterrows():
            peer = PeerCompany(
                ticker=row['ticker'],
                cik=row['cik'],
                name=row.get('name', ''),
                sector=row.get('sector', ''),
                industry=row.get('industry', ''),
                market_cap=row.get('market_cap', 0),
                similarity_score=row['similarity_score']
            )
            peers.append(peer)
        
        logger.info(f"Identified {len(peers)} peers by fundamentals")
        return peers
    
    def benchmark_against_peers(self,
                               company_metrics: Dict[str, float],
                               peer_metrics: pd.DataFrame,
                               metrics_to_compare: Optional[List[str]] = None) -> Dict[str, BenchmarkResult]:
        """
        Benchmark company against peer group
        
        Args:
            company_metrics: Target company's metrics
            peer_metrics: DataFrame with peer metrics
            metrics_to_compare: List of metrics to benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        if peer_metrics.empty:
            return results
        
        # Use all available metrics if not specified
        if metrics_to_compare is None:
            metrics_to_compare = list(company_metrics.keys())
        
        for metric in metrics_to_compare:
            if metric not in company_metrics or metric not in peer_metrics.columns:
                continue
            
            company_value = company_metrics[metric]
            peer_values = peer_metrics[metric].dropna()
            
            if len(peer_values) == 0:
                continue
            
            # Calculate statistics
            peer_median = peer_values.median()
            peer_mean = peer_values.mean()
            peer_std = peer_values.std()
            peer_min = peer_values.min()
            peer_max = peer_values.max()
            peer_q1 = peer_values.quantile(0.25)
            peer_q3 = peer_values.quantile(0.75)
            
            # Calculate percentile rank
            percentile_rank = (peer_values < company_value).mean() * 100
            
            # Calculate z-score
            z_score = (company_value - peer_mean) / peer_std if peer_std > 0 else 0
            
            # Determine relative performance
            if percentile_rank >= 75:
                performance = "Top Quartile"
            elif percentile_rank >= 50:
                performance = "Above Median"
            elif percentile_rank >= 25:
                performance = "Below Median"
            else:
                performance = "Bottom Quartile"
            
            results[metric] = BenchmarkResult(
                metric_name=metric,
                company_value=company_value,
                peer_median=peer_median,
                peer_mean=peer_mean,
                peer_min=peer_min,
                peer_max=peer_max,
                peer_q1=peer_q1,
                peer_q3=peer_q3,
                percentile_rank=percentile_rank,
                z_score=z_score,
                relative_performance=performance,
                peer_count=len(peer_values)
            )
        
        logger.info(f"Benchmarked {len(results)} metrics against {len(peer_metrics)} peers")
        return results
    
    def create_benchmark_visualization(self,
                                     benchmark_results: Dict[str, BenchmarkResult],
                                     category: Optional[str] = None) -> go.Figure:
        """
        Create visualization of benchmark results
        
        Args:
            benchmark_results: Dictionary of benchmark results
            category: Optional category to filter metrics
            
        Returns:
            Plotly figure
        """
        # Filter by category if specified
        if category and category in self.COMPARISON_METRICS:
            metrics_to_plot = [
                m for m in self.COMPARISON_METRICS[category]
                if m in benchmark_results
            ]
        else:
            metrics_to_plot = list(benchmark_results.keys())
        
        if not metrics_to_plot:
            return go.Figure()
        
        # Create box plot visualization
        fig = go.Figure()
        
        for i, metric in enumerate(metrics_to_plot):
            result = benchmark_results[metric]
            
            # Add box plot for peer distribution
            fig.add_trace(go.Box(
                name=metric.replace('_', ' ').title(),
                y=[result.peer_min, result.peer_q1, result.peer_median, 
                   result.peer_q3, result.peer_max],
                boxmean='sd',
                marker_color='lightblue',
                showlegend=False
            ))
            
            # Add company value as scatter point
            fig.add_trace(go.Scatter(
                x=[metric.replace('_', ' ').title()],
                y=[result.company_value],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red' if result.percentile_rank < 50 else 'green',
                    symbol='diamond'
                ),
                name='Company',
                showlegend=i == 0,
                text=f"Percentile: {result.percentile_rank:.1f}",
                hovertemplate='%{text}<br>Value: %{y:.2f}'
            ))
        
        fig.update_layout(
            title="Peer Benchmark Comparison",
            xaxis_title="Metrics",
            yaxis_title="Value",
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_percentile_chart(self,
                               benchmark_results: Dict[str, BenchmarkResult]) -> go.Figure:
        """
        Create percentile ranking chart
        
        Args:
            benchmark_results: Dictionary of benchmark results
            
        Returns:
            Plotly figure
        """
        # Prepare data for chart
        metrics = []
        percentiles = []
        colors = []
        
        for metric, result in benchmark_results.items():
            metrics.append(metric.replace('_', ' ').title())
            percentiles.append(result.percentile_rank)
            
            # Color based on performance
            if result.percentile_rank >= 75:
                colors.append('green')
            elif result.percentile_rank >= 50:
                colors.append('lightgreen')
            elif result.percentile_rank >= 25:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=percentiles,
                y=metrics,
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in percentiles],
                textposition='auto',
            )
        ])
        
        # Add reference lines
        fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                      annotation_text="Median")
        fig.add_vline(x=75, line_dash="dot", line_color="gray", 
                      annotation_text="Top Quartile")
        fig.add_vline(x=25, line_dash="dot", line_color="gray", 
                      annotation_text="Bottom Quartile")
        
        fig.update_layout(
            title="Percentile Ranking vs Peers",
            xaxis_title="Percentile Rank",
            yaxis_title="Metric",
            xaxis=dict(range=[0, 100]),
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_spider_chart(self,
                          company_metrics: Dict[str, float],
                          peer_medians: Dict[str, float],
                          metrics_to_show: Optional[List[str]] = None) -> go.Figure:
        """
        Create spider/radar chart for peer comparison
        
        Args:
            company_metrics: Company's metrics
            peer_medians: Peer median values
            metrics_to_show: Metrics to display
            
        Returns:
            Plotly figure
        """
        if metrics_to_show is None:
            metrics_to_show = list(set(company_metrics.keys()) & set(peer_medians.keys()))
        
        # Normalize values to 0-100 scale
        categories = []
        company_values = []
        peer_values = []
        
        for metric in metrics_to_show:
            if metric in company_metrics and metric in peer_medians:
                categories.append(metric.replace('_', ' ').title())
                
                # Normalize to percentage of peer median
                if peer_medians[metric] != 0:
                    company_norm = (company_metrics[metric] / peer_medians[metric]) * 50
                    company_values.append(min(100, company_norm))  # Cap at 100
                else:
                    company_values.append(50)
                
                peer_values.append(50)  # Peer median always at 50
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=company_values,
            theta=categories,
            fill='toself',
            name='Company',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=peer_values,
            theta=categories,
            fill='toself',
            name='Peer Median',
            line_color='gray',
            fillcolor='lightgray',
            opacity=0.5
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Company vs Peer Median - Spider Chart",
            height=600
        )
        
        return fig
    
    def identify_strengths_weaknesses(self,
                                    benchmark_results: Dict[str, BenchmarkResult],
                                    threshold: float = 25.0) -> Tuple[List[str], List[str]]:
        """
        Identify company strengths and weaknesses vs peers
        
        Args:
            benchmark_results: Benchmark results
            threshold: Percentile threshold for strength/weakness
            
        Returns:
            Tuple of (strengths, weaknesses)
        """
        strengths = []
        weaknesses = []
        
        for metric, result in benchmark_results.items():
            if result.percentile_rank >= (100 - threshold):
                strengths.append({
                    'metric': metric,
                    'percentile': result.percentile_rank,
                    'vs_median': (result.company_value - result.peer_median) / result.peer_median * 100
                })
            elif result.percentile_rank <= threshold:
                weaknesses.append({
                    'metric': metric,
                    'percentile': result.percentile_rank,
                    'vs_median': (result.company_value - result.peer_median) / result.peer_median * 100
                })
        
        # Sort by percentile rank
        strengths.sort(key=lambda x: x['percentile'], reverse=True)
        weaknesses.sort(key=lambda x: x['percentile'])
        
        logger.info(f"Identified {len(strengths)} strengths and {len(weaknesses)} weaknesses")
        
        return strengths, weaknesses
    
    def calculate_composite_score(self,
                                 benchmark_results: Dict[str, BenchmarkResult],
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate composite performance score
        
        Args:
            benchmark_results: Benchmark results
            weights: Optional weights for metrics
            
        Returns:
            Composite score (0-100)
        """
        if not benchmark_results:
            return 50.0
        
        # Default equal weights if not provided
        if weights is None:
            weights = {metric: 1.0 for metric in benchmark_results.keys()}
        
        total_weight = sum(weights.values())
        weighted_sum = 0
        
        for metric, result in benchmark_results.items():
            if metric in weights:
                weighted_sum += result.percentile_rank * weights[metric]
        
        composite_score = weighted_sum / total_weight if total_weight > 0 else 50.0
        
        logger.info(f"Calculated composite score: {composite_score:.1f}")
        return composite_score
    
    def generate_benchmark_report(self,
                                 benchmark_results: Dict[str, BenchmarkResult],
                                 strengths_weaknesses: Optional[Tuple[List, List]] = None) -> pd.DataFrame:
        """
        Generate comprehensive benchmark report
        
        Args:
            benchmark_results: Benchmark results
            strengths_weaknesses: Optional strengths and weaknesses
            
        Returns:
            DataFrame with benchmark report
        """
        report_data = []
        
        for metric, result in benchmark_results.items():
            report_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Company Value': f"{result.company_value:.2f}",
                'Peer Median': f"{result.peer_median:.2f}",
                'Percentile Rank': f"{result.percentile_rank:.1f}%",
                'Z-Score': f"{result.z_score:.2f}",
                'Performance': result.relative_performance,
                'vs Median %': f"{((result.company_value - result.peer_median) / result.peer_median * 100):.1f}%" if result.peer_median != 0 else "N/A",
                'Peer Count': result.peer_count
            })
        
        df = pd.DataFrame(report_data)
        
        # Sort by percentile rank
        df = df.sort_values('Percentile Rank', ascending=False)
        
        return df