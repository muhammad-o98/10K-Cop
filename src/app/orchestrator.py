"""
ML Orchestrator & Insights Visualization Module
Main orchestrator for all ML models with advanced visualization for quant insights
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Import our ML modules (in production, these would be separate files)
# from ml_forecaster import AdvancedForecaster, ForecastResult, ModelConfig
# from risk_model import QuantitativeRiskModel, RiskMetrics, StressScenario
# from sentiment_analyzer import FinancialSentimentAnalyzer, DocumentAnalysis
# from anomaly_factor_module import AnomalyDetector, FactorAnalyzer, AlphaSignal

logger = logging.getLogger(__name__)


@dataclass
class QuantInsights:
    """Container for comprehensive quantitative insights"""
    
    # Core metrics
    company_ticker: str
    analysis_date: datetime
    
    # Forecasting results
    revenue_forecast: Dict[str, Any]
    earnings_forecast: Dict[str, Any]
    best_forecast_model: str
    forecast_confidence: float
    
    # Risk metrics
    risk_metrics: Dict[str, float]
    var_95: float
    expected_shortfall: float
    risk_adjusted_returns: float
    stress_test_results: Dict[str, float]
    
    # Sentiment analysis
    overall_sentiment: float
    sentiment_breakdown: Dict[str, float]
    key_topics: List[str]
    risk_mentions: int
    
    # Anomaly detection
    anomalies_detected: List[Dict]
    anomaly_severity: str
    
    # Factor analysis
    factor_exposures: Dict[str, float]
    alpha_signals: List[Dict]
    expected_alpha: float
    
    # Peer comparison
    peer_percentile: Dict[str, float]
    relative_performance: str
    
    # Investment recommendation
    investment_signal: str  # BUY, HOLD, SELL
    confidence_score: float
    key_drivers: List[str]
    risk_warnings: List[str]


class MLOrchestrator:
    """
    Main orchestrator for all ML models and quantitative analysis
    """
    
    def __init__(self,
                 enable_parallel: bool = True,
                 cache_results: bool = True,
                 config: Optional[Dict] = None):
        """
        Initialize ML orchestrator
        
        Args:
            enable_parallel: Enable parallel processing
            cache_results: Cache model results
            config: Configuration dictionary
        """
        self.enable_parallel = enable_parallel
        self.cache_results = cache_results
        self.config = config or {}
        
        # Initialize models
        self._initialize_models()
        
        # Results cache
        self.cache = {}
        
        logger.info("MLOrchestrator initialized")
    
    def _initialize_models(self):
        """Initialize all ML models"""
        
        # Forecasting
        from ml_forecaster import AdvancedForecaster, ModelConfig
        self.forecaster = AdvancedForecaster(
            config=ModelConfig(
                forecast_horizon=4,
                enable_ensemble=True,
                use_external_regressors=True
            )
        )
        
        # Risk modeling
        from risk_model import QuantitativeRiskModel
        self.risk_model = QuantitativeRiskModel(
            confidence_levels=[0.95, 0.99],
            use_cornish_fisher=True
        )
        
        # Sentiment analysis
        from sentiment_analyzer import FinancialSentimentAnalyzer
        self.sentiment_analyzer = FinancialSentimentAnalyzer(
            use_finbert=True,
            use_topic_modeling=True
        )
        
        # Anomaly detection
        from anomaly_factor_module import AnomalyDetector, FactorAnalyzer
        self.anomaly_detector = AnomalyDetector(
            methods=['isolation_forest', 'statistical'],
            contamination=0.05
        )
        
        self.factor_analyzer = FactorAnalyzer(
            n_factors=5,
            factor_method='pca'
        )
    
    def analyze_company(self,
                       ticker: str,
                       financial_data: pd.DataFrame,
                       price_data: pd.DataFrame,
                       text_data: Dict[str, str],
                       macro_data: Optional[pd.DataFrame] = None,
                       peer_data: Optional[pd.DataFrame] = None) -> QuantInsights:
        """
        Comprehensive company analysis with all ML models
        
        Args:
            ticker: Company ticker
            financial_data: Financial metrics
            price_data: Stock price history
            text_data: Text from 10-K sections
            macro_data: Macroeconomic data
            peer_data: Peer company data
            
        Returns:
            QuantInsights object with all results
        """
        logger.info(f"Starting comprehensive analysis for {ticker}")
        
        # Parallel execution if enabled
        if self.enable_parallel:
            results = self._parallel_analysis(
                ticker, financial_data, price_data, text_data, macro_data, peer_data
            )
        else:
            results = self._sequential_analysis(
                ticker, financial_data, price_data, text_data, macro_data, peer_data
            )
        
        # Generate investment signal
        investment_signal, confidence = self._generate_investment_signal(results)
        
        # Identify key drivers and risks
        key_drivers = self._identify_key_drivers(results)
        risk_warnings = self._identify_risk_warnings(results)
        
        # Create insights object
        insights = QuantInsights(
            company_ticker=ticker,
            analysis_date=datetime.now(),
            
            # Forecasting
            revenue_forecast=results['forecast']['revenue'],
            earnings_forecast=results['forecast']['earnings'],
            best_forecast_model=results['forecast']['best_model'],
            forecast_confidence=results['forecast']['confidence'],
            
            # Risk
            risk_metrics=results['risk']['metrics'],
            var_95=results['risk']['var_95'],
            expected_shortfall=results['risk']['expected_shortfall'],
            risk_adjusted_returns=results['risk']['sharpe_ratio'],
            stress_test_results=results['risk']['stress_tests'],
            
            # Sentiment
            overall_sentiment=results['sentiment']['overall'],
            sentiment_breakdown=results['sentiment']['breakdown'],
            key_topics=results['sentiment']['topics'],
            risk_mentions=results['sentiment']['risk_mentions'],
            
            # Anomalies
            anomalies_detected=results['anomalies']['detected'],
            anomaly_severity=results['anomalies']['severity'],
            
            # Factors
            factor_exposures=results['factors']['exposures'],
            alpha_signals=results['factors']['signals'],
            expected_alpha=results['factors']['expected_alpha'],
            
            # Peer comparison
            peer_percentile=results.get('peer', {}).get('percentiles', {}),
            relative_performance=results.get('peer', {}).get('performance', 'average'),
            
            # Investment recommendation
            investment_signal=investment_signal,
            confidence_score=confidence,
            key_drivers=key_drivers,
            risk_warnings=risk_warnings
        )
        
        # Cache results
        if self.cache_results:
            self.cache[ticker] = insights
        
        logger.info(f"Analysis complete for {ticker}")
        
        return insights
    
    def _parallel_analysis(self, ticker, financial_data, price_data, 
                          text_data, macro_data, peer_data) -> Dict:
        """Run analysis in parallel"""
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            forecast_future = executor.submit(
                self._run_forecasting, financial_data, price_data, macro_data
            )
            risk_future = executor.submit(
                self._run_risk_analysis, price_data, financial_data
            )
            sentiment_future = executor.submit(
                self._run_sentiment_analysis, text_data
            )
            anomaly_future = executor.submit(
                self._run_anomaly_detection, financial_data, price_data
            )
            factor_future = executor.submit(
                self._run_factor_analysis, price_data, financial_data
            )
            
            # Collect results
            results = {
                'forecast': forecast_future.result(),
                'risk': risk_future.result(),
                'sentiment': sentiment_future.result(),
                'anomalies': anomaly_future.result(),
                'factors': factor_future.result()
            }
            
            # Peer analysis if data available
            if peer_data is not None:
                peer_future = executor.submit(self._run_peer_analysis, financial_data, peer_data)
                results['peer'] = peer_future.result()
        
        return results
    
    def _sequential_analysis(self, ticker, financial_data, price_data,
                           text_data, macro_data, peer_data) -> Dict:
        """Run analysis sequentially"""
        
        results = {
            'forecast': self._run_forecasting(financial_data, price_data, macro_data),
            'risk': self._run_risk_analysis(price_data, financial_data),
            'sentiment': self._run_sentiment_analysis(text_data),
            'anomalies': self._run_anomaly_detection(financial_data, price_data),
            'factors': self._run_factor_analysis(price_data, financial_data)
        }
        
        if peer_data is not None:
            results['peer'] = self._run_peer_analysis(financial_data, peer_data)
        
        return results
    
    def _run_forecasting(self, financial_data, price_data, macro_data) -> Dict:
        """Run forecasting models"""
        
        # Prepare data
        prepared_data = self.forecaster.prepare_data(
            financial_data, price_data, macro_data
        )
        
        # Fit all models
        forecast_results = self.forecaster.fit_all_models(prepared_data)
        
        # Get best model
        best_model = self.forecaster.get_best_model()
        
        # Extract revenue and earnings forecasts
        revenue_forecast = {}
        earnings_forecast = {}
        
        if 'revenue' in prepared_data.columns:
            revenue_forecast = {
                'values': forecast_results[best_model].forecast,
                'lower_bound': forecast_results[best_model].lower_bound,
                'upper_bound': forecast_results[best_model].upper_bound,
                'dates': pd.date_range(
                    start=prepared_data.index[-1] + pd.Timedelta(days=90),
                    periods=len(forecast_results[best_model].forecast),
                    freq='Q'
                )
            }
        
        return {
            'revenue': revenue_forecast,
            'earnings': earnings_forecast,
            'best_model': best_model,
            'confidence': forecast_results[best_model].confidence,
            'all_models': {k: v.metrics for k, v in forecast_results.items()}
        }
    
    def _run_risk_analysis(self, price_data, financial_data) -> Dict:
        """Run risk analysis"""
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # Calculate risk metrics
        risk_metrics = self.risk_model.calculate_all_risk_metrics(returns)
        
        # Run stress tests
        stress_scenarios = [
            {'name': 'Market Crash', 'shock': -0.30},
            {'name': 'Interest Rate Shock', 'shock': -0.15},
            {'name': 'Credit Crisis', 'shock': -0.20}
        ]
        
        stress_results = {}
        for scenario in stress_scenarios:
            stressed_returns = returns + scenario['shock']
            stressed_var = self.risk_model.calculate_var(stressed_returns, 0.99)
            stress_results[scenario['name']] = stressed_var
        
        return {
            'metrics': {
                'volatility': returns.std() * np.sqrt(252),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'max_drawdown': risk_metrics.max_drawdown
            },
            'var_95': risk_metrics.var_95,
            'var_99': risk_metrics.var_99,
            'expected_shortfall': risk_metrics.expected_shortfall,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'stress_tests': stress_results
        }
    
    def _run_sentiment_analysis(self, text_data) -> Dict:
        """Run sentiment analysis"""
        
        # Analyze document
        full_text = ' '.join(text_data.values())
        analysis = self.sentiment_analyzer.analyze_document(
            full_text,
            sections=text_data
        )
        
        # Extract key metrics
        breakdown = {
            section: result.sentiment_score
            for section, result in analysis.sections.items()
        }
        
        topics = [topic for topic, _ in analysis.key_topics[:5]]
        
        return {
            'overall': analysis.overall_sentiment,
            'breakdown': breakdown,
            'topics': topics,
            'risk_mentions': sum(r.risk_mentions for r in analysis.sections.values()),
            'volatility': analysis.sentiment_volatility,
            'trend': analysis.sentiment_trend
        }
    
    def _run_anomaly_detection(self, financial_data, price_data) -> Dict:
        """Run anomaly detection"""
        
        # Combine data for anomaly detection
        combined_data = pd.concat([
            financial_data,
            price_data[['close', 'volume']].resample('Q').agg({'close': 'last', 'volume': 'sum'})
        ], axis=1)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(combined_data)
        
        # Process results
        detected = [
            {
                'date': a.timestamp,
                'type': a.anomaly_type,
                'score': a.anomaly_score,
                'metrics': a.affected_metrics
            }
            for a in anomalies[:10]  # Top 10 anomalies
        ]
        
        # Determine overall severity
        if any(a.severity == 'critical' for a in anomalies):
            severity = 'critical'
        elif any(a.severity == 'high' for a in anomalies):
            severity = 'high'
        elif any(a.severity == 'medium' for a in anomalies):
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'detected': detected,
            'severity': severity,
            'count': len(anomalies)
        }
    
    def _run_factor_analysis(self, price_data, financial_data) -> Dict:
        """Run factor analysis"""
        
        # Calculate returns
        returns = price_data[['close']].pct_change().dropna()
        
        # Extract factors
        factor_model = self.factor_analyzer.extract_factors(returns)
        
        # Generate alpha signals
        alpha_signals = self.factor_analyzer.generate_alpha_signals(
            factor_model, returns
        )
        
        # Process results
        exposures = {}
        for i, name in factor_model.factor_names.items():
            exposures[name] = factor_model.loadings.iloc[0, i]
        
        signals = [
            {
                'name': s.signal_name,
                'value': s.signal_value,
                'expected_return': s.expected_return,
                'entry': s.entry_signal
            }
            for s in alpha_signals[:5]
        ]
        
        expected_alpha = np.mean([s.expected_return for s in alpha_signals])
        
        return {
            'exposures': exposures,
            'signals': signals,
            'expected_alpha': expected_alpha,
            'variance_explained': factor_model.total_variance_explained
        }
    
    def _run_peer_analysis(self, financial_data, peer_data) -> Dict:
        """Run peer comparison analysis"""
        
        # Calculate key metrics
        metrics = ['revenue_growth', 'profit_margin', 'roe', 'debt_to_equity']
        
        percentiles = {}
        for metric in metrics:
            if metric in financial_data.columns:
                company_value = financial_data[metric].iloc[-1]
                peer_values = peer_data[metric] if metric in peer_data.columns else []
                
                if len(peer_values) > 0:
                    percentile = (peer_values < company_value).mean() * 100
                    percentiles[metric] = percentile
        
        # Determine relative performance
        avg_percentile = np.mean(list(percentiles.values()))
        
        if avg_percentile > 75:
            performance = 'outperforming'
        elif avg_percentile > 25:
            performance = 'average'
        else:
            performance = 'underperforming'
        
        return {
            'percentiles': percentiles,
            'performance': performance,
            'avg_percentile': avg_percentile
        }
    
    def _generate_investment_signal(self, results: Dict) -> Tuple[str, float]:
        """Generate investment recommendation"""
        
        # Score components
        scores = []
        weights = []
        
        # Forecast score (30% weight)
        forecast_confidence = results['forecast']['confidence']
        scores.append(forecast_confidence)
        weights.append(0.30)
        
        # Risk score (25% weight)
        sharpe = results['risk']['sharpe_ratio']
        risk_score = min(1.0, max(0, (sharpe + 2) / 4))  # Normalize to 0-1
        scores.append(risk_score)
        weights.append(0.25)
        
        # Sentiment score (20% weight)
        sentiment = results['sentiment']['overall']
        sentiment_score = (sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
        scores.append(sentiment_score)
        weights.append(0.20)
        
        # Anomaly score (15% weight)
        anomaly_severity = results['anomalies']['severity']
        anomaly_score = {
            'low': 0.8,
            'medium': 0.5,
            'high': 0.2,
            'critical': 0.0
        }.get(anomaly_severity, 0.5)
        scores.append(anomaly_score)
        weights.append(0.15)
        
        # Factor score (10% weight)
        expected_alpha = results['factors']['expected_alpha']
        factor_score = min(1.0, max(0, expected_alpha + 0.5))
        scores.append(factor_score)
        weights.append(0.10)
        
        # Calculate weighted score
        total_score = np.average(scores, weights=weights)
        
        # Generate signal
        if total_score > 0.65:
            signal = 'BUY'
        elif total_score > 0.35:
            signal = 'HOLD'
        else:
            signal = 'SELL'
        
        # Confidence is based on score distance from thresholds
        if signal == 'HOLD':
            confidence = 1 - 2 * abs(total_score - 0.5)
        else:
            confidence = abs(total_score - 0.5) * 2
        
        return signal, confidence
    
    def _identify_key_drivers(self, results: Dict) -> List[str]:
        """Identify key drivers of performance"""
        
        drivers = []
        
        # Check forecast
        if results['forecast']['confidence'] > 0.7:
            drivers.append(f"Strong {results['forecast']['best_model']} forecast confidence")
        
        # Check risk metrics
        if results['risk']['sharpe_ratio'] > 1.5:
            drivers.append("Excellent risk-adjusted returns")
        
        # Check sentiment
        if results['sentiment']['overall'] > 0.5:
            drivers.append("Positive sentiment in financial disclosures")
        
        # Check factors
        if results['factors']['expected_alpha'] > 0.1:
            drivers.append(f"High alpha potential ({results['factors']['expected_alpha']:.1%})")
        
        # Check peer performance
        if 'peer' in results and results['peer']['avg_percentile'] > 75:
            drivers.append("Outperforming industry peers")
        
        return drivers[:5]  # Top 5 drivers
    
    def _identify_risk_warnings(self, results: Dict) -> List[str]:
        """Identify risk warnings"""
        
        warnings = []
        
        # Check VaR
        if results['risk']['var_95'] > 0.1:
            warnings.append(f"High Value at Risk: {results['risk']['var_95']:.1%}")
        
        # Check anomalies
        if results['anomalies']['severity'] in ['high', 'critical']:
            warnings.append(f"{results['anomalies']['severity'].title()} severity anomalies detected")
        
        # Check sentiment volatility
        if results['sentiment']['volatility'] > 0.3:
            warnings.append("High sentiment volatility across sections")
        
        # Check risk mentions
        if results['sentiment']['risk_mentions'] > 50:
            warnings.append(f"Elevated risk mentions ({results['sentiment']['risk_mentions']})")
        
        # Check stress tests
        worst_stress = min(results['risk']['stress_tests'].values())
        if worst_stress > 0.2:
            warnings.append(f"Vulnerable to stress scenarios (worst: {worst_stress:.1%})")
        
        return warnings[:5]  # Top 5 warnings


class QuantInsightsVisualizer:
    """
    Advanced visualization for quantitative insights
    """
    
    @staticmethod
    def create_comprehensive_dashboard(insights: QuantInsights) -> go.Figure:
        """Create comprehensive dashboard with all insights"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Revenue Forecast', 'Risk Metrics', 'Sentiment Analysis',
                'Factor Exposures', 'Anomaly Detection', 'Alpha Signals',
                'Peer Comparison', 'Investment Signal', 'Key Metrics'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Revenue Forecast
        if insights.revenue_forecast:
            fig.add_trace(
                go.Scatter(
                    x=insights.revenue_forecast['dates'],
                    y=insights.revenue_forecast['values'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=insights.revenue_forecast['dates'],
                    y=insights.revenue_forecast['upper_bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=insights.revenue_forecast['dates'],
                    y=insights.revenue_forecast['lower_bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Confidence Band'
                ),
                row=1, col=1
            )
        
        # 2. Risk Metrics
        risk_names = list(insights.risk_metrics.keys())[:5]
        risk_values = [insights.risk_metrics[k] for k in risk_names]
        
        fig.add_trace(
            go.Bar(
                x=risk_names,
                y=risk_values,
                marker_color='red',
                name='Risk Metrics'
            ),
            row=1, col=2
        )
        
        # 3. Sentiment Analysis
        sent_sections = list(insights.sentiment_breakdown.keys())
        sent_values = list(insights.sentiment_breakdown.values())
        colors = ['green' if v > 0 else 'red' for v in sent_values]
        
        fig.add_trace(
            go.Bar(
                x=sent_sections,
                y=sent_values,
                marker_color=colors,
                name='Sentiment'
            ),
            row=1, col=3
        )
        
        # 4. Factor Exposures
        factor_names = list(insights.factor_exposures.keys())
        factor_values = list(insights.factor_exposures.values())
        
        fig.add_trace(
            go.Bar(
                x=factor_names,
                y=factor_values,
                marker_color='purple',
                name='Factors'
            ),
            row=2, col=1
        )
        
        # 5. Anomaly Detection Timeline
        if insights.anomalies_detected:
            anomaly_dates = [a['date'] for a in insights.anomalies_detected[:20]]
            anomaly_scores = [a['score'] for a in insights.anomalies_detected[:20]]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_scores,
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    name='Anomalies'
                ),
                row=2, col=2
            )
        
        # 6. Alpha Signals
        if insights.alpha_signals:
            signal_names = [s['name'] for s in insights.alpha_signals]
            signal_returns = [s['expected_return'] for s in insights.alpha_signals]
            
            fig.add_trace(
                go.Bar(
                    x=signal_names,
                    y=signal_returns,
                    marker_color='cyan',
                    name='Alpha'
                ),
                row=2, col=3
            )
        
        # 7. Peer Comparison
        if insights.peer_percentile:
            peer_metrics = list(insights.peer_percentile.keys())
            peer_values = list(insights.peer_percentile.values())
            
            fig.add_trace(
                go.Bar(
                    x=peer_metrics,
                    y=peer_values,
                    marker_color='teal',
                    name='Percentile'
                ),
                row=3, col=1
            )
        
        # 8. Investment Signal Gauge
        signal_value = {'BUY': 75, 'HOLD': 50, 'SELL': 25}[insights.investment_signal]
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=signal_value,
                title={'text': insights.investment_signal},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': signal_value
                    }
                }
            ),
            row=3, col=2
        )
        
        # 9. Key Metrics Table
        table_data = {
            'Metric': [
                'VaR (95%)',
                'Sharpe Ratio',
                'Expected Alpha',
                'Sentiment Score',
                'Confidence'
            ],
            'Value': [
                f"{insights.var_95:.2%}",
                f"{insights.risk_adjusted_returns:.2f}",
                f"{insights.expected_alpha:.2%}",
                f"{insights.overall_sentiment:.3f}",
                f"{insights.confidence_score:.1%}"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(table_data.keys())),
                cells=dict(values=list(table_data.values()))
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Quantitative Analysis Dashboard - {insights.company_ticker}",
            showlegend=False,
            height=1200,
            template="plotly_dark"
        )
        
        return fig
    
    @staticmethod
    def create_risk_heatmap(risk_metrics: RiskMetrics) -> go.Figure:
        """Create risk metrics heatmap"""
        
        # Prepare data for heatmap
        metrics = [
            ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%'],
            ['Sharpe', 'Sortino', 'Calmar', 'Info Ratio'],
            ['Max DD', 'Volatility', 'Skew', 'Kurtosis']
        ]
        
        values = [
            [risk_metrics.var_95, risk_metrics.var_99, 
             risk_metrics.cvar_95, risk_metrics.cvar_99],
            [risk_metrics.sharpe_ratio, risk_metrics.sortino_ratio,
             risk_metrics.calmar_ratio, risk_metrics.information_ratio],
            [risk_metrics.max_drawdown, 0.2, 0.1, 3.0]  # Placeholder values
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            text=metrics,
            texttemplate="%{text}",
            colorscale='RdYlGn_r',
            showscale=True
        ))
        
        fig.update_layout(
            title="Risk Metrics Heatmap",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_factor_analysis_plot(factor_model) -> go.Figure:
        """Create factor analysis visualization"""
        
        # Create scree plot for variance explained
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Variance Explained', 'Factor Loadings'),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}]]
        )
        
        # Variance explained
        fig.add_trace(
            go.Bar(
                x=[f'Factor {i+1}' for i in range(len(factor_model.variance_explained))],
                y=factor_model.variance_explained,
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Factor loadings heatmap
        fig.add_trace(
            go.Heatmap(
                z=factor_model.loadings.values,
                x=factor_model.loadings.columns,
                y=factor_model.loadings.index,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Factor Analysis Results",
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_forecast_comparison(forecast_results: Dict) -> go.Figure:
        """Create forecast model comparison"""
        
        models = list(forecast_results.keys())
        metrics = ['rmse', 'mape', 'r2']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [forecast_results[model].metrics.get(metric, 0) for model in models]
            
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=models,
                y=values
            ))
        
        fig.update_layout(
            title="Forecast Model Comparison",
            xaxis_title="Model",
            yaxis_title="Metric Value",
            barmode='group',
            height=400
        )
        
        return fig


# Streamlit App Integration
def create_ml_insights_app():
    """Create Streamlit app for ML insights"""
    
    st.set_page_config(
        page_title="10K Cop - Quant Insights",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🎯 10K Cop - Quantitative Analysis Platform")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        ticker = st.text_input("Company Ticker", "AAPL")
        
        analysis_options = st.multiselect(
            "Analysis Components",
            ["Forecasting", "Risk Analysis", "Sentiment", "Anomaly Detection", "Factor Analysis"],
            default=["Forecasting", "Risk Analysis", "Sentiment"]
        )
        
        forecast_horizon = st.slider("Forecast Horizon (Quarters)", 1, 8, 4)
        
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        if st.button("Run Analysis", type="primary"):
            run_analysis = True
        else:
            run_analysis = False
    
    # Main content
    if run_analysis:
        with st.spinner("Running comprehensive analysis..."):
            # Initialize orchestrator
            orchestrator = MLOrchestrator()
            
            # Load data (placeholder - would load real data)
            financial_data = pd.DataFrame()  # Load financial data
            price_data = pd.DataFrame()  # Load price data
            text_data = {}  # Load 10-K text
            
            # Run analysis
            insights = orchestrator.analyze_company(
                ticker, financial_data, price_data, text_data
            )
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Investment Signal",
                    insights.investment_signal,
                    f"{insights.confidence_score:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "Expected Alpha",
                    f"{insights.expected_alpha:.2%}",
                    "Annualized"
                )
            
            with col3:
                st.metric(
                    "Risk (VaR 95%)",
                    f"{insights.var_95:.2%}",
                    "Daily"
                )
            
            with col4:
                st.metric(
                    "Sentiment Score",
                    f"{insights.overall_sentiment:.3f}",
                    "Overall"
                )
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Dashboard", "📈 Forecasts", "⚠️ Risk", "💭 Sentiment", "🎯 Signals"
            ])
            
            with tab1:
                st.subheader("Comprehensive Analysis Dashboard")
                visualizer = QuantInsightsVisualizer()
                fig = visualizer.create_comprehensive_dashboard(insights)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Forecast Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Revenue Forecast**")
                    # Display forecast chart
                
                with col2:
                    st.write("**Model Performance**")
                    # Display model comparison
            
            with tab3:
                st.subheader("Risk Analysis")
                
                # Risk metrics table
                risk_df = pd.DataFrame({
                    'Metric': ['VaR 95%', 'CVaR 95%', 'Max Drawdown', 'Sharpe Ratio'],
                    'Value': [
                        f"{insights.var_95:.2%}",
                        f"{insights.expected_shortfall:.2%}",
                        "15.3%",  # Placeholder
                        f"{insights.risk_adjusted_returns:.2f}"
                    ]
                })
                st.dataframe(risk_df, use_container_width=True)
                
                # Stress test results
                st.write("**Stress Test Results**")
                stress_df = pd.DataFrame(insights.stress_test_results.items(),
                                        columns=['Scenario', 'Impact'])
                st.dataframe(stress_df, use_container_width=True)
            
            with tab4:
                st.subheader("Sentiment Analysis")
                
                # Sentiment breakdown
                st.write("**Section Sentiment**")
                for section, score in insights.sentiment_breakdown.items():
                    color = "green" if score > 0 else "red"
                    st.markdown(f"- **{section}**: :{color}[{score:.3f}]")
                
                # Key topics
                st.write("**Key Topics**")
                st.write(", ".join(insights.key_topics))
            
            with tab5:
                st.subheader("Trading Signals")
                
                # Alpha signals
                if insights.alpha_signals:
                    signals_df = pd.DataFrame(insights.alpha_signals)
                    st.dataframe(signals_df, use_container_width=True)
                
                # Key drivers and warnings
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Drivers**")
                    for driver in insights.key_drivers:
                        st.write(f"✅ {driver}")
                
                with col2:
                    st.write("**Risk Warnings**")
                    for warning in insights.risk_warnings:
                        st.write(f"⚠️ {warning}")


if __name__ == "__main__":
    # Run the Streamlit app
    create_ml_insights_app()