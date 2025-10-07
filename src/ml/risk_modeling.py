"""
Quantitative Risk Modeling Module
Advanced risk metrics, VaR/CVaR calculations, and stress testing
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t, genpareto, genextreme
import matplotlib.pyplot as plt
import seaborn as sns

# Risk modeling libraries
from arch import arch_model
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap
import copulas
from copulas.multivariate import GaussianMultivariate, VineCopula
import riskfolio as rp

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics results"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    max_drawdown: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    beta: float
    alpha: float
    tracking_error: float
    downside_deviation: float
    upside_potential: float
    omega_ratio: float
    tail_ratio: float
    var_breach_prob: float
    stress_test_results: Dict[str, float]
    risk_decomposition: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class StressScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    shocks: Dict[str, float]  # Factor -> shock magnitude
    probability: float = 0.01
    correlation_adjustment: Optional[np.ndarray] = None


class QuantitativeRiskModel:
    """
    Advanced risk modeling system for financial analysis
    """
    
    def __init__(self,
                 confidence_levels: List[float] = [0.95, 0.99],
                 risk_free_rate: float = 0.03,
                 lookback_window: int = 252,
                 use_cornish_fisher: bool = True):
        """
        Initialize risk model
        
        Args:
            confidence_levels: VaR/CVaR confidence levels
            risk_free_rate: Risk-free rate for Sharpe ratio
            lookback_window: Historical window for risk calculations
            use_cornish_fisher: Use Cornish-Fisher expansion for non-normal distributions
        """
        self.confidence_levels = confidence_levels
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.use_cornish_fisher = use_cornish_fisher
        
        # Risk models
        self.garch_model = None
        self.copula_model = None
        self.evt_model = None
        
        logger.info("QuantitativeRiskModel initialized")
    
    def calculate_all_risk_metrics(self,
                                  returns: pd.DataFrame,
                                  benchmark: Optional[pd.Series] = None,
                                  factors: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Returns data (can be single series or portfolio)
            benchmark: Benchmark returns for relative metrics
            factors: Factor returns for attribution
            
        Returns:
            RiskMetrics object with all calculations
        """
        # Convert to Series if single column DataFrame
        if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
            returns = returns.iloc[:, 0]
        
        # Basic risk metrics
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        es = self.calculate_expected_shortfall(returns)
        
        # Drawdown metrics
        max_dd = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(returns, max_dd)
        
        # Risk-adjusted returns
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        
        # Relative metrics if benchmark provided
        if benchmark is not None:
            info_ratio = self.calculate_information_ratio(returns, benchmark)
            tracking_err = self.calculate_tracking_error(returns, benchmark)
            beta, alpha = self.calculate_beta_alpha(returns, benchmark)
        else:
            info_ratio = tracking_err = beta = alpha = np.nan
        
        # Advanced metrics
        downside_dev = self.calculate_downside_deviation(returns)
        upside_pot = self.calculate_upside_potential(returns)
        omega = self.calculate_omega_ratio(returns)
        tail_ratio = self.calculate_tail_ratio(returns)
        
        # VaR breach probability
        var_breach = self.calculate_var_breach_probability(returns, var_95)
        
        # Stress testing
        stress_results = self.run_stress_tests(returns)
        
        # Risk decomposition if factors provided
        risk_decomp = None
        if factors is not None:
            risk_decomp = self.factor_risk_decomposition(returns, factors)
        
        # Confidence intervals
        ci = self.calculate_confidence_intervals(returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=es,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            information_ratio=info_ratio,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_err,
            downside_deviation=downside_dev,
            upside_potential=upside_pot,
            omega_ratio=omega,
            tail_ratio=tail_ratio,
            var_breach_prob=var_breach,
            stress_test_results=stress_results,
            risk_decomposition=risk_decomp,
            confidence_intervals=ci
        )
    
    def calculate_var(self, returns: Union[pd.Series, np.ndarray], 
                     confidence: float = 0.95,
                     method: str = 'parametric') -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Return series
            confidence: Confidence level
            method: 'parametric', 'historical', 'cornish_fisher', 'evt'
            
        Returns:
            VaR value
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns[~np.isnan(returns)]
        
        if method == 'parametric':
            # Assume normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = -norm.ppf(1 - confidence, mu, sigma)
            
        elif method == 'historical':
            # Historical simulation
            var = -np.percentile(returns, (1 - confidence) * 100)
            
        elif method == 'cornish_fisher' and self.use_cornish_fisher:
            # Cornish-Fisher expansion for non-normal distributions
            var = self._cornish_fisher_var(returns, confidence)
            
        elif method == 'evt':
            # Extreme Value Theory
            var = self._evt_var(returns, confidence)
            
        else:
            # Default to historical
            var = -np.percentile(returns, (1 - confidence) * 100)
        
        return var
    
    def calculate_cvar(self, returns: Union[pd.Series, np.ndarray],
                      confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Return series
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns[~np.isnan(returns)]
        
        var = self.calculate_var(returns, confidence)
        # CVaR is the average of returns below VaR
        cvar = -np.mean(returns[returns <= -var])
        
        return cvar if not np.isnan(cvar) else var
    
    def calculate_expected_shortfall(self, returns: Union[pd.Series, np.ndarray],
                                    confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (same as CVaR)"""
        return self.calculate_cvar(returns, confidence)
    
    def _cornish_fisher_var(self, returns: np.ndarray, confidence: float) -> float:
        """
        Calculate VaR using Cornish-Fisher expansion
        Adjusts for skewness and kurtosis
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6 + \
               (z**3 - 3*z) * (kurt - 3) / 24 - \
               (2*z**3 - 5*z) * skew**2 / 36
        
        var = -mu - sigma * z_cf
        
        return var
    
    def _evt_var(self, returns: np.ndarray, confidence: float) -> float:
        """
        Calculate VaR using Extreme Value Theory (Peak Over Threshold)
        """
        # Set threshold at 95th percentile
        threshold = np.percentile(returns, 5)
        
        # Get exceedances
        exceedances = returns[returns < threshold] - threshold
        
        if len(exceedances) < 10:
            # Not enough data for EVT
            return self.calculate_var(returns, confidence, method='historical')
        
        # Fit Generalized Pareto Distribution
        params = genpareto.fit(-exceedances)
        
        # Calculate VaR
        n = len(returns)
        nu = len(exceedances)
        
        if params[0] != 0:  # xi != 0
            var = threshold - params[1]/params[0] * \
                  ((n/nu * (1-confidence))**(-params[0]) - 1)
        else:  # xi == 0 (exponential case)
            var = threshold - params[1] * np.log(n/nu * (1-confidence))
        
        return -var
    
    def calculate_garch_var(self, returns: pd.Series, 
                           confidence: float = 0.95,
                           horizon: int = 1) -> float:
        """
        Calculate VaR using GARCH model for volatility forecasting
        
        Args:
            returns: Return series
            confidence: Confidence level
            horizon: Forecast horizon
            
        Returns:
            GARCH-based VaR
        """
        # Scale returns to percentage
        returns_pct = returns * 100
        
        # Fit GARCH(1,1) model
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='t')
        res = model.fit(disp='off')
        
        # Forecast volatility
        forecasts = res.forecast(horizon=horizon)
        variance_forecast = forecasts.variance.values[-1, :]
        
        # Calculate VaR
        mu = returns_pct.mean()
        
        # Use Student's t-distribution
        nu = res.params['nu'] if 'nu' in res.params else 10
        var = -mu/100 - np.sqrt(variance_forecast[-1])/100 * t.ppf(1-confidence, nu) * np.sqrt(nu/(nu-2))
        
        self.garch_model = res
        
        return var
    
    def calculate_max_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown (positive value)
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        return -drawdown.min()
    
    def calculate_calmar_ratio(self, returns: Union[pd.Series, np.ndarray],
                              max_drawdown: Optional[float] = None) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Max Drawdown)
        """
        if max_drawdown is None:
            max_drawdown = self.calculate_max_drawdown(returns)
        
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        annual_return = returns.mean() * 252  # Annualize
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else -np.inf
        
        return annual_return / max_drawdown
    
    def calculate_sharpe_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Sharpe Ratio
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        excess_returns = returns - self.risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self, returns: Union[pd.Series, np.ndarray],
                               target_return: float = 0) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation)
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.sqrt(np.mean(downside_returns**2))
        
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_information_ratio(self, returns: pd.Series, 
                                   benchmark: pd.Series) -> float:
        """
        Calculate Information Ratio
        """
        active_returns = returns - benchmark
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return np.sqrt(252) * active_returns.mean() / tracking_error
    
    def calculate_tracking_error(self, returns: pd.Series,
                                benchmark: pd.Series) -> float:
        """Calculate Tracking Error"""
        return np.sqrt(252) * (returns - benchmark).std()
    
    def calculate_beta_alpha(self, returns: pd.Series,
                           benchmark: pd.Series) -> Tuple[float, float]:
        """
        Calculate Beta and Alpha relative to benchmark
        """
        # Align series
        aligned = pd.DataFrame({'returns': returns, 'benchmark': benchmark}).dropna()
        
        if len(aligned) < 2:
            return np.nan, np.nan
        
        # Calculate beta
        covariance = aligned.cov().iloc[0, 1]
        benchmark_var = aligned['benchmark'].var()
        
        if benchmark_var == 0:
            beta = 0
        else:
            beta = covariance / benchmark_var
        
        # Calculate alpha (annualized)
        alpha = (aligned['returns'].mean() - beta * aligned['benchmark'].mean()) * 252
        
        return beta, alpha
    
    def calculate_downside_deviation(self, returns: Union[pd.Series, np.ndarray],
                                    target: float = 0) -> float:
        """Calculate Downside Deviation"""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        downside_returns = returns[returns < target] - target
        
        if len(downside_returns) == 0:
            return 0
        
        return np.sqrt(252) * np.sqrt(np.mean(downside_returns**2))
    
    def calculate_upside_potential(self, returns: Union[pd.Series, np.ndarray],
                                  target: float = 0) -> float:
        """Calculate Upside Potential Ratio"""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        upside_returns = returns[returns > target] - target
        downside_returns = returns[returns < target] - target
        
        if len(downside_returns) == 0:
            return np.inf
        
        upside_potential = np.mean(upside_returns) if len(upside_returns) > 0 else 0
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        if downside_deviation == 0:
            return np.inf if upside_potential > 0 else 0
        
        return upside_potential / downside_deviation
    
    def calculate_omega_ratio(self, returns: Union[pd.Series, np.ndarray],
                            threshold: float = 0) -> float:
        """
        Calculate Omega Ratio
        Probability-weighted ratio of gains to losses
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf if gains.sum() > 0 else 0
        
        return gains.sum() / losses.sum()
    
    def calculate_tail_ratio(self, returns: Union[pd.Series, np.ndarray],
                           percentile: float = 5) -> float:
        """
        Calculate Tail Ratio
        Ratio of right tail to left tail
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        right_tail = returns.quantile(1 - percentile/100)
        left_tail = returns.quantile(percentile/100)
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 0
        
        return abs(right_tail / left_tail)
    
    def calculate_var_breach_probability(self, returns: Union[pd.Series, np.ndarray],
                                        var_threshold: float) -> float:
        """
        Calculate probability of VaR breach
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        breaches = returns < -var_threshold
        return breaches.mean()
    
    def run_stress_tests(self, returns: pd.Series,
                        scenarios: Optional[List[StressScenario]] = None) -> Dict[str, float]:
        """
        Run stress test scenarios
        
        Args:
            returns: Historical returns
            scenarios: List of stress scenarios
            
        Returns:
            Dictionary of scenario results
        """
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        results = {}
        
        for scenario in scenarios:
            # Apply shocks
            stressed_returns = returns.copy()
            
            # Simple shock application (can be made more sophisticated)
            shock_magnitude = sum(scenario.shocks.values()) / len(scenario.shocks)
            stressed_returns = stressed_returns + shock_magnitude
            
            # Calculate stressed metrics
            stressed_var = self.calculate_var(stressed_returns, 0.99)
            stressed_loss = -stressed_returns.mean() * len(stressed_returns)
            
            results[scenario.name] = {
                'var_99': stressed_var,
                'expected_loss': stressed_loss,
                'probability': scenario.probability
            }
        
        return results
    
    def _get_default_stress_scenarios(self) -> List[StressScenario]:
        """Get default stress test scenarios"""
        return [
            StressScenario(
                name="Market Crash",
                description="Severe market downturn similar to 2008",
                shocks={'equity': -0.40, 'credit': -0.30, 'rates': 0.02},
                probability=0.02
            ),
            StressScenario(
                name="Interest Rate Shock",
                description="Rapid interest rate increase",
                shocks={'rates': 0.03, 'equity': -0.15, 'credit': -0.10},
                probability=0.05
            ),
            StressScenario(
                name="Credit Crisis",
                description="Credit spread widening",
                shocks={'credit': -0.25, 'equity': -0.20, 'rates': -0.01},
                probability=0.03
            ),
            StressScenario(
                name="Inflation Surge",
                description="Unexpected inflation spike",
                shocks={'rates': 0.02, 'equity': -0.10, 'commodities': 0.20},
                probability=0.10
            )
        ]
    
    def monte_carlo_var(self, returns: pd.Series,
                       n_simulations: int = 10000,
                       horizon: int = 10,
                       confidence: float = 0.95) -> Tuple[float, np.ndarray]:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            returns: Historical returns
            n_simulations: Number of Monte Carlo paths
            horizon: Time horizon for VaR
            confidence: Confidence level
            
        Returns:
            VaR value and simulated paths
        """
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random paths
        dt = 1 / 252  # Daily steps
        paths = np.zeros((n_simulations, horizon))
        
        for i in range(n_simulations):
            shocks = np.random.normal(0, 1, horizon)
            paths[i] = mu * dt + sigma * np.sqrt(dt) * shocks
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + paths, axis=1) - 1
        terminal_returns = cum_returns[:, -1]
        
        # Calculate VaR
        var = -np.percentile(terminal_returns, (1 - confidence) * 100)
        
        return var, paths
    
    def copula_var(self, returns_matrix: pd.DataFrame,
                  confidence: float = 0.95,
                  copula_type: str = 'gaussian') -> Dict[str, float]:
        """
        Calculate portfolio VaR using copulas for dependency modeling
        
        Args:
            returns_matrix: DataFrame with asset returns
            confidence: Confidence level
            copula_type: Type of copula ('gaussian', 'vine', 't')
            
        Returns:
            Dictionary with VaR estimates
        """
        # Fit marginal distributions
        marginals = {}
        uniform_data = pd.DataFrame(index=returns_matrix.index)
        
        for col in returns_matrix.columns:
            # Fit best distribution to each marginal
            returns = returns_matrix[col].dropna()
            
            # Transform to uniform using empirical CDF
            ecdf = stats.rankdata(returns) / (len(returns) + 1)
            uniform_data[col] = ecdf
            
            # Store marginal info
            marginals[col] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skew': stats.skew(returns),
                'kurt': stats.kurtosis(returns)
            }
        
        # Fit copula
        if copula_type == 'gaussian':
            copula = GaussianMultivariate()
        elif copula_type == 'vine':
            copula = VineCopula('center')
        else:
            copula = GaussianMultivariate()  # Default
        
        copula.fit(uniform_data)
        
        # Simulate from copula
        n_sim = 10000
        simulated = copula.sample(n_sim)
        
        # Transform back to returns using inverse marginal CDFs
        simulated_returns = pd.DataFrame()
        
        for col in returns_matrix.columns:
            # Use inverse normal CDF with marginal parameters
            uniform_samples = simulated[col]
            mu = marginals[col]['mean']
            sigma = marginals[col]['std']
            
            simulated_returns[col] = norm.ppf(uniform_samples, mu, sigma)
        
        # Calculate portfolio returns (equal weighted for simplicity)
        weights = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)
        portfolio_returns = simulated_returns @ weights
        
        # Calculate VaR
        var = -portfolio_returns.quantile(1 - confidence)
        
        # Also calculate component VaR
        component_var = {}
        for col in returns_matrix.columns:
            component_var[col] = -simulated_returns[col].quantile(1 - confidence)
        
        return {
            'portfolio_var': var,
            'component_var': component_var,
            'copula_type': copula_type
        }
    
    def factor_risk_decomposition(self, returns: pd.Series,
                                factors: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose risk into factor contributions
        
        Args:
            returns: Asset returns
            factors: Factor returns (columns are different factors)
            
        Returns:
            Dictionary with risk decomposition
        """
        # Align data
        data = pd.DataFrame({'returns': returns})
        data = data.join(factors, how='inner')
        
        # Run factor regression
        X = data[factors.columns]
        y = data['returns']
        
        # Add constant for intercept
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        
        # OLS regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Get factor loadings
        loadings = model.coef_
        
        # Calculate factor contributions to variance
        factor_cov = X.cov()
        
        total_variance = y.var()
        factor_contributions = {}
        
        for i, factor in enumerate(X.columns):
            # Contribution = loading^2 * factor_variance
            contribution = loadings[i]**2 * factor_cov.iloc[i, i]
            factor_contributions[factor] = contribution / total_variance
        
        # Idiosyncratic risk
        fitted_values = model.predict(X)
        residuals = y - fitted_values
        idio_risk = residuals.var() / total_variance
        
        factor_contributions['idiosyncratic'] = idio_risk
        
        return factor_contributions
    
    def calculate_risk_parity_weights(self, returns_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate Risk Parity portfolio weights
        Each asset contributes equally to portfolio risk
        
        Args:
            returns_matrix: DataFrame with asset returns
            
        Returns:
            Array of portfolio weights
        """
        cov_matrix = returns_matrix.cov().values
        n_assets = len(returns_matrix.columns)
        
        # Objective: Equal risk contribution
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)
            marginal_contrib = cov_matrix @ weights.T
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal contribution
            target = portfolio_vol / n_assets
            
            return np.sum((contrib - target)**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds (long only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_contribution, x0, 
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        return result.x
    
    def calculate_kelly_fraction(self, returns: pd.Series,
                                confidence: float = 0.25) -> float:
        """
        Calculate Kelly Criterion for position sizing
        
        Args:
            returns: Historical returns
            confidence: Confidence factor (fractional Kelly)
            
        Returns:
            Optimal betting fraction
        """
        mu = returns.mean()
        sigma2 = returns.var()
        
        if sigma2 == 0:
            return 0
        
        # Kelly fraction = μ / σ²
        kelly = mu / sigma2
        
        # Apply confidence factor (fractional Kelly is more conservative)
        return kelly * confidence
    
    def calculate_confidence_intervals(self, returns: pd.Series) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for risk metrics using bootstrap
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with confidence intervals
        """
        # Bootstrap parameters
        n_bootstrap = 1000
        
        # Storage for bootstrap samples
        var_samples = []
        sharpe_samples = []
        sortino_samples = []
        
        # Stationary bootstrap for time series
        bs = StationaryBootstrap(10, returns)
        
        for data in bs.bootstrap(n_bootstrap):
            sample = data[0][0]
            
            # Calculate metrics for bootstrap sample
            var_samples.append(self.calculate_var(sample, 0.95))
            sharpe_samples.append(self.calculate_sharpe_ratio(sample))
            sortino_samples.append(self.calculate_sortino_ratio(sample))
        
        # Calculate confidence intervals (95%)
        ci = {
            'var_95': (np.percentile(var_samples, 2.5), np.percentile(var_samples, 97.5)),
            'sharpe': (np.percentile(sharpe_samples, 2.5), np.percentile(sharpe_samples, 97.5)),
            'sortino': (np.percentile(sortino_samples, 2.5), np.percentile(sortino_samples, 97.5))
        }
        
        return ci
    
    def create_risk_report(self, metrics: RiskMetrics) -> pd.DataFrame:
        """
        Create comprehensive risk report
        
        Args:
            metrics: RiskMetrics object
            
        Returns:
            DataFrame with formatted risk report
        """
        report_data = {
            'Metric': [],
            'Value': [],
            'Category': []
        }
        
        # VaR metrics
        report_data['Metric'].extend(['VaR (95%)', 'VaR (99%)', 'CVaR (95%)', 'CVaR (99%)'])
        report_data['Value'].extend([f'{metrics.var_95:.2%}', f'{metrics.var_99:.2%}',
                                    f'{metrics.cvar_95:.2%}', f'{metrics.cvar_99:.2%}'])
        report_data['Category'].extend(['Risk'] * 4)
        
        # Performance metrics
        report_data['Metric'].extend(['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'])
        report_data['Value'].extend([f'{metrics.sharpe_ratio:.2f}', 
                                    f'{metrics.sortino_ratio:.2f}',
                                    f'{metrics.calmar_ratio:.2f}'])
        report_data['Category'].extend(['Performance'] * 3)
        
        # Drawdown
        report_data['Metric'].append('Max Drawdown')
        report_data['Value'].append(f'{metrics.max_drawdown:.2%}')
        report_data['Category'].append('Drawdown')
        
        # Advanced metrics
        report_data['Metric'].extend(['Omega Ratio', 'Tail Ratio'])
        report_data['Value'].extend([f'{metrics.omega_ratio:.2f}', 
                                    f'{metrics.tail_ratio:.2f}'])
        report_data['Category'].extend(['Advanced'] * 2)
        
        return pd.DataFrame(report_data)