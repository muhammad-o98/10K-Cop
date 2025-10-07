"""
Backtesting & Portfolio Optimization Module
Production-ready backtesting framework and advanced portfolio optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, shgo
from scipy.stats import norm, t
import cvxpy as cp

# Portfolio optimization
from pypfopt import (
    EfficientFrontier, 
    BlackLittermanModel,
    HRPOpt,
    CLA,
    risk_models,
    expected_returns,
    objective_functions
)
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Backtesting
import vectorbt as vbt
import backtrader as bt
import empyrical as ep

# Machine Learning
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    
    # Returns
    total_return: float
    annualized_return: float
    cumulative_returns: pd.Series
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    number_of_trades: int
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    
    # Efficiency metrics
    calmar_ratio: float
    information_ratio: float
    alpha: float
    beta: float
    
    # Time metrics
    avg_holding_period: float
    time_in_market: float
    
    # Costs
    total_commission: float
    total_slippage: float
    
    # Trade log
    trades: pd.DataFrame
    equity_curve: pd.Series
    
    # Statistics
    monthly_returns: pd.Series
    annual_returns: pd.Series
    rolling_sharpe: pd.Series


@dataclass
class PortfolioResult:
    """Container for portfolio optimization results"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    
    # Risk decomposition
    marginal_risk_contributions: Dict[str, float]
    component_risk_contributions: Dict[str, float]
    
    # Efficient frontier
    efficient_frontier: Optional[pd.DataFrame] = None
    
    # Discrete allocation
    allocation: Optional[Dict[str, int]] = None
    leftover_cash: Optional[float] = None
    
    # Performance attribution
    factor_exposures: Optional[Dict[str, float]] = None
    
    # Constraints satisfaction
    constraints_satisfied: bool = True
    optimization_status: str = "optimal"


class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to positions"""
        pass


class MomentumStrategy(Strategy):
    """Momentum trading strategy"""
    
    def __init__(self, lookback: int = 20, holding_period: int = 5):
        self.lookback = lookback
        self.holding_period = holding_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[momentum > momentum.quantile(0.7)] = 1
        signals[momentum < momentum.quantile(0.3)] = -1
        
        return signals
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to positions with holding period"""
        positions = signals.copy()
        
        # Hold positions for specified period
        for i in range(len(positions)):
            if positions.iloc[i] != 0:
                end_idx = min(i + self.holding_period, len(positions))
                positions.iloc[i:end_idx] = positions.iloc[i]
        
        return positions


class MeanReversionStrategy(Strategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        price = data['close']
        
        # Calculate z-score
        rolling_mean = price.rolling(self.window).mean()
        rolling_std = price.rolling(self.window).std()
        z_score = (price - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[z_score > self.z_threshold] = -1  # Sell when overbought
        signals[z_score < -self.z_threshold] = 1  # Buy when oversold
        
        return signals
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to positions"""
        return signals


class MLStrategy(Strategy):
    """Machine learning based strategy"""
    
    def __init__(self, model, features: List[str], threshold: float = 0.6):
        self.model = model
        self.features = features
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ML-based signals"""
        # Prepare features
        X = data[self.features].fillna(0)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
            signals = pd.Series(0, index=data.index)
            signals[probabilities > self.threshold] = 1
            signals[probabilities < (1 - self.threshold)] = -1
        else:
            predictions = self.model.predict(X)
            signals = pd.Series(predictions, index=data.index)
        
        return signals
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to positions"""
        return signals


class Backtester:
    """
    Comprehensive backtesting framework
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        logger.info("Backtester initialized")
    
    def backtest_strategy(self,
                         strategy: Strategy,
                         data: pd.DataFrame,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> BacktestResult:
        """
        Backtest a trading strategy
        
        Args:
            strategy: Trading strategy object
            data: Price data with OHLCV
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult object
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Get positions
        positions = strategy.get_positions(signals)
        
        # Calculate returns
        returns = data['close'].pct_change()
        strategy_returns = positions.shift(1) * returns
        
        # Apply transaction costs
        trades = positions.diff().abs()
        costs = trades * (self.commission + self.slippage)
        strategy_returns = strategy_returns - costs
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        
        # Calculate metrics
        result = self._calculate_metrics(
            strategy_returns,
            positions,
            trades,
            equity_curve,
            strategy.__class__.__name__
        )
        
        return result
    
    def _calculate_metrics(self,
                          returns: pd.Series,
                          positions: pd.Series,
                          trades: pd.Series,
                          equity_curve: pd.Series,
                          strategy_name: str) -> BacktestResult:
        """Calculate backtest metrics"""
        
        # Clean returns
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return self._empty_result(strategy_name)
        
        # Returns metrics
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        annualized_return = ep.annual_return(returns_clean, period='daily')
        cumulative_returns = (1 + returns_clean).cumprod() - 1
        
        # Risk metrics
        sharpe = ep.sharpe_ratio(returns_clean, risk_free=self.risk_free_rate, period='daily')
        sortino = ep.sortino_ratio(returns_clean, required_return=self.risk_free_rate, period='daily')
        max_dd = ep.max_drawdown(returns_clean)
        var_95 = np.percentile(returns_clean, 5)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        
        # Trading metrics
        trade_returns = returns_clean[trades > 0]
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0
        
        profit_factor = (
            winning_trades.sum() / abs(losing_trades.sum())
            if len(losing_trades) > 0 and losing_trades.sum() != 0
            else np.inf if len(winning_trades) > 0 else 0
        )
        
        # Additional metrics
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0
        
        # Time in market
        time_in_market = (positions != 0).mean()
        
        # Trade log
        trade_log = self._create_trade_log(positions, returns_clean, equity_curve)
        
        # Rolling metrics
        rolling_sharpe = returns_clean.rolling(252).apply(
            lambda x: ep.sharpe_ratio(x, period='daily')
        )
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_returns=cumulative_returns,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            number_of_trades=len(trade_returns),
            avg_trade_return=trade_returns.mean() if len(trade_returns) > 0 else 0,
            avg_winning_trade=winning_trades.mean() if len(winning_trades) > 0 else 0,
            avg_losing_trade=losing_trades.mean() if len(losing_trades) > 0 else 0,
            calmar_ratio=calmar,
            information_ratio=0,  # Placeholder
            alpha=0,  # Placeholder
            beta=1,  # Placeholder
            avg_holding_period=self._calculate_avg_holding_period(positions),
            time_in_market=time_in_market,
            total_commission=trades.sum() * self.commission * self.initial_capital,
            total_slippage=trades.sum() * self.slippage * self.initial_capital,
            trades=trade_log,
            equity_curve=equity_curve,
            monthly_returns=returns_clean.resample('M').apply(lambda x: (1+x).prod()-1),
            annual_returns=returns_clean.resample('Y').apply(lambda x: (1+x).prod()-1),
            rolling_sharpe=rolling_sharpe
        )
    
    def _create_trade_log(self, positions: pd.Series, returns: pd.Series, 
                         equity: pd.Series) -> pd.DataFrame:
        """Create detailed trade log"""
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        
        for date, pos in positions.items():
            if pos != current_position:
                if current_position != 0:
                    # Close position
                    exit_price = equity.loc[date]
                    trade_return = (exit_price - entry_price) / entry_price
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': current_position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'holding_period': (date - entry_date).days
                    })
                
                if pos != 0:
                    # Open new position
                    entry_price = equity.loc[date]
                    entry_date = date
                
                current_position = pos
        
        return pd.DataFrame(trades)
    
    def _calculate_avg_holding_period(self, positions: pd.Series) -> float:
        """Calculate average holding period"""
        holding_periods = []
        current_holding = 0
        
        for pos in positions:
            if pos != 0:
                current_holding += 1
            elif current_holding > 0:
                holding_periods.append(current_holding)
                current_holding = 0
        
        return np.mean(holding_periods) if holding_periods else 0
    
    def _empty_result(self, strategy_name: str) -> BacktestResult:
        """Return empty result for failed backtest"""
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=0,
            annualized_return=0,
            cumulative_returns=pd.Series(),
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            win_rate=0,
            profit_factor=0,
            number_of_trades=0,
            avg_trade_return=0,
            avg_winning_trade=0,
            avg_losing_trade=0,
            calmar_ratio=0,
            information_ratio=0,
            alpha=0,
            beta=0,
            avg_holding_period=0,
            time_in_market=0,
            total_commission=0,
            total_slippage=0,
            trades=pd.DataFrame(),
            equity_curve=pd.Series(),
            monthly_returns=pd.Series(),
            annual_returns=pd.Series(),
            rolling_sharpe=pd.Series()
        )
    
    def walk_forward_analysis(self,
                            strategy: Strategy,
                            data: pd.DataFrame,
                            train_periods: int = 252,
                            test_periods: int = 63,
                            step_size: int = 21) -> List[BacktestResult]:
        """
        Walk-forward analysis for robust strategy testing
        
        Args:
            strategy: Trading strategy
            data: Price data
            train_periods: Training window size
            test_periods: Testing window size
            step_size: Step size for rolling window
            
        Returns:
            List of backtest results for each window
        """
        results = []
        
        for i in range(train_periods, len(data) - test_periods, step_size):
            # Training data
            train_data = data.iloc[i-train_periods:i]
            
            # Test data
            test_data = data.iloc[i:i+test_periods]
            
            # Train strategy if it has a fit method
            if hasattr(strategy, 'fit'):
                strategy.fit(train_data)
            
            # Backtest on test data
            result = self.backtest_strategy(strategy, test_data)
            results.append(result)
        
        return results
    
    def monte_carlo_simulation(self,
                             returns: pd.Series,
                             n_simulations: int = 1000,
                             n_days: int = 252) -> Dict[str, Any]:
        """
        Monte Carlo simulation for strategy robustness
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            n_days: Number of days to simulate
            
        Returns:
            Simulation results
        """
        # Calculate return statistics
        mu = returns.mean()
        sigma = returns.std()
        
        # Run simulations
        simulated_returns = np.random.normal(mu, sigma, (n_simulations, n_days))
        
        # Calculate final values
        final_values = (1 + simulated_returns).prod(axis=1)
        
        # Calculate statistics
        results = {
            'mean_return': final_values.mean() - 1,
            'median_return': np.median(final_values) - 1,
            'std_return': final_values.std(),
            'var_95': np.percentile(final_values - 1, 5),
            'cvar_95': (final_values[final_values <= np.percentile(final_values, 5)] - 1).mean(),
            'prob_profit': (final_values > 1).mean(),
            'prob_loss_10pct': (final_values < 0.9).mean(),
            'prob_loss_20pct': (final_values < 0.8).mean(),
            'best_case': final_values.max() - 1,
            'worst_case': final_values.min() - 1,
            'paths': simulated_returns[:100]  # Store first 100 paths for visualization
        }
        
        return results


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple methods
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 frequency: int = 252):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Risk-free rate
            frequency: Return frequency (252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        logger.info("PortfolioOptimizer initialized")
    
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         method: str = 'max_sharpe',
                         constraints: Optional[Dict] = None,
                         target_return: Optional[float] = None,
                         target_risk: Optional[float] = None) -> PortfolioResult:
        """
        Optimize portfolio weights
        
        Args:
            returns: Asset returns DataFrame
            method: Optimization method
            constraints: Portfolio constraints
            target_return: Target return for efficient frontier
            target_risk: Target risk for efficient frontier
            
        Returns:
            PortfolioResult object
        """
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(returns, frequency=self.frequency)
        S = risk_models.sample_cov(returns, frequency=self.frequency)
        
        # Apply shrinkage to covariance matrix
        S = self._shrink_covariance(returns)
        
        if method == 'max_sharpe':
            weights = self._max_sharpe(mu, S, constraints)
        elif method == 'min_volatility':
            weights = self._min_volatility(S, constraints)
        elif method == 'efficient_return':
            weights = self._efficient_return(mu, S, target_return, constraints)
        elif method == 'efficient_risk':
            weights = self._efficient_risk(mu, S, target_risk, constraints)
        elif method == 'risk_parity':
            weights = self._risk_parity(S)
        elif method == 'hrp':
            weights = self._hierarchical_risk_parity(returns)
        elif method == 'black_litterman':
            weights = self._black_litterman(returns, constraints)
        elif method == 'robust':
            weights = self._robust_optimization(mu, S, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(S, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Risk contributions
        marginal_contrib, component_contrib = self._calculate_risk_contributions(weights, S)
        
        # Create efficient frontier
        efficient_frontier_df = self._create_efficient_frontier(mu, S)
        
        return PortfolioResult(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            marginal_risk_contributions=dict(zip(returns.columns, marginal_contrib)),
            component_risk_contributions=dict(zip(returns.columns, component_contrib)),
            efficient_frontier=efficient_frontier_df
        )
    
    def _shrink_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to covariance matrix"""
        lw = LedoitWolf()
        shrunk_cov, _ = lw.fit(returns).covariance_, lw.shrinkage_
        return shrunk_cov * self.frequency
    
    def _max_sharpe(self, mu: np.ndarray, S: np.ndarray, 
                   constraints: Optional[Dict]) -> np.ndarray:
        """Maximize Sharpe ratio"""
        ef = EfficientFrontier(mu, S)
        
        # Add constraints
        if constraints:
            if 'weight_bounds' in constraints:
                ef.add_constraint(lambda w: w >= constraints['weight_bounds'][0])
                ef.add_constraint(lambda w: w <= constraints['weight_bounds'][1])
            if 'sector_limits' in constraints:
                for sector, limit in constraints['sector_limits'].items():
                    # Add sector constraints
                    pass
        
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        return ef.clean_weights()
    
    def _min_volatility(self, S: np.ndarray, constraints: Optional[Dict]) -> np.ndarray:
        """Minimize portfolio volatility"""
        ef = EfficientFrontier(None, S)
        weights = ef.min_volatility()
        return ef.clean_weights()
    
    def _efficient_return(self, mu: np.ndarray, S: np.ndarray,
                         target_return: float, constraints: Optional[Dict]) -> np.ndarray:
        """Efficient portfolio for target return"""
        ef = EfficientFrontier(mu, S)
        weights = ef.efficient_return(target_return)
        return ef.clean_weights()
    
    def _efficient_risk(self, mu: np.ndarray, S: np.ndarray,
                       target_risk: float, constraints: Optional[Dict]) -> np.ndarray:
        """Efficient portfolio for target risk"""
        ef = EfficientFrontier(mu, S)
        weights = ef.efficient_risk(target_risk)
        return ef.clean_weights()
    
    def _risk_parity(self, S: np.ndarray) -> np.ndarray:
        """Risk parity portfolio"""
        n_assets = S.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(weights @ S @ weights)
            marginal_contrib = S @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Equal risk contribution
            target = portfolio_vol / n_assets
            
            return np.sum((contrib - target)**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_budget_objective, x0,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        return result.x
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity (HRP) portfolio"""
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        return np.array(list(weights.values()))
    
    def _black_litterman(self, returns: pd.DataFrame,
                        constraints: Optional[Dict],
                        views: Optional[Dict] = None) -> np.ndarray:
        """Black-Litterman portfolio optimization"""
        S = risk_models.sample_cov(returns, frequency=self.frequency)
        
        # Market equilibrium weights (market cap weighted)
        market_weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Calculate equilibrium returns
        delta = 2.5  # Risk aversion coefficient
        equilibrium_returns = delta * S @ market_weights
        
        if views:
            # Incorporate views
            bl = BlackLittermanModel(
                S, 
                pi=equilibrium_returns,
                Q=views['returns'],
                P=views['picking_matrix'],
                omega=views['confidence']
            )
            mu_bl = bl.bl_returns()
            S_bl = bl.bl_cov()
        else:
            mu_bl = equilibrium_returns
            S_bl = S
        
        # Optimize with Black-Litterman estimates
        ef = EfficientFrontier(mu_bl, S_bl)
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        return ef.clean_weights()
    
    def _robust_optimization(self, mu: np.ndarray, S: np.ndarray,
                           constraints: Optional[Dict]) -> np.ndarray:
        """Robust portfolio optimization with uncertainty"""
        n_assets = len(mu)
        
        # Define uncertainty sets
        kappa_mu = 0.1  # Uncertainty in expected returns
        kappa_sigma = 0.05  # Uncertainty in covariance
        
        # CVX optimization
        w = cp.Variable(n_assets)
        
        # Robust expected return (worst-case)
        robust_return = mu @ w - kappa_mu * cp.norm(w, 1)
        
        # Robust risk (worst-case)
        robust_risk = cp.quad_form(w, S) + kappa_sigma * cp.norm(w, 2)**2
        
        # Objective: maximize worst-case Sharpe ratio
        objective = cp.Maximize((robust_return - self.risk_free_rate) / cp.sqrt(robust_risk))
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        # Solve
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        return w.value if w.value is not None else np.ones(n_assets) / n_assets
    
    def _calculate_risk_contributions(self, weights: np.ndarray, 
                                     S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate marginal and component risk contributions"""
        portfolio_vol = np.sqrt(weights @ S @ weights)
        
        # Marginal contributions
        marginal_contrib = S @ weights / portfolio_vol
        
        # Component contributions
        component_contrib = weights * marginal_contrib
        
        return marginal_contrib, component_contrib
    
    def _create_efficient_frontier(self, mu: np.ndarray, S: np.ndarray,
                                  n_points: int = 100) -> pd.DataFrame:
        """Create efficient frontier"""
        ef = EfficientFrontier(mu, S)
        
        # Generate efficient frontier
        returns = []
        risks = []
        sharpes = []
        
        # Get min and max returns
        min_ret = mu.min()
        max_ret = mu.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        for target in target_returns:
            try:
                ef_copy = EfficientFrontier(mu, S)
                weights = ef_copy.efficient_return(target)
                
                ret = ef_copy.portfolio_performance()[0]
                risk = ef_copy.portfolio_performance()[1]
                sharpe = ef_copy.portfolio_performance()[2]
                
                returns.append(ret)
                risks.append(risk)
                sharpes.append(sharpe)
            except:
                continue
        
        return pd.DataFrame({
            'return': returns,
            'risk': risks,
            'sharpe': sharpes
        })
    
    def portfolio_rebalancing(self,
                            current_weights: Dict[str, float],
                            target_weights: Dict[str, float],
                            prices: Dict[str, float],
                            total_value: float,
                            threshold: float = 0.05) -> Dict[str, int]:
        """
        Calculate rebalancing trades
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current asset prices
            total_value: Total portfolio value
            threshold: Rebalancing threshold
            
        Returns:
            Dictionary of trades (positive = buy, negative = sell)
        """
        trades = {}
        
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            
            # Check if rebalancing needed
            if abs(target - current) > threshold:
                # Calculate shares to trade
                value_diff = (target - current) * total_value
                shares = int(value_diff / prices[asset])
                
                if shares != 0:
                    trades[asset] = shares
        
        return trades
    
    def calculate_discrete_allocation(self,
                                    weights: Dict[str, float],
                                    prices: pd.Series,
                                    total_portfolio_value: float) -> Tuple[Dict[str, int], float]:
        """
        Convert continuous weights to discrete share allocation
        
        Args:
            weights: Portfolio weights
            prices: Latest prices
            total_portfolio_value: Total value to invest
            
        Returns:
            Share allocation and leftover cash
        """
        da = DiscreteAllocation(weights, prices, total_portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        
        return allocation, leftover


class PerformanceAnalyzer:
    """Analyze and visualize portfolio/strategy performance"""
    
    @staticmethod
    def create_performance_report(backtest_result: BacktestResult) -> Dict[str, Any]:
        """Create comprehensive performance report"""
        
        report = {
            'summary': {
                'Total Return': f"{backtest_result.total_return:.2%}",
                'Annual Return': f"{backtest_result.annualized_return:.2%}",
                'Sharpe Ratio': f"{backtest_result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{backtest_result.max_drawdown:.2%}",
                'Win Rate': f"{backtest_result.win_rate:.2%}",
                'Profit Factor': f"{backtest_result.profit_factor:.2f}"
            },
            'risk_metrics': {
                'Sortino Ratio': backtest_result.sortino_ratio,
                'Calmar Ratio': backtest_result.calmar_ratio,
                'VaR (95%)': backtest_result.var_95,
                'CVaR (95%)': backtest_result.cvar_95
            },
            'trading_stats': {
                'Number of Trades': backtest_result.number_of_trades,
                'Avg Trade Return': f"{backtest_result.avg_trade_return:.2%}",
                'Avg Win': f"{backtest_result.avg_winning_trade:.2%}",
                'Avg Loss': f"{backtest_result.avg_losing_trade:.2%}",
                'Time in Market': f"{backtest_result.time_in_market:.1%}"
            },
            'costs': {
                'Total Commission': f"${backtest_result.total_commission:,.2f}",
                'Total Slippage': f"${backtest_result.total_slippage:,.2f}"
            }
        }
        
        return report
    
    @staticmethod
    def plot_backtest_results(result: BacktestResult) -> go.Figure:
        """Create interactive backtest visualization"""
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Equity Curve', 'Returns Distribution',
                'Rolling Sharpe', 'Drawdown',
                'Monthly Returns', 'Trade Analysis',
                'Return vs Risk', 'Trade Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'box'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12
        )
        
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=result.cumulative_returns.dropna(),
                nbinsx=50,
                name='Returns',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=result.rolling_sharpe.index,
                y=result.rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # 4. Drawdown
        drawdown = (result.equity_curve / result.equity_curve.expanding().max() - 1)
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # 5. Monthly Returns Heatmap (simplified as bar chart)
        fig.add_trace(
            go.Bar(
                x=result.monthly_returns.index,
                y=result.monthly_returns.values,
                name='Monthly Returns',
                marker_color=['green' if r > 0 else 'red' for r in result.monthly_returns]
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results - {result.strategy_name}",
            showlegend=False,
            height=1200,
            template="plotly_white"
        )
        
        return fig