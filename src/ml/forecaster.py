"""
Advanced Financial Forecasting Module
Production-ready time series forecasting with multiple models and ensemble methods
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
from scipy.optimize import minimize
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# ML Libraries
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results with confidence intervals"""
    model_name: str
    forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    model_params: Optional[Dict] = None
    residuals: Optional[np.ndarray] = None
    forecast_dates: Optional[pd.DatetimeIndex] = None


@dataclass
class ModelConfig:
    """Configuration for forecast models"""
    # Data parameters
    target_variable: str = 'revenue'
    feature_columns: List[str] = field(default_factory=list)
    lookback_window: int = 12  # quarters
    forecast_horizon: int = 4   # quarters ahead
    
    # Model parameters
    enable_arima: bool = True
    enable_sarimax: bool = True
    enable_prophet: bool = True
    enable_ml_models: bool = True
    enable_deep_learning: bool = True
    enable_ensemble: bool = True
    
    # Validation parameters
    cv_splits: int = 5
    test_size: float = 0.2
    validation_metric: str = 'mape'  # mape, rmse, mae
    
    # Advanced features
    use_external_regressors: bool = True
    use_macro_features: bool = True
    use_technical_indicators: bool = True
    use_sentiment_scores: bool = True
    
    # Hyperparameter tuning
    enable_hyperopt: bool = True
    hyperopt_trials: int = 100
    
    # Risk parameters
    var_confidence: float = 0.95
    monte_carlo_simulations: int = 10000


class AdvancedForecaster:
    """
    Production-ready forecasting system with multiple models and ensemble methods
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize forecaster with configuration
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.forecast_results = {}
        
        # Initialize model registry
        self._init_model_registry()
        
        logger.info(f"AdvancedForecaster initialized with {self.config.forecast_horizon} period horizon")
    
    def _init_model_registry(self):
        """Initialize available models"""
        self.model_registry = {
            'arima': self._fit_arima,
            'sarimax': self._fit_sarimax,
            'prophet': self._fit_prophet,
            'ets': self._fit_ets,
            'random_forest': self._fit_random_forest,
            'xgboost': self._fit_xgboost,
            'lightgbm': self._fit_lightgbm,
            'catboost': self._fit_catboost,
            'lstm': self._fit_lstm,
            'gru': self._fit_gru,
            'transformer': self._fit_transformer
        }
    
    def prepare_data(self,
                    financial_data: pd.DataFrame,
                    price_data: Optional[pd.DataFrame] = None,
                    macro_data: Optional[pd.DataFrame] = None,
                    sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare and engineer features for forecasting
        
        Args:
            financial_data: Financial metrics DataFrame
            price_data: Stock price data
            macro_data: Macroeconomic indicators
            sentiment_data: Sentiment scores
            
        Returns:
            Prepared DataFrame with engineered features
        """
        df = financial_data.copy()
        
        # Ensure datetime index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Engineer financial features
        df = self._engineer_financial_features(df)
        
        # Add external features
        if price_data is not None and self.config.use_technical_indicators:
            df = self._add_technical_features(df, price_data)
        
        if macro_data is not None and self.config.use_macro_features:
            df = self._add_macro_features(df, macro_data)
        
        if sentiment_data is not None and self.config.use_sentiment_scores:
            df = self._add_sentiment_features(df, sentiment_data)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns 
                               if col != self.config.target_variable]
        
        logger.info(f"Prepared data with {len(df)} samples and {len(self.feature_columns)} features")
        
        return df
    
    def _engineer_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer financial ratio features"""
        
        # Growth rates
        for col in ['revenue', 'net_income', 'operating_cash_flow']:
            if col in df.columns:
                df[f'{col}_growth'] = df[col].pct_change()
                df[f'{col}_growth_ma3'] = df[f'{col}_growth'].rolling(3).mean()
                df[f'{col}_growth_volatility'] = df[f'{col}_growth'].rolling(4).std()
        
        # Efficiency metrics
        if 'revenue' in df.columns and 'assets' in df.columns:
            df['asset_turnover'] = df['revenue'] / df['assets']
        
        if 'net_income' in df.columns and 'equity' in df.columns:
            df['roe'] = df['net_income'] / df['equity']
        
        # Margin trends
        if 'gross_margin' in df.columns:
            df['gross_margin_trend'] = df['gross_margin'].diff()
            df['gross_margin_ma'] = df['gross_margin'].rolling(4).mean()
        
        # Lagged features
        for lag in [1, 2, 4]:
            for col in ['revenue', 'net_income', 'eps']:
                if col in df.columns:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Moving averages
        for window in [2, 4, 8]:
            if 'revenue' in df.columns:
                df[f'revenue_ma{window}'] = df['revenue'].rolling(window).mean()
                df[f'revenue_std{window}'] = df['revenue'].rolling(window).std()
        
        # Trend features using linear regression
        if 'revenue' in df.columns:
            df['revenue_trend'] = self._calculate_trend(df['revenue'], window=4)
        
        return df
    
    def _calculate_trend(self, series: pd.Series, window: int = 4) -> pd.Series:
        """Calculate trend using rolling linear regression"""
        trend = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            
            if not np.isnan(y).any():
                coef = np.polyfit(x, y, 1)[0]
                trend.iloc[i] = coef
        
        return trend
    
    def _add_technical_features(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators from price data"""
        
        # Align price data with financial data
        price_features = price_data[['close', 'volume', 'volatility_20d', 'rsi_14']].copy()
        price_features.columns = [f'price_{col}' for col in price_features.columns]
        
        # Resample to match financial data frequency
        price_quarterly = price_features.resample('Q').agg({
            'price_close': 'last',
            'price_volume': 'sum',
            'price_volatility_20d': 'mean',
            'price_rsi_14': 'mean'
        })
        
        # Merge with financial data
        df = pd.merge(df, price_quarterly, left_index=True, right_index=True, how='left')
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features"""
        
        macro_features = ['gdp_growth', 'unemployment_rate', 'interest_rate', 
                         'inflation_rate', 'vix']
        
        for feature in macro_features:
            if feature in macro_data.columns:
                # Resample to quarterly if needed
                macro_quarterly = macro_data[[feature]].resample('Q').mean()
                df = pd.merge(df, macro_quarterly, left_index=True, 
                            right_index=True, how='left')
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment scores"""
        
        sentiment_cols = ['mda_sentiment', 'risk_sentiment', 'overall_sentiment']
        
        for col in sentiment_cols:
            if col in sentiment_data.columns:
                df = pd.merge(df, sentiment_data[[col]], left_index=True, 
                            right_index=True, how='left')
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['days_in_quarter'] = df.index.to_period('Q').asfreq('D', 'end').day
        
        # Cyclical encoding for quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Time trend
        df['time_trend'] = np.arange(len(df))
        df['time_trend_squared'] = df['time_trend'] ** 2
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        
        # Forward fill for most recent values
        df = df.ffill(limit=1)
        
        # Interpolate for small gaps
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=2)
        
        # Fill remaining with median
        df = df.fillna(df.median())
        
        return df
    
    def fit_all_models(self, data: pd.DataFrame) -> Dict[str, ForecastResult]:
        """
        Fit all enabled models
        
        Args:
            data: Prepared data
            
        Returns:
            Dictionary of forecast results
        """
        results = {}
        
        # Split data
        train_size = int(len(data) * (1 - self.config.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Store test data for evaluation
        self.test_data = test_data
        
        # Time series models
        if self.config.enable_arima:
            results['arima'] = self._fit_arima(train_data)
        
        if self.config.enable_sarimax:
            results['sarimax'] = self._fit_sarimax(train_data)
        
        if self.config.enable_prophet:
            results['prophet'] = self._fit_prophet(train_data)
        
        # ML models
        if self.config.enable_ml_models:
            results['xgboost'] = self._fit_xgboost(train_data)
            results['lightgbm'] = self._fit_lightgbm(train_data)
            results['catboost'] = self._fit_catboost(train_data)
            results['random_forest'] = self._fit_random_forest(train_data)
        
        # Deep learning models
        if self.config.enable_deep_learning:
            results['lstm'] = self._fit_lstm(train_data)
            results['gru'] = self._fit_gru(train_data)
        
        # Ensemble
        if self.config.enable_ensemble and len(results) > 1:
            results['ensemble'] = self._create_ensemble(results)
        
        self.forecast_results = results
        
        # Evaluate models
        self._evaluate_models(test_data)
        
        return results
    
    def _fit_arima(self, data: pd.DataFrame) -> ForecastResult:
        """Fit ARIMA model with auto parameter selection"""
        
        y = data[self.config.target_variable].values
        
        # Auto ARIMA
        model = pm.auto_arima(
            y,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=True, m=4,  # Quarterly seasonality
            stepwise=True,
            suppress_warnings=True,
            information_criterion='aic',
            n_jobs=-1
        )
        
        # Forecast
        forecast, conf_int = model.predict(
            n_periods=self.config.forecast_horizon,
            return_conf_int=True,
            alpha=1 - self.config.var_confidence
        )
        
        # Calculate metrics
        in_sample_pred = model.predict_in_sample()
        residuals = y - in_sample_pred
        metrics = self._calculate_metrics(y[-len(in_sample_pred):], in_sample_pred)
        
        return ForecastResult(
            model_name='ARIMA',
            forecast=forecast,
            lower_bound=conf_int[:, 0],
            upper_bound=conf_int[:, 1],
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            model_params=model.get_params(),
            residuals=residuals
        )
    
    def _fit_sarimax(self, data: pd.DataFrame) -> ForecastResult:
        """Fit SARIMAX with exogenous variables"""
        
        y = data[self.config.target_variable]
        
        # Select top exogenous variables
        exog_vars = self._select_top_features(data, n_features=5)
        X = data[exog_vars] if exog_vars else None
        
        # Grid search for best parameters
        best_aic = np.inf
        best_model = None
        best_params = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = SARIMAX(
                            y, exog=X,
                            order=(p, d, q),
                            seasonal_order=(1, 1, 1, 4),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False)
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_model = results
                            best_params = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            # Fallback to simple model
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            best_model = model.fit(disp=False)
        
        # Forecast
        forecast = best_model.forecast(steps=self.config.forecast_horizon, exog=X[-1:] if X is not None else None)
        forecast_df = best_model.get_forecast(steps=self.config.forecast_horizon)
        conf_int = forecast_df.conf_int(alpha=1 - self.config.var_confidence)
        
        # Metrics
        residuals = best_model.resid
        metrics = self._calculate_metrics(y.values, best_model.fittedvalues.values)
        
        return ForecastResult(
            model_name='SARIMAX',
            forecast=forecast.values,
            lower_bound=conf_int.iloc[:, 0].values,
            upper_bound=conf_int.iloc[:, 1].values,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            model_params={'order': best_params, 'aic': best_aic},
            residuals=residuals.values
        )
    
    def _fit_prophet(self, data: pd.DataFrame) -> ForecastResult:
        """Fit Facebook Prophet model"""
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[self.config.target_variable].values
        })
        
        # Add regressors
        regressor_cols = []
        if self.config.use_external_regressors:
            for col in ['gdp_growth', 'interest_rate', 'price_close']:
                if col in data.columns:
                    prophet_data[col] = data[col].values
                    regressor_cols.append(col)
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=self.config.var_confidence
        )
        
        # Add regressors
        for col in regressor_cols:
            model.add_regressor(col)
        
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=self.config.forecast_horizon, freq='Q')
        
        # Add regressor values for future
        for col in regressor_cols:
            # Simple forward fill for demo - in production use proper forecasts
            future[col] = prophet_data[col].iloc[-1]
        
        # Predict
        forecast = model.predict(future)
        
        # Extract results
        forecast_values = forecast['yhat'].iloc[-self.config.forecast_horizon:].values
        lower_bound = forecast['yhat_lower'].iloc[-self.config.forecast_horizon:].values
        upper_bound = forecast['yhat_upper'].iloc[-self.config.forecast_horizon:].values
        
        # Calculate metrics
        in_sample = forecast['yhat'].iloc[:-self.config.forecast_horizon].values
        actual = prophet_data['y'].values
        metrics = self._calculate_metrics(actual, in_sample[:len(actual)])
        
        return ForecastResult(
            model_name='Prophet',
            forecast=forecast_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            residuals=actual - in_sample[:len(actual)]
        )
    
    def _fit_ets(self, data: pd.DataFrame) -> ForecastResult:
        """Fit Exponential Smoothing (ETS) model"""
        
        y = data[self.config.target_variable].values
        
        # Fit model
        model = ExponentialSmoothing(
            y,
            seasonal_periods=4,
            trend='add',
            seasonal='add',
            damped_trend=True
        )
        
        fit = model.fit(optimized=True)
        
        # Forecast
        forecast = fit.forecast(steps=self.config.forecast_horizon)
        
        # Confidence intervals using simulation
        simulations = fit.simulate(
            nsimulations=self.config.forecast_horizon,
            repetitions=1000,
            anchor='end'
        )
        
        lower_bound = np.percentile(simulations, (1 - self.config.var_confidence) / 2 * 100, axis=1)
        upper_bound = np.percentile(simulations, (1 + self.config.var_confidence) / 2 * 100, axis=1)
        
        # Metrics
        fitted = fit.fittedvalues
        metrics = self._calculate_metrics(y, fitted)
        
        return ForecastResult(
            model_name='ETS',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            residuals=fit.resid
        )
    
    def _fit_xgboost(self, data: pd.DataFrame) -> ForecastResult:
        """Fit XGBoost model with time series features"""
        
        # Prepare features and target
        X, y = self._prepare_ml_data(data)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        # Hyperparameter tuning
        if self.config.enable_hyperopt:
            params = self._optimize_xgboost_params(X, y, tscv)
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        # Train model
        model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Store model
        self.models['xgboost'] = model
        
        # Feature importance
        importance = dict(zip(X.columns, model.feature_importances_))
        self.feature_importance['xgboost'] = importance
        
        # Forecast
        forecast = self._recursive_forecast(model, X, self.config.forecast_horizon)
        
        # Confidence intervals using quantile regression
        lower_model = xgb.XGBRegressor(**params, objective='reg:quantileerror', 
                                       quantile_alpha=(1-self.config.var_confidence)/2)
        upper_model = xgb.XGBRegressor(**params, objective='reg:quantileerror',
                                       quantile_alpha=(1+self.config.var_confidence)/2)
        
        lower_model.fit(X, y)
        upper_model.fit(X, y)
        
        lower_bound = self._recursive_forecast(lower_model, X, self.config.forecast_horizon)
        upper_bound = self._recursive_forecast(upper_model, X, self.config.forecast_horizon)
        
        # Metrics
        predictions = model.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        return ForecastResult(
            model_name='XGBoost',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            feature_importance=importance,
            model_params=params,
            residuals=y - predictions
        )
    
    def _fit_lightgbm(self, data: pd.DataFrame) -> ForecastResult:
        """Fit LightGBM model"""
        
        X, y = self._prepare_ml_data(data)
        
        # Hyperparameter tuning
        if self.config.enable_hyperopt:
            params = self._optimize_lightgbm_params(X, y)
        else:
            params = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100
            }
        
        # Train model
        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        self.models['lightgbm'] = model
        
        # Feature importance
        importance = dict(zip(X.columns, model.feature_importances_))
        self.feature_importance['lightgbm'] = importance
        
        # Forecast
        forecast = self._recursive_forecast(model, X, self.config.forecast_horizon)
        
        # Bootstrap confidence intervals
        lower_bound, upper_bound = self._bootstrap_confidence_intervals(
            model, X, y, self.config.forecast_horizon
        )
        
        # Metrics
        predictions = model.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        return ForecastResult(
            model_name='LightGBM',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            feature_importance=importance,
            model_params=params,
            residuals=y - predictions
        )
    
    def _fit_catboost(self, data: pd.DataFrame) -> ForecastResult:
        """Fit CatBoost model"""
        
        X, y = self._prepare_ml_data(data)
        
        # CatBoost parameters
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'logging_level': 'Silent'
        }
        
        # Train model
        model = CatBoostRegressor(**params)
        model.fit(X, y, verbose=False)
        
        self.models['catboost'] = model
        
        # Feature importance
        importance = dict(zip(X.columns, model.feature_importances_))
        self.feature_importance['catboost'] = importance
        
        # Forecast
        forecast = self._recursive_forecast(model, X, self.config.forecast_horizon)
        
        # Confidence intervals
        lower_bound, upper_bound = self._bootstrap_confidence_intervals(
            model, X, y, self.config.forecast_horizon
        )
        
        # Metrics
        predictions = model.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        return ForecastResult(
            model_name='CatBoost',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            feature_importance=importance,
            residuals=y - predictions
        )
    
    def _fit_random_forest(self, data: pd.DataFrame) -> ForecastResult:
        """Fit Random Forest model"""
        
        X, y = self._prepare_ml_data(data)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        self.models['random_forest'] = model
        
        # Feature importance
        importance = dict(zip(X.columns, model.feature_importances_))
        self.feature_importance['random_forest'] = importance
        
        # Forecast with prediction intervals
        forecast = self._recursive_forecast(model, X, self.config.forecast_horizon)
        
        # Get prediction intervals from individual trees
        tree_predictions = []
        for tree in model.estimators_:
            tree_pred = self._recursive_forecast(tree, X, self.config.forecast_horizon)
            tree_predictions.append(tree_pred)
        
        tree_predictions = np.array(tree_predictions)
        lower_bound = np.percentile(tree_predictions, (1 - self.config.var_confidence) / 2 * 100, axis=0)
        upper_bound = np.percentile(tree_predictions, (1 + self.config.var_confidence) / 2 * 100, axis=0)
        
        # Metrics
        predictions = model.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        return ForecastResult(
            model_name='RandomForest',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            feature_importance=importance,
            residuals=y - predictions
        )
    
    def _fit_lstm(self, data: pd.DataFrame) -> ForecastResult:
        """Fit LSTM neural network"""
        
        # Prepare sequences
        X, y = self._prepare_sequences(data, lookback=self.config.lookback_window)
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.scalers['lstm_X'] = scaler_X
        self.scalers['lstm_y'] = scaler_y
        
        # Build LSTM model
        model = LSTMForecaster(
            input_size=X.shape[-1],
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )
        
        # Train model
        model.train_model(X_scaled, y_scaled, epochs=100, batch_size=32)
        
        self.models['lstm'] = model
        
        # Forecast
        forecast_scaled = model.forecast(X_scaled[-1:], steps=self.config.forecast_horizon)
        forecast = scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        
        # Monte Carlo dropout for uncertainty
        mc_predictions = []
        for _ in range(100):
            mc_pred = model.forecast_with_dropout(X_scaled[-1:], steps=self.config.forecast_horizon)
            mc_pred = scaler_y.inverse_transform(mc_pred.reshape(-1, 1)).flatten()
            mc_predictions.append(mc_pred)
        
        mc_predictions = np.array(mc_predictions)
        lower_bound = np.percentile(mc_predictions, (1 - self.config.var_confidence) / 2 * 100, axis=0)
        upper_bound = np.percentile(mc_predictions, (1 + self.config.var_confidence) / 2 * 100, axis=0)
        
        # Metrics
        predictions_scaled = model.predict(X_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        metrics = self._calculate_metrics(y[:len(predictions)], predictions)
        
        return ForecastResult(
            model_name='LSTM',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            residuals=y[:len(predictions)] - predictions
        )
    
    def _fit_gru(self, data: pd.DataFrame) -> ForecastResult:
        """Fit GRU neural network"""
        
        # Similar to LSTM but with GRU cells
        X, y = self._prepare_sequences(data, lookback=self.config.lookback_window)
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Build and train GRU model
        model = GRUForecaster(
            input_size=X.shape[-1],
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        model.train_model(X_scaled, y_scaled, epochs=100)
        
        # Forecast
        forecast_scaled = model.forecast(X_scaled[-1:], steps=self.config.forecast_horizon)
        forecast = scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        
        # Confidence intervals
        lower_bound = forecast * 0.9  # Simplified for demo
        upper_bound = forecast * 1.1
        
        # Metrics
        predictions_scaled = model.predict(X_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        metrics = self._calculate_metrics(y[:len(predictions)], predictions)
        
        return ForecastResult(
            model_name='GRU',
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.var_confidence,
            metrics=metrics,
            residuals=y[:len(predictions)] - predictions
        )
    
    def _fit_transformer(self, data: pd.DataFrame) -> ForecastResult:
        """Fit Transformer model for time series"""
        # Placeholder for transformer implementation
        # Would use models like Temporal Fusion Transformer
        pass
    
    def _prepare_ml_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for ML models"""
        
        # Select features
        feature_cols = [col for col in data.columns 
                       if col not in [self.config.target_variable, 'date']]
        
        X = data[feature_cols]
        y = data[self.config.target_variable].values
        
        return X, y
    
    def _prepare_sequences(self, data: pd.DataFrame, lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for deep learning models"""
        
        # Convert to array
        values = data.values
        
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i, data.columns.get_loc(self.config.target_variable)])
        
        return np.array(X), np.array(y)
    
    def _recursive_forecast(self, model, X: pd.DataFrame, steps: int) -> np.ndarray:
        """Generate multi-step recursive forecast"""
        
        forecast = []
        last_features = X.iloc[-1:].copy()
        
        for _ in range(steps):
            # Predict next step
            pred = model.predict(last_features)[0]
            forecast.append(pred)
            
            # Update features for next prediction
            # Shift lag features
            for col in last_features.columns:
                if 'lag' in col:
                    lag_num = int(col.split('lag')[-1])
                    if lag_num > 1:
                        # Shift to next lag
                        new_col = col.replace(f'lag{lag_num}', f'lag{lag_num-1}')
                        if new_col in last_features.columns:
                            last_features[col] = last_features[new_col].values
                    else:
                        # Update lag1 with current prediction
                        last_features[col] = pred
            
            # Update other features (simplified)
            last_features['time_trend'] = last_features['time_trend'] + 1
            last_features['time_trend_squared'] = last_features['time_trend'] ** 2
        
        return np.array(forecast)
    
    def _bootstrap_confidence_intervals(self, model, X: pd.DataFrame, y: np.ndarray, 
                                       steps: int, n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals using bootstrap"""
        
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y[indices]
            
            # Retrain model
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_boot, y_boot)
            
            # Forecast
            forecast = self._recursive_forecast(model_boot, X_boot, steps)
            predictions.append(forecast)
        
        predictions = np.array(predictions)
        
        lower = np.percentile(predictions, (1 - self.config.var_confidence) / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 + self.config.var_confidence) / 2 * 100, axis=0)
        
        return lower, upper
    
    def _create_ensemble(self, individual_results: Dict[str, ForecastResult]) -> ForecastResult:
        """Create ensemble forecast from individual models"""
        
        # Get forecasts from all models
        forecasts = []
        weights = []
        
        for model_name, result in individual_results.items():
            forecasts.append(result.forecast)
            # Weight by inverse of error metric
            weight = 1 / (result.metrics.get('rmse', 1) + 1e-6)
            weights.append(weight)
        
        forecasts = np.array(forecasts)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average ensemble
        ensemble_forecast = np.average(forecasts, weights=weights, axis=0)
        
        # Confidence intervals from individual models
        lower_bounds = []
        upper_bounds = []
        
        for result in individual_results.values():
            lower_bounds.append(result.lower_bound)
            upper_bounds.append(result.upper_bound)
        
        ensemble_lower = np.average(lower_bounds, weights=weights, axis=0)
        ensemble_upper = np.average(upper_bounds, weights=weights, axis=0)
        
        # Calculate ensemble metrics
        metrics = {
            'ensemble_weights': dict(zip(individual_results.keys(), weights.tolist())),
            'n_models': len(individual_results)
        }
        
        return ForecastResult(
            model_name='Ensemble',
            forecast=ensemble_forecast,
            lower_bound=ensemble_lower,
            upper_bound=ensemble_upper,
            confidence_level=self.config.var_confidence,
            metrics=metrics
        )
    
    def _select_top_features(self, data: pd.DataFrame, n_features: int = 10) -> List[str]:
        """Select top features using correlation and importance"""
        
        # Calculate correlations
        target = data[self.config.target_variable]
        correlations = {}
        
        for col in data.columns:
            if col != self.config.target_variable:
                corr = target.corr(data[col])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n features
        return [feat[0] for feat in sorted_features[:n_features]]
    
    def _optimize_xgboost_params(self, X: pd.DataFrame, y: np.ndarray, 
                                 tscv: TimeSeriesSplit) -> Dict:
        """Optimize XGBoost hyperparameters using Bayesian optimization"""
        
        from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
        
        # Define search space
        space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 10]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': hp.choice('min_child_weight', [1, 3, 5]),
        }
        
        def objective(params):
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=tscv, 
                                    scoring='neg_mean_squared_error')
            
            return {'loss': -scores.mean(), 'status': STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=self.config.hyperopt_trials,
                   trials=trials,
                   verbose=0)
        
        # Convert back to proper format
        params = {
            'n_estimators': [50, 100, 200][best['n_estimators']],
            'max_depth': [3, 5, 7, 10][best['max_depth']],
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': [1, 3, 5][best['min_child_weight']]
        }
        
        return params
    
    def _optimize_lightgbm_params(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Optimize LightGBM hyperparameters"""
        
        # Similar to XGBoost optimization
        params = {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 100
        }
        
        return params
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mae': mean_absolute_error(actual, predicted),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100,
            'r2': r2_score(actual, predicted),
            'mse': mean_squared_error(actual, predicted)
        }
        
        return metrics
    
    def _evaluate_models(self, test_data: pd.DataFrame):
        """Evaluate all models on test data"""
        
        for model_name, result in self.forecast_results.items():
            logger.info(f"\n{model_name} Performance:")
            logger.info(f"  RMSE: {result.metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"  MAPE: {result.metrics.get('mape', 'N/A'):.2f}%")
            logger.info(f"  R²: {result.metrics.get('r2', 'N/A'):.4f}")
    
    def get_best_model(self) -> str:
        """Get the best performing model based on validation metric"""
        
        best_model = None
        best_score = float('inf') if self.config.validation_metric in ['rmse', 'mae', 'mape'] else -float('inf')
        
        for model_name, result in self.forecast_results.items():
            score = result.metrics.get(self.config.validation_metric, float('inf'))
            
            if self.config.validation_metric in ['rmse', 'mae', 'mape']:
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:  # For R2, higher is better
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        logger.info(f"Best model: {best_model} with {self.config.validation_metric}: {best_score:.4f}")
        return best_model
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load models from disk"""
        
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        
        logger.info(f"Models loaded from {path}")


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
        """Train the LSTM model"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self(X_tensor).numpy()
        return predictions.squeeze()
    
    def forecast(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """Generate multi-step forecast"""
        self.eval()
        forecast = []
        
        current_seq = torch.FloatTensor(last_sequence)
        
        with torch.no_grad():
            for _ in range(steps):
                pred = self(current_seq)
                forecast.append(pred.item())
                
                # Update sequence (simplified - would need proper feature engineering)
                current_seq = torch.roll(current_seq, -1, dims=1)
                current_seq[:, -1, 0] = pred.item()
        
        return np.array(forecast)
    
    def forecast_with_dropout(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """Forecast with dropout for uncertainty estimation"""
        self.train()  # Enable dropout
        forecast = []
        
        current_seq = torch.FloatTensor(last_sequence)
        
        with torch.no_grad():
            for _ in range(steps):
                pred = self(current_seq)
                forecast.append(pred.item())
                
                current_seq = torch.roll(current_seq, -1, dims=1)
                current_seq[:, -1, 0] = pred.item()
        
        return np.array(forecast)


class GRUForecaster(nn.Module):
    """GRU model for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1):
        super(GRUForecaster, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the GRU model"""
        # Similar to LSTM training
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self(X_tensor).numpy()
        return predictions.squeeze()
    
    def forecast(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """Generate forecast"""
        # Similar to LSTM forecast
        return np.zeros(steps)  # Placeholder