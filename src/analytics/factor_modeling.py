"""
Anomaly Detection & Factor Analysis Module
Advanced anomaly detection, factor modeling, and alpha generation
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
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization

# Factor Analysis
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.rolling import RollingOLS

# Deep Learning
import torch
import torch.nn as nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""
    timestamp: pd.Timestamp
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    
    # Anomaly details
    anomaly_type: str  # outlier, structural_break, regime_change
    affected_metrics: List[str]
    deviation_from_normal: Dict[str, float]
    
    # Context
    explanation: str
    similar_historical: List[pd.Timestamp]
    expected_range: Tuple[float, float]
    
    # Impact
    business_impact: Optional[str] = None
    severity: str = 'medium'  # low, medium, high, critical
    action_required: bool = False


@dataclass
class FactorModel:
    """Container for factor model results"""
    factors: pd.DataFrame
    loadings: pd.DataFrame
    eigenvalues: np.ndarray
    variance_explained: np.ndarray
    
    # Model metrics
    total_variance_explained: float
    factor_scores: pd.DataFrame
    residuals: pd.DataFrame
    
    # Factor interpretation
    factor_names: Dict[int, str]
    factor_correlations: pd.DataFrame
    
    # Statistical tests
    kaiser_meyer_olkin: float
    bartlett_sphericity: Tuple[float, float]  # statistic, p-value


@dataclass
class AlphaSignal:
    """Container for alpha generation signals"""
    signal_name: str
    signal_value: float
    expected_return: float
    confidence_interval: Tuple[float, float]
    
    # Signal components
    factor_exposures: Dict[str, float]
    idiosyncratic_component: float
    
    # Risk metrics
    signal_volatility: float
    information_ratio: float
    max_drawdown: float
    
    # Timing
    entry_signal: bool
    exit_signal: bool
    holding_period: int
    
    # Backtesting results
    historical_returns: Optional[pd.Series] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None


class AnomalyDetector:
    """
    Multi-method anomaly detection system for financial data
    """
    
    def __init__(self,
                 methods: List[str] = ['isolation_forest', 'autoencoder', 'statistical'],
                 contamination: float = 0.05,
                 sensitivity: float = 0.95):
        """
        Initialize anomaly detector
        
        Args:
            methods: Detection methods to use
            contamination: Expected proportion of anomalies
            sensitivity: Detection sensitivity (1 - false negative rate)
        """
        self.methods = methods
        self.contamination = contamination
        self.sensitivity = sensitivity
        
        # Initialize detectors
        self.detectors = self._initialize_detectors()
        
        # Historical anomalies for pattern matching
        self.historical_anomalies = []
        
        logger.info(f"AnomalyDetector initialized with {len(methods)} methods")
    
    def _initialize_detectors(self) -> Dict:
        """Initialize anomaly detection models"""
        detectors = {}
        
        if 'isolation_forest' in self.methods:
            detectors['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        
        if 'one_class_svm' in self.methods:
            detectors['one_class_svm'] = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        
        if 'lof' in self.methods:
            detectors['lof'] = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
        
        if 'elliptic' in self.methods:
            detectors['elliptic'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        
        if 'knn' in self.methods:
            detectors['knn'] = KNN(contamination=self.contamination)
        
        return detectors
    
    def detect_anomalies(self,
                        data: pd.DataFrame,
                        target_column: Optional[str] = None,
                        real_time: bool = False) -> List[AnomalyResult]:
        """
        Detect anomalies in financial data
        
        Args:
            data: Financial data DataFrame
            target_column: Specific column to analyze
            real_time: Enable real-time detection mode
            
        Returns:
            List of anomaly results
        """
        anomalies = []
        
        # Prepare data
        X = self._prepare_data(data, target_column)
        
        # Statistical anomaly detection
        if 'statistical' in self.methods:
            stat_anomalies = self._detect_statistical_anomalies(data, target_column)
            anomalies.extend(stat_anomalies)
        
        # Machine learning based detection
        ml_anomalies = self._detect_ml_anomalies(X, data.index)
        anomalies.extend(ml_anomalies)
        
        # Structural break detection
        if not real_time:
            breaks = self._detect_structural_breaks(data, target_column)
            anomalies.extend(breaks)
        
        # Regime change detection
        regime_changes = self._detect_regime_changes(data)
        anomalies.extend(regime_changes)
        
        # Combine and rank anomalies
        anomalies = self._combine_anomaly_results(anomalies)
        
        # Store for historical pattern matching
        self.historical_anomalies.extend(anomalies)
        
        return anomalies
    
    def _prepare_data(self, data: pd.DataFrame, target_column: Optional[str]) -> np.ndarray:
        """Prepare data for anomaly detection"""
        if target_column:
            X = data[[target_column]].values
        else:
            # Use all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X))
        
        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    
    def _detect_statistical_anomalies(self,
                                     data: pd.DataFrame,
                                     target_column: Optional[str]) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        cols_to_check = [target_column] if target_column else data.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_check:
            if col not in data.columns:
                continue
            
            series = data[col].dropna()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            threshold = stats.norm.ppf(self.sensitivity)
            
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            for idx in anomaly_indices:
                timestamp = series.index[idx]
                value = series.iloc[idx]
                z_score = z_scores[idx]
                
                # Calculate expected range
                mean = series.mean()
                std = series.std()
                expected_range = (mean - 2*std, mean + 2*std)
                
                # Find similar historical anomalies
                similar = self._find_similar_anomalies(value, col)
                
                anomaly = AnomalyResult(
                    timestamp=timestamp,
                    is_anomaly=True,
                    anomaly_score=z_score,
                    confidence=min(0.99, 1 - stats.norm.cdf(-abs(z_score))),
                    anomaly_type='outlier',
                    affected_metrics=[col],
                    deviation_from_normal={col: (value - mean) / std},
                    explanation=f"{col} value {value:.2f} is {z_score:.1f} standard deviations from mean",
                    similar_historical=similar,
                    expected_range=expected_range,
                    severity='high' if z_score > 4 else 'medium' if z_score > 3 else 'low'
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_ml_anomalies(self, X: np.ndarray, index: pd.Index) -> List[AnomalyResult]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        # Train detectors
        predictions = {}
        scores = {}
        
        for name, detector in self.detectors.items():
            try:
                if name == 'autoencoder':
                    # Special handling for autoencoder
                    anomaly_labels, anomaly_scores = self._detect_autoencoder_anomalies(X)
                    predictions[name] = anomaly_labels
                    scores[name] = anomaly_scores
                else:
                    # Fit and predict
                    detector.fit(X)
                    predictions[name] = detector.predict(X)
                    
                    # Get anomaly scores
                    if hasattr(detector, 'score_samples'):
                        scores[name] = -detector.score_samples(X)
                    elif hasattr(detector, 'decision_function'):
                        scores[name] = -detector.decision_function(X)
                    else:
                        scores[name] = predictions[name]
            except Exception as e:
                logger.warning(f"Error in {name} detector: {e}")
                continue
        
        # Ensemble voting
        if predictions:
            ensemble_predictions = self._ensemble_voting(predictions)
            ensemble_scores = self._ensemble_scores(scores)
            
            # Create anomaly results
            for i, is_anomaly in enumerate(ensemble_predictions):
                if is_anomaly == -1:  # Anomaly detected
                    anomaly = AnomalyResult(
                        timestamp=index[i],
                        is_anomaly=True,
                        anomaly_score=ensemble_scores[i],
                        confidence=self._calculate_confidence(scores, i),
                        anomaly_type='ml_detected',
                        affected_metrics=['multiple'],
                        deviation_from_normal={},
                        explanation=f"ML ensemble detected anomaly with score {ensemble_scores[i]:.3f}",
                        similar_historical=[],
                        expected_range=(0, 0),
                        severity=self._calculate_severity(ensemble_scores[i])
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_autoencoder_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using autoencoder"""
        # Build autoencoder
        autoencoder = AutoEncoderDetector(
            input_dim=X.shape[1],
            encoding_dim=max(2, X.shape[1] // 2)
        )
        
        # Train
        autoencoder.fit(X)
        
        # Get reconstruction error
        predictions = autoencoder.predict(X)
        scores = autoencoder.decision_scores_
        
        return predictions, scores
    
    def _detect_structural_breaks(self,
                                 data: pd.DataFrame,
                                 target_column: Optional[str]) -> List[AnomalyResult]:
        """Detect structural breaks in time series"""
        anomalies = []
        
        cols = [target_column] if target_column else data.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in data.columns:
                continue
            
            series = data[col].dropna()
            
            # CUSUM test for structural breaks
            breaks = self._cusum_test(series)
            
            for break_point in breaks:
                anomaly = AnomalyResult(
                    timestamp=break_point,
                    is_anomaly=True,
                    anomaly_score=1.0,
                    confidence=0.95,
                    anomaly_type='structural_break',
                    affected_metrics=[col],
                    deviation_from_normal={},
                    explanation=f"Structural break detected in {col} at {break_point}",
                    similar_historical=[],
                    expected_range=(0, 0),
                    severity='high',
                    action_required=True
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _cusum_test(self, series: pd.Series, threshold: float = 0.05) -> List[pd.Timestamp]:
        """CUSUM test for structural breaks"""
        breaks = []
        
        # Calculate cumulative sum of recursive residuals
        mean = series.mean()
        cusum = np.cumsum(series - mean)
        
        # Normalize
        n = len(series)
        cusum_normalized = cusum / (series.std() * np.sqrt(n))
        
        # Find break points where CUSUM exceeds critical value
        critical_value = 1.358  # 95% confidence
        
        break_indices = np.where(np.abs(cusum_normalized) > critical_value)[0]
        
        if len(break_indices) > 0:
            # Group consecutive indices
            groups = np.split(break_indices, np.where(np.diff(break_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    breaks.append(series.index[group[0]])
        
        return breaks
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect regime changes using Hidden Markov Models or similar"""
        anomalies = []
        
        # Simplified regime detection using rolling statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            
            if len(series) < 20:
                continue
            
            # Calculate rolling mean and std
            rolling_mean = series.rolling(window=10).mean()
            rolling_std = series.rolling(window=10).std()
            
            # Detect significant changes
            mean_change = rolling_mean.diff().abs()
            std_change = rolling_std.diff().abs()
            
            # Threshold for regime change
            mean_threshold = mean_change.quantile(0.95)
            std_threshold = std_change.quantile(0.95)
            
            regime_changes = series.index[
                (mean_change > mean_threshold) | (std_change > std_threshold)
            ]
            
            for timestamp in regime_changes:
                if pd.notna(timestamp):
                    anomaly = AnomalyResult(
                        timestamp=timestamp,
                        is_anomaly=True,
                        anomaly_score=0.8,
                        confidence=0.8,
                        anomaly_type='regime_change',
                        affected_metrics=[col],
                        deviation_from_normal={},
                        explanation=f"Regime change detected in {col}",
                        similar_historical=[],
                        expected_range=(0, 0),
                        severity='medium'
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _ensemble_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Ensemble voting for anomaly detection"""
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Majority voting
        ensemble = pred_df.mode(axis=1).iloc[:, 0].values
        
        return ensemble
    
    def _ensemble_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine anomaly scores from multiple detectors"""
        if not scores:
            return np.array([])
        
        # Average scores
        scores_array = np.column_stack(list(scores.values()))
        return np.mean(scores_array, axis=1)
    
    def _calculate_confidence(self, scores: Dict[str, np.ndarray], index: int) -> float:
        """Calculate confidence of anomaly detection"""
        if not scores:
            return 0.5
        
        # Get all scores for this point
        point_scores = [score[index] for score in scores.values()]
        
        # Calculate agreement between detectors
        agreement = np.std(point_scores)
        
        # Lower std means higher confidence
        confidence = max(0.5, min(0.99, 1 - agreement))
        
        return confidence
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity"""
        if score > np.percentile(self.historical_anomalies, 95) if self.historical_anomalies else 3:
            return 'critical'
        elif score > np.percentile(self.historical_anomalies, 80) if self.historical_anomalies else 2:
            return 'high'
        elif score > np.percentile(self.historical_anomalies, 50) if self.historical_anomalies else 1:
            return 'medium'
        else:
            return 'low'
    
    def _find_similar_anomalies(self, value: float, metric: str) -> List[pd.Timestamp]:
        """Find similar historical anomalies"""
        similar = []
        
        for historical in self.historical_anomalies:
            if metric in historical.affected_metrics:
                # Simple similarity based on value range
                if abs(historical.anomaly_score - value) < 0.5:
                    similar.append(historical.timestamp)
        
        return similar[:5]  # Return top 5 similar
    
    def _combine_anomaly_results(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
        """Combine and deduplicate anomaly results"""
        # Group by timestamp
        grouped = defaultdict(list)
        for anomaly in anomalies:
            grouped[anomaly.timestamp].append(anomaly)
        
        # Combine anomalies at same timestamp
        combined = []
        for timestamp, group in grouped.items():
            if len(group) == 1:
                combined.append(group[0])
            else:
                # Merge multiple detections
                merged = AnomalyResult(
                    timestamp=timestamp,
                    is_anomaly=True,
                    anomaly_score=max(a.anomaly_score for a in group),
                    confidence=max(a.confidence for a in group),
                    anomaly_type=', '.join(set(a.anomaly_type for a in group)),
                    affected_metrics=list(set(sum([a.affected_metrics for a in group], []))),
                    deviation_from_normal={},
                    explanation=f"Multiple anomalies detected: {', '.join(a.anomaly_type for a in group)}",
                    similar_historical=[],
                    expected_range=(0, 0),
                    severity=max(group, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.severity)).severity
                )
                combined.append(merged)
        
        return combined


class FactorAnalyzer:
    """
    Factor analysis and alpha generation for quantitative strategies
    """
    
    def __init__(self,
                 n_factors: int = 5,
                 factor_method: str = 'pca',
                 rotation: Optional[str] = 'varimax'):
        """
        Initialize factor analyzer
        
        Args:
            n_factors: Number of factors to extract
            factor_method: Method for factor extraction ('pca', 'fa', 'ica')
            rotation: Factor rotation method
        """
        self.n_factors = n_factors
        self.factor_method = factor_method
        self.rotation = rotation
        
        # Initialize models
        self.factor_model = None
        self.scaler = StandardScaler()
        
        logger.info(f"FactorAnalyzer initialized with {n_factors} factors using {factor_method}")
    
    def extract_factors(self, returns_data: pd.DataFrame) -> FactorModel:
        """
        Extract factors from return data
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            FactorModel object
        """
        # Standardize data
        X = self.scaler.fit_transform(returns_data.fillna(0))
        
        # Extract factors based on method
        if self.factor_method == 'pca':
            factors, loadings, explained_var = self._pca_factors(X, returns_data.index, returns_data.columns)
        elif self.factor_method == 'fa':
            factors, loadings, explained_var = self._fa_factors(X, returns_data.index, returns_data.columns)
        elif self.factor_method == 'ica':
            factors, loadings, explained_var = self._ica_factors(X, returns_data.index, returns_data.columns)
        else:
            raise ValueError(f"Unknown factor method: {self.factor_method}")
        
        # Calculate eigenvalues
        cov_matrix = np.cov(X.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        
        # Statistical tests
        kmo = self._calculate_kmo(returns_data)
        bartlett = self._bartlett_test(returns_data)
        
        # Factor scores
        factor_scores = pd.DataFrame(
            np.dot(X, loadings),
            index=returns_data.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Residuals
        reconstructed = np.dot(factor_scores, loadings.T)
        residuals = pd.DataFrame(
            X - reconstructed,
            index=returns_data.index,
            columns=returns_data.columns
        )
        
        # Name factors based on loadings
        factor_names = self._interpret_factors(loadings)
        
        # Factor correlations
        factor_correlations = factor_scores.corr()
        
        return FactorModel(
            factors=factors,
            loadings=loadings,
            eigenvalues=eigenvalues[:self.n_factors],
            variance_explained=explained_var,
            total_variance_explained=explained_var.sum(),
            factor_scores=factor_scores,
            residuals=residuals,
            factor_names=factor_names,
            factor_correlations=factor_correlations,
            kaiser_meyer_olkin=kmo,
            bartlett_sphericity=bartlett
        )
    
    def _pca_factors(self, X: np.ndarray, index: pd.Index, columns: pd.Index) -> Tuple:
        """Extract factors using PCA"""
        pca = PCA(n_components=self.n_factors)
        factors = pca.fit_transform(X)
        
        factors_df = pd.DataFrame(
            factors,
            index=index,
            columns=[f'PC{i+1}' for i in range(self.n_factors)]
        )
        
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=columns,
            columns=[f'PC{i+1}' for i in range(self.n_factors)]
        )
        
        explained_var = pca.explained_variance_ratio_
        
        return factors_df, loadings_df, explained_var
    
    def _fa_factors(self, X: np.ndarray, index: pd.Index, columns: pd.Index) -> Tuple:
        """Extract factors using Factor Analysis"""
        fa = FactorAnalysis(n_components=self.n_factors, rotation=self.rotation)
        factors = fa.fit_transform(X)
        
        factors_df = pd.DataFrame(
            factors,
            index=index,
            columns=[f'Factor{i+1}' for i in range(self.n_factors)]
        )
        
        loadings_df = pd.DataFrame(
            fa.components_.T,
            index=columns,
            columns=[f'Factor{i+1}' for i in range(self.n_factors)]
        )
        
        # Calculate variance explained
        explained_var = np.var(factors, axis=0) / np.var(X).sum()
        
        self.factor_model = fa
        
        return factors_df, loadings_df, explained_var
    
    def _ica_factors(self, X: np.ndarray, index: pd.Index, columns: pd.Index) -> Tuple:
        """Extract factors using Independent Component Analysis"""
        ica = FastICA(n_components=self.n_factors, random_state=42)
        factors = ica.fit_transform(X)
        
        factors_df = pd.DataFrame(
            factors,
            index=index,
            columns=[f'IC{i+1}' for i in range(self.n_factors)]
        )
        
        loadings_df = pd.DataFrame(
            ica.mixing_,
            index=columns,
            columns=[f'IC{i+1}' for i in range(self.n_factors)]
        )
        
        explained_var = np.ones(self.n_factors) / self.n_factors  # Equal for ICA
        
        return factors_df, loadings_df, explained_var
    
    def _calculate_kmo(self, data: pd.DataFrame) -> float:
        """Calculate Kaiser-Meyer-Olkin measure"""
        # Simplified KMO calculation
        corr_matrix = data.corr()
        
        # Calculate partial correlations
        try:
            inv_corr = inv(corr_matrix)
            partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
            np.fill_diagonal(partial_corr, 0)
            
            # KMO calculation
            corr_sum = (corr_matrix**2).sum().sum() - len(corr_matrix)
            partial_sum = (partial_corr**2).sum().sum()
            
            kmo = corr_sum / (corr_sum + partial_sum)
        except:
            kmo = 0.5  # Default if calculation fails
        
        return kmo
    
    def _bartlett_test(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Bartlett's test of sphericity"""
        corr_matrix = data.corr()
        n = len(data)
        p = len(data.columns)
        
        # Calculate test statistic
        det = np.linalg.det(corr_matrix)
        if det <= 0:
            return 0, 1  # Invalid
        
        chi_square = -(n - 1 - (2*p + 5)/6) * np.log(det)
        
        # Degrees of freedom
        df = p * (p - 1) / 2
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        
        return chi_square, p_value
    
    def _interpret_factors(self, loadings: pd.DataFrame) -> Dict[int, str]:
        """Interpret and name factors based on loadings"""
        factor_names = {}
        
        for i, col in enumerate(loadings.columns):
            # Get top loading variables
            top_loadings = loadings[col].abs().nlargest(3)
            
            # Name based on top variables
            if 'market' in ' '.join(top_loadings.index).lower():
                factor_names[i] = 'Market Factor'
            elif 'size' in ' '.join(top_loadings.index).lower():
                factor_names[i] = 'Size Factor'
            elif 'value' in ' '.join(top_loadings.index).lower():
                factor_names[i] = 'Value Factor'
            elif 'momentum' in ' '.join(top_loadings.index).lower():
                factor_names[i] = 'Momentum Factor'
            else:
                factor_names[i] = f'Factor {i+1}'
        
        return factor_names
    
    def generate_alpha_signals(self,
                              factor_model: FactorModel,
                              returns_data: pd.DataFrame,
                              lookback: int = 252) -> List[AlphaSignal]:
        """
        Generate alpha signals from factor model
        
        Args:
            factor_model: Fitted factor model
            returns_data: Asset returns
            lookback: Lookback period for signal generation
            
        Returns:
            List of alpha signals
        """
        signals = []
        
        # Factor momentum signals
        momentum_signals = self._factor_momentum_signals(factor_model, lookback)
        signals.extend(momentum_signals)
        
        # Factor reversal signals
        reversal_signals = self._factor_reversal_signals(factor_model)
        signals.extend(reversal_signals)
        
        # Residual momentum
        residual_signals = self._residual_momentum_signals(factor_model, returns_data)
        signals.extend(residual_signals)
        
        # Cross-sectional signals
        cross_signals = self._cross_sectional_signals(factor_model)
        signals.extend(cross_signals)
        
        return signals
    
    def _factor_momentum_signals(self,
                                factor_model: FactorModel,
                                lookback: int) -> List[AlphaSignal]:
        """Generate factor momentum signals"""
        signals = []
        
        for i, factor_name in factor_model.factor_names.items():
            factor_returns = factor_model.factor_scores.iloc[:, i].pct_change()
            
            # Calculate momentum
            momentum = factor_returns.rolling(lookback).mean()
            
            # Generate signal
            current_momentum = momentum.iloc[-1] if len(momentum) > 0 else 0
            
            # Entry/exit logic
            entry = current_momentum > momentum.quantile(0.7)
            exit_signal = current_momentum < momentum.quantile(0.3)
            
            signal = AlphaSignal(
                signal_name=f"{factor_name}_Momentum",
                signal_value=current_momentum,
                expected_return=current_momentum * 252,  # Annualized
                confidence_interval=(current_momentum * 252 * 0.5, current_momentum * 252 * 1.5),
                factor_exposures={factor_name: 1.0},
                idiosyncratic_component=0,
                signal_volatility=factor_returns.std() * np.sqrt(252),
                information_ratio=current_momentum / factor_returns.std() if factor_returns.std() > 0 else 0,
                max_drawdown=self._calculate_max_drawdown(factor_returns),
                entry_signal=entry,
                exit_signal=exit_signal,
                holding_period=20,  # days
                historical_returns=factor_returns
            )
            
            signals.append(signal)
        
        return signals
    
    def _factor_reversal_signals(self, factor_model: FactorModel) -> List[AlphaSignal]:
        """Generate mean reversion signals"""
        signals = []
        
        for i, factor_name in factor_model.factor_names.items():
            factor_values = factor_model.factor_scores.iloc[:, i]
            
            # Z-score for reversal
            z_score = (factor_values.iloc[-1] - factor_values.mean()) / factor_values.std()
            
            # Reversal signal
            if abs(z_score) > 2:
                expected_return = -z_score * 0.1  # Expect 10% reversal per std
                
                signal = AlphaSignal(
                    signal_name=f"{factor_name}_Reversal",
                    signal_value=-z_score,
                    expected_return=expected_return,
                    confidence_interval=(expected_return * 0.5, expected_return * 1.5),
                    factor_exposures={factor_name: -1.0},
                    idiosyncratic_component=0,
                    signal_volatility=factor_values.std(),
                    information_ratio=abs(expected_return) / factor_values.std(),
                    max_drawdown=0,
                    entry_signal=abs(z_score) > 2,
                    exit_signal=abs(z_score) < 0.5,
                    holding_period=10
                )
                
                signals.append(signal)
        
        return signals
    
    def _residual_momentum_signals(self,
                                  factor_model: FactorModel,
                                  returns_data: pd.DataFrame) -> List[AlphaSignal]:
        """Generate signals from residual momentum"""
        signals = []
        
        # Average residual across assets
        avg_residual = factor_model.residuals.mean(axis=1)
        residual_momentum = avg_residual.rolling(20).mean()
        
        if len(residual_momentum) > 0:
            current_momentum = residual_momentum.iloc[-1]
            
            signal = AlphaSignal(
                signal_name="Residual_Momentum",
                signal_value=current_momentum,
                expected_return=current_momentum * 252,
                confidence_interval=(current_momentum * 126, current_momentum * 378),
                factor_exposures={},
                idiosyncratic_component=1.0,
                signal_volatility=avg_residual.std() * np.sqrt(252),
                information_ratio=current_momentum / avg_residual.std() if avg_residual.std() > 0 else 0,
                max_drawdown=self._calculate_max_drawdown(avg_residual),
                entry_signal=current_momentum > 0,
                exit_signal=current_momentum < 0,
                holding_period=5
            )
            
            signals.append(signal)
        
        return signals
    
    def _cross_sectional_signals(self, factor_model: FactorModel) -> List[AlphaSignal]:
        """Generate cross-sectional dispersion signals"""
        signals = []
        
        # Factor dispersion signal
        factor_dispersion = factor_model.factor_scores.std(axis=1)
        current_dispersion = factor_dispersion.iloc[-1] if len(factor_dispersion) > 0 else 0
        mean_dispersion = factor_dispersion.mean()
        
        if current_dispersion > mean_dispersion * 1.5:
            signal = AlphaSignal(
                signal_name="Factor_Dispersion",
                signal_value=current_dispersion / mean_dispersion,
                expected_return=0.05,  # 5% expected return in high dispersion
                confidence_interval=(0.02, 0.08),
                factor_exposures={'dispersion': 1.0},
                idiosyncratic_component=0.5,
                signal_volatility=factor_dispersion.std(),
                information_ratio=0.5,
                max_drawdown=0.1,
                entry_signal=True,
                exit_signal=False,
                holding_period=20
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return -drawdown.min()


class AutoEncoderDetector(nn.Module):
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(AutoEncoderDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
        
        self.decision_scores_ = None
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """Train autoencoder"""
        X_tensor = torch.FloatTensor(X)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            outputs = self(X_tensor)
            loss = criterion(outputs, X_tensor)
            
            loss.backward()
            optimizer.step()
        
        # Calculate decision scores (reconstruction error)
        self.eval()
        with torch.no_grad():
            outputs = self(X_tensor)
            mse = torch.mean((outputs - X_tensor) ** 2, dim=1)
            self.decision_scores_ = mse.numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        threshold = np.percentile(self.decision_scores_, 95)
        
        X_tensor = torch.FloatTensor(X)
        self.eval()
        
        with torch.no_grad():
            outputs = self(X_tensor)
            mse = torch.mean((outputs - X_tensor) ** 2, dim=1).numpy()
        
        predictions = np.where(mse > threshold, -1, 1)
        return predictions