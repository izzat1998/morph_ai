"""
Confidence Interval Calculation Service

This module implements multiple methods for calculating confidence intervals
for morphometric measurements, including bootstrap, parametric, and 
non-parametric approaches.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from enum import Enum

from .bootstrap_analysis import BootstrapEngine, BootstrapResult
from ..models import FeatureStatistics, StatisticalAnalysis


class CIMethod(Enum):
    """Available confidence interval methods"""
    BOOTSTRAP_PERCENTILE = "bootstrap_percentile"
    BOOTSTRAP_BCA = "bootstrap_bca" 
    PARAMETRIC_T = "parametric_t"
    PARAMETRIC_NORMAL = "parametric_normal"
    NONPARAMETRIC_WILCOXON = "nonparametric_wilcoxon"
    AUTO = "auto"


@dataclass
class ConfidenceInterval:
    """Single confidence interval result"""
    feature_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method_used: str
    standard_error: Optional[float] = None
    degrees_of_freedom: Optional[float] = None
    coverage_probability: Optional[float] = None
    interval_width: Optional[float] = None
    relative_width: Optional[float] = None


@dataclass
class CIAnalysisResult:
    """Complete confidence interval analysis result"""
    confidence_intervals: Dict[str, ConfidenceInterval]
    method_recommendations: Dict[str, str]
    quality_assessment: Dict[str, float]
    summary_statistics: Dict[str, Union[float, int]]
    warnings: List[str]


class ConfidenceIntervalCalculator:
    """
    Comprehensive confidence interval calculator for morphometric features
    
    Supports multiple methods and automatically selects the most appropriate
    method based on data characteristics.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 default_method: CIMethod = CIMethod.AUTO,
                 bootstrap_samples: int = 2000):
        """
        Initialize confidence interval calculator
        
        Args:
            confidence_level: Confidence level (0.80-0.99)
            default_method: Default method to use
            bootstrap_samples: Number of bootstrap samples for bootstrap methods
        """
        if not 0.8 <= confidence_level <= 0.99:
            raise ValueError("Confidence level must be between 0.8 and 0.99")
            
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.default_method = default_method
        self.bootstrap_samples = bootstrap_samples
        
        # Initialize bootstrap engine
        self.bootstrap_engine = BootstrapEngine(
            n_bootstrap=bootstrap_samples,
            confidence_level=confidence_level
        )
    
    def calculate_single_cell_cis(self,
                                 cell_mask: np.ndarray,
                                 feature_calculator: callable,
                                 method: CIMethod = None) -> CIAnalysisResult:
        """
        Calculate confidence intervals for single cell features
        
        Args:
            cell_mask: Binary segmentation mask
            feature_calculator: Function to calculate features from mask
            method: CI method to use (None for auto-selection)
            
        Returns:
            CIAnalysisResult with confidence intervals
        """
        if method is None:
            method = self.default_method
        
        # Get original feature values
        original_features = feature_calculator(cell_mask)
        
        confidence_intervals = {}
        method_recommendations = {}
        quality_assessment = {}
        warnings_list = []
        
        if method == CIMethod.AUTO:
            # Auto-select method for each feature
            for feature_name, value in original_features.items():
                best_method = self._select_best_method_single_cell(feature_name, value, cell_mask)
                method_recommendations[feature_name] = best_method.value
                
                ci = self._calculate_single_feature_ci(
                    feature_name, value, cell_mask, feature_calculator, best_method
                )
                confidence_intervals[feature_name] = ci
                quality_assessment[feature_name] = self._assess_ci_quality(ci)
        else:
            # Use specified method for all features
            for feature_name, value in original_features.items():
                method_recommendations[feature_name] = method.value
                
                ci = self._calculate_single_feature_ci(
                    feature_name, value, cell_mask, feature_calculator, method
                )
                confidence_intervals[feature_name] = ci
                quality_assessment[feature_name] = self._assess_ci_quality(ci)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(confidence_intervals)
        
        return CIAnalysisResult(
            confidence_intervals=confidence_intervals,
            method_recommendations=method_recommendations,
            quality_assessment=quality_assessment,
            summary_statistics=summary_stats,
            warnings=warnings_list
        )
    
    def calculate_population_cis(self,
                                cell_features: List[Dict[str, float]],
                                method: CIMethod = None,
                                statistic_func: callable = np.mean) -> CIAnalysisResult:
        """
        Calculate confidence intervals for population statistics
        
        Args:
            cell_features: List of feature dictionaries from multiple cells
            method: CI method to use (None for auto-selection)
            statistic_func: Population statistic function (mean, median, etc.)
            
        Returns:
            CIAnalysisResult with confidence intervals
        """
        if len(cell_features) < 3:
            raise ValueError("Need at least 3 cells for population confidence intervals")
        
        if method is None:
            method = self.default_method
        
        # Extract feature arrays
        feature_arrays = self._extract_feature_arrays(cell_features)
        
        confidence_intervals = {}
        method_recommendations = {}
        quality_assessment = {}
        warnings_list = []
        
        for feature_name, values in feature_arrays.items():
            if len(values) < 3:
                warnings_list.append(f"Insufficient data for {feature_name} (n={len(values)})")
                continue
            
            if method == CIMethod.AUTO:
                best_method = self._select_best_method_population(feature_name, values)
                method_recommendations[feature_name] = best_method.value
                
                ci = self._calculate_population_feature_ci(
                    feature_name, values, best_method, statistic_func
                )
            else:
                method_recommendations[feature_name] = method.value
                ci = self._calculate_population_feature_ci(
                    feature_name, values, method, statistic_func
                )
            
            confidence_intervals[feature_name] = ci
            quality_assessment[feature_name] = self._assess_ci_quality(ci)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(confidence_intervals)
        
        return CIAnalysisResult(
            confidence_intervals=confidence_intervals,
            method_recommendations=method_recommendations,
            quality_assessment=quality_assessment,
            summary_statistics=summary_stats,
            warnings=warnings_list
        )
    
    def _calculate_single_feature_ci(self,
                                   feature_name: str,
                                   original_value: float,
                                   cell_mask: np.ndarray,
                                   feature_calculator: callable,
                                   method: CIMethod) -> ConfidenceInterval:
        """Calculate CI for single cell feature using specified method"""
        
        if method in [CIMethod.BOOTSTRAP_PERCENTILE, CIMethod.BOOTSTRAP_BCA]:
            # Use bootstrap methods
            bootstrap_results = self.bootstrap_engine.bootstrap_single_cell_features(
                cell_mask, feature_calculator
            )
            
            if feature_name not in bootstrap_results:
                # Fallback to uncertainty-based CI
                return self._uncertainty_based_ci(feature_name, original_value)
            
            bootstrap_result = bootstrap_results[feature_name]
            
            if method == CIMethod.BOOTSTRAP_BCA:
                lower = bootstrap_result.confidence_interval_lower
                upper = bootstrap_result.confidence_interval_upper
                method_str = "bootstrap_bca"
            else:  # BOOTSTRAP_PERCENTILE
                alpha_lower = (self.alpha / 2) * 100
                alpha_upper = (1 - self.alpha / 2) * 100
                lower = np.percentile(bootstrap_result.bootstrap_distribution, alpha_lower)
                upper = np.percentile(bootstrap_result.bootstrap_distribution, alpha_upper)
                method_str = "bootstrap_percentile"
            
            return ConfidenceInterval(
                feature_name=feature_name,
                point_estimate=original_value,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=self.confidence_level,
                method_used=method_str,
                standard_error=bootstrap_result.bootstrap_std,
                interval_width=upper - lower,
                relative_width=(upper - lower) / abs(original_value) if original_value != 0 else float('inf')
            )
        
        else:
            # Use uncertainty-based approximation for single cells
            return self._uncertainty_based_ci(feature_name, original_value)
    
    def _calculate_population_feature_ci(self,
                                       feature_name: str,
                                       values: List[float],
                                       method: CIMethod,
                                       statistic_func: callable) -> ConfidenceInterval:
        """Calculate CI for population feature using specified method"""
        
        values_array = np.array(values)
        point_estimate = statistic_func(values_array)
        n = len(values_array)
        
        if method == CIMethod.PARAMETRIC_T:
            # t-distribution based CI
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            se = std_val / np.sqrt(n)
            
            t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
            margin_error = t_critical * se
            
            return ConfidenceInterval(
                feature_name=feature_name,
                point_estimate=point_estimate,
                lower_bound=mean_val - margin_error,
                upper_bound=mean_val + margin_error,
                confidence_level=self.confidence_level,
                method_used="parametric_t",
                standard_error=se,
                degrees_of_freedom=n-1,
                interval_width=2 * margin_error,
                relative_width=(2 * margin_error) / abs(mean_val) if mean_val != 0 else float('inf')
            )
        
        elif method == CIMethod.PARAMETRIC_NORMAL:
            # Normal distribution based CI
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            se = std_val / np.sqrt(n)
            
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            margin_error = z_critical * se
            
            return ConfidenceInterval(
                feature_name=feature_name,
                point_estimate=point_estimate,
                lower_bound=mean_val - margin_error,
                upper_bound=mean_val + margin_error,
                confidence_level=self.confidence_level,
                method_used="parametric_normal",
                standard_error=se,
                interval_width=2 * margin_error,
                relative_width=(2 * margin_error) / abs(mean_val) if mean_val != 0 else float('inf')
            )
        
        elif method == CIMethod.NONPARAMETRIC_WILCOXON:
            # Wilcoxon signed-rank based CI
            try:
                ci_lower, ci_upper = self._wilcoxon_ci(values_array, self.confidence_level)
                
                return ConfidenceInterval(
                    feature_name=feature_name,
                    point_estimate=point_estimate,
                    lower_bound=ci_lower,
                    upper_bound=ci_upper,
                    confidence_level=self.confidence_level,
                    method_used="nonparametric_wilcoxon",
                    interval_width=ci_upper - ci_lower,
                    relative_width=(ci_upper - ci_lower) / abs(point_estimate) if point_estimate != 0 else float('inf')
                )
            except:
                # Fallback to percentile method
                alpha_lower = (self.alpha / 2) * 100
                alpha_upper = (1 - self.alpha / 2) * 100
                lower = np.percentile(values_array, alpha_lower)
                upper = np.percentile(values_array, alpha_upper)
                
                return ConfidenceInterval(
                    feature_name=feature_name,
                    point_estimate=point_estimate,
                    lower_bound=lower,
                    upper_bound=upper,
                    confidence_level=self.confidence_level,
                    method_used="percentile_fallback",
                    interval_width=upper - lower,
                    relative_width=(upper - lower) / abs(point_estimate) if point_estimate != 0 else float('inf')
                )
        
        elif method in [CIMethod.BOOTSTRAP_PERCENTILE, CIMethod.BOOTSTRAP_BCA]:
            # Bootstrap methods for population
            cell_features = [{'temp_feature': val} for val in values]
            bootstrap_results = self.bootstrap_engine.bootstrap_population_features(
                cell_features, statistic_func
            )
            
            feature_key = 'temp_feature'
            if feature_key not in bootstrap_results:
                # Fallback to t-distribution
                return self._calculate_population_feature_ci(
                    feature_name, values, CIMethod.PARAMETRIC_T, statistic_func
                )
            
            bootstrap_result = bootstrap_results[feature_key]
            
            if method == CIMethod.BOOTSTRAP_BCA:
                lower = bootstrap_result.confidence_interval_lower
                upper = bootstrap_result.confidence_interval_upper
                method_str = "bootstrap_bca"
            else:  # BOOTSTRAP_PERCENTILE
                alpha_lower = (self.alpha / 2) * 100
                alpha_upper = (1 - self.alpha / 2) * 100
                lower = np.percentile(bootstrap_result.bootstrap_distribution, alpha_lower)
                upper = np.percentile(bootstrap_result.bootstrap_distribution, alpha_upper)
                method_str = "bootstrap_percentile"
            
            return ConfidenceInterval(
                feature_name=feature_name,
                point_estimate=point_estimate,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=self.confidence_level,
                method_used=method_str,
                standard_error=bootstrap_result.bootstrap_std,
                interval_width=upper - lower,
                relative_width=(upper - lower) / abs(point_estimate) if point_estimate != 0 else float('inf')
            )
        
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _select_best_method_single_cell(self, feature_name: str, value: float, mask: np.ndarray) -> CIMethod:
        """Select best CI method for single cell analysis"""
        # For single cells, bootstrap is generally preferred
        mask_size = np.sum(mask)
        
        if mask_size > 100:  # Sufficient pixels for bootstrap
            return CIMethod.BOOTSTRAP_BCA
        else:
            return CIMethod.BOOTSTRAP_PERCENTILE
    
    def _select_best_method_population(self, feature_name: str, values: List[float]) -> CIMethod:
        """Select best CI method for population analysis"""
        n = len(values)
        values_array = np.array(values)
        
        # Test normality
        if n >= 8:  # Minimum for meaningful normality test
            _, normality_p = stats.normaltest(values_array)
            is_normal = normality_p > 0.05
        else:
            is_normal = False
        
        # Check sample size
        if n >= 30:
            # Large sample
            if is_normal:
                return CIMethod.PARAMETRIC_NORMAL
            else:
                return CIMethod.BOOTSTRAP_BCA
        elif n >= 8:
            # Medium sample
            if is_normal:
                return CIMethod.PARAMETRIC_T
            else:
                return CIMethod.BOOTSTRAP_BCA
        else:
            # Small sample
            return CIMethod.NONPARAMETRIC_WILCOXON
    
    def _uncertainty_based_ci(self, feature_name: str, value: float) -> ConfidenceInterval:
        """Create CI based on measurement uncertainty estimation"""
        # Estimate uncertainty based on feature type and value
        if 'area' in feature_name.lower():
            relative_uncertainty = 0.05  # 5% for area measurements
        elif 'perimeter' in feature_name.lower():
            relative_uncertainty = 0.08  # 8% for perimeter measurements
        elif 'diameter' in feature_name.lower() or 'axis' in feature_name.lower():
            relative_uncertainty = 0.06  # 6% for length measurements
        else:
            relative_uncertainty = 0.10  # 10% for other features
        
        # Calculate uncertainty
        absolute_uncertainty = abs(value) * relative_uncertainty
        
        # Assume normal distribution for CI calculation
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        margin_error = z_critical * absolute_uncertainty
        
        return ConfidenceInterval(
            feature_name=feature_name,
            point_estimate=value,
            lower_bound=value - margin_error,
            upper_bound=value + margin_error,
            confidence_level=self.confidence_level,
            method_used="uncertainty_based",
            standard_error=absolute_uncertainty,
            interval_width=2 * margin_error,
            relative_width=(2 * margin_error) / abs(value) if value != 0 else float('inf')
        )
    
    def _wilcoxon_ci(self, values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate Wilcoxon signed-rank confidence interval"""
        n = len(values)
        if n < 6:
            raise ValueError("Need at least 6 values for Wilcoxon CI")
        
        # Calculate all pairwise averages
        pairwise_avgs = []
        for i in range(n):
            for j in range(i, n):
                pairwise_avgs.append((values[i] + values[j]) / 2)
        
        pairwise_avgs = np.array(sorted(pairwise_avgs))
        
        # Find critical value for Wilcoxon
        alpha = 1 - confidence_level
        k = int(np.floor(0.5 * (len(pairwise_avgs) - stats.norm.ppf(1 - alpha/2) * np.sqrt(n * (n + 1) / 6))))
        
        # Ensure valid bounds
        k = max(0, min(k, len(pairwise_avgs) - 1))
        
        lower_bound = pairwise_avgs[k]
        upper_bound = pairwise_avgs[-(k+1)]
        
        return lower_bound, upper_bound
    
    def _extract_feature_arrays(self, cell_features: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """Extract feature arrays from list of cell feature dictionaries"""
        feature_arrays = {}
        
        for cell_features_dict in cell_features:
            for feature_name, value in cell_features_dict.items():
                if feature_name not in feature_arrays:
                    feature_arrays[feature_name] = []
                if np.isfinite(value) and value is not None:
                    feature_arrays[feature_name].append(value)
        
        return feature_arrays
    
    def _assess_ci_quality(self, ci: ConfidenceInterval) -> float:
        """Assess quality of confidence interval (0-1 score)"""
        quality_factors = []
        
        # 1. Relative width assessment
        if ci.relative_width is not None and np.isfinite(ci.relative_width):
            if ci.relative_width <= 0.1:  # ≤10% relative width is excellent
                quality_factors.append(1.0)
            elif ci.relative_width <= 0.2:  # ≤20% is good
                quality_factors.append(0.8)
            elif ci.relative_width <= 0.5:  # ≤50% is acceptable
                quality_factors.append(0.5)
            else:  # >50% is poor
                quality_factors.append(0.2)
        else:
            quality_factors.append(0.5)  # Neutral if can't assess
        
        # 2. Method appropriateness
        method_scores = {
            'bootstrap_bca': 1.0,
            'bootstrap_percentile': 0.9,
            'parametric_t': 0.8,
            'parametric_normal': 0.7,
            'nonparametric_wilcoxon': 0.6,
            'uncertainty_based': 0.4,
            'percentile_fallback': 0.3
        }
        
        method_score = method_scores.get(ci.method_used, 0.5)
        quality_factors.append(method_score)
        
        # 3. Interval bounds reasonableness
        if ci.lower_bound < ci.upper_bound and np.isfinite(ci.lower_bound) and np.isfinite(ci.upper_bound):
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        return np.mean(quality_factors)
    
    def _calculate_summary_statistics(self, confidence_intervals: Dict[str, ConfidenceInterval]) -> Dict[str, Union[float, int]]:
        """Calculate summary statistics for CI analysis"""
        if not confidence_intervals:
            return {}
        
        relative_widths = [ci.relative_width for ci in confidence_intervals.values() 
                          if ci.relative_width is not None and np.isfinite(ci.relative_width)]
        
        methods_used = [ci.method_used for ci in confidence_intervals.values()]
        method_counts = {method: methods_used.count(method) for method in set(methods_used)}
        
        return {
            'total_features': len(confidence_intervals),
            'mean_relative_width': np.mean(relative_widths) if relative_widths else None,
            'median_relative_width': np.median(relative_widths) if relative_widths else None,
            'max_relative_width': np.max(relative_widths) if relative_widths else None,
            'narrow_intervals_count': sum(1 for w in relative_widths if w <= 0.1),
            'wide_intervals_count': sum(1 for w in relative_widths if w > 0.3),
            'method_distribution': method_counts,
            'confidence_level': self.confidence_level
        }
    
    def create_feature_statistics_from_cis(self,
                                         statistical_analysis: 'StatisticalAnalysis',
                                         detected_cell,
                                         ci_result: CIAnalysisResult) -> List['FeatureStatistics']:
        """
        Create FeatureStatistics instances from confidence interval analysis
        
        Args:
            statistical_analysis: StatisticalAnalysis instance
            detected_cell: DetectedCell instance
            ci_result: CIAnalysisResult instance
            
        Returns:
            List of FeatureStatistics instances
        """
        feature_stats_list = []
        
        for feature_name, ci in ci_result.confidence_intervals.items():
            # Calculate uncertainty metrics
            uncertainty_absolute = ci.standard_error if ci.standard_error else (ci.upper_bound - ci.lower_bound) / (2 * 1.96)
            uncertainty_percent = (uncertainty_absolute / abs(ci.point_estimate) * 100) if ci.point_estimate != 0 else 100
            
            # Get quality score
            reliability_score = ci_result.quality_assessment.get(feature_name, 0.5)
            
            feature_stats = FeatureStatistics(
                statistical_analysis=statistical_analysis,
                detected_cell=detected_cell,
                feature_name=feature_name,
                measured_value=ci.point_estimate,
                mean_value=ci.point_estimate,  # For single measurements
                std_error=uncertainty_absolute,
                confidence_interval_lower=ci.lower_bound,
                confidence_interval_upper=ci.upper_bound,
                confidence_interval_width=ci.interval_width or (ci.upper_bound - ci.lower_bound),
                uncertainty_absolute=uncertainty_absolute,
                uncertainty_percent=uncertainty_percent,
                uncertainty_source=f"ci_{ci.method_used}",
                measurement_reliability_score=reliability_score
            )
            
            feature_stats_list.append(feature_stats)
        
        return feature_stats_list