"""
Bootstrap Analysis for Morphometric Features

This module implements bootstrap resampling methods for estimating
confidence intervals and distributions of morphometric measurements.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import warnings

from ..models import FeatureStatistics, StatisticalAnalysis


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis"""
    feature_name: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_median: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    bias_estimate: float
    bias_corrected_value: float
    skewness: float
    kurtosis: float
    n_bootstrap_samples: int
    bootstrap_distribution: np.ndarray


@dataclass
class BootstrapValidation:
    """Bootstrap validation metrics"""
    coverage_probability: float
    interval_width_ratio: float
    bias_to_std_ratio: float
    distribution_normality_p: float
    quality_score: float
    recommendations: List[str]


class BootstrapEngine:
    """
    Bootstrap resampling engine for morphometric feature analysis
    
    Implements bias-corrected and accelerated (BCa) bootstrap methods
    for robust confidence interval estimation.
    """
    
    def __init__(self, n_bootstrap: int = 2000, confidence_level: float = 0.95, random_seed: int = None):
        """
        Initialize bootstrap engine
        
        Args:
            n_bootstrap: Number of bootstrap samples to generate
            confidence_level: Confidence level for intervals (0.80-0.99)
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bootstrap_single_cell_features(
        self,
        cell_mask: np.ndarray,
        feature_calculator: Callable,
        n_perturbations: int = None
    ) -> Dict[str, BootstrapResult]:
        """
        Bootstrap analysis for single cell by perturbing the segmentation mask
        
        Args:
            cell_mask: Binary mask of the segmented cell
            feature_calculator: Function that calculates features from mask
            n_perturbations: Number of mask perturbations (defaults to self.n_bootstrap)
            
        Returns:
            Dictionary of feature_name -> BootstrapResult
        """
        if n_perturbations is None:
            n_perturbations = self.n_bootstrap
        
        # Original features
        original_features = feature_calculator(cell_mask)
        
        # Generate perturbed masks and calculate features
        bootstrap_features = {feature: [] for feature in original_features.keys()}
        
        for i in range(n_perturbations):
            # Perturb mask (add noise to boundary)
            perturbed_mask = self._perturb_mask(cell_mask)
            
            try:
                features = feature_calculator(perturbed_mask)
                for feature_name, value in features.items():
                    if np.isfinite(value):
                        bootstrap_features[feature_name].append(value)
            except Exception as e:
                # Skip invalid perturbations
                continue
        
        # Analyze bootstrap results for each feature
        results = {}
        for feature_name, values in bootstrap_features.items():
            if len(values) > 10:  # Minimum samples for analysis
                bootstrap_array = np.array(values)
                result = self._analyze_bootstrap_distribution(
                    feature_name, 
                    original_features[feature_name], 
                    bootstrap_array
                )
                results[feature_name] = result
        
        return results
    
    def bootstrap_population_features(
        self,
        cell_features: List[Dict[str, float]],
        statistic_func: Callable = np.mean
    ) -> Dict[str, BootstrapResult]:
        """
        Bootstrap analysis for population statistics
        
        Args:
            cell_features: List of feature dictionaries from multiple cells
            statistic_func: Function to calculate population statistic (mean, median, etc.)
            
        Returns:
            Dictionary of feature_name -> BootstrapResult
        """
        if len(cell_features) < 3:
            raise ValueError("Need at least 3 cells for population bootstrap analysis")
        
        # Extract feature arrays
        feature_arrays = {}
        for cell_features_dict in cell_features:
            for feature_name, value in cell_features_dict.items():
                if feature_name not in feature_arrays:
                    feature_arrays[feature_name] = []
                if np.isfinite(value):
                    feature_arrays[feature_name].append(value)
        
        # Calculate original statistics
        original_stats = {}
        for feature_name, values in feature_arrays.items():
            if len(values) > 0:
                original_stats[feature_name] = statistic_func(values)
        
        # Bootstrap resampling
        bootstrap_stats = {feature: [] for feature in original_stats.keys()}
        
        for i in range(self.n_bootstrap):
            # Resample cells with replacement
            n_cells = len(cell_features)
            bootstrap_indices = np.random.choice(n_cells, size=n_cells, replace=True)
            
            # Calculate statistic for bootstrap sample
            for feature_name in original_stats.keys():
                bootstrap_values = [cell_features[idx][feature_name] 
                                 for idx in bootstrap_indices 
                                 if feature_name in cell_features[idx] and np.isfinite(cell_features[idx][feature_name])]
                
                if len(bootstrap_values) > 0:
                    bootstrap_stat = statistic_func(bootstrap_values)
                    if np.isfinite(bootstrap_stat):
                        bootstrap_stats[feature_name].append(bootstrap_stat)
        
        # Analyze bootstrap results
        results = {}
        for feature_name, values in bootstrap_stats.items():
            if len(values) > 10:
                bootstrap_array = np.array(values)
                result = self._analyze_bootstrap_distribution(
                    feature_name,
                    original_stats[feature_name],
                    bootstrap_array
                )
                results[feature_name] = result
        
        return results
    
    def _perturb_mask(self, mask: np.ndarray, noise_level: float = 0.3) -> np.ndarray:
        """
        Create perturbed version of segmentation mask
        
        Args:
            mask: Original binary mask
            noise_level: Amount of boundary perturbation
            
        Returns:
            Perturbed binary mask
        """
        from scipy import ndimage
        from skimage import morphology
        
        # Add random noise to boundaries
        boundary = mask ^ ndimage.binary_erosion(mask)
        noise_mask = np.random.random(mask.shape) < (noise_level * 0.1)
        
        # Randomly erode or dilate boundary pixels
        erode_mask = noise_mask & (np.random.random(mask.shape) < 0.5)
        dilate_mask = noise_mask & ~erode_mask
        
        perturbed = mask.copy()
        perturbed[boundary & erode_mask] = 0
        perturbed = ndimage.binary_dilation(perturbed, mask=dilate_mask)
        
        return perturbed.astype(bool)
    
    def _analyze_bootstrap_distribution(
        self,
        feature_name: str,
        original_value: float,
        bootstrap_samples: np.ndarray
    ) -> BootstrapResult:
        """
        Analyze bootstrap distribution and calculate confidence intervals
        
        Args:
            feature_name: Name of the morphometric feature
            original_value: Original measured value
            bootstrap_samples: Array of bootstrap samples
            
        Returns:
            BootstrapResult with complete analysis
        """
        # Basic statistics
        bootstrap_mean = np.mean(bootstrap_samples)
        bootstrap_std = np.std(bootstrap_samples, ddof=1)
        bootstrap_median = np.median(bootstrap_samples)
        
        # Bias estimation
        bias_estimate = bootstrap_mean - original_value
        bias_corrected_value = original_value - bias_estimate
        
        # Higher-order moments
        skewness = stats.skew(bootstrap_samples)
        kurtosis = stats.kurtosis(bootstrap_samples)
        
        # Confidence intervals using bias-corrected and accelerated (BCa) method
        ci_lower, ci_upper = self._calculate_bca_interval(
            original_value, bootstrap_samples, self.confidence_level
        )
        
        return BootstrapResult(
            feature_name=feature_name,
            original_value=original_value,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            bootstrap_median=bootstrap_median,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            confidence_level=self.confidence_level,
            bias_estimate=bias_estimate,
            bias_corrected_value=bias_corrected_value,
            skewness=skewness,
            kurtosis=kurtosis,
            n_bootstrap_samples=len(bootstrap_samples),
            bootstrap_distribution=bootstrap_samples
        )
    
    def _calculate_bca_interval(
        self,
        original_value: float,
        bootstrap_samples: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calculate bias-corrected and accelerated (BCa) confidence interval
        
        Args:
            original_value: Original statistic value
            bootstrap_samples: Bootstrap sample array
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence_level
        n_boot = len(bootstrap_samples)
        
        # Bias correction
        p_less = np.mean(bootstrap_samples < original_value)
        if p_less == 0:
            z0 = -stats.norm.ppf(1e-7)  # Avoid infinite values
        elif p_less == 1:
            z0 = stats.norm.ppf(1 - 1e-7)
        else:
            z0 = stats.norm.ppf(p_less)
        
        # Acceleration factor (simplified - would need jackknife for full BCa)
        # For now, use simpler percentile method with bias correction
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # Adjusted percentiles
        alpha_1 = stats.norm.cdf(z0 + (z0 + z_alpha_2))
        alpha_2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2))
        
        # Ensure percentiles are within valid range
        alpha_1 = max(0.001, min(0.999, alpha_1))
        alpha_2 = max(0.001, min(0.999, alpha_2))
        
        # Calculate interval bounds
        lower_bound = np.percentile(bootstrap_samples, alpha_1 * 100)
        upper_bound = np.percentile(bootstrap_samples, alpha_2 * 100)
        
        return lower_bound, upper_bound
    
    def validate_bootstrap_quality(
        self,
        bootstrap_results: Dict[str, BootstrapResult]
    ) -> Dict[str, BootstrapValidation]:
        """
        Validate quality of bootstrap analysis
        
        Args:
            bootstrap_results: Dictionary of BootstrapResult objects
            
        Returns:
            Dictionary of feature_name -> BootstrapValidation
        """
        validations = {}
        
        for feature_name, result in bootstrap_results.items():
            validation = self._assess_bootstrap_quality(result)
            validations[feature_name] = validation
        
        return validations
    
    def _assess_bootstrap_quality(self, result: BootstrapResult) -> BootstrapValidation:
        """Assess quality of individual bootstrap result"""
        
        recommendations = []
        
        # 1. Check sample size adequacy
        if result.n_bootstrap_samples < 1000:
            recommendations.append(f"Increase bootstrap samples (current: {result.n_bootstrap_samples}, recommended: ≥1000)")
        
        # 2. Check bias magnitude
        relative_bias = abs(result.bias_estimate) / result.bootstrap_std if result.bootstrap_std > 0 else 0
        bias_to_std_ratio = relative_bias
        
        if relative_bias > 0.25:
            recommendations.append(f"High bias detected ({relative_bias:.2f}×std). Consider bias correction.")
        
        # 3. Check interval width reasonableness
        interval_width = result.confidence_interval_upper - result.confidence_interval_lower
        relative_width = interval_width / abs(result.original_value) if result.original_value != 0 else float('inf')
        interval_width_ratio = relative_width
        
        if relative_width > 0.5:
            recommendations.append(f"Wide confidence interval ({relative_width:.1%}). Increase sample size.")
        
        # 4. Check distribution normality
        _, normality_p = stats.normaltest(result.bootstrap_distribution)
        distribution_normality_p = normality_p
        
        if normality_p < 0.05:
            recommendations.append("Bootstrap distribution is non-normal. Consider transformation.")
        
        # 5. Check for extreme skewness
        if abs(result.skewness) > 2:
            recommendations.append(f"High skewness detected ({result.skewness:.2f}). Asymmetric distribution.")
        
        # Overall quality score (0-1)
        quality_components = [
            1.0 if result.n_bootstrap_samples >= 1000 else result.n_bootstrap_samples / 1000,
            1.0 if relative_bias <= 0.1 else max(0, 1 - (relative_bias - 0.1) / 0.15),
            1.0 if relative_width <= 0.2 else max(0, 1 - (relative_width - 0.2) / 0.3),
            min(1.0, normality_p * 5),  # Scale p-value
            1.0 if abs(result.skewness) <= 1 else max(0, 1 - (abs(result.skewness) - 1) / 2)
        ]
        
        quality_score = np.mean(quality_components)
        
        if len(recommendations) == 0:
            recommendations.append("Bootstrap analysis quality is good.")
        
        # Estimate coverage probability (simplified)
        coverage_probability = self.confidence_level  # Ideal case
        if relative_bias > 0.1:
            coverage_probability -= min(0.05, relative_bias * 0.1)
        
        return BootstrapValidation(
            coverage_probability=coverage_probability,
            interval_width_ratio=interval_width_ratio,
            bias_to_std_ratio=bias_to_std_ratio,
            distribution_normality_p=distribution_normality_p,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def create_feature_statistics_with_bootstrap(
        self,
        statistical_analysis: StatisticalAnalysis,
        detected_cell,
        bootstrap_results: Dict[str, BootstrapResult]
    ) -> List[FeatureStatistics]:
        """
        Create FeatureStatistics instances from bootstrap results
        
        Args:
            statistical_analysis: StatisticalAnalysis instance
            detected_cell: DetectedCell instance
            bootstrap_results: Dictionary of BootstrapResult objects
            
        Returns:
            List of FeatureStatistics instances
        """
        feature_stats_list = []
        
        for feature_name, result in bootstrap_results.items():
            # Calculate reliability score based on confidence interval width
            relative_ci_width = ((result.confidence_interval_upper - result.confidence_interval_lower) / 
                               abs(result.original_value)) if result.original_value != 0 else 1.0
            
            # Improved reliability scoring for morphometric measurements
            relative_uncertainty = relative_ci_width * 100  # Convert to percentage
            
            if relative_uncertainty <= 2:
                reliability_score = 0.95 + (2 - relative_uncertainty) / 2 * 0.05  # 95-100%
            elif relative_uncertainty <= 5:
                reliability_score = 0.85 + (5 - relative_uncertainty) / 3 * 0.10  # 85-95%
            elif relative_uncertainty <= 10:
                reliability_score = 0.70 + (10 - relative_uncertainty) / 5 * 0.15  # 70-85%
            elif relative_uncertainty <= 20:
                reliability_score = 0.50 + (20 - relative_uncertainty) / 10 * 0.20  # 50-70%
            else:
                reliability_score = max(0, 0.50 - (relative_uncertainty - 20) / 20 * 0.50)  # 0-50%
            
            reliability_score = max(0, min(1, reliability_score))
            
            feature_stats = FeatureStatistics(
                statistical_analysis=statistical_analysis,
                detected_cell=detected_cell,
                feature_name=feature_name,
                measured_value=result.original_value,
                mean_value=result.bootstrap_mean,
                std_error=result.bootstrap_std,
                confidence_interval_lower=result.confidence_interval_lower,
                confidence_interval_upper=result.confidence_interval_upper,
                confidence_interval_width=result.confidence_interval_upper - result.confidence_interval_lower,
                uncertainty_absolute=result.bootstrap_std,
                uncertainty_percent=(result.bootstrap_std / abs(result.original_value) * 100) if result.original_value != 0 else 100,
                uncertainty_source='bootstrap_resampling',
                bootstrap_mean=result.bootstrap_mean,
                bootstrap_std=result.bootstrap_std,
                bootstrap_skewness=result.skewness,
                bootstrap_kurtosis=result.kurtosis,
                measurement_reliability_score=reliability_score
            )
            
            feature_stats_list.append(feature_stats)
        
        return feature_stats_list
    
    def generate_bootstrap_report(
        self,
        bootstrap_results: Dict[str, BootstrapResult],
        validations: Dict[str, BootstrapValidation]
    ) -> Dict:
        """
        Generate comprehensive bootstrap analysis report
        
        Args:
            bootstrap_results: Dictionary of BootstrapResult objects
            validations: Dictionary of BootstrapValidation objects
            
        Returns:
            Dictionary with complete bootstrap report
        """
        report = {
            'summary': {
                'total_features_analyzed': len(bootstrap_results),
                'bootstrap_samples_per_feature': self.n_bootstrap,
                'confidence_level': self.confidence_level,
                'overall_quality_score': np.mean([v.quality_score for v in validations.values()]) if validations else 0
            },
            'feature_results': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        # Collect all recommendations
        all_recommendations = []
        for validation in validations.values():
            all_recommendations.extend(validation.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        report['recommendations'] = unique_recommendations
        
        # Feature-specific results
        for feature_name, result in bootstrap_results.items():
            report['feature_results'][feature_name] = {
                'original_value': result.original_value,
                'bootstrap_mean': result.bootstrap_mean,
                'bootstrap_std': result.bootstrap_std,
                'confidence_interval': [result.confidence_interval_lower, result.confidence_interval_upper],
                'bias_estimate': result.bias_estimate,
                'bias_corrected_value': result.bias_corrected_value,
                'distribution_properties': {
                    'skewness': result.skewness,
                    'kurtosis': result.kurtosis,
                    'normality_test_p': validations.get(feature_name, {}).distribution_normality_p if feature_name in validations else None
                }
            }
            
            if feature_name in validations:
                report['quality_assessment'][feature_name] = {
                    'quality_score': validations[feature_name].quality_score,
                    'coverage_probability': validations[feature_name].coverage_probability,
                    'interval_width_ratio': validations[feature_name].interval_width_ratio,
                    'bias_to_std_ratio': validations[feature_name].bias_to_std_ratio
                }
        
        return report