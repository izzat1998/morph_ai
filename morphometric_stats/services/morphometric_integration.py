"""
Morphometric Integration Service

This module integrates statistical analysis (including confidence intervals)
with the main morphometric analysis pipeline.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from django.db import transaction
from django.utils import timezone

from .confidence_intervals import ConfidenceIntervalCalculator, CIMethod
from .bootstrap_analysis import BootstrapEngine
from ..models import StatisticalAnalysis, FeatureStatistics

logger = logging.getLogger(__name__)


class MorphometricStatisticalIntegrator:
    """
    Integrates statistical analysis with morphometric feature extraction
    
    This class handles the integration of confidence intervals, bootstrap analysis,
    and uncertainty quantification with the main morphometric analysis pipeline.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 2000,
                 enable_bootstrap: bool = True,
                 enable_uncertainty_propagation: bool = True):
        """
        Initialize the statistical integrator
        
        Args:
            confidence_level: Confidence level for intervals (0.80-0.99)
            bootstrap_samples: Number of bootstrap samples
            enable_bootstrap: Enable bootstrap analysis
            enable_uncertainty_propagation: Enable uncertainty propagation
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.enable_bootstrap = enable_bootstrap
        self.enable_uncertainty_propagation = enable_uncertainty_propagation
        
        # Initialize calculators
        self.ci_calculator = ConfidenceIntervalCalculator(
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples
        )
        
        self.bootstrap_engine = BootstrapEngine(
            n_bootstrap=bootstrap_samples,
            confidence_level=confidence_level
        )
    
    def analyze_single_cell_with_statistics(self,
                                          cell_analysis,
                                          detected_cell,
                                          cell_mask: np.ndarray,
                                          feature_calculator: callable) -> Optional[StatisticalAnalysis]:
        """
        Perform statistical analysis for a single detected cell
        
        Args:
            cell_analysis: CellAnalysis instance
            detected_cell: DetectedCell instance
            cell_mask: Binary segmentation mask
            feature_calculator: Function to calculate morphometric features
            
        Returns:
            StatisticalAnalysis instance or None if analysis fails
        """
        try:
            with transaction.atomic():
                # Create or get statistical analysis
                statistical_analysis, created = StatisticalAnalysis.objects.get_or_create(
                    analysis=cell_analysis,
                    defaults={
                        'confidence_level': self.confidence_level,
                        'bootstrap_iterations': self.bootstrap_samples,
                        'include_confidence_intervals': True,
                        'include_uncertainty_propagation': self.enable_uncertainty_propagation,
                        'include_bootstrap_analysis': self.enable_bootstrap
                    }
                )
                
                start_time = timezone.now()
                
                # Calculate confidence intervals
                ci_result = self.ci_calculator.calculate_single_cell_cis(
                    cell_mask=cell_mask,
                    feature_calculator=feature_calculator,
                    method=CIMethod.AUTO
                )
                
                # Create FeatureStatistics instances
                feature_stats_list = self.ci_calculator.create_feature_statistics_from_cis(
                    statistical_analysis=statistical_analysis,
                    detected_cell=detected_cell,
                    ci_result=ci_result
                )
                
                # Save feature statistics
                FeatureStatistics.objects.bulk_create(feature_stats_list)
                
                # Update computation time
                computation_time = (timezone.now() - start_time).total_seconds()
                statistical_analysis.computation_time_seconds = computation_time
                statistical_analysis.save()
                
                logger.info(f"Statistical analysis completed for cell {detected_cell.id} in {computation_time:.2f}s")
                
                return statistical_analysis
                
        except Exception as e:
            logger.error(f"Failed to analyze cell {detected_cell.id} with statistics: {str(e)}")
            return None
    
    def analyze_population_with_statistics(self,
                                         cell_analysis,
                                         detected_cells: List,
                                         population_name: str = "default") -> Optional[StatisticalAnalysis]:
        """
        Perform population-level statistical analysis
        
        Args:
            cell_analysis: CellAnalysis instance
            detected_cells: List of DetectedCell instances
            population_name: Name identifier for this population
            
        Returns:
            StatisticalAnalysis instance or None if analysis fails
        """
        try:
            if len(detected_cells) < 3:
                logger.warning(f"Insufficient cells for population analysis: {len(detected_cells)}")
                return None
            
            with transaction.atomic():
                # Create statistical analysis for population
                statistical_analysis = StatisticalAnalysis.objects.create(
                    analysis=cell_analysis,
                    confidence_level=self.confidence_level,
                    bootstrap_iterations=self.bootstrap_samples,
                    include_confidence_intervals=True,
                    include_uncertainty_propagation=self.enable_uncertainty_propagation,
                    include_bootstrap_analysis=self.enable_bootstrap
                )
                
                start_time = timezone.now()
                
                # Extract features from all cells
                cell_features = []
                for cell in detected_cells:
                    cell_dict = self._extract_cell_features(cell)
                    if cell_dict:
                        cell_features.append(cell_dict)
                
                if not cell_features:
                    logger.warning("No valid cell features found for population analysis")
                    return None
                
                # Calculate population confidence intervals
                ci_result = self.ci_calculator.calculate_population_cis(
                    cell_features=cell_features,
                    method=CIMethod.AUTO
                )
                
                # Create summary feature statistics for population
                population_stats = self._create_population_feature_statistics(
                    statistical_analysis=statistical_analysis,
                    ci_result=ci_result,
                    cell_features=cell_features
                )
                
                # Save population statistics
                FeatureStatistics.objects.bulk_create(population_stats)
                
                # Update computation time
                computation_time = (timezone.now() - start_time).total_seconds()
                statistical_analysis.computation_time_seconds = computation_time
                statistical_analysis.save()
                
                logger.info(f"Population analysis completed for {len(detected_cells)} cells in {computation_time:.2f}s")
                
                return statistical_analysis
                
        except Exception as e:
            logger.error(f"Failed to analyze population with statistics: {str(e)}")
            return None
    
    def _extract_cell_features(self, detected_cell) -> Optional[Dict[str, float]]:
        """Extract morphometric features from a DetectedCell instance"""
        try:
            features = {}
            
            # Basic measurements
            if hasattr(detected_cell, 'area') and detected_cell.area is not None:
                features['area'] = float(detected_cell.area)
            if hasattr(detected_cell, 'perimeter') and detected_cell.perimeter is not None:
                features['perimeter'] = float(detected_cell.perimeter)
            
            # Shape descriptors
            if hasattr(detected_cell, 'circularity') and detected_cell.circularity is not None:
                features['circularity'] = float(detected_cell.circularity)
            if hasattr(detected_cell, 'eccentricity') and detected_cell.eccentricity is not None:
                features['eccentricity'] = float(detected_cell.eccentricity)
            if hasattr(detected_cell, 'solidity') and detected_cell.solidity is not None:
                features['solidity'] = float(detected_cell.solidity)
            if hasattr(detected_cell, 'extent') and detected_cell.extent is not None:
                features['extent'] = float(detected_cell.extent)
            
            # Ellipse fitting
            if hasattr(detected_cell, 'major_axis_length') and detected_cell.major_axis_length is not None:
                features['major_axis_length'] = float(detected_cell.major_axis_length)
            if hasattr(detected_cell, 'minor_axis_length') and detected_cell.minor_axis_length is not None:
                features['minor_axis_length'] = float(detected_cell.minor_axis_length)
            if hasattr(detected_cell, 'aspect_ratio') and detected_cell.aspect_ratio is not None:
                features['aspect_ratio'] = float(detected_cell.aspect_ratio)
            
            # Micron-based measurements if available
            if hasattr(detected_cell, 'area_microns_sq') and detected_cell.area_microns_sq is not None:
                features['area_microns_sq'] = float(detected_cell.area_microns_sq)
            if hasattr(detected_cell, 'perimeter_microns') and detected_cell.perimeter_microns is not None:
                features['perimeter_microns'] = float(detected_cell.perimeter_microns)
            
            return features if features else None
            
        except Exception as e:
            logger.warning(f"Failed to extract features from cell {detected_cell.id}: {str(e)}")
            return None
    
    def _create_population_feature_statistics(self,
                                            statistical_analysis: StatisticalAnalysis,
                                            ci_result,
                                            cell_features: List[Dict[str, float]]) -> List[FeatureStatistics]:
        """Create FeatureStatistics instances for population analysis"""
        
        feature_stats_list = []
        
        # Calculate population statistics for each feature
        feature_arrays = {}
        for cell_dict in cell_features:
            for feature_name, value in cell_dict.items():
                if feature_name not in feature_arrays:
                    feature_arrays[feature_name] = []
                if np.isfinite(value):
                    feature_arrays[feature_name].append(value)
        
        for feature_name, ci in ci_result.confidence_intervals.items():
            if feature_name in feature_arrays:
                values = np.array(feature_arrays[feature_name])
                
                # Calculate population statistics
                mean_value = np.mean(values)
                std_value = np.std(values, ddof=1)
                
                # Create a representative "population" detected cell entry
                # (This is a workaround since FeatureStatistics expects a DetectedCell)
                # In practice, you might want to create a separate PopulationStatistics model
                
                uncertainty_absolute = ci.standard_error if ci.standard_error else std_value / np.sqrt(len(values))
                uncertainty_percent = (uncertainty_absolute / abs(mean_value) * 100) if mean_value != 0 else 100
                
                # Get quality score
                reliability_score = ci_result.quality_assessment.get(feature_name, 0.5)
                
                # Note: This creates a population-level statistic
                # You might want to handle this differently based on your specific needs
                feature_stats = FeatureStatistics(
                    statistical_analysis=statistical_analysis,
                    detected_cell=None,  # Population-level statistic
                    feature_name=f"population_{feature_name}",
                    measured_value=mean_value,
                    mean_value=mean_value,
                    std_error=uncertainty_absolute,
                    confidence_interval_lower=ci.lower_bound,
                    confidence_interval_upper=ci.upper_bound,
                    confidence_interval_width=ci.interval_width or (ci.upper_bound - ci.lower_bound),
                    uncertainty_absolute=uncertainty_absolute,
                    uncertainty_percent=uncertainty_percent,
                    uncertainty_source=f"population_ci_{ci.method_used}",
                    measurement_reliability_score=reliability_score,
                    bootstrap_mean=mean_value,
                    bootstrap_std=std_value
                )
                
                feature_stats_list.append(feature_stats)
        
        return feature_stats_list
    
    def get_statistical_summary(self, statistical_analysis: StatisticalAnalysis) -> Dict[str, Any]:
        """
        Generate a comprehensive statistical summary
        
        Args:
            statistical_analysis: StatisticalAnalysis instance
            
        Returns:
            Dictionary with statistical summary information
        """
        try:
            feature_stats = statistical_analysis.feature_stats.all()
            
            if not feature_stats.exists():
                return {'error': 'No statistical data available'}
            
            summary = {
                'analysis_info': {
                    'confidence_level': statistical_analysis.confidence_level,
                    'bootstrap_iterations': statistical_analysis.bootstrap_iterations,
                    'computation_time_seconds': statistical_analysis.computation_time_seconds,
                    'total_features': feature_stats.count()
                },
                'confidence_intervals': {},
                'uncertainty_analysis': {},
                'quality_assessment': {}
            }
            
            # Process individual feature statistics
            for feature_stat in feature_stats:
                feature_name = feature_stat.feature_name
                
                # Confidence interval information
                summary['confidence_intervals'][feature_name] = {
                    'point_estimate': feature_stat.measured_value,
                    'lower_bound': feature_stat.confidence_interval_lower,
                    'upper_bound': feature_stat.confidence_interval_upper,
                    'width': feature_stat.confidence_interval_width,
                    'relative_width_percent': (feature_stat.confidence_interval_width / abs(feature_stat.measured_value) * 100) if feature_stat.measured_value != 0 else 0
                }
                
                # Uncertainty analysis
                summary['uncertainty_analysis'][feature_name] = {
                    'absolute_uncertainty': feature_stat.uncertainty_absolute,
                    'relative_uncertainty_percent': feature_stat.uncertainty_percent,
                    'uncertainty_source': feature_stat.uncertainty_source,
                    'standard_error': feature_stat.std_error
                }
                
                # Quality assessment
                summary['quality_assessment'][feature_name] = {
                    'reliability_score': feature_stat.measurement_reliability_score,
                    'is_within_tolerance': feature_stat.is_within_tolerance(),
                    'outlier_score': feature_stat.outlier_score
                }
            
            # Overall quality metrics
            reliability_scores = [fs.measurement_reliability_score for fs in feature_stats]
            relative_widths = []
            
            for feature_stat in feature_stats:
                if feature_stat.measured_value != 0:
                    rel_width = feature_stat.confidence_interval_width / abs(feature_stat.measured_value)
                    relative_widths.append(rel_width)
            
            summary['overall_quality'] = {
                'mean_reliability_score': np.mean(reliability_scores) if reliability_scores else 0,
                'mean_relative_ci_width': np.mean(relative_widths) if relative_widths else 0,
                'features_with_narrow_cis': sum(1 for w in relative_widths if w <= 0.1),
                'features_with_wide_cis': sum(1 for w in relative_widths if w > 0.3),
                'high_quality_features': sum(1 for score in reliability_scores if score >= 0.8)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate statistical summary: {str(e)}")
            return {'error': f'Failed to generate summary: {str(e)}'}
    
    def validate_statistical_analysis(self, statistical_analysis: StatisticalAnalysis) -> Dict[str, Any]:
        """
        Validate the quality of statistical analysis results
        
        Args:
            statistical_analysis: StatisticalAnalysis instance to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            feature_stats = statistical_analysis.feature_stats.all()
            
            validation_results = {
                'overall_status': 'unknown',
                'validation_score': 0.0,
                'issues': [],
                'recommendations': [],
                'feature_validation': {}
            }
            
            if not feature_stats.exists():
                validation_results['overall_status'] = 'failed'
                validation_results['issues'].append('No statistical data found')
                return validation_results
            
            issues = []
            recommendations = []
            feature_scores = []
            
            for feature_stat in feature_stats:
                feature_name = feature_stat.feature_name
                feature_validation = {
                    'reliability_score': feature_stat.measurement_reliability_score,
                    'issues': [],
                    'recommendations': []
                }
                
                # Check reliability score
                if feature_stat.measurement_reliability_score < 0.5:
                    feature_validation['issues'].append('Low reliability score')
                    feature_validation['recommendations'].append('Consider increasing sample size or improving measurement precision')
                
                # Check confidence interval width
                if feature_stat.measured_value != 0:
                    relative_width = feature_stat.confidence_interval_width / abs(feature_stat.measured_value)
                    if relative_width > 0.3:
                        feature_validation['issues'].append('Wide confidence interval')
                        feature_validation['recommendations'].append('Increase sample size to narrow confidence interval')
                
                # Check uncertainty level
                if feature_stat.uncertainty_percent > 20:
                    feature_validation['issues'].append('High measurement uncertainty')
                    feature_validation['recommendations'].append('Improve measurement precision or increase replication')
                
                validation_results['feature_validation'][feature_name] = feature_validation
                feature_scores.append(feature_stat.measurement_reliability_score)
            
            # Overall assessment
            mean_score = np.mean(feature_scores)
            validation_results['validation_score'] = mean_score
            
            if mean_score >= 0.8:
                validation_results['overall_status'] = 'excellent'
            elif mean_score >= 0.6:
                validation_results['overall_status'] = 'good'
            elif mean_score >= 0.4:
                validation_results['overall_status'] = 'acceptable'
            else:
                validation_results['overall_status'] = 'poor'
            
            # General recommendations
            if statistical_analysis.bootstrap_iterations < 1000:
                recommendations.append('Increase bootstrap iterations to at least 1000 for more reliable estimates')
            
            if len(feature_stats) < 5:
                recommendations.append('Consider analyzing additional morphometric features for comprehensive characterization')
            
            validation_results['recommendations'] = recommendations
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate statistical analysis: {str(e)}")
            return {
                'overall_status': 'error',
                'validation_score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'recommendations': ['Check analysis data integrity']
            }