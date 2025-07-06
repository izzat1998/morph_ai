"""
Statistical Analysis Reporting System

This module provides comprehensive reporting capabilities for statistical analysis results,
generating publication-ready reports and summaries.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from django.template.loader import render_to_string
from django.template import Template, Context

from .models import StatisticalAnalysis, FeatureStatistics, HypothesisTest, PopulationComparison
from .services.morphometric_integration import MorphometricStatisticalIntegrator


@dataclass
class AnalysisReport:
    """Container for complete analysis report data."""
    analysis_id: int
    cell_name: str
    timestamp: str
    summary: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    uncertainty_analysis: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComparisonReport:
    """Container for population comparison report data."""
    comparison_name: str
    timestamp: str
    analyses_compared: List[Dict[str, Any]]
    features_tested: List[str]
    significant_differences: List[Dict[str, Any]]
    effect_sizes: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class StatisticalReportGenerator:
    """
    Generates comprehensive statistical reports for morphometric analysis.
    
    This class creates publication-ready reports including:
    - Individual analysis statistical summaries
    - Population comparison reports
    - Validation reports
    - Quality assessment reports
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.integrator = MorphometricStatisticalIntegrator()
    
    def generate_analysis_report(self, statistical_analysis: StatisticalAnalysis) -> AnalysisReport:
        """
        Generate comprehensive report for a single statistical analysis.
        
        Args:
            statistical_analysis: StatisticalAnalysis instance
            
        Returns:
            AnalysisReport with complete statistical summary
        """
        # Get comprehensive statistical summary
        summary = self.integrator.get_statistical_summary(statistical_analysis)
        
        # Generate validation results
        validation = self.integrator.validate_statistical_analysis(statistical_analysis)
        
        # Extract key information
        analysis = statistical_analysis.analysis
        feature_stats = statistical_analysis.feature_stats.all()
        
        # Build confidence intervals summary
        confidence_intervals = {}
        uncertainty_analysis = {}
        
        for feature_stat in feature_stats:
            feature_name = feature_stat.feature_name
            
            confidence_intervals[feature_name] = {
                'point_estimate': feature_stat.measured_value,
                'lower_bound': feature_stat.confidence_interval_lower,
                'upper_bound': feature_stat.confidence_interval_upper,
                'width': feature_stat.confidence_interval_width,
                'confidence_level': statistical_analysis.confidence_level
            }
            
            uncertainty_analysis[feature_name] = {
                'absolute_uncertainty': feature_stat.uncertainty_absolute,
                'relative_uncertainty_percent': feature_stat.uncertainty_percent,
                'uncertainty_source': feature_stat.uncertainty_source,
                'reliability_score': feature_stat.measurement_reliability_score
            }
        
        # Quality assessment
        quality_assessment = {
            'overall_status': validation['overall_status'],
            'validation_score': validation['validation_score'],
            'computation_time': statistical_analysis.computation_time_seconds,
            'bootstrap_iterations': statistical_analysis.bootstrap_iterations,
            'total_features_analyzed': feature_stats.count()
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation, summary)
        
        return AnalysisReport(
            analysis_id=analysis.id,
            cell_name=analysis.cell.name,
            timestamp=datetime.now().isoformat(),
            summary=summary,
            confidence_intervals=confidence_intervals,
            uncertainty_analysis=uncertainty_analysis,
            quality_assessment=quality_assessment,
            recommendations=recommendations
        )
    
    def generate_comparison_report(self, population_comparison: PopulationComparison) -> ComparisonReport:
        """
        Generate comprehensive report for population comparison.
        
        Args:
            population_comparison: PopulationComparison instance
            
        Returns:
            ComparisonReport with complete comparison analysis
        """
        # Get all hypothesis tests for this comparison
        hypothesis_tests = HypothesisTest.objects.filter(
            test_date__gte=population_comparison.comparison_date,
            performed_by=population_comparison.created_by
        ).order_by('test_date')
        
        # Extract analyses compared
        analyses_compared = []
        analysis_ids = set()
        
        for test in hypothesis_tests:
            if test.analysis_1.id not in analysis_ids:
                analyses_compared.append({
                    'id': test.analysis_1.id,
                    'name': test.analysis_1.cell.name,
                    'model': test.analysis_1.cellpose_model,
                    'cells_detected': test.group1_n
                })
                analysis_ids.add(test.analysis_1.id)
            
            if test.analysis_2 and test.analysis_2.id not in analysis_ids:
                analyses_compared.append({
                    'id': test.analysis_2.id,
                    'name': test.analysis_2.cell.name,
                    'model': test.analysis_2.cellpose_model,
                    'cells_detected': test.group2_n
                })
                analysis_ids.add(test.analysis_2.id)
        
        # Extract features tested
        features_tested = list(set(test.feature_name for test in hypothesis_tests))
        
        # Identify significant differences
        significant_differences = []
        for test in hypothesis_tests:
            if test.is_significant:
                significant_differences.append({
                    'feature': test.feature_name,
                    'test_type': test.test_type,
                    'p_value': test.p_value,
                    'effect_size': test.effect_size,
                    'effect_size_type': test.effect_size_type,
                    'effect_interpretation': test.effect_size_interpretation,
                    'group1_mean': test.group1_mean,
                    'group2_mean': test.group2_mean,
                    'test_name': test.test_name
                })
        
        # Calculate effect sizes summary
        effect_sizes = {}
        for feature in features_tested:
            feature_tests = [t for t in hypothesis_tests if t.feature_name == feature]
            if feature_tests:
                effect_sizes[feature] = {
                    'mean_effect_size': np.mean([t.effect_size for t in feature_tests]),
                    'max_effect_size': max(t.effect_size for t in feature_tests),
                    'min_effect_size': min(t.effect_size for t in feature_tests),
                    'effect_size_type': feature_tests[0].effect_size_type
                }
        
        # Statistical summary
        statistical_summary = {
            'total_tests': len(hypothesis_tests),
            'significant_tests': len(significant_differences),
            'significance_rate': len(significant_differences) / len(hypothesis_tests) if hypothesis_tests else 0,
            'features_with_differences': len(set(diff['feature'] for diff in significant_differences)),
            'most_discriminating_feature': population_comparison.most_discriminating_feature,
            'alpha_level': hypothesis_tests[0].alpha_level if hypothesis_tests else 0.05
        }
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(
            statistical_summary, significant_differences, effect_sizes
        )
        
        return ComparisonReport(
            comparison_name=population_comparison.comparison_name,
            timestamp=population_comparison.comparison_date.isoformat(),
            analyses_compared=analyses_compared,
            features_tested=features_tested,
            significant_differences=significant_differences,
            effect_sizes=effect_sizes,
            statistical_summary=statistical_summary,
            recommendations=recommendations
        )
    
    def generate_validation_report(self, analyses: List[StatisticalAnalysis]) -> Dict[str, Any]:
        """
        Generate validation report for multiple statistical analyses.
        
        Args:
            analyses: List of StatisticalAnalysis instances
            
        Returns:
            Dictionary with validation summary
        """
        validation_results = []
        overall_scores = []
        
        for analysis in analyses:
            validation = self.integrator.validate_statistical_analysis(analysis)
            validation_results.append({
                'analysis_id': analysis.analysis.id,
                'cell_name': analysis.analysis.cell.name,
                'status': validation['overall_status'],
                'score': validation['validation_score'],
                'issues': validation['issues'],
                'recommendations': validation['recommendations']
            })
            overall_scores.append(validation['validation_score'])
        
        # Calculate overall statistics
        if overall_scores:
            mean_score = np.mean(overall_scores)
            min_score = np.min(overall_scores)
            max_score = np.max(overall_scores)
            
            # Count by status
            status_counts = {}
            for result in validation_results:
                status = result['status']
                status_counts[status] = status_counts.get(status, 0) + 1
        else:
            mean_score = min_score = max_score = 0
            status_counts = {}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_analyses': len(analyses),
            'validation_results': validation_results,
            'overall_statistics': {
                'mean_score': mean_score,
                'min_score': min_score,
                'max_score': max_score,
                'status_distribution': status_counts
            },
            'system_recommendations': self._generate_system_recommendations(validation_results)
        }
    
    def export_report_json(self, report: Any, filename: str) -> str:
        """
        Export report to JSON file.
        
        Args:
            report: Report object (AnalysisReport or ComparisonReport)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if hasattr(report, 'to_dict'):
            data = report.to_dict()
        else:
            data = report
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    def _generate_recommendations(self, validation: Dict, summary: Dict) -> List[str]:
        """Generate recommendations based on validation and summary."""
        recommendations = []
        
        # Based on validation score
        validation_score = validation.get('validation_score', 0)
        if validation_score < 0.6:
            recommendations.append("Consider increasing sample size for more reliable statistical estimates")
        
        if validation_score < 0.4:
            recommendations.append("Review measurement precision - high uncertainty detected")
        
        # Based on confidence intervals
        ci_data = summary.get('confidence_intervals', {})
        wide_intervals = []
        for feature, ci_info in ci_data.items():
            if 'relative_width_percent' in ci_info and ci_info['relative_width_percent'] > 30:
                wide_intervals.append(feature)
        
        if wide_intervals:
            recommendations.append(f"Confidence intervals are wide for: {', '.join(wide_intervals)}. "
                                 f"Consider increasing sample size.")
        
        # Based on quality assessment
        quality = summary.get('quality_assessment', {})
        if quality.get('high_quality_features', 0) < quality.get('features_with_narrow_cis', 0) / 2:
            recommendations.append("Consider improving image quality or calibration precision")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Statistical analysis quality is good. Results are reliable for publication.")
        
        return recommendations
    
    def _generate_comparison_recommendations(self, statistical_summary: Dict, 
                                          significant_differences: List, 
                                          effect_sizes: Dict) -> List[str]:
        """Generate recommendations for population comparisons."""
        recommendations = []
        
        significance_rate = statistical_summary.get('significance_rate', 0)
        
        if significance_rate == 0:
            recommendations.append("No significant differences detected. Consider increasing sample sizes "
                                 "or examining different morphometric features.")
        elif significance_rate < 0.1:
            recommendations.append("Few significant differences detected. Results suggest populations "
                                 "are morphologically similar.")
        elif significance_rate > 0.8:
            recommendations.append("Many significant differences detected. Populations appear "
                                 "morphologically distinct.")
            recommendations.append("Consider applying multiple comparison corrections if not already done.")
        
        # Effect size recommendations
        large_effects = [feature for feature, data in effect_sizes.items() 
                        if data.get('mean_effect_size', 0) > 0.8]
        
        if large_effects:
            recommendations.append(f"Large effect sizes detected for: {', '.join(large_effects)}. "
                                 f"These features show strong biological differences.")
        
        # Feature-specific recommendations
        if statistical_summary.get('most_discriminating_feature'):
            recommendations.append(f"Most discriminating feature: "
                                 f"{statistical_summary['most_discriminating_feature']}. "
                                 f"Focus on this feature for classification or characterization.")
        
        return recommendations
    
    def _generate_system_recommendations(self, validation_results: List) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Count issues across all analyses
        poor_analyses = [r for r in validation_results if r['score'] < 0.4]
        if len(poor_analyses) > len(validation_results) * 0.3:
            recommendations.append("Consider system-wide calibration improvements - "
                                 "many analyses show high uncertainty.")
        
        # Check for common issues
        common_issues = {}
        for result in validation_results:
            for issue in result.get('issues', []):
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        if common_issues:
            most_common = max(common_issues.items(), key=lambda x: x[1])
            if most_common[1] > len(validation_results) * 0.5:
                recommendations.append(f"Common issue detected: {most_common[0]}. "
                                     f"Consider addressing this systematically.")
        
        if not recommendations:
            recommendations.append("Overall system performance is good. "
                                 "Statistical framework is operating within expected parameters.")
        
        return recommendations