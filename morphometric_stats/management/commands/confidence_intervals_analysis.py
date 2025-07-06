"""
Management command for confidence intervals analysis

This command demonstrates the complete Week 6 confidence intervals framework
by running statistical analysis on cell data with confidence intervals.
"""

import os
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List

from cells.models import CellAnalysis, DetectedCell
from morphometric_stats.models import StatisticalAnalysis, FeatureStatistics
from morphometric_stats.services.confidence_intervals import ConfidenceIntervalCalculator, CIMethod
from morphometric_stats.services.morphometric_integration import MorphometricStatisticalIntegrator
from morphometric_stats.services.ci_visualization import CIVisualizer


class Command(BaseCommand):
    help = 'Run confidence intervals analysis on morphometric data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--analysis-id',
            type=int,
            help='Specific CellAnalysis ID to analyze'
        )
        parser.add_argument(
            '--confidence-level',
            type=float,
            default=0.95,
            help='Confidence level (default: 0.95)'
        )
        parser.add_argument(
            '--bootstrap-samples',
            type=int,
            default=2000,
            help='Number of bootstrap samples (default: 2000)'
        )
        parser.add_argument(
            '--method',
            choices=['auto', 'bootstrap_bca', 'bootstrap_percentile', 'parametric_t', 'parametric_normal'],
            default='auto',
            help='CI method to use (default: auto)'
        )
        parser.add_argument(
            '--create-visualizations',
            action='store_true',
            help='Create visualization plots'
        )
        parser.add_argument(
            '--output-dir',
            default='statistical_analysis_output',
            help='Output directory for results (default: statistical_analysis_output)'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting Confidence Intervals Analysis'))
        
        # Parse options
        analysis_id = options.get('analysis_id')
        confidence_level = options['confidence_level']
        bootstrap_samples = options['bootstrap_samples']
        method_str = options['method']
        create_viz = options['create_visualizations']
        output_dir = options['output_dir']
        
        # Validate confidence level
        if not 0.8 <= confidence_level <= 0.99:
            raise CommandError('Confidence level must be between 0.8 and 0.99')
        
        # Convert method string to enum
        method_map = {
            'auto': CIMethod.AUTO,
            'bootstrap_bca': CIMethod.BOOTSTRAP_BCA,
            'bootstrap_percentile': CIMethod.BOOTSTRAP_PERCENTILE,
            'parametric_t': CIMethod.PARAMETRIC_T,
            'parametric_normal': CIMethod.PARAMETRIC_NORMAL
        }
        method = method_map[method_str]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize integrator
        integrator = MorphometricStatisticalIntegrator(
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            enable_bootstrap=True,
            enable_uncertainty_propagation=True
        )
        
        if analysis_id:
            # Analyze specific analysis
            self._analyze_single_analysis(analysis_id, integrator, method, create_viz, output_dir)
        else:
            # Analyze recent completed analyses
            self._analyze_recent_analyses(integrator, method, create_viz, output_dir)
        
        self.stdout.write(self.style.SUCCESS('✅ Confidence Intervals Analysis Complete'))
    
    def _analyze_single_analysis(self, 
                                analysis_id: int, 
                                integrator: MorphometricStatisticalIntegrator,
                                method: CIMethod,
                                create_viz: bool,
                                output_dir: str):
        """Analyze a single CellAnalysis with confidence intervals"""
        
        try:
            analysis = CellAnalysis.objects.get(id=analysis_id)
        except CellAnalysis.DoesNotExist:
            raise CommandError(f'CellAnalysis with ID {analysis_id} not found')
        
        if analysis.status != 'completed':
            raise CommandError(f'Analysis {analysis_id} is not completed (status: {analysis.status})')
        
        detected_cells = analysis.detected_cells.all()
        if not detected_cells.exists():
            raise CommandError(f'No detected cells found for analysis {analysis_id}')
        
        self.stdout.write(f'📊 Analyzing CellAnalysis {analysis_id} with {detected_cells.count()} cells')
        
        # Perform population-level statistical analysis
        statistical_analysis = integrator.analyze_population_with_statistics(
            cell_analysis=analysis,
            detected_cells=list(detected_cells),
            population_name=f"analysis_{analysis_id}"
        )
        
        if not statistical_analysis:
            self.stdout.write(self.style.ERROR(f'❌ Failed to create statistical analysis for {analysis_id}'))
            return
        
        # Get statistical summary
        summary = integrator.get_statistical_summary(statistical_analysis)
        self._display_summary(summary, analysis_id)
        
        # Validate analysis quality
        validation = integrator.validate_statistical_analysis(statistical_analysis)
        self._display_validation(validation)
        
        # Create visualizations if requested
        if create_viz:
            self._create_visualizations(statistical_analysis, analysis_id, output_dir)
        
        # Save detailed results
        self._save_detailed_results(statistical_analysis, summary, validation, analysis_id, output_dir)
    
    def _analyze_recent_analyses(self,
                               integrator: MorphometricStatisticalIntegrator,
                               method: CIMethod,
                               create_viz: bool,
                               output_dir: str):
        """Analyze recent completed analyses"""
        
        # Get recent completed analyses
        recent_analyses = CellAnalysis.objects.filter(
            status='completed'
        ).exclude(
            detected_cells__isnull=True
        ).order_by('-analysis_date')[:5]
        
        if not recent_analyses.exists():
            raise CommandError('No completed analyses with detected cells found')
        
        self.stdout.write(f'📊 Found {recent_analyses.count()} recent analyses to process')
        
        for analysis in recent_analyses:
            self.stdout.write(f'\n🔍 Processing Analysis {analysis.id}...')
            
            detected_cells = analysis.detected_cells.all()
            if detected_cells.count() < 3:
                self.stdout.write(f'⚠️ Skipping analysis {analysis.id} - insufficient cells ({detected_cells.count()})')
                continue
            
            # Perform population-level analysis
            statistical_analysis = integrator.analyze_population_with_statistics(
                cell_analysis=analysis,
                detected_cells=list(detected_cells),
                population_name=f"analysis_{analysis.id}"
            )
            
            if statistical_analysis:
                summary = integrator.get_statistical_summary(statistical_analysis)
                self._display_summary(summary, analysis.id)
                
                if create_viz:
                    self._create_visualizations(statistical_analysis, analysis.id, output_dir)
            else:
                self.stdout.write(f'❌ Failed to analyze {analysis.id}')
    
    def _display_summary(self, summary: Dict, analysis_id: int):
        """Display statistical summary in console"""
        
        self.stdout.write(f'\n📋 STATISTICAL SUMMARY - Analysis {analysis_id}')
        self.stdout.write('=' * 50)
        
        if 'error' in summary:
            self.stdout.write(self.style.ERROR(f'❌ Error: {summary["error"]}'))
            return
        
        # Analysis info
        info = summary.get('analysis_info', {})
        self.stdout.write(f"🔬 Confidence Level: {info.get('confidence_level', 0)*100:.0f}%")
        self.stdout.write(f"🔄 Bootstrap Iterations: {info.get('bootstrap_iterations', 0):,}")
        self.stdout.write(f"⏱️ Computation Time: {info.get('computation_time_seconds', 0):.2f}s")
        self.stdout.write(f"📊 Total Features: {info.get('total_features', 0)}")
        
        # Confidence intervals
        ci_data = summary.get('confidence_intervals', {})
        if ci_data:
            self.stdout.write(f'\n📏 CONFIDENCE INTERVALS:')
            for feature, ci_info in ci_data.items():
                feature_display = feature.replace('population_', '').replace('_', ' ').title()
                point_est = ci_info['point_estimate']
                lower = ci_info['lower_bound']
                upper = ci_info['upper_bound']
                rel_width = ci_info['relative_width_percent']
                
                self.stdout.write(
                    f"  {feature_display:20s}: {point_est:8.3f} [{lower:8.3f}, {upper:8.3f}] "
                    f"(±{rel_width:5.1f}%)"
                )
        
        # Overall quality
        quality = summary.get('overall_quality', {})
        if quality:
            self.stdout.write(f'\n⭐ QUALITY ASSESSMENT:')
            self.stdout.write(f"  Mean Reliability: {quality.get('mean_reliability_score', 0):.3f}")
            self.stdout.write(f"  Mean CI Width: {quality.get('mean_relative_ci_width', 0)*100:.1f}%")
            self.stdout.write(f"  High Quality Features: {quality.get('high_quality_features', 0)}")
            self.stdout.write(f"  Narrow CIs (≤10%): {quality.get('features_with_narrow_cis', 0)}")
    
    def _display_validation(self, validation: Dict):
        """Display validation results"""
        
        self.stdout.write(f'\n🔍 VALIDATION RESULTS')
        self.stdout.write('-' * 30)
        
        status = validation.get('overall_status', 'unknown')
        score = validation.get('validation_score', 0)
        
        status_symbols = {
            'excellent': '🌟',
            'good': '✅',
            'acceptable': '⚠️',
            'poor': '❌',
            'error': '💥'
        }
        
        symbol = status_symbols.get(status, '❓')
        self.stdout.write(f"{symbol} Overall Status: {status.upper()}")
        self.stdout.write(f"📊 Validation Score: {score:.3f}")
        
        # Display issues and recommendations
        issues = validation.get('issues', [])
        if issues:
            self.stdout.write(f"\n⚠️ Issues Found:")
            for issue in issues:
                self.stdout.write(f"  • {issue}")
        
        recommendations = validation.get('recommendations', [])
        if recommendations:
            self.stdout.write(f"\n💡 Recommendations:")
            for rec in recommendations:
                self.stdout.write(f"  • {rec}")
    
    def _create_visualizations(self, 
                             statistical_analysis: StatisticalAnalysis,
                             analysis_id: int,
                             output_dir: str):
        """Create visualization plots"""
        
        self.stdout.write(f'🎨 Creating visualizations for analysis {analysis_id}...')
        
        try:
            visualizer = CIVisualizer()
            
            # Create statistical dashboard
            dashboard_path = os.path.join(output_dir, f'statistical_dashboard_{analysis_id}.png')
            dashboard_file = visualizer.create_statistical_dashboard(
                statistical_analysis=statistical_analysis,
                save_path=dashboard_path
            )
            
            if dashboard_file:
                self.stdout.write(f'📊 Dashboard saved: {dashboard_file}')
            else:
                self.stdout.write(self.style.WARNING('⚠️ Failed to create dashboard'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Visualization error: {str(e)}'))
    
    def _save_detailed_results(self,
                             statistical_analysis: StatisticalAnalysis,
                             summary: Dict,
                             validation: Dict,
                             analysis_id: int,
                             output_dir: str):
        """Save detailed results to files"""
        
        import json
        from datetime import datetime
        
        # Prepare comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_id': analysis_id,
            'statistical_analysis_id': statistical_analysis.id,
            'summary': summary,
            'validation': validation,
            'feature_details': {}
        }
        
        # Add individual feature details
        feature_stats = statistical_analysis.feature_stats.all()
        for fs in feature_stats:
            results['feature_details'][fs.feature_name] = {
                'measured_value': fs.measured_value,
                'confidence_interval': {
                    'lower': fs.confidence_interval_lower,
                    'upper': fs.confidence_interval_upper,
                    'width': fs.confidence_interval_width
                },
                'uncertainty': {
                    'absolute': fs.uncertainty_absolute,
                    'percent': fs.uncertainty_percent,
                    'source': fs.uncertainty_source
                },
                'quality': {
                    'reliability_score': fs.measurement_reliability_score,
                    'within_tolerance': fs.is_within_tolerance()
                }
            }
        
        # Save to JSON file
        results_file = os.path.join(output_dir, f'confidence_intervals_analysis_{analysis_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.stdout.write(f'💾 Detailed results saved: {results_file}')
    
    def _extract_features_from_detected_cells(self, detected_cells) -> List[Dict[str, float]]:
        """Extract feature dictionaries from detected cells"""
        
        cell_features = []
        
        for cell in detected_cells:
            features = {}
            
            # Basic measurements
            if hasattr(cell, 'area') and cell.area is not None:
                features['area'] = float(cell.area)
            if hasattr(cell, 'perimeter') and cell.perimeter is not None:
                features['perimeter'] = float(cell.perimeter)
            
            # Shape descriptors
            if hasattr(cell, 'circularity') and cell.circularity is not None:
                features['circularity'] = float(cell.circularity)
            if hasattr(cell, 'eccentricity') and cell.eccentricity is not None:
                features['eccentricity'] = float(cell.eccentricity)
            if hasattr(cell, 'solidity') and cell.solidity is not None:
                features['solidity'] = float(cell.solidity)
            if hasattr(cell, 'extent') and cell.extent is not None:
                features['extent'] = float(cell.extent)
            
            # Only add if we have some features
            if features:
                cell_features.append(features)
        
        return cell_features