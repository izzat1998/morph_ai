"""
Django management command for statistical analysis of morphometric data

Usage:
    python manage.py statistical_analysis --analysis-id 123
    python manage.py statistical_analysis --compare-groups 123 124
    python manage.py statistical_analysis --all-analyses --bootstrap
"""

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from pathlib import Path
import json
from datetime import datetime

from morphometric_stats.models import StatisticalAnalysis, HypothesisTest, PopulationComparison
from morphometric_stats.services.bootstrap_analysis import BootstrapEngine
from morphometric_stats.services.hypothesis_testing import HypothesisTestingEngine
from morphometric_stats.services.uncertainty_propagation import UncertaintyPropagationEngine
from cells.models import CellAnalysis, DetectedCell

User = get_user_model()


class Command(BaseCommand):
    help = 'Perform statistical analysis on morphometric data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--analysis-id',
            type=int,
            help='ID of specific CellAnalysis to analyze'
        )
        
        parser.add_argument(
            '--compare-groups',
            nargs=2,
            type=int,
            metavar=('ID1', 'ID2'),
            help='Compare two CellAnalysis groups'
        )
        
        parser.add_argument(
            '--all-analyses',
            action='store_true',
            help='Analyze all CellAnalysis instances'
        )
        
        parser.add_argument(
            '--bootstrap',
            action='store_true',
            help='Include bootstrap analysis'
        )
        
        parser.add_argument(
            '--confidence-level',
            type=float,
            default=0.95,
            help='Confidence level for statistical analysis (default: 0.95)'
        )
        
        parser.add_argument(
            '--bootstrap-iterations',
            type=int,
            default=2000,
            help='Number of bootstrap iterations (default: 2000)'
        )
        
        parser.add_argument(
            '--feature',
            type=str,
            default='area',
            choices=['area', 'perimeter', 'circularity', 'eccentricity', 'aspect_ratio'],
            help='Feature to analyze for group comparisons'
        )
        
        parser.add_argument(
            '--output-dir',
            type=str,
            default='statistical_reports',
            help='Directory to save statistical reports'
        )
        
        parser.add_argument(
            '--user',
            type=str,
            help='Username to associate with statistical analyses'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('📊 Starting Statistical Analysis')
        )
        
        # Get user
        if options['user']:
            try:
                user = User.objects.get(username=options['user'])
            except User.DoesNotExist:
                raise CommandError(f"User '{options['user']}' does not exist")
        else:
            user = User.objects.filter(is_superuser=True).first()
            if not user:
                raise CommandError("No superuser found. Create a superuser first or specify --user")
        
        # Create output directory
        output_dir = Path(options['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engines
        bootstrap_engine = BootstrapEngine(
            n_bootstrap=options['bootstrap_iterations'],
            confidence_level=options['confidence_level']
        )
        hypothesis_engine = HypothesisTestingEngine()
        uncertainty_engine = UncertaintyPropagationEngine(confidence_level=options['confidence_level'])
        
        # Determine operation mode
        if options['analysis_id']:
            self._analyze_single_population(
                options['analysis_id'], user, bootstrap_engine, 
                uncertainty_engine, output_dir, options
            )
        elif options['compare_groups']:
            self._compare_two_groups(
                options['compare_groups'][0], options['compare_groups'][1],
                user, hypothesis_engine, bootstrap_engine, output_dir, options
            )
        elif options['all_analyses']:
            self._analyze_all_populations(
                user, bootstrap_engine, uncertainty_engine, output_dir, options
            )
        else:
            raise CommandError("Specify --analysis-id, --compare-groups, or --all-analyses")

    def _analyze_single_population(self, analysis_id, user, bootstrap_engine, uncertainty_engine, output_dir, options):
        """Analyze single population with bootstrap and uncertainty analysis"""
        
        try:
            analysis = CellAnalysis.objects.get(id=analysis_id)
        except CellAnalysis.DoesNotExist:
            raise CommandError(f"CellAnalysis {analysis_id} not found")
        
        cells = analysis.detected_cells.all()
        if not cells.exists():
            raise CommandError(f"No detected cells found for analysis {analysis_id}")
        
        self.stdout.write(f"🔄 Analyzing population: Analysis {analysis_id} ({cells.count()} cells)")
        
        # Create StatisticalAnalysis record
        stat_analysis, created = StatisticalAnalysis.objects.get_or_create(
            analysis=analysis,
            defaults={
                'confidence_level': options['confidence_level'],
                'bootstrap_iterations': options['bootstrap_iterations'],
                'include_bootstrap_analysis': options['bootstrap']
            }
        )
        
        if created:
            self.stdout.write("✅ Created statistical analysis record")
        
        # Extract cell features for bootstrap analysis
        cell_features = []
        for cell in cells:
            features = {
                'area': cell.area,
                'perimeter': cell.perimeter,
            }
            
            # Add optional features if available
            if hasattr(cell, 'circularity') and cell.circularity is not None:
                features['circularity'] = cell.circularity
            if hasattr(cell, 'eccentricity') and cell.eccentricity is not None:
                features['eccentricity'] = cell.eccentricity
            if hasattr(cell, 'aspect_ratio') and cell.aspect_ratio is not None:
                features['aspect_ratio'] = cell.aspect_ratio
            
            cell_features.append(features)
        
        results = {}
        
        # Bootstrap analysis
        if options['bootstrap']:
            self.stdout.write("🔄 Running bootstrap analysis...")
            bootstrap_results = bootstrap_engine.bootstrap_population_features(cell_features)
            
            # Validate bootstrap quality
            bootstrap_validations = bootstrap_engine.validate_bootstrap_quality(bootstrap_results)
            
            results['bootstrap'] = {
                'results': self._serialize_bootstrap_results(bootstrap_results),
                'validations': self._serialize_bootstrap_validations(bootstrap_validations)
            }
            
            # Create FeatureStatistics records
            feature_stats = bootstrap_engine.create_feature_statistics_with_bootstrap(
                stat_analysis, cells.first(), bootstrap_results
            )
            
            for fs in feature_stats:
                fs.save()
            
            self.stdout.write(f"✅ Bootstrap analysis complete ({len(bootstrap_results)} features)")
        
        # Individual cell uncertainty analysis
        self.stdout.write("🔄 Calculating individual cell uncertainties...")
        individual_uncertainties = []
        
        for cell in cells:
            if cell.area > 0 and cell.perimeter > 0:
                area_uncertainty = uncertainty_engine.calculate_area_uncertainty(cell.area, cell.perimeter)
                perimeter_uncertainty = uncertainty_engine.calculate_perimeter_uncertainty(cell.perimeter)
                
                cell_uncertainty = {
                    'cell_id': cell.id,
                    'area': {
                        'value': cell.area,
                        'uncertainty': area_uncertainty.standard_uncertainty,
                        'relative_percent': (area_uncertainty.standard_uncertainty / cell.area * 100) if cell.area > 0 else 0
                    },
                    'perimeter': {
                        'value': cell.perimeter,
                        'uncertainty': perimeter_uncertainty.standard_uncertainty,
                        'relative_percent': (perimeter_uncertainty.standard_uncertainty / cell.perimeter * 100) if cell.perimeter > 0 else 0
                    }
                }
                individual_uncertainties.append(cell_uncertainty)
        
        results['individual_uncertainties'] = individual_uncertainties
        
        # Generate report
        self._generate_population_report(analysis, results, output_dir, options)
        
        # Update statistical analysis record
        stat_analysis.computation_time_seconds = 0  # Would measure actual time
        stat_analysis.save()
        
        self.stdout.write(f"✅ Population analysis complete for analysis {analysis_id}")

    def _compare_two_groups(self, analysis_id1, analysis_id2, user, hypothesis_engine, bootstrap_engine, output_dir, options):
        """Compare two groups using hypothesis testing"""
        
        try:
            analysis1 = CellAnalysis.objects.get(id=analysis_id1)
            analysis2 = CellAnalysis.objects.get(id=analysis_id2)
        except CellAnalysis.DoesNotExist as e:
            raise CommandError(f"CellAnalysis not found: {e}")
        
        cells1 = analysis1.detected_cells.all()
        cells2 = analysis2.detected_cells.all()
        
        if not cells1.exists() or not cells2.exists():
            raise CommandError("Both analyses must have detected cells")
        
        feature_name = options['feature']
        self.stdout.write(f"🔄 Comparing {feature_name} between groups:")
        self.stdout.write(f"   Group 1: Analysis {analysis_id1} ({cells1.count()} cells)")
        self.stdout.write(f"   Group 2: Analysis {analysis_id2} ({cells2.count()} cells)")
        
        # Extract feature values
        def get_feature_values(cells, feature):
            values = []
            for cell in cells:
                if feature == 'area':
                    val = cell.area
                elif feature == 'perimeter':
                    val = cell.perimeter
                elif feature == 'circularity' and hasattr(cell, 'circularity'):
                    val = cell.circularity
                elif feature == 'eccentricity' and hasattr(cell, 'eccentricity'):
                    val = cell.eccentricity
                elif feature == 'aspect_ratio' and hasattr(cell, 'aspect_ratio'):
                    val = cell.aspect_ratio
                else:
                    continue
                
                if val is not None and val > 0:
                    values.append(val)
            return values
        
        group1_values = get_feature_values(cells1, feature_name)
        group2_values = get_feature_values(cells2, feature_name)
        
        if len(group1_values) < 3 or len(group2_values) < 3:
            raise CommandError("Each group needs at least 3 valid measurements")
        
        # Perform hypothesis test
        test_result = hypothesis_engine.compare_two_groups(
            group1_values, group2_values, feature_name,
            test_name=f"Compare {feature_name}: Analysis {analysis_id1} vs {analysis_id2}"
        )
        
        # Create HypothesisTest record
        hypothesis_test = hypothesis_engine.create_hypothesis_test_record(
            test_result, analysis1, analysis2, user
        )
        hypothesis_test.save()
        
        # Create PopulationComparison record
        population_comparison = PopulationComparison(
            comparison_name=f"Analysis {analysis_id1} vs {analysis_id2}",
            description=f"Comparison of {feature_name} between two cell populations",
            features_compared=[feature_name],
            total_comparisons=1,
            significant_comparisons=1 if test_result.p_value < 0.05 else 0,
            largest_effect_size=test_result.effect_size,
            most_discriminating_feature=feature_name,
            created_by=user
        )
        population_comparison.save()
        population_comparison.populations.add(analysis1, analysis2)
        population_comparison.hypothesis_tests.add(hypothesis_test)
        
        # Generate comparison report
        self._generate_comparison_report(test_result, analysis1, analysis2, output_dir, options)
        
        self.stdout.write(f"✅ Group comparison complete")

    def _analyze_all_populations(self, user, bootstrap_engine, uncertainty_engine, output_dir, options):
        """Analyze all available populations"""
        
        analyses = CellAnalysis.objects.filter(detected_cells__isnull=False).distinct()
        
        if not analyses.exists():
            raise CommandError("No analyses with detected cells found")
        
        self.stdout.write(f"🔄 Analyzing {analyses.count()} populations...")
        
        for analysis in analyses:
            try:
                self._analyze_single_population(
                    analysis.id, user, bootstrap_engine, uncertainty_engine, 
                    output_dir, options
                )
                self.stdout.write(f"   ✅ Analysis {analysis.id} complete")
            except Exception as e:
                self.stdout.write(f"   ❌ Analysis {analysis.id} failed: {e}")
        
        self.stdout.write(f"✅ All population analyses complete")

    def _serialize_bootstrap_results(self, bootstrap_results):
        """Convert bootstrap results to serializable format"""
        serialized = {}
        for feature_name, result in bootstrap_results.items():
            serialized[feature_name] = {
                'original_value': result.original_value,
                'bootstrap_mean': result.bootstrap_mean,
                'bootstrap_std': result.bootstrap_std,
                'confidence_interval': [result.confidence_interval_lower, result.confidence_interval_upper],
                'bias_estimate': result.bias_estimate,
                'skewness': result.skewness,
                'kurtosis': result.kurtosis,
                'n_samples': result.n_bootstrap_samples
            }
        return serialized

    def _serialize_bootstrap_validations(self, bootstrap_validations):
        """Convert bootstrap validations to serializable format"""
        serialized = {}
        for feature_name, validation in bootstrap_validations.items():
            serialized[feature_name] = {
                'quality_score': validation.quality_score,
                'coverage_probability': validation.coverage_probability,
                'recommendations': validation.recommendations
            }
        return serialized

    def _generate_population_report(self, analysis, results, output_dir, options):
        """Generate population analysis report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = output_dir / f"population_analysis_{analysis.id}_{timestamp}.json"
        
        report_data = {
            'analysis_info': {
                'analysis_id': analysis.id,
                'cell_count': analysis.detected_cells.count(),
                'analysis_date': str(analysis.analysis_date) if analysis.analysis_date else None
            },
            'statistical_analysis': results,
            'generated_at': timestamp
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.stdout.write(f"📄 Population report: {json_path}")

    def _generate_comparison_report(self, test_result, analysis1, analysis2, output_dir, options):
        """Generate group comparison report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = output_dir / f"group_comparison_{analysis1.id}_vs_{analysis2.id}_{timestamp}.json"
        
        report_data = {
            'comparison_info': {
                'analysis1_id': analysis1.id,
                'analysis2_id': analysis2.id,
                'feature_compared': test_result.feature_name
            },
            'test_results': {
                'test_type': test_result.test_type,
                'test_statistic': test_result.test_statistic,
                'p_value': test_result.p_value,
                'effect_size': test_result.effect_size,
                'effect_size_interpretation': test_result.effect_size_interpretation,
                'is_significant': test_result.p_value < 0.05,
                'confidence_interval': test_result.confidence_interval_diff,
                'statistical_power': test_result.statistical_power
            },
            'group_statistics': test_result.group_statistics,
            'generated_at': timestamp
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Console summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write("📊 GROUP COMPARISON RESULTS")
        self.stdout.write("="*60)
        
        g1_stats = test_result.group_statistics['group1']
        g2_stats = test_result.group_statistics['group2']
        
        self.stdout.write(f"Feature: {test_result.feature_name}")
        self.stdout.write(f"Test: {test_result.test_type}")
        self.stdout.write(f"Group 1: {g1_stats['mean']:.2f} ± {g1_stats['std']:.2f} (n={g1_stats['n']})")
        self.stdout.write(f"Group 2: {g2_stats['mean']:.2f} ± {g2_stats['std']:.2f} (n={g2_stats['n']})")
        self.stdout.write(f"p-value: {test_result.p_value:.6f}")
        self.stdout.write(f"Effect size: {test_result.effect_size:.3f} ({test_result.effect_size_interpretation})")
        
        if test_result.p_value < 0.001:
            self.stdout.write("✅ HIGHLY SIGNIFICANT")
        elif test_result.p_value < 0.01:
            self.stdout.write("✅ SIGNIFICANT")
        elif test_result.p_value < 0.05:
            self.stdout.write("✅ SIGNIFICANT")
        else:
            self.stdout.write("❌ NOT SIGNIFICANT")
        
        self.stdout.write(f"📄 Detailed report: {json_path}")