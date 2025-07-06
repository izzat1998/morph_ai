"""
Population Comparison Management Command

Performs statistical comparisons between cell populations with 
comprehensive hypothesis testing and multiple comparison corrections.
"""

import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from django.db.models import Q

from cells.models import Cell, CellAnalysis, DetectedCell
from morphometric_stats.models import HypothesisTest, PopulationComparison, MultipleComparisonCorrection
from morphometric_stats.services.hypothesis_testing import HypothesisTestingEngine
# Removed import - using direct database queries instead

User = get_user_model()


class Command(BaseCommand):
    help = """
    Perform statistical comparisons between cell populations.
    
    Examples:
        # Compare two specific analyses
        python manage.py population_comparison --analysis-ids 1 2 --features area perimeter
        
        # Compare analyses by name pattern
        python manage.py population_comparison --name-pattern "control*" "treatment*" --features area
        
        # Multi-group comparison with correction
        python manage.py population_comparison --analysis-ids 1 2 3 --features area --multi-group --correction benjamini_hochberg
        
        # Compare all features with automatic test selection
        python manage.py population_comparison --analysis-ids 1 2 --all-features --auto-test
    """
    
    def add_arguments(self, parser):
        """Add command arguments"""
        
        # Analysis selection
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '--analysis-ids',
            nargs='+',
            type=int,
            help='Analysis IDs to compare'
        )
        group.add_argument(
            '--name-pattern',
            nargs='+',
            help='Name patterns to match analyses (supports wildcards)'
        )
        
        # Feature selection
        feature_group = parser.add_mutually_exclusive_group(required=True)
        feature_group.add_argument(
            '--features',
            nargs='+',
            help='Specific morphometric features to compare'
        )
        feature_group.add_argument(
            '--all-features',
            action='store_true',
            help='Compare all morphometric features'
        )
        
        # Statistical options
        parser.add_argument(
            '--multi-group',
            action='store_true',
            help='Perform multi-group comparison (ANOVA/Kruskal-Wallis)'
        )
        parser.add_argument(
            '--test-type',
            choices=['auto', 'ttest_ind', 'ttest_rel', 'mannwhitney', 'wilcoxon', 'anova', 'kruskal'],
            default='auto',
            help='Statistical test type (default: auto-select)'
        )
        parser.add_argument(
            '--alpha',
            type=float,
            default=0.05,
            help='Significance level (default: 0.05)'
        )
        parser.add_argument(
            '--correction',
            choices=['none', 'bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli'],
            default='benjamini_hochberg',
            help='Multiple comparison correction method'
        )
        
        # Output options
        parser.add_argument(
            '--output-file',
            help='Save results to JSON file'
        )
        parser.add_argument(
            '--comparison-name',
            help='Name for the population comparison'
        )
        parser.add_argument(
            '--save-to-db',
            action='store_true',
            help='Save results to database'
        )
        parser.add_argument(
            '--user-email',
            help='Email of user performing comparison (required if saving to DB)'
        )
        
        # Additional options
        parser.add_argument(
            '--min-cells',
            type=int,
            default=5,
            help='Minimum cells required per analysis (default: 5)'
        )
        parser.add_argument(
            '--effect-size-only',
            action='store_true',
            help='Calculate only effect sizes (no hypothesis tests)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        
        try:
            # Get analyses
            analyses = self.get_analyses(options)
            self.stdout.write(f"Found {len(analyses)} analyses for comparison")
            
            # Get features
            features = self.get_features(options)
            self.stdout.write(f"Comparing {len(features)} features: {', '.join(features)}")
            
            # Validate data
            self.validate_data(analyses, options['min_cells'])
            
            # Setup statistical engine
            engine = HypothesisTestingEngine(default_alpha=options['alpha'])
            
            # Perform comparisons
            if options['multi_group'] and len(analyses) > 2:
                results = self.perform_multi_group_comparison(
                    analyses, features, engine, options
                )
            else:
                results = self.perform_pairwise_comparisons(
                    analyses, features, engine, options
                )
            
            # Apply multiple comparison correction
            if len(results.get('tests', [])) > 1 and options['correction'] != 'none':
                results['correction'] = self.apply_multiple_comparison_correction(
                    results['tests'], engine, options
                )
            
            # Output results
            self.output_results(results, options)
            
            # Save to database if requested
            if options['save_to_db']:
                self.save_to_database(results, analyses, options)
            
            self.stdout.write(
                self.style.SUCCESS(f"Population comparison completed successfully")
            )
            
        except Exception as e:
            raise CommandError(f"Population comparison failed: {str(e)}")
    
    def get_analyses(self, options) -> List[CellAnalysis]:
        """Get analyses based on command options"""
        
        if options['analysis_ids']:
            analyses = list(CellAnalysis.objects.filter(
                id__in=options['analysis_ids']
            ).select_related('cell'))
            
            if len(analyses) != len(options['analysis_ids']):
                missing = set(options['analysis_ids']) - set(a.id for a in analyses)
                raise CommandError(f"Analyses not found: {missing}")
        
        elif options['name_pattern']:
            q_objects = Q()
            for pattern in options['name_pattern']:
                # Simple wildcard support
                if '*' in pattern:
                    q_objects |= Q(cell__name__icontains=pattern.replace('*', ''))
                else:
                    q_objects |= Q(cell__name__iexact=pattern)
            
            analyses = list(CellAnalysis.objects.filter(q_objects).select_related('cell'))
            
            if not analyses:
                raise CommandError(f"No analyses found matching patterns: {options['name_pattern']}")
        
        else:
            raise CommandError("Must specify either --analysis-ids or --name-pattern")
        
        if len(analyses) < 2:
            raise CommandError("Need at least 2 analyses for comparison")
        
        return analyses
    
    def get_features(self, options) -> List[str]:
        """Get features to compare"""
        
        if options['all_features']:
            # All morphometric features
            features = [
                'area', 'perimeter', 'major_axis_length', 'minor_axis_length',
                'eccentricity', 'circularity', 'aspect_ratio', 'extent',
                'solidity', 'compactness', 'roundness', 'convex_area'
            ]
        else:
            features = options['features']
        
        return features
    
    def validate_data(self, analyses: List[CellAnalysis], min_cells: int):
        """Validate that analyses have sufficient data"""
        
        for analysis in analyses:
            cell_count = DetectedCell.objects.filter(analysis=analysis).count()
            if cell_count < min_cells:
                raise CommandError(
                    f"Analysis {analysis.id} ({analysis.cell.name}) has only "
                    f"{cell_count} cells, minimum {min_cells} required"
                )
    
    def perform_pairwise_comparisons(
        self, 
        analyses: List[CellAnalysis], 
        features: List[str],
        engine: HypothesisTestingEngine,
        options: Dict
    ) -> Dict:
        """Perform pairwise comparisons between analyses"""
        
        results = {
            'comparison_type': 'pairwise',
            'analyses': [{'id': a.id, 'name': a.cell.name} for a in analyses],
            'features': features,
            'tests': []
        }
        
        # Compare all pairs
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                analysis1, analysis2 = analyses[i], analyses[j]
                
                if options['verbose']:
                    self.stdout.write(
                        f"Comparing {analysis1.cell.name} vs {analysis2.cell.name}"
                    )
                
                # Get feature data
                for feature in features:
                    try:
                        # Extract feature values
                        cells1 = DetectedCell.objects.filter(analysis=analysis1)
                        cells2 = DetectedCell.objects.filter(analysis=analysis2)
                        
                        values1 = [getattr(cell, feature) for cell in cells1 if getattr(cell, feature) is not None]
                        values2 = [getattr(cell, feature) for cell in cells2 if getattr(cell, feature) is not None]
                        
                        if not values1 or not values2:
                            self.stdout.write(
                                self.style.WARNING(f"Skipping {feature}: insufficient data")
                            )
                            continue
                        
                        # Perform statistical test
                        test_type = None if options['test_type'] == 'auto' else options['test_type']
                        
                        result = engine.compare_two_groups(
                            values1, values2,
                            feature_name=feature,
                            test_name=f"{analysis1.cell.name} vs {analysis2.cell.name}",
                            alpha=options['alpha'],
                            auto_select_test=(options['test_type'] == 'auto'),
                            test_type=test_type
                        )
                        
                        results['tests'].append({
                            'analysis1_id': analysis1.id,
                            'analysis1_name': analysis1.cell.name,
                            'analysis2_id': analysis2.id,
                            'analysis2_name': analysis2.cell.name,
                            'feature': feature,
                            'result': self._serialize_test_result(result)
                        })
                        
                        if options['verbose']:
                            self.print_test_result(result)
                    
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"Error comparing {feature}: {str(e)}")
                        )
        
        return results
    
    def perform_multi_group_comparison(
        self,
        analyses: List[CellAnalysis],
        features: List[str],
        engine: HypothesisTestingEngine,
        options: Dict
    ) -> Dict:
        """Perform multi-group comparison (ANOVA/Kruskal-Wallis)"""
        
        results = {
            'comparison_type': 'multi_group',
            'analyses': [{'id': a.id, 'name': a.cell.name} for a in analyses],
            'features': features,
            'tests': []
        }
        
        for feature in features:
            try:
                # Extract feature data for all groups
                group_data = []
                group_names = []
                
                for analysis in analyses:
                    cells = DetectedCell.objects.filter(analysis=analysis)
                    values = [getattr(cell, feature) for cell in cells if getattr(cell, feature) is not None]
                    
                    if values:
                        group_data.append(values)
                        group_names.append(analysis.cell.name)
                
                if len(group_data) < 2:
                    self.stdout.write(
                        self.style.WARNING(f"Skipping {feature}: insufficient groups with data")
                    )
                    continue
                
                # Perform multi-group test
                test_results = engine.compare_multiple_groups(
                    group_data, group_names, feature,
                    test_name=f"Multi-group comparison: {feature}",
                    alpha=options['alpha'],
                    post_hoc=True
                )
                
                # Store overall result and post-hoc tests
                overall_result = test_results['overall']
                results['tests'].append({
                    'feature': feature,
                    'test_type': 'overall',
                    'result': self._serialize_test_result(overall_result)
                })
                
                # Add post-hoc comparisons
                for key, post_hoc_result in test_results.items():
                    if key.startswith('posthoc'):
                        results['tests'].append({
                            'feature': feature,
                            'test_type': 'posthoc',
                            'comparison': key,
                            'result': self._serialize_test_result(post_hoc_result)
                        })
                
                if options['verbose']:
                    self.stdout.write(f"\n=== {feature} ===")
                    self.print_test_result(overall_result)
                    for key, result in test_results.items():
                        if key.startswith('posthoc'):
                            self.stdout.write(f"  {key}:")
                            self.print_test_result(result, indent=4)
            
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error in multi-group comparison for {feature}: {str(e)}")
                )
        
        return results
    
    def apply_multiple_comparison_correction(
        self,
        test_results: List[Dict],
        engine: HypothesisTestingEngine,
        options: Dict
    ) -> Dict:
        """Apply multiple comparison correction"""
        
        # Extract TestResult objects (handling nested structure)
        test_result_objects = []
        for test_data in test_results:
            # Reconstruct TestResult from serialized data
            result_data = test_data['result']
            from morphometric_stats.services.hypothesis_testing import TestResult
            
            test_result = TestResult(
                test_name=result_data['test_name'],
                test_type=result_data['test_type'],
                feature_name=result_data['feature_name'],
                test_statistic=result_data['test_statistic'],
                p_value=result_data['p_value'],
                degrees_of_freedom=result_data.get('degrees_of_freedom'),
                effect_size=result_data['effect_size'],
                effect_size_type=result_data['effect_size_type'],
                effect_size_interpretation=result_data['effect_size_interpretation'],
                confidence_interval_diff=tuple(result_data['confidence_interval_diff']),
                statistical_power=result_data.get('statistical_power'),
                minimum_detectable_effect=result_data.get('minimum_detectable_effect'),
                assumptions_met=result_data['assumptions_met'],
                assumption_tests=result_data['assumption_tests'],
                group_statistics=result_data['group_statistics']
            )
            test_result_objects.append(test_result)
        
        # Apply correction
        correction = engine.correct_multiple_comparisons(
            test_result_objects,
            method=options['correction'],
            family_wise_alpha=options['alpha']
        )
        
        return {
            'method': correction.correction_method,
            'family_wise_alpha': correction.family_wise_alpha,
            'total_tests': correction.total_tests,
            'significant_after_correction': correction.significant_after_correction,
            'adjusted_alpha': correction.adjusted_alpha,
            'false_discovery_rate': correction.false_discovery_rate
        }
    
    def _serialize_test_result(self, result) -> Dict:
        """Serialize TestResult object to dictionary"""
        return {
            'test_name': result.test_name,
            'test_type': result.test_type,
            'feature_name': result.feature_name,
            'test_statistic': result.test_statistic,
            'p_value': result.p_value,
            'degrees_of_freedom': result.degrees_of_freedom,
            'effect_size': result.effect_size,
            'effect_size_type': result.effect_size_type,
            'effect_size_interpretation': result.effect_size_interpretation,
            'confidence_interval_diff': list(result.confidence_interval_diff),
            'statistical_power': result.statistical_power,
            'minimum_detectable_effect': result.minimum_detectable_effect,
            'assumptions_met': result.assumptions_met,
            'assumption_tests': result.assumption_tests,
            'group_statistics': result.group_statistics
        }
    
    def print_test_result(self, result, indent: int = 0):
        """Print formatted test result"""
        prefix = ' ' * indent
        
        self.stdout.write(f"{prefix}Test: {result.test_name}")
        self.stdout.write(f"{prefix}Feature: {result.feature_name}")
        self.stdout.write(f"{prefix}Test type: {result.test_type}")
        self.stdout.write(f"{prefix}p-value: {result.p_value:.6f}")
        self.stdout.write(f"{prefix}Effect size: {result.effect_size:.4f} ({result.effect_size_type})")
        self.stdout.write(f"{prefix}Effect magnitude: {result.effect_size_interpretation}")
        
        if result.p_value < 0.05:
            self.stdout.write(self.style.SUCCESS(f"{prefix}** SIGNIFICANT **"))
        else:
            self.stdout.write(f"{prefix}Not significant")
        
        self.stdout.write("")
    
    def output_results(self, results: Dict, options: Dict):
        """Output comparison results"""
        
        # Print summary
        total_tests = len(results['tests'])
        if 'correction' in results:
            significant_count = results['correction']['significant_after_correction']
            self.stdout.write(f"\nSummary:")
            self.stdout.write(f"Total tests: {total_tests}")
            self.stdout.write(f"Significant after correction: {significant_count}")
            self.stdout.write(f"Correction method: {results['correction']['method']}")
        else:
            significant_count = sum(1 for test in results['tests'] if test['result']['p_value'] < options['alpha'])
            self.stdout.write(f"\nSummary:")
            self.stdout.write(f"Total tests: {total_tests}")
            self.stdout.write(f"Significant tests: {significant_count}")
        
        # Save to file if requested
        if options['output_file']:
            results['timestamp'] = datetime.now().isoformat()
            results['command_options'] = {k: v for k, v in options.items() if v is not None}
            
            with open(options['output_file'], 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.stdout.write(f"Results saved to: {options['output_file']}")
    
    def save_to_database(self, results: Dict, analyses: List[CellAnalysis], options: Dict):
        """Save results to database"""
        
        if not options['user_email']:
            raise CommandError("--user-email required when saving to database")
        
        try:
            user = User.objects.get(email=options['user_email'])
        except User.DoesNotExist:
            raise CommandError(f"User not found: {options['user_email']}")
        
        # Create PopulationComparison record
        comparison_name = options.get('comparison_name') or f"Population comparison {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        population_comparison = PopulationComparison.objects.create(
            comparison_name=comparison_name,
            description=f"Command-line comparison with {len(analyses)} analyses",
            created_by=user,
            total_comparisons=len(results['tests']),
            significant_comparisons=results.get('correction', {}).get('significant_after_correction', 0),
            most_discriminating_feature=results['features'][0] if results['features'] else 'unknown'
        )
        
        # Save individual hypothesis tests
        for test_data in results['tests']:
            result_data = test_data['result']
            
            analysis_1 = None
            analysis_2 = None
            
            if 'analysis1_id' in test_data:
                analysis_1 = CellAnalysis.objects.get(id=test_data['analysis1_id'])
            if 'analysis2_id' in test_data:
                analysis_2 = CellAnalysis.objects.get(id=test_data['analysis2_id'])
            
            hypothesis_test = HypothesisTest.objects.create(
                test_name=result_data['test_name'],
                test_type=result_data['test_type'],
                feature_name=result_data['feature_name'],
                analysis_1=analysis_1,
                analysis_2=analysis_2,
                performed_by=user,
                
                # Group statistics
                group1_n=result_data['group_statistics'].get('group1', {}).get('n', 0),
                group2_n=result_data['group_statistics'].get('group2', {}).get('n', 0),
                group1_mean=result_data['group_statistics'].get('group1', {}).get('mean', 0.0),
                group1_std=result_data['group_statistics'].get('group1', {}).get('std', 0.0),
                group2_mean=result_data['group_statistics'].get('group2', {}).get('mean'),
                group2_std=result_data['group_statistics'].get('group2', {}).get('std'),
                
                # Test results
                test_statistic=result_data['test_statistic'],
                p_value=result_data['p_value'],
                degrees_of_freedom=result_data.get('degrees_of_freedom'),
                effect_size=result_data['effect_size'],
                effect_size_type=result_data['effect_size_type'],
                effect_size_interpretation=result_data['effect_size_interpretation'],
                alpha_level=options['alpha'],
                is_significant=result_data['p_value'] < options['alpha'],
                confidence_interval_diff_lower=result_data['confidence_interval_diff'][0],
                confidence_interval_diff_upper=result_data['confidence_interval_diff'][1],
                statistical_power=result_data.get('statistical_power'),
                minimum_detectable_effect=result_data.get('minimum_detectable_effect'),
                assumptions_met=result_data['assumptions_met'],
                
                notes=f"Generated by population_comparison command"
            )
        
        # Save multiple comparison correction if applied
        if 'correction' in results:
            correction_obj = MultipleComparisonCorrection.objects.create(
                correction_name=f"Multiple comparison correction ({results['correction']['method']})",
                correction_method=results['correction']['method'],
                family_wise_alpha=results['correction']['family_wise_alpha'],
                total_tests=results['correction']['total_tests'],
                adjusted_alpha=results['correction'].get('adjusted_alpha'),
                significant_after_correction=results['correction']['significant_after_correction'],
                false_discovery_rate=results['correction'].get('false_discovery_rate'),
                family_wise_error_rate=results['correction']['family_wise_alpha'],
                performed_by=user
            )
        
        self.stdout.write(f"Results saved to database. PopulationComparison ID: {population_comparison.id}")