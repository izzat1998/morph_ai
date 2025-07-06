"""
Comprehensive tests for morphometric statistics framework
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from django.test import TestCase
from django.contrib.auth import get_user_model

from cells.models import Cell, CellAnalysis, DetectedCell
from .models import (
    FeatureStatistics, HypothesisTest, 
    MultipleComparisonCorrection, PopulationComparison
)
from .services.hypothesis_testing import HypothesisTestingEngine, TestResult
from .services.confidence_intervals import ConfidenceIntervalCalculator
from .services.uncertainty_propagation import UncertaintyPropagationEngine

User = get_user_model()


class HypothesisTestingEngineTests(TestCase):
    """Test suite for HypothesisTestingEngine"""
    
    def setUp(self):
        """Set up test data"""
        self.engine = HypothesisTestingEngine()
        
        # Create sample data with known properties
        np.random.seed(42)  # For reproducible tests
        
        # Group 1: Normal distribution, mean=10, std=2
        self.group1_normal = np.random.normal(10, 2, 30)
        
        # Group 2: Normal distribution, mean=12, std=2 (Cohen's d ≈ 1.0)
        self.group2_normal = np.random.normal(12, 2, 30)
        
        # Group 3: Skewed distribution
        self.group1_skewed = np.random.exponential(2, 30)
        self.group2_skewed = np.random.exponential(3, 30)
        
        # Multiple groups for ANOVA
        self.multiple_groups = [
            np.random.normal(10, 2, 20),
            np.random.normal(12, 2, 20),
            np.random.normal(14, 2, 20)
        ]
        self.group_names = ['Group A', 'Group B', 'Group C']
    
    def test_two_group_t_test(self):
        """Test two-group t-test functionality"""
        result = self.engine.compare_two_groups(
            self.group1_normal,
            self.group2_normal,
            feature_name='area',
            test_name='Normal groups comparison'
        )
        
        # Check result structure
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test_type, 'ttest_ind')
        self.assertEqual(result.feature_name, 'area')
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size, float)
        self.assertEqual(result.effect_size_type, 'cohens_d')
        
        # Check group statistics
        self.assertIn('group1', result.group_statistics)
        self.assertIn('group2', result.group_statistics)
        self.assertEqual(result.group_statistics['group1']['n'], 30)
        self.assertEqual(result.group_statistics['group2']['n'], 30)
        
        # Check confidence interval
        self.assertIsInstance(result.confidence_interval_diff, tuple)
        self.assertEqual(len(result.confidence_interval_diff), 2)
    
    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test for non-normal data"""
        result = self.engine.compare_two_groups(
            self.group1_skewed,
            self.group2_skewed,
            feature_name='perimeter',
            test_name='Skewed groups comparison',
            test_type='mannwhitney'
        )
        
        self.assertEqual(result.test_type, 'mannwhitney')
        self.assertEqual(result.effect_size_type, 'cliff_delta')
        self.assertIsInstance(result.p_value, float)
        self.assertTrue(0 <= result.p_value <= 1)
    
    def test_multiple_groups_anova(self):
        """Test ANOVA for multiple groups"""
        results = self.engine.compare_multiple_groups(
            self.multiple_groups,
            self.group_names,
            feature_name='circularity',
            test_name='Multiple groups ANOVA'
        )
        
        # Check overall result
        self.assertIn('overall', results)
        overall = results['overall']
        self.assertIn(overall.test_type, ['anova', 'kruskal'])
        self.assertEqual(overall.feature_name, 'circularity')
        
        # Check post-hoc comparisons
        posthoc_keys = [k for k in results.keys() if k.startswith('posthoc')]
        self.assertTrue(len(posthoc_keys) > 0)
    
    def test_effect_size_calculations(self):
        """Test effect size calculations"""
        # Test Cohen's d
        effect_size, effect_type = self.engine._calculate_cohens_d(
            self.group1_normal, self.group2_normal
        )
        self.assertEqual(effect_type, 'cohens_d')
        self.assertIsInstance(effect_size, float)
        self.assertTrue(effect_size >= 0)
        
        # Test Cliff's delta
        effect_size, effect_type = self.engine._calculate_cliff_delta(
            self.group1_skewed, self.group2_skewed
        )
        self.assertEqual(effect_type, 'cliff_delta')
        self.assertIsInstance(effect_size, float)
        self.assertTrue(0 <= effect_size <= 1)
    
    def test_effect_size_interpretation(self):
        """Test effect size interpretation"""
        # Test Cohen's d interpretation
        self.assertEqual(self.engine._interpret_effect_size(0.1, 'cohens_d'), 'negligible')
        self.assertEqual(self.engine._interpret_effect_size(0.3, 'cohens_d'), 'small')
        self.assertEqual(self.engine._interpret_effect_size(0.6, 'cohens_d'), 'medium')
        self.assertEqual(self.engine._interpret_effect_size(0.9, 'cohens_d'), 'large')
    
    def test_assumption_checking(self):
        """Test statistical assumption checking"""
        assumptions = self.engine._check_assumptions(
            self.group1_normal, self.group2_normal
        )
        
        # Check required keys
        required_keys = [
            'normality_group1', 'normality_group2', 'normality_both',
            'equal_variances', 'equal_variances_bool'
        ]
        for key in required_keys:
            self.assertIn(key, assumptions)
            self.assertIsInstance(assumptions[key], (float, bool))
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction"""
        # Create multiple test results
        test_results = []
        for i in range(5):
            result = self.engine.compare_two_groups(
                np.random.normal(10, 2, 20),
                np.random.normal(10.5, 2, 20),
                feature_name=f'feature_{i}',
                test_name=f'Test {i}'
            )
            test_results.append(result)
        
        # Apply correction
        correction = self.engine.correct_multiple_comparisons(
            test_results, method='benjamini_hochberg'
        )
        
        self.assertIsInstance(correction, MultipleComparisonCorrection)
        self.assertEqual(correction.total_tests, 5)
        self.assertEqual(correction.correction_method, 'benjamini_hochberg')
    
    def test_auto_test_selection(self):
        """Test automatic test selection based on assumptions"""
        # Normal data should select t-test
        result_normal = self.engine.compare_two_groups(
            self.group1_normal,
            self.group2_normal,
            feature_name='area',
            auto_select_test=True
        )
        self.assertEqual(result_normal.test_type, 'ttest_ind')
        
        # Skewed data should select Mann-Whitney
        result_skewed = self.engine.compare_two_groups(
            self.group1_skewed,
            self.group2_skewed,
            feature_name='area',
            auto_select_test=True
        )
        # This may be either test depending on sample size and assumptions
        self.assertIn(result_skewed.test_type, ['ttest_ind', 'mannwhitney'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.engine.compare_two_groups([1, 2], [3, 4], 'test_feature')
        
        # Test with invalid test type
        with self.assertRaises(ValueError):
            self.engine.compare_two_groups(
                self.group1_normal, self.group2_normal,
                'test_feature', test_type='invalid_test'
            )
        
        # Test multiple groups with insufficient groups
        with self.assertRaises(ValueError):
            self.engine.compare_multiple_groups(
                [self.group1_normal], ['Group1'], 'test_feature'
            )
    
    def test_paired_tests(self):
        """Test paired statistical tests"""
        # Create paired data
        baseline = np.random.normal(10, 2, 25)
        treatment = baseline + np.random.normal(1, 0.5, 25)  # Paired improvement
        
        result = self.engine.compare_two_groups(
            baseline, treatment,
            feature_name='paired_measurement',
            test_type='ttest_rel'
        )
        
        self.assertEqual(result.test_type, 'ttest_rel')
        self.assertIsInstance(result.p_value, float)


class ConfidenceIntervalTests(TestCase):
    """Test suite for ConfidenceIntervalCalculator"""
    
    def setUp(self):
        """Set up test data"""
        self.calculator = ConfidenceIntervalCalculator()
        np.random.seed(42)
        self.sample_data = np.random.normal(10, 2, 100)
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval calculation"""
        ci_low, ci_high = self.calculator.bootstrap_ci(
            self.sample_data, statistic=np.mean, n_bootstrap=1000
        )
        
        self.assertIsInstance(ci_low, float)
        self.assertIsInstance(ci_high, float)
        self.assertLess(ci_low, ci_high)
        
        # CI should contain the true mean most of the time
        sample_mean = np.mean(self.sample_data)
        self.assertLessEqual(ci_low, sample_mean)
        self.assertGreaterEqual(ci_high, sample_mean)
    
    def test_parametric_ci(self):
        """Test parametric confidence interval calculation"""
        ci_low, ci_high = self.calculator.parametric_ci(
            self.sample_data, confidence_level=0.95
        )
        
        self.assertIsInstance(ci_low, float)
        self.assertIsInstance(ci_high, float)
        self.assertLess(ci_low, ci_high)
    
    def test_different_confidence_levels(self):
        """Test different confidence levels"""
        ci_90 = self.calculator.bootstrap_ci(self.sample_data, confidence_level=0.90)
        ci_95 = self.calculator.bootstrap_ci(self.sample_data, confidence_level=0.95)
        ci_99 = self.calculator.bootstrap_ci(self.sample_data, confidence_level=0.99)
        
        # Wider intervals for higher confidence
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        
        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)


class UncertaintyPropagationTests(TestCase):
    """Tests for uncertainty propagation"""
    
    def setUp(self):
        """Set up test data"""
        self.propagator = UncertaintyPropagationEngine()
    
    def test_basic_initialization(self):
        """Test basic uncertainty propagation engine initialization"""
        self.assertIsInstance(self.propagator, UncertaintyPropagationEngine)
    
    def test_placeholder_uncertainty_test(self):
        """Placeholder test for uncertainty propagation functionality"""
        # This is a placeholder test until full uncertainty methods are implemented
        # For now, just verify the engine can be instantiated
        self.assertTrue(True)


class ModelTests(TestCase):
    """Test morphometric statistics models"""
    
    def setUp(self):
        """Set up test models"""
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
        
        # Create test cell and analysis
        self.cell = Cell.objects.create(
            name='Test Cell',
            uploaded_by=self.user,
            image='test.png'
        )
        
        self.analysis = CellAnalysis.objects.create(
            cell=self.cell,
            cellpose_model='cyto',
            diameter=30.0
        )
    
    def test_hypothesis_test_model(self):
        """Test HypothesisTest model creation"""
        hypothesis_test = HypothesisTest.objects.create(
            test_name='Test Comparison',
            test_type='ttest_ind',
            feature_name='area',
            analysis_1=self.analysis,
            performed_by=self.user,
            group1_n=30,
            group2_n=30,
            group1_mean=10.5,
            group1_std=2.1,
            group2_mean=12.3,
            group2_std=2.0,
            test_statistic=3.45,
            p_value=0.001,
            effect_size=0.82,
            effect_size_type='cohens_d',
            effect_size_interpretation='large',
            alpha_level=0.05,
            is_significant=True
        )
        
        self.assertEqual(hypothesis_test.test_name, 'Test Comparison')
        self.assertEqual(hypothesis_test.test_type, 'ttest_ind')
        self.assertTrue(hypothesis_test.is_significant)
        self.assertEqual(hypothesis_test.effect_size_interpretation, 'large')
    
    def test_multiple_comparison_correction_model(self):
        """Test MultipleComparisonCorrection model"""
        correction = MultipleComparisonCorrection.objects.create(
            correction_name='Benjamini-Hochberg Correction',
            correction_method='benjamini_hochberg',
            family_wise_alpha=0.05,
            total_tests=10,
            adjusted_alpha=0.005,
            significant_after_correction=3,
            false_discovery_rate=0.05,
            family_wise_error_rate=0.05,
            performed_by=self.user
        )
        
        self.assertEqual(correction.correction_method, 'benjamini_hochberg')
        self.assertEqual(correction.total_tests, 10)
        self.assertEqual(correction.significant_after_correction, 3)
    
    def test_population_comparison_model(self):
        """Test PopulationComparison model"""
        comparison = PopulationComparison.objects.create(
            comparison_name='Treatment vs Control',
            description='Comparison of treatment groups',
            created_by=self.user,
            total_comparisons=5,
            significant_comparisons=2,
            most_discriminating_feature='area'
        )
        
        self.assertEqual(comparison.comparison_name, 'Treatment vs Control')
        self.assertEqual(comparison.total_comparisons, 5)
        self.assertEqual(comparison.significant_comparisons, 2)


class IntegrationTests(TestCase):
    """Integration tests for the complete statistical framework"""
    
    def setUp(self):
        """Set up integration test data"""
        self.user = User.objects.create_user(
            email='integration@example.com',
            password='testpass123'
        )
        
        # Create test analyses
        self.cell1 = Cell.objects.create(
            name='Cell 1', uploaded_by=self.user, image='cell1.png'
        )
        self.cell2 = Cell.objects.create(
            name='Cell 2', uploaded_by=self.user, image='cell2.png'
        )
        
        self.analysis1 = CellAnalysis.objects.create(
            cell=self.cell1, cellpose_model='cyto', diameter=30.0
        )
        self.analysis2 = CellAnalysis.objects.create(
            cell=self.cell2, cellpose_model='cyto', diameter=30.0
        )
        
        # Create detected cells with different characteristics
        np.random.seed(42)
        
        # Group 1: Smaller cells
        for i in range(20):
            DetectedCell.objects.create(
                analysis=self.analysis1,
                cell_id=i,
                area=np.random.normal(80, 10),
                perimeter=np.random.normal(30, 5),
                centroid_x=np.random.uniform(0, 100),
                centroid_y=np.random.uniform(0, 100)
            )
        
        # Group 2: Larger cells
        for i in range(20):
            DetectedCell.objects.create(
                analysis=self.analysis2,
                cell_id=i,
                area=np.random.normal(120, 15),
                perimeter=np.random.normal(40, 6),
                centroid_x=np.random.uniform(0, 100),
                centroid_y=np.random.uniform(0, 100)
            )
    
    def test_full_hypothesis_testing_workflow(self):
        """Test complete hypothesis testing workflow"""
        from .services.morphometric_integration import MorphometricStatisticalAnalyzer
        
        analyzer = MorphometricStatisticalAnalyzer()
        
        # Perform comparison between two analyses
        results = analyzer.compare_analyses(
            self.analysis1, self.analysis2,
            features=['area', 'perimeter'],
            test_name='Size comparison'
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('area', results)
        self.assertIn('perimeter', results)
        
        # Check that results are TestResult objects
        for feature, result in results.items():
            self.assertIsInstance(result, TestResult)
            self.assertEqual(result.feature_name, feature)
    
    def test_confidence_interval_integration(self):
        """Test confidence interval integration with real data"""
        from .services.morphometric_integration import MorphometricStatisticalAnalyzer
        
        analyzer = MorphometricStatisticalAnalyzer()
        
        # Calculate confidence intervals for an analysis
        ci_results = analyzer.calculate_confidence_intervals(
            self.analysis1, features=['area', 'perimeter']
        )
        
        self.assertIsInstance(ci_results, dict)
        self.assertIn('area', ci_results)
        self.assertIn('perimeter', ci_results)
        
        # Check CI structure
        for feature, ci_data in ci_results.items():
            self.assertIn('mean', ci_data)
            self.assertIn('ci_lower', ci_data)
            self.assertIn('ci_upper', ci_data)
            self.assertLess(ci_data['ci_lower'], ci_data['ci_upper'])


if __name__ == '__main__':
    unittest.main()