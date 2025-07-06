"""
Hypothesis Testing Framework for Morphometric Analysis

This module provides comprehensive statistical testing capabilities for
comparing morphometric features between groups and populations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap
import warnings

from ..models import HypothesisTest, MultipleComparisonCorrection, PopulationComparison


@dataclass
class TestConfiguration:
    """Configuration for statistical tests"""
    test_type: str
    alpha_level: float
    alternative: str  # 'two-sided', 'less', 'greater'
    equal_variances: bool
    normality_assumption: bool
    effect_size_type: str


@dataclass
class TestResult:
    """Results from statistical hypothesis test"""
    test_name: str
    test_type: str
    feature_name: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[float]
    effect_size: float
    effect_size_type: str
    effect_size_interpretation: str
    confidence_interval_diff: Tuple[float, float]
    statistical_power: Optional[float]
    minimum_detectable_effect: Optional[float]
    assumptions_met: bool
    assumption_tests: Dict[str, float]
    group_statistics: Dict[str, Dict[str, float]]


@dataclass
class PowerAnalysisResult:
    """Results from statistical power analysis"""
    test_type: str
    effect_size: float
    alpha: float
    power: float
    required_n_per_group: int
    actual_n_per_group: int
    minimum_detectable_effect: float
    power_curve_data: Optional[Dict] = None


class HypothesisTestingEngine:
    """
    Comprehensive hypothesis testing engine for morphometric comparisons
    
    Supports multiple test types, effect size calculations, power analysis,
    and assumption checking with detailed reporting.
    """
    
    def __init__(self, default_alpha: float = 0.05):
        """
        Initialize hypothesis testing engine
        
        Args:
            default_alpha: Default significance level
        """
        self.default_alpha = default_alpha
        
        # Effect size interpretation thresholds (Cohen's conventions)
        self.effect_size_thresholds = {
            'cohens_d': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
            'eta_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
            'cramers_v': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
            'cliff_delta': {'small': 0.11, 'medium': 0.28, 'large': 0.43}
        }
    
    def compare_two_groups(
        self,
        group1_data: List[float],
        group2_data: List[float],
        feature_name: str,
        test_name: str = "Two-group comparison",
        alpha: float = None,
        auto_select_test: bool = True,
        test_type: str = None
    ) -> TestResult:
        """
        Compare morphometric feature between two groups
        
        Args:
            group1_data: Feature values for group 1
            group2_data: Feature values for group 2
            feature_name: Name of the morphometric feature
            test_name: Descriptive name for the test
            alpha: Significance level (uses default if None)
            auto_select_test: Automatically select appropriate test
            test_type: Force specific test type
            
        Returns:
            TestResult with complete analysis
        """
        if alpha is None:
            alpha = self.default_alpha
        
        group1_array = np.array(group1_data)
        group2_array = np.array(group2_data)
        
        # Remove invalid values
        group1_clean = group1_array[np.isfinite(group1_array)]
        group2_clean = group2_array[np.isfinite(group2_array)]
        
        if len(group1_clean) < 3 or len(group2_clean) < 3:
            raise ValueError("Each group must have at least 3 valid observations")
        
        # Check assumptions
        assumption_tests = self._check_assumptions(group1_clean, group2_clean)
        
        # Select test type
        if auto_select_test and test_type is None:
            test_type = self._select_appropriate_test(assumption_tests, len(group1_clean), len(group2_clean))
        elif test_type is None:
            test_type = 'ttest_ind'  # Default
        
        # Perform statistical test
        if test_type == 'ttest_ind':
            result = self._perform_t_test(group1_clean, group2_clean, paired=False, equal_var=assumption_tests['equal_variances'])
        elif test_type == 'ttest_rel':
            if len(group1_clean) != len(group2_clean):
                raise ValueError("Paired t-test requires equal sample sizes")
            result = self._perform_t_test(group1_clean, group2_clean, paired=True)
        elif test_type == 'mannwhitney':
            result = self._perform_mann_whitney(group1_clean, group2_clean)
        elif test_type == 'wilcoxon':
            if len(group1_clean) != len(group2_clean):
                raise ValueError("Wilcoxon test requires equal sample sizes")
            result = self._perform_wilcoxon(group1_clean, group2_clean)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Calculate group statistics
        group_stats = {
            'group1': {
                'n': len(group1_clean),
                'mean': np.mean(group1_clean),
                'std': np.std(group1_clean, ddof=1),
                'median': np.median(group1_clean),
                'min': np.min(group1_clean),
                'max': np.max(group1_clean)
            },
            'group2': {
                'n': len(group2_clean),
                'mean': np.mean(group2_clean),
                'std': np.std(group2_clean, ddof=1),
                'median': np.median(group2_clean),
                'min': np.min(group2_clean),
                'max': np.max(group2_clean)
            }
        }
        
        # Calculate effect size
        if test_type in ['ttest_ind', 'ttest_rel']:
            effect_size, effect_type = self._calculate_cohens_d(group1_clean, group2_clean)
        elif test_type in ['mannwhitney', 'wilcoxon']:
            effect_size, effect_type = self._calculate_cliff_delta(group1_clean, group2_clean)
        else:
            effect_size, effect_type = 0.0, 'unknown'
        
        # Effect size interpretation
        effect_interpretation = self._interpret_effect_size(effect_size, effect_type)
        
        # Confidence interval for difference
        ci_diff = self._calculate_difference_ci(group1_clean, group2_clean, alpha)
        
        # Power analysis
        power_result = self._calculate_post_hoc_power(group1_clean, group2_clean, effect_size, alpha, test_type)
        
        # Check if assumptions are met
        assumptions_met = self._evaluate_assumptions(assumption_tests, test_type)
        
        return TestResult(
            test_name=test_name,
            test_type=test_type,
            feature_name=feature_name,
            test_statistic=result['statistic'],
            p_value=result['p_value'],
            degrees_of_freedom=result.get('df'),
            effect_size=effect_size,
            effect_size_type=effect_type,
            effect_size_interpretation=effect_interpretation,
            confidence_interval_diff=ci_diff,
            statistical_power=power_result.power if power_result else None,
            minimum_detectable_effect=power_result.minimum_detectable_effect if power_result else None,
            assumptions_met=assumptions_met,
            assumption_tests=assumption_tests,
            group_statistics=group_stats
        )
    
    def compare_multiple_groups(
        self,
        group_data: List[List[float]],
        group_names: List[str],
        feature_name: str,
        test_name: str = "Multiple-group comparison",
        alpha: float = None,
        post_hoc: bool = True
    ) -> Dict[str, TestResult]:
        """
        Compare morphometric feature across multiple groups (ANOVA/Kruskal-Wallis)
        
        Args:
            group_data: List of feature value lists for each group
            group_names: Names of the groups
            feature_name: Name of the morphometric feature
            test_name: Descriptive name for the test
            alpha: Significance level
            post_hoc: Perform post-hoc pairwise comparisons
            
        Returns:
            Dictionary with overall test result and post-hoc comparisons
        """
        if alpha is None:
            alpha = self.default_alpha
        
        if len(group_data) < 2:
            raise ValueError("Need at least 2 groups for comparison")
        
        # Clean data
        clean_groups = []
        for group in group_data:
            clean_group = np.array(group)[np.isfinite(group)]
            if len(clean_group) < 3:
                raise ValueError("Each group must have at least 3 valid observations")
            clean_groups.append(clean_group)
        
        # Check assumptions for ANOVA
        normality_ok = all(self._test_normality(group)[1] > 0.05 for group in clean_groups)
        equal_var_ok = self._test_equal_variances(*clean_groups)[1] > 0.05
        
        # Select appropriate test
        if normality_ok and equal_var_ok:
            test_type = 'anova'
            statistic, p_value = stats.f_oneway(*clean_groups)
            df_between = len(clean_groups) - 1
            df_within = sum(len(group) for group in clean_groups) - len(clean_groups)
            df = (df_between, df_within)
        else:
            test_type = 'kruskal'
            statistic, p_value = stats.kruskal(*clean_groups)
            df = len(clean_groups) - 1
        
        # Calculate effect size (eta-squared for ANOVA, epsilon-squared for Kruskal-Wallis)
        if test_type == 'anova':
            effect_size = self._calculate_eta_squared(clean_groups, statistic, df)
            effect_type = 'eta_squared'
        else:
            effect_size = self._calculate_epsilon_squared(statistic, sum(len(g) for g in clean_groups))
            effect_type = 'epsilon_squared'
        
        effect_interpretation = self._interpret_effect_size(effect_size, effect_type)
        
        # Overall test result
        overall_result = TestResult(
            test_name=f"{test_name} (Overall)",
            test_type=test_type,
            feature_name=feature_name,
            test_statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df if isinstance(df, (int, float)) else df[0],
            effect_size=effect_size,
            effect_size_type=effect_type,
            effect_size_interpretation=effect_interpretation,
            confidence_interval_diff=(np.nan, np.nan),  # Not applicable for overall test
            statistical_power=None,  # Complex calculation for multiple groups
            minimum_detectable_effect=None,
            assumptions_met=normality_ok and equal_var_ok if test_type == 'anova' else True,
            assumption_tests={
                'normality_all_groups': min(self._test_normality(group)[1] for group in clean_groups),
                'equal_variances': equal_var_ok
            },
            group_statistics={
                name: {
                    'n': len(group),
                    'mean': np.mean(group),
                    'std': np.std(group, ddof=1),
                    'median': np.median(group)
                } for name, group in zip(group_names, clean_groups)
            }
        )
        
        results = {'overall': overall_result}
        
        # Post-hoc pairwise comparisons
        if post_hoc and p_value < alpha:
            for i in range(len(clean_groups)):
                for j in range(i + 1, len(clean_groups)):
                    comparison_name = f"{group_names[i]} vs {group_names[j]}"
                    try:
                        pairwise_result = self.compare_two_groups(
                            clean_groups[i],
                            clean_groups[j],
                            feature_name,
                            comparison_name,
                            alpha,
                            auto_select_test=True
                        )
                        results[f"posthoc_{i}_{j}"] = pairwise_result
                    except Exception as e:
                        warnings.warn(f"Post-hoc comparison {comparison_name} failed: {e}")
        
        return results
    
    def _check_assumptions(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """Check statistical test assumptions"""
        
        # Normality tests
        _, norm1_p = self._test_normality(group1)
        _, norm2_p = self._test_normality(group2)
        
        # Equal variances test
        _, equal_var_p = self._test_equal_variances(group1, group2)
        
        return {
            'normality_group1': norm1_p,
            'normality_group2': norm2_p,
            'normality_both': min(norm1_p, norm2_p),
            'equal_variances': equal_var_p,
            'equal_variances_bool': equal_var_p > 0.05
        }
    
    def _test_normality(self, data: np.ndarray) -> Tuple[float, float]:
        """Test normality using Shapiro-Wilk or Anderson-Darling"""
        if len(data) <= 5000:
            return stats.shapiro(data)
        else:
            # Use Anderson-Darling for large samples
            ad_result = stats.anderson(data, dist='norm')
            # Convert to p-value approximation
            critical_val = ad_result.critical_values[2]  # 5% level
            p_approx = 0.05 if ad_result.statistic > critical_val else 0.1
            return ad_result.statistic, p_approx
    
    def _test_equal_variances(self, *groups: np.ndarray) -> Tuple[float, float]:
        """Test equal variances using Levene's test"""
        return stats.levene(*groups)
    
    def _select_appropriate_test(self, assumptions: Dict, n1: int, n2: int) -> str:
        """Select appropriate statistical test based on assumptions"""
        
        normality_ok = assumptions['normality_both'] > 0.05
        equal_var_ok = assumptions['equal_variances_bool']
        sample_size_ok = min(n1, n2) >= 15
        
        if normality_ok and equal_var_ok:
            return 'ttest_ind'
        elif normality_ok and not equal_var_ok:
            return 'ttest_ind'  # Welch's t-test
        elif not normality_ok or not sample_size_ok:
            return 'mannwhitney'
        else:
            return 'ttest_ind'  # Default
    
    def _perform_t_test(self, group1: np.ndarray, group2: np.ndarray, paired: bool = False, equal_var: bool = True) -> Dict:
        """Perform t-test"""
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
            df = len(group1) - 1
        else:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
            if equal_var:
                df = len(group1) + len(group2) - 2
            else:
                # Welch's t-test degrees of freedom
                s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        return {'statistic': statistic, 'p_value': p_value, 'df': df}
    
    def _perform_mann_whitney(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Perform Mann-Whitney U test"""
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return {'statistic': statistic, 'p_value': p_value}
    
    def _perform_wilcoxon(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Perform Wilcoxon signed-rank test"""
        statistic, p_value = stats.wilcoxon(group1, group2, alternative='two-sided')
        return {'statistic': statistic, 'p_value': p_value}
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        return abs(cohens_d), 'cohens_d'
    
    def _calculate_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """Calculate Cliff's delta effect size for non-parametric tests"""
        n1, n2 = len(group1), len(group2)
        
        # Count pairs where group1[i] > group2[j]
        greater = sum(x > y for x in group1 for y in group2)
        less = sum(x < y for x in group1 for y in group2)
        
        cliff_delta = (greater - less) / (n1 * n2)
        return abs(cliff_delta), 'cliff_delta'
    
    def _calculate_eta_squared(self, groups: List[np.ndarray], f_statistic: float, df: Tuple) -> float:
        """Calculate eta-squared for ANOVA"""
        df_between, df_within = df
        eta_squared = (f_statistic * df_between) / (f_statistic * df_between + df_within)
        return eta_squared
    
    def _calculate_epsilon_squared(self, h_statistic: float, total_n: int) -> float:
        """Calculate epsilon-squared for Kruskal-Wallis"""
        return h_statistic / (total_n - 1)
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """Interpret effect size magnitude"""
        if effect_type not in self.effect_size_thresholds:
            return 'unknown'
        
        thresholds = self.effect_size_thresholds[effect_type]
        
        if effect_size < thresholds['small']:
            return 'negligible'
        elif effect_size < thresholds['medium']:
            return 'small'
        elif effect_size < thresholds['large']:
            return 'medium'
        else:
            return 'large'
    
    def _calculate_difference_ci(self, group1: np.ndarray, group2: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean_diff = np.mean(group1) - np.mean(group2)
        
        # Use bootstrap for robust CI
        def diff_statistic(x1, x2):
            return np.mean(x1) - np.mean(x2)
        
        try:
            # Scipy bootstrap (if available)
            rng = np.random.default_rng()
            res = bootstrap((group1, group2), diff_statistic, 
                          n_resamples=1000, confidence_level=1-alpha, 
                          random_state=rng, method='percentile')
            return res.confidence_interval.low, res.confidence_interval.high
        except:
            # Fallback: simple parametric CI
            se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
            t_crit = stats.t.ppf(1 - alpha/2, len(group1) + len(group2) - 2)
            margin = t_crit * se_diff
            return mean_diff - margin, mean_diff + margin
    
    def _calculate_post_hoc_power(self, group1: np.ndarray, group2: np.ndarray, 
                                 effect_size: float, alpha: float, test_type: str) -> Optional[PowerAnalysisResult]:
        """Calculate post-hoc statistical power"""
        try:
            from statsmodels.stats.power import ttest_power, TTestPower
            
            if test_type in ['ttest_ind', 'ttest_rel']:
                n_per_group = min(len(group1), len(group2))
                power = ttest_power(effect_size, n_per_group, alpha, alternative='two-sided')
                
                # Calculate minimum detectable effect with 80% power
                power_analysis = TTestPower()
                min_effect = power_analysis.solve_power(effect_size=None, nobs=n_per_group, 
                                                      alpha=alpha, power=0.8, alternative='two-sided')
                
                return PowerAnalysisResult(
                    test_type=test_type,
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    required_n_per_group=int(power_analysis.solve_power(effect_size=effect_size, nobs=None, 
                                                                      alpha=alpha, power=0.8, alternative='two-sided')),
                    actual_n_per_group=n_per_group,
                    minimum_detectable_effect=min_effect
                )
        except ImportError:
            # statsmodels not available
            pass
        
        return None
    
    def _evaluate_assumptions(self, assumption_tests: Dict, test_type: str) -> bool:
        """Evaluate if test assumptions are reasonably met"""
        if test_type in ['ttest_ind', 'ttest_rel']:
            # For t-tests, we need reasonable normality OR large sample sizes
            normality_ok = assumption_tests['normality_both'] > 0.01  # Relaxed threshold
            return normality_ok
        elif test_type in ['mannwhitney', 'wilcoxon']:
            # Non-parametric tests have fewer assumptions
            return True
        else:
            return True
    
    def correct_multiple_comparisons(
        self,
        test_results: List[TestResult],
        method: str = 'benjamini_hochberg',
        family_wise_alpha: float = 0.05
    ) -> MultipleComparisonCorrection:
        """
        Apply multiple comparison correction to a family of tests
        
        Args:
            test_results: List of TestResult objects
            method: Correction method ('bonferroni', 'holm', 'benjamini_hochberg', etc.)
            family_wise_alpha: Family-wise error rate
            
        Returns:
            MultipleComparisonCorrection with correction results
        """
        from statsmodels.stats.multitest import multipletests
        
        p_values = [result.p_value for result in test_results]
        
        # Apply correction
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=family_wise_alpha, method=method, returnsorted=False
        )
        
        # Count significant results
        significant_before = sum(1 for p in p_values if p < family_wise_alpha)
        significant_after = sum(rejected)
        
        # Create correction record
        correction = MultipleComparisonCorrection(
            correction_name=f"Multiple comparison correction ({method})",
            correction_method=method,
            family_wise_alpha=family_wise_alpha,
            total_tests=len(test_results),
            adjusted_alpha=alpha_bonf if method == 'bonferroni' else family_wise_alpha,
            significant_after_correction=significant_after,
            false_discovery_rate=family_wise_alpha if 'benjamini' in method else None,
            family_wise_error_rate=family_wise_alpha
        )
        
        return correction
    
    def create_hypothesis_test_record(
        self,
        test_result: TestResult,
        analysis_1,
        analysis_2,
        performed_by
    ) -> HypothesisTest:
        """
        Create HypothesisTest model instance from test results
        
        Args:
            test_result: TestResult object
            analysis_1: First CellAnalysis instance
            analysis_2: Second CellAnalysis instance (optional)
            performed_by: User who performed the test
            
        Returns:
            HypothesisTest model instance
        """
        
        group1_stats = test_result.group_statistics.get('group1', {})
        group2_stats = test_result.group_statistics.get('group2', {})
        
        hypothesis_test = HypothesisTest(
            test_name=test_result.test_name,
            test_type=test_result.test_type,
            feature_name=test_result.feature_name,
            analysis_1=analysis_1,
            analysis_2=analysis_2,
            performed_by=performed_by,
            
            # Sample sizes and descriptive statistics
            group1_n=group1_stats.get('n', 0),
            group2_n=group2_stats.get('n', 0),
            group1_mean=group1_stats.get('mean', 0.0),
            group1_std=group1_stats.get('std', 0.0),
            group1_median=group1_stats.get('median'),
            group2_mean=group2_stats.get('mean'),
            group2_std=group2_stats.get('std'),
            group2_median=group2_stats.get('median'),
            
            # Test results
            test_statistic=test_result.test_statistic,
            p_value=test_result.p_value,
            degrees_of_freedom=test_result.degrees_of_freedom,
            effect_size=test_result.effect_size,
            effect_size_type=test_result.effect_size_type,
            effect_size_interpretation=test_result.effect_size_interpretation,
            
            # Additional statistics
            alpha_level=self.default_alpha,
            is_significant=test_result.p_value < self.default_alpha,
            confidence_interval_diff_lower=test_result.confidence_interval_diff[0],
            confidence_interval_diff_upper=test_result.confidence_interval_diff[1],
            statistical_power=test_result.statistical_power,
            minimum_detectable_effect=test_result.minimum_detectable_effect,
            
            # Assumption checking
            normality_test_p_value=test_result.assumption_tests.get('normality_both'),
            equal_variance_test_p_value=test_result.assumption_tests.get('equal_variances'),
            assumptions_met=test_result.assumptions_met,
            
            notes=f"Test performed automatically. Assumptions: {test_result.assumptions_met}"
        )
        
        return hypothesis_test