from django.db import models
from django.contrib.auth import get_user_model
import json

User = get_user_model()


class StatisticalAnalysis(models.Model):
    """Statistical analysis configuration and metadata"""
    analysis = models.OneToOneField('cells.CellAnalysis', on_delete=models.CASCADE, related_name='statistical_analysis')
    
    # Statistical parameters
    confidence_level = models.FloatField(default=0.95, help_text="Confidence level for intervals (0.80-0.99)")
    bootstrap_iterations = models.IntegerField(default=1000, help_text="Number of bootstrap iterations")
    pixel_uncertainty = models.FloatField(default=0.5, help_text="Pixel-level measurement uncertainty")
    
    # Analysis flags
    include_confidence_intervals = models.BooleanField(default=True)
    include_uncertainty_propagation = models.BooleanField(default=True)
    include_bootstrap_analysis = models.BooleanField(default=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    computation_time_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        verbose_name = 'Statistical Analysis'
        verbose_name_plural = 'Statistical Analyses'
    
    def __str__(self):
        return f"Statistical Analysis - Analysis {self.analysis.id} - {self.confidence_level*100:.0f}% CI"


class FeatureStatistics(models.Model):
    """Statistical information for individual morphometric features"""
    statistical_analysis = models.ForeignKey(StatisticalAnalysis, on_delete=models.CASCADE, related_name='feature_stats')
    detected_cell = models.ForeignKey('cells.DetectedCell', on_delete=models.CASCADE, related_name='statistics', null=True, blank=True)
    
    # Feature identification
    feature_name = models.CharField(max_length=100, help_text="Name of the morphometric feature")
    
    # Basic statistics
    measured_value = models.FloatField(help_text="Raw measured value")
    mean_value = models.FloatField(help_text="Mean value (for bootstrap analysis)")
    std_error = models.FloatField(help_text="Standard error of measurement")
    
    # Confidence intervals
    confidence_interval_lower = models.FloatField(help_text="Lower bound of confidence interval")
    confidence_interval_upper = models.FloatField(help_text="Upper bound of confidence interval")
    confidence_interval_width = models.FloatField(help_text="Width of confidence interval")
    
    # Uncertainty analysis
    uncertainty_absolute = models.FloatField(help_text="Absolute uncertainty")
    uncertainty_percent = models.FloatField(help_text="Relative uncertainty as percentage")
    uncertainty_source = models.CharField(max_length=50, default='pixel_level', help_text="Primary source of uncertainty")
    
    # Bootstrap statistics
    bootstrap_mean = models.FloatField(null=True, blank=True)
    bootstrap_std = models.FloatField(null=True, blank=True)
    bootstrap_skewness = models.FloatField(null=True, blank=True)
    bootstrap_kurtosis = models.FloatField(null=True, blank=True)
    
    # Quality metrics
    measurement_reliability_score = models.FloatField(help_text="0-1 score indicating measurement reliability")
    outlier_score = models.FloatField(null=True, blank=True, help_text="Statistical outlier score")
    
    class Meta:
        ordering = ['statistical_analysis', 'detected_cell', 'feature_name']
        # Note: unique_together removed to allow population-level statistics (detected_cell=null)
        verbose_name = 'Feature Statistics'
        verbose_name_plural = 'Feature Statistics'
    
    def __str__(self):
        cell_info = f"Cell {self.detected_cell.id}" if self.detected_cell else "Population"
        return f"{self.feature_name} - {cell_info} - {self.measured_value:.2f}"
    
    def get_confidence_interval_display(self):
        """Return formatted confidence interval"""
        return f"[{self.confidence_interval_lower:.2f}, {self.confidence_interval_upper:.2f}]"
    
    def is_within_tolerance(self, tolerance_percent=5.0):
        """Check if uncertainty is within acceptable tolerance"""
        return self.uncertainty_percent <= tolerance_percent


class HypothesisTest(models.Model):
    """Results of statistical hypothesis tests"""
    TEST_TYPES = [
        ('ttest_ind', 'Independent t-test'),
        ('ttest_rel', 'Paired t-test'),
        ('mannwhitney', 'Mann-Whitney U test'),
        ('wilcoxon', 'Wilcoxon signed-rank test'),
        ('anova', 'One-way ANOVA'),
        ('kruskal', 'Kruskal-Wallis test'),
        ('chi2', 'Chi-square test')
    ]
    
    # Test identification
    test_name = models.CharField(max_length=200, help_text="Descriptive name for this test")
    test_type = models.CharField(max_length=20, choices=TEST_TYPES)
    feature_name = models.CharField(max_length=100, help_text="Morphometric feature being tested")
    
    # Groups being compared
    analysis_1 = models.ForeignKey('cells.CellAnalysis', related_name='hypothesis_tests_as_group1', on_delete=models.CASCADE)
    analysis_2 = models.ForeignKey('cells.CellAnalysis', related_name='hypothesis_tests_as_group2', on_delete=models.CASCADE, null=True, blank=True)
    
    # Sample sizes
    group1_n = models.IntegerField(help_text="Sample size of group 1")
    group2_n = models.IntegerField(null=True, blank=True, help_text="Sample size of group 2")
    
    # Descriptive statistics
    group1_mean = models.FloatField()
    group1_std = models.FloatField()
    group1_median = models.FloatField(null=True, blank=True)
    group2_mean = models.FloatField(null=True, blank=True)
    group2_std = models.FloatField(null=True, blank=True)
    group2_median = models.FloatField(null=True, blank=True)
    
    # Test results
    test_statistic = models.FloatField(help_text="Test statistic value")
    p_value = models.FloatField(help_text="p-value of the test")
    degrees_of_freedom = models.FloatField(null=True, blank=True)
    
    # Effect size
    effect_size = models.FloatField(help_text="Effect size (Cohen's d, eta-squared, etc.)")
    effect_size_type = models.CharField(max_length=20, default='cohens_d', help_text="Type of effect size calculated")
    effect_size_interpretation = models.CharField(max_length=20, help_text="Interpretation (small, medium, large)")
    
    # Statistical significance
    alpha_level = models.FloatField(default=0.05, help_text="Significance level used")
    is_significant = models.BooleanField(help_text="Whether result is statistically significant")
    confidence_interval_diff_lower = models.FloatField(null=True, blank=True)
    confidence_interval_diff_upper = models.FloatField(null=True, blank=True)
    
    # Power analysis
    statistical_power = models.FloatField(null=True, blank=True, help_text="Post-hoc statistical power")
    minimum_detectable_effect = models.FloatField(null=True, blank=True, help_text="Minimum effect size detectable")
    
    # Test assumptions
    normality_test_p_value = models.FloatField(null=True, blank=True)
    equal_variance_test_p_value = models.FloatField(null=True, blank=True)
    assumptions_met = models.BooleanField(default=True, help_text="Whether test assumptions are satisfied")
    
    # Metadata
    test_date = models.DateTimeField(auto_now_add=True)
    performed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-test_date']
        verbose_name = 'Hypothesis Test'
        verbose_name_plural = 'Hypothesis Tests'
    
    def __str__(self):
        return f"{self.test_name} - {self.feature_name} - p={self.p_value:.4f}"
    
    def get_significance_display(self):
        """Return formatted significance result"""
        if self.is_significant:
            return f"Significant (p={self.p_value:.4f})"
        else:
            return f"Not significant (p={self.p_value:.4f})"
    
    def get_effect_size_display(self):
        """Return formatted effect size with interpretation"""
        return f"{self.effect_size:.3f} ({self.effect_size_interpretation})"


class MultipleComparisonCorrection(models.Model):
    """Results of multiple comparison corrections"""
    CORRECTION_METHODS = [
        ('bonferroni', 'Bonferroni'),
        ('holm', 'Holm-Bonferroni'),
        ('benjamini_hochberg', 'Benjamini-Hochberg (FDR)'),
        ('benjamini_yekutieli', 'Benjamini-Yekutieli'),
        ('sidak', 'Šidák')
    ]
    
    # Correction identification
    correction_name = models.CharField(max_length=200, help_text="Name for this correction analysis")
    correction_method = models.CharField(max_length=30, choices=CORRECTION_METHODS)
    family_wise_alpha = models.FloatField(default=0.05, help_text="Family-wise error rate")
    
    # Tests included in correction
    hypothesis_tests = models.ManyToManyField(HypothesisTest, related_name='corrections')
    
    # Correction results
    total_tests = models.IntegerField(help_text="Total number of tests in family")
    adjusted_alpha = models.FloatField(help_text="Adjusted significance level")
    significant_after_correction = models.IntegerField(help_text="Number of tests significant after correction")
    
    # Quality metrics
    false_discovery_rate = models.FloatField(null=True, blank=True)
    family_wise_error_rate = models.FloatField(null=True, blank=True)
    
    # Metadata
    correction_date = models.DateTimeField(auto_now_add=True)
    performed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    class Meta:
        ordering = ['-correction_date']
        verbose_name = 'Multiple Comparison Correction'
        verbose_name_plural = 'Multiple Comparison Corrections'
    
    def __str__(self):
        return f"{self.correction_name} - {self.get_correction_method_display()}"


class PopulationComparison(models.Model):
    """Comprehensive comparison between cell populations"""
    # Comparison identification
    comparison_name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Populations being compared
    populations = models.ManyToManyField('cells.CellAnalysis', related_name='population_comparisons')
    
    # Features analyzed
    features_compared = models.JSONField(default=list, help_text="List of morphometric features compared")
    
    # Statistical tests performed
    hypothesis_tests = models.ManyToManyField(HypothesisTest, blank=True)
    multiple_comparison_correction = models.ForeignKey(MultipleComparisonCorrection, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Summary results
    total_comparisons = models.IntegerField(default=0)
    significant_comparisons = models.IntegerField(default=0)
    largest_effect_size = models.FloatField(null=True, blank=True)
    most_discriminating_feature = models.CharField(max_length=100, blank=True)
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Population Comparison'
        verbose_name_plural = 'Population Comparisons'
    
    def __str__(self):
        return f"{self.comparison_name} - {self.populations.count()} populations"


class StatisticalReport(models.Model):
    """Comprehensive statistical analysis reports"""
    REPORT_TYPES = [
        ('feature_statistics', 'Feature Statistics Report'),
        ('population_comparison', 'Population Comparison Report'),
        ('hypothesis_testing', 'Hypothesis Testing Report'),
        ('comprehensive', 'Comprehensive Statistical Report')
    ]
    
    # Report identification
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=30, choices=REPORT_TYPES)
    description = models.TextField(blank=True)
    
    # Related data
    statistical_analyses = models.ManyToManyField(StatisticalAnalysis, blank=True)
    population_comparisons = models.ManyToManyField(PopulationComparison, blank=True)
    
    # Report content
    report_data = models.JSONField(default=dict, help_text="Complete report data in JSON format")
    
    # Summary statistics
    total_cells_analyzed = models.IntegerField(default=0)
    total_features_analyzed = models.IntegerField(default=0)
    total_statistical_tests = models.IntegerField(default=0)
    
    # Report files
    html_report_path = models.CharField(max_length=255, blank=True)
    pdf_report_path = models.CharField(max_length=255, blank=True)
    csv_data_path = models.CharField(max_length=255, blank=True)
    
    # Metadata
    generated_by = models.ForeignKey(User, on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-generated_at']
        verbose_name = 'Statistical Report'
        verbose_name_plural = 'Statistical Reports'
    
    def __str__(self):
        return f"{self.title} - {self.generated_at.date()}"
