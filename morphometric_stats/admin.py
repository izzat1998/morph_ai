"""
Django admin interface for morphometric statistics models
"""

from django.contrib import admin
from .models import (
    StatisticalAnalysis, FeatureStatistics, HypothesisTest, 
    MultipleComparisonCorrection, PopulationComparison, StatisticalReport
)


@admin.register(StatisticalAnalysis)
class StatisticalAnalysisAdmin(admin.ModelAdmin):
    list_display = ('analysis', 'confidence_level', 'bootstrap_iterations', 'created_at')
    list_filter = ('confidence_level', 'include_confidence_intervals', 'include_bootstrap_analysis', 'created_at')
    search_fields = ('analysis__id',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(FeatureStatistics)  
class FeatureStatisticsAdmin(admin.ModelAdmin):
    list_display = ('feature_name', 'detected_cell', 'measured_value', 'uncertainty_percent', 'measurement_reliability_score')
    list_filter = ('feature_name', 'uncertainty_source')
    search_fields = ('feature_name', 'detected_cell__id')


@admin.register(HypothesisTest)
class HypothesisTestAdmin(admin.ModelAdmin):
    list_display = ('test_name', 'test_type', 'feature_name', 'p_value', 'is_significant', 'test_date')
    list_filter = ('test_type', 'feature_name', 'is_significant', 'test_date')
    search_fields = ('test_name', 'feature_name')
    readonly_fields = ('test_date',)


@admin.register(PopulationComparison)
class PopulationComparisonAdmin(admin.ModelAdmin):
    list_display = ('comparison_name', 'total_comparisons', 'significant_comparisons', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('comparison_name',)
    readonly_fields = ('created_at',)
