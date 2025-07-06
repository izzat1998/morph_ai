"""
Generate Validation Report Management Command

Creates comprehensive validation reports for the statistical analysis framework.
"""

import json
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

from morphometric_stats.models import StatisticalAnalysis, PopulationComparison
from morphometric_stats.reports import StatisticalReportGenerator
from cells.models import CellAnalysis

User = get_user_model()


class Command(BaseCommand):
    help = """
    Generate comprehensive validation reports for the statistical analysis framework.
    
    Examples:
        # Generate system-wide validation report
        python manage.py generate_validation_report --system-wide
        
        # Generate report for specific analyses
        python manage.py generate_validation_report --analysis-ids 1 2 3
        
        # Generate report and save to file
        python manage.py generate_validation_report --system-wide --output validation_report.json
        
        # Generate individual analysis reports
        python manage.py generate_validation_report --analysis-ids 1 2 --individual-reports
    """
    
    def add_arguments(self, parser):
        """Add command arguments."""
        
        # Analysis selection
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '--system-wide',
            action='store_true',
            help='Generate report for all statistical analyses'
        )
        group.add_argument(
            '--analysis-ids',
            nargs='+',
            type=int,
            help='Specific analysis IDs to include in report'
        )
        group.add_argument(
            '--recent',
            type=int,
            metavar='DAYS',
            help='Include analyses from the last N days'
        )
        
        # Report options
        parser.add_argument(
            '--output',
            help='Output file path for JSON report'
        )
        parser.add_argument(
            '--individual-reports',
            action='store_true',
            help='Generate individual reports for each analysis'
        )
        parser.add_argument(
            '--include-comparisons',
            action='store_true',
            help='Include population comparison reports'
        )
        parser.add_argument(
            '--format',
            choices=['json', 'summary'],
            default='summary',
            help='Output format (default: summary)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output with detailed information'
        )
    
    def handle(self, *args, **options):
        """Main command handler."""
        
        try:
            # Initialize report generator
            report_generator = StatisticalReportGenerator()
            
            # Get statistical analyses to include
            statistical_analyses = self.get_statistical_analyses(options)
            
            if not statistical_analyses:
                self.stdout.write(
                    self.style.WARNING("No statistical analyses found matching criteria")
                )
                return
            
            self.stdout.write(
                f"Generating validation report for {len(statistical_analyses)} statistical analyses..."
            )
            
            # Generate main validation report
            validation_report = report_generator.generate_validation_report(statistical_analyses)
            
            # Output results
            if options['format'] == 'summary':
                self.output_summary_report(validation_report, options)
            else:
                self.output_json_report(validation_report, options)
            
            # Generate individual reports if requested
            if options['individual_reports']:
                self.generate_individual_reports(statistical_analyses, report_generator, options)
            
            # Generate comparison reports if requested
            if options['include_comparisons']:
                self.generate_comparison_reports(report_generator, options)
            
            # Save to file if requested
            if options['output']:
                self.save_report_to_file(validation_report, options['output'])
                self.stdout.write(
                    self.style.SUCCESS(f"Report saved to: {options['output']}")
                )
            
            self.stdout.write(
                self.style.SUCCESS("Validation report generation completed successfully")
            )
            
        except Exception as e:
            raise CommandError(f"Report generation failed: {str(e)}")
    
    def get_statistical_analyses(self, options):
        """Get statistical analyses based on options."""
        
        if options['system_wide']:
            return list(StatisticalAnalysis.objects.all().select_related('analysis__cell'))
        
        elif options['analysis_ids']:
            analyses = list(StatisticalAnalysis.objects.filter(
                analysis__id__in=options['analysis_ids']
            ).select_related('analysis__cell'))
            
            if len(analyses) != len(options['analysis_ids']):
                found_ids = set(a.analysis.id for a in analyses)
                missing_ids = set(options['analysis_ids']) - found_ids
                raise CommandError(f"Statistical analyses not found for IDs: {missing_ids}")
            
            return analyses
        
        elif options['recent']:
            from datetime import timedelta
            from django.utils import timezone
            
            cutoff_date = timezone.now() - timedelta(days=options['recent'])
            return list(StatisticalAnalysis.objects.filter(
                created_at__gte=cutoff_date
            ).select_related('analysis__cell'))
        
        else:
            raise CommandError("Must specify analysis selection criteria")
    
    def output_summary_report(self, validation_report, options):
        """Output human-readable summary report."""
        
        self.stdout.write(f"\n{self.style.HTTP_INFO('=== STATISTICAL FRAMEWORK VALIDATION REPORT ===')}")
        self.stdout.write(f"Generated: {validation_report['timestamp']}")
        self.stdout.write(f"Total Analyses: {validation_report['total_analyses']}")
        
        # Overall statistics
        stats = validation_report['overall_statistics']
        self.stdout.write(f"\n{self.style.HTTP_INFO('Overall Statistics:')}")
        self.stdout.write(f"  Mean Validation Score: {stats['mean_score']:.3f}")
        self.stdout.write(f"  Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        
        # Status distribution
        self.stdout.write(f"\n{self.style.HTTP_INFO('Status Distribution:')}")
        for status, count in stats['status_distribution'].items():
            percentage = count / validation_report['total_analyses'] * 100
            status_style = self.get_status_style(status)
            self.stdout.write(f"  {status_style(status.title())}: {count} ({percentage:.1f}%)")
        
        # Individual analysis results
        if options['verbose']:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Individual Analysis Results:')}")
            for result in validation_report['validation_results']:
                status_style = self.get_status_style(result['status'])
                self.stdout.write(
                    f"  Analysis {result['analysis_id']} ({result['cell_name']}): "
                    f"{status_style(result['status'])} (Score: {result['score']:.3f})"
                )
                
                if result['issues']:
                    for issue in result['issues']:
                        self.stdout.write(f"    ⚠️  {issue}")
        
        # System recommendations
        self.stdout.write(f"\n{self.style.HTTP_INFO('System Recommendations:')}")
        for i, recommendation in enumerate(validation_report['system_recommendations'], 1):
            self.stdout.write(f"  {i}. {recommendation}")
    
    def output_json_report(self, validation_report, options):
        """Output JSON format report."""
        
        json_output = json.dumps(validation_report, indent=2, default=str)
        self.stdout.write(json_output)
    
    def generate_individual_reports(self, statistical_analyses, report_generator, options):
        """Generate individual reports for each analysis."""
        
        self.stdout.write(f"\n{self.style.HTTP_INFO('Generating individual analysis reports...')}")
        
        for stat_analysis in statistical_analyses:
            try:
                report = report_generator.generate_analysis_report(stat_analysis)
                
                self.stdout.write(f"\n--- Analysis {report.analysis_id}: {report.cell_name} ---")
                self.stdout.write(f"Quality Score: {report.quality_assessment['validation_score']:.3f}")
                self.stdout.write(f"Status: {report.quality_assessment['overall_status']}")
                
                # Show key confidence intervals
                if options['verbose']:
                    self.stdout.write("Key Confidence Intervals:")
                    for feature, ci in list(report.confidence_intervals.items())[:3]:
                        self.stdout.write(
                            f"  {feature}: {ci['point_estimate']:.2f} "
                            f"[{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]"
                        )
                
                # Show recommendations
                self.stdout.write("Recommendations:")
                for rec in report.recommendations[:2]:
                    self.stdout.write(f"  • {rec}")
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Failed to generate report for analysis {stat_analysis.analysis.id}: {str(e)}")
                )
    
    def generate_comparison_reports(self, report_generator, options):
        """Generate population comparison reports."""
        
        comparisons = PopulationComparison.objects.all().order_by('-comparison_date')[:5]
        
        if not comparisons:
            self.stdout.write(
                self.style.WARNING("No population comparisons found")
            )
            return
        
        self.stdout.write(f"\n{self.style.HTTP_INFO('Population Comparison Reports:')}")
        
        for comparison in comparisons:
            try:
                report = report_generator.generate_comparison_report(comparison)
                
                self.stdout.write(f"\n--- {report.comparison_name} ---")
                self.stdout.write(f"Analyses Compared: {len(report.analyses_compared)}")
                self.stdout.write(f"Features Tested: {len(report.features_tested)}")
                self.stdout.write(f"Significant Differences: {len(report.significant_differences)}")
                
                if report.significant_differences and options['verbose']:
                    self.stdout.write("Key Differences:")
                    for diff in report.significant_differences[:3]:
                        self.stdout.write(
                            f"  {diff['feature']}: p={diff['p_value']:.4f}, "
                            f"effect={diff['effect_interpretation']}"
                        )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Failed to generate comparison report: {str(e)}")
                )
    
    def save_report_to_file(self, validation_report, filename):
        """Save report to JSON file."""
        
        with open(filename, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
    
    def get_status_style(self, status):
        """Get appropriate style for status."""
        
        status_styles = {
            'excellent': self.style.SUCCESS,
            'good': self.style.SUCCESS,
            'acceptable': self.style.WARNING,
            'poor': self.style.ERROR,
            'error': self.style.ERROR
        }
        
        return status_styles.get(status.lower(), self.style.NOTICE)