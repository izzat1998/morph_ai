"""
Confidence Interval Visualization Service

This module provides visualization tools for confidence intervals and
statistical analysis results in morphometric analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from ..models import StatisticalAnalysis, FeatureStatistics
from .confidence_intervals import ConfidenceInterval, CIAnalysisResult

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class CIVisualizer:
    """
    Confidence Interval Visualization Service
    
    Creates publication-ready visualizations of confidence intervals,
    uncertainty analysis, and statistical comparisons.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the CI visualizer
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'error': '#FF6B6B',
            'ci_fill': '#E8F4FD',
            'ci_border': '#2E86AB'
        }
        
        # Set up matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 10,
            'font.family': 'serif',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })
    
    def plot_feature_confidence_intervals(self,
                                        ci_result: CIAnalysisResult,
                                        title: str = "Feature Confidence Intervals",
                                        save_path: Optional[str] = None) -> str:
        """
        Create forest plot of confidence intervals for multiple features
        
        Args:
            ci_result: CIAnalysisResult containing confidence intervals
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            fig, ax = plt.subplots(figsize=(12, max(6, len(ci_result.confidence_intervals) * 0.8)))
            
            features = list(ci_result.confidence_intervals.keys())
            y_positions = range(len(features))
            
            # Plot confidence intervals
            for i, (feature_name, ci) in enumerate(ci_result.confidence_intervals.items()):
                y_pos = len(features) - i - 1  # Reverse order for better readability
                
                # Point estimate
                ax.plot(ci.point_estimate, y_pos, 'o', 
                       color=self.colors['primary'], markersize=8, zorder=3)
                
                # Confidence interval
                ax.plot([ci.lower_bound, ci.upper_bound], [y_pos, y_pos], 
                       color=self.colors['ci_border'], linewidth=3, zorder=2)
                
                # CI caps
                cap_height = 0.1
                ax.plot([ci.lower_bound, ci.lower_bound], 
                       [y_pos - cap_height, y_pos + cap_height],
                       color=self.colors['ci_border'], linewidth=2, zorder=2)
                ax.plot([ci.upper_bound, ci.upper_bound], 
                       [y_pos - cap_height, y_pos + cap_height],
                       color=self.colors['ci_border'], linewidth=2, zorder=2)
                
                # Add confidence interval text
                ci_width = ci.upper_bound - ci.lower_bound
                relative_width = (ci_width / abs(ci.point_estimate) * 100) if ci.point_estimate != 0 else 0
                
                ax.text(max(ci.upper_bound, max([c.upper_bound for c in ci_result.confidence_intervals.values()])) * 1.05,
                       y_pos, f'[{ci.lower_bound:.3f}, {ci.upper_bound:.3f}]\n±{relative_width:.1f}%',
                       va='center', fontsize=9, color='black')
            
            # Customize plot
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f.replace('_', ' ').title() for f in reversed(features)])
            ax.set_xlabel('Feature Value')
            ax.set_title(f'{title}\n{int(ci_result.confidence_intervals[features[0]].confidence_level*100)}% Confidence Intervals', 
                        fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            
            # Add vertical line at zero if appropriate
            x_min = min([ci.lower_bound for ci in ci_result.confidence_intervals.values()])
            x_max = max([ci.upper_bound for ci in ci_result.confidence_intervals.values()])
            
            if x_min < 0 < x_max:
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"CI forest plot saved to {save_path}")
            else:
                save_path = f"feature_confidence_intervals_{np.random.randint(1000, 9999)}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create confidence interval plot: {str(e)}")
            plt.close()
            return ""
    
    def plot_uncertainty_analysis(self,
                                ci_result: CIAnalysisResult,
                                title: str = "Measurement Uncertainty Analysis",
                                save_path: Optional[str] = None) -> str:
        """
        Create uncertainty analysis visualization
        
        Args:
            ci_result: CIAnalysisResult containing uncertainty information
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            features = list(ci_result.confidence_intervals.keys())
            relative_widths = []
            quality_scores = []
            
            for feature_name in features:
                ci = ci_result.confidence_intervals[feature_name]
                relative_widths.append(ci.relative_width * 100 if ci.relative_width else 0)
                quality_scores.append(ci_result.quality_assessment.get(feature_name, 0.5))
            
            # Plot 1: Relative uncertainty
            bars1 = ax1.bar(range(len(features)), relative_widths, 
                           color=self.colors['primary'], alpha=0.7, edgecolor='black')
            
            # Color bars based on uncertainty level
            for i, (bar, width) in enumerate(zip(bars1, relative_widths)):
                if width <= 10:
                    bar.set_color(self.colors['success'])
                elif width <= 20:
                    bar.set_color(self.colors['accent'])
                else:
                    bar.set_color(self.colors['error'])
            
            ax1.set_xticks(range(len(features)))
            ax1.set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45, ha='right')
            ax1.set_ylabel('Relative CI Width (%)')
            ax1.set_title('Relative Uncertainty by Feature')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add reference lines
            ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Good (<10%)')
            ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<20%)')
            ax1.legend()
            
            # Plot 2: Quality assessment
            bars2 = ax2.bar(range(len(features)), quality_scores, 
                           color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            
            # Color bars based on quality score
            for i, (bar, score) in enumerate(zip(bars2, quality_scores)):
                if score >= 0.8:
                    bar.set_color(self.colors['success'])
                elif score >= 0.6:
                    bar.set_color(self.colors['accent'])
                else:
                    bar.set_color(self.colors['error'])
            
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45, ha='right')
            ax2.set_ylabel('Quality Score')
            ax2.set_title('Measurement Quality by Feature')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add reference lines
            ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.8)')
            ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.6)')
            ax2.legend()
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Uncertainty analysis plot saved to {save_path}")
            else:
                save_path = f"uncertainty_analysis_{np.random.randint(1000, 9999)}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create uncertainty analysis plot: {str(e)}")
            plt.close()
            return ""
    
    def plot_bootstrap_distributions(self,
                                   bootstrap_results: Dict[str, Any],
                                   title: str = "Bootstrap Distributions",
                                   save_path: Optional[str] = None) -> str:
        """
        Create visualization of bootstrap distributions
        
        Args:
            bootstrap_results: Dictionary containing bootstrap results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            n_features = len(bootstrap_results)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (feature_name, result) in enumerate(bootstrap_results.items()):
                ax = axes[i] if n_features > 1 else axes[0]
                
                # Plot histogram of bootstrap distribution
                if hasattr(result, 'bootstrap_distribution'):
                    bootstrap_values = result.bootstrap_distribution
                    
                    ax.hist(bootstrap_values, bins=50, alpha=0.7, 
                           color=self.colors['primary'], edgecolor='black', density=True)
                    
                    # Add original value
                    ax.axvline(result.original_value, color='red', linestyle='--', 
                              linewidth=2, label=f'Original: {result.original_value:.3f}')
                    
                    # Add confidence interval
                    ax.axvline(result.confidence_interval_lower, color='orange', 
                              linestyle=':', linewidth=2, alpha=0.8)
                    ax.axvline(result.confidence_interval_upper, color='orange', 
                              linestyle=':', linewidth=2, alpha=0.8,
                              label=f'CI: [{result.confidence_interval_lower:.3f}, {result.confidence_interval_upper:.3f}]')
                    
                    # Add bootstrap mean
                    ax.axvline(result.bootstrap_mean, color='green', linestyle='-', 
                              linewidth=2, alpha=0.8, label=f'Bootstrap Mean: {result.bootstrap_mean:.3f}')
                    
                    ax.set_title(f'{feature_name.replace("_", " ").title()}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No bootstrap data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{feature_name.replace("_", " ").title()}')
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Bootstrap distribution plot saved to {save_path}")
            else:
                save_path = f"bootstrap_distributions_{np.random.randint(1000, 9999)}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create bootstrap distribution plot: {str(e)}")
            plt.close()
            return ""
    
    def plot_population_comparison(self,
                                 population_data: Dict[str, List[float]],
                                 title: str = "Population Comparison",
                                 save_path: Optional[str] = None) -> str:
        """
        Create box plot comparison of multiple populations
        
        Args:
            population_data: Dictionary of population_name -> [values]
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            n_features = len(population_data)
            n_cols = min(2, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, (feature_name, values) in enumerate(population_data.items()):
                ax = axes[i] if n_features > 1 else axes[0]
                
                # Create box plot
                bp = ax.boxplot(values, patch_artist=True, notch=True)
                
                # Customize box plot colors
                for patch in bp['boxes']:
                    patch.set_facecolor(self.colors['primary'])
                    patch.set_alpha(0.7)
                
                for whisker in bp['whiskers']:
                    whisker.set_color(self.colors['ci_border'])
                    whisker.set_linewidth(2)
                
                for cap in bp['caps']:
                    cap.set_color(self.colors['ci_border'])
                    cap.set_linewidth(2)
                
                for median in bp['medians']:
                    median.set_color('red')
                    median.set_linewidth(2)
                
                # Add statistics text
                mean_val = np.mean(values)
                std_val = np.std(values)
                n_samples = len(values)
                
                stats_text = f'n = {n_samples}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{feature_name.replace("_", " ").title()}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Population comparison plot saved to {save_path}")
            else:
                save_path = f"population_comparison_{np.random.randint(1000, 9999)}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create population comparison plot: {str(e)}")
            plt.close()
            return ""
    
    def create_statistical_dashboard(self,
                                   statistical_analysis: StatisticalAnalysis,
                                   save_path: Optional[str] = None) -> str:
        """
        Create comprehensive statistical analysis dashboard
        
        Args:
            statistical_analysis: StatisticalAnalysis instance
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard file
        """
        try:
            feature_stats = statistical_analysis.feature_stats.all()
            
            if not feature_stats.exists():
                logger.warning("No feature statistics found for dashboard creation")
                return ""
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Extract data
            features = []
            point_estimates = []
            ci_lowers = []
            ci_uppers = []
            uncertainties = []
            quality_scores = []
            
            for fs in feature_stats:
                features.append(fs.feature_name.replace('_', ' ').title())
                point_estimates.append(fs.measured_value)
                ci_lowers.append(fs.confidence_interval_lower)
                ci_uppers.append(fs.confidence_interval_upper)
                uncertainties.append(fs.uncertainty_percent)
                quality_scores.append(fs.measurement_reliability_score)
            
            # 1. Confidence intervals forest plot
            ax1 = fig.add_subplot(gs[0, :2])
            y_pos = range(len(features))
            
            for i in range(len(features)):
                ax1.plot(point_estimates[i], i, 'o', color=self.colors['primary'], markersize=8)
                ax1.plot([ci_lowers[i], ci_uppers[i]], [i, i], 
                        color=self.colors['ci_border'], linewidth=3)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Feature Value')
            ax1.set_title('Confidence Intervals')
            ax1.grid(True, alpha=0.3)
            
            # 2. Uncertainty levels
            ax2 = fig.add_subplot(gs[0, 2:])
            bars = ax2.bar(range(len(features)), uncertainties, color=self.colors['accent'], alpha=0.7)
            
            # Color bars by uncertainty level
            for bar, unc in zip(bars, uncertainties):
                if unc <= 5:
                    bar.set_color(self.colors['success'])
                elif unc <= 15:
                    bar.set_color(self.colors['accent'])
                else:
                    bar.set_color(self.colors['error'])
            
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels(features, rotation=45, ha='right')
            ax2.set_ylabel('Uncertainty (%)')
            ax2.set_title('Measurement Uncertainty')
            ax2.grid(True, alpha=0.3)
            
            # 3. Quality scores
            ax3 = fig.add_subplot(gs[1, :2])
            bars = ax3.bar(range(len(features)), quality_scores, color=self.colors['secondary'], alpha=0.7)
            
            for bar, score in zip(bars, quality_scores):
                if score >= 0.8:
                    bar.set_color(self.colors['success'])
                elif score >= 0.6:
                    bar.set_color(self.colors['accent'])
                else:
                    bar.set_color(self.colors['error'])
            
            ax3.set_xticks(range(len(features)))
            ax3.set_xticklabels(features, rotation=45, ha='right')
            ax3.set_ylabel('Quality Score')
            ax3.set_title('Measurement Quality')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics table
            ax4 = fig.add_subplot(gs[1, 2:])
            ax4.axis('off')
            
            summary_data = [
                ['Feature', 'Value', 'CI Width', 'Uncertainty', 'Quality'],
                ['', '', '', '(%)', 'Score']
            ]
            
            for i in range(len(features)):
                ci_width = ci_uppers[i] - ci_lowers[i]
                row = [
                    features[i][:15] + '...' if len(features[i]) > 15 else features[i],
                    f'{point_estimates[i]:.3f}',
                    f'{ci_width:.3f}',
                    f'{uncertainties[i]:.1f}',
                    f'{quality_scores[i]:.2f}'
                ]
                summary_data.append(row)
            
            table = ax4.table(cellText=summary_data[2:], colLabels=summary_data[0],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Color table cells based on quality
            for i in range(len(features)):
                if quality_scores[i] >= 0.8:
                    table[(i+1, 4)].set_facecolor('#90EE90')  # Light green
                elif quality_scores[i] >= 0.6:
                    table[(i+1, 4)].set_facecolor('#FFE4B5')  # Light orange
                else:
                    table[(i+1, 4)].set_facecolor('#FFB6C1')  # Light red
            
            ax4.set_title('Summary Statistics')
            
            # 5. Analysis information
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            
            info_text = f"""
            Statistical Analysis Summary
            
            Confidence Level: {statistical_analysis.confidence_level*100:.0f}%
            Bootstrap Iterations: {statistical_analysis.bootstrap_iterations:,}
            Computation Time: {statistical_analysis.computation_time_seconds:.2f} seconds
            Total Features Analyzed: {len(features)}
            
            High Quality Features (≥0.8): {sum(1 for score in quality_scores if score >= 0.8)}
            Low Uncertainty Features (≤5%): {sum(1 for unc in uncertainties if unc <= 5)}
            
            Analysis Date: {statistical_analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # Main title
            fig.suptitle(f'Statistical Analysis Dashboard - Analysis {statistical_analysis.id}',
                        fontsize=18, fontweight='bold')
            
            # Save dashboard
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Statistical dashboard saved to {save_path}")
            else:
                save_path = f"statistical_dashboard_{statistical_analysis.id}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create statistical dashboard: {str(e)}")
            plt.close()
            return ""