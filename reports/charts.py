"""
Statistical chart generation utilities for morphometric analysis reports.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set matplotlib and seaborn style for professional reports
plt.style.use('default')
sns.set_palette("husl")

class MorphometricChartGenerator:
    """Generate statistical charts for morphometric analysis data."""
    
    def __init__(self, dpi: int = 300, figsize: Tuple[float, float] = (10, 6)):
        """
        Initialize chart generator.
        
        Args:
            dpi: Resolution for chart images
            figsize: Default figure size (width, height) in inches
        """
        self.dpi = dpi
        self.figsize = figsize
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#198754',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'info': '#0DCAF0',
            'gray': '#6C757D'
        }
    
    def create_distribution_histogram(self, data: List[float], title: str, 
                                    xlabel: str, bins: int = 30) -> io.BytesIO:
        """
        Create a histogram showing the distribution of morphometric values.
        
        Args:
            data: List of numerical values
            title: Chart title
            xlabel: X-axis label
            bins: Number of histogram bins
            
        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Create histogram with statistical overlay
            n, bins_arr, patches = ax.hist(data, bins=bins, alpha=0.7, 
                                         color=self.colors['primary'], 
                                         edgecolor='white', linewidth=0.5)
            
            # Add statistical information
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color=self.colors['danger'], linestyle='--', 
                      linewidth=2, label=f'Среднее: {mean_val:.2f}')
            ax.axvline(median_val, color=self.colors['warning'], linestyle='-', 
                      linewidth=2, label=f'Медиана: {median_val:.2f}')
            
            # Add normal distribution overlay
            x = np.linspace(min(data), max(data), 100)
            y = ((1/(std_val * np.sqrt(2 * np.pi))) * 
                 np.exp(-0.5 * ((x - mean_val) / std_val) ** 2))
            # Scale to histogram
            y = y * len(data) * (bins_arr[1] - bins_arr[0])
            ax.plot(x, y, color=self.colors['secondary'], linewidth=2, 
                   label='Нормальное распределение')
            
            # Styling
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Частота', fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'n = {len(data)}\nСреднее = {mean_val:.2f}\nСтд.откл. = {std_val:.2f}\nМедиана = {median_val:.2f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating distribution histogram: {e}")
            try:
                return self._create_error_chart(title)
            except:
                # If even error chart fails, return None
                return None
    
    def create_correlation_matrix(self, data_dict: Dict[str, List[float]], 
                                title: str = "Корреляционная матрица") -> io.BytesIO:
        """
        Create a correlation matrix heatmap.
        
        Args:
            data_dict: Dictionary of parameter names and their values
            title: Chart title
            
        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            correlation_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                       fmt='.2f', ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            try:
                return self._create_error_chart(title)
            except:
                return None
    
    def create_box_plot(self, data_dict: Dict[str, List[float]], 
                       title: str, ylabel: str) -> io.BytesIO:
        """
        Create box plots for multiple parameters.
        
        Args:
            data_dict: Dictionary of parameter names and their values
            title: Chart title
            ylabel: Y-axis label
            
        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Prepare data for box plot
            data_list = list(data_dict.values())
            labels = list(data_dict.keys())
            
            # Create box plot
            box_plot = ax.boxplot(data_list, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['success'], self.colors['warning']]
            for patch, color in zip(box_plot['boxes'], colors * len(box_plot['boxes'])):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Styling
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate labels if needed
            if len(max(labels, key=len)) > 8:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            try:
                return self._create_error_chart(title)
            except:
                return None
    
    def create_scatter_plot(self, x_data: List[float], y_data: List[float],
                          title: str, xlabel: str, ylabel: str,
                          add_regression: bool = True) -> io.BytesIO:
        """
        Create scatter plot with optional regression line.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            add_regression: Whether to add regression line
            
        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Create scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, color=self.colors['primary'],
                      s=30, edgecolors='white', linewidth=0.5)
            
            # Add regression line if requested
            if add_regression and len(x_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), color=self.colors['danger'],
                       linewidth=2, linestyle='--', alpha=0.8)
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Styling
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return self._create_error_chart(title)
    
    def create_quality_metrics_chart(self, metrics: Dict[str, float],
                                   title: str = "Метрики качества") -> io.BytesIO:
        """
        Create a radar chart for quality metrics.
        
        Args:
            metrics: Dictionary of metric names and values (0-1 scale)
            title: Chart title
            
        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi, 
                                 subplot_kw=dict(projection='polar'))
            
            # Prepare data
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            # Number of variables
            N = len(categories)
            
            # Angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Values
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   color=self.colors['primary'])
            ax.fill(angles, values, color=self.colors['primary'], alpha=0.25)
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            
            # Add grid
            ax.grid(True)
            
            # Title
            ax.set_title(title, size=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating quality metrics chart: {e}")
            return self._create_error_chart(title)
    
    def create_summary_statistics_table(self, data_dict: Dict[str, List[float]],
                                      title: str = "Сводная статистика") -> io.BytesIO:
        """
        Create a visual table of summary statistics.
        
        Args:
            data_dict: Dictionary of parameter names and their values
            title: Table title
            
        Returns:
            BytesIO buffer containing the table image
        """
        try:
            # Calculate statistics
            stats = []
            for param_name, values in data_dict.items():
                stats.append({
                    'Параметр': param_name,
                    'Количество': len(values),
                    'Среднее': np.mean(values),
                    'Стд.откл.': np.std(values),
                    'Мин': np.min(values),
                    'Q25': np.percentile(values, 25),
                    'Медиана': np.median(values),
                    'Q75': np.percentile(values, 75),
                    'Макс': np.max(values)
                })
            
            df = pd.DataFrame(stats)
            
            fig, ax = plt.subplots(figsize=(12, len(data_dict) * 0.5 + 2), dpi=self.dpi)
            ax.axis('tight')
            ax.axis('off')
            
            # Create table
            table = ax.table(cellText=df.round(3).values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Header styling
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor(self.colors['primary'])
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(df) + 1):
                for j in range(len(df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f8f9fa')
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating statistics table: {e}")
            return self._create_error_chart(title)
    
    def _create_error_chart(self, title: str) -> io.BytesIO:
        """
        Create an error chart when chart generation fails.
        
        Args:
            title: Original chart title
            
        Returns:
            BytesIO buffer containing error chart
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.text(0.5, 0.5, 'Ошибка создания диаграммы', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color=self.colors['danger'])
        ax.set_title(f"Ошибка: {title}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)
        
        return buffer