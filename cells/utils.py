"""
Utility Functions for Morphometric Analysis

This module provides utility functions used across the morphometric analysis pipeline,
including analysis execution, quality assessment, and statistical summary generation.
"""

import numpy as np
from typing import Dict, Optional, Union, Any
import logging

from .exceptions import (
    MorphometricAnalysisError, 
    ImageQualityError, 
    DataValidationError,
    DependencyError
)

logger = logging.getLogger(__name__)


def run_cell_analysis(analysis_id: int) -> bool:
    """
    Public function to run complete cell analysis pipeline.
    
    Args:
        analysis_id: ID of the CellAnalysis instance to process
        
    Returns:
        True if analysis completed successfully, False otherwise
        
    Raises:
        MorphometricAnalysisError: If analysis fails
    """
    try:
        # Import here to avoid circular import
        from .analysis import CellAnalysisProcessor
        
        processor = CellAnalysisProcessor(analysis_id)
        return processor.run_analysis()
        
    except Exception as e:
        logger.error(f"Failed to run cell analysis for ID {analysis_id}: {str(e)}")
        raise MorphometricAnalysisError(f"Analysis failed: {str(e)}")


def get_image_quality_summary(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive image quality summary for a given image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing quality metrics and assessment results
        
    Raises:
        ImageQualityError: If quality assessment fails
    """
    try:
        # Import here to avoid circular imports
        from .quality_assessment import ImageQualityAssessment
        
        # Try to import cellpose
        try:
            from cellpose.io import imread
        except ImportError:
            raise DependencyError("Cellpose is required for image loading")
        
        if not image_path or not isinstance(image_path, str):
            raise DataValidationError("Invalid image path provided")
            
        # Load image
        image_array = imread(image_path)
        
        # Convert to RGB if needed (remove alpha channel)
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        
        # Perform quality assessment
        quality_assessment = ImageQualityAssessment.assess_overall_quality(image_array)
        return quality_assessment
        
    except (ImageQualityError, DependencyError, DataValidationError):
        raise  # Re-raise our custom exceptions
    except Exception as e:
        logger.error(f"Error getting image quality summary for {image_path}: {str(e)}")
        return {
            'overall_score': 0,
            'quality_category': 'error',
            'error': str(e),
            'blur_score': 0,
            'contrast_score': 0,
            'noise_score': 0,
            'recommendations': ['Unable to assess image quality - check file format and integrity']
        }


def get_analysis_summary(analysis) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive summary statistics for a completed analysis including all morphometric features.
    
    Args:
        analysis: CellAnalysis instance (Django model)
        
    Returns:
        Dictionary containing statistical summaries of detected cells,
        or None if analysis is not completed or has no cells
        
    Raises:
        DataValidationError: If analysis data is invalid
    """
    try:
        # Validate analysis status
        if not hasattr(analysis, 'status'):
            raise DataValidationError("Analysis object missing status attribute")
            
        if analysis.status != 'completed':
            logger.info(f"Analysis {getattr(analysis, 'id', 'unknown')} is not completed (status: {analysis.status})")
            return None
        
        # Get detected cells
        if not hasattr(analysis, 'detected_cells'):
            raise DataValidationError("Analysis object missing detected_cells attribute")
            
        detected_cells = analysis.detected_cells.all()
        if not detected_cells.exists():
            logger.info(f"Analysis {getattr(analysis, 'id', 'unknown')} has no detected cells")
            return None
        
        # Initialize comprehensive data collectors
        morphometric_data = {
            # Basic morphometric features
            'areas': [cell.area for cell in detected_cells if cell.area is not None],
            'perimeters': [cell.perimeter for cell in detected_cells if cell.perimeter is not None],
            'circularities': [cell.circularity for cell in detected_cells if cell.circularity is not None],
            'eccentricities': [cell.eccentricity for cell in detected_cells if cell.eccentricity is not None],
            'solidities': [cell.solidity for cell in detected_cells if cell.solidity is not None],
            'extents': [cell.extent for cell in detected_cells if cell.extent is not None],
            'major_axis_lengths': [cell.major_axis_length for cell in detected_cells if cell.major_axis_length is not None],
            'minor_axis_lengths': [cell.minor_axis_length for cell in detected_cells if cell.minor_axis_length is not None],
            'aspect_ratios': [cell.aspect_ratio for cell in detected_cells if cell.aspect_ratio is not None],
            
            # GLCM texture features
            'glcm_contrasts': [cell.glcm_contrast for cell in detected_cells if cell.glcm_contrast is not None],
            'glcm_correlations': [cell.glcm_correlation for cell in detected_cells if cell.glcm_correlation is not None],
            'glcm_energies': [cell.glcm_energy for cell in detected_cells if cell.glcm_energy is not None],
            'glcm_homogeneities': [cell.glcm_homogeneity for cell in detected_cells if cell.glcm_homogeneity is not None],
            'glcm_entropies': [cell.glcm_entropy for cell in detected_cells if cell.glcm_entropy is not None],
            'glcm_variances': [cell.glcm_variance for cell in detected_cells if cell.glcm_variance is not None],
            
            # Intensity features
            'intensity_means': [cell.intensity_mean for cell in detected_cells if cell.intensity_mean is not None],
            'intensity_stds': [cell.intensity_std for cell in detected_cells if cell.intensity_std is not None],
            'intensity_variances': [cell.intensity_variance for cell in detected_cells if cell.intensity_variance is not None],
            'intensity_ranges': [cell.intensity_range for cell in detected_cells if cell.intensity_range is not None],
            'intensity_entropies': [cell.intensity_entropy for cell in detected_cells if cell.intensity_entropy is not None],
            'intensity_energies': [cell.intensity_energy for cell in detected_cells if cell.intensity_energy is not None],
            
            # Physical measurements (if scale available)
            'areas_microns': [cell.area_microns_sq for cell in detected_cells if cell.area_microns_sq is not None],
            'perimeters_microns': [cell.perimeter_microns for cell in detected_cells if cell.perimeter_microns is not None],
            'major_axis_lengths_microns': [cell.major_axis_length_microns for cell in detected_cells if cell.major_axis_length_microns is not None],
            'minor_axis_lengths_microns': [cell.minor_axis_length_microns for cell in detected_cells if cell.minor_axis_length_microns is not None],
        }
        
        # Validate we have basic data
        if not morphometric_data['areas']:
            logger.warning(f"Analysis {getattr(analysis, 'id', 'unknown')} has no valid area measurements")
            return None
        
        # Build comprehensive summary statistics
        summary = {
            'total_cells': len(morphometric_data['areas']),
            'scale_available': getattr(analysis.cell, 'scale_set', False),
            'pixels_per_micron': getattr(analysis.cell, 'pixels_per_micron', None) if getattr(analysis.cell, 'scale_set', False) else None,
        }
        
        # Add comprehensive statistics for all features
        for feature_name, values in morphometric_data.items():
            if values and not feature_name.endswith('_microns'):  # Handle microns separately
                summary[f'{feature_name[:-1]}_stats'] = _calculate_comprehensive_stats(values, feature_name)
        
        # Add physical measurements if scale is available
        if getattr(analysis.cell, 'scale_set', False):
            _add_comprehensive_physical_measurements(summary, morphometric_data)
        
        return summary
        
    except DataValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error generating analysis summary: {str(e)}")
        raise MorphometricAnalysisError(f"Failed to generate analysis summary: {str(e)}")


def _calculate_stats(values: list, metric_name: str) -> Dict[str, float]:
    """
    Calculate statistical summary for a list of values.
    
    Args:
        values: List of numerical values
        metric_name: Name of the metric (for logging)
        
    Returns:
        Dictionary with mean, std, min, max statistics
    """
    if not values:
        logger.warning(f"No values provided for {metric_name} statistics")
        return {}
    
    try:
        values_array = np.array(values)
        
        # Filter out invalid values
        valid_values = values_array[np.isfinite(values_array)]
        
        if len(valid_values) == 0:
            logger.warning(f"No valid values found for {metric_name} statistics")
            return {}
        
        return {
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'count': len(valid_values)
        }
        
    except Exception as e:
        logger.error(f"Error calculating statistics for {metric_name}: {str(e)}")
        return {}


def _calculate_comprehensive_stats(values: list, metric_name: str) -> Dict[str, float]:
    """
    Calculate comprehensive statistical summary for a list of values.
    
    Args:
        values: List of numerical values
        metric_name: Name of the metric (for logging)
        
    Returns:
        Dictionary with comprehensive statistics (mean, std, min, max, median, quartiles, CV, etc.)
    """
    if not values:
        logger.warning(f"No values provided for {metric_name} statistics")
        return {}
    
    try:
        values_array = np.array(values)
        
        # Filter out invalid values
        valid_values = values_array[np.isfinite(values_array)]
        
        if len(valid_values) == 0:
            logger.warning(f"No valid values found for {metric_name} statistics")
            return {}
        
        import statistics
        
        stats = {
            'count': len(valid_values),
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'median': float(np.median(valid_values)),
            'range': float(np.max(valid_values) - np.min(valid_values)),
        }
        
        # Add quartiles and IQR for datasets with multiple values
        if len(valid_values) > 1:
            stats.update({
                'q25': float(np.percentile(valid_values, 25)),
                'q75': float(np.percentile(valid_values, 75)),
                'iqr': float(np.percentile(valid_values, 75) - np.percentile(valid_values, 25)),
                'cv': float((np.std(valid_values) / np.mean(valid_values)) * 100) if np.mean(valid_values) != 0 else 0.0
            })
        else:
            stats.update({
                'q25': stats['median'],
                'q75': stats['median'],
                'iqr': 0.0,
                'cv': 0.0
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive statistics for {metric_name}: {str(e)}")
        return {}


def _add_physical_measurements(summary: Dict[str, Any], detected_cells) -> None:
    """
    Add physical measurements (in microns) to the summary.
    
    Args:
        summary: Summary dictionary to update
        detected_cells: QuerySet of DetectedCell instances
    """
    try:
        # Extract physical measurements
        areas_microns = [cell.area_microns_sq for cell in detected_cells 
                        if cell.area_microns_sq is not None]
        perimeters_microns = [cell.perimeter_microns for cell in detected_cells 
                             if cell.perimeter_microns is not None]
        major_axes_microns = [cell.major_axis_length_microns for cell in detected_cells 
                             if cell.major_axis_length_microns is not None]
        minor_axes_microns = [cell.minor_axis_length_microns for cell in detected_cells 
                             if cell.minor_axis_length_microns is not None]
        
        # Add to summary if data is available
        if areas_microns:
            summary['area_stats_microns'] = _calculate_stats(areas_microns, 'area_microns')
        
        if perimeters_microns:
            summary['perimeter_stats_microns'] = _calculate_stats(perimeters_microns, 'perimeter_microns')
        
        if major_axes_microns:
            summary['major_axis_stats_microns'] = _calculate_stats(major_axes_microns, 'major_axis_microns')
        
        if minor_axes_microns:
            summary['minor_axis_stats_microns'] = _calculate_stats(minor_axes_microns, 'minor_axis_microns')
            
    except Exception as e:
        logger.error(f"Error adding physical measurements to summary: {str(e)}")
        # Don't raise - just log the error and continue without physical measurements


def _add_comprehensive_physical_measurements(summary: Dict[str, Any], morphometric_data: Dict[str, list]) -> None:
    """
    Add comprehensive physical measurements (in microns) to the summary.
    
    Args:
        summary: Summary dictionary to update
        morphometric_data: Dictionary containing morphometric data including micron measurements
    """
    try:
        # Add physical measurements if data is available
        if morphometric_data['areas_microns']:
            summary['area_stats_microns'] = _calculate_comprehensive_stats(
                morphometric_data['areas_microns'], 'area_microns'
            )
        
        if morphometric_data['perimeters_microns']:
            summary['perimeter_stats_microns'] = _calculate_comprehensive_stats(
                morphometric_data['perimeters_microns'], 'perimeter_microns'
            )
        
        if morphometric_data['major_axis_lengths_microns']:
            summary['major_axis_stats_microns'] = _calculate_comprehensive_stats(
                morphometric_data['major_axis_lengths_microns'], 'major_axis_microns'
            )
        
        if morphometric_data['minor_axis_lengths_microns']:
            summary['minor_axis_stats_microns'] = _calculate_comprehensive_stats(
                morphometric_data['minor_axis_lengths_microns'], 'minor_axis_microns'
            )
            
    except Exception as e:
        logger.error(f"Error adding comprehensive physical measurements to summary: {str(e)}")
        # Don't raise - just log the error and continue without physical measurements


def validate_image_array(image: np.ndarray, min_size: int = 10) -> None:
    """
    Validate image array for processing.
    
    Args:
        image: Image array to validate
        min_size: Minimum required size for each dimension
        
    Raises:
        DataValidationError: If image is invalid
    """
    if image is None:
        raise DataValidationError("Image array is None")
    
    if not isinstance(image, np.ndarray):
        raise DataValidationError("Image must be a numpy array")
    
    if image.size == 0:
        raise DataValidationError("Image array is empty")
    
    if len(image.shape) < 2:
        raise DataValidationError("Image must be at least 2-dimensional")
    
    if any(dim < min_size for dim in image.shape[:2]):
        raise DataValidationError(f"Image dimensions {image.shape[:2]} are too small (minimum: {min_size}x{min_size})")
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise DataValidationError(f"Invalid number of channels: {image.shape[2]} (expected 1, 3, or 4)")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    except (ZeroDivisionError, TypeError, ValueError):
        return default