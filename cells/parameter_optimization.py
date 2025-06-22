
"""
Parameter Optimization Module

This module provides automatic parameter optimization for Cellpose segmentation
based on image characteristics and quality metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

try:
    from skimage import filters, feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from .exceptions import ParameterOptimizationError, DependencyError
from .utils import validate_image_array, safe_divide

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Automatic parameter optimization for Cellpose based on image characteristics.
    
    This class provides methods to automatically estimate optimal parameters
    for Cellpose segmentation including diameter, thresholds, and model selection.
    """
    
    # Valid Cellpose models
    VALID_MODELS = ['cyto', 'nuclei', 'cyto2', 'cyto3']
    
    # Parameter ranges
    DIAMETER_RANGE = (5, 500)
    FLOW_THRESHOLD_RANGE = (0.1, 3.0)
    CELLPROB_THRESHOLD_RANGE = (-6.0, 6.0)
    
    def __init__(self):
        """Initialize the parameter optimizer."""
        if not SKIMAGE_AVAILABLE:
            raise DependencyError("scikit-image is required for parameter optimization")
        
        logger.info("ParameterOptimizer initialized")
    
    @staticmethod
    def estimate_cell_diameter(
        image_array: np.ndarray, 
        sample_size: int = 5,
        min_sigma: float = 3.0,
        max_sigma: float = 50.0
    ) -> float:
        """
        Estimate optimal cell diameter using blob detection and statistical analysis.
        
        Args:
            image_array: Input image array (2D grayscale or 3D RGB)
            sample_size: Number of samples for analysis (currently not used)
            min_sigma: Minimum sigma for blob detection
            max_sigma: Maximum sigma for blob detection
            
        Returns:
            Estimated cell diameter in pixels
            
        Raises:
            ParameterOptimizationError: If diameter estimation fails
        """
        try:
            validate_image_array(image_array, min_size=20)
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array.astype(np.uint8)
            
            logger.debug(f"Estimating cell diameter for image shape: {gray.shape}")
            
            # Apply gentle preprocessing for better blob detection
            smoothed = filters.gaussian(gray, sigma=1.0)
            
            # Use multiple blob detection methods for robustness
            blobs_log = feature.blob_log(
                smoothed, 
                min_sigma=min_sigma, 
                max_sigma=max_sigma, 
                num_sigma=20, 
                threshold=0.02
            )
            blobs_dog = feature.blob_dog(
                smoothed, 
                min_sigma=min_sigma, 
                max_sigma=max_sigma, 
                sigma_ratio=1.6, 
                threshold=0.02
            )
            
            # Combine results
            all_blobs = []
            if len(blobs_log) > 0:
                # Convert LoG blobs to diameter (radius * 2 * sqrt(2))
                diameters_log = blobs_log[:, 2] * 2 * np.sqrt(2)
                all_blobs.extend(diameters_log)
                logger.debug(f"LoG detected {len(blobs_log)} blobs")
            
            if len(blobs_dog) > 0:
                # Convert DoG blobs to diameter
                diameters_dog = blobs_dog[:, 2] * 2 * np.sqrt(2)
                all_blobs.extend(diameters_dog)
                logger.debug(f"DoG detected {len(blobs_dog)} blobs")
            
            if len(all_blobs) == 0:
                # Fallback: estimate from image dimensions
                min_dim = min(gray.shape)
                estimated_diameter = min_dim / 10  # Assume cells are roughly 1/10 of image size
                fallback_diameter = max(20, min(100, estimated_diameter))
                logger.warning(f"No blobs detected, using fallback diameter: {fallback_diameter}")
                return float(fallback_diameter)
            
            # Statistical analysis of detected blob sizes
            all_blobs = np.array(all_blobs)
            
            # Remove outliers (beyond 2 standard deviations)
            mean_diameter = np.mean(all_blobs)
            std_diameter = np.std(all_blobs)
            
            if std_diameter > 0:
                filtered_blobs = all_blobs[np.abs(all_blobs - mean_diameter) <= 2 * std_diameter]
            else:
                filtered_blobs = all_blobs
            
            if len(filtered_blobs) == 0:
                estimated_diameter = mean_diameter
            else:
                # Return median for robustness
                estimated_diameter = np.median(filtered_blobs)
            
            # Clamp to reasonable range
            final_diameter = max(
                ParameterOptimizer.DIAMETER_RANGE[0], 
                min(ParameterOptimizer.DIAMETER_RANGE[1], estimated_diameter)
            )
            
            logger.info(f"Estimated cell diameter: {final_diameter:.1f} pixels "
                       f"(from {len(all_blobs)} blobs, {len(filtered_blobs)} after filtering)")
            
            return float(final_diameter)
            
        except Exception as e:
            logger.error(f"Error estimating cell diameter: {str(e)}")
            raise ParameterOptimizationError(f"Failed to estimate cell diameter: {str(e)}")
    
    @staticmethod
    def optimize_thresholds(
        image_array: np.ndarray, 
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Union[float, str]]:
        """
        Optimize flow and cellprob thresholds based on image quality.
        
        Args:
            image_array: Input image array
            quality_metrics: Dictionary containing image quality assessment results
            
        Returns:
            Dictionary containing optimized thresholds and reasoning
            
        Raises:
            ParameterOptimizationError: If threshold optimization fails
        """
        try:
            validate_image_array(image_array)
            
            if not isinstance(quality_metrics, dict):
                raise ParameterOptimizationError("Quality metrics must be a dictionary")
            
            # Extract quality scores with safe defaults
            overall_score = quality_metrics.get('overall_score', 50)
            blur_metrics = quality_metrics.get('blur_metrics', {})
            contrast_metrics = quality_metrics.get('contrast_metrics', {})
            noise_metrics = quality_metrics.get('noise_metrics', {})
            
            blur_score = blur_metrics.get('blur_score', 50) if isinstance(blur_metrics, dict) else 50
            contrast_score = contrast_metrics.get('contrast_score', 50) if isinstance(contrast_metrics, dict) else 50
            noise_score = noise_metrics.get('noise_score', 50) if isinstance(noise_metrics, dict) else 50
            
            logger.debug(f"Optimizing thresholds for quality scores - "
                        f"overall: {overall_score}, blur: {blur_score}, "
                        f"contrast: {contrast_score}, noise: {noise_score}")
            
            # Default values
            flow_threshold = 0.4
            cellprob_threshold = 0.0
            
            # Adjust based on overall image quality
            if overall_score < 40:  # Poor quality
                flow_threshold = 0.8  # More permissive for poor quality
                cellprob_threshold = -1.0
            elif overall_score < 60:  # Fair quality
                flow_threshold = 0.6
                cellprob_threshold = -0.5
            elif overall_score < 80:  # Good quality
                flow_threshold = 0.4
                cellprob_threshold = 0.0
            else:  # Excellent quality
                flow_threshold = 0.3  # More strict for high quality
                cellprob_threshold = 0.5
            
            # Fine-tune based on specific metrics
            if blur_score < 30:  # Very blurry
                flow_threshold += 0.2
                cellprob_threshold -= 0.5
            
            if contrast_score < 30:  # Very low contrast
                flow_threshold += 0.1
                cellprob_threshold -= 0.3
            
            if noise_score < 30:  # Very noisy
                flow_threshold += 0.1
                cellprob_threshold -= 0.2
            
            # Clamp to valid ranges
            flow_threshold = max(
                ParameterOptimizer.FLOW_THRESHOLD_RANGE[0], 
                min(ParameterOptimizer.FLOW_THRESHOLD_RANGE[1], flow_threshold)
            )
            cellprob_threshold = max(
                ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[0], 
                min(ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[1], cellprob_threshold)
            )
            
            reasoning = (f"Optimized for image quality score {overall_score:.1f} "
                        f"(blur: {blur_score:.1f}, contrast: {contrast_score:.1f}, "
                        f"noise: {noise_score:.1f})")
            
            logger.info(f"Optimized thresholds - flow: {flow_threshold:.2f}, "
                       f"cellprob: {cellprob_threshold:.2f}")
            
            return {
                'flow_threshold': float(flow_threshold),
                'cellprob_threshold': float(cellprob_threshold),
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error optimizing thresholds: {str(e)}")
            raise ParameterOptimizationError(f"Failed to optimize thresholds: {str(e)}")
    
    @staticmethod
    def select_optimal_model(
        image_array: np.ndarray, 
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Union[str, float]]:
        """
        Select the best Cellpose model based on image characteristics.
        
        Args:
            image_array: Input image array
            quality_metrics: Dictionary containing image quality assessment results
            
        Returns:
            Dictionary containing recommended model, confidence, and reasoning
            
        Raises:
            ParameterOptimizationError: If model selection fails
        """
        try:
            validate_image_array(image_array)
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array.astype(np.uint8)
            
            logger.debug(f"Selecting optimal model for image shape: {gray.shape}")
            
            # Analyze image characteristics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Check if image looks like nuclei (high contrast, round objects)
            # Use edge detection to estimate roundness
            edges = feature.canny(gray, sigma=2.0)
            edge_density = np.sum(edges) / edges.size
            
            # Estimate texture using local binary patterns
            texture_uniformity = std_intensity  # Default fallback
            try:
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
                texture_uniformity = np.std(lbp)
            except ImportError:
                logger.debug("Local binary pattern not available, using intensity std as texture measure")
            except Exception as e:
                logger.warning(f"Error computing local binary pattern: {str(e)}")
            
            logger.debug(f"Image characteristics - intensity: {mean_intensity:.1f}, "
                        f"edges: {edge_density:.4f}, texture: {texture_uniformity:.1f}")
            
            # Decision logic for model selection
            if mean_intensity > 150 and edge_density > 0.05 and texture_uniformity < 50:
                # High contrast, well-defined edges, uniform texture -> likely nuclei
                recommended_model = 'nuclei'
                confidence = 0.8
                reasoning = "High contrast with uniform texture suggests nuclear staining"
            elif mean_intensity < 100 and edge_density < 0.03:
                # Low contrast, fewer edges -> might be cytoplasm
                recommended_model = 'cyto2'  # More robust version
                confidence = 0.6
                reasoning = "Low contrast with diffuse edges suggests cytoplasmic staining"
            elif edge_density > 0.1:
                # High edge density -> complex structures
                recommended_model = 'cyto2'
                confidence = 0.7
                reasoning = "High edge density suggests complex cellular structures"
            else:
                # Default to cyto2 for general purpose
                recommended_model = 'cyto2'
                confidence = 0.5
                reasoning = "Default selection for general cellular morphology"
            
            # Validate selected model
            if recommended_model not in ParameterOptimizer.VALID_MODELS:
                logger.warning(f"Invalid model selected: {recommended_model}, defaulting to cyto2")
                recommended_model = 'cyto2'
                confidence = 0.3
                reasoning += " (fallback due to invalid selection)"
            
            full_reasoning = (f"{reasoning}. Based on intensity={mean_intensity:.1f}, "
                            f"edges={edge_density:.3f}, texture={texture_uniformity:.1f}")
            
            logger.info(f"Selected model: {recommended_model} (confidence: {confidence:.2f})")
            
            return {
                'recommended_model': recommended_model,
                'confidence': float(confidence),
                'reasoning': full_reasoning
            }
            
        except Exception as e:
            logger.error(f"Error selecting optimal model: {str(e)}")
            raise ParameterOptimizationError(f"Failed to select optimal model: {str(e)}")
    
    @staticmethod
    def optimize_all_parameters(
        image_array: np.ndarray, 
        quality_metrics: Dict[str, Any], 
        current_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive parameter optimization combining all methods.
        
        Args:
            image_array: Input image array
            quality_metrics: Dictionary containing image quality assessment results
            current_params: Current parameter values (optional)
            
        Returns:
            Dictionary containing all optimized parameters and metadata
            
        Raises:
            ParameterOptimizationError: If comprehensive optimization fails
        """
        try:
            validate_image_array(image_array)
            
            current_params = current_params or {}
            
            logger.info("Starting comprehensive parameter optimization")
            
            # Estimate optimal diameter
            optimal_diameter = ParameterOptimizer.estimate_cell_diameter(image_array)
            
            # Optimize thresholds
            threshold_optimization = ParameterOptimizer.optimize_thresholds(
                image_array, quality_metrics
            )
            
            # Select optimal model
            model_optimization = ParameterOptimizer.select_optimal_model(
                image_array, quality_metrics
            )
            
            # Compile recommendations
            recommendations = {
                'cellpose_diameter': optimal_diameter,
                'flow_threshold': threshold_optimization['flow_threshold'],
                'cellprob_threshold': threshold_optimization['cellprob_threshold'],
                'cellpose_model': model_optimization['recommended_model'],
                'confidence_scores': {
                    'diameter_confidence': 0.7,  # Blob detection is fairly reliable
                    'threshold_confidence': 0.8,  # Quality-based optimization is well-tested
                    'model_confidence': model_optimization['confidence']
                },
                'optimization_notes': [
                    f"Estimated diameter: {optimal_diameter:.1f} pixels",
                    threshold_optimization['reasoning'],
                    model_optimization['reasoning']
                ],
                'parameter_bounds': {
                    'diameter_range': ParameterOptimizer.DIAMETER_RANGE,
                    'flow_threshold_range': ParameterOptimizer.FLOW_THRESHOLD_RANGE,
                    'cellprob_threshold_range': ParameterOptimizer.CELLPROB_THRESHOLD_RANGE,
                    'valid_models': ParameterOptimizer.VALID_MODELS
                }
            }
            
            # Add comparison with current parameters if provided
            if current_params:
                changes = []
                for param in ['cellpose_diameter', 'flow_threshold', 'cellprob_threshold', 'cellpose_model']:
                    old_val = current_params.get(param)
                    new_val = recommendations.get(param)
                    if old_val is not None and old_val != new_val:
                        changes.append(f"{param}: {old_val} → {new_val}")
                
                recommendations['parameter_changes'] = changes
                logger.info(f"Parameter changes: {len(changes)} parameters updated")
            
            overall_confidence = np.mean(list(recommendations['confidence_scores'].values()))
            recommendations['overall_confidence'] = float(overall_confidence)
            
            logger.info(f"Parameter optimization completed. Overall confidence: {overall_confidence:.2f}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in comprehensive parameter optimization: {str(e)}")
            raise ParameterOptimizationError(f"Failed to optimize parameters: {str(e)}")
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """
        Get default Cellpose parameters.
        
        Returns:
            Dictionary with default parameter values
        """
        return {
            'cellpose_diameter': 30.0,
            'flow_threshold': 0.4,
            'cellprob_threshold': 0.0,
            'cellpose_model': 'cyto2'
        }
    
    @staticmethod
    def validate_parameters(params: Dict[str, Any]) -> List[str]:
        """
        Validate Cellpose parameters.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Check diameter
        diameter = params.get('cellpose_diameter')
        if diameter is not None:
            if not isinstance(diameter, (int, float)) or diameter <= 0:
                errors.append("cellpose_diameter must be a positive number")
            elif not (ParameterOptimizer.DIAMETER_RANGE[0] <= diameter <= ParameterOptimizer.DIAMETER_RANGE[1]):
                errors.append(f"cellpose_diameter must be between {ParameterOptimizer.DIAMETER_RANGE[0]} and {ParameterOptimizer.DIAMETER_RANGE[1]}")
        
        # Check flow threshold
        flow_thresh = params.get('flow_threshold')
        if flow_thresh is not None:
            if not isinstance(flow_thresh, (int, float)):
                errors.append("flow_threshold must be a number")
            elif not (ParameterOptimizer.FLOW_THRESHOLD_RANGE[0] <= flow_thresh <= ParameterOptimizer.FLOW_THRESHOLD_RANGE[1]):
                errors.append(f"flow_threshold must be between {ParameterOptimizer.FLOW_THRESHOLD_RANGE[0]} and {ParameterOptimizer.FLOW_THRESHOLD_RANGE[1]}")
        
        # Check cellprob threshold
        cellprob_thresh = params.get('cellprob_threshold')
        if cellprob_thresh is not None:
            if not isinstance(cellprob_thresh, (int, float)):
                errors.append("cellprob_threshold must be a number")
            elif not (ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[0] <= cellprob_thresh <= ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[1]):
                errors.append(f"cellprob_threshold must be between {ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[0]} and {ParameterOptimizer.CELLPROB_THRESHOLD_RANGE[1]}")
        
        # Check model
        model = params.get('cellpose_model')
        if model is not None:
            if not isinstance(model, str):
                errors.append("cellpose_model must be a string")
            elif model not in ParameterOptimizer.VALID_MODELS:
                errors.append(f"cellpose_model must be one of: {ParameterOptimizer.VALID_MODELS}")
        
        return errors