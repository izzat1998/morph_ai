"""
Image Preprocessing Module

This module provides comprehensive image preprocessing operations for morphometric analysis,
including noise reduction, contrast enhancement, normalization, sharpening, and morphological operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from skimage import filters, morphology, exposure, restoration
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from .exceptions import ImagePreprocessingError, DependencyError
from .utils import validate_image_array

logger = logging.getLogger(__name__)


class GPUImagePreprocessor:
    """
    GPU-accelerated image preprocessing operations using CuPy.
    
    This class provides GPU-accelerated versions of common preprocessing operations
    with automatic fallback to CPU when GPU is not available.
    """
    
    def __init__(self):
        """Initialize GPU image preprocessor."""
        self.cupy_available = False
        self.gpu_info = None
        
        try:
            import cupy as cp
            self.cp = cp
            self.cupy_available = True
            logger.info("GPU preprocessing enabled with CuPy")
        except ImportError:
            self.cupy_available = False
            self.cp = None
            logger.info("CuPy not available, using CPU preprocessing")
    
    def _to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Move array to GPU if available."""
        if self.cupy_available:
            try:
                return self.cp.asarray(array)
            except Exception as e:
                logger.warning(f"Failed to move array to GPU: {str(e)}")
                return array
        return array
    
    def _to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Move array back to CPU."""
        if self.cupy_available and hasattr(array, 'get'):
            return array.get()
        return array
    
    def gaussian_filter_gpu(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """GPU-accelerated Gaussian filtering."""
        if not self.cupy_available:
            # Fallback to CPU
            return filters.gaussian(image, sigma=sigma)
        
        try:
            gpu_image = self._to_gpu(image)
            # Use CuPy's equivalent of scipy.ndimage.gaussian_filter
            from cupyx.scipy import ndimage as cp_ndimage
            filtered = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)
            return self._to_cpu(filtered)
        except Exception as e:
            logger.warning(f"GPU Gaussian filter failed, using CPU: {str(e)}")
            return filters.gaussian(image, sigma=sigma)
    
    def median_filter_gpu(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """GPU-accelerated median filtering."""
        if not self.cupy_available:
            return filters.median(image, morphology.disk(size))
        
        try:
            gpu_image = self._to_gpu(image)
            from cupyx.scipy import ndimage as cp_ndimage
            filtered = cp_ndimage.median_filter(gpu_image, size=size)
            return self._to_cpu(filtered)
        except Exception as e:
            logger.warning(f"GPU median filter failed, using CPU: {str(e)}")
            return filters.median(image, morphology.disk(size))
    
    def histogram_equalization_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated histogram equalization."""
        if not self.cupy_available:
            return exposure.equalize_hist(image)
        
        try:
            gpu_image = self._to_gpu(image)
            
            # Compute histogram on GPU
            hist, bin_edges = self.cp.histogram(gpu_image.ravel(), bins=256, range=(0, 1))
            
            # Compute cumulative distribution function
            cdf = self.cp.cumsum(hist).astype(self.cp.float32)
            cdf = cdf / cdf[-1]  # Normalize
            
            # Linear interpolation to map pixel values
            equalized = self.cp.interp(gpu_image.ravel(), bin_edges[:-1], cdf)
            equalized = equalized.reshape(gpu_image.shape)
            
            return self._to_cpu(equalized)
        except Exception as e:
            logger.warning(f"GPU histogram equalization failed, using CPU: {str(e)}")
            return exposure.equalize_hist(image)
    
    def contrast_enhancement_gpu(self, image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
        """GPU-accelerated CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if not self.cupy_available:
            return exposure.equalize_adapthist(image, clip_limit=clip_limit)
        
        try:
            # For now, fall back to CPU for CLAHE as it's complex to implement on GPU
            # In a production system, you might use specialized GPU libraries
            return exposure.equalize_adapthist(image, clip_limit=clip_limit)
        except Exception as e:
            logger.warning(f"GPU CLAHE failed, using CPU: {str(e)}")
            return exposure.equalize_adapthist(image, clip_limit=clip_limit)
    
    def morphological_operations_gpu(self, image: np.ndarray, operation: str, 
                                   kernel_size: int = 3) -> np.ndarray:
        """GPU-accelerated morphological operations."""
        if not self.cupy_available:
            kernel = morphology.disk(kernel_size)
            if operation == 'opening':
                return morphology.opening(image, kernel)
            elif operation == 'closing':
                return morphology.closing(image, kernel)
            elif operation == 'erosion':
                return morphology.erosion(image, kernel)
            elif operation == 'dilation':
                return morphology.dilation(image, kernel)
        
        try:
            gpu_image = self._to_gpu(image)
            from cupyx.scipy import ndimage as cp_ndimage
            
            # Create circular kernel
            kernel = self.cp.ones((kernel_size, kernel_size))
            
            if operation == 'erosion':
                result = cp_ndimage.binary_erosion(gpu_image, kernel)
            elif operation == 'dilation':
                result = cp_ndimage.binary_dilation(gpu_image, kernel)
            elif operation == 'opening':
                eroded = cp_ndimage.binary_erosion(gpu_image, kernel)
                result = cp_ndimage.binary_dilation(eroded, kernel)
            elif operation == 'closing':
                dilated = cp_ndimage.binary_dilation(gpu_image, kernel)
                result = cp_ndimage.binary_erosion(dilated, kernel)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return self._to_cpu(result)
        except Exception as e:
            logger.warning(f"GPU morphological operation failed, using CPU: {str(e)}")
            kernel = morphology.disk(kernel_size)
            if operation == 'opening':
                return morphology.opening(image, kernel)
            elif operation == 'closing':
                return morphology.closing(image, kernel)
            elif operation == 'erosion':
                return morphology.erosion(image, kernel)
            elif operation == 'dilation':
                return morphology.dilation(image, kernel)


class ImagePreprocessor:
    """
    Advanced image preprocessing operations for morphometric analysis.
    
    This class provides a pipeline of preprocessing operations that can be applied
    to images before segmentation to improve analysis quality.
    """
    
    NOISE_REDUCTION_METHODS = ['gaussian', 'median', 'bilateral']
    CONTRAST_METHODS = ['clahe', 'histogram_eq', 'rescale']
    NORMALIZATION_METHODS = ['zscore', 'minmax']
    SHARPENING_METHODS = ['unsharp_mask']
    MORPHOLOGICAL_OPERATIONS = ['opening', 'closing', 'erosion', 'dilation']
    
    def __init__(self, preprocessing_options: Optional[Dict[str, Any]] = None):
        """
        Initialize the image preprocessor.
        
        Args:
            preprocessing_options: Dictionary of preprocessing configuration options
        """
        if not SKIMAGE_AVAILABLE:
            raise DependencyError("scikit-image is required for image preprocessing")
            
        self.options = preprocessing_options or {}
        self._validate_options()
        
        logger.info(f"ImagePreprocessor initialized with options: {self.options}")
    
    def _validate_options(self) -> None:
        """Validate preprocessing options."""
        # Validate noise reduction method
        noise_method = self.options.get('noise_reduction_method', 'gaussian')
        if noise_method not in self.NOISE_REDUCTION_METHODS:
            raise ImagePreprocessingError(
                f"Invalid noise reduction method: {noise_method}. "
                f"Valid options: {self.NOISE_REDUCTION_METHODS}"
            )
        
        # Validate contrast method
        contrast_method = self.options.get('contrast_method', 'clahe')
        if contrast_method not in self.CONTRAST_METHODS:
            raise ImagePreprocessingError(
                f"Invalid contrast method: {contrast_method}. "
                f"Valid options: {self.CONTRAST_METHODS}"
            )
        
        # Validate normalization method
        norm_method = self.options.get('normalization_method', 'zscore')
        if norm_method not in self.NORMALIZATION_METHODS:
            raise ImagePreprocessingError(
                f"Invalid normalization method: {norm_method}. "
                f"Valid options: {self.NORMALIZATION_METHODS}"
            )
        
        # Validate morphological operation
        morph_op = self.options.get('morphological_operation', 'opening')
        if morph_op not in self.MORPHOLOGICAL_OPERATIONS:
            raise ImagePreprocessingError(
                f"Invalid morphological operation: {morph_op}. "
                f"Valid options: {self.MORPHOLOGICAL_OPERATIONS}"
            )
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Apply complete preprocessing pipeline based on configured options.
        
        Args:
            image: Input image array (2D grayscale or 3D RGB)
            
        Returns:
            Tuple of (processed_image, preprocessing_steps_applied)
            
        Raises:
            ImagePreprocessingError: If preprocessing fails
        """
        try:
            validate_image_array(image)
            
            processed_image = image.copy()
            preprocessing_steps = []
            
            logger.info(f"Starting image preprocessing. Image shape: {image.shape}, dtype: {image.dtype}")
            
            # Convert to float for processing
            was_uint8 = False
            if processed_image.dtype == np.uint8:
                processed_image = processed_image.astype(np.float32) / 255.0
                was_uint8 = True
            
            # Apply preprocessing steps in order
            if self.options.get('apply_noise_reduction', False):
                processed_image, step_info = self._apply_noise_reduction(processed_image)
                preprocessing_steps.append(step_info)
                logger.debug(f"Applied noise reduction: {step_info}")
            
            if self.options.get('apply_contrast_enhancement', False):
                processed_image, step_info = self._apply_contrast_enhancement(processed_image)
                preprocessing_steps.append(step_info)
                logger.debug(f"Applied contrast enhancement: {step_info}")
            
            if self.options.get('apply_normalization', False):
                processed_image, step_info = self._apply_normalization(processed_image)
                preprocessing_steps.append(step_info)
                logger.debug(f"Applied normalization: {step_info}")
            
            if self.options.get('apply_sharpening', False):
                processed_image, step_info = self._apply_sharpening(processed_image)
                preprocessing_steps.append(step_info)
                logger.debug(f"Applied sharpening: {step_info}")
            
            if self.options.get('apply_morphological', False):
                processed_image, step_info = self._apply_morphological_operations(processed_image)
                preprocessing_steps.append(step_info)
                logger.debug(f"Applied morphological operations: {step_info}")
            
            # Convert back to original data type
            if was_uint8:
                processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            
            logger.info(f"Preprocessing completed. Applied {len(preprocessing_steps)} operations.")
            return processed_image, preprocessing_steps
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")
    
    def _apply_noise_reduction(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply noise reduction filtering.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (filtered_image, step_description)
        """
        method = self.options.get('noise_reduction_method', 'gaussian')
        
        try:
            if method == 'gaussian':
                sigma = self.options.get('gaussian_sigma', 0.5)
                if sigma <= 0:
                    raise ImagePreprocessingError("Gaussian sigma must be positive")
                    
                filtered_image = filters.gaussian(image, sigma=sigma, preserve_range=True)
                step_info = f'Gaussian blur (σ={sigma})'
            
            elif method == 'median':
                disk_size = self.options.get('median_disk_size', 2)
                if disk_size <= 0:
                    raise ImagePreprocessingError("Median disk size must be positive")
                
                if len(image.shape) == 3:
                    filtered_image = np.stack([
                        filters.median(image[:,:,i], morphology.disk(disk_size))
                        for i in range(image.shape[2])
                    ], axis=2)
                else:
                    filtered_image = filters.median(image, morphology.disk(disk_size))
                step_info = f'Median filter (disk size={disk_size})'
            
            elif method == 'bilateral':
                # Bilateral filtering (preserves edges while reducing noise)
                sigma_color = self.options.get('bilateral_sigma_color', 0.1)
                sigma_spatial = self.options.get('bilateral_sigma_spatial', 1.0)
                
                if len(image.shape) == 3:
                    filtered_image = np.stack([
                        restoration.denoise_bilateral(
                            image[:,:,i], 
                            sigma_color=sigma_color, 
                            sigma_spatial=sigma_spatial
                        )
                        for i in range(image.shape[2])
                    ], axis=2)
                else:
                    filtered_image = restoration.denoise_bilateral(
                        image, 
                        sigma_color=sigma_color, 
                        sigma_spatial=sigma_spatial
                    )
                step_info = f'Bilateral filtering (σ_color={sigma_color}, σ_spatial={sigma_spatial})'
            
            else:
                filtered_image = image.copy()
                step_info = 'No noise reduction applied'
            
            return filtered_image, step_info
            
        except Exception as e:
            raise ImagePreprocessingError(f"Noise reduction failed: {str(e)}")
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply contrast enhancement.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (enhanced_image, step_description)
        """
        method = self.options.get('contrast_method', 'clahe')
        
        try:
            if method == 'clahe':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clip_limit = self.options.get('clahe_clip_limit', 0.03)
                
                if len(image.shape) == 3:
                    enhanced_image = np.stack([
                        exposure.equalize_adapthist(image[:,:,i], clip_limit=clip_limit)
                        for i in range(image.shape[2])
                    ], axis=2)
                else:
                    enhanced_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)
                step_info = f'CLAHE contrast enhancement (clip_limit={clip_limit})'
            
            elif method == 'histogram_eq':
                # Global histogram equalization
                if len(image.shape) == 3:
                    enhanced_image = np.stack([
                        exposure.equalize_hist(image[:,:,i])
                        for i in range(image.shape[2])
                    ], axis=2)
                else:
                    enhanced_image = exposure.equalize_hist(image)
                step_info = 'Histogram equalization'
            
            elif method == 'rescale':
                # Simple rescaling to full range
                p_low = self.options.get('rescale_p_low', 2)
                p_high = self.options.get('rescale_p_high', 98)
                
                p_low_val, p_high_val = np.percentile(image, (p_low, p_high))
                enhanced_image = exposure.rescale_intensity(image, in_range=(p_low_val, p_high_val))
                step_info = f'Intensity rescaling ({p_low}-{p_high} percentile)'
            
            else:
                enhanced_image = image.copy()
                step_info = 'No contrast enhancement applied'
            
            return enhanced_image, step_info
            
        except Exception as e:
            raise ImagePreprocessingError(f"Contrast enhancement failed: {str(e)}")
    
    def _apply_normalization(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply intensity normalization.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (normalized_image, step_description)
        """
        method = self.options.get('normalization_method', 'zscore')
        
        try:
            if method == 'zscore':
                # Z-score normalization
                mean_val = np.mean(image)
                std_val = np.std(image)
                if std_val > 1e-10:
                    normalized_image = (image - mean_val) / std_val
                    # Rescale to 0-1 range
                    min_val = normalized_image.min()
                    max_val = normalized_image.max()
                    if max_val > min_val:
                        normalized_image = (normalized_image - min_val) / (max_val - min_val)
                else:
                    normalized_image = image.copy()
                    logger.warning("Image has zero standard deviation, skipping z-score normalization")
                step_info = 'Z-score normalization'
            
            elif method == 'minmax':
                # Min-max normalization
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    normalized_image = (image - min_val) / (max_val - min_val)
                else:
                    normalized_image = image.copy()
                    logger.warning("Image has constant intensity, skipping min-max normalization")
                step_info = 'Min-max normalization'
            
            else:
                normalized_image = image.copy()
                step_info = 'No normalization applied'
            
            return normalized_image, step_info
            
        except Exception as e:
            raise ImagePreprocessingError(f"Normalization failed: {str(e)}")
    
    def _apply_sharpening(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply image sharpening.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (sharpened_image, step_description)
        """
        method = self.options.get('sharpening_method', 'unsharp_mask')
        
        try:
            if method == 'unsharp_mask':
                # Unsharp masking
                radius = self.options.get('unsharp_radius', 1.0)
                amount = self.options.get('unsharp_amount', 1.0)
                
                if radius <= 0 or amount <= 0:
                    raise ImagePreprocessingError("Unsharp mask radius and amount must be positive")
                
                if len(image.shape) == 3:
                    sharpened_image = np.stack([
                        filters.unsharp_mask(image[:,:,i], radius=radius, amount=amount)
                        for i in range(image.shape[2])
                    ], axis=2)
                else:
                    sharpened_image = filters.unsharp_mask(image, radius=radius, amount=amount)
                step_info = f'Unsharp masking (radius={radius}, amount={amount})'
            
            else:
                sharpened_image = image.copy()
                step_info = 'No sharpening applied'
            
            return sharpened_image, step_info
            
        except Exception as e:
            raise ImagePreprocessingError(f"Sharpening failed: {str(e)}")
    
    def _apply_morphological_operations(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply morphological operations for artifact removal.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (processed_image, step_description)
        """
        operation = self.options.get('morphological_operation', 'opening')
        disk_size = self.options.get('morphological_disk_size', 1)
        
        try:
            if disk_size <= 0:
                raise ImagePreprocessingError("Morphological disk size must be positive")
            
            # Convert to grayscale for morphological operations
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image.copy()
            
            selem = morphology.disk(disk_size)
            
            if operation == 'opening':
                processed_gray = morphology.opening(gray, selem)
                step_info = f'Morphological opening (disk size={disk_size})'
            elif operation == 'closing':
                processed_gray = morphology.closing(gray, selem)
                step_info = f'Morphological closing (disk size={disk_size})'
            elif operation == 'erosion':
                processed_gray = morphology.erosion(gray, selem)
                step_info = f'Morphological erosion (disk size={disk_size})'
            elif operation == 'dilation':
                processed_gray = morphology.dilation(gray, selem)
                step_info = f'Morphological dilation (disk size={disk_size})'
            else:
                processed_gray = gray.copy()
                step_info = 'No morphological operations applied'
            
            # If original was color, maintain the color information
            if len(image.shape) == 3:
                # Apply the same transformation to all channels proportionally
                ratio = processed_gray / (gray + 1e-10)
                # Clip ratio to reasonable bounds
                ratio = np.clip(ratio, 0.1, 10.0)
                processed_image = image * ratio[:, :, np.newaxis]
            else:
                processed_image = processed_gray
            
            return processed_image, step_info
            
        except Exception as e:
            raise ImagePreprocessingError(f"Morphological operations failed: {str(e)}")
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """
        Get available preprocessing methods.
        
        Returns:
            Dictionary of available methods for each preprocessing type
        """
        return {
            'noise_reduction': self.NOISE_REDUCTION_METHODS,
            'contrast_enhancement': self.CONTRAST_METHODS,
            'normalization': self.NORMALIZATION_METHODS,
            'sharpening': self.SHARPENING_METHODS,
            'morphological': self.MORPHOLOGICAL_OPERATIONS
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """
        Get default preprocessing options.
        
        Returns:
            Dictionary with default preprocessing options
        """
        return {
            'apply_noise_reduction': False,
            'noise_reduction_method': 'gaussian',
            'gaussian_sigma': 0.5,
            'median_disk_size': 2,
            'bilateral_sigma_color': 0.1,
            'bilateral_sigma_spatial': 1.0,
            
            'apply_contrast_enhancement': False,
            'contrast_method': 'clahe',
            'clahe_clip_limit': 0.03,
            'rescale_p_low': 2,
            'rescale_p_high': 98,
            
            'apply_normalization': False,
            'normalization_method': 'zscore',
            
            'apply_sharpening': False,
            'sharpening_method': 'unsharp_mask',
            'unsharp_radius': 1.0,
            'unsharp_amount': 1.0,
            
            'apply_morphological': False,
            'morphological_operation': 'opening',
            'morphological_disk_size': 1
        }