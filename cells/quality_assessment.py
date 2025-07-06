"""
Image Quality Assessment Module

This module provides comprehensive image quality metrics for morphometric analysis,
including blur detection, contrast measurement, and noise estimation.
"""

import numpy as np
from typing import Dict, List, Union, Optional
import logging

try:
    from skimage import filters
    from scipy.stats import entropy as scipy_entropy
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageQualityError(Exception):
    """Custom exception for image quality assessment errors"""
    pass


class ImageQualityAssessment:
    """
    Class for assessing image quality metrics for morphometric analysis.
    
    This class provides static methods to calculate various image quality metrics
    including blur, contrast, and noise measurements that are essential for
    determining preprocessing requirements.
    """
    
    @staticmethod
    def calculate_blur_metrics(image: np.ndarray) -> Dict[str, float]:
        """
        Calculate blur metrics using Laplacian variance and Tenengrad gradient.
        
        Args:
            image: Input image array (2D grayscale or 3D RGB)
            
        Returns:
            Dictionary containing blur metrics:
            - laplacian_variance: Laplacian variance (higher = less blurry)
            - tenengrad_gradient: Tenengrad gradient (higher = sharper)
            - blur_score: Primary blur metric
            - sharpness_score: Primary sharpness metric
            
        Raises:
            ImageQualityError: If image processing fails
        """
        if not SKIMAGE_AVAILABLE:
            raise ImageQualityError("scikit-image is required for blur metrics calculation")
            
        if image is None or image.size == 0:
            raise ImageQualityError("Invalid input image")
            
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image.copy()
            
            # Laplacian variance (higher = less blurry)
            laplacian_var = filters.laplace(gray).var()
            
            # Tenengrad gradient (higher = sharper)
            sobel_x = filters.sobel_h(gray)
            sobel_y = filters.sobel_v(gray)
            tenengrad = np.sqrt(sobel_x**2 + sobel_y**2).mean()
            
            return {
                'laplacian_variance': float(laplacian_var),
                'tenengrad_gradient': float(tenengrad),
                'blur_score': float(laplacian_var),  # Higher = less blurry
                'sharpness_score': float(tenengrad)  # Higher = sharper
            }
            
        except Exception as e:
            logger.error(f"Error calculating blur metrics: {str(e)}")
            raise ImageQualityError(f"Failed to calculate blur metrics: {str(e)}")
    
    @staticmethod
    def calculate_contrast_metrics(image: np.ndarray) -> Dict[str, float]:
        """
        Calculate contrast metrics including RMS, Michelson, and histogram entropy.
        
        Args:
            image: Input image array (2D grayscale or 3D RGB)
            
        Returns:
            Dictionary containing contrast metrics:
            - rms_contrast: RMS contrast
            - michelson_contrast: Michelson contrast ratio
            - std_contrast: Standard deviation contrast
            - histogram_entropy: Histogram entropy
            - contrast_score: Primary contrast metric
            
        Raises:
            ImageQualityError: If image processing fails
        """
        if image is None or image.size == 0:
            raise ImageQualityError("Invalid input image")
            
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image.copy()
            
            # Normalize to 0-255 for consistent metrics
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            
            # RMS contrast
            rms_contrast = np.sqrt(np.mean((gray - gray.mean()) ** 2))
            
            # Michelson contrast
            max_val = float(gray.max())
            min_val = float(gray.min())
            michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
            
            # Standard deviation (another contrast measure)
            std_contrast = gray.std()
            
            # Histogram entropy
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist_entropy = scipy_entropy(hist + 1e-10)  # Add small value to avoid log(0)
            
            return {
                'rms_contrast': float(rms_contrast),
                'michelson_contrast': float(michelson_contrast),
                'std_contrast': float(std_contrast),
                'histogram_entropy': float(hist_entropy),
                'contrast_score': float(rms_contrast)  # Primary contrast metric
            }
            
        except Exception as e:
            logger.error(f"Error calculating contrast metrics: {str(e)}")
            raise ImageQualityError(f"Failed to calculate contrast metrics: {str(e)}")
    
    @staticmethod
    def calculate_noise_metrics(image: np.ndarray) -> Dict[str, float]:
        """
        Calculate noise estimation metrics using signal-to-noise ratio.
        
        Args:
            image: Input image array (2D grayscale or 3D RGB)
            
        Returns:
            Dictionary containing noise metrics:
            - noise_power: Estimated noise power
            - snr_linear: Signal-to-noise ratio (linear scale)
            - snr_db: Signal-to-noise ratio (decibel scale)
            - noise_score: Primary noise metric (higher = less noisy)
            
        Raises:
            ImageQualityError: If image processing fails
        """
        if not SKIMAGE_AVAILABLE:
            raise ImageQualityError("scikit-image is required for noise metrics calculation")
            
        if image is None or image.size == 0:
            raise ImageQualityError("Invalid input image")
            
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image.copy()
            
            # Normalize to 0-1 for consistent processing
            if gray.max() > 1.0:
                gray = gray / 255.0
            
            # Wiener filter estimation of noise
            # Apply small Gaussian to estimate noise
            denoised = filters.gaussian(gray, sigma=0.5)
            noise_estimate = np.abs(gray - denoised)
            noise_power = np.mean(noise_estimate**2)
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(denoised**2)
            snr = signal_power / (noise_power + 1e-10)
            snr_db = 10 * np.log10(snr + 1e-10)
            
            return {
                'noise_power': float(noise_power),
                'snr_linear': float(snr),
                'snr_db': float(snr_db),
                'noise_score': float(snr_db)  # Higher = less noisy
            }
            
        except Exception as e:
            logger.warning(f"Error calculating noise metrics, using defaults: {str(e)}")
            return {
                'noise_power': 0.0,
                'snr_linear': 1.0,
                'snr_db': 0.0,
                'noise_score': 0.0
            }
    
    @staticmethod
    def assess_overall_quality(image: np.ndarray) -> Dict[str, Union[float, str, Dict, List]]:
        """
        Comprehensive image quality assessment combining all metrics.
        
        Args:
            image: Input image array (2D grayscale or 3D RGB)
            
        Returns:
            Dictionary containing:
            - blur_metrics: Detailed blur measurements
            - contrast_metrics: Detailed contrast measurements  
            - noise_metrics: Detailed noise measurements
            - blur_score: Normalized blur score (0-100)
            - contrast_score: Normalized contrast score (0-100)
            - noise_score: Normalized noise score (0-100)
            - overall_score: Weighted overall quality score (0-100)
            - quality_category: Quality category (excellent/good/fair/poor)
            - recommendations: List of preprocessing recommendations
            
        Raises:
            ImageQualityError: If quality assessment fails
        """
        if image is None or image.size == 0:
            raise ImageQualityError("Invalid input image")
            
        try:
            blur_metrics = ImageQualityAssessment.calculate_blur_metrics(image)
            contrast_metrics = ImageQualityAssessment.calculate_contrast_metrics(image)
            noise_metrics = ImageQualityAssessment.calculate_noise_metrics(image)
            
            # Combine metrics into overall scores
            # Normalize scores to 0-100 scale with improved thresholds
            blur_score = min(100, max(0, (blur_metrics['blur_score'] / 500) * 100))  # Adjusted threshold
            contrast_score = min(100, max(0, (contrast_metrics['contrast_score'] / 40) * 100))  # Adjusted threshold  
            noise_score = min(100, max(0, (noise_metrics['snr_db'] + 10) * 3))  # Adjusted threshold
            
            # Overall quality score (weighted average)
            overall_score = (blur_score * 0.4 + contrast_score * 0.3 + noise_score * 0.3)
            
            # Quality categories
            if overall_score >= 80:
                quality_category = 'excellent'
            elif overall_score >= 60:
                quality_category = 'good'
            elif overall_score >= 40:
                quality_category = 'fair'
            else:
                quality_category = 'poor'
            
            return {
                'blur_metrics': blur_metrics,
                'contrast_metrics': contrast_metrics,
                'noise_metrics': noise_metrics,
                'blur_score': blur_score,
                'contrast_score': contrast_score,
                'noise_score': noise_score,
                'overall_score': overall_score,
                'quality_category': quality_category,
                'recommendations': ImageQualityAssessment._generate_recommendations(
                    blur_score, contrast_score, noise_score
                )
            }
            
        except Exception as e:
            logger.error(f"Error in overall quality assessment: {str(e)}")
            raise ImageQualityError(f"Failed to assess overall quality: {str(e)}")
    
    @staticmethod
    def _generate_recommendations(blur_score: float, contrast_score: float, noise_score: float) -> List[str]:
        """
        Generate preprocessing recommendations based on quality scores.
        
        Args:
            blur_score: Blur quality score (0-100)
            contrast_score: Contrast quality score (0-100)
            noise_score: Noise quality score (0-100)
            
        Returns:
            List of string recommendations for image preprocessing
        """
        recommendations = []
        
        if blur_score < 40:
            recommendations.append('Consider image sharpening')
        if contrast_score < 40:
            recommendations.append('Apply contrast enhancement (CLAHE)')
        if noise_score < 40:
            recommendations.append('Apply noise reduction filtering')
        
        if not recommendations:
            recommendations.append('Image quality is good - minimal preprocessing needed')
        
        return recommendations