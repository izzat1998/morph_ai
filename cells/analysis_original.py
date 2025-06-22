import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
from django.core.files.base import ContentFile
from django.utils import timezone
from django.conf import settings
from django.utils.translation import gettext as _

try:
    from cellpose import models
    from cellpose.io import imread
    from cellpose import plot, utils
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    from skimage import measure, morphology, exposure, filters, restoration, feature, segmentation
    from skimage.segmentation import clear_border, watershed
    from skimage.morphology import disk, opening, closing, erosion, dilation
    from skimage.filters import rank, gaussian, median, unsharp_mask
    from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
    from skimage.feature import graycomatrix, graycoprops, blob_log, blob_dog, canny, peak_local_max
    from scipy import ndimage
    from scipy.stats import entropy
    from scipy.ndimage import binary_fill_holes, label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from .models import CellAnalysis, DetectedCell


class ImageQualityAssessment:
    """Class for assessing image quality metrics"""
    
    @staticmethod
    def calculate_blur_metrics(image):
        """Calculate blur metrics using Laplacian variance"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
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
    
    @staticmethod
    def calculate_contrast_metrics(image):
        """Calculate contrast metrics"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
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
        from scipy.stats import entropy as scipy_entropy
        hist_entropy = scipy_entropy(hist + 1e-10)  # Add small value to avoid log(0)
        
        return {
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'std_contrast': float(std_contrast),
            'histogram_entropy': float(hist_entropy),
            'contrast_score': float(rms_contrast)  # Primary contrast metric
        }
    
    @staticmethod
    def calculate_noise_metrics(image):
        """Calculate noise estimation metrics"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize to 0-1 for consistent processing
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        try:
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
        except Exception:
            return {
                'noise_power': 0.0,
                'snr_linear': 1.0,
                'snr_db': 0.0,
                'noise_score': 0.0
            }
    
    @staticmethod
    def assess_overall_quality(image):
        """Comprehensive image quality assessment"""
        blur_metrics = ImageQualityAssessment.calculate_blur_metrics(image)
        contrast_metrics = ImageQualityAssessment.calculate_contrast_metrics(image)
        noise_metrics = ImageQualityAssessment.calculate_noise_metrics(image)
        
        # Combine metrics into overall scores
        # Normalize scores to 0-100 scale
        blur_score = min(100, max(0, (blur_metrics['blur_score'] / 1000) * 100))  # Scale laplacian variance
        contrast_score = min(100, max(0, (contrast_metrics['contrast_score'] / 64) * 100))  # Scale RMS contrast
        noise_score = min(100, max(0, (noise_metrics['snr_db'] + 20) * 2))  # Scale SNR dB
        
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
    
    @staticmethod
    def _generate_recommendations(blur_score, contrast_score, noise_score):
        """Generate preprocessing recommendations based on quality scores"""
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


class ImagePreprocessor:
    """Class for advanced image preprocessing operations"""
    
    def __init__(self, preprocessing_options=None):
        self.options = preprocessing_options or {}
    
    def preprocess_image(self, image):
        """Apply preprocessing pipeline based on options"""
        processed_image = image.copy()
        preprocessing_steps = []
        
        # Convert to float for processing
        if processed_image.dtype == np.uint8:
            processed_image = processed_image.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            was_uint8 = False
        
        # Apply preprocessing steps in order
        if self.options.get('apply_noise_reduction', False):
            processed_image, step_info = self._apply_noise_reduction(processed_image)
            preprocessing_steps.append(step_info)
        
        if self.options.get('apply_contrast_enhancement', False):
            processed_image, step_info = self._apply_contrast_enhancement(processed_image)
            preprocessing_steps.append(step_info)
        
        if self.options.get('apply_normalization', False):
            processed_image, step_info = self._apply_normalization(processed_image)
            preprocessing_steps.append(step_info)
        
        if self.options.get('apply_sharpening', False):
            processed_image, step_info = self._apply_sharpening(processed_image)
            preprocessing_steps.append(step_info)
        
        if self.options.get('apply_morphological', False):
            processed_image, step_info = self._apply_morphological_operations(processed_image)
            preprocessing_steps.append(step_info)
        
        # Convert back to original data type
        if was_uint8:
            processed_image = (processed_image * 255).astype(np.uint8)
        
        return processed_image, preprocessing_steps
    
    def _apply_noise_reduction(self, image):
        """Apply noise reduction filtering"""
        method = self.options.get('noise_reduction_method', 'gaussian')
        
        if method == 'gaussian':
            sigma = self.options.get('gaussian_sigma', 0.5)
            filtered_image = filters.gaussian(image, sigma=sigma, preserve_range=True)
            step_info = f'Gaussian blur (σ={sigma})'
        
        elif method == 'median':
            disk_size = self.options.get('median_disk_size', 2)
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
            if len(image.shape) == 3:
                filtered_image = np.stack([
                    restoration.denoise_bilateral(image[:,:,i], 
                                                 sigma_color=0.1, 
                                                 sigma_spatial=1.0)
                    for i in range(image.shape[2])
                ], axis=2)
            else:
                filtered_image = restoration.denoise_bilateral(image, 
                                                             sigma_color=0.1, 
                                                             sigma_spatial=1.0)
            step_info = 'Bilateral filtering'
        
        else:
            filtered_image = image
            step_info = 'No noise reduction applied'
        
        return filtered_image, step_info
    
    def _apply_contrast_enhancement(self, image):
        """Apply contrast enhancement"""
        method = self.options.get('contrast_method', 'clahe')
        
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(image.shape) == 3:
                enhanced_image = np.stack([
                    exposure.equalize_adapthist(image[:,:,i], clip_limit=0.03)
                    for i in range(image.shape[2])
                ], axis=2)
            else:
                enhanced_image = exposure.equalize_adapthist(image, clip_limit=0.03)
            step_info = 'CLAHE contrast enhancement'
        
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
            p2, p98 = np.percentile(image, (2, 98))
            enhanced_image = exposure.rescale_intensity(image, in_range=(p2, p98))
            step_info = 'Intensity rescaling (2-98 percentile)'
        
        else:
            enhanced_image = image
            step_info = 'No contrast enhancement applied'
        
        return enhanced_image, step_info
    
    def _apply_normalization(self, image):
        """Apply intensity normalization"""
        method = self.options.get('normalization_method', 'zscore')
        
        if method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(image)
            std_val = np.std(image)
            if std_val > 0:
                normalized_image = (image - mean_val) / std_val
                # Rescale to 0-1 range
                normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())
            else:
                normalized_image = image
            step_info = 'Z-score normalization'
        
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized_image = (image - min_val) / (max_val - min_val)
            else:
                normalized_image = image
            step_info = 'Min-max normalization'
        
        else:
            normalized_image = image
            step_info = 'No normalization applied'
        
        return normalized_image, step_info
    
    def _apply_sharpening(self, image):
        """Apply image sharpening"""
        method = self.options.get('sharpening_method', 'unsharp_mask')
        
        if method == 'unsharp_mask':
            # Unsharp masking
            radius = self.options.get('unsharp_radius', 1.0)
            amount = self.options.get('unsharp_amount', 1.0)
            
            if len(image.shape) == 3:
                sharpened_image = np.stack([
                    filters.unsharp_mask(image[:,:,i], radius=radius, amount=amount)
                    for i in range(image.shape[2])
                ], axis=2)
            else:
                sharpened_image = filters.unsharp_mask(image, radius=radius, amount=amount)
            step_info = f'Unsharp masking (radius={radius}, amount={amount})'
        
        else:
            sharpened_image = image
            step_info = 'No sharpening applied'
        
        return sharpened_image, step_info
    
    def _apply_morphological_operations(self, image):
        """Apply morphological operations for artifact removal"""
        operation = self.options.get('morphological_operation', 'opening')
        disk_size = self.options.get('morphological_disk_size', 1)
        
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
            processed_gray = gray
            step_info = 'No morphological operations applied'
        
        # If original was color, maintain the color information
        if len(image.shape) == 3:
            # Apply the same transformation to all channels proportionally
            ratio = processed_gray / (gray + 1e-10)
            processed_image = image * ratio[:, :, np.newaxis]
        else:
            processed_image = processed_gray
        
        return processed_image, step_info


class ParameterOptimizer:
    """
    Automatic parameter optimization for Cellpose based on image characteristics
    """
    
    @staticmethod
    def estimate_cell_diameter(image_array, sample_size=5):
        """
        Estimate optimal cell diameter using blob detection and statistical analysis
        """
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array.astype(np.uint8)
        
        # Apply gentle preprocessing for better blob detection
        smoothed = filters.gaussian(gray, sigma=1.0)
        
        # Use multiple blob detection methods for robustness
        blobs_log = feature.blob_log(smoothed, min_sigma=3, max_sigma=50, num_sigma=20, threshold=0.02)
        blobs_dog = feature.blob_dog(smoothed, min_sigma=3, max_sigma=50, sigma_ratio=1.6, threshold=0.02)
        
        # Combine results
        all_blobs = []
        if len(blobs_log) > 0:
            # Convert LoG blobs to diameter (radius * 2 * sqrt(2))
            diameters_log = blobs_log[:, 2] * 2 * np.sqrt(2)
            all_blobs.extend(diameters_log)
        
        if len(blobs_dog) > 0:
            # Convert DoG blobs to diameter
            diameters_dog = blobs_dog[:, 2] * 2 * np.sqrt(2)
            all_blobs.extend(diameters_dog)
        
        if len(all_blobs) == 0:
            # Fallback: estimate from image dimensions
            min_dim = min(gray.shape)
            estimated_diameter = min_dim / 10  # Assume cells are roughly 1/10 of image size
            return max(20, min(100, estimated_diameter))
        
        # Statistical analysis of detected blob sizes
        all_blobs = np.array(all_blobs)
        
        # Remove outliers (beyond 2 standard deviations)
        mean_diameter = np.mean(all_blobs)
        std_diameter = np.std(all_blobs)
        filtered_blobs = all_blobs[np.abs(all_blobs - mean_diameter) <= 2 * std_diameter]
        
        if len(filtered_blobs) == 0:
            return mean_diameter
        
        # Return median for robustness
        estimated_diameter = np.median(filtered_blobs)
        
        # Clamp to reasonable range
        return max(10, min(200, estimated_diameter))
    
    @staticmethod
    def optimize_thresholds(image_array, quality_metrics):
        """
        Optimize flow and cellprob thresholds based on image quality
        """
        overall_score = quality_metrics.get('overall_score', 50)
        blur_score = quality_metrics.get('blur_metrics', {}).get('blur_score', 50)
        contrast_score = quality_metrics.get('contrast_metrics', {}).get('contrast_score', 50)
        noise_score = quality_metrics.get('noise_metrics', {}).get('noise_score', 50)
        
        # Default values
        flow_threshold = 0.4
        cellprob_threshold = 0.0
        
        # Adjust based on image quality
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
        flow_threshold = max(0.1, min(3.0, flow_threshold))
        cellprob_threshold = max(-6.0, min(6.0, cellprob_threshold))
        
        return {
            'flow_threshold': float(flow_threshold),
            'cellprob_threshold': float(cellprob_threshold),
            'reasoning': f"Optimized for image quality score {overall_score:.1f}"
        }
    
    @staticmethod
    def select_optimal_model(image_array, quality_metrics):
        """
        Select the best Cellpose model based on image characteristics
        """
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array.astype(np.uint8)
        
        # Analyze image characteristics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Check if image looks like nuclei (high contrast, round objects)
        # Use edge detection to estimate roundness
        edges = feature.canny(gray, sigma=2.0)
        edge_density = np.sum(edges) / edges.size
        
        # Estimate texture using local binary patterns
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            texture_uniformity = np.std(lbp)
        except:
            texture_uniformity = std_intensity
        
        # Decision logic
        if mean_intensity > 150 and edge_density > 0.05 and texture_uniformity < 50:
            # High contrast, well-defined edges, uniform texture -> likely nuclei
            recommended_model = 'nuclei'
            confidence = 0.8
        elif mean_intensity < 100 and edge_density < 0.03:
            # Low contrast, fewer edges -> might be cytoplasm
            recommended_model = 'cyto2'  # More robust version
            confidence = 0.6
        else:
            # Default to cyto2 for general purpose
            recommended_model = 'cyto2'
            confidence = 0.5
        
        return {
            'recommended_model': recommended_model,
            'confidence': confidence,
            'reasoning': f"Based on intensity={mean_intensity:.1f}, edges={edge_density:.3f}, texture={texture_uniformity:.1f}"
        }
    
    @staticmethod
    def optimize_all_parameters(image_array, quality_metrics, current_params=None):
        """
        Comprehensive parameter optimization
        """
        current_params = current_params or {}
        
        # Estimate optimal diameter
        optimal_diameter = ParameterOptimizer.estimate_cell_diameter(image_array)
        
        # Optimize thresholds
        threshold_optimization = ParameterOptimizer.optimize_thresholds(image_array, quality_metrics)
        
        # Select optimal model
        model_optimization = ParameterOptimizer.select_optimal_model(image_array, quality_metrics)
        
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
            ]
        }
        
        return recommendations


class SegmentationRefinement:
    """
    Post-processing refinement for improving segmentation accuracy
    """
    
    @staticmethod
    def filter_by_size(masks, min_area=50, max_area=None):
        """
        Remove objects that are too small or too large to be cells
        """
        if max_area is None:
            # Set max area to 1/4 of image area as default
            max_area = (masks.shape[0] * masks.shape[1]) // 4
        
        refined_masks = np.zeros_like(masks)
        new_label = 1
        
        # Get properties of all regions
        props = measure.regionprops(masks)
        
        for prop in props:
            if min_area <= prop.area <= max_area:
                # Keep this region, assign new label
                refined_masks[masks == prop.label] = new_label
                new_label += 1
        
        return refined_masks
    
    @staticmethod
    def filter_by_shape(masks, min_circularity=0.1, max_eccentricity=0.95):
        """
        Remove objects with non-cellular shapes
        """
        refined_masks = np.zeros_like(masks)
        new_label = 1
        
        props = measure.regionprops(masks)
        
        for prop in props:
            # Calculate circularity
            area = prop.area
            perimeter = prop.perimeter
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Check shape criteria
            if (circularity >= min_circularity and 
                prop.eccentricity <= max_eccentricity and
                prop.solidity >= 0.7):  # Require reasonable solidity
                
                refined_masks[masks == prop.label] = new_label
                new_label += 1
        
        return refined_masks
    
    @staticmethod
    def split_touching_cells(masks, min_distance=10):
        """
        Use watershed segmentation to split touching cells
        """
        # Create binary mask
        binary = masks > 0
        
        # Calculate distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima (cell centers)
        local_maxima = feature.peak_local_max(
            distance, 
            min_distance=min_distance,
            threshold_abs=min_distance//2
        )
        
        # Create markers for watershed
        markers = np.zeros_like(masks, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        refined_masks = segmentation.watershed(-distance, markers, mask=binary)
        
        return refined_masks
    
    @staticmethod
    def smooth_boundaries(masks, smoothing_factor=1.0):
        """
        Smooth cell boundaries using morphological operations
        """
        refined_masks = np.zeros_like(masks)
        
        unique_labels = np.unique(masks)[1:]  # Skip background
        
        for label in unique_labels:
            # Extract single cell mask
            cell_mask = (masks == label).astype(np.uint8)
            
            # Apply morphological operations
            kernel = disk(int(smoothing_factor))
            
            # Closing to fill small gaps
            cell_mask = morphology.closing(cell_mask, kernel)
            
            # Opening to smooth boundaries
            cell_mask = morphology.opening(cell_mask, kernel)
            
            # Fill holes
            cell_mask = binary_fill_holes(cell_mask)
            
            refined_masks[cell_mask > 0] = label
        
        return refined_masks
    
    @staticmethod
    def remove_edge_cells(masks, border_width=5):
        """
        Remove cells touching the image edges
        """
        return clear_border(masks, buffer_size=border_width)
    
    @staticmethod
    def refine_segmentation(masks, image_array=None, options=None):
        """
        Apply comprehensive segmentation refinement
        """
        options = options or {}
        refined_masks = masks.copy()
        refinement_steps = []
        
        original_count = len(np.unique(refined_masks)) - 1  # Subtract background
        
        # Step 1: Size filtering
        if options.get('apply_size_filtering', True):
            min_area = options.get('min_cell_area', 50)
            max_area = options.get('max_cell_area', None)
            refined_masks = SegmentationRefinement.filter_by_size(
                refined_masks, min_area, max_area
            )
            after_size = len(np.unique(refined_masks)) - 1
            refinement_steps.append(f"Size filtering: {original_count} → {after_size} cells")
        
        # Step 2: Shape filtering
        if options.get('apply_shape_filtering', True):
            min_circularity = options.get('min_circularity', 0.1)
            max_eccentricity = options.get('max_eccentricity', 0.95)
            refined_masks = SegmentationRefinement.filter_by_shape(
                refined_masks, min_circularity, max_eccentricity
            )
            after_shape = len(np.unique(refined_masks)) - 1
            refinement_steps.append(f"Shape filtering: {after_size} → {after_shape} cells")
        
        # Step 3: Split touching cells
        if options.get('apply_watershed', False):
            min_distance = options.get('watershed_min_distance', 10)
            refined_masks = SegmentationRefinement.split_touching_cells(
                refined_masks, min_distance
            )
            after_watershed = len(np.unique(refined_masks)) - 1
            refinement_steps.append(f"Watershed splitting: {after_shape} → {after_watershed} cells")
        
        # Step 4: Boundary smoothing
        if options.get('apply_smoothing', True):
            smoothing_factor = options.get('smoothing_factor', 1.0)
            refined_masks = SegmentationRefinement.smooth_boundaries(
                refined_masks, smoothing_factor
            )
            refinement_steps.append("Applied boundary smoothing")
        
        # Step 5: Remove edge cells
        if options.get('remove_edge_cells', False):
            border_width = options.get('border_width', 5)
            refined_masks = SegmentationRefinement.remove_edge_cells(
                refined_masks, border_width
            )
            final_count = len(np.unique(refined_masks)) - 1
            refinement_steps.append(f"Edge removal: {final_count} final cells")
        
        final_count = len(np.unique(refined_masks)) - 1
        refinement_steps.append(f"Final result: {final_count} cells")
        
        return refined_masks, refinement_steps


class IntelligentROI:
    """
    Intelligent ROI suggestion based on cell density and image characteristics
    """
    
    @staticmethod
    def detect_cell_density_regions(image_array, grid_size=8):
        """
        Detect regions with high cell density for ROI suggestions
        """
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array.astype(np.uint8)
        
        h, w = gray.shape
        cell_density_map = np.zeros((grid_size, grid_size))
        
        # Divide image into grid
        grid_h = h // grid_size
        grid_w = w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract grid cell
                start_h = i * grid_h
                end_h = min((i + 1) * grid_h, h)
                start_w = j * grid_w
                end_w = min((j + 1) * grid_w, w)
                
                grid_cell = gray[start_h:end_h, start_w:end_w]
                
                # Estimate cell density using blob detection
                blobs = feature.blob_log(
                    grid_cell, 
                    min_sigma=2, 
                    max_sigma=15, 
                    num_sigma=10, 
                    threshold=0.1
                )
                
                cell_density_map[i, j] = len(blobs)
        
        return cell_density_map, (grid_h, grid_w)
    
    @staticmethod
    def suggest_roi_regions(image_array, num_regions=3, min_density_threshold=None):
        """
        Suggest optimal ROI regions based on cell density analysis
        """
        density_map, (grid_h, grid_w) = IntelligentROI.detect_cell_density_regions(image_array)
        
        if min_density_threshold is None:
            min_density_threshold = np.mean(density_map) + 0.5 * np.std(density_map)
        
        # Find high-density regions
        high_density_coords = np.where(density_map >= min_density_threshold)
        
        if len(high_density_coords[0]) == 0:
            # If no high density regions, use regions with above-average density
            avg_density = np.mean(density_map)
            high_density_coords = np.where(density_map >= avg_density)
        
        # Convert grid coordinates to image coordinates
        suggested_rois = []
        h, w = image_array.shape[:2]
        
        for i, j in zip(high_density_coords[0], high_density_coords[1]):
            roi = {
                'x': j * grid_w,
                'y': i * grid_h,
                'width': min(grid_w, w - j * grid_w),
                'height': min(grid_h, h - i * grid_h),
                'density_score': float(density_map[i, j]),
                'confidence': float(density_map[i, j] / np.max(density_map))
            }
            suggested_rois.append(roi)
        
        # Sort by density score and return top regions
        suggested_rois.sort(key=lambda x: x['density_score'], reverse=True)
        return suggested_rois[:num_regions]
    
    @staticmethod
    def optimize_roi_size(image_array, roi_region, target_cell_count=10):
        """
        Optimize ROI size to capture approximately target number of cells
        """
        x, y, w, h = roi_region['x'], roi_region['y'], roi_region['width'], roi_region['height']
        
        # Extract current ROI
        if len(image_array.shape) == 3:
            roi_image = np.mean(image_array[y:y+h, x:x+w], axis=2).astype(np.uint8)
        else:
            roi_image = image_array[y:y+h, x:x+w]
        
        # Estimate current cell count
        blobs = feature.blob_log(roi_image, min_sigma=3, max_sigma=20, num_sigma=15, threshold=0.02)
        current_count = len(blobs)
        
        if current_count == 0:
            return roi_region  # Return original if no cells detected
        
        # Calculate scaling factor
        scale_factor = np.sqrt(target_cell_count / current_count)
        scale_factor = max(0.5, min(2.0, scale_factor))  # Clamp scaling
        
        # Apply scaling
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Adjust position to keep ROI centered
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        
        # Ensure ROI stays within image bounds
        max_x = image_array.shape[1]
        max_y = image_array.shape[0]
        
        if new_x + new_w > max_x:
            new_w = max_x - new_x
        if new_y + new_h > max_y:
            new_h = max_y - new_y
        
        optimized_roi = {
            'x': new_x,
            'y': new_y,
            'width': new_w,
            'height': new_h,
            'estimated_cells': int(current_count * scale_factor**2),
            'scale_factor': scale_factor
        }
        
        return optimized_roi
    
    @staticmethod
    def merge_overlapping_rois(roi_list, overlap_threshold=0.3):
        """
        Merge ROI regions that overlap significantly
        """
        if len(roi_list) <= 1:
            return roi_list
        
        merged_rois = []
        used_indices = set()
        
        for i, roi1 in enumerate(roi_list):
            if i in used_indices:
                continue
            
            # Find overlapping ROIs
            overlapping_group = [roi1]
            used_indices.add(i)
            
            for j, roi2 in enumerate(roi_list[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                # Calculate intersection over union (IoU)
                x1_max = max(roi1['x'], roi2['x'])
                y1_max = max(roi1['y'], roi2['y'])
                x2_min = min(roi1['x'] + roi1['width'], roi2['x'] + roi2['width'])
                y2_min = min(roi1['y'] + roi1['height'], roi2['y'] + roi2['height'])
                
                if x2_min > x1_max and y2_min > y1_max:
                    intersection = (x2_min - x1_max) * (y2_min - y1_max)
                    area1 = roi1['width'] * roi1['height']
                    area2 = roi2['width'] * roi2['height']
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= overlap_threshold:
                        overlapping_group.append(roi2)
                        used_indices.add(j)
            
            # Merge overlapping ROIs
            if len(overlapping_group) > 1:
                # Find bounding box of all overlapping ROIs
                min_x = min(roi['x'] for roi in overlapping_group)
                min_y = min(roi['y'] for roi in overlapping_group)
                max_x = max(roi['x'] + roi['width'] for roi in overlapping_group)
                max_y = max(roi['y'] + roi['height'] for roi in overlapping_group)
                
                merged_roi = {
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y,
                    'merged_from': len(overlapping_group),
                    'density_score': np.mean([roi.get('density_score', 0) for roi in overlapping_group])
                }
                merged_rois.append(merged_roi)
            else:
                merged_rois.append(roi1)
        
        return merged_rois


class TextureAnalyzer:
    """
    Advanced texture analysis using Gray-Level Co-occurrence Matrix (GLCM) 
    and first-order statistical features for medical-grade morphometric analysis
    """
    
    @staticmethod
    def calculate_glcm_features(intensity_image, mask, distances=[1], angles=[0, 45, 90, 135]):
        """
        Calculate Gray-Level Co-occurrence Matrix (GLCM) texture features
        
        Args:
            intensity_image: Grayscale intensity image
            mask: Binary mask defining the region of interest
            distances: List of pixel distances for GLCM calculation
            angles: List of angles (in degrees) for GLCM calculation
            
        Returns:
            Dictionary containing GLCM texture features
        """
        if not SKIMAGE_AVAILABLE:
            return {}
        
        # Convert angles to radians
        angles_rad = [np.radians(angle) for angle in angles]
        
        # Extract masked region
        masked_region = intensity_image[mask > 0]
        if len(masked_region) == 0:
            return {}
        
        # Normalize intensity to 0-255 range for GLCM
        if masked_region.max() <= 1.0:
            masked_region = (masked_region * 255).astype(np.uint8)
        else:
            masked_region = masked_region.astype(np.uint8)
        
        # Create region image for GLCM calculation
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return {}
        
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        
        # Extract bounding box region
        region_intensity = intensity_image[min_y:max_y+1, min_x:max_x+1]
        region_mask = mask[min_y:max_y+1, min_x:max_x+1]
        
        # Apply mask to intensity image
        masked_intensity = region_intensity * region_mask
        
        # Normalize to 8-bit grayscale
        if masked_intensity.max() <= 1.0:
            masked_intensity = (masked_intensity * 255).astype(np.uint8)
        else:
            masked_intensity = masked_intensity.astype(np.uint8)
        
        try:
            # Calculate GLCM
            glcm = graycomatrix(
                masked_intensity, 
                distances=distances, 
                angles=angles_rad, 
                levels=256,
                symmetric=True, 
                normed=True
            )
            
            # Calculate GLCM properties
            features = {}
            
            # Basic GLCM features
            features['glcm_contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
            features['glcm_correlation'] = float(np.mean(graycoprops(glcm, 'correlation')))
            features['glcm_energy'] = float(np.mean(graycoprops(glcm, 'energy')))
            features['glcm_homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))
            
            # Additional GLCM features calculated manually
            features.update(TextureAnalyzer._calculate_advanced_glcm_features(glcm))
            
            return features
            
        except Exception as e:
            # Return empty dict if GLCM calculation fails
            return {}
    
    @staticmethod
    def _calculate_advanced_glcm_features(glcm):
        """
        Calculate advanced GLCM features not provided by scikit-image
        """
        features = {}
        
        try:
            # Average GLCM across distances and angles
            glcm_mean = np.mean(glcm, axis=(2, 3))
            
            # Create coordinate matrices
            i, j = np.ogrid[0:glcm_mean.shape[0], 0:glcm_mean.shape[1]]
            
            # Calculate additional features
            # Entropy
            glcm_entropy = -np.sum(glcm_mean * np.log2(glcm_mean + 1e-10))
            features['glcm_entropy'] = float(glcm_entropy)
            
            # Variance
            mu_i = np.sum(i * np.sum(glcm_mean, axis=1))
            mu_j = np.sum(j * np.sum(glcm_mean, axis=0))
            var_i = np.sum(((i - mu_i) ** 2) * np.sum(glcm_mean, axis=1))
            var_j = np.sum(((j - mu_j) ** 2) * np.sum(glcm_mean, axis=0))
            features['glcm_variance'] = float((var_i + var_j) / 2)
            
            # Sum Average
            k = np.arange(2, 2 * glcm_mean.shape[0])
            p_x_plus_y = np.array([np.sum(glcm_mean[i + j == k_val]) for k_val in k])
            features['glcm_sum_average'] = float(np.sum(k * p_x_plus_y))
            
            # Sum Variance
            sum_avg = features['glcm_sum_average']
            features['glcm_sum_variance'] = float(np.sum(((k - sum_avg) ** 2) * p_x_plus_y))
            
            # Sum Entropy
            features['glcm_sum_entropy'] = float(-np.sum(p_x_plus_y * np.log2(p_x_plus_y + 1e-10)))
            
            # Difference features
            k_diff = np.arange(0, glcm_mean.shape[0])
            p_x_minus_y = np.array([np.sum(glcm_mean[np.abs(i - j) == k_val]) for k_val in k_diff])
            
            features['glcm_difference_average'] = float(np.sum(k_diff * p_x_minus_y))
            diff_avg = features['glcm_difference_average']
            features['glcm_difference_variance'] = float(np.sum(((k_diff - diff_avg) ** 2) * p_x_minus_y))
            features['glcm_difference_entropy'] = float(-np.sum(p_x_minus_y * np.log2(p_x_minus_y + 1e-10)))
            
        except Exception:
            # Set default values if calculation fails
            for feature_name in ['glcm_entropy', 'glcm_variance', 'glcm_sum_average', 
                               'glcm_sum_variance', 'glcm_sum_entropy', 'glcm_difference_average',
                               'glcm_difference_variance', 'glcm_difference_entropy']:
                features[feature_name] = 0.0
        
        return features
    
    @staticmethod
    def calculate_first_order_features(intensity_image, mask):
        """
        Calculate first-order statistical features from intensity distribution
        
        Args:
            intensity_image: Grayscale intensity image
            mask: Binary mask defining the region of interest
            
        Returns:
            Dictionary containing first-order statistical features
        """
        # Extract intensity values within the mask
        intensity_values = intensity_image[mask > 0]
        
        if len(intensity_values) == 0:
            return {}
        
        features = {}
        
        try:
            # Basic statistical moments
            features['intensity_mean'] = float(np.mean(intensity_values))
            features['intensity_std'] = float(np.std(intensity_values))
            features['intensity_variance'] = float(np.var(intensity_values))
            
            # Skewness and Kurtosis
            from scipy.stats import skew, kurtosis
            features['intensity_skewness'] = float(skew(intensity_values))
            features['intensity_kurtosis'] = float(kurtosis(intensity_values))
            
            # Range features
            features['intensity_min'] = float(np.min(intensity_values))
            features['intensity_max'] = float(np.max(intensity_values))
            features['intensity_range'] = features['intensity_max'] - features['intensity_min']
            
            # Percentile features
            percentiles = [10, 25, 75, 90]
            for p in percentiles:
                features[f'intensity_p{p}'] = float(np.percentile(intensity_values, p))
            
            # Interquartile range
            features['intensity_iqr'] = features['intensity_p75'] - features['intensity_p25']
            
            # Entropy
            hist, _ = np.histogram(intensity_values, bins=256, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            features['intensity_entropy'] = float(-np.sum(hist * np.log2(hist)))
            
            # Energy (uniformity)
            features['intensity_energy'] = float(np.sum(hist ** 2))
            
            # Robust statistical measures
            features['intensity_median'] = float(np.median(intensity_values))
            features['intensity_mad'] = float(np.median(np.abs(intensity_values - features['intensity_median'])))
            
            # Coefficient of variation
            if features['intensity_mean'] != 0:
                features['intensity_cv'] = features['intensity_std'] / features['intensity_mean']
            else:
                features['intensity_cv'] = 0.0
            
        except Exception:
            # Return empty dict if calculation fails
            return {}
        
        return features
    
    @staticmethod
    def extract_all_texture_features(intensity_image, mask):
        """
        Extract all texture features (GLCM + first-order) for a cell region
        
        Args:
            intensity_image: Grayscale intensity image
            mask: Binary mask defining the cell region
            
        Returns:
            Dictionary containing all texture features
        """
        features = {}
        
        # Calculate GLCM features
        glcm_features = TextureAnalyzer.calculate_glcm_features(intensity_image, mask)
        features.update(glcm_features)
        
        # Calculate first-order statistical features
        first_order_features = TextureAnalyzer.calculate_first_order_features(intensity_image, mask)
        features.update(first_order_features)
        
        return features


class MorphometricValidator:
    """
    Validation and outlier detection for morphometric measurements
    """
    
    @staticmethod
    def detect_measurement_outliers(measurements, method='iqr', threshold=1.5):
        """
        Detect outliers in morphometric measurements using various methods
        """
        measurements = np.array(measurements)
        
        if method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(measurements, 25)
            q3 = np.percentile(measurements, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (measurements < lower_bound) | (measurements > upper_bound)
        
        elif method == 'zscore':
            # Z-score method
            mean_val = np.mean(measurements)
            std_val = np.std(measurements)
            z_scores = np.abs((measurements - mean_val) / std_val)
            outliers = z_scores > threshold
        
        elif method == 'modified_zscore':
            # Modified Z-score using median
            median_val = np.median(measurements)
            mad = np.median(np.abs(measurements - median_val))
            modified_z_scores = 0.6745 * (measurements - median_val) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    @staticmethod
    def validate_cell_measurements(detected_cells, enable_outlier_removal=True, outlier_method='iqr', 
                                 outlier_threshold=1.5, enable_physics_validation=True):
        """
        Comprehensive validation of cell measurements with configurable options
        """
        if not detected_cells:
            return {'valid_cells': [], 'outliers': [], 'validation_summary': {}}
        
        # Initialize outlier detection arrays
        combined_outliers = np.zeros(len(detected_cells), dtype=bool)
        physics_outliers = np.zeros(len(detected_cells), dtype=bool)
        
        outlier_reasons = {
            'area_outliers': 0,
            'perimeter_outliers': 0,
            'circularity_outliers': 0,
            'eccentricity_outliers': 0,
            'physics_violations': 0
        }
        
        # Statistical outlier detection (if enabled)
        if enable_outlier_removal:
            # Extract measurements
            areas = [cell.area for cell in detected_cells]
            perimeters = [cell.perimeter for cell in detected_cells]
            circularities = [cell.circularity for cell in detected_cells]
            eccentricities = [cell.eccentricity for cell in detected_cells]
            
            # Detect outliers for each measurement using specified method
            area_outliers = MorphometricValidator.detect_measurement_outliers(areas, outlier_method, outlier_threshold)
            perimeter_outliers = MorphometricValidator.detect_measurement_outliers(perimeters, outlier_method, outlier_threshold)
            circularity_outliers = MorphometricValidator.detect_measurement_outliers(circularities, outlier_method, outlier_threshold)
            eccentricity_outliers = MorphometricValidator.detect_measurement_outliers(eccentricities, outlier_method, outlier_threshold)
            
            # Count outliers by type
            outlier_reasons['area_outliers'] = int(np.sum(area_outliers))
            outlier_reasons['perimeter_outliers'] = int(np.sum(perimeter_outliers))
            outlier_reasons['circularity_outliers'] = int(np.sum(circularity_outliers))
            outlier_reasons['eccentricity_outliers'] = int(np.sum(eccentricity_outliers))
            
            # Combine outlier detection
            combined_outliers = area_outliers | perimeter_outliers | circularity_outliers | eccentricity_outliers
        
        # Physics-based validation (if enabled)
        if enable_physics_validation:
            for i, cell in enumerate(detected_cells):
                # Check area-perimeter relationship (should be reasonable)
                theoretical_radius = np.sqrt(cell.area / np.pi)
                theoretical_perimeter = 2 * np.pi * theoretical_radius
                perimeter_ratio = cell.perimeter / theoretical_perimeter
                
                # Flag if perimeter is too different from theoretical (indicates noise/artifacts)
                if perimeter_ratio < 0.5 or perimeter_ratio > 3.0:
                    physics_outliers[i] = True
                
                # Check aspect ratio reasonableness (cells shouldn't be extremely elongated)
                if cell.aspect_ratio > 10:
                    physics_outliers[i] = True
            
            outlier_reasons['physics_violations'] = int(np.sum(physics_outliers))
        
        final_outliers = combined_outliers | physics_outliers
        
        # Separate valid cells and outliers
        valid_cells = [cell for i, cell in enumerate(detected_cells) if not final_outliers[i]]
        outlier_cells = [cell for i, cell in enumerate(detected_cells) if final_outliers[i]]
        
        validation_summary = {
            'total_cells': len(detected_cells),
            'valid_cells': len(valid_cells),
            'outliers_detected': len(outlier_cells),
            'outlier_percentage': (len(outlier_cells) / len(detected_cells)) * 100,
            'outlier_reasons': {
                'area_outliers': int(np.sum(area_outliers)),
                'perimeter_outliers': int(np.sum(perimeter_outliers)),
                'circularity_outliers': int(np.sum(circularity_outliers)),
                'eccentricity_outliers': int(np.sum(eccentricity_outliers)),
                'physics_violations': int(np.sum(physics_outliers))
            }
        }
        
        return {
            'valid_cells': valid_cells,
            'outlier_cells': outlier_cells,
            'validation_summary': validation_summary
        }


class CellAnalysisProcessor:
    
    def __init__(self, analysis_id):
        self.analysis = CellAnalysis.objects.get(id=analysis_id)
        self.cell = self.analysis.cell
        
    def run_analysis(self):
        """Main analysis pipeline"""
        if not CELLPOSE_AVAILABLE or not SKIMAGE_AVAILABLE:
            self._mark_failed("Required dependencies (cellpose/scikit-image) not available")
            return False
            
        try:
            # Update status to processing
            self.analysis.status = 'processing'
            self.analysis.save()
            
            start_time = time.time()
            
            # Step 1: Load and preprocess image
            image_array = self._load_image()
            
            # Step 2: Run cellpose segmentation
            masks, flows, styles, diameters = self._run_cellpose_segmentation(image_array)
            
            # Step 2.5: Apply post-processing refinement with user-configured filtering
            original_mask_count = len(np.unique(masks)) - 1
            refinement_options = self.analysis.get_filtering_options()
            
            refined_masks, refinement_steps = SegmentationRefinement.refine_segmentation(
                masks, image_array, refinement_options
            )
            final_mask_count = len(np.unique(refined_masks)) - 1
            
            # Store refinement information
            refinement_info = {
                'original_cell_count': int(original_mask_count),
                'refined_cell_count': int(final_mask_count),
                'refinement_steps': refinement_steps,
                'options_used': refinement_options
            }
            
            # Add refinement info to quality metrics
            if not hasattr(self.analysis, 'quality_metrics') or not self.analysis.quality_metrics:
                self.analysis.quality_metrics = {}
            self.analysis.quality_metrics['segmentation_refinement'] = refinement_info
            
            # Use refined masks for further processing
            masks = refined_masks
            
            # Debug information
            print(f"DEBUG: Original cells detected: {original_mask_count}")
            print(f"DEBUG: Final cells after refinement: {final_mask_count}")
            print(f"DEBUG: Refinement steps: {refinement_steps}")
            
            # Step 3: Save all visualization images
            self._save_all_visualizations(image_array, masks, flows, styles, diameters)
            
            # Step 4: Extract morphometric features
            self._extract_morphometric_features(masks)
            
            # Step 5: Update analysis record
            processing_time = time.time() - start_time
            self.analysis.processing_time = processing_time
            self.analysis.completed_at = timezone.now()
            self.analysis.status = 'completed'
            self.analysis.save()
            
            return True
            
        except Exception as e:
            self._mark_failed(str(e))
            return False
    
    def _load_image(self):
        """Load and preprocess image from file"""
        image_path = self.cell.image.path
        
        # Use cellpose's imread for better compatibility
        image_array = imread(image_path)
        
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        
        # Perform quality assessment
        quality_assessment = ImageQualityAssessment.assess_overall_quality(image_array)
        
        # Store quality metrics in analysis
        self.analysis.quality_metrics = quality_assessment
        
        # Auto-optimize parameters if diameter is 0 (auto-detection requested)
        if self.analysis.cellpose_diameter == 0:
            current_params = {
                'cellpose_model': self.analysis.cellpose_model,
                'flow_threshold': self.analysis.flow_threshold,
                'cellprob_threshold': self.analysis.cellprob_threshold
            }
            
            parameter_recommendations = ParameterOptimizer.optimize_all_parameters(
                image_array, quality_assessment, current_params
            )
            
            # Update analysis parameters with recommendations
            self.analysis.cellpose_diameter = parameter_recommendations['cellpose_diameter']
            
            # Store optimization info for user feedback
            optimization_info = {
                'auto_optimized': True,
                'original_parameters': current_params,
                'optimized_parameters': {
                    'cellpose_diameter': parameter_recommendations['cellpose_diameter'],
                    'flow_threshold': parameter_recommendations['flow_threshold'],
                    'cellprob_threshold': parameter_recommendations['cellprob_threshold'],
                    'cellpose_model': parameter_recommendations['cellpose_model']
                },
                'confidence_scores': parameter_recommendations['confidence_scores'],
                'optimization_notes': parameter_recommendations['optimization_notes']
            }
            
            # Update quality metrics to include optimization info
            self.analysis.quality_metrics['parameter_optimization'] = optimization_info
            
            # Optionally update other parameters if confidence is high enough
            if parameter_recommendations['confidence_scores']['threshold_confidence'] > 0.7:
                self.analysis.flow_threshold = parameter_recommendations['flow_threshold']
                self.analysis.cellprob_threshold = parameter_recommendations['cellprob_threshold']
            
            if parameter_recommendations['confidence_scores']['model_confidence'] > 0.7:
                self.analysis.cellpose_model = parameter_recommendations['cellpose_model']
        
        # Apply preprocessing if requested
        preprocessing_options = getattr(self.analysis, 'preprocessing_options', {})
        if preprocessing_options and any(preprocessing_options.values()):
            preprocessor = ImagePreprocessor(preprocessing_options)
            processed_image, preprocessing_steps = preprocessor.preprocess_image(image_array)
            
            # Store preprocessing information
            self.analysis.preprocessing_applied = True
            self.analysis.preprocessing_steps = preprocessing_steps
            
            return processed_image
        else:
            # No preprocessing requested
            self.analysis.preprocessing_applied = False
            self.analysis.preprocessing_steps = []
            
            return image_array
    
    def _run_cellpose_segmentation(self, image_array):
        """Run cellpose segmentation with optional ROI processing"""
        # Initialize cellpose model
        model_name = self.analysis.cellpose_model
        model = models.CellposeModel(gpu=True, model_type=model_name)
        
        # Set parameters
        diameter = self.analysis.cellpose_diameter if self.analysis.cellpose_diameter > 0 else None
        flow_threshold = self.analysis.flow_threshold
        cellprob_threshold = self.analysis.cellprob_threshold
        
        if self.analysis.use_roi and self.analysis.roi_regions:
            # Debug ROI processing
            print(f"DEBUG: ROI enabled with {len(self.analysis.roi_regions)} regions")
            print(f"DEBUG: ROI data: {self.analysis.roi_regions}")
            print(f"DEBUG: Image shape: {image_array.shape}")
            
            # Process ROI regions
            masks = self._process_roi_regions(image_array, model, diameter, flow_threshold, cellprob_threshold)
            flows = None
            styles = None
            diameters = None
        else:
            # Run standard full-image segmentation
            result = model.eval(
                image_array, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            # Handle different return formats from Cellpose versions
            if len(result) == 4:
                masks, flows, styles, diameters = result
            elif len(result) == 3:
                masks, flows, styles = result
                diameters = None
            else:
                masks = result[0]
                flows = result[1] if len(result) > 1 else None
                styles = result[2] if len(result) > 2 else None
                diameters = None
        
        # Update cell count
        unique_masks = np.unique(masks)
        # Remove background (0)
        num_cells = len(unique_masks) - 1 if 0 in unique_masks else len(unique_masks)
        self.analysis.num_cells_detected = num_cells
        self.analysis.save()
        
        return masks, flows, styles, diameters
    
    def _process_roi_regions(self, image_array, model, diameter, flow_threshold, cellprob_threshold):
        """Process multiple ROI regions and combine results"""
        print(f"DEBUG: _process_roi_regions called with {len(self.analysis.roi_regions)} ROI regions")
        
        # Initialize full image mask
        full_masks = np.zeros(image_array.shape[:2], dtype=np.int32)
        current_cell_id = 1
        
        for roi_idx, roi in enumerate(self.analysis.roi_regions):
            print(f"DEBUG: Processing ROI {roi_idx + 1}: {roi}")
            
            # Extract ROI coordinates (ensure they're integers and within bounds)
            x = max(0, int(roi['x']))
            y = max(0, int(roi['y']))
            w = min(image_array.shape[1] - x, int(roi['width']))
            h = min(image_array.shape[0] - y, int(roi['height']))
            
            print(f"DEBUG: ROI {roi_idx + 1} coordinates: x={x}, y={y}, w={w}, h={h}")
            
            if w <= 0 or h <= 0:
                print(f"DEBUG: Skipping invalid ROI {roi_idx + 1}")
                continue  # Skip invalid ROI
            
            # Extract ROI from image
            roi_image = image_array[y:y+h, x:x+w]
            
            if roi_image.size == 0:
                continue  # Skip empty ROI
            
            print(f"DEBUG: ROI image shape: {roi_image.shape}")
            
            try:
                # Run cellpose on ROI
                print(f"DEBUG: Running cellpose on ROI {roi_idx + 1}")
                roi_masks, _, _ = model.eval(
                    roi_image,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )
                
                unique_roi_masks = np.unique(roi_masks)
                print(f"DEBUG: ROI {roi_idx + 1} detected {len(unique_roi_masks) - 1} cells")
                
                # Adjust cell IDs to avoid conflicts
                roi_masks_adjusted = np.zeros_like(roi_masks)
                
                for mask_id in unique_roi_masks:
                    if mask_id == 0:  # Skip background
                        continue
                    roi_masks_adjusted[roi_masks == mask_id] = current_cell_id
                    current_cell_id += 1
                
                # Place ROI results back into full image
                full_masks[y:y+h, x:x+w] = np.maximum(
                    full_masks[y:y+h, x:x+w], 
                    roi_masks_adjusted
                )
                
                print(f"DEBUG: ROI {roi_idx + 1} processed successfully")
                
            except Exception as e:
                print(f"DEBUG: Error processing ROI {roi_idx + 1}: {str(e)}")
                continue
        
        return full_masks
    
    def _save_all_visualizations(self, original_image, masks, flows=None, styles=None, diameters=None):
        """Save all 4 visualization pages"""
        # Page 1: Core Pipeline (6 panels)
        self._save_core_pipeline_visualization(original_image, masks, flows, styles, diameters)
        
        # Only create advanced visualizations if we have flow data
        if flows is not None:
            # Page 2: Advanced Flow Analysis
            self._save_flow_analysis_visualization(original_image, masks, flows, styles, diameters)
            
            # Page 3: Style & Quality Analysis  
            self._save_style_quality_visualization(original_image, masks, flows, styles, diameters)
            
            # Page 4: Edge & Boundary Analysis
            self._save_edge_boundary_visualization(original_image, masks, flows, styles, diameters)

    def _save_core_pipeline_visualization(self, original_image, masks, flows=None, styles=None, diameters=None):
        """Page 1: Core Pipeline - Create and save comprehensive visualization of Cellpose pipeline results"""
        unique_masks = np.unique(masks)[1:]  # Skip background
        
        # Determine number of panels based on available data
        if flows is not None:
            # Full pipeline with intermediate steps
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
        else:
            # ROI mode - simpler visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Panel 1: Original Image
        ax_idx = 0
        if len(original_image.shape) == 3:
            axes[ax_idx].imshow(original_image)
        else:
            axes[ax_idx].imshow(original_image, cmap='gray')
        axes[ax_idx].set_title('1. Original Image', fontsize=12, fontweight='bold')
        axes[ax_idx].axis('off')
        ax_idx += 1
        
        if flows is not None:
            # Panel 2: Predicted Outlines
            if len(original_image.shape) == 3:
                axes[ax_idx].imshow(original_image)
            else:
                axes[ax_idx].imshow(original_image, cmap='gray')
            
            # Extract and plot outlines
            try:
                outlines = utils.outlines_list(masks)
                for outline in outlines:
                    axes[ax_idx].plot(outline[:, 0], outline[:, 1], linewidth=1.5, color='cyan')
            except:
                # Fallback: draw contours manually
                from skimage import measure
                contours = measure.find_contours(masks > 0, 0.5)
                for contour in contours:
                    axes[ax_idx].plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan')
            
            axes[ax_idx].set_title('2. Predicted Outlines', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            # Panel 3: Flow Fields
            try:
                # Convert flow field to RGB visualization
                if flows is not None and len(flows) > 0 and len(flows[0]) > 1:
                    xy_flows = flows[0][1]  # XY flows at each pixel
                    # Ensure correct shape and data type for dx_to_circ
                    if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                        # dx_to_circ expects [dY, dX] format (note: Y first, then X)
                        flow_data = np.stack([xy_flows[:,:,1], xy_flows[:,:,0]], axis=-1)
                        flow_rgb = plot.dx_to_circ(flow_data)
                        axes[ax_idx].imshow(flow_rgb)
                    else:
                        # Fallback for incorrect shape
                        axes[ax_idx].imshow(np.zeros_like(original_image), cmap='gray')
                        axes[ax_idx].text(0.5, 0.5, 'Flow shape error', 
                                        transform=axes[ax_idx].transAxes, ha='center', va='center')
                else:
                    # Fallback visualization
                    axes[ax_idx].imshow(np.zeros_like(original_image), cmap='gray')
                    axes[ax_idx].text(0.5, 0.5, 'Flow data unavailable', 
                                    transform=axes[ax_idx].transAxes, ha='center', va='center')
            except Exception as e:
                axes[ax_idx].imshow(np.zeros_like(original_image), cmap='gray')
                axes[ax_idx].text(0.5, 0.5, f'Flow visualization error', 
                                transform=axes[ax_idx].transAxes, ha='center', va='center')
            
            axes[ax_idx].set_title('3. Flow Fields', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            # Panel 4: Cell Probability Map
            try:
                if len(flows) > 0 and len(flows[0]) > 2:
                    cell_prob = flows[0][2]  # Cell probability
                    im = axes[ax_idx].imshow(cell_prob, cmap='viridis')
                    plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
                else:
                    axes[ax_idx].imshow(np.zeros_like(original_image), cmap='gray')
                    axes[ax_idx].text(0.5, 0.5, 'Probability data unavailable', 
                                    transform=axes[ax_idx].transAxes, ha='center', va='center')
            except Exception as e:
                axes[ax_idx].imshow(np.zeros_like(original_image), cmap='gray')
                axes[ax_idx].text(0.5, 0.5, f'Probability visualization error', 
                                transform=axes[ax_idx].transAxes, ha='center', va='center')
            
            axes[ax_idx].set_title('4. Cell Probability Map', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            # Panel 5: Final Masks
            if len(original_image.shape) == 3:
                base_img = np.mean(original_image, axis=2)
            else:
                base_img = original_image
            axes[ax_idx].imshow(base_img, cmap='gray', alpha=0.7)
            
            # Create colored overlay for masks
            colored_masks = np.zeros((*masks.shape, 3))
            if len(unique_masks) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_masks)))
                
                for i, mask_id in enumerate(unique_masks):
                    mask_pixels = masks == mask_id
                    colored_masks[mask_pixels] = colors[i][:3]
                    
                axes[ax_idx].imshow(colored_masks, alpha=0.8)
            else:
                # No cells detected - show message
                axes[ax_idx].text(0.5, 0.5, 'No cells detected after filtering', 
                                transform=axes[ax_idx].transAxes, ha='center', va='center', 
                                fontsize=12, color='red', fontweight='bold')
            
            axes[ax_idx].set_title(f'5. Final Masks ({len(unique_masks)} cells)', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            # Panel 6: Cell Centers/Poses
            if len(original_image.shape) == 3:
                axes[ax_idx].imshow(original_image)
            else:
                axes[ax_idx].imshow(original_image, cmap='gray')
            
            # Extract cell centers from masks
            if len(unique_masks) > 0:
                from skimage import measure
                props = measure.regionprops(masks)
                for prop in props:
                    if prop.label > 0:
                        y, x = prop.centroid
                        axes[ax_idx].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
                        axes[ax_idx].text(x+5, y+5, str(prop.label), color='red', fontsize=8, fontweight='bold')
            else:
                # No cells detected - show message
                axes[ax_idx].text(0.5, 0.5, 'No cell centers - no cells detected after filtering', 
                                transform=axes[ax_idx].transAxes, ha='center', va='center', 
                                fontsize=10, color='red', fontweight='bold')
            
            axes[ax_idx].set_title(f'6. Cell Centers ({len(unique_masks)} cells)', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            
        else:
            # ROI mode - simple segmentation overlay
            if len(original_image.shape) == 3:
                axes[ax_idx].imshow(original_image)
            else:
                axes[ax_idx].imshow(original_image, cmap='gray')
            
            # Create colored overlay for masks
            colored_masks = np.zeros((*masks.shape, 3))
            if len(unique_masks) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_masks)))
                
                for i, mask_id in enumerate(unique_masks):
                    mask_pixels = masks == mask_id
                    colored_masks[mask_pixels] = colors[i][:3]
                
                axes[ax_idx].imshow(colored_masks, alpha=0.6)
            else:
                # No cells detected - show message
                axes[ax_idx].text(0.5, 0.5, 'No cells detected after filtering', 
                                transform=axes[ax_idx].transAxes, ha='center', va='center', 
                                fontsize=12, color='red', fontweight='bold')
            
            # Draw ROI boundaries if ROI selection was used
            if self.analysis.use_roi and self.analysis.roi_regions:
                for roi_idx, roi in enumerate(self.analysis.roi_regions):
                    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='--')
                    axes[ax_idx].add_patch(rect)
                    # Add ROI label
                    axes[ax_idx].text(x + 5, y + 15, f'ROI {roi_idx + 1}', 
                                   color='red', fontsize=10, weight='bold')
                
                axes[ax_idx].set_title(f'2. Segmentation ({len(unique_masks)} cells, {len(self.analysis.roi_regions)} ROI regions)')
            else:
                axes[ax_idx].set_title(f'2. Segmentation ({len(unique_masks)} cells)')
            
            axes[ax_idx].axis('off')
        
        plt.tight_layout(pad=2.0)
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Save to model
        filename = f"segmentation_{self.analysis.id}_{self.cell.id}.png"
        self.analysis.segmentation_image.save(
            filename,
            ContentFile(buffer.getvalue()),
            save=False
        )
        self.analysis.save()

    def _save_flow_analysis_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 2: Advanced Flow Analysis - HSV flows, trajectories, magnitude, convergence"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Panel 1: HSV Flow Visualization
        try:
            if len(flows) > 0 and len(flows[0]) > 0:
                hsv_flow = flows[0][0]  # XY flow in HSV 0-255
                axes[0].imshow(hsv_flow)
            else:
                axes[0].imshow(np.zeros_like(original_image), cmap='gray')
                axes[0].text(0.5, 0.5, 'HSV flow data unavailable', 
                           transform=axes[0].transAxes, ha='center', va='center')
        except Exception:
            axes[0].imshow(np.zeros_like(original_image), cmap='gray')
            axes[0].text(0.5, 0.5, 'HSV flow error', 
                       transform=axes[0].transAxes, ha='center', va='center')
        
        axes[0].set_title('1. HSV Flow Representation', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Pixel Trajectories/Final Locations
        try:
            if len(flows) > 0 and len(flows[0]) > 3:
                final_locations = flows[0][3]  # Final pixel locations after Euler integration
                if len(original_image.shape) == 3:
                    base_img = np.mean(original_image, axis=2)
                else:
                    base_img = original_image
                axes[1].imshow(base_img, cmap='gray', alpha=0.7)
                
                # Visualize pixel trajectories as a heatmap
                if final_locations is not None and final_locations.size > 0:
                    axes[1].imshow(final_locations, alpha=0.6, cmap='plasma')
                else:
                    axes[1].text(0.5, 0.5, 'Trajectory data unavailable', 
                               transform=axes[1].transAxes, ha='center', va='center', color='white')
            else:
                axes[1].imshow(np.zeros_like(original_image), cmap='gray')
        except Exception:
            axes[1].imshow(np.zeros_like(original_image), cmap='gray')
            axes[1].text(0.5, 0.5, 'Trajectory error', 
                       transform=axes[1].transAxes, ha='center', va='center')
        
        axes[1].set_title('2. Pixel Trajectories', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Flow Magnitude Heatmap
        try:
            if len(flows) > 0 and len(flows[0]) > 1:
                xy_flows = flows[0][1]  # XY flows at each pixel
                if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                    # Calculate flow magnitude
                    flow_magnitude = np.sqrt(xy_flows[:,:,0]**2 + xy_flows[:,:,1]**2)
                    im = axes[2].imshow(flow_magnitude, cmap='hot')
                    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                else:
                    axes[2].imshow(np.zeros_like(original_image), cmap='gray')
                    axes[2].text(0.5, 0.5, 'Flow magnitude unavailable', 
                               transform=axes[2].transAxes, ha='center', va='center')
            else:
                axes[2].imshow(np.zeros_like(original_image), cmap='gray')
        except Exception:
            axes[2].imshow(np.zeros_like(original_image), cmap='gray')
            axes[2].text(0.5, 0.5, 'Magnitude error', 
                       transform=axes[2].transAxes, ha='center', va='center')
        
        axes[2].set_title('3. Flow Magnitude Heatmap', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Panel 4: Flow Convergence Points (Advanced Cell Centers)
        if len(original_image.shape) == 3:
            axes[3].imshow(original_image)
        else:
            axes[3].imshow(original_image, cmap='gray')
        
        try:
            # Calculate flow convergence using advanced analysis
            from skimage import measure
            props = measure.regionprops(masks)
            
            # Extract convergence points from flow analysis
            if len(flows) > 0 and len(flows[0]) > 1:
                xy_flows = flows[0][1]
                if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                    # Find convergence points by analyzing flow directions
                    for prop in props:
                        if prop.label > 0:
                            y, x = prop.centroid
                            # Plot enhanced center markers
                            axes[3].plot(x, y, 'yo', markersize=12, markeredgecolor='red', markeredgewidth=2)
                            axes[3].plot(x, y, 'r+', markersize=8, markeredgewidth=3)
                            axes[3].text(x+8, y+8, str(prop.label), color='yellow', fontsize=10, 
                                       fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            else:
                # Fallback to simple centroids
                for prop in props:
                    if prop.label > 0:
                        y, x = prop.centroid
                        axes[3].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
                        axes[3].text(x+5, y+5, str(prop.label), color='red', fontsize=8, fontweight='bold')
        except Exception:
            axes[3].text(0.5, 0.95, 'Convergence analysis error', 
                       transform=axes[3].transAxes, ha='center', va='top', color='red')
        
        unique_masks = np.unique(masks)[1:]
        axes[3].set_title(f'4. Flow Convergence Points ({len(unique_masks)} cells)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout(pad=2.0)
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Save to model
        filename = f"flow_analysis_{self.analysis.id}_{self.cell.id}.png"
        self.analysis.flow_analysis_image.save(
            filename,
            ContentFile(buffer.getvalue()),
            save=False
        )

    def _save_style_quality_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 3: Style & Quality Analysis - style vectors, diameter distribution, multi-threshold"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Panel 1: Style Vector Visualization
        if styles is not None and len(styles) > 0:
            # Plot style vector as bar chart
            style_data = styles[0] if isinstance(styles, list) else styles
            if hasattr(style_data, '__len__') and len(style_data) > 1:
                x_pos = np.arange(min(len(style_data), 50))  # Show first 50 components
                axes[0].bar(x_pos, style_data[:len(x_pos)], alpha=0.7, color='steelblue')
                axes[0].set_xlabel('Style Component')
                axes[0].set_ylabel('Value')
            else:
                axes[0].text(0.5, 0.5, 'Style data unavailable', 
                           transform=axes[0].transAxes, ha='center', va='center')
        else:
            axes[0].text(0.5, 0.5, 'Style data unavailable', 
                       transform=axes[0].transAxes, ha='center', va='center')
        
        axes[0].set_title('1. Style Vector Components', fontsize=12, fontweight='bold')
        
        # Panel 2: Diameter Distribution
        if diameters is not None and len(diameters) > 0:
            diameter_data = diameters[0] if isinstance(diameters, list) else diameters
            if hasattr(diameter_data, '__len__') and len(diameter_data) > 1:
                axes[1].hist(diameter_data, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1].set_xlabel('Diameter (pixels)')
                axes[1].set_ylabel('Frequency')
                axes[1].axvline(np.mean(diameter_data), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(diameter_data):.1f}')
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, f'Single diameter: {diameter_data:.1f}', 
                           transform=axes[1].transAxes, ha='center', va='center')
        else:
            axes[1].text(0.5, 0.5, 'Diameter data unavailable', 
                       transform=axes[1].transAxes, ha='center', va='center')
        
        axes[1].set_title('2. Diameter Distribution', fontsize=12, fontweight='bold')
        
        # Panel 3: Multi-threshold Probability Analysis
        try:
            if len(flows) > 0 and len(flows[0]) > 2:
                cell_prob = flows[0][2]
                thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                
                # Create subplot for threshold analysis
                threshold_data = []
                for thresh in thresholds:
                    above_thresh = np.sum(cell_prob > thresh)
                    threshold_data.append(above_thresh)
                
                axes[2].plot(thresholds, threshold_data, 'bo-', linewidth=2, markersize=8)
                axes[2].set_xlabel('Probability Threshold')
                axes[2].set_ylabel('Pixels Above Threshold')
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, 'Probability data unavailable', 
                           transform=axes[2].transAxes, ha='center', va='center')
        except Exception:
            axes[2].text(0.5, 0.5, 'Threshold analysis error', 
                       transform=axes[2].transAxes, ha='center', va='center')
        
        axes[2].set_title('3. Multi-threshold Analysis', fontsize=12, fontweight='bold')
        
        # Panel 4: Quality Metrics Summary
        quality_metrics = getattr(self.analysis, 'quality_metrics', {})
        if quality_metrics:
            # Create a text summary of quality metrics
            axes[3].axis('off')
            
            quality_text = []
            if 'overall_score' in quality_metrics:
                quality_text.append(f"Overall Score: {quality_metrics['overall_score']:.1f}/100")
            if 'quality_category' in quality_metrics:
                quality_text.append(f"Category: {quality_metrics['quality_category'].title()}")
            
            # Add blur, contrast, noise metrics if available
            for metric_type in ['blur_metrics', 'contrast_metrics', 'noise_metrics']:
                if metric_type in quality_metrics:
                    metrics = quality_metrics[metric_type]
                    score_key = metric_type.replace('_metrics', '_score')
                    if score_key in metrics:
                        quality_text.append(f"{metric_type.replace('_', ' ').title()}: {metrics[score_key]:.1f}")
            
            # Display as formatted text
            text_str = '\n'.join(quality_text)
            axes[3].text(0.1, 0.9, text_str, transform=axes[3].transAxes, 
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        else:
            axes[3].text(0.5, 0.5, 'Quality metrics unavailable', 
                       transform=axes[3].transAxes, ha='center', va='center')
        
        axes[3].set_title('4. Quality Assessment Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout(pad=2.0)
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Save to model
        filename = f"style_quality_{self.analysis.id}_{self.cell.id}.png"
        self.analysis.style_quality_image.save(
            filename,
            ContentFile(buffer.getvalue()),
            save=False
        )

    def _save_edge_boundary_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 4: Edge & Boundary Analysis - edges, gradients, flow arrows"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Panel 1: Mask Edges/Boundaries
        try:
            # Create binary mask edges
            from skimage import feature, filters
            if len(original_image.shape) == 3:
                gray = np.mean(original_image, axis=2)
            else:
                gray = original_image
            
            # Method 1: Mask boundaries
            mask_edges = np.zeros_like(masks, dtype=bool)
            unique_masks = np.unique(masks)[1:]
            
            for mask_id in unique_masks:
                single_mask = (masks == mask_id)
                edges = feature.canny(single_mask.astype(float), sigma=1.0)
                mask_edges |= edges
            
            axes[0].imshow(gray, cmap='gray', alpha=0.7)
            axes[0].imshow(mask_edges, alpha=0.8, cmap='Reds')
        except Exception:
            axes[0].imshow(np.zeros_like(original_image), cmap='gray')
            axes[0].text(0.5, 0.5, 'Edge detection error', 
                       transform=axes[0].transAxes, ha='center', va='center')
        
        axes[0].set_title('1. Mask Boundaries/Edges', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Gradient Analysis
        try:
            if len(flows) > 0 and len(flows[0]) > 1:
                xy_flows = flows[0][1]
                if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                    # Calculate gradients of flow field
                    flow_x = xy_flows[:,:,0]
                    flow_y = xy_flows[:,:,1]
                    
                    grad_x = np.gradient(flow_x)[1]
                    grad_y = np.gradient(flow_y)[0]
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    im = axes[1].imshow(gradient_magnitude, cmap='viridis')
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                else:
                    axes[1].imshow(np.zeros_like(original_image), cmap='gray')
            else:
                axes[1].imshow(np.zeros_like(original_image), cmap='gray')
        except Exception:
            axes[1].imshow(np.zeros_like(original_image), cmap='gray')
            axes[1].text(0.5, 0.5, 'Gradient error', 
                       transform=axes[1].transAxes, ha='center', va='center')
        
        axes[1].set_title('2. Flow Gradient Analysis', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Flow Direction Arrows
        if len(original_image.shape) == 3:
            axes[2].imshow(original_image, alpha=0.7)
        else:
            axes[2].imshow(original_image, cmap='gray', alpha=0.7)
        
        try:
            if len(flows) > 0 and len(flows[0]) > 1:
                xy_flows = flows[0][1]
                if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                    # Subsample for arrow visualization
                    step = max(1, min(xy_flows.shape[:2]) // 20)
                    y_coords, x_coords = np.meshgrid(
                        np.arange(0, xy_flows.shape[0], step),
                        np.arange(0, xy_flows.shape[1], step),
                        indexing='ij'
                    )
                    
                    u = xy_flows[::step, ::step, 0]
                    v = xy_flows[::step, ::step, 1]
                    
                    axes[2].quiver(x_coords, y_coords, u, v, 
                                 alpha=0.8, color='cyan', scale=50, width=0.003)
        except Exception:
            axes[2].text(0.95, 0.95, 'Flow arrows error', 
                       transform=axes[2].transAxes, ha='right', va='top', color='red')
        
        axes[2].set_title('3. Flow Direction Arrows', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Panel 4: Combined Edge + Flow Analysis
        try:
            if len(original_image.shape) == 3:
                base_img = np.mean(original_image, axis=2)
            else:
                base_img = original_image
            
            axes[3].imshow(base_img, cmap='gray', alpha=0.6)
            
            # Overlay mask edges
            if 'mask_edges' in locals():
                axes[3].imshow(mask_edges, alpha=0.6, cmap='Reds')
            
            # Add cell centers
            from skimage import measure
            props = measure.regionprops(masks)
            for prop in props:
                if prop.label > 0:
                    y, x = prop.centroid
                    axes[3].plot(x, y, 'go', markersize=6, markeredgecolor='yellow', markeredgewidth=1)
        except Exception:
            axes[3].imshow(np.zeros_like(original_image), cmap='gray')
            axes[3].text(0.5, 0.5, 'Combined analysis error', 
                       transform=axes[3].transAxes, ha='center', va='center')
        
        unique_masks = np.unique(masks)[1:]
        axes[3].set_title(f'4. Combined Edge+Flow Analysis ({len(unique_masks)} cells)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout(pad=2.0)
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Save to model
        filename = f"edge_boundary_{self.analysis.id}_{self.cell.id}.png"
        self.analysis.edge_boundary_image.save(
            filename,
            ContentFile(buffer.getvalue()),
            save=False
        )
    
    def _extract_morphometric_features(self, masks):
        """Extract morphometric features from segmentation masks"""
        # Clear existing detected cells
        self.analysis.detected_cells.all().delete()
        
        # Get unique cell IDs (excluding background 0)
        unique_masks = np.unique(masks)
        cell_ids = unique_masks[unique_masks > 0]
        
        if len(cell_ids) == 0:
            return
        
        # Load original image for texture analysis
        image_path = self.cell.image.path
        original_image = imread(image_path)
        
        # Convert to grayscale for texture analysis
        if len(original_image.shape) == 3:
            intensity_image = np.mean(original_image, axis=2).astype(np.float32)
        else:
            intensity_image = original_image.astype(np.float32)
        
        # Normalize intensity image to 0-1 range
        if intensity_image.max() > 1.0:
            intensity_image = intensity_image / 255.0
        
        # Create binary masks for regionprops
        binary_masks = masks > 0
        
        # Use regionprops to get measurements
        props = measure.regionprops(masks, intensity_image=intensity_image)
        
        detected_cells = []
        
        for prop in props:
            if prop.label == 0:  # Skip background
                continue
                
            # Basic measurements
            area = prop.area
            perimeter = prop.perimeter
            
            # Shape descriptors
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            extent = prop.extent
            
            # Ellipse fitting
            major_axis_length = prop.major_axis_length
            minor_axis_length = prop.minor_axis_length
            aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0
            
            # Position
            centroid_y, centroid_x = prop.centroid
            
            # Bounding box
            min_row, min_col, max_row, max_col = prop.bbox
            
            # Extract texture features for this cell
            cell_mask = (masks == prop.label).astype(np.uint8)
            texture_features = TextureAnalyzer.extract_all_texture_features(intensity_image, cell_mask)
            
            # Calculate physical measurements if scale is available
            area_microns_sq = None
            perimeter_microns = None
            major_axis_length_microns = None
            minor_axis_length_microns = None
            
            if self.cell.scale_set and self.cell.pixels_per_micron:
                area_microns_sq = self.cell.convert_area_to_microns_squared(area)
                perimeter_microns = self.cell.convert_pixels_to_microns(perimeter)
                major_axis_length_microns = self.cell.convert_pixels_to_microns(major_axis_length)
                minor_axis_length_microns = self.cell.convert_pixels_to_microns(minor_axis_length)
            
            # Create DetectedCell instance
            detected_cell = DetectedCell(
                analysis=self.analysis,
                cell_id=prop.label,
                area=area,
                perimeter=perimeter,
                area_microns_sq=area_microns_sq,
                perimeter_microns=perimeter_microns,
                circularity=circularity,
                eccentricity=eccentricity,
                solidity=solidity,
                extent=extent,
                major_axis_length=major_axis_length,
                minor_axis_length=minor_axis_length,
                major_axis_length_microns=major_axis_length_microns,
                minor_axis_length_microns=minor_axis_length_microns,
                aspect_ratio=aspect_ratio,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                bounding_box_x=min_col,
                bounding_box_y=min_row,
                bounding_box_width=max_col - min_col,
                bounding_box_height=max_row - min_row,
                # GLCM Texture Features
                glcm_contrast=texture_features.get('glcm_contrast'),
                glcm_correlation=texture_features.get('glcm_correlation'),
                glcm_energy=texture_features.get('glcm_energy'),
                glcm_homogeneity=texture_features.get('glcm_homogeneity'),
                glcm_entropy=texture_features.get('glcm_entropy'),
                glcm_variance=texture_features.get('glcm_variance'),
                glcm_sum_average=texture_features.get('glcm_sum_average'),
                glcm_sum_variance=texture_features.get('glcm_sum_variance'),
                glcm_sum_entropy=texture_features.get('glcm_sum_entropy'),
                glcm_difference_average=texture_features.get('glcm_difference_average'),
                glcm_difference_variance=texture_features.get('glcm_difference_variance'),
                glcm_difference_entropy=texture_features.get('glcm_difference_entropy'),
                # First-Order Statistical Features
                intensity_mean=texture_features.get('intensity_mean'),
                intensity_std=texture_features.get('intensity_std'),
                intensity_variance=texture_features.get('intensity_variance'),
                intensity_skewness=texture_features.get('intensity_skewness'),
                intensity_kurtosis=texture_features.get('intensity_kurtosis'),
                intensity_min=texture_features.get('intensity_min'),
                intensity_max=texture_features.get('intensity_max'),
                intensity_range=texture_features.get('intensity_range'),
                intensity_p10=texture_features.get('intensity_p10'),
                intensity_p25=texture_features.get('intensity_p25'),
                intensity_p75=texture_features.get('intensity_p75'),
                intensity_p90=texture_features.get('intensity_p90'),
                intensity_iqr=texture_features.get('intensity_iqr'),
                intensity_entropy=texture_features.get('intensity_entropy'),
                intensity_energy=texture_features.get('intensity_energy'),
                intensity_median=texture_features.get('intensity_median'),
                intensity_mad=texture_features.get('intensity_mad'),
                intensity_cv=texture_features.get('intensity_cv')
            )
            
            detected_cells.append(detected_cell)
        
        # Apply morphometric validation based on user settings
        if self.analysis.enable_outlier_removal or self.analysis.enable_physics_validation:
            validation_results = MorphometricValidator.validate_cell_measurements(
                detected_cells,
                enable_outlier_removal=self.analysis.enable_outlier_removal,
                outlier_method=self.analysis.outlier_method,
                outlier_threshold=self.analysis.outlier_threshold,
                enable_physics_validation=self.analysis.enable_physics_validation
            )
            
            # Store validation information
            if not hasattr(self.analysis, 'quality_metrics') or not self.analysis.quality_metrics:
                self.analysis.quality_metrics = {}
            self.analysis.quality_metrics['morphometric_validation'] = validation_results['validation_summary']
            
            # Use only validated cells (remove outliers)
            valid_detected_cells = validation_results['valid_cells']
            outlier_count = len(detected_cells) - len(valid_detected_cells)
            
            if outlier_count > 0:
                print(f"Filtered out {outlier_count} outlier cells during validation")
        else:
            # No validation filtering - use all detected cells
            valid_detected_cells = detected_cells
            print("Morphometric validation disabled - keeping all detected cells")
        
        # Bulk create for efficiency (using validated cells only)
        DetectedCell.objects.bulk_create(valid_detected_cells)
    
    def _mark_failed(self, error_message):
        """Mark analysis as failed with error message"""
        self.analysis.status = 'failed'
        self.analysis.error_message = error_message
        self.analysis.save()


def run_cell_analysis(analysis_id):
    """
    Public function to run cell analysis
    Usage: run_cell_analysis(analysis.id)
    """
    processor = CellAnalysisProcessor(analysis_id)
    return processor.run_analysis()


def get_image_quality_summary(image_path):
    """Get image quality summary for a given image file"""
    try:
        from cellpose.io import imread
        image_array = imread(image_path)
        
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        
        quality_assessment = ImageQualityAssessment.assess_overall_quality(image_array)
        return quality_assessment
    except Exception as e:
        return {
            'overall_score': 0,
            'quality_category': 'error',
            'error': str(e)
        }


def get_analysis_summary(analysis):
    """Get summary statistics for an analysis"""
    if analysis.status != 'completed':
        return None
    
    detected_cells = analysis.detected_cells.all()
    if not detected_cells.exists():
        return None
    
    # Calculate summary statistics (pixels)
    areas = [cell.area for cell in detected_cells]
    perimeters = [cell.perimeter for cell in detected_cells]
    circularities = [cell.circularity for cell in detected_cells]
    eccentricities = [cell.eccentricity for cell in detected_cells]
    major_axes = [cell.major_axis_length for cell in detected_cells]
    minor_axes = [cell.minor_axis_length for cell in detected_cells]
    
    summary = {
        'total_cells': len(areas),
        'scale_available': analysis.cell.scale_set,
        'pixels_per_micron': analysis.cell.pixels_per_micron if analysis.cell.scale_set else None,
        'area_stats': {
            'mean': np.mean(areas),
            'std': np.std(areas),
            'min': np.min(areas),
            'max': np.max(areas),
        },
        'perimeter_stats': {
            'mean': np.mean(perimeters),
            'std': np.std(perimeters),
            'min': np.min(perimeters),
            'max': np.max(perimeters),
        },
        'circularity_stats': {
            'mean': np.mean(circularities),
            'std': np.std(circularities),
            'min': np.min(circularities),
            'max': np.max(circularities),
        },
        'eccentricity_stats': {
            'mean': np.mean(eccentricities),
            'std': np.std(eccentricities),
            'min': np.min(eccentricities),
            'max': np.max(eccentricities),
        },
        'major_axis_stats': {
            'mean': np.mean(major_axes),
            'std': np.std(major_axes),
            'min': np.min(major_axes),
            'max': np.max(major_axes),
        },
        'minor_axis_stats': {
            'mean': np.mean(minor_axes),
            'std': np.std(minor_axes),
            'min': np.min(minor_axes),
            'max': np.max(minor_axes),
        }
    }
    
    # Add physical measurements if scale is available
    if analysis.cell.scale_set:
        areas_microns = [cell.area_microns_sq for cell in detected_cells if cell.area_microns_sq is not None]
        perimeters_microns = [cell.perimeter_microns for cell in detected_cells if cell.perimeter_microns is not None]
        major_axes_microns = [cell.major_axis_length_microns for cell in detected_cells if cell.major_axis_length_microns is not None]
        minor_axes_microns = [cell.minor_axis_length_microns for cell in detected_cells if cell.minor_axis_length_microns is not None]
        
        if areas_microns:
            summary['area_stats_microns'] = {
                'mean': np.mean(areas_microns),
                'std': np.std(areas_microns),
                'min': np.min(areas_microns),
                'max': np.max(areas_microns),
            }
        
        if perimeters_microns:
            summary['perimeter_stats_microns'] = {
                'mean': np.mean(perimeters_microns),
                'std': np.std(perimeters_microns),
                'min': np.min(perimeters_microns),
                'max': np.max(perimeters_microns),
            }
        
        if major_axes_microns:
            summary['major_axis_stats_microns'] = {
                'mean': np.mean(major_axes_microns),
                'std': np.std(major_axes_microns),
                'min': np.min(major_axes_microns),
                'max': np.max(major_axes_microns),
            }
        
        if minor_axes_microns:
            summary['minor_axis_stats_microns'] = {
                'mean': np.mean(minor_axes_microns),
                'std': np.std(minor_axes_microns),
                'min': np.min(minor_axes_microns),
                'max': np.max(minor_axes_microns),
            }
    
    return summary