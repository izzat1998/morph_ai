"""
Morphometric Analysis Module

This is the main entry point for morphometric analysis functionality.
The original monolithic file has been refactored into focused modules.

New modular structure:
- quality_assessment.py: Image quality metrics
- image_preprocessing.py: Image preprocessing operations  
- parameter_optimization.py: Parameter optimization for Cellpose
- segmentation_refinement.py: Post-processing refinement
- utils.py: Utility functions
- exceptions.py: Custom exception classes

Legacy classes and functions are imported here for backward compatibility.
"""

# Standard library imports
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import logging

# Django imports
from django.core.files.base import ContentFile
from django.utils import timezone
from django.conf import settings
from django.utils.translation import gettext as _

# Import simplified modules
from .quality_assessment import ImageQualityAssessment
from .image_preprocessing import ImagePreprocessor
from .segmentation_refinement import SegmentationRefinement
from .utils import run_cell_analysis, get_image_quality_summary, get_analysis_summary
from .cellpose_sam_proper import CellposeSAMSegmenter, is_cellpose_sam_available
from .exceptions import *

# Model imports
from .models import CellAnalysis, DetectedCell

# Configure logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from cellpose import models
    from cellpose.io import imread
    from cellpose import plot, utils
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logger.warning("Cellpose not available - some functionality will be limited")

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
    logger.warning("scikit-image not available - some functionality will be limited")


class CellAnalysisProcessor:
    """
    Main cell analysis processor class.
    
    This class coordinates the complete analysis pipeline including:
    - Image loading and quality assessment
    - Parameter optimization
    - Cellpose segmentation  
    - Post-processing refinement
    - Morphometric feature extraction
    - Visualization generation
    """
    
    def __init__(self, analysis_id):
        """
        Initialize the analysis processor.
        
        Args:
            analysis_id: ID of the CellAnalysis instance to process
        """
        self.analysis = CellAnalysis.objects.get(id=analysis_id)
        self.cell = self.analysis.cell
        logger.info(f"CellAnalysisProcessor initialized for analysis ID: {analysis_id}")
        
    def run_analysis(self):
        """
        Execute the complete analysis pipeline with GPU memory management.
        
        Returns:
            True if analysis completed successfully, False otherwise
        """
        if not CELLPOSE_AVAILABLE or not SKIMAGE_AVAILABLE:
            self._mark_failed("Required dependencies (cellpose/scikit-image) not available")
            return False
            
        try:
            # Update status to processing
            self.analysis.status = 'processing'
            self.analysis.save()
            
            start_time = time.time()
            logger.info(f"Starting analysis for cell: {self.cell.name}")
            
            # Simplified GPU operations - no complex memory management needed
            # Step 1: Load and assess image quality
            image_array = self._load_image()
            
            # Step 2: Run cellpose segmentation
            masks, flows, styles, diameters = self._run_cellpose_segmentation(image_array)
            
            # Step 3: Apply post-processing refinement
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
            
            logger.info(f"Segmentation completed: {original_mask_count} → {final_mask_count} cells after refinement")
            
            # Add ROI metadata if ROI processing was used
            if hasattr(self, '_roi_metadata'):
                self.analysis.quality_metrics['roi_analysis'] = self._roi_metadata
                logger.info(f"ROI metadata preserved: {len(self._roi_metadata.get('roi_regions', []))} regions processed")
            
            # Step 4: Save all comprehensive visualizations
            self._save_all_visualizations(image_array, masks, flows, styles, diameters)
            
            # Step 5: Extract morphometric features
            self._extract_morphometric_features(masks)
            
            # Step 6: Update analysis record
            processing_time = time.time() - start_time
            self.analysis.processing_time = processing_time
            self.analysis.completed_at = timezone.now()
            self.analysis.status = 'completed'
            self.analysis.save()
            
            # Final validation - ensure core visualization is accessible
            try:
                if self.analysis.segmentation_image and self.analysis.segmentation_image.url:
                    logger.info(f"Core visualization accessible at: {self.analysis.segmentation_image.url}")
                else:
                    logger.error("Core visualization is not accessible after analysis completion")
            except Exception as validation_error:
                logger.warning(f"Could not validate visualization accessibility: {str(validation_error)}")
            
            logger.info(f"Analysis completed successfully in {processing_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            self._mark_failed(str(e))
            return False
    
    def _load_image(self):
        """Load and assess image quality."""
        image_path = self.cell.image.path
        
        # Use cellpose's imread for better compatibility
        image_array = imread(image_path)
        
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        
        # Perform quality assessment using our new module
        quality_assessment = ImageQualityAssessment.assess_overall_quality(image_array)
        
        # Store quality metrics in analysis
        self.analysis.quality_metrics = quality_assessment
        
        # Apply GPU-accelerated preprocessing if enabled
        if self.analysis.apply_preprocessing and getattr(settings, 'ENABLE_GPU_PREPROCESSING', False):
            try:
                from .image_preprocessing import GPUImagePreprocessor
                gpu_preprocessor = GPUImagePreprocessor()
                
                # Apply basic preprocessing operations
                preprocessing_start = time.time()
                
                # Apply Gaussian smoothing for noise reduction
                if gpu_preprocessor.cupy_available:
                    logger.info("Applying GPU-accelerated preprocessing")
                    image_array = gpu_preprocessor.gaussian_filter_gpu(image_array, sigma=0.5)
                    
                    # Apply contrast enhancement if image quality is poor
                    if quality_assessment.get('overall_score', 50) < 60:
                        image_array = gpu_preprocessor.histogram_equalization_gpu(image_array)
                        
                    preprocessing_time = time.time() - preprocessing_start
                    logger.info(f"GPU preprocessing completed in {preprocessing_time:.2f}s")
                else:
                    logger.info("GPU preprocessing not available, skipping")
                    
            except Exception as preprocessing_error:
                logger.warning(f"GPU preprocessing failed: {str(preprocessing_error)}")
        
        # Use simple, proven defaults - no optimization needed
        # CellposeSAM works best with standard parameters
        if self.analysis.cellpose_diameter == 0:
            # Keep diameter=0 for auto-detection (CellposeSAM is excellent at this)
            pass
        
        # Ensure we have proven default parameters
        if self.analysis.flow_threshold is None or self.analysis.flow_threshold == 0:
            self.analysis.flow_threshold = 0.4  # Proven optimal default
        if self.analysis.cellprob_threshold is None:
            self.analysis.cellprob_threshold = 0.0  # Proven optimal default
        if not self.analysis.cellpose_model:
            self.analysis.cellpose_model = 'cpsam'  # Most robust model
            
        logger.info(f"Using proven parameters: diameter={self.analysis.cellpose_diameter}, "
                   f"flow={self.analysis.flow_threshold}, cellprob={self.analysis.cellprob_threshold}, "
                   f"model={self.analysis.cellpose_model}")
        
        self.analysis.save()
        return image_array
        
    def _extract_roi_regions(self, image_array):
        """
        Extract ROI regions from the full image array.
        
        Returns:
            list: List of (roi_data, cropped_image) tuples
        """
        if not self.analysis.use_roi or not self.analysis.roi_regions:
            return None
            
        # Validate ROI regions data
        if not isinstance(self.analysis.roi_regions, list):
            logger.warning("ROI regions data is not a list, disabling ROI processing")
            return None
            
        if len(self.analysis.roi_regions) == 0:
            logger.warning("No ROI regions defined, disabling ROI processing")
            return None
            
        roi_extracts = []
        
        for roi in self.analysis.roi_regions:
            # Validate ROI data structure
            if not isinstance(roi, dict):
                logger.warning(f"Invalid ROI data structure: {type(roi)}, skipping")
                continue
                
            # Extract coordinates and ensure they're within image bounds
            try:
                x = max(0, int(roi.get('x', 0)))
                y = max(0, int(roi.get('y', 0)))
                width = max(1, int(roi.get('width', 1)))
                height = max(1, int(roi.get('height', 1)))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid ROI coordinates: {roi}, error: {e}")
                continue
            
            # Ensure ROI doesn't exceed image bounds
            x = min(x, image_array.shape[1] - 1)
            y = min(y, image_array.shape[0] - 1)
            width = min(width, image_array.shape[1] - x)
            height = min(height, image_array.shape[0] - y)
            
            # Skip invalid ROIs
            if width <= 0 or height <= 0:
                logger.warning(f"Skipping invalid ROI: x={x}, y={y}, w={width}, h={height}")
                continue
                
            # Extract the ROI region
            if len(image_array.shape) == 2:
                cropped = image_array[y:y+height, x:x+width]
            else:
                cropped = image_array[y:y+height, x:x+width, :]
                
            roi_data = {
                'id': roi.get('id', len(roi_extracts)),
                'x': x, 'y': y, 'width': width, 'height': height,
                'original_coords': roi
            }
            
            roi_extracts.append((roi_data, cropped))
            logger.info(f"Extracted ROI {roi_data['id']}: {width}x{height} at ({x}, {y})")
            
        # Check for overlapping ROI regions and warn user
        if len(roi_extracts) > 1:
            overlap_count = 0
            for i in range(len(roi_extracts)):
                for j in range(i + 1, len(roi_extracts)):
                    roi1_data, _ = roi_extracts[i]
                    roi2_data, _ = roi_extracts[j]
                    
                    region1 = (roi1_data['x'], roi1_data['y'], 
                              roi1_data['x'] + roi1_data['width'], 
                              roi1_data['y'] + roi1_data['height'])
                    region2 = (roi2_data['x'], roi2_data['y'], 
                              roi2_data['x'] + roi2_data['width'], 
                              roi2_data['y'] + roi2_data['height'])
                    
                    if self._roi_regions_overlap(region1, region2):
                        overlap_count += 1
                        logger.info(f"ROI overlap detected: ROI {roi1_data['id']} and ROI {roi2_data['id']} - "
                                  f"deduplication will be applied")
            
            if overlap_count > 0:
                logger.info(f"Total ROI overlaps detected: {overlap_count} - using intelligent cell merging")
            else:
                logger.info(f"No ROI overlaps detected in {len(roi_extracts)} regions - standard processing")
            
        return roi_extracts if roi_extracts else None

    def _merge_overlapping_cells(self, combined_mask, new_mask, roi_data, overlap_threshold=0.5):
        """
        Merge overlapping cells from new ROI mask into existing combined mask.
        
        Args:
            combined_mask: Existing combined mask
            new_mask: New ROI mask to merge
            roi_data: ROI coordinates and metadata
            overlap_threshold: Minimum overlap ratio to consider cells as duplicates
            
        Returns:
            Updated combined mask with merged cells
        """
        from scipy import ndimage
        from skimage import measure
        
        x, y = roi_data['x'], roi_data['y']
        height, width = new_mask.shape[:2]
        
        # Get region where new mask will be placed
        roi_region = combined_mask[y:y+height, x:x+width]
        
        # Find existing cells in the ROI region
        existing_labels = np.unique(roi_region)
        existing_labels = existing_labels[existing_labels > 0]
        
        # Find new cells in the new mask
        new_labels = np.unique(new_mask)
        new_labels = new_labels[new_labels > 0]
        
        if len(existing_labels) == 0:
            # No existing cells in this region, just place the new mask
            combined_mask[y:y+height, x:x+width] = np.maximum(roi_region, new_mask)
            logger.debug(f"ROI {roi_data['id']}: No existing cells, placed {len(new_labels)} new cells")
            return combined_mask
        
        # Track which new cells have been merged
        merged_new_cells = set()
        merge_count = 0
        
        # Check each new cell against existing cells for overlap
        for new_label in new_labels:
            new_cell_mask = (new_mask == new_label)
            best_overlap = 0
            best_existing_label = None
            
            # Check overlap with each existing cell
            for existing_label in existing_labels:
                existing_cell_mask = (roi_region == existing_label)
                
                # Calculate overlap
                intersection = np.logical_and(new_cell_mask, existing_cell_mask)
                union = np.logical_or(new_cell_mask, existing_cell_mask)
                
                if union.sum() > 0:
                    overlap_ratio = intersection.sum() / union.sum()
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_existing_label = existing_label
            
            # If significant overlap found, merge the cells
            if best_overlap > overlap_threshold and best_existing_label is not None:
                # Merge by assigning new cell pixels to existing cell ID
                merge_pixels = new_cell_mask
                combined_mask[y:y+height, x:x+width][merge_pixels] = best_existing_label
                merged_new_cells.add(new_label)
                merge_count += 1
                logger.debug(f"ROI {roi_data['id']}: Merged cell {new_label} with existing cell {best_existing_label} (overlap: {best_overlap:.2f})")
        
        # Place remaining new cells (those that weren't merged)
        remaining_labels = [label for label in new_labels if label not in merged_new_cells]
        if remaining_labels:
            # Find the next available cell ID
            max_existing_id = combined_mask.max()
            current_new_id = max_existing_id + 1
            
            for old_label in remaining_labels:
                new_cell_mask = (new_mask == old_label)
                combined_mask[y:y+height, x:x+width][new_cell_mask] = current_new_id
                current_new_id += 1
            
            logger.debug(f"ROI {roi_data['id']}: Added {len(remaining_labels)} new cells, merged {merge_count} duplicates")
        else:
            logger.debug(f"ROI {roi_data['id']}: All {len(new_labels)} cells were merged with existing cells")
        
        return combined_mask

    def _roi_regions_overlap(self, region1, region2):
        """
        Check if two ROI regions overlap.
        
        Args:
            region1: (x1, y1, x2, y2) bounds of first region
            region2: (x1, y1, x2, y2) bounds of second region
            
        Returns:
            bool: True if regions overlap
        """
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Check if there's no overlap (easier to check)
        if (x2_1 <= x1_2 or x2_2 <= x1_1 or 
            y2_1 <= y1_2 or y2_2 <= y1_1):
            return False
            
        return True

    def _combine_roi_masks(self, roi_results, full_image_shape):
        """
        Combine masks from multiple ROI regions into a single full-image mask.
        
        Args:
            roi_results: List of (roi_data, masks, flows, styles, diameters) tuples
            full_image_shape: Shape of the original full image
            
        Returns:
            Combined masks, flows, styles, diameters
        """
        # Create empty full-size mask
        if len(full_image_shape) == 2:
            combined_mask = np.zeros(full_image_shape, dtype=np.uint16)
        else:
            combined_mask = np.zeros(full_image_shape[:2], dtype=np.uint16)
            
        # Reconstruct full flow field from ROI data
        combined_flows = self._reconstruct_full_flow_field(roi_results, full_image_shape)
        combined_styles = []
        combined_diameters = []
        
        total_detections = 0
        overlapping_regions = []
        
        for roi_idx, (roi_data, masks, flows, styles, diameters) in enumerate(roi_results):
            if masks is None:
                continue
                
            # Get ROI coordinates
            x, y = roi_data['x'], roi_data['y']
            width, height = roi_data['width'], roi_data['height']
            
            # Check for overlaps with previously processed ROIs
            current_roi_bounds = (x, y, x + width, y + height)
            overlaps_detected = False
            
            for prev_roi_bounds in overlapping_regions:
                if self._roi_regions_overlap(current_roi_bounds, prev_roi_bounds):
                    overlaps_detected = True
                    break
            
            if overlaps_detected:
                logger.info(f"ROI {roi_data['id']}: Overlap detected, using intelligent merging")
            
            overlapping_regions.append(current_roi_bounds)
            
            # Prepare ROI mask for merging
            roi_mask = masks.copy()
            if roi_mask.max() > 0:
                unique_labels = np.unique(roi_mask)
                unique_labels = unique_labels[unique_labels > 0]  # Exclude background
                
                # Use intelligent merging instead of simple np.maximum
                try:
                    combined_mask = self._merge_overlapping_cells(
                        combined_mask, roi_mask, roi_data, 
                        overlap_threshold=0.5
                    )
                    
                    # Count actual detections in final mask (to avoid double counting)
                    roi_region = combined_mask[y:y+height, x:x+width]
                    actual_detections = len(np.unique(roi_region)) - 1  # Exclude background
                    
                    logger.info(f"ROI {roi_data['id']}: Processed {len(unique_labels)} detected cells, "
                              f"resulted in {actual_detections} final cells in region")
                    
                except ValueError as e:
                    logger.warning(f"Failed to merge ROI mask for region {roi_data['id']}: {e}")
                    continue
                    
            # Collect metadata
            if styles is not None:
                combined_styles.extend(styles if isinstance(styles, list) else [styles])
            if diameters is not None:
                combined_diameters.extend(diameters if isinstance(diameters, list) else [diameters])
        
        # Calculate final total detections from the combined mask
        total_detections = len(np.unique(combined_mask)) - 1  # Exclude background
                
        logger.info(f"Combined {len(roi_results)} ROI regions with {total_detections} total detections")
        
        # Return in the same format as single-image segmentation
        return combined_mask, combined_flows, combined_styles or None, combined_diameters or None

    def _reconstruct_full_flow_field(self, roi_results, full_image_shape):
        """
        Reconstruct full-image flow field from ROI flow data.
        
        Args:
            roi_results: List of (roi_data, masks, flows, styles, diameters) tuples
            full_image_shape: Shape of the original full image
            
        Returns:
            Combined flow data in same format as Cellpose output
        """
        logger.info("Reconstructing full-image flow field from ROI data")
        
        # Create full-size flow array - same format as Cellpose output
        if len(full_image_shape) == 2:
            h, w = full_image_shape
        else:
            h, w = full_image_shape[:2]
            
        # Initialize flow arrays matching Cellpose format
        # flows[0] contains [prob, dY, dX] where prob is cell probability
        # flows[1] contains XY flow data
        full_flow_prob = np.zeros((h, w), dtype=np.float32)
        full_flow_xy = np.zeros((h, w, 2), dtype=np.float32)
        
        overlap_count = np.zeros((h, w), dtype=np.int32)  # Track overlapping regions
        
        valid_roi_count = 0
        
        for roi_idx, (roi_data, masks, flows, styles, diameters) in enumerate(roi_results):
            if flows is None or len(flows) < 2:
                logger.debug(f"ROI {roi_data.get('id', roi_idx)}: No flow data available")
                continue
                
            try:
                # Get ROI coordinates
                x = max(0, int(roi_data['x']))
                y = max(0, int(roi_data['y']))
                roi_width = int(roi_data['width'])
                roi_height = int(roi_data['height'])
                
                # Ensure ROI doesn't exceed image bounds
                x = min(x, w - 1)
                y = min(y, h - 1)
                roi_width = min(roi_width, w - x)
                roi_height = min(roi_height, h - y)
                
                if roi_width <= 0 or roi_height <= 0:
                    logger.warning(f"ROI {roi_data.get('id', roi_idx)}: Invalid dimensions, skipping")
                    continue
                
                # Extract flow data from ROI
                roi_flow_data = flows[0]  # [prob, dY, dX] format
                roi_xy_flows = flows[1] if len(flows) > 1 else None
                
                # Handle different flow data formats
                if isinstance(roi_flow_data, list) and len(roi_flow_data) >= 3:
                    # Standard format: [prob, dY, dX]
                    roi_prob = roi_flow_data[0]
                    if roi_xy_flows is not None and roi_xy_flows.shape[:2] == (roi_height, roi_width):
                        # Place flow data in full image coordinates
                        full_flow_xy[y:y+roi_height, x:x+roi_width] += roi_xy_flows
                        full_flow_prob[y:y+roi_height, x:x+roi_width] += roi_prob
                        overlap_count[y:y+roi_height, x:x+roi_width] += 1
                        valid_roi_count += 1
                        
                        logger.debug(f"ROI {roi_data.get('id', roi_idx)}: Added flow data "
                                   f"at ({x},{y}) size ({roi_width}x{roi_height})")
                
            except Exception as roi_error:
                logger.warning(f"ROI {roi_data.get('id', roi_idx)}: Flow reconstruction failed: {roi_error}")
                continue
        
        # Average overlapping regions
        overlap_mask = overlap_count > 1
        if np.any(overlap_mask):
            full_flow_xy[overlap_mask] /= overlap_count[overlap_mask, np.newaxis]
            full_flow_prob[overlap_mask] /= overlap_count[overlap_mask]
            overlap_regions = np.sum(overlap_mask)
            logger.info(f"Averaged {overlap_regions} overlapping pixels from multiple ROIs")
        
        # Construct flow data in Cellpose format
        # flows[0] = [prob, dY, dX] 
        # flows[1] = XY flow field
        if valid_roi_count > 0:
            dY = full_flow_xy[:, :, 0]  # Y component
            dX = full_flow_xy[:, :, 1]  # X component
            
            combined_flows = [
                [full_flow_prob, dY, dX],  # Standard Cellpose format
                full_flow_xy                # XY flow field for visualization
            ]
            
            logger.info(f"Successfully reconstructed flow field from {valid_roi_count} ROI regions")
            return combined_flows
        else:
            logger.warning("No valid flow data found in any ROI region")
            return None

    def _run_cellpose_segmentation(self, image_array):
        """Run Cellpose segmentation with GPU acceleration and fallback."""
        try:
            # Simple GPU detection
            use_gpu = getattr(settings, 'CELLPOSE_USE_GPU', False)
            
            # Check if GPU is actually available
            if use_gpu:
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except ImportError:
                    use_gpu = False
            
            # Enhanced GPU logging
            if use_gpu:
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"🚀 GPU ACCELERATION ENABLED")
                    logger.info(f"   Device: {gpu_name}")
                    logger.info(f"   Memory: {gpu_memory:.1f}GB total, {memory_allocated:.1f}MB currently allocated")
                except Exception as e:
                    logger.warning(f"GPU info retrieval failed: {str(e)}")
            else:
                logger.info("⚠️  Using CPU (GPU disabled or unavailable)")
            
            # Initialize CellposeSAM segmenter (proper implementation)
            try:
                logger.info("Initializing CellposeSAM segmenter")
                
                # Check if SAM is available
                if not is_cellpose_sam_available():
                    raise RuntimeError(
                        "CellposeSAM is not available. Please ensure Cellpose 4.0+ is installed: "
                        "pip install cellpose[gui]>=4.0.0"
                    )
                
                # Initialize proper CellposeSAM segmenter
                segmenter = CellposeSAMSegmenter(
                    gpu=use_gpu
                )
                
                model_info = segmenter.get_model_info()
                logger.info(f"CellposeSAM segmenter initialized successfully (GPU: {use_gpu})")
                logger.info(f"Model info: {model_info}")
                    
            except Exception as model_error:
                logger.error(f"CellposeSAM initialization failed: {str(model_error)}")
                raise RuntimeError(
                    f"Failed to initialize CellposeSAM: {str(model_error)}. "
                    f"Please ensure Cellpose 4.0+ is properly installed."
                )
            
            # Determine channels based on image
            if len(image_array.shape) == 2:
                # Grayscale image
                channels = [0, 0]
            elif len(image_array.shape) == 3:
                if image_array.shape[2] == 1:
                    # Single channel
                    channels = [0, 0]
                elif image_array.shape[2] >= 3:
                    # RGB or more channels - use grayscale for segmentation
                    channels = [0, 0]
                else:
                    channels = [0, 0]
            else:
                channels = [0, 0]
            
            # Check if ROI processing is enabled
            roi_extracts = self._extract_roi_regions(image_array)
            
            if roi_extracts:
                # ROI-based segmentation
                logger.info(f"Running ROI-based CellposeSAM segmentation on {len(roi_extracts)} regions")
                roi_results = []
                
                for roi_data, roi_image in roi_extracts:
                    logger.info(f"Processing ROI {roi_data['id']}: {roi_data['width']}x{roi_data['height']}")
                    
                    try:
                        # Run segmentation on this ROI
                        roi_result = segmenter.segment(
                            image_array=roi_image,
                            diameter=self.analysis.cellpose_diameter if self.analysis.cellpose_diameter > 0 else None,
                            flow_threshold=self.analysis.flow_threshold,
                            cellprob_threshold=self.analysis.cellprob_threshold,
                            do_3D=False
                        )
                        
                        # Handle different return formats
                        if len(roi_result) == 3:
                            roi_masks, roi_flows, roi_prob = roi_result
                            roi_styles = None
                            roi_diameters = [self.analysis.cellpose_diameter]
                        elif len(roi_result) == 4:
                            roi_masks, roi_flows, roi_styles, roi_diameters = roi_result
                        else:
                            roi_masks = roi_result[0]
                            roi_flows = roi_result[1] if len(roi_result) > 1 else None
                            roi_styles = None
                            roi_diameters = [self.analysis.cellpose_diameter]
                        
                        roi_detections = len(np.unique(roi_masks)) - 1
                        logger.info(f"ROI {roi_data['id']}: {roi_detections} cells detected")
                        
                        roi_results.append((roi_data, roi_masks, roi_flows, roi_styles, roi_diameters))
                        
                    except Exception as roi_error:
                        logger.warning(f"ROI {roi_data['id']} segmentation failed: {str(roi_error)}")
                        # Continue with other ROIs
                        continue
                
                if roi_results:
                    # Combine all ROI results into single full-image masks
                    masks, flows, styles, diameters = self._combine_roi_masks(roi_results, image_array.shape)
                    
                    # Store ROI metadata for quality metrics
                    self._roi_metadata = {
                        'roi_regions': [
                            {
                                'id': roi_data['id'],
                                'coordinates': {'x': roi_data['x'], 'y': roi_data['y'], 
                                              'width': roi_data['width'], 'height': roi_data['height']},
                                'cells_detected': len(np.unique(roi_masks)) - 1,
                                'area_coverage': roi_data['width'] * roi_data['height'],
                                'flow_data_available': roi_flows is not None
                            }
                            for (roi_data, roi_masks, roi_flows, roi_styles, roi_diameters) in roi_results
                        ],
                        'total_rois': len(roi_results),
                        'flow_reconstruction': flows is not None,
                        'analysis_type': 'roi_enhanced'
                    }
                    
                    logger.info("ROI-based segmentation completed, masks combined")
                else:
                    logger.warning("All ROI segmentations failed, falling back to full image")
                    # Fallback to full image segmentation
                    result = segmenter.segment(
                        image_array=image_array,
                        diameter=self.analysis.cellpose_diameter if self.analysis.cellpose_diameter > 0 else None,
                        flow_threshold=self.analysis.flow_threshold,
                        cellprob_threshold=self.analysis.cellprob_threshold,
                        do_3D=False
                    )
                    
                    if len(result) == 3:
                        masks, flows, prob = result
                        styles = None
                        diameters = [self.analysis.cellpose_diameter]
                    elif len(result) == 4:
                        masks, flows, styles, diameters = result
                    else:
                        masks = result[0]
                        flows = result[1] if len(result) > 1 else None
                        styles = None
                        diameters = [self.analysis.cellpose_diameter]
            else:
                # Standard full-image segmentation
                logger.info("Running standard CellposeSAM segmentation on full image")
                result = segmenter.segment(
                    image_array=image_array,
                    diameter=self.analysis.cellpose_diameter if self.analysis.cellpose_diameter > 0 else None,
                    flow_threshold=self.analysis.flow_threshold,
                    cellprob_threshold=self.analysis.cellprob_threshold,
                    do_3D=False
                )
                
                # Handle different return formats between Cellpose versions
                if len(result) == 3:
                    # v4.0.4+ format: (masks, flows, prob)
                    masks, flows, prob = result
                    styles = None  # Not returned in v4.0.4+
                    diameters = [self.analysis.cellpose_diameter]  # Use configured diameter
                elif len(result) == 4:
                    # Legacy format: (masks, flows, styles, diameters)
                    masks, flows, styles, diameters = result
                else:
                    # Fallback
                    masks = result[0]
                    flows = result[1] if len(result) > 1 else None
                    styles = None
                    diameters = [self.analysis.cellpose_diameter]
            
            # Log completion with performance info
            num_detections = len(np.unique(masks)) - 1
            backend_used = "GPU" if use_gpu else "CPU"
            
            # Enhanced completion logging with GPU memory info
            if use_gpu:
                try:
                    import torch
                    final_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"✅ Cellpose-SAM segmentation completed: {num_detections} cells detected")
                    logger.info(f"   Backend: {backend_used} | GPU Memory: {final_memory:.1f}MB")
                except Exception:
                    logger.info(f"✅ Cellpose-SAM segmentation completed: {num_detections} detections using {backend_used}")
            else:
                logger.info(f"✅ Cellpose-SAM segmentation completed: {num_detections} detections using {backend_used}")
            
            # Simple GPU cleanup
            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("GPU memory cleaned up after segmentation")
                except ImportError:
                    pass
            
            return masks, flows, styles, diameters
            
        except Exception as e:
            logger.error(f"Cellpose-SAM segmentation failed: {str(e)}")
            # Simple GPU cleanup on error
            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            raise SegmentationError(f"Cellpose-SAM segmentation failed: {str(e)}")
    
    def _save_all_visualizations(self, original_image, masks, flows=None, styles=None, diameters=None):
        """Save all 4 comprehensive visualization pages"""
        try:
            logger.info("Starting comprehensive visualization generation")
            
            # Page 1: Core Pipeline (6 panels) - CRITICAL
            self._save_core_pipeline_visualization(original_image, masks, flows, styles, diameters)
            
            # Validate core pipeline was saved
            if not self.analysis.segmentation_image:
                logger.error("CRITICAL: Core pipeline visualization was not saved properly")
                raise RuntimeError("Core pipeline visualization failed - this is required for analysis completion")
            else:
                logger.info("Core pipeline visualization validated successfully")
            
            # Only create advanced visualizations if we have flow data
            if flows is not None:
                logger.debug("Flow data available, generating advanced visualizations")
                try:
                    # Page 2: Advanced Flow Analysis
                    self._save_flow_analysis_visualization(original_image, masks, flows, styles, diameters)
                except Exception as flow_viz_error:
                    logger.warning(f"Flow analysis visualization failed: {str(flow_viz_error)}")
                
                try:
                    # Page 3: Style & Quality Analysis  
                    self._save_style_quality_visualization(original_image, masks, flows, styles, diameters)
                except Exception as style_viz_error:
                    logger.warning(f"Style & quality visualization failed: {str(style_viz_error)}")
                
                try:
                    # Page 4: Edge & Boundary Analysis
                    self._save_edge_boundary_visualization(original_image, masks, flows, styles, diameters)
                except Exception as edge_viz_error:
                    logger.warning(f"Edge & boundary visualization failed: {str(edge_viz_error)}")
            else:
                logger.info("No flow data available, skipping advanced visualizations")
            
            logger.info("All visualizations processing completed")
            
        except Exception as e:
            logger.error(f"Comprehensive visualization generation failed: {str(e)}")
            # Don't fail the entire analysis for visualization errors unless it's the core pipeline
            if "Core pipeline visualization failed" in str(e):
                raise  # Re-raise critical core pipeline errors
    
    def _save_core_pipeline_visualization(self, original_image, masks, flows=None, styles=None, diameters=None):
        """Page 1: Core Pipeline - Create and save comprehensive visualization of Cellpose pipeline results"""
        try:
            logger.info("Starting core pipeline visualization generation")
            
            # Input validation
            if original_image is None or original_image.size == 0:
                raise ValueError("Original image is None or empty")
            if masks is None or masks.size == 0:
                raise ValueError("Masks array is None or empty")
            
            logger.debug(f"Input shapes - image: {original_image.shape}, masks: {masks.shape}")
            
            unique_masks = np.unique(masks)[1:]  # Skip background
            logger.debug(f"Found {len(unique_masks)} unique cell masks")
            
            # Determine number of panels based on available data
            if flows is not None:
                logger.debug("Generating full 6-panel visualization with flow data")
                # Full pipeline with intermediate steps
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
            else:
                logger.debug("Generating simplified 2-panel visualization (no flow data)")
                # Simple version without flows
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            if fig is None or axes is None:
                raise RuntimeError("Failed to create matplotlib figure or axes")
            
            logger.debug(f"Created figure with {len(axes)} panels")
            ax_idx = 0
            
            # Panel 1: Original Image
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
                    logger.debug("Extracting cell outlines using Cellpose utils")
                    outlines = utils.outlines_list(masks)
                    logger.debug(f"Extracted {len(outlines)} outlines")
                    for i, outline in enumerate(outlines):
                        if outline.shape[0] > 0:  # Check outline is not empty
                            axes[ax_idx].plot(outline[:, 0], outline[:, 1], linewidth=1.5, color='cyan')
                        else:
                            logger.warning(f"Empty outline found for cell {i+1}")
                except Exception as outline_error:
                    logger.warning(f"Cellpose outline extraction failed: {str(outline_error)}, using fallback contour method")
                    # Fallback: draw contours manually
                    try:
                        from skimage import measure
                        contours = measure.find_contours(masks > 0, 0.5)
                        logger.debug(f"Fallback: extracted {len(contours)} contours")
                        for contour in contours:
                            if contour.shape[0] > 0:  # Check contour is not empty
                                axes[ax_idx].plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan')
                    except Exception as contour_error:
                        logger.error(f"Both outline extraction methods failed: {str(contour_error)}")
                        # Continue without outlines
                
                axes[ax_idx].set_title('2. Predicted Outlines', fontsize=12, fontweight='bold')
                axes[ax_idx].axis('off')
                ax_idx += 1
                
                # Panel 3: Flow Fields (with enhanced error handling)
                try:
                    logger.debug("Processing flow field visualization")
                    if flows is not None and len(flows) > 0 and len(flows[0]) > 1:
                        xy_flows = flows[0][1]  # XY flows at each pixel
                        logger.debug(f"Flow data shape: {xy_flows.shape}")
                        
                        # Handle different flow data formats from Cellpose
                        if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                            logger.debug("Converting 3D flow data to circular RGB representation")
                            # dx_to_circ expects [dY, dX] format (note: Y first, then X)
                            flow_data = np.stack([xy_flows[:,:,1], xy_flows[:,:,0]], axis=-1)
                            
                            # Validate flow data before processing
                            if np.isfinite(flow_data).all():
                                flow_rgb = plot.dx_to_circ(flow_data)
                                if flow_rgb is not None and flow_rgb.size > 0:
                                    axes[ax_idx].imshow(flow_rgb)
                                    logger.debug("Flow field visualization completed successfully")
                                else:
                                    raise ValueError("dx_to_circ returned empty result")
                            else:
                                raise ValueError("Flow data contains non-finite values")
                        elif len(xy_flows.shape) == 2 and xy_flows.shape[1] >= 2:
                            logger.debug(f"Handling 2D flow data shape: {xy_flows.shape}")
                            # Some Cellpose versions return 2D flow data - check for valid reshape
                            h, w = masks.shape
                            expected_size = h * w
                            actual_size = xy_flows.shape[0]
                            
                            if actual_size == expected_size and xy_flows.shape[1] >= 2:
                                # Reshape to 3D format - valid case
                                xy_flows_3d = xy_flows.reshape(h, w, xy_flows.shape[1])
                                flow_data = np.stack([xy_flows_3d[:,:,1], xy_flows_3d[:,:,0]], axis=-1)
                                
                                if np.isfinite(flow_data).all():
                                    flow_rgb = plot.dx_to_circ(flow_data)
                                    if flow_rgb is not None and flow_rgb.size > 0:
                                        axes[ax_idx].imshow(flow_rgb)
                                        logger.debug("Flow field visualization completed successfully (reshaped)")
                                    else:
                                        raise ValueError("dx_to_circ returned empty result after reshape")
                                else:
                                    raise ValueError("Reshaped flow data contains non-finite values")
                            else:
                                # Invalid dimensions - log warning and skip flow visualization
                                logger.warning(f"Flow data dimensions mismatch: flow shape {xy_flows.shape} (size={actual_size}) cannot be reshaped for mask shape {masks.shape} (size={expected_size})")
                                # Skip flow visualization instead of raising error
                                axes[ax_idx].text(0.5, 0.5, 'Flow data\ndimensions mismatch', 
                                                ha='center', va='center', transform=axes[ax_idx].transAxes,
                                                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                        else:
                            raise ValueError(f"Unsupported flow shape: {xy_flows.shape}, expected 3D with >=2 channels or reshapeable 2D")
                    else:
                        raise ValueError("Flow data is None or incomplete")
                        
                except Exception as flow_error:
                    logger.warning(f"Flow field visualization failed: {str(flow_error)}, using fallback")
                    # Robust fallback visualization
                    try:
                        if len(original_image.shape) == 3:
                            fallback_base = np.mean(original_image, axis=2)
                        else:
                            fallback_base = original_image
                        axes[ax_idx].imshow(fallback_base, cmap='gray', alpha=0.3)
                        axes[ax_idx].text(0.5, 0.5, 'Flow visualization unavailable', 
                                        transform=axes[ax_idx].transAxes, ha='center', va='center',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception as fallback_error:
                        logger.error(f"Even fallback flow visualization failed: {str(fallback_error)}")
                
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
            
            # Panel: Final Masks (always show this)
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
            
            title_idx = 5 if flows is not None else 2
            axes[ax_idx].set_title(f'{title_idx}. Final Masks ({len(unique_masks)} cells)', fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            if flows is not None:
                # Panel 6: Enhanced Morphometric Visualization
                if len(original_image.shape) == 3:
                    base_img = np.mean(original_image, axis=2)
                else:
                    base_img = original_image
                axes[ax_idx].imshow(base_img, cmap='gray', alpha=0.6)
                
                # Get morphometric data from regionprops and DetectedCell model
                from skimage import measure
                props = measure.regionprops(masks)
                
                # Try to get DetectedCell data if analysis has been saved
                morphometric_data = {}
                try:
                    # Check if we have detected cells data for this analysis
                    detected_cells = self.analysis.detected_cells.all()
                    if detected_cells.exists():
                        for cell in detected_cells:
                            morphometric_data[cell.cell_id] = {
                                'circularity': cell.circularity,
                                'area': cell.area,
                                'solidity': cell.solidity,
                                'eccentricity': cell.eccentricity
                            }
                        logger.debug(f"Using stored morphometric data for {len(morphometric_data)} cells")
                    else:
                        logger.debug("No stored DetectedCell data available, using regionprops only")
                except Exception as e:
                    logger.debug(f"Could not access DetectedCell data: {str(e)}, using regionprops")
                
                # Calculate morphometric properties for visualization
                cell_areas = []
                cell_circularities = []
                cell_positions = []
                cell_labels = []
                
                for prop in props:
                    if prop.label > 0:
                        # Get position
                        y, x = prop.centroid
                        cell_positions.append((x, y))
                        cell_labels.append(prop.label)
                        
                        # Use stored data if available, otherwise calculate from regionprops
                        if prop.label in morphometric_data:
                            circularity = morphometric_data[prop.label]['circularity']
                            area = morphometric_data[prop.label]['area']
                        else:
                            # Calculate circularity: 4π×area/perimeter²
                            if prop.perimeter > 0:
                                circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
                            else:
                                circularity = 0
                            area = prop.area
                        
                        cell_areas.append(area)
                        cell_circularities.append(circularity)
                
                if cell_areas:
                    # Normalize areas for size coding (10-100 pixel range for markers)
                    areas_array = np.array(cell_areas)
                    if len(areas_array) > 1 and np.std(areas_array) > 0:
                        normalized_sizes = 10 + 90 * (areas_array - np.min(areas_array)) / (np.max(areas_array) - np.min(areas_array))
                    else:
                        normalized_sizes = np.full(len(areas_array), 30)  # Default size
                    
                    # Color coding based on circularity (blue=round, red=elongated)
                    circularities_array = np.array(cell_circularities)
                    # Clamp circularity to reasonable range for color mapping
                    circularities_clamped = np.clip(circularities_array, 0, 1)
                    
                    # Create colormap: high circularity (round) = blue, low circularity (elongated) = red
                    from matplotlib.colors import LinearSegmentedColormap
                    colors_list = ['red', 'orange', 'yellow', 'lightblue', 'blue']
                    circularity_cmap = LinearSegmentedColormap.from_list('circularity', colors_list, N=256)
                    
                    # Plot enhanced cell visualization
                    scatter = axes[ax_idx].scatter(
                        [pos[0] for pos in cell_positions],
                        [pos[1] for pos in cell_positions],
                        s=normalized_sizes,
                        c=circularities_clamped,
                        cmap=circularity_cmap,
                        alpha=0.8,
                        edgecolors='white',
                        linewidth=1.5,
                        vmin=0,
                        vmax=1
                    )
                    
                    # Smart labeling: only show labels for largest 20% of cells or cells with extreme properties
                    if len(cell_areas) > 5:  # Only apply smart labeling if we have enough cells
                        area_threshold = np.percentile(areas_array, 80)  # Top 20% by area
                        circularity_extremes = (circularities_array < 0.3) | (circularities_array > 0.8)  # Very elongated or very round
                        
                        for i, (pos, label, area, circularity) in enumerate(zip(cell_positions, cell_labels, cell_areas, cell_circularities)):
                            # Show label if cell is large or has extreme shape
                            if area >= area_threshold or circularity_extremes[i]:
                                # Use white text with black outline for better visibility
                                axes[ax_idx].text(pos[0]+5, pos[1]-5, str(label), 
                                                fontsize=8, color='white', fontweight='bold',
                                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    else:
                        # For small numbers of cells, show all labels
                        for pos, label in zip(cell_positions, cell_labels):
                            axes[ax_idx].text(pos[0]+5, pos[1]-5, str(label), 
                                            fontsize=8, color='white', fontweight='bold',
                                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    
                    # Add mini colorbar for circularity
                    from matplotlib.colorbar import ColorbarBase
                    from matplotlib.ticker import FuncFormatter
                    
                    # Create inset axes for colorbar (top-right corner)
                    colorbar_ax = axes[ax_idx].inset_axes([0.75, 0.85, 0.2, 0.03])
                    cb = plt.colorbar(scatter, cax=colorbar_ax, orientation='horizontal')
                    cb.set_label('Circularity', fontsize=8, color='white', fontweight='bold')
                    cb.ax.tick_params(labelsize=7, colors='white')
                    
                    # Add size legend (bottom-right corner)
                    legend_ax = axes[ax_idx].inset_axes([0.75, 0.05, 0.2, 0.15])
                    legend_ax.set_xlim(0, 1)
                    legend_ax.set_ylim(0, 1)
                    legend_ax.axis('off')
                    
                    # Size legend circles
                    if len(areas_array) > 1:
                        min_area, max_area = np.min(areas_array), np.max(areas_array)
                        
                        # Show 3 size examples
                        sizes_for_legend = [10, 50, 90]  # Small, medium, large marker sizes
                        areas_for_legend = [min_area, np.mean(areas_array), max_area]
                        
                        for i, (size, area) in enumerate(zip(sizes_for_legend, areas_for_legend)):
                            y_pos = 0.8 - i * 0.25
                            legend_ax.scatter(0.2, y_pos, s=size, c='gray', alpha=0.7, edgecolors='white')
                            legend_ax.text(0.4, y_pos, f'{int(area)} px²', fontsize=7, color='white', 
                                         verticalalignment='center', fontweight='bold')
                    
                    # Add title for size legend
                    legend_ax.text(0.1, 0.95, 'Size = Area', fontsize=8, color='white', fontweight='bold')
                    
                    logger.info(f"Enhanced morphometric visualization: {len(cell_areas)} cells displayed")
                    
                else:
                    # Fallback if no cells detected
                    axes[ax_idx].text(0.5, 0.5, 'No cells detected', 
                                    transform=axes[ax_idx].transAxes, ha='center', va='center',
                                    fontsize=12, color='red', fontweight='bold')
                
                axes[ax_idx].set_title('6. Morphometric Analysis\n(Size=Area, Color=Circularity)', 
                                     fontsize=12, fontweight='bold')
                axes[ax_idx].axis('off')
            
            plt.tight_layout()
            logger.debug("Applied tight layout to figure")
            
            # Save visualization with enhanced error handling
            buffer = None
            try:
                logger.debug("Creating BytesIO buffer for figure")
                buffer = BytesIO()
                
                logger.debug("Saving figure to buffer")
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buffer.seek(0)
                
                # Validate buffer contents
                buffer_size = len(buffer.getvalue())
                if buffer_size == 0:
                    raise ValueError("Generated image buffer is empty")
                
                logger.debug(f"Figure saved to buffer successfully ({buffer_size} bytes)")
                
                # Save to analysis model
                filename = f'analysis_{self.analysis.id}_core_pipeline.png'
                logger.debug(f"Saving to model field with filename: {filename}")
                
                self.analysis.segmentation_image.save(
                    filename,
                    ContentFile(buffer.getvalue()),
                    save=False  # Don't save the analysis model yet
                )
                
                # Verify the image was actually saved
                if not self.analysis.segmentation_image:
                    raise RuntimeError("Image field is empty after save operation")
                
                logger.info(f"Core pipeline visualization saved successfully ({buffer_size} bytes)")
                
            except Exception as save_error:
                logger.error(f"Failed to save visualization: {str(save_error)}")
                # Try to save a minimal fallback visualization
                try:
                    logger.warning("Attempting to create fallback visualization")
                    if buffer:
                        buffer.close()
                    
                    # Create minimal fallback figure
                    fallback_fig, fallback_ax = plt.subplots(1, 1, figsize=(8, 6))
                    if len(original_image.shape) == 3:
                        fallback_ax.imshow(original_image)
                    else:
                        fallback_ax.imshow(original_image, cmap='gray')
                    fallback_ax.set_title(f'Analysis Results - {len(unique_masks)} cells detected', fontweight='bold')
                    fallback_ax.axis('off')
                    
                    fallback_buffer = BytesIO()
                    fallback_fig.savefig(fallback_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                    fallback_buffer.seek(0)
                    
                    filename = f'analysis_{self.analysis.id}_core_pipeline.png'
                    self.analysis.segmentation_image.save(
                        filename,
                        ContentFile(fallback_buffer.getvalue()),
                        save=False
                    )
                    
                    plt.close(fallback_fig)
                    fallback_buffer.close()
                    logger.info("Fallback visualization saved successfully")
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback visualization also failed: {str(fallback_error)}")
                    raise save_error  # Re-raise original error
            
            finally:
                # Ensure proper cleanup
                try:
                    plt.close(fig)
                    if buffer:
                        buffer.close()
                    logger.debug("Cleaned up matplotlib figure and buffer")
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup: {str(cleanup_error)}")
            
        except Exception as e:
            logger.error(f"Core pipeline visualization failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't re-raise - let analysis continue with other visualizations
    
    def _save_flow_analysis_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 2: Advanced Flow Analysis"""
        try:
            # Validate flow data at the start
            flow_data_valid = False
            if flows is not None and len(flows) > 0:
                if isinstance(flows[0], (list, tuple)) and len(flows[0]) > 1:
                    xy_flows = flows[0][1]
                    if xy_flows is not None and hasattr(xy_flows, 'shape'):
                        flow_data_valid = True
                        logger.debug(f"Flow data validation passed - shape: {xy_flows.shape}, type: {type(xy_flows)}")
                    else:
                        logger.debug(f"Flow data is None or has no shape attribute: {type(xy_flows)}")
                elif isinstance(flows[0], np.ndarray):
                    # Handle case where flows[0] is directly the flow array
                    logger.debug(f"Flow data is directly an array: shape {flows[0].shape}, type: {type(flows[0])}")
                    # This case may need special handling depending on Cellpose version
                else:
                    logger.debug(f"Flow data structure unexpected but not critical: {type(flows[0]) if flows else 'None'}")
            else:
                logger.debug(f"No flows data provided: {type(flows)} - will skip flow visualization")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Panel 1: Mask Boundaries/Edges  
            if len(original_image.shape) == 3:
                base_img = np.mean(original_image, axis=2)
            else:
                base_img = original_image
            axes[0].imshow(base_img, cmap='gray', alpha=0.7)
            
            # Add mask boundaries
            from skimage import measure
            contours = measure.find_contours(masks > 0, 0.5)
            for contour in contours:
                axes[0].plot(contour[:, 1], contour[:, 0], linewidth=2, color='lime')
            
            axes[0].set_title('1. Mask Boundaries/Edges', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Panel 2: Gradient Analysis
            try:
                if flow_data_valid and len(flows) > 0 and len(flows[0]) > 1:
                    xy_flows = flows[0][1]
                    logger.debug(f"Flow data shape for gradient: {xy_flows.shape}")
                    
                    if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                        # Calculate gradients of flow field
                        flow_x = xy_flows[:,:,0]
                        flow_y = xy_flows[:,:,1]
                        
                        # Check for valid flow data
                        if np.isfinite(flow_x).any() and np.isfinite(flow_y).any():
                            grad_x = np.gradient(flow_x)[1]
                            grad_y = np.gradient(flow_y)[0]
                            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                            
                            # Normalize gradient magnitude for better visualization
                            if gradient_magnitude.max() > 0:
                                gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
                                im = axes[1].imshow(gradient_magnitude, cmap='viridis', vmin=0, vmax=1)
                                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                                logger.debug("Gradient visualization completed successfully")
                            else:
                                logger.warning("Gradient magnitude is zero everywhere")
                                axes[1].imshow(np.zeros_like(masks), cmap='gray')
                                axes[1].text(0.5, 0.5, 'No gradient data', 
                                           transform=axes[1].transAxes, ha='center', va='center')
                        else:
                            logger.warning("Flow data contains no finite values")
                            axes[1].imshow(np.zeros_like(masks), cmap='gray')
                            axes[1].text(0.5, 0.5, 'Invalid flow data', 
                                       transform=axes[1].transAxes, ha='center', va='center')
                    else:
                        logger.warning(f"Unexpected flow shape: {xy_flows.shape}")
                        axes[1].imshow(np.zeros_like(masks), cmap='gray')
                        axes[1].text(0.5, 0.5, 'Flow shape error', 
                                   transform=axes[1].transAxes, ha='center', va='center')
                else:
                    logger.info("Creating alternative gradient visualization using mask edges")
                    # Alternative: show edge gradients from masks
                    from skimage import filters
                    if len(original_image.shape) == 3:
                        gray_img = np.mean(original_image, axis=2)
                    else:
                        gray_img = original_image
                    
                    # Use Sobel edge detection as alternative
                    edges = filters.sobel(gray_img)
                    if edges.max() > 0:
                        edges_norm = edges / edges.max()
                        im = axes[1].imshow(edges_norm, cmap='viridis', vmin=0, vmax=1)
                        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                        axes[1].text(0.02, 0.98, 'Alternative: Sobel edges', 
                                   transform=axes[1].transAxes, ha='left', va='top',
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=8)
                    else:
                        axes[1].imshow(np.zeros_like(masks), cmap='gray')
                        axes[1].text(0.5, 0.5, 'No gradient data available', 
                                   transform=axes[1].transAxes, ha='center', va='center')
            except Exception as grad_error:
                logger.error(f"Gradient analysis failed: {str(grad_error)}")
                axes[1].imshow(np.zeros_like(masks), cmap='gray')
                axes[1].text(0.5, 0.5, f'Gradient error\n{str(grad_error)[:20]}...', 
                           transform=axes[1].transAxes, ha='center', va='center', fontsize=10)
            
            axes[1].set_title('2. Flow Gradient Analysis', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Panel 3: Flow Magnitude Heatmap
            try:
                if flow_data_valid and len(flows) > 0 and len(flows[0]) > 1:
                    xy_flows = flows[0][1]  # XY flows at each pixel
                    logger.debug(f"Flow data shape for magnitude: {xy_flows.shape}")
                    
                    if len(xy_flows.shape) == 3 and xy_flows.shape[2] >= 2:
                        # Calculate flow magnitude
                        flow_x = xy_flows[:,:,0]
                        flow_y = xy_flows[:,:,1]
                        
                        # Check for valid flow data
                        if np.isfinite(flow_x).any() and np.isfinite(flow_y).any():
                            flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
                            
                            # Normalize magnitude for better visualization
                            if flow_magnitude.max() > 0:
                                flow_magnitude_norm = flow_magnitude / flow_magnitude.max()
                                im = axes[2].imshow(flow_magnitude_norm, cmap='hot', vmin=0, vmax=1)
                                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                                logger.debug(f"Flow magnitude visualization completed (max: {flow_magnitude.max():.3f})")
                            else:
                                logger.warning("Flow magnitude is zero everywhere")
                                axes[2].imshow(np.zeros_like(masks), cmap='gray')
                                axes[2].text(0.5, 0.5, 'No flow magnitude', 
                                           transform=axes[2].transAxes, ha='center', va='center')
                        else:
                            logger.warning("Flow data contains no finite values for magnitude")
                            axes[2].imshow(np.zeros_like(masks), cmap='gray')
                            axes[2].text(0.5, 0.5, 'Invalid flow data', 
                                       transform=axes[2].transAxes, ha='center', va='center')
                    else:
                        logger.warning(f"Unexpected flow shape for magnitude: {xy_flows.shape}")
                        axes[2].imshow(np.zeros_like(masks), cmap='gray')
                        axes[2].text(0.5, 0.5, 'Flow shape error', 
                                   transform=axes[2].transAxes, ha='center', va='center')
                else:
                    logger.info("Creating alternative magnitude visualization using distance transform")
                    # Alternative: distance transform from mask boundaries  
                    from scipy import ndimage
                    binary_masks = masks > 0
                    if binary_masks.any():
                        # Distance transform from cell boundaries
                        distance_transform = ndimage.distance_transform_edt(binary_masks)
                        if distance_transform.max() > 0:
                            distance_norm = distance_transform / distance_transform.max()
                            im = axes[2].imshow(distance_norm, cmap='hot', vmin=0, vmax=1)
                            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                            axes[2].text(0.02, 0.98, 'Alternative: Distance from edges', 
                                       transform=axes[2].transAxes, ha='left', va='top',
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=8)
                        else:
                            axes[2].imshow(np.zeros_like(masks), cmap='gray')
                            axes[2].text(0.5, 0.5, 'No distance data', 
                                       transform=axes[2].transAxes, ha='center', va='center')
                    else:
                        axes[2].imshow(np.zeros_like(masks), cmap='gray')
                        axes[2].text(0.5, 0.5, 'No masks detected', 
                                   transform=axes[2].transAxes, ha='center', va='center')
            except Exception as mag_error:
                logger.error(f"Flow magnitude analysis failed: {str(mag_error)}")
                axes[2].imshow(np.zeros_like(masks), cmap='gray')
                axes[2].text(0.5, 0.5, f'Magnitude error\n{str(mag_error)[:20]}...', 
                           transform=axes[2].transAxes, ha='center', va='center', fontsize=10)
            
            axes[2].set_title('3. Flow Magnitude Heatmap', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            # Panel 4: Flow Vector Field (with arrows)
            if len(original_image.shape) == 3:
                base_img = np.mean(original_image, axis=2)
            else:
                base_img = original_image
            axes[3].imshow(base_img, cmap='gray', alpha=0.5)
            
            try:
                if flow_data_valid and len(flows) > 0 and len(flows[0]) > 1:
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
                        
                        axes[3].quiver(x_coords, y_coords, u, -v, 
                                     angles='xy', scale_units='xy', scale=0.5,
                                     color='red', alpha=0.7, width=0.003)
            except Exception:
                pass
            
            axes[3].set_title('4. Flow Vector Field', fontsize=12, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            filename = f'analysis_{self.analysis.id}_flow_analysis.png'
            self.analysis.flow_analysis_image.save(
                filename,
                ContentFile(buffer.getvalue()),
                save=False
            )
            
            plt.close()
            logger.info("Flow analysis visualization saved successfully")
            
        except Exception as e:
            logger.error(f"Flow analysis visualization failed: {str(e)}")
    
    def _save_style_quality_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 3: Style & Quality Analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Panel 1: Quality Metrics Display
            axes[0].axis('off')
            quality_text = "Image Quality Assessment\n\n"
            if hasattr(self.analysis, 'quality_metrics') and self.analysis.quality_metrics:
                qm = self.analysis.quality_metrics
                quality_text += f"Overall Score: {qm.get('overall_score', 0):.1f}/100\n"
                quality_text += f"Category: {qm.get('quality_category', 'unknown').title()}\n\n"
                
                if 'blur_score' in qm:
                    quality_text += f"Blur Score: {qm['blur_score']:.1f}/100\n"
                if 'contrast_score' in qm:
                    quality_text += f"Contrast Score: {qm['contrast_score']:.1f}/100\n"
                if 'noise_score' in qm:
                    quality_text += f"Noise Score: {qm['noise_score']:.1f}/100\n"
            else:
                quality_text += "Quality metrics not available"
            
            axes[0].text(0.1, 0.9, quality_text, transform=axes[0].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
            axes[0].set_title('1. Quality Assessment', fontsize=12, fontweight='bold')
            
            # Panel 2: Cell Size Distribution
            props = measure.regionprops(masks)
            if props:
                areas = [prop.area for prop in props if prop.label > 0]
                if areas:
                    axes[1].hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[1].axvline(np.mean(areas), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(areas):.1f}')
                    axes[1].legend()
                    axes[1].set_xlabel('Cell Area (pixels²)')
                    axes[1].set_ylabel('Frequency')
                else:
                    axes[1].text(0.5, 0.5, 'No cells detected', transform=axes[1].transAxes, ha='center', va='center')
            else:
                axes[1].text(0.5, 0.5, 'No cells detected', transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('2. Cell Size Distribution', fontsize=12, fontweight='bold')
            
            # Panel 3: Shape Analysis
            if props:
                circularities = [4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0 
                               for prop in props if prop.label > 0]
                eccentricities = [prop.eccentricity for prop in props if prop.label > 0]
                
                if circularities and eccentricities:
                    scatter = axes[2].scatter(circularities, eccentricities, alpha=0.6, c='green', s=30)
                    axes[2].set_xlabel('Circularity')
                    axes[2].set_ylabel('Eccentricity')
                    axes[2].grid(True, alpha=0.3)
                else:
                    axes[2].text(0.5, 0.5, 'No shape data', transform=axes[2].transAxes, ha='center', va='center')
            else:
                axes[2].text(0.5, 0.5, 'No cells detected', transform=axes[2].transAxes, ha='center', va='center')
            axes[2].set_title('3. Shape Analysis', fontsize=12, fontweight='bold')
            
            # Panel 4: Processing Summary
            axes[3].axis('off')
            summary_text = "Analysis Summary\n\n"
            summary_text += f"Cells Detected: {len(np.unique(masks))-1}\n"
            summary_text += f"Model Used: {self.analysis.cellpose_model}\n"
            summary_text += f"Diameter: {self.analysis.cellpose_diameter:.1f} px\n"
            summary_text += f"Flow Threshold: {self.analysis.flow_threshold}\n"
            summary_text += f"CellProb Threshold: {self.analysis.cellprob_threshold}\n\n"
            
            if hasattr(self.analysis, 'quality_metrics') and 'segmentation_refinement' in self.analysis.quality_metrics:
                ref_info = self.analysis.quality_metrics['segmentation_refinement']
                summary_text += "Refinement Applied:\n"
                for step in ref_info.get('refinement_steps', []):
                    summary_text += f"• {step}\n"
            
            axes[3].text(0.1, 0.9, summary_text, transform=axes[3].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[3].set_title('4. Processing Summary', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save visualization
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            filename = f'analysis_{self.analysis.id}_style_quality.png'
            self.analysis.style_quality_image.save(
                filename,
                ContentFile(buffer.getvalue()),
                save=False
            )
            
            plt.close()
            logger.info("Style & quality visualization saved successfully")
            
        except Exception as e:
            logger.error(f"Style & quality visualization failed: {str(e)}")
    
    def _save_edge_boundary_visualization(self, original_image, masks, flows, styles, diameters):
        """Page 4: Edge & Boundary Analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Panel 1: Edge Detection
            if len(original_image.shape) == 3:
                gray_img = np.mean(original_image, axis=2)
            else:
                gray_img = original_image
            
            from skimage import feature
            edges = feature.canny(gray_img, sigma=1, low_threshold=0.1, high_threshold=0.2)
            axes[0].imshow(edges, cmap='gray')
            axes[0].set_title('1. Edge Detection (Canny)', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Panel 2: Boundary Analysis
            axes[1].imshow(gray_img, cmap='gray', alpha=0.7)
            
            # Overlay cell boundaries
            from skimage import measure
            props = measure.regionprops(masks)
            for prop in props:
                if prop.label > 0:
                    # Get boundary coordinates
                    boundary = measure.find_contours(masks == prop.label, 0.5)
                    for contour in boundary:
                        axes[1].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.8)
            
            axes[1].set_title('2. Cell Boundary Overlay', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Panel 3: Morphological Operations
            binary_masks = masks > 0
            from skimage import morphology
            
            # Apply morphological operations for comparison
            opened = morphology.opening(binary_masks, morphology.disk(2))
            closed = morphology.closing(binary_masks, morphology.disk(2))
            
            # Create composite image showing differences
            composite = np.zeros((*binary_masks.shape, 3))
            composite[:,:,0] = binary_masks  # Original in red
            composite[:,:,1] = opened       # Opened in green  
            composite[:,:,2] = closed       # Closed in blue
            
            axes[2].imshow(composite)
            axes[2].set_title('3. Morphological Analysis\n(Red: Original, Green: Opened, Blue: Closed)', fontsize=10, fontweight='bold')
            axes[2].axis('off')
            
            # Panel 4: Size and Shape Validation
            if props:
                areas = [prop.area for prop in props if prop.label > 0]
                perimeters = [prop.perimeter for prop in props if prop.label > 0]
                
                if areas and perimeters:
                    # Create scatter plot of area vs perimeter
                    axes[3].scatter(areas, perimeters, alpha=0.6, c='purple', s=30)
                    axes[3].set_xlabel('Area (pixels²)')
                    axes[3].set_ylabel('Perimeter (pixels)')
                    axes[3].grid(True, alpha=0.3)
                    
                    # Add theoretical circle line for reference
                    area_range = np.linspace(min(areas), max(areas), 100)
                    circle_perimeter = 2 * np.sqrt(np.pi * area_range)
                    axes[3].plot(area_range, circle_perimeter, 'r--', alpha=0.7, label='Perfect Circle')
                    axes[3].legend()
                else:
                    axes[3].text(0.5, 0.5, 'No measurement data', transform=axes[3].transAxes, ha='center', va='center')
            else:
                axes[3].text(0.5, 0.5, 'No cells detected', transform=axes[3].transAxes, ha='center', va='center')
            
            axes[3].set_title('4. Area vs Perimeter Analysis', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save visualization
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            filename = f'analysis_{self.analysis.id}_edge_boundary.png'
            self.analysis.edge_boundary_image.save(
                filename,
                ContentFile(buffer.getvalue()),
                save=False
            )
            
            plt.close()
            logger.info("Edge & boundary visualization saved successfully")
            
        except Exception as e:
            logger.error(f"Edge & boundary visualization failed: {str(e)}")
            
    def _extract_morphometric_features(self, masks):
        """Extract morphometric features from segmented cells using scikit-image."""
        try:
            # Clear existing detected cells
            self.analysis.detected_cells.all().delete()
            
            # Use standard scikit-image morphometric calculations
            logger.info("Using CPU morphometric calculations")
            cpu_start_time = time.time()
            
            # Get region properties
            props = measure.regionprops(masks)
            
            for prop in props:
                if prop.label > 0:  # Skip background
                    # Calculate additional metrics
                    area = prop.area
                    perimeter = prop.perimeter
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    
                    # Calculate aspect ratio
                    aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
                    
                    # Create DetectedCell instance with correct field names
                    detected_cell = DetectedCell(
                        analysis=self.analysis,
                        cell_id=prop.label,
                        area=area,
                        perimeter=perimeter,
                        circularity=circularity,
                        eccentricity=prop.eccentricity,
                        solidity=prop.solidity,
                        extent=prop.extent,
                        major_axis_length=prop.major_axis_length,
                        minor_axis_length=prop.minor_axis_length,
                        aspect_ratio=aspect_ratio,
                        centroid_x=prop.centroid[1],
                        centroid_y=prop.centroid[0],
                        bounding_box_x=prop.bbox[1],  # min_col 
                        bounding_box_y=prop.bbox[0],  # min_row
                        bounding_box_width=prop.bbox[3] - prop.bbox[1],  # max_col - min_col
                        bounding_box_height=prop.bbox[2] - prop.bbox[0]  # max_row - min_row
                    )
                    
                    # Calculate physical measurements if scale is available
                    if self.analysis.cell.scale_set and self.analysis.cell.pixels_per_micron:
                        ppm = self.analysis.cell.pixels_per_micron
                        detected_cell.area_microns_sq = area / (ppm ** 2)
                        detected_cell.perimeter_microns = perimeter / ppm
                        detected_cell.major_axis_length_microns = prop.major_axis_length / ppm
                        detected_cell.minor_axis_length_microns = prop.minor_axis_length / ppm
                    
                    detected_cell.save()
            
            cpu_time = time.time() - cpu_start_time        
            logger.info(f"CPU morphometric feature extraction completed for {len(props)} cells in {cpu_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise MorphometricAnalysisError(f"Feature extraction failed: {str(e)}")
    
    def _mark_failed(self, error_message):
        """Mark analysis as failed with error message."""
        self.analysis.status = 'failed'
        self.analysis.error_message = error_message
        self.analysis.save()
        logger.error(f"Analysis marked as failed: {error_message}")


# Legacy support - maintain backward compatibility
# These classes are now imported from their respective modules
__all__ = [
    'ImageQualityAssessment',
    'ImagePreprocessor', 
    'ParameterOptimizer',
    'SegmentationRefinement',
    'CellAnalysisProcessor',
    'run_cell_analysis',
    'get_image_quality_summary', 
    'get_analysis_summary'
]