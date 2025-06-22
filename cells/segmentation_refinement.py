"""
Segmentation Refinement Module

This module provides post-processing refinement methods for improving 
segmentation accuracy including size filtering, shape filtering, 
watershed splitting, and boundary smoothing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

try:
    from skimage import measure, morphology, feature, segmentation
    from skimage.segmentation import clear_border
    from skimage.morphology import disk
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from .exceptions import SegmentationRefinementError, DependencyError
from .utils import validate_image_array, safe_divide

logger = logging.getLogger(__name__)


class SegmentationRefinement:
    """
    Post-processing refinement for improving segmentation accuracy.
    
    This class provides static methods for refining segmentation masks
    to improve cell detection accuracy and reduce false positives.
    """
    
    def __init__(self):
        """Initialize segmentation refinement."""
        if not SKIMAGE_AVAILABLE:
            raise DependencyError("scikit-image and scipy are required for segmentation refinement")
    
    @staticmethod
    def filter_by_size(
        masks: np.ndarray, 
        min_area: int = 50, 
        max_area: Optional[int] = None
    ) -> np.ndarray:
        """
        Remove objects that are too small or too large to be cells.
        
        Args:
            masks: Input segmentation masks (2D integer array)
            min_area: Minimum cell area in pixels
            max_area: Maximum cell area in pixels (None for auto-calculation)
            
        Returns:
            Refined masks with size filtering applied
            
        Raises:
            SegmentationRefinementError: If size filtering fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            if min_area < 0:
                raise SegmentationRefinementError("Minimum area must be non-negative")
            
            if max_area is None:
                # Set max area to 1/4 of image area as default
                max_area = (masks.shape[0] * masks.shape[1]) // 4
            
            if max_area <= min_area:
                raise SegmentationRefinementError("Maximum area must be greater than minimum area")
            
            logger.debug(f"Filtering by size: min_area={min_area}, max_area={max_area}")
            
            refined_masks = np.zeros_like(masks)
            new_label = 1
            
            # Get properties of all regions
            props = measure.regionprops(masks)
            original_count = len(props)
            kept_count = 0
            
            for prop in props:
                if min_area <= prop.area <= max_area:
                    # Keep this region, assign new label
                    refined_masks[masks == prop.label] = new_label
                    new_label += 1
                    kept_count += 1
            
            logger.info(f"Size filtering: kept {kept_count}/{original_count} cells")
            return refined_masks
            
        except Exception as e:
            logger.error(f"Error in size filtering: {str(e)}")
            raise SegmentationRefinementError(f"Size filtering failed: {str(e)}")
    
    @staticmethod
    def filter_by_shape(
        masks: np.ndarray, 
        min_circularity: float = 0.1, 
        max_eccentricity: float = 0.95,
        min_solidity: float = 0.7
    ) -> np.ndarray:
        """
        Remove objects with non-cellular shapes.
        
        Args:
            masks: Input segmentation masks
            min_circularity: Minimum circularity (0-1)
            max_eccentricity: Maximum eccentricity (0-1)
            min_solidity: Minimum solidity (0-1)
            
        Returns:
            Refined masks with shape filtering applied
            
        Raises:
            SegmentationRefinementError: If shape filtering fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            # Validate parameters
            if not (0 <= min_circularity <= 1):
                raise SegmentationRefinementError("min_circularity must be between 0 and 1")
            if not (0 <= max_eccentricity <= 1):
                raise SegmentationRefinementError("max_eccentricity must be between 0 and 1")
            if not (0 <= min_solidity <= 1):
                raise SegmentationRefinementError("min_solidity must be between 0 and 1")
            
            logger.debug(f"Filtering by shape: circularity>={min_circularity}, "
                        f"eccentricity<={max_eccentricity}, solidity>={min_solidity}")
            
            refined_masks = np.zeros_like(masks)
            new_label = 1
            
            props = measure.regionprops(masks)
            original_count = len(props)
            kept_count = 0
            
            for prop in props:
                # Calculate circularity
                area = prop.area
                perimeter = prop.perimeter
                circularity = safe_divide(4 * np.pi * area, perimeter ** 2, default=0)
                
                # Check shape criteria
                if (circularity >= min_circularity and 
                    prop.eccentricity <= max_eccentricity and
                    prop.solidity >= min_solidity):
                    
                    refined_masks[masks == prop.label] = new_label
                    new_label += 1
                    kept_count += 1
            
            logger.info(f"Shape filtering: kept {kept_count}/{original_count} cells")
            return refined_masks
            
        except Exception as e:
            logger.error(f"Error in shape filtering: {str(e)}")
            raise SegmentationRefinementError(f"Shape filtering failed: {str(e)}")
    
    @staticmethod
    def split_touching_cells(
        masks: np.ndarray, 
        min_distance: int = 10
    ) -> np.ndarray:
        """
        Use watershed segmentation to split touching cells.
        
        Args:
            masks: Input segmentation masks
            min_distance: Minimum distance between cell centers
            
        Returns:
            Refined masks with watershed splitting applied
            
        Raises:
            SegmentationRefinementError: If watershed splitting fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            if min_distance <= 0:
                raise SegmentationRefinementError("min_distance must be positive")
            
            logger.debug(f"Splitting touching cells with min_distance={min_distance}")
            
            # Create binary mask
            binary = masks > 0
            
            if not np.any(binary):
                logger.warning("No cells found in masks for watershed splitting")
                return masks.copy()
            
            # Calculate distance transform
            distance = ndimage.distance_transform_edt(binary)
            
            # Find local maxima (cell centers)
            local_maxima = feature.peak_local_max(
                distance, 
                min_distance=min_distance,
                threshold_abs=max(1, min_distance//2),
                exclude_border=False
            )
            
            if len(local_maxima) == 0:
                logger.warning("No local maxima found for watershed splitting")
                return masks.copy()
            
            # Create markers for watershed
            markers = np.zeros_like(masks, dtype=int)
            for i, (y, x) in enumerate(local_maxima):
                if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
                    markers[y, x] = i + 1
            
            if np.max(markers) == 0:
                logger.warning("No valid markers created for watershed")
                return masks.copy()
            
            # Apply watershed
            refined_masks = segmentation.watershed(-distance, markers, mask=binary)
            
            original_count = len(np.unique(masks)) - 1
            new_count = len(np.unique(refined_masks)) - 1
            
            logger.info(f"Watershed splitting: {original_count} → {new_count} cells")
            return refined_masks
            
        except Exception as e:
            logger.error(f"Error in watershed splitting: {str(e)}")
            raise SegmentationRefinementError(f"Watershed splitting failed: {str(e)}")
    
    @staticmethod
    def smooth_boundaries(
        masks: np.ndarray, 
        smoothing_factor: float = 1.0
    ) -> np.ndarray:
        """
        Smooth cell boundaries using morphological operations.
        
        Args:
            masks: Input segmentation masks
            smoothing_factor: Size of morphological kernel
            
        Returns:
            Refined masks with smoothed boundaries
            
        Raises:
            SegmentationRefinementError: If boundary smoothing fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            if smoothing_factor <= 0:
                raise SegmentationRefinementError("smoothing_factor must be positive")
            
            logger.debug(f"Smoothing boundaries with factor={smoothing_factor}")
            
            refined_masks = np.zeros_like(masks)
            unique_labels = np.unique(masks)[1:]  # Skip background
            
            if len(unique_labels) == 0:
                logger.warning("No cells found for boundary smoothing")
                return masks.copy()
            
            kernel_size = max(1, int(smoothing_factor))
            kernel = disk(kernel_size)
            
            for label in unique_labels:
                try:
                    # Extract single cell mask
                    cell_mask = (masks == label).astype(np.uint8)
                    
                    # Apply morphological operations
                    # Closing to fill small gaps
                    cell_mask = morphology.closing(cell_mask, kernel)
                    
                    # Opening to smooth boundaries
                    cell_mask = morphology.opening(cell_mask, kernel)
                    
                    # Fill holes
                    cell_mask = binary_fill_holes(cell_mask)
                    
                    # Assign to refined masks
                    refined_masks[cell_mask > 0] = label
                    
                except Exception as e:
                    logger.warning(f"Error smoothing cell {label}: {str(e)}")
                    # Keep original mask for this cell
                    refined_masks[masks == label] = label
            
            logger.info("Boundary smoothing completed")
            return refined_masks
            
        except Exception as e:
            logger.error(f"Error in boundary smoothing: {str(e)}")
            raise SegmentationRefinementError(f"Boundary smoothing failed: {str(e)}")
    
    @staticmethod
    def remove_edge_cells(
        masks: np.ndarray, 
        border_width: int = 5
    ) -> np.ndarray:
        """
        Remove cells touching the image edges.
        
        Args:
            masks: Input segmentation masks
            border_width: Width of border region to clear
            
        Returns:
            Refined masks with edge cells removed
            
        Raises:
            SegmentationRefinementError: If edge removal fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            if border_width < 0:
                raise SegmentationRefinementError("border_width must be non-negative")
            
            logger.debug(f"Removing edge cells with border_width={border_width}")
            
            original_count = len(np.unique(masks)) - 1
            refined_masks = clear_border(masks, buffer_size=border_width)
            final_count = len(np.unique(refined_masks)) - 1
            
            logger.info(f"Edge removal: {original_count} → {final_count} cells")
            return refined_masks
            
        except Exception as e:
            logger.error(f"Error removing edge cells: {str(e)}")
            raise SegmentationRefinementError(f"Edge removal failed: {str(e)}")
    
    @staticmethod
    def refine_segmentation(
        masks: np.ndarray, 
        image_array: Optional[np.ndarray] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply comprehensive segmentation refinement pipeline.
        
        Args:
            masks: Input segmentation masks
            image_array: Original image (optional, for future enhancements)
            options: Refinement options dictionary
            
        Returns:
            Tuple of (refined_masks, refinement_steps)
            
        Raises:
            SegmentationRefinementError: If refinement pipeline fails
        """
        try:
            if masks is None or masks.size == 0:
                raise SegmentationRefinementError("Invalid input masks")
            
            options = options or {}
            refined_masks = masks.copy()
            refinement_steps = []
            
            original_count = len(np.unique(refined_masks)) - 1  # Subtract background
            logger.info(f"Starting refinement with {original_count} cells")
            
            current_count = original_count
            
            # Step 1: Size filtering
            if options.get('apply_size_filtering', True):
                min_area = options.get('min_cell_area', 50)
                max_area = options.get('max_cell_area', None)
                
                refined_masks = SegmentationRefinement.filter_by_size(
                    refined_masks, min_area, max_area
                )
                after_size = len(np.unique(refined_masks)) - 1
                refinement_steps.append(f"Size filtering: {current_count} → {after_size} cells")
                current_count = after_size
            
            # Step 2: Shape filtering
            if options.get('apply_shape_filtering', True):
                min_circularity = options.get('min_circularity', 0.1)
                max_eccentricity = options.get('max_eccentricity', 0.95)
                min_solidity = options.get('min_solidity', 0.7)
                
                refined_masks = SegmentationRefinement.filter_by_shape(
                    refined_masks, min_circularity, max_eccentricity, min_solidity
                )
                after_shape = len(np.unique(refined_masks)) - 1
                refinement_steps.append(f"Shape filtering: {current_count} → {after_shape} cells")
                current_count = after_shape
            
            # Step 3: Split touching cells
            if options.get('apply_watershed', False):
                min_distance = options.get('watershed_min_distance', 10)
                refined_masks = SegmentationRefinement.split_touching_cells(
                    refined_masks, min_distance
                )
                after_watershed = len(np.unique(refined_masks)) - 1
                refinement_steps.append(f"Watershed splitting: {current_count} → {after_watershed} cells")
                current_count = after_watershed
            
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
                refinement_steps.append(f"Edge removal: {current_count} → {final_count} cells")
                current_count = final_count
            
            refinement_steps.append(f"Final result: {current_count} cells")
            
            logger.info(f"Refinement completed: {original_count} → {current_count} cells")
            return refined_masks, refinement_steps
            
        except Exception as e:
            logger.error(f"Error in segmentation refinement: {str(e)}")
            raise SegmentationRefinementError(f"Segmentation refinement failed: {str(e)}")
    
    @staticmethod
    def get_default_options() -> Dict[str, Any]:
        """
        Get default refinement options.
        
        Returns:
            Dictionary with default refinement options
        """
        return {
            'apply_size_filtering': True,
            'min_cell_area': 50,
            'max_cell_area': None,
            
            'apply_shape_filtering': True,
            'min_circularity': 0.1,
            'max_eccentricity': 0.95,
            'min_solidity': 0.7,
            
            'apply_watershed': False,
            'watershed_min_distance': 10,
            
            'apply_smoothing': True,
            'smoothing_factor': 1.0,
            
            'remove_edge_cells': False,
            'border_width': 5
        }
    
    @staticmethod
    def validate_options(options: Dict[str, Any]) -> List[str]:
        """
        Validate refinement options.
        
        Args:
            options: Dictionary of options to validate
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Validate size filtering options
        min_area = options.get('min_cell_area')
        if min_area is not None and (not isinstance(min_area, int) or min_area < 0):
            errors.append("min_cell_area must be a non-negative integer")
        
        max_area = options.get('max_cell_area')
        if max_area is not None and (not isinstance(max_area, int) or max_area <= 0):
            errors.append("max_cell_area must be a positive integer")
        
        if min_area is not None and max_area is not None and max_area <= min_area:
            errors.append("max_cell_area must be greater than min_cell_area")
        
        # Validate shape filtering options
        for param, name in [('min_circularity', 'min_circularity'), 
                           ('max_eccentricity', 'max_eccentricity'),
                           ('min_solidity', 'min_solidity')]:
            value = options.get(param)
            if value is not None and (not isinstance(value, (int, float)) or not (0 <= value <= 1)):
                errors.append(f"{name} must be a number between 0 and 1")
        
        # Validate watershed options
        min_distance = options.get('watershed_min_distance')
        if min_distance is not None and (not isinstance(min_distance, int) or min_distance <= 0):
            errors.append("watershed_min_distance must be a positive integer")
        
        # Validate smoothing options
        smoothing_factor = options.get('smoothing_factor')
        if smoothing_factor is not None and (not isinstance(smoothing_factor, (int, float)) or smoothing_factor <= 0):
            errors.append("smoothing_factor must be a positive number")
        
        # Validate edge removal options
        border_width = options.get('border_width')
        if border_width is not None and (not isinstance(border_width, int) or border_width < 0):
            errors.append("border_width must be a non-negative integer")
        
        return errors