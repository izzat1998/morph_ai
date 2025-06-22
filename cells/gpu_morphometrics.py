"""
GPU-Accelerated Morphometric Calculations Module

This module provides GPU-accelerated implementations of morphometric feature extraction
using CuPy and PyTorch for improved performance on large datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from .exceptions import MorphometricAnalysisError, DependencyError
from .gpu_utils import gpu_manager, cleanup_gpu_memory

logger = logging.getLogger(__name__)


class GPUMorphometrics:
    """
    GPU-accelerated morphometric feature extraction.
    
    This class provides GPU implementations of common morphometric calculations
    with automatic fallback to CPU when GPU is not available.
    """
    
    def __init__(self):
        """Initialize GPU morphometrics calculator."""
        self.cupy_available = False
        self.torch_available = False
        self.gpu_info = None
        self.cp = None
        self.torch = None
        
        # Try to initialize CuPy
        try:
            import cupy as cp
            self.cp = cp
            self.gpu_info = gpu_manager.detect_gpu_capabilities()
            if self.gpu_info.backend == 'cuda':
                self.cupy_available = True
                logger.info("GPU morphometrics enabled with CuPy")
        except ImportError:
            logger.debug("CuPy not available for GPU morphometrics")
        
        # Try to initialize PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.torch_available = True
                logger.info("PyTorch CUDA available for GPU morphometrics")
        except ImportError:
            logger.debug("PyTorch not available for GPU morphometrics")
        
        if not self.cupy_available and not self.torch_available:
            logger.info("No GPU libraries available, using CPU morphometrics")
    
    def _to_gpu_cupy(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Move array to GPU using CuPy."""
        if self.cupy_available:
            try:
                return self.cp.asarray(array)
            except Exception as e:
                logger.warning(f"Failed to move array to GPU with CuPy: {str(e)}")
        return array
    
    def _to_gpu_torch(self, array: np.ndarray) -> Union[np.ndarray, 'torch.Tensor']:
        """Move array to GPU using PyTorch."""
        if self.torch_available:
            try:
                return self.torch.from_numpy(array).cuda()
            except Exception as e:
                logger.warning(f"Failed to move array to GPU with PyTorch: {str(e)}")
        return array
    
    def _to_cpu_cupy(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Move array back to CPU from CuPy."""
        if self.cupy_available and hasattr(array, 'get'):
            return array.get()
        return array
    
    def _to_cpu_torch(self, tensor: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Move tensor back to CPU from PyTorch."""
        if self.torch_available and hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        return tensor
    
    def calculate_areas_gpu(self, masks: np.ndarray) -> Dict[int, float]:
        """
        Calculate cell areas using GPU acceleration.
        
        Args:
            masks: Labeled mask array where each unique value represents a cell
            
        Returns:
            Dictionary mapping cell_id to area in pixels
        """
        if not self.cupy_available:
            # Fallback to CPU using skimage
            props = measure.regionprops(masks)
            return {prop.label: float(prop.area) for prop in props if prop.label > 0}
        
        try:
            start_time = time.time()
            gpu_masks = self._to_gpu_cupy(masks)
            
            # Get unique labels (cell IDs)
            unique_labels = self.cp.unique(gpu_masks)
            unique_labels = unique_labels[unique_labels > 0]  # Skip background
            
            areas = {}
            for label in unique_labels:
                # Count pixels for each label
                area = self.cp.sum(gpu_masks == label)
                areas[int(label)] = float(area)
            
            elapsed = time.time() - start_time
            logger.debug(f"GPU area calculation completed in {elapsed:.3f}s for {len(areas)} cells")
            
            return areas
            
        except Exception as e:
            logger.warning(f"GPU area calculation failed, using CPU: {str(e)}")
            props = measure.regionprops(masks)
            return {prop.label: float(prop.area) for prop in props if prop.label > 0}
    
    def calculate_perimeters_gpu(self, masks: np.ndarray) -> Dict[int, float]:
        """
        Calculate cell perimeters using GPU acceleration.
        
        Args:
            masks: Labeled mask array
            
        Returns:
            Dictionary mapping cell_id to perimeter in pixels
        """
        if not self.cupy_available:
            # Fallback to CPU
            props = measure.regionprops(masks)
            return {prop.label: float(prop.perimeter) for prop in props if prop.label > 0}
        
        try:
            start_time = time.time()
            gpu_masks = self._to_gpu_cupy(masks)
            
            unique_labels = self.cp.unique(gpu_masks)
            unique_labels = unique_labels[unique_labels > 0]
            
            perimeters = {}
            
            # Sobel kernels for edge detection
            sobel_x = self.cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=self.cp.float32)
            sobel_y = self.cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=self.cp.float32)
            
            for label in unique_labels:
                # Create binary mask for this cell
                cell_mask = (gpu_masks == label).astype(self.cp.float32)
                
                # Apply Sobel filters to detect edges
                from cupyx.scipy import ndimage as cp_ndimage
                grad_x = cp_ndimage.convolve(cell_mask, sobel_x)
                grad_y = cp_ndimage.convolve(cell_mask, sobel_y)
                
                # Calculate gradient magnitude
                gradient_magnitude = self.cp.sqrt(grad_x**2 + grad_y**2)
                
                # Threshold to get perimeter pixels
                perimeter_pixels = self.cp.sum(gradient_magnitude > 0.1)
                perimeters[int(label)] = float(perimeter_pixels)
            
            elapsed = time.time() - start_time
            logger.debug(f"GPU perimeter calculation completed in {elapsed:.3f}s for {len(perimeters)} cells")
            
            return perimeters
            
        except Exception as e:
            logger.warning(f"GPU perimeter calculation failed, using CPU: {str(e)}")
            props = measure.regionprops(masks)
            return {prop.label: float(prop.perimeter) for prop in props if prop.label > 0}
    
    def calculate_centroids_gpu(self, masks: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """
        Calculate cell centroids using GPU acceleration.
        
        Args:
            masks: Labeled mask array
            
        Returns:
            Dictionary mapping cell_id to (y, x) centroid coordinates
        """
        if not self.cupy_available:
            # Fallback to CPU
            props = measure.regionprops(masks)
            return {prop.label: prop.centroid for prop in props if prop.label > 0}
        
        try:
            start_time = time.time()
            gpu_masks = self._to_gpu_cupy(masks)
            
            unique_labels = self.cp.unique(gpu_masks)
            unique_labels = unique_labels[unique_labels > 0]
            
            centroids = {}
            
            # Create coordinate grids
            height, width = gpu_masks.shape
            y_coords, x_coords = self.cp.mgrid[0:height, 0:width]
            
            for label in unique_labels:
                # Get mask for this cell
                cell_mask = (gpu_masks == label)
                
                # Calculate centroid
                total_pixels = self.cp.sum(cell_mask)
                if total_pixels > 0:
                    y_center = self.cp.sum(y_coords[cell_mask]) / total_pixels
                    x_center = self.cp.sum(x_coords[cell_mask]) / total_pixels
                    centroids[int(label)] = (float(y_center), float(x_center))
            
            elapsed = time.time() - start_time
            logger.debug(f"GPU centroid calculation completed in {elapsed:.3f}s for {len(centroids)} cells")
            
            return centroids
            
        except Exception as e:
            logger.warning(f"GPU centroid calculation failed, using CPU: {str(e)}")
            props = measure.regionprops(masks)
            return {prop.label: prop.centroid for prop in props if prop.label > 0}
    
    def calculate_shape_descriptors_gpu(self, masks: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Calculate shape descriptors (circularity, eccentricity, etc.) using GPU.
        
        Args:
            masks: Labeled mask array
            
        Returns:
            Dictionary mapping cell_id to shape descriptor dictionary
        """
        try:
            # For complex shape calculations, combine GPU and CPU approaches
            start_time = time.time()
            
            # Use GPU for basic calculations
            areas = self.calculate_areas_gpu(masks)
            perimeters = self.calculate_perimeters_gpu(masks)
            
            shape_descriptors = {}
            
            # Calculate shape descriptors from areas and perimeters
            for cell_id in areas:
                area = areas[cell_id]
                perimeter = perimeters.get(cell_id, 0)
                
                # Circularity = 4π * area / perimeter²
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # For more complex descriptors, fall back to CPU regionprops
                try:
                    # Extract single cell mask
                    cell_mask = (masks == cell_id).astype(np.uint8)
                    props = measure.regionprops(cell_mask)
                    
                    if props:
                        prop = props[0]
                        shape_descriptors[cell_id] = {
                            'area': area,
                            'perimeter': perimeter,
                            'circularity': circularity,
                            'eccentricity': prop.eccentricity,
                            'solidity': prop.solidity,
                            'extent': prop.extent,
                            'major_axis_length': prop.major_axis_length,
                            'minor_axis_length': prop.minor_axis_length,
                            'aspect_ratio': prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
                        }
                    else:
                        # Basic descriptors only
                        shape_descriptors[cell_id] = {
                            'area': area,
                            'perimeter': perimeter,
                            'circularity': circularity
                        }
                        
                except Exception as desc_error:
                    logger.warning(f"Failed to calculate descriptors for cell {cell_id}: {str(desc_error)}")
                    shape_descriptors[cell_id] = {
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity
                    }
            
            elapsed = time.time() - start_time
            logger.debug(f"GPU shape descriptor calculation completed in {elapsed:.3f}s for {len(shape_descriptors)} cells")
            
            return shape_descriptors
            
        except Exception as e:
            logger.warning(f"GPU shape descriptor calculation failed, using CPU: {str(e)}")
            # Full fallback to CPU
            props = measure.regionprops(masks)
            shape_descriptors = {}
            
            for prop in props:
                if prop.label > 0:
                    shape_descriptors[prop.label] = {
                        'area': float(prop.area),
                        'perimeter': float(prop.perimeter),
                        'circularity': (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0,
                        'eccentricity': prop.eccentricity,
                        'solidity': prop.solidity,
                        'extent': prop.extent,
                        'major_axis_length': prop.major_axis_length,
                        'minor_axis_length': prop.minor_axis_length,
                        'aspect_ratio': prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
                    }
            
            return shape_descriptors
    
    def batch_calculate_morphometrics(self, masks: np.ndarray) -> Dict[str, Dict[int, Any]]:
        """
        Efficiently calculate all morphometric features in a single GPU operation.
        
        Args:
            masks: Labeled mask array
            
        Returns:
            Dictionary with morphometric features for all cells
        """
        try:
            start_time = time.time()
            logger.info("Starting batch GPU morphometric calculation")
            
            # Calculate all features
            areas = self.calculate_areas_gpu(masks)
            perimeters = self.calculate_perimeters_gpu(masks)
            centroids = self.calculate_centroids_gpu(masks)
            shape_descriptors = self.calculate_shape_descriptors_gpu(masks)
            
            # Combine results
            results = {
                'areas': areas,
                'perimeters': perimeters,
                'centroids': centroids,
                'shape_descriptors': shape_descriptors
            }
            
            # Clean up GPU memory
            if self.cupy_available or self.torch_available:
                cleanup_gpu_memory()
            
            elapsed = time.time() - start_time
            num_cells = len(areas)
            logger.info(f"Batch GPU morphometric calculation completed in {elapsed:.3f}s for {num_cells} cells")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch GPU morphometric calculation failed: {str(e)}")
            cleanup_gpu_memory()
            raise MorphometricAnalysisError(f"GPU morphometric calculation failed: {str(e)}")


# Global GPU morphometrics instance
gpu_morphometrics = GPUMorphometrics()


def calculate_morphometrics_gpu(masks: np.ndarray) -> Dict[str, Dict[int, Any]]:
    """
    Calculate morphometric features using GPU acceleration.
    
    Args:
        masks: Labeled mask array
        
    Returns:
        Dictionary with morphometric features
    """
    return gpu_morphometrics.batch_calculate_morphometrics(masks)


def is_gpu_morphometrics_available() -> bool:
    """Check if GPU morphometrics are available."""
    return gpu_morphometrics.cupy_available or gpu_morphometrics.torch_available