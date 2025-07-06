"""
Proper Cellpose-SAM Integration for Morph AI

This module provides correct integration with Cellpose 4.0+ CellposeSAM
using the official API as documented at https://cellpose.readthedocs.io/en/latest/api.html

This replaces the incorrect cellpose_sam_integration.py that tried to fake SAM support.
"""

import logging
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Import Cellpose components
try:
    from cellpose import models
    from cellpose.io import imread
    from cellpose import plot, utils
    CELLPOSE_AVAILABLE = True
    logger.info("Cellpose 4.0+ with CellposeSAM available")
except ImportError as e:
    CELLPOSE_AVAILABLE = False
    logger.error(f"Cellpose not available: {e}")


class CellposeSAMSegmenter:
    """
    Proper Cellpose-SAM segmentation using the official Cellpose 4.0+ API.
    
    Uses the 'cpsam' pretrained model which is the default SAM model in Cellpose 4.0+.
    """
    
    def __init__(self, gpu=True, device=None, nchan=None, diam_mean=None):
        """
        Initialize CellposeSAM model using official API.
        
        Args:
            gpu: Whether to use GPU acceleration
            device: Optional device specification
            nchan: Number of channels
            diam_mean: Mean diameter for normalization
        """
        if not CELLPOSE_AVAILABLE:
            raise RuntimeError("Cellpose not available")
        
        self.gpu = gpu and self._check_gpu_availability()
        self.device = device
        self.nchan = nchan
        self.diam_mean = diam_mean
        self.model = None
        
        self._initialize_sam_model()
    
    def _check_gpu_availability(self):
        """Check if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.info("GPU not available, using CPU")
                return False
        except ImportError:
            logger.info("PyTorch not available, using CPU")
            return False
    
    def _initialize_sam_model(self):
        """Initialize CellposeSAM model using official API."""
        try:
            logger.info("Initializing CellposeSAM model (pretrained_model='cpsam')")
            
            # Use official API from Cellpose documentation
            self.model = models.CellposeModel(
                gpu=self.gpu,
                pretrained_model='cpsam',  # This is the SAM model
                model_type=None,  # Not needed for cpsam
                diam_mean=self.diam_mean,  # Mean diameter for normalization
                device=self.device,  # Device specification
                nchan=self.nchan  # Number of channels
            )
            
            logger.info("CellposeSAM model initialized successfully")
            
        except Exception as e:
            logger.error(f"CellposeSAM initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize CellposeSAM: {e}")
    
    def segment(self, image_array, diameter=None, flow_threshold=0.4, 
                cellprob_threshold=0.0, do_3D=False):
        """
        Run CellposeSAM segmentation using official API.
        
        Args:
            image_array: Input image array
            diameter: Cell diameter in pixels (None for auto-detection)
            flow_threshold: Flow error threshold (default 0.4)
            cellprob_threshold: Cell probability threshold (default 0.0)
            do_3D: Whether to perform 3D segmentation
            
        Returns:
            Tuple of (masks, flows, styles, diameters)
        """
        if not self.model:
            raise RuntimeError("CellposeSAM model not initialized")
        
        try:
            logger.info("Running CellposeSAM segmentation")
            
            # Use official eval API
            result = self.model.eval(
                x=image_array,
                do_3D=do_3D,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            # Handle different return formats between Cellpose versions
            if len(result) == 3:
                # Cellpose 4.0.4+ returns (masks, flows, diameters)
                masks, flows, diameters = result
                styles = None
            elif len(result) == 4:
                # Legacy format (masks, flows, styles, diameters)
                masks, flows, styles, diameters = result
            else:
                # Fallback
                masks = result[0]
                flows = result[1] if len(result) > 1 else None
                styles = None
                diameters = [diameter] if diameter else [30.0]
            
            # Log results
            num_detections = len(np.unique(masks)) - 1
            logger.info(f"CellposeSAM segmentation completed: {num_detections} cells detected")
            logger.info(f"Detected diameters: {diameters}")
            
            return masks, flows, styles, diameters
            
        except Exception as e:
            logger.error(f"CellposeSAM segmentation failed: {e}")
            raise RuntimeError(f"Segmentation failed: {e}")
    
    def get_model_info(self):
        """Get information about the CellposeSAM model."""
        return {
            'model_type': 'cpsam',
            'gpu_enabled': self.gpu,
            'device': self.device,
            'nchan': self.nchan,
            'diam_mean': self.diam_mean,
            'model_initialized': self.model is not None
        }
    
    def cleanup(self):
        """Clean up GPU memory if needed."""
        if self.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cleared")
            except ImportError:
                pass


def segment_with_cellpose_sam(image_array, diameter=None, flow_threshold=0.4, 
                             cellprob_threshold=0.0, gpu=True, do_3D=False):
    """
    High-level function for CellposeSAM segmentation.
    
    Args:
        image_array: Input image array
        diameter: Cell diameter in pixels (None for auto-detection)
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold
        gpu: Whether to use GPU
        do_3D: Whether to perform 3D segmentation
        
    Returns:
        Tuple of (masks, flows, styles, diameters)
    """
    segmenter = CellposeSAMSegmenter(gpu=gpu)
    
    try:
        return segmenter.segment(
            image_array=image_array,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D
        )
    finally:
        segmenter.cleanup()


def is_cellpose_sam_available():
    """Check if CellposeSAM is available."""
    return CELLPOSE_AVAILABLE


# Backward compatibility aliases
CellposeSAMModel = CellposeSAMSegmenter
segment_cellular_image = segment_with_cellpose_sam