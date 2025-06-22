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

# Import refactored modules
from .quality_assessment import ImageQualityAssessment
from .image_preprocessing import ImagePreprocessor
from .parameter_optimization import ParameterOptimizer
from .segmentation_refinement import SegmentationRefinement
from .utils import run_cell_analysis, get_image_quality_summary, get_analysis_summary
from .gpu_utils import gpu_manager, log_gpu_status, cleanup_gpu_memory
from .gpu_memory_manager import memory_manager, gpu_memory_context
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
            
            # Start memory monitoring for this analysis
            memory_manager.start_monitoring()
            
            # Use managed memory context for GPU operations
            with gpu_memory_context(reserve_mb=1000):
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
                
                # Log memory status after analysis
                memory_status = memory_manager.get_status()
                logger.info(f"Analysis completed successfully in {processing_time:.2f} seconds")
                logger.info(f"Final GPU memory usage: {memory_status['current_memory']['usage_ratio']:.1%}")
                
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
        
        # Auto-optimize parameters if diameter is 0 (auto-detection requested)
        if self.analysis.cellpose_diameter == 0:
            current_params = {
                'cellpose_model': self.analysis.cellpose_model,
                'flow_threshold': self.analysis.flow_threshold,
                'cellprob_threshold': self.analysis.cellprob_threshold
            }
            
            # Use our new parameter optimization module
            optimized_params = ParameterOptimizer.optimize_all_parameters(
                image_array, quality_assessment, current_params
            )
            
            # Update analysis with optimized parameters
            self.analysis.cellpose_diameter = optimized_params['cellpose_diameter']
            self.analysis.flow_threshold = optimized_params['flow_threshold'] 
            self.analysis.cellprob_threshold = optimized_params['cellprob_threshold']
            self.analysis.cellpose_model = optimized_params['cellpose_model']
            
            # Store optimization info
            self.analysis.quality_metrics['parameter_optimization'] = optimized_params
            
            logger.info(f"Auto-optimized parameters: diameter={optimized_params['cellpose_diameter']:.1f}, "
                       f"model={optimized_params['cellpose_model']}")
        
        self.analysis.save()
        return image_array
        
    def _run_cellpose_segmentation(self, image_array):
        """Run Cellpose segmentation with GPU acceleration and fallback."""
        try:
            # Log GPU status for debugging
            log_gpu_status()
            
            # Detect GPU capabilities
            gpu_info = gpu_manager.detect_gpu_capabilities()
            use_gpu = getattr(settings, 'CELLPOSE_USE_GPU', False) and gpu_info.backend != 'cpu'
            
            logger.info(f"Cellpose segmentation starting - GPU: {use_gpu} ({gpu_info.backend} backend)")
            
            # Initialize Cellpose model with GPU settings
            try:
                model = models.CellposeModel(
                    model_type=self.analysis.cellpose_model,
                    gpu=use_gpu
                )
                logger.info(f"Cellpose model initialized: {self.analysis.cellpose_model} (GPU: {use_gpu})")
            except Exception as model_error:
                logger.warning(f"Failed to initialize GPU model, falling back to CPU: {str(model_error)}")
                # Fallback to CPU
                model = models.CellposeModel(
                    model_type=self.analysis.cellpose_model,
                    gpu=False
                )
                use_gpu = False
            
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
            
            # Run segmentation (Cellpose v4.0.4+ returns masks, flows, prob)
            result = model.eval(
                image_array,
                diameter=self.analysis.cellpose_diameter,
                flow_threshold=self.analysis.flow_threshold,
                cellprob_threshold=self.analysis.cellprob_threshold,
                channels=channels
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
            logger.info(f"Cellpose segmentation completed: {num_detections} detections using {backend_used}")
            
            # Clean up GPU memory if used
            if use_gpu:
                try:
                    cleanup_gpu_memory()
                    logger.debug("GPU memory cleaned up after segmentation")
                except Exception as cleanup_error:
                    logger.warning(f"GPU memory cleanup failed: {str(cleanup_error)}")
            
            return masks, flows, styles, diameters
            
        except Exception as e:
            logger.error(f"Cellpose segmentation failed: {str(e)}")
            # Clean up GPU memory on error
            try:
                cleanup_gpu_memory()
            except:
                pass
            raise SegmentationError(f"Segmentation failed: {str(e)}")
    
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
                            # Some Cellpose versions return 2D flow data - reshape if needed
                            h, w = masks.shape
                            if xy_flows.shape[0] == h * w and xy_flows.shape[1] >= 2:
                                # Reshape to 3D format
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
                                raise ValueError(f"Cannot reshape 2D flow data: {xy_flows.shape} for mask shape {masks.shape}")
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
                # Panel 6: Cell Centers/Poses
                if len(original_image.shape) == 3:
                    base_img = np.mean(original_image, axis=2)
                else:
                    base_img = original_image
                axes[ax_idx].imshow(base_img, cmap='gray', alpha=0.7)
                
                # Plot cell centers
                from skimage import measure
                props = measure.regionprops(masks)
                for prop in props:
                    if prop.label > 0:
                        y, x = prop.centroid
                        axes[ax_idx].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
                        axes[ax_idx].text(x+10, y, str(prop.label), fontsize=8, color='red', fontweight='bold')
                
                axes[ax_idx].set_title('6. Cell Centers', fontsize=12, fontweight='bold')
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
                        logger.warning(f"Flow data is None or has no shape attribute: {type(xy_flows)}")
                else:
                    logger.warning(f"Flow data structure unexpected: {type(flows[0]) if flows else 'None'}")
            else:
                logger.warning(f"No flows data provided: {type(flows)}")
            
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
        """Extract morphometric features from segmented cells with GPU acceleration."""
        try:
            # Clear existing detected cells
            self.analysis.detected_cells.all().delete()
            
            # Try GPU-accelerated morphometrics first
            use_gpu_morphometrics = getattr(settings, 'ENABLE_GPU_PREPROCESSING', False)
            
            if use_gpu_morphometrics:
                try:
                    from .gpu_morphometrics import calculate_morphometrics_gpu, is_gpu_morphometrics_available
                    
                    if is_gpu_morphometrics_available():
                        logger.info("Using GPU-accelerated morphometric calculations")
                        gpu_start_time = time.time()
                        
                        # Calculate all features using GPU
                        gpu_results = calculate_morphometrics_gpu(masks)
                        areas = gpu_results['areas']
                        perimeters = gpu_results['perimeters']
                        centroids = gpu_results['centroids']
                        shape_descriptors = gpu_results['shape_descriptors']
                        
                        gpu_time = time.time() - gpu_start_time
                        logger.info(f"GPU morphometric calculations completed in {gpu_time:.3f}s")
                        
                        # Create DetectedCell instances from GPU results
                        for cell_id in areas:
                            descriptors = shape_descriptors.get(cell_id, {})
                            centroid = centroids.get(cell_id, (0, 0))
                            
                            # Get bounding box from CPU regionprops for complex shapes
                            try:
                                cell_mask = (masks == cell_id).astype(np.uint8)
                                props = measure.regionprops(cell_mask)
                                if props:
                                    bbox = props[0].bbox
                                    bounding_box_y, bounding_box_x = bbox[0], bbox[1]
                                    bounding_box_height, bounding_box_width = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                else:
                                    bounding_box_x = bounding_box_y = 0
                                    bounding_box_width = bounding_box_height = 1
                            except:
                                bounding_box_x = bounding_box_y = 0
                                bounding_box_width = bounding_box_height = 1
                            
                            detected_cell = DetectedCell(
                                analysis=self.analysis,
                                cell_id=cell_id,
                                area=descriptors.get('area', areas[cell_id]),
                                perimeter=descriptors.get('perimeter', perimeters.get(cell_id, 0)),
                                circularity=descriptors.get('circularity', 0),
                                eccentricity=descriptors.get('eccentricity', 0),
                                solidity=descriptors.get('solidity', 1.0),
                                extent=descriptors.get('extent', 1.0),
                                major_axis_length=descriptors.get('major_axis_length', 0),
                                minor_axis_length=descriptors.get('minor_axis_length', 0),
                                aspect_ratio=descriptors.get('aspect_ratio', 1.0),
                                centroid_x=centroid[1],
                                centroid_y=centroid[0],
                                bounding_box_x=bounding_box_x,
                                bounding_box_y=bounding_box_y,
                                bounding_box_width=bounding_box_width,
                                bounding_box_height=bounding_box_height
                            )
                            
                            # Calculate physical measurements if scale is available
                            if self.analysis.cell.scale_set and self.analysis.cell.pixels_per_micron:
                                ppm = self.analysis.cell.pixels_per_micron
                                detected_cell.area_microns_sq = detected_cell.area / (ppm ** 2)
                                detected_cell.perimeter_microns = detected_cell.perimeter / ppm
                                detected_cell.major_axis_length_microns = detected_cell.major_axis_length / ppm
                                detected_cell.minor_axis_length_microns = detected_cell.minor_axis_length / ppm
                            
                            detected_cell.save()
                        
                        num_cells = len(areas)
                        logger.info(f"GPU morphometric feature extraction completed for {num_cells} cells")
                        return
                        
                    else:
                        logger.info("GPU morphometrics not available, falling back to CPU")
                        
                except Exception as gpu_error:
                    logger.warning(f"GPU morphometric calculation failed: {str(gpu_error)}")
                    logger.info("Falling back to CPU morphometric calculations")
            
            # CPU fallback - original implementation
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