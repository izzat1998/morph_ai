"""
Enhanced Morphometric Analysis with Statistical Integration

This module provides OPTIONAL statistical enhancement to the existing analysis pipeline.
It does NOT modify existing functionality - it only ADDS statistical capabilities.

Design Principles:
- 100% backward compatible
- Existing code continues to work unchanged
- Statistical features are opt-in only
- All new data goes to separate statistical tables
"""

import logging
import time
from typing import Optional, Dict, Any, List
from django.db import transaction
from django.utils import timezone
from django.conf import settings

# Import existing analysis components (no changes to these)
from .analysis import CellAnalysisProcessor
from .models import CellAnalysis, DetectedCell

# Import statistical framework components
from morphometric_stats.models import StatisticalAnalysis, FeatureStatistics
from morphometric_stats.services.confidence_intervals import ConfidenceIntervalCalculator, CIMethod
from morphometric_stats.services.uncertainty_propagation import UncertaintyPropagationEngine
from morphometric_stats.services.morphometric_integration import MorphometricStatisticalIntegrator

logger = logging.getLogger(__name__)


class EnhancedMorphometricAnalysis:
    """
    OPTIONAL enhancement wrapper for existing morphometric analysis.
    
    This class provides statistical rigor without modifying existing functionality.
    Your existing code continues to work exactly as before.
    
    Usage:
        # Existing way (unchanged):
        processor = CellAnalysisProcessor(analysis_id)
        processor.run_analysis()
        
        # Enhanced way (optional):
        enhanced = EnhancedMorphometricAnalysis(analysis_id)
        enhanced.run_enhanced_analysis()  # Does everything + statistics
    """
    
    def __init__(self, analysis_id: int, enable_statistics: bool = True):
        """
        Initialize enhanced analysis.
        
        Args:
            analysis_id: CellAnalysis ID to process
            enable_statistics: Whether to add statistical analysis (default: True)
        """
        self.analysis_id = analysis_id
        self.enable_statistics = enable_statistics
        
        # Initialize the existing processor (no changes to existing code)
        self.processor = CellAnalysisProcessor(analysis_id)
        self.analysis = self.processor.analysis
        
        # Initialize statistical components only if enabled
        if self.enable_statistics:
            self.statistical_integrator = MorphometricStatisticalIntegrator(
                confidence_level=getattr(settings, 'DEFAULT_CONFIDENCE_LEVEL', 0.95),
                bootstrap_samples=getattr(settings, 'DEFAULT_BOOTSTRAP_SAMPLES', 2000)
            )
            self.uncertainty_engine = UncertaintyPropagationEngine()
        
        logger.info(f"EnhancedMorphometricAnalysis initialized for analysis {analysis_id} "
                   f"(statistics: {'enabled' if enable_statistics else 'disabled'})")
    
    def run_enhanced_analysis(self) -> bool:
        """
        Run the complete enhanced analysis pipeline.
        
        This method:
        1. Runs the existing analysis (unchanged)
        2. Optionally adds statistical analysis
        3. Returns success/failure status
        
        Returns:
            True if analysis completed successfully, False otherwise
        """
        try:
            logger.info(f"Starting enhanced analysis for {self.analysis.cell.name}")
            
            # Step 1: Run existing analysis (100% unchanged)
            logger.info("Running standard morphometric analysis...")
            success = self.processor.run_analysis()
            
            if not success:
                logger.error("Standard analysis failed - skipping statistical enhancement")
                return False
            
            logger.info("Standard analysis completed successfully")
            
            # Step 2: Add statistical enhancement if enabled
            if self.enable_statistics:
                logger.info("Adding statistical enhancement...")
                statistical_success = self._add_statistical_analysis()
                
                if not statistical_success:
                    logger.warning("Statistical analysis failed, but standard analysis succeeded")
                    # Don't fail the entire analysis - statistics are enhancement only
                
            logger.info("Enhanced analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {str(e)}")
            return False
    
    def _add_statistical_analysis(self) -> bool:
        """
        Add statistical analysis to completed morphometric analysis.
        This is completely separate from existing functionality.
        
        Returns:
            True if statistical analysis succeeded, False otherwise
        """
        try:
            # Verify we have detected cells from the standard analysis
            detected_cells = self.analysis.detected_cells.all()
            if not detected_cells.exists():
                logger.warning("No detected cells found - skipping statistical analysis")
                return False
            
            logger.info(f"Adding statistical analysis for {detected_cells.count()} detected cells")
            
            # Auto-calibrate uncertainty parameters based on image quality
            analysis_data = {
                'areas': [float(cell.area) for cell in detected_cells],
                'circularities': [float(cell.circularity) for cell in detected_cells]
            }
            self.uncertainty_engine.auto_calibrate_uncertainties(analysis_data)
            logger.info("Auto-calibrated uncertainty parameters based on image quality")
            
            with transaction.atomic():
                # Create or get statistical analysis record
                statistical_analysis, created = StatisticalAnalysis.objects.get_or_create(
                    analysis=self.analysis,
                    defaults={
                        'confidence_level': self.statistical_integrator.confidence_level,
                        'bootstrap_iterations': self.statistical_integrator.bootstrap_samples,
                        'include_confidence_intervals': True,
                        'include_uncertainty_propagation': True,
                        'include_bootstrap_analysis': True
                    }
                )
                
                start_time = timezone.now()
                
                # Process each detected cell with statistical analysis
                feature_stats_list = []
                processed_count = 0
                
                for detected_cell in detected_cells:
                    try:
                        cell_stats = self._calculate_cell_statistics(detected_cell, statistical_analysis)
                        feature_stats_list.extend(cell_stats)
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            logger.debug(f"Processed {processed_count}/{detected_cells.count()} cells")
                            
                    except Exception as cell_error:
                        logger.warning(f"Failed to process cell {detected_cell.cell_id}: {str(cell_error)}")
                        continue
                
                # Bulk create all feature statistics
                if feature_stats_list:
                    FeatureStatistics.objects.bulk_create(feature_stats_list)
                    logger.info(f"Created {len(feature_stats_list)} feature statistics records")
                
                # Update computation time
                computation_time = (timezone.now() - start_time).total_seconds()
                statistical_analysis.computation_time_seconds = computation_time
                statistical_analysis.save()
                
                logger.info(f"Statistical analysis completed in {computation_time:.2f}s "
                           f"for {processed_count} cells")
                
                return True
                
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            return False
    
    def _calculate_cell_statistics(self, detected_cell: DetectedCell, 
                                 statistical_analysis: StatisticalAnalysis) -> List[FeatureStatistics]:
        """
        Calculate statistical properties for a single detected cell.
        
        Args:
            detected_cell: DetectedCell instance from standard analysis
            statistical_analysis: StatisticalAnalysis container instance
            
        Returns:
            List of FeatureStatistics instances (not yet saved to DB)
        """
        feature_stats_list = []
        
        # Define morphometric features to analyze
        features_to_analyze = {
            'area': detected_cell.area,
            'perimeter': detected_cell.perimeter,
            'circularity': detected_cell.circularity,
            'eccentricity': detected_cell.eccentricity,
            'solidity': detected_cell.solidity,
            'extent': detected_cell.extent,
            'major_axis_length': detected_cell.major_axis_length,
            'minor_axis_length': detected_cell.minor_axis_length,
            'aspect_ratio': detected_cell.aspect_ratio
        }
        
        # Add physical measurements if available
        if detected_cell.area_microns_sq is not None:
            features_to_analyze['area_microns_sq'] = detected_cell.area_microns_sq
        if detected_cell.perimeter_microns is not None:
            features_to_analyze['perimeter_microns'] = detected_cell.perimeter_microns
        
        # Calculate uncertainty propagation for each feature
        for feature_name, measured_value in features_to_analyze.items():
            if measured_value is None or not isinstance(measured_value, (int, float)):
                continue
                
            try:
                # Calculate uncertainty based on feature type
                if feature_name == 'area' or feature_name == 'area_microns_sq':
                    uncertainty_result = self.uncertainty_engine.calculate_area_uncertainty(
                        area_pixels=detected_cell.area,
                        perimeter_pixels=detected_cell.perimeter
                    )
                elif feature_name == 'perimeter' or feature_name == 'perimeter_microns':
                    uncertainty_result = self.uncertainty_engine.calculate_perimeter_uncertainty(
                        perimeter_pixels=detected_cell.perimeter
                    )
                elif feature_name == 'circularity':
                    input_uncertainties = {
                        'area': (detected_cell.area, detected_cell.area * 0.05),  # 5% assumed uncertainty
                        'perimeter': (detected_cell.perimeter, detected_cell.perimeter * 0.03)  # 3% assumed uncertainty
                    }
                    uncertainty_result = self.uncertainty_engine.calculate_derived_feature_uncertainty(
                        feature_name, measured_value, input_uncertainties
                    )
                elif feature_name == 'aspect_ratio':
                    input_uncertainties = {
                        'major_axis': (detected_cell.major_axis_length, detected_cell.major_axis_length * 0.02),
                        'minor_axis': (detected_cell.minor_axis_length, detected_cell.minor_axis_length * 0.02)
                    }
                    uncertainty_result = self.uncertainty_engine.calculate_derived_feature_uncertainty(
                        feature_name, measured_value, input_uncertainties
                    )
                else:
                    # Generic uncertainty for other features
                    input_uncertainties = {
                        'value': (measured_value, measured_value * 0.02)  # 2% assumed uncertainty
                    }
                    uncertainty_result = self.uncertainty_engine.calculate_derived_feature_uncertainty(
                        feature_name, measured_value, input_uncertainties
                    )
                
                # Create FeatureStatistics instance
                feature_stat = self.uncertainty_engine.create_feature_statistics(
                    statistical_analysis=statistical_analysis,
                    detected_cell=detected_cell,
                    feature_name=feature_name,
                    measured_value=measured_value,
                    propagation_result=uncertainty_result
                )
                
                feature_stats_list.append(feature_stat)
                
            except Exception as feature_error:
                logger.warning(f"Failed to calculate statistics for {feature_name}: {str(feature_error)}")
                continue
        
        return feature_stats_list
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary including statistical data.
        
        Returns:
            Dictionary with complete analysis information
        """
        summary = {
            'analysis_id': self.analysis.id,
            'cell_name': self.analysis.cell.name,
            'status': self.analysis.status,
            'standard_analysis': self._get_standard_summary(),
            'statistical_analysis': None
        }
        
        if self.enable_statistics:
            try:
                statistical_analysis = StatisticalAnalysis.objects.filter(
                    analysis=self.analysis
                ).first()
                
                if statistical_analysis:
                    summary['statistical_analysis'] = self.statistical_integrator.get_statistical_summary(
                        statistical_analysis
                    )
            except Exception as e:
                logger.warning(f"Failed to get statistical summary: {str(e)}")
                summary['statistical_analysis'] = {'error': str(e)}
        
        return summary
    
    def _get_standard_summary(self) -> Dict[str, Any]:
        """Get summary of standard analysis results."""
        detected_cells = self.analysis.detected_cells.all()
        
        if not detected_cells.exists():
            return {'cells_detected': 0}
        
        # Calculate basic statistics from detected cells
        areas = [cell.area for cell in detected_cells if cell.area is not None]
        circularities = [cell.circularity for cell in detected_cells if cell.circularity is not None]
        
        summary = {
            'cells_detected': detected_cells.count(),
            'processing_time': self.analysis.processing_time,
            'model_used': self.analysis.cellpose_model,
            'diameter': self.analysis.cellpose_diameter
        }
        
        if areas:
            import numpy as np
            summary['area_stats'] = {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas))
            }
        
        if circularities:
            import numpy as np
            summary['circularity_stats'] = {
                'mean': float(np.mean(circularities)),
                'std': float(np.std(circularities))
            }
        
        return summary


def run_enhanced_analysis(analysis_id: int, enable_statistics: bool = True) -> bool:
    """
    Convenience function to run enhanced analysis.
    
    This provides a simple interface similar to existing run_cell_analysis.
    
    Args:
        analysis_id: CellAnalysis ID to process
        enable_statistics: Whether to include statistical analysis
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    try:
        enhanced = EnhancedMorphometricAnalysis(analysis_id, enable_statistics)
        return enhanced.run_enhanced_analysis()
    except Exception as e:
        logger.error(f"Enhanced analysis failed for ID {analysis_id}: {str(e)}")
        return False


# Backward compatibility - existing code continues to work
from .utils import run_cell_analysis as _original_run_cell_analysis

def run_cell_analysis_with_stats(analysis_id: int) -> bool:
    """
    Drop-in replacement for run_cell_analysis that adds statistics.
    
    Your existing code can use this by simply changing the import:
    
    # Old way:
    from cells.utils import run_cell_analysis
    
    # New way (same interface):
    from cells.enhanced_analysis import run_cell_analysis_with_stats as run_cell_analysis
    """
    return run_enhanced_analysis(analysis_id, enable_statistics=True)