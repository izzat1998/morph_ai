"""
Custom Exceptions for Morphometric Analysis

This module defines custom exception classes used throughout the morphometric
analysis pipeline to provide specific error handling and meaningful error messages.
"""


class MorphometricAnalysisError(Exception):
    """Base exception for all morphometric analysis errors"""
    pass


class ImageQualityError(MorphometricAnalysisError):
    """Exception raised for image quality assessment errors"""
    pass


class ImagePreprocessingError(MorphometricAnalysisError):
    """Exception raised during image preprocessing operations"""
    pass


class ParameterOptimizationError(MorphometricAnalysisError):
    """Exception raised during parameter optimization"""
    pass


class SegmentationError(MorphometricAnalysisError):
    """Exception raised during segmentation operations"""
    pass


class SegmentationRefinementError(MorphometricAnalysisError):
    """Exception raised during segmentation refinement operations"""
    pass


class ROIAnalysisError(MorphometricAnalysisError):
    """Exception raised during ROI (Region of Interest) analysis"""
    pass


class TextureAnalysisError(MorphometricAnalysisError):
    """Exception raised during texture analysis operations"""
    pass


class MorphometricValidationError(MorphometricAnalysisError):
    """Exception raised during morphometric validation"""
    pass


class VisualizationError(MorphometricAnalysisError):
    """Exception raised during visualization generation"""
    pass


class CellposeError(MorphometricAnalysisError):
    """Exception raised for Cellpose-specific errors"""
    pass


class DependencyError(MorphometricAnalysisError):
    """Exception raised when required dependencies are not available"""
    pass


class GPUMemoryError(MorphometricAnalysisError):
    """Exception raised for GPU memory-related errors"""
    pass


class PerformanceBenchmarkError(MorphometricAnalysisError):
    """Exception raised during performance benchmarking operations"""
    pass


class ConfigurationError(MorphometricAnalysisError):
    """Exception raised for configuration-related errors"""
    pass


class DataValidationError(MorphometricAnalysisError):
    """Exception raised for data validation errors"""
    pass