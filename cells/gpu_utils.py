"""
GPU Utilities and Hardware Detection Module

This module provides utilities for GPU detection, validation, and management
for the morphometric analysis pipeline.
"""

import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Container for GPU information."""
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    cuda_capability: Optional[Tuple[int, int]]
    driver_version: Optional[str]
    is_available: bool
    backend: str  # 'cuda', 'opencl', 'cpu'


class GPUManager:
    """
    GPU detection and management utilities.
    
    Handles detection of available GPU hardware, memory management,
    and provides fallback strategies for CPU-only systems.
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self._gpu_info = None
        self._backends_checked = False
        self._available_backends = []
        logger.info("GPUManager initialized")
    
    def detect_gpu_capabilities(self) -> GPUInfo:
        """
        Detect and return comprehensive GPU capabilities.
        
        Returns:
            GPUInfo object with detailed GPU information
        """
        if self._gpu_info is not None:
            return self._gpu_info
        
        logger.info("Detecting GPU capabilities...")
        
        # Try different GPU backends in order of preference
        backends_to_try = [
            ('cuda', self._detect_cuda_gpu),
            ('opencl', self._detect_opencl_gpu),
            ('cpu', self._detect_cpu_fallback)
        ]
        
        for backend_name, detector_func in backends_to_try:
            try:
                gpu_info = detector_func()
                if gpu_info.is_available:
                    self._gpu_info = gpu_info
                    logger.info(f"GPU detection successful: {backend_name} backend, {gpu_info.name}")
                    return gpu_info
            except Exception as e:
                logger.warning(f"Failed to detect {backend_name} GPU: {str(e)}")
                continue
        
        # Fallback to CPU
        self._gpu_info = self._detect_cpu_fallback()
        logger.info("No GPU detected, falling back to CPU processing")
        return self._gpu_info
    
    def _detect_cuda_gpu(self) -> GPUInfo:
        """Detect CUDA-capable GPU."""
        try:
            # Try PyTorch CUDA detection first
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        # Get info for the first GPU
                        device_props = torch.cuda.get_device_properties(0)
                        memory_total = device_props.total_memory // (1024 * 1024)  # Convert to MB
                        memory_available = memory_total - (torch.cuda.memory_allocated(0) // (1024 * 1024))
                        
                        return GPUInfo(
                            name=device_props.name,
                            memory_total=memory_total,
                            memory_available=memory_available,
                            cuda_capability=(device_props.major, device_props.minor),
                            driver_version=torch.version.cuda,
                            is_available=True,
                            backend='cuda'
                        )
            except ImportError:
                logger.debug("PyTorch not available, trying alternative CUDA detection")
            
            # Try CuPy detection
            try:
                import cupy
                # Check if CUDA is actually available
                with cupy.cuda.Device(0):
                    meminfo = cupy.cuda.runtime.memGetInfo()
                    memory_available = meminfo[0] // (1024 * 1024)  # Convert to MB
                    memory_total = meminfo[1] // (1024 * 1024)
                    
                    # Get device properties
                    device_props = cupy.cuda.runtime.getDeviceProperties(0)
                    device_name = f"GPU Device {cupy.cuda.runtime.getDevice()}"
                    
                    return GPUInfo(
                        name=device_name,
                        memory_total=memory_total,
                        memory_available=memory_available,
                        cuda_capability=(device_props['major'], device_props['minor']),
                        driver_version=str(cupy.cuda.runtime.runtimeGetVersion()),
                        is_available=True,
                        backend='cuda'
                    )
            except ImportError:
                logger.debug("CuPy not available")
            except Exception as e:
                logger.debug(f"CuPy CUDA detection failed: {str(e)}")
            
            # Try nvidia-smi command
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        parts = lines[0].split(', ')
                        if len(parts) >= 4:
                            return GPUInfo(
                                name=parts[0],
                                memory_total=int(parts[1]),
                                memory_available=int(parts[2]),
                                cuda_capability=None,
                                driver_version=parts[3],
                                is_available=True,
                                backend='cuda'
                            )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                logger.debug("nvidia-smi command not available or failed")
            
        except Exception as e:
            logger.error(f"CUDA GPU detection failed: {str(e)}")
        
        raise RuntimeError("No CUDA GPU detected")
    
    def _detect_opencl_gpu(self) -> GPUInfo:
        """Detect OpenCL-capable GPU."""
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    device = devices[0]  # Use first GPU
                    
                    return GPUInfo(
                        name=device.name,
                        memory_total=device.global_mem_size // (1024 * 1024),
                        memory_available=device.global_mem_size // (1024 * 1024),  # Approximation
                        cuda_capability=None,
                        driver_version=device.driver_version,
                        is_available=True,
                        backend='opencl'
                    )
        except ImportError:
            logger.debug("PyOpenCL not available")
        except Exception as e:
            logger.debug(f"OpenCL GPU detection failed: {str(e)}")
        
        raise RuntimeError("No OpenCL GPU detected")
    
    def _detect_cpu_fallback(self) -> GPUInfo:
        """Create CPU fallback info."""
        # Get system memory info
        memory = psutil.virtual_memory()
        available_memory = memory.available // (1024 * 1024)  # Convert to MB
        total_memory = memory.total // (1024 * 1024)
        
        return GPUInfo(
            name=f"CPU ({psutil.cpu_count()} cores)",
            memory_total=total_memory,
            memory_available=available_memory,
            cuda_capability=None,
            driver_version=None,
            is_available=True,  # CPU is always available
            backend='cpu'
        )
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available for processing.
        
        Returns:
            True if GPU is available and functional
        """
        gpu_info = self.detect_gpu_capabilities()
        return gpu_info.is_available and gpu_info.backend != 'cpu'
    
    def get_optimal_batch_size(self, image_size: Tuple[int, int], base_batch_size: int = 4) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            image_size: (height, width) of input images
            base_batch_size: Base batch size to start calculations from
            
        Returns:
            Recommended batch size
        """
        gpu_info = self.detect_gpu_capabilities()
        
        if gpu_info.backend == 'cpu':
            # For CPU, limit batch size to avoid memory issues
            return min(base_batch_size, 2)
        
        # Estimate memory usage per image (rough approximation)
        height, width = image_size
        channels = 3  # Assume RGB
        
        # Memory per image in MB (including intermediate buffers)
        memory_per_image = (height * width * channels * 4) / (1024 * 1024)  # 4 bytes per float32
        memory_per_image *= 3  # Account for intermediate processing buffers
        
        # Use 80% of available memory
        usable_memory = gpu_info.memory_available * 0.8
        
        if memory_per_image > 0:
            optimal_batch_size = max(1, int(usable_memory / memory_per_image))
            return min(optimal_batch_size, base_batch_size * 2)  # Cap at 2x base
        
        return base_batch_size
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        gpu_info = self.detect_gpu_capabilities()
        
        memory_info = {
            'backend': gpu_info.backend,
            'total_mb': gpu_info.memory_total,
            'available_mb': gpu_info.memory_available,
            'device_name': gpu_info.name
        }
        
        if gpu_info.backend == 'cuda':
            try:
                # Try to get current memory usage
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
                    cached = torch.cuda.memory_reserved(0) // (1024 * 1024)
                    memory_info.update({
                        'allocated_mb': allocated,
                        'cached_mb': cached,
                        'free_mb': gpu_info.memory_total - allocated
                    })
            except ImportError:
                try:
                    import cupy
                    meminfo = cupy.cuda.runtime.memGetInfo()
                    free_memory = meminfo[0] // (1024 * 1024)
                    memory_info.update({
                        'free_mb': free_memory,
                        'used_mb': gpu_info.memory_total - free_memory
                    })
                except ImportError:
                    pass
        
        return memory_info
    
    def validate_gpu_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate GPU setup and return status with any issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            gpu_info = self.detect_gpu_capabilities()
            
            if gpu_info.backend == 'cpu':
                issues.append("No GPU detected - will use CPU processing (slower)")
                return True, issues  # CPU fallback is valid
            
            # Check memory availability
            if gpu_info.memory_available < 1024:  # Less than 1GB
                issues.append(f"Low GPU memory: {gpu_info.memory_available}MB available")
            
            # Check CUDA capability for deep learning
            if gpu_info.backend == 'cuda':
                if gpu_info.cuda_capability and gpu_info.cuda_capability[0] < 3:
                    issues.append(f"Old GPU architecture: {gpu_info.cuda_capability} (may have limited support)")
                
                # Test if we can actually use the GPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Try to allocate a small tensor
                        test_tensor = torch.randn(100, 100).cuda()
                        del test_tensor
                        torch.cuda.empty_cache()
                except Exception as e:
                    issues.append(f"GPU allocation test failed: {str(e)}")
                    return False, issues
            
            return True, issues
            
        except Exception as e:
            issues.append(f"GPU validation failed: {str(e)}")
            return False, issues
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory caches."""
        gpu_info = self.detect_gpu_capabilities()
        
        if gpu_info.backend == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("PyTorch GPU memory cache cleared")
            except ImportError:
                pass
            
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                logger.debug("CuPy GPU memory cache cleared")
            except ImportError:
                pass


# Global GPU manager instance
gpu_manager = GPUManager()


def get_gpu_info() -> GPUInfo:
    """Get GPU information using the global manager."""
    return gpu_manager.detect_gpu_capabilities()


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return gpu_manager.is_gpu_available()


def get_optimal_batch_size(image_size: Tuple[int, int], base_batch_size: int = 4) -> int:
    """Get optimal batch size for the given image size."""
    return gpu_manager.get_optimal_batch_size(image_size, base_batch_size)


def validate_gpu_setup() -> Tuple[bool, List[str]]:
    """Validate GPU setup."""
    return gpu_manager.validate_gpu_setup()


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    gpu_manager.cleanup_gpu_memory()


def log_gpu_status():
    """Log current GPU status for debugging."""
    try:
        gpu_info = get_gpu_info()
        memory_info = gpu_manager.get_memory_info()
        
        logger.info(f"GPU Status: {gpu_info.backend.upper()} backend")
        logger.info(f"Device: {gpu_info.name}")
        logger.info(f"Memory: {memory_info['available_mb']}MB available / {memory_info['total_mb']}MB total")
        
        if gpu_info.cuda_capability:
            logger.info(f"CUDA Capability: {gpu_info.cuda_capability[0]}.{gpu_info.cuda_capability[1]}")
        
        is_valid, issues = validate_gpu_setup()
        if issues:
            logger.warning(f"GPU validation issues: {'; '.join(issues)}")
        else:
            logger.info("GPU setup validation passed")
            
    except Exception as e:
        logger.error(f"Failed to log GPU status: {str(e)}")