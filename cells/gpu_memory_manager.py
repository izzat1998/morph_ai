"""
GPU Memory Management and Monitoring Module

This module provides comprehensive GPU memory management, monitoring,
and optimization for the morphometric analysis pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import threading
import gc
from dataclasses import dataclass
from contextlib import contextmanager

from .gpu_utils import gpu_manager
from .exceptions import GPUMemoryError, DependencyError

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Container for GPU memory snapshot."""
    timestamp: float
    total_mb: int
    allocated_mb: int
    cached_mb: int
    free_mb: int
    fragmentation_ratio: float
    backend: str


class GPUMemoryMonitor:
    """
    GPU memory monitoring and alerting system.
    
    Tracks GPU memory usage, detects memory leaks, and provides
    alerts when memory usage exceeds thresholds.
    """
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Memory usage ratio to trigger warnings
            critical_threshold: Memory usage ratio to trigger critical alerts
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.gpu_info = gpu_manager.detect_gpu_capabilities()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        
        # Memory tracking
        self.snapshots = []
        self.max_snapshots = 100
        self.alerts = []
        
        # Backend-specific initialization
        self.torch_available = False
        self.cupy_available = False
        
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.torch_available = True
        except ImportError:
            pass
        
        try:
            import cupy as cp
            if self.gpu_info.backend == 'cuda':
                self.cp = cp
                self.cupy_available = True
        except ImportError:
            pass
        
        logger.info(f"GPU Memory Monitor initialized - backend: {self.gpu_info.backend}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information."""
        try:
            if self.torch_available:
                return self._get_torch_memory_info()
            elif self.cupy_available:
                return self._get_cupy_memory_info()
            else:
                return self._get_fallback_memory_info()
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {str(e)}")
            return self._get_fallback_memory_info()
    
    def _get_torch_memory_info(self) -> Dict[str, Any]:
        """Get memory info using PyTorch."""
        device = 0  # Use first GPU
        
        allocated = self.torch.cuda.memory_allocated(device) // (1024 * 1024)
        cached = self.torch.cuda.memory_reserved(device) // (1024 * 1024)
        total = self.torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
        free = total - allocated
        
        return {
            'backend': 'torch',
            'total_mb': total,
            'allocated_mb': allocated,
            'cached_mb': cached,
            'free_mb': free,
            'usage_ratio': allocated / total if total > 0 else 0,
            'fragmentation_ratio': (cached - allocated) / total if total > 0 else 0
        }
    
    def _get_cupy_memory_info(self) -> Dict[str, Any]:
        """Get memory info using CuPy."""
        try:
            meminfo = self.cp.cuda.runtime.memGetInfo()
            free_bytes = meminfo[0]
            total_bytes = meminfo[1]
            
            allocated_bytes = total_bytes - free_bytes
            
            total_mb = total_bytes // (1024 * 1024)
            allocated_mb = allocated_bytes // (1024 * 1024)
            free_mb = free_bytes // (1024 * 1024)
            
            # Get memory pool info for cached memory
            try:
                pool = self.cp.get_default_memory_pool()
                cached_mb = pool.total_bytes() // (1024 * 1024)
            except:
                cached_mb = 0
            
            return {
                'backend': 'cupy',
                'total_mb': total_mb,
                'allocated_mb': allocated_mb,
                'cached_mb': cached_mb,
                'free_mb': free_mb,
                'usage_ratio': allocated_mb / total_mb if total_mb > 0 else 0,
                'fragmentation_ratio': cached_mb / total_mb if total_mb > 0 else 0
            }
        except Exception as e:
            logger.error(f"CuPy memory info failed: {str(e)}")
            return self._get_fallback_memory_info()
    
    def _get_fallback_memory_info(self) -> Dict[str, Any]:
        """Fallback memory info."""
        return {
            'backend': 'cpu',
            'total_mb': self.gpu_info.memory_total,
            'allocated_mb': 0,
            'cached_mb': 0,
            'free_mb': self.gpu_info.memory_available,
            'usage_ratio': 0.0,
            'fragmentation_ratio': 0.0
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        info = self.get_memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_mb=info['total_mb'],
            allocated_mb=info['allocated_mb'],
            cached_mb=info['cached_mb'],
            free_mb=info['free_mb'],
            fragmentation_ratio=info['fragmentation_ratio'],
            backend=info['backend']
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        # Check for alerts
        self._check_memory_alerts(snapshot)
        
        return snapshot
    
    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check for memory usage alerts."""
        usage_ratio = snapshot.allocated_mb / snapshot.total_mb if snapshot.total_mb > 0 else 0
        
        alert = None
        if usage_ratio >= self.critical_threshold:
            alert = {
                'level': 'critical',
                'message': f"Critical GPU memory usage: {usage_ratio:.1%} ({snapshot.allocated_mb}MB/{snapshot.total_mb}MB)",
                'timestamp': snapshot.timestamp,
                'usage_ratio': usage_ratio
            }
        elif usage_ratio >= self.warning_threshold:
            alert = {
                'level': 'warning',
                'message': f"High GPU memory usage: {usage_ratio:.1%} ({snapshot.allocated_mb}MB/{snapshot.total_mb}MB)",
                'timestamp': snapshot.timestamp,
                'usage_ratio': usage_ratio
            }
        
        if alert:
            self.alerts.append(alert)
            logger.warning(alert['message'])
            
            # Keep only recent alerts
            cutoff_time = time.time() - 3600  # 1 hour
            self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("GPU memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                self.take_snapshot()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
                time.sleep(self.monitor_interval)
    
    def get_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        if len(self.snapshots) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
        
        # Calculate trends
        usage_ratios = [s.allocated_mb / s.total_mb for s in recent_snapshots if s.total_mb > 0]
        fragmentation_ratios = [s.fragmentation_ratio for s in recent_snapshots]
        
        if usage_ratios:
            avg_usage = np.mean(usage_ratios)
            max_usage = np.max(usage_ratios)
            usage_trend = 'increasing' if usage_ratios[-1] > usage_ratios[0] else 'decreasing'
        else:
            avg_usage = max_usage = 0.0
            usage_trend = 'stable'
        
        return {
            'avg_usage_ratio': avg_usage,
            'max_usage_ratio': max_usage,
            'usage_trend': usage_trend,
            'avg_fragmentation': np.mean(fragmentation_ratios) if fragmentation_ratios else 0.0,
            'snapshots_count': len(self.snapshots),
            'recent_alerts': len([a for a in self.alerts if a['timestamp'] > time.time() - 300])  # Last 5 min
        }


class GPUMemoryManager:
    """
    Advanced GPU memory management with automatic optimization.
    
    Provides memory pool management, automatic cleanup, and optimization
    strategies for GPU memory usage.
    """
    
    def __init__(self):
        """Initialize GPU memory manager."""
        self.monitor = GPUMemoryMonitor()
        self.gpu_info = gpu_manager.detect_gpu_capabilities()
        
        # Memory management settings
        self.auto_cleanup_threshold = 0.9  # Auto cleanup at 90% usage
        self.aggressive_cleanup_threshold = 0.95  # Aggressive cleanup at 95%
        
        # Backend-specific initialization
        self.torch_available = self.monitor.torch_available
        self.cupy_available = self.monitor.cupy_available
        
        if self.torch_available:
            self.torch = self.monitor.torch
        if self.cupy_available:
            self.cp = self.monitor.cp
        
        logger.info("GPU Memory Manager initialized")
    
    @contextmanager
    def managed_memory(self, reserve_mb: int = 500):
        """
        Context manager for managed GPU memory operations.
        
        Args:
            reserve_mb: Memory to reserve for operations
        """
        initial_snapshot = self.monitor.take_snapshot()
        
        try:
            # Check if we have enough memory
            if initial_snapshot.free_mb < reserve_mb:
                self.cleanup_memory()
                
                # Check again after cleanup
                snapshot = self.monitor.take_snapshot()
                if snapshot.free_mb < reserve_mb:
                    raise GPUMemoryError(
                        f"Insufficient GPU memory: need {reserve_mb}MB, "
                        f"have {snapshot.free_mb}MB after cleanup"
                    )
            
            yield
            
        finally:
            # Cleanup after operation
            self.cleanup_memory()
            
            final_snapshot = self.monitor.take_snapshot()
            logger.debug(
                f"Memory operation completed: "
                f"{initial_snapshot.allocated_mb}MB -> {final_snapshot.allocated_mb}MB"
            )
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Clean up GPU memory.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        logger.debug(f"Cleaning up GPU memory (aggressive: {aggressive})")
        
        # Python garbage collection first
        gc.collect()
        
        if self.torch_available:
            self._cleanup_torch_memory(aggressive)
        
        if self.cupy_available:
            self._cleanup_cupy_memory(aggressive)
        
        # Force another garbage collection
        gc.collect()
    
    def _cleanup_torch_memory(self, aggressive: bool = False):
        """Clean up PyTorch GPU memory."""
        try:
            # Empty cache
            self.torch.cuda.empty_cache()
            
            if aggressive:
                # Synchronize all devices
                self.torch.cuda.synchronize()
                
                # Try to reset memory stats
                try:
                    self.torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            
            logger.debug("PyTorch GPU memory cleaned up")
            
        except Exception as e:
            logger.warning(f"PyTorch memory cleanup failed: {str(e)}")
    
    def _cleanup_cupy_memory(self, aggressive: bool = False):
        """Clean up CuPy GPU memory."""
        try:
            # Free memory pool
            pool = self.cp.get_default_memory_pool()
            pool.free_all_blocks()
            
            if aggressive:
                # Also free pinned memory
                pinned_pool = self.cp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
            
            logger.debug("CuPy GPU memory cleaned up")
            
        except Exception as e:
            logger.warning(f"CuPy memory cleanup failed: {str(e)}")
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage based on current state."""
        snapshot = self.monitor.take_snapshot()
        usage_ratio = snapshot.allocated_mb / snapshot.total_mb if snapshot.total_mb > 0 else 0
        
        if usage_ratio >= self.aggressive_cleanup_threshold:
            logger.warning(f"High memory usage detected ({usage_ratio:.1%}), performing aggressive cleanup")
            self.cleanup_memory(aggressive=True)
        elif usage_ratio >= self.auto_cleanup_threshold:
            logger.info(f"Moderate memory usage detected ({usage_ratio:.1%}), performing cleanup")
            self.cleanup_memory(aggressive=False)
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        snapshot = self.monitor.take_snapshot()
        trends = self.monitor.get_memory_trends()
        recommendations = []
        
        usage_ratio = snapshot.allocated_mb / snapshot.total_mb if snapshot.total_mb > 0 else 0
        
        if usage_ratio > 0.9:
            recommendations.append("Consider reducing batch size to lower memory usage")
            recommendations.append("Enable automatic memory cleanup")
        
        if trends['avg_fragmentation'] > 0.3:
            recommendations.append("High memory fragmentation detected - consider memory pool optimization")
        
        if trends['usage_trend'] == 'increasing' and trends['recent_alerts'] > 0:
            recommendations.append("Memory usage is increasing - check for memory leaks")
        
        if self.gpu_info.memory_total < 4096:  # Less than 4GB
            recommendations.append("Limited GPU memory - consider using CPU fallback for large operations")
        
        return recommendations
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory management status."""
        snapshot = self.monitor.take_snapshot()
        trends = self.monitor.get_memory_trends()
        
        return {
            'current_memory': {
                'total_mb': snapshot.total_mb,
                'allocated_mb': snapshot.allocated_mb,
                'free_mb': snapshot.free_mb,
                'usage_ratio': snapshot.allocated_mb / snapshot.total_mb if snapshot.total_mb > 0 else 0
            },
            'trends': trends,
            'recommendations': self.get_memory_recommendations(),
            'backend': snapshot.backend,
            'monitoring_active': self.monitor.monitoring
        }


# Global memory manager instance
memory_manager = GPUMemoryManager()


@contextmanager
def gpu_memory_context(reserve_mb: int = 500):
    """
    Context manager for GPU memory management.
    
    Args:
        reserve_mb: Memory to reserve for operations
    """
    with memory_manager.managed_memory(reserve_mb):
        yield


def cleanup_gpu_memory_advanced(aggressive: bool = False):
    """
    Advanced GPU memory cleanup.
    
    Args:
        aggressive: Whether to perform aggressive cleanup
    """
    memory_manager.cleanup_memory(aggressive)


def get_gpu_memory_status() -> Dict[str, Any]:
    """Get GPU memory status."""
    return memory_manager.get_status()


def start_memory_monitoring():
    """Start GPU memory monitoring."""
    memory_manager.start_monitoring()


def stop_memory_monitoring():
    """Stop GPU memory monitoring."""
    memory_manager.stop_monitoring()