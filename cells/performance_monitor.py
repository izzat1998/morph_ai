"""
Performance Monitoring and Benchmarking Module

This module provides comprehensive performance monitoring, benchmarking,
and adaptive processing selection for GPU vs CPU operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import statistics
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import os

from .gpu_utils import gpu_manager, is_gpu_available
from .gpu_memory_manager import memory_manager
from .exceptions import PerformanceBenchmarkError

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    backend: str
    execution_time: float
    memory_used_mb: int
    data_size: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: str = ""


@dataclass
class PerformanceProfile:
    """Container for operation performance profile."""
    operation: str
    gpu_avg_time: float
    cpu_avg_time: float
    gpu_speedup: float
    optimal_backend: str
    confidence: float
    sample_count: int
    data_size_range: Tuple[int, int]


class PerformanceBenchmark:
    """
    Performance benchmarking system for GPU vs CPU operations.
    
    Automatically benchmarks operations to determine optimal backend
    for different data sizes and operation types.
    """
    
    def __init__(self, results_file: str = None):
        """
        Initialize performance benchmark.
        
        Args:
            results_file: Optional file to save benchmark results
        """
        self.results_file = results_file or os.path.join(
            os.path.dirname(__file__), 'benchmark_results.json'
        )
        
        self.results = []
        self.profiles = {}
        
        # Load existing results if available
        self._load_results()
        
        logger.info(f"Performance benchmark initialized with {len(self.results)} historical results")
    
    def _load_results(self):
        """Load benchmark results from file."""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    
                self.results = [
                    BenchmarkResult(**result) for result in data.get('results', [])
                ]
                
                profiles_data = data.get('profiles', {})
                self.profiles = {
                    op: PerformanceProfile(**profile) 
                    for op, profile in profiles_data.items()
                }
                
                logger.info(f"Loaded {len(self.results)} benchmark results from {self.results_file}")
        except Exception as e:
            logger.warning(f"Failed to load benchmark results: {str(e)}")
            self.results = []
            self.profiles = {}
    
    def _save_results(self):
        """Save benchmark results to file."""
        try:
            data = {
                'results': [
                    {
                        'operation': r.operation,
                        'backend': r.backend,
                        'execution_time': r.execution_time,
                        'memory_used_mb': r.memory_used_mb,
                        'data_size': r.data_size,
                        'parameters': r.parameters,
                        'timestamp': r.timestamp,
                        'success': r.success,
                        'error_message': r.error_message
                    }
                    for r in self.results
                ],
                'profiles': {
                    op: {
                        'operation': profile.operation,
                        'gpu_avg_time': profile.gpu_avg_time,
                        'cpu_avg_time': profile.cpu_avg_time,
                        'gpu_speedup': profile.gpu_speedup,
                        'optimal_backend': profile.optimal_backend,
                        'confidence': profile.confidence,
                        'sample_count': profile.sample_count,
                        'data_size_range': profile.data_size_range
                    }
                    for op, profile in self.profiles.items()
                }
            }
            
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved benchmark results to {self.results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
    
    @contextmanager
    def benchmark_context(self, operation: str, backend: str, 
                         data_size: Tuple[int, ...], parameters: Dict[str, Any] = None):
        """
        Context manager for benchmarking operations.
        
        Args:
            operation: Name of the operation being benchmarked
            backend: Backend being used ('gpu' or 'cpu')
            data_size: Size of data being processed
            parameters: Additional parameters
        """
        parameters = parameters or {}
        
        # Get initial memory state
        if backend == 'gpu' and is_gpu_available():
            initial_memory = memory_manager.get_status()['current_memory']['allocated_mb']
        else:
            initial_memory = 0
        
        start_time = time.time()
        success = True
        error_message = ""
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Calculate memory usage
            if backend == 'gpu' and is_gpu_available():
                final_memory = memory_manager.get_status()['current_memory']['allocated_mb']
                memory_used = max(0, final_memory - initial_memory)
            else:
                memory_used = 0
            
            # Create benchmark result
            result = BenchmarkResult(
                operation=operation,
                backend=backend,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                data_size=data_size,
                parameters=parameters,
                success=success,
                error_message=error_message
            )
            
            self.results.append(result)
            logger.debug(f"Benchmark result: {operation} ({backend}) - {execution_time:.3f}s")
    
    def benchmark_operation(self, operation_func: Callable, operation_name: str,
                          test_data: Any, parameters: Dict[str, Any] = None,
                          test_both_backends: bool = True) -> Dict[str, BenchmarkResult]:
        """
        Benchmark an operation on both GPU and CPU.
        
        Args:
            operation_func: Function to benchmark
            operation_name: Name of the operation
            test_data: Test data for the operation
            parameters: Additional parameters
            test_both_backends: Whether to test both GPU and CPU
            
        Returns:
            Dictionary of benchmark results by backend
        """
        parameters = parameters or {}
        results = {}
        
        # Determine data size
        if hasattr(test_data, 'shape'):
            data_size = test_data.shape
        elif hasattr(test_data, '__len__'):
            data_size = (len(test_data),)
        else:
            data_size = (1,)
        
        # Test GPU if available
        if is_gpu_available():
            try:
                with self.benchmark_context(operation_name, 'gpu', data_size, parameters):
                    gpu_result = operation_func(test_data, backend='gpu', **parameters)
                results['gpu'] = self.results[-1]
            except Exception as e:
                logger.warning(f"GPU benchmark failed for {operation_name}: {str(e)}")
        
        # Test CPU
        if test_both_backends or not is_gpu_available():
            try:
                with self.benchmark_context(operation_name, 'cpu', data_size, parameters):
                    cpu_result = operation_func(test_data, backend='cpu', **parameters)
                results['cpu'] = self.results[-1]
            except Exception as e:
                logger.warning(f"CPU benchmark failed for {operation_name}: {str(e)}")
        
        return results
    
    def analyze_performance(self, operation: str, min_samples: int = 3) -> Optional[PerformanceProfile]:
        """
        Analyze performance for a specific operation.
        
        Args:
            operation: Operation name to analyze
            min_samples: Minimum number of samples required
            
        Returns:
            Performance profile or None if insufficient data
        """
        # Get results for this operation
        operation_results = [r for r in self.results if r.operation == operation and r.success]
        
        if len(operation_results) < min_samples:
            logger.debug(f"Insufficient samples for {operation}: {len(operation_results)} < {min_samples}")
            return None
        
        # Separate GPU and CPU results
        gpu_results = [r for r in operation_results if r.backend == 'gpu']
        cpu_results = [r for r in operation_results if r.backend == 'cpu']
        
        if not gpu_results or not cpu_results:
            logger.debug(f"Missing backend results for {operation}")
            return None
        
        # Calculate average times
        gpu_times = [r.execution_time for r in gpu_results]
        cpu_times = [r.execution_time for r in cpu_results]
        
        gpu_avg_time = statistics.mean(gpu_times)
        cpu_avg_time = statistics.mean(cpu_times)
        
        # Calculate speedup
        gpu_speedup = cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 0
        
        # Determine optimal backend
        optimal_backend = 'gpu' if gpu_speedup > 1.1 else 'cpu'  # 10% threshold
        
        # Calculate confidence based on sample count and variance
        confidence = min(1.0, len(operation_results) / 10.0)  # Max confidence at 10 samples
        
        # Data size range
        all_sizes = [r.data_size for r in operation_results]
        if all_sizes and all(isinstance(s, tuple) and s for s in all_sizes):
            size_products = [np.prod(s) for s in all_sizes]
            data_size_range = (min(size_products), max(size_products))
        else:
            data_size_range = (0, 0)
        
        profile = PerformanceProfile(
            operation=operation,
            gpu_avg_time=gpu_avg_time,
            cpu_avg_time=cpu_avg_time,
            gpu_speedup=gpu_speedup,
            optimal_backend=optimal_backend,
            confidence=confidence,
            sample_count=len(operation_results),
            data_size_range=data_size_range
        )
        
        self.profiles[operation] = profile
        self._save_results()
        
        return profile
    
    def get_recommendation(self, operation: str, data_size: Tuple[int, ...] = None) -> str:
        """
        Get backend recommendation for an operation.
        
        Args:
            operation: Operation name
            data_size: Optional data size for size-based recommendations
            
        Returns:
            Recommended backend ('gpu' or 'cpu')
        """
        # Check if we have a profile for this operation
        if operation in self.profiles:
            profile = self.profiles[operation]
            
            # If confidence is high, use the optimal backend
            if profile.confidence > 0.5:
                return profile.optimal_backend
        
        # Fallback to GPU if available for common GPU-friendly operations
        gpu_friendly_ops = ['morphometrics', 'preprocessing', 'segmentation']
        if any(op in operation.lower() for op in gpu_friendly_ops):
            return 'gpu' if is_gpu_available() else 'cpu'
        
        # Default to CPU for unknown operations
        return 'cpu'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        total_results = len(self.results)
        successful_results = len([r for r in self.results if r.success])
        
        gpu_results = [r for r in self.results if r.backend == 'gpu' and r.success]
        cpu_results = [r for r in self.results if r.backend == 'cpu' and r.success]
        
        stats = {
            'total_benchmarks': total_results,
            'successful_benchmarks': successful_results,
            'success_rate': successful_results / total_results if total_results > 0 else 0,
            'gpu_benchmarks': len(gpu_results),
            'cpu_benchmarks': len(cpu_results),
            'operations_profiled': len(self.profiles),
            'avg_gpu_speedup': statistics.mean([p.gpu_speedup for p in self.profiles.values()]) if self.profiles else 0
        }
        
        return stats


class AdaptiveProcessor:
    """
    Adaptive processor that automatically selects optimal backend for operations.
    
    Uses performance benchmarks and real-time monitoring to make intelligent
    backend selection decisions.
    """
    
    def __init__(self):
        """Initialize adaptive processor."""
        self.benchmark = PerformanceBenchmark()
        self.runtime_stats = {}
        
        logger.info("Adaptive processor initialized")
    
    def select_backend(self, operation: str, data_size: Tuple[int, ...] = None,
                      force_backend: str = None) -> str:
        """
        Select optimal backend for an operation.
        
        Args:
            operation: Operation name
            data_size: Data size (optional)
            force_backend: Force specific backend (optional)
            
        Returns:
            Selected backend ('gpu' or 'cpu')
        """
        if force_backend:
            return force_backend
        
        # Check GPU availability
        if not is_gpu_available():
            return 'cpu'
        
        # Get benchmark recommendation
        recommendation = self.benchmark.get_recommendation(operation, data_size)
        
        # Check current GPU memory status
        memory_status = memory_manager.get_status()
        if memory_status['current_memory']['usage_ratio'] > 0.9:
            logger.warning("High GPU memory usage, preferring CPU")
            return 'cpu'
        
        return recommendation
    
    def execute_adaptive(self, operation_func: Callable, operation_name: str,
                        data: Any, **kwargs) -> Any:
        """
        Execute operation with adaptive backend selection.
        
        Args:
            operation_func: Function to execute
            operation_name: Name of the operation
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Operation result
        """
        # Determine data size
        if hasattr(data, 'shape'):
            data_size = data.shape
        else:
            data_size = None
        
        # Select backend
        backend = self.select_backend(operation_name, data_size, kwargs.get('backend'))
        
        # Execute with benchmarking
        start_time = time.time()
        
        try:
            with self.benchmark.benchmark_context(operation_name, backend, data_size or (1,)):
                result = operation_func(data, backend=backend, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Update runtime stats
            if operation_name not in self.runtime_stats:
                self.runtime_stats[operation_name] = {'count': 0, 'total_time': 0.0}
            
            self.runtime_stats[operation_name]['count'] += 1
            self.runtime_stats[operation_name]['total_time'] += execution_time
            
            logger.debug(f"Adaptive execution: {operation_name} ({backend}) - {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive execution failed: {operation_name} ({backend}) - {str(e)}")
            
            # Try fallback backend if original failed
            if backend == 'gpu':
                logger.info("Retrying with CPU backend")
                return operation_func(data, backend='cpu', **kwargs)
            else:
                raise
    
    def update_profiles(self):
        """Update performance profiles based on recent results."""
        operations = set(r.operation for r in self.benchmark.results)
        
        for operation in operations:
            profile = self.benchmark.analyze_performance(operation)
            if profile:
                logger.debug(f"Updated profile for {operation}: {profile.optimal_backend} "
                           f"(speedup: {profile.gpu_speedup:.2f}x)")
    
    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get runtime execution statistics."""
        stats = {}
        
        for operation, data in self.runtime_stats.items():
            if data['count'] > 0:
                stats[operation] = {
                    'execution_count': data['count'],
                    'total_time': data['total_time'],
                    'avg_time': data['total_time'] / data['count']
                }
        
        return stats


# Global adaptive processor instance
adaptive_processor = AdaptiveProcessor()


def execute_with_adaptive_backend(operation_func: Callable, operation_name: str, 
                                data: Any, **kwargs) -> Any:
    """
    Execute operation with adaptive backend selection.
    
    Args:
        operation_func: Function to execute
        operation_name: Operation name
        data: Input data
        **kwargs: Additional arguments
        
    Returns:
        Operation result
    """
    return adaptive_processor.execute_adaptive(operation_func, operation_name, data, **kwargs)


def get_performance_recommendations() -> List[str]:
    """Get performance optimization recommendations."""
    stats = adaptive_processor.benchmark.get_statistics()
    recommendations = []
    
    if stats['total_benchmarks'] < 10:
        recommendations.append("Run more analyses to improve performance profiling")
    
    if stats['avg_gpu_speedup'] > 2.0:
        recommendations.append("GPU acceleration is highly effective - ensure GPU is properly configured")
    elif stats['avg_gpu_speedup'] < 1.1:
        recommendations.append("Limited GPU benefit detected - consider CPU-optimized workflows")
    
    memory_status = memory_manager.get_status()
    if memory_status['current_memory']['usage_ratio'] > 0.8:
        recommendations.append("High GPU memory usage - consider reducing batch sizes")
    
    return recommendations