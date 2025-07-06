"""
Comprehensive tests for GPU acceleration functionality.

This module tests all aspects of GPU acceleration including detection,
memory management, processing, and fallback mechanisms.
"""

import unittest
import numpy as np
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from ..gpu_utils import GPUManager, GPUInfo
from ..gpu_memory_manager import GPUMemoryManager, GPUMemoryMonitor
# GPU morphometrics import with fallback for tests
try:
    from ..gpu_morphometrics import GPUMorphometrics
    GPU_MORPHOMETRICS_AVAILABLE = True
except ImportError:
    GPU_MORPHOMETRICS_AVAILABLE = False
    GPUMorphometrics = None
from ..image_preprocessing import GPUImagePreprocessor
from ..performance_monitor import PerformanceBenchmark, AdaptiveProcessor
from ..batch_processing import GPUBatchProcessor, BatchTask
from ..exceptions import GPUMemoryError, DependencyError


class TestGPUDetection(unittest.TestCase):
    """Test GPU detection and hardware validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gpu_manager = GPUManager()
    
    def test_gpu_detection(self):
        """Test GPU detection functionality."""
        gpu_info = self.gpu_manager.detect_gpu_capabilities()
        
        # GPU info should be valid
        self.assertIsInstance(gpu_info, GPUInfo)
        self.assertIn(gpu_info.backend, ['cuda', 'opencl', 'cpu'])
        self.assertGreaterEqual(gpu_info.memory_total, 0)
        self.assertGreaterEqual(gpu_info.memory_available, 0)
    
    def test_gpu_validation(self):
        """Test GPU setup validation."""
        is_valid, issues = self.gpu_manager.validate_gpu_setup()
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(issues, list)
        
        # All issues should be strings
        for issue in issues:
            self.assertIsInstance(issue, str)
    
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        batch_size = self.gpu_manager.get_optimal_batch_size((512, 512), 4)
        
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 16)  # Reasonable upper bound
    
    def test_memory_info(self):
        """Test memory information retrieval."""
        memory_info = self.gpu_manager.get_memory_info()
        
        self.assertIn('backend', memory_info)
        self.assertIn('total_mb', memory_info)
        self.assertIn('available_mb', memory_info)
        self.assertIn('device_name', memory_info)


class TestGPUMemoryManagement(unittest.TestCase):
    """Test GPU memory management and monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = GPUMemoryManager()
        self.memory_monitor = GPUMemoryMonitor()
    
    def test_memory_snapshot(self):
        """Test memory snapshot functionality."""
        snapshot = self.memory_monitor.take_snapshot()
        
        self.assertIsNotNone(snapshot.timestamp)
        self.assertGreaterEqual(snapshot.total_mb, 0)
        self.assertGreaterEqual(snapshot.allocated_mb, 0)
        self.assertGreaterEqual(snapshot.free_mb, 0)
        self.assertIn(snapshot.backend, ['torch', 'cupy', 'cpu'])
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # This should not raise any exceptions
        self.memory_manager.cleanup_memory()
        self.memory_manager.cleanup_memory(aggressive=True)
    
    def test_memory_context_manager(self):
        """Test memory context manager."""
        with self.memory_manager.managed_memory(reserve_mb=100):
            # Should execute without errors
            pass
    
    def test_memory_trends(self):
        """Test memory trend analysis."""
        # Take a few snapshots
        for _ in range(3):
            self.memory_monitor.take_snapshot()
            time.sleep(0.1)
        
        trends = self.memory_monitor.get_memory_trends()
        
        self.assertIn('avg_usage_ratio', trends)
        self.assertIn('usage_trend', trends)
    
    def test_memory_recommendations(self):
        """Test memory optimization recommendations."""
        recommendations = self.memory_manager.get_memory_recommendations()
        
        self.assertIsInstance(recommendations, list)
        for rec in recommendations:
            self.assertIsInstance(rec, str)


@unittest.skipUnless(GPU_MORPHOMETRICS_AVAILABLE, "GPU morphometrics module not available")
class TestGPUMorphometrics(unittest.TestCase):
    """Test GPU-accelerated morphometric calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gpu_morphometrics = GPUMorphometrics()
        
        # Create test mask data
        self.test_masks = np.zeros((100, 100), dtype=np.int32)
        # Add some circular masks
        for i in range(1, 4):
            center_x, center_y = 25 * i, 25 * i
            y, x = np.ogrid[:100, :100]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 10**2
            self.test_masks[mask] = i
    
    def test_area_calculation(self):
        """Test GPU area calculation."""
        areas = self.gpu_morphometrics.calculate_areas_gpu(self.test_masks)
        
        self.assertIsInstance(areas, dict)
        self.assertEqual(len(areas), 3)  # Three cells
        
        # All areas should be positive
        for area in areas.values():
            self.assertGreater(area, 0)
    
    def test_perimeter_calculation(self):
        """Test GPU perimeter calculation."""
        perimeters = self.gpu_morphometrics.calculate_perimeters_gpu(self.test_masks)
        
        self.assertIsInstance(perimeters, dict)
        self.assertEqual(len(perimeters), 3)
        
        # All perimeters should be positive
        for perimeter in perimeters.values():
            self.assertGreater(perimeter, 0)
    
    def test_centroid_calculation(self):
        """Test GPU centroid calculation."""
        centroids = self.gpu_morphometrics.calculate_centroids_gpu(self.test_masks)
        
        self.assertIsInstance(centroids, dict)
        self.assertEqual(len(centroids), 3)
        
        # All centroids should be valid coordinates
        for centroid in centroids.values():
            self.assertIsInstance(centroid, tuple)
            self.assertEqual(len(centroid), 2)
            self.assertGreaterEqual(centroid[0], 0)
            self.assertGreaterEqual(centroid[1], 0)
    
    def test_batch_morphometrics(self):
        """Test batch morphometric calculation."""
        results = self.gpu_morphometrics.batch_calculate_morphometrics(self.test_masks)
        
        self.assertIn('areas', results)
        self.assertIn('perimeters', results)
        self.assertIn('centroids', results)
        self.assertIn('shape_descriptors', results)
        
        # Verify consistency across results
        self.assertEqual(len(results['areas']), 3)
        self.assertEqual(len(results['perimeters']), 3)
        self.assertEqual(len(results['centroids']), 3)


class TestGPUImagePreprocessing(unittest.TestCase):
    """Test GPU-accelerated image preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gpu_preprocessor = GPUImagePreprocessor()
        
        # Create test image
        self.test_image = np.random.rand(100, 100).astype(np.float32)
    
    def test_gaussian_filter(self):
        """Test GPU Gaussian filtering."""
        filtered = self.gpu_preprocessor.gaussian_filter_gpu(self.test_image, sigma=1.0)
        
        self.assertEqual(filtered.shape, self.test_image.shape)
        self.assertEqual(filtered.dtype, self.test_image.dtype)
    
    def test_median_filter(self):
        """Test GPU median filtering."""
        filtered = self.gpu_preprocessor.median_filter_gpu(self.test_image, size=3)
        
        self.assertEqual(filtered.shape, self.test_image.shape)
    
    def test_histogram_equalization(self):
        """Test GPU histogram equalization."""
        equalized = self.gpu_preprocessor.histogram_equalization_gpu(self.test_image)
        
        self.assertEqual(equalized.shape, self.test_image.shape)
        
        # Check that values are in [0, 1] range
        self.assertGreaterEqual(np.min(equalized), 0.0)
        self.assertLessEqual(np.max(equalized), 1.0)
    
    def test_morphological_operations(self):
        """Test GPU morphological operations."""
        binary_image = (self.test_image > 0.5).astype(np.float32)
        
        operations = ['erosion', 'dilation', 'opening', 'closing']
        
        for operation in operations:
            with self.subTest(operation=operation):
                result = self.gpu_preprocessor.morphological_operations_gpu(
                    binary_image, operation, kernel_size=3
                )
                self.assertEqual(result.shape, binary_image.shape)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_processor = GPUBatchProcessor(max_batch_size=2)
    
    def test_task_creation(self):
        """Test batch task creation."""
        task = BatchTask(
            task_id="test_task",
            data=np.array([1, 2, 3]),
            metadata={'type': 'test'},
            priority=1
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertIsNotNone(task.created_at)
    
    def test_task_queue(self):
        """Test task queuing."""
        task = BatchTask(
            task_id="test_task",
            data=np.array([1, 2, 3]),
            metadata={'type': 'test'}
        )
        
        success = self.batch_processor.add_task(task)
        self.assertTrue(success)
    
    def test_batch_statistics(self):
        """Test batch processing statistics."""
        stats = self.batch_processor.get_statistics()
        
        self.assertIn('queue_size', stats)
        self.assertIn('max_batch_size', stats)
        self.assertIn('gpu_backend', stats)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking and adaptive processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary file for benchmark results
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        self.benchmark = PerformanceBenchmark(results_file=self.temp_file.name)
        self.adaptive_processor = AdaptiveProcessor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_benchmark_context(self):
        """Test benchmark context manager."""
        with self.benchmark.benchmark_context(
            'test_operation', 'cpu', (100, 100), {'param': 'value'}
        ):
            time.sleep(0.01)  # Simulate some work
        
        # Should have recorded a result
        self.assertEqual(len(self.benchmark.results), 1)
        result = self.benchmark.results[0]
        
        self.assertEqual(result.operation, 'test_operation')
        self.assertEqual(result.backend, 'cpu')
        self.assertEqual(result.data_size, (100, 100))
        self.assertGreater(result.execution_time, 0)
    
    def test_performance_analysis(self):
        """Test performance profile analysis."""
        # Add some mock results
        test_data = np.random.rand(50, 50)
        
        def mock_operation(data, backend='cpu'):
            time.sleep(0.001 if backend == 'gpu' else 0.002)
            return data
        
        # Benchmark the operation multiple times
        for _ in range(5):
            self.benchmark.benchmark_operation(
                mock_operation, 'mock_operation', test_data
            )
        
        # Analyze performance
        profile = self.benchmark.analyze_performance('mock_operation')
        
        if profile:  # May be None if insufficient samples
            self.assertEqual(profile.operation, 'mock_operation')
            self.assertGreater(profile.cpu_avg_time, 0)
    
    def test_backend_recommendation(self):
        """Test backend recommendation system."""
        recommendation = self.benchmark.get_recommendation('morphometrics')
        self.assertIn(recommendation, ['gpu', 'cpu'])
    
    def test_adaptive_execution(self):
        """Test adaptive execution."""
        def test_func(data, backend='cpu'):
            return data * 2
        
        test_data = np.array([1, 2, 3])
        result = self.adaptive_processor.execute_adaptive(
            test_func, 'test_operation', test_data
        )
        
        np.testing.assert_array_equal(result, test_data * 2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and fallback mechanisms."""
    
    def test_gpu_memory_error(self):
        """Test GPU memory error handling."""
        error = GPUMemoryError("Test memory error")
        self.assertEqual(str(error), "Test memory error")
    
    def test_dependency_error(self):
        """Test dependency error handling."""
        error = DependencyError("Test dependency error")
        self.assertEqual(str(error), "Test dependency error")
    
    @patch('cells.gpu_utils.gpu_manager.detect_gpu_capabilities')
    def test_gpu_detection_failure(self, mock_detect):
        """Test handling of GPU detection failures."""
        # Mock GPU detection failure
        mock_detect.side_effect = Exception("GPU detection failed")
        
        gpu_manager = GPUManager()
        
        # Should not raise exception, should fall back gracefully
        try:
            gpu_info = gpu_manager.detect_gpu_capabilities()
            # Should get CPU fallback
            self.assertEqual(gpu_info.backend, 'cpu')
        except Exception:
            self.fail("GPU detection failure should be handled gracefully")
    
    @unittest.skipUnless(GPU_MORPHOMETRICS_AVAILABLE, "GPU morphometrics module not available")
    def test_morphometrics_fallback(self):
        """Test morphometrics fallback to CPU."""
        gpu_morphometrics = GPUMorphometrics()
        
        # Create test data
        test_masks = np.zeros((50, 50), dtype=np.int32)
        test_masks[20:30, 20:30] = 1
        
        # Should work regardless of GPU availability
        areas = gpu_morphometrics.calculate_areas_gpu(test_masks)
        self.assertIsInstance(areas, dict)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete GPU acceleration pipeline."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end GPU acceleration workflow."""
        # Create test image and masks
        test_image = np.random.rand(100, 100).astype(np.float32)
        test_masks = np.zeros((100, 100), dtype=np.int32)
        test_masks[25:75, 25:75] = 1
        
        # Test preprocessing
        gpu_preprocessor = GPUImagePreprocessor()
        processed_image = gpu_preprocessor.gaussian_filter_gpu(test_image)
        
        # Test morphometrics
        gpu_morphometrics = GPUMorphometrics()
        morphometric_results = gpu_morphometrics.batch_calculate_morphometrics(test_masks)
        
        # Verify results
        self.assertEqual(processed_image.shape, test_image.shape)
        self.assertIn('areas', morphometric_results)
        self.assertGreater(len(morphometric_results['areas']), 0)
    
    def test_memory_management_integration(self):
        """Test memory management throughout processing pipeline."""
        memory_manager = GPUMemoryManager()
        
        with memory_manager.managed_memory(reserve_mb=100):
            # Simulate some GPU operations
            test_data = np.random.rand(100, 100)
            
            gpu_preprocessor = GPUImagePreprocessor()
            result = gpu_preprocessor.gaussian_filter_gpu(test_data)
            
            self.assertEqual(result.shape, test_data.shape)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)