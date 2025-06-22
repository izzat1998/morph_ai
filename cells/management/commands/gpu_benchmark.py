"""
Django management command for GPU performance benchmarking.

Usage:
    python manage.py gpu_benchmark
    python manage.py gpu_benchmark --operation morphometrics
    python manage.py gpu_benchmark --full-suite
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import numpy as np
import time
import sys

from cells.gpu_utils import gpu_manager, log_gpu_status, is_gpu_available
from cells.gpu_memory_manager import memory_manager, get_gpu_memory_status
from cells.gpu_morphometrics import GPUMorphometrics, calculate_morphometrics_gpu
from cells.image_preprocessing import GPUImagePreprocessor
from cells.performance_monitor import PerformanceBenchmark, get_performance_recommendations


class Command(BaseCommand):
    help = 'Run GPU performance benchmarks and diagnostics'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--operation',
            type=str,
            choices=['morphometrics', 'preprocessing', 'all'],
            default='all',
            help='Specific operation to benchmark'
        )
        
        parser.add_argument(
            '--full-suite',
            action='store_true',
            help='Run comprehensive benchmark suite'
        )
        
        parser.add_argument(
            '--data-size',
            type=str,
            default='512x512',
            help='Data size for benchmarks (e.g., 512x512)'
        )
        
        parser.add_argument(
            '--iterations',
            type=int,
            default=5,
            help='Number of iterations per benchmark'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )
    
    def handle(self, *args, **options):
        """Main command handler."""
        self.stdout.write(self.style.SUCCESS('GPU Performance Benchmark'))
        self.stdout.write('=' * 50)
        
        # Parse data size
        try:
            width, height = map(int, options['data_size'].split('x'))
            data_size = (height, width)
        except ValueError:
            raise CommandError(f"Invalid data size format: {options['data_size']}")
        
        # Initialize benchmark
        benchmark = PerformanceBenchmark()
        
        try:
            # Run system diagnostics first
            self._run_diagnostics()
            
            # Run benchmarks based on options
            if options['full_suite']:
                self._run_full_benchmark_suite(benchmark, data_size, options['iterations'])
            else:
                if options['operation'] in ['morphometrics', 'all']:
                    self._benchmark_morphometrics(benchmark, data_size, options['iterations'])
                
                if options['operation'] in ['preprocessing', 'all']:
                    self._benchmark_preprocessing(benchmark, data_size, options['iterations'])
            
            # Show results and recommendations
            self._show_results(benchmark)
            
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nBenchmark interrupted by user'))
        except Exception as e:
            raise CommandError(f'Benchmark failed: {str(e)}')
    
    def _run_diagnostics(self):
        """Run GPU diagnostics."""
        self.stdout.write('\n' + self.style.HTTP_INFO('GPU Diagnostics:'))
        
        # GPU detection
        gpu_info = gpu_manager.detect_gpu_capabilities()
        self.stdout.write(f'  Backend: {gpu_info.backend}')
        self.stdout.write(f'  Device: {gpu_info.name}')
        self.stdout.write(f'  Memory: {gpu_info.memory_total}MB total, {gpu_info.memory_available}MB available')
        
        if gpu_info.cuda_capability:
            self.stdout.write(f'  CUDA Capability: {gpu_info.cuda_capability[0]}.{gpu_info.cuda_capability[1]}')
        
        # GPU validation
        is_valid, issues = gpu_manager.validate_gpu_setup()
        if is_valid:
            self.stdout.write(self.style.SUCCESS('  GPU validation: PASSED'))
        else:
            self.stdout.write(self.style.WARNING('  GPU validation: ISSUES FOUND'))
            for issue in issues:
                self.stdout.write(f'    - {issue}')
        
        # Settings
        self.stdout.write(f'  Cellpose GPU enabled: {getattr(settings, "CELLPOSE_USE_GPU", False)}')
        self.stdout.write(f'  GPU preprocessing enabled: {getattr(settings, "ENABLE_GPU_PREPROCESSING", False)}')
    
    def _benchmark_morphometrics(self, benchmark, data_size, iterations):
        """Benchmark morphometric calculations."""
        self.stdout.write('\n' + self.style.HTTP_INFO('Benchmarking Morphometrics:'))
        
        # Create test mask data
        test_masks = self._create_test_masks(data_size)
        
        gpu_morphometrics = GPUMorphometrics()
        
        # Benchmark GPU vs CPU
        gpu_times = []
        cpu_times = []
        
        for i in range(iterations):
            self.stdout.write(f'  Iteration {i+1}/{iterations}', ending='')
            
            # GPU benchmark
            if is_gpu_available():
                start_time = time.time()
                try:
                    gpu_result = calculate_morphometrics_gpu(test_masks)
                    gpu_time = time.time() - start_time
                    gpu_times.append(gpu_time)
                    self.stdout.write(f' | GPU: {gpu_time:.3f}s', ending='')
                except Exception as e:
                    self.stdout.write(f' | GPU: FAILED ({str(e)[:30]})', ending='')
            
            # CPU benchmark (using skimage)
            start_time = time.time()
            try:
                from skimage import measure
                props = measure.regionprops(test_masks)
                cpu_result = {prop.label: prop.area for prop in props if prop.label > 0}
                cpu_time = time.time() - start_time
                cpu_times.append(cpu_time)
                self.stdout.write(f' | CPU: {cpu_time:.3f}s')
            except Exception as e:
                self.stdout.write(f' | CPU: FAILED ({str(e)[:30]})')
        
        # Show results
        if gpu_times and cpu_times:
            avg_gpu = np.mean(gpu_times)
            avg_cpu = np.mean(cpu_times)
            speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
            
            self.stdout.write(f'  Average GPU time: {avg_gpu:.3f}s')
            self.stdout.write(f'  Average CPU time: {avg_cpu:.3f}s')
            self.stdout.write(f'  GPU speedup: {speedup:.2f}x')
            
            if speedup > 1.1:
                self.stdout.write(self.style.SUCCESS(f'  GPU is {speedup:.1f}x faster'))
            else:
                self.stdout.write(self.style.WARNING('  Limited GPU benefit'))
    
    def _benchmark_preprocessing(self, benchmark, data_size, iterations):
        """Benchmark image preprocessing."""
        self.stdout.write('\n' + self.style.HTTP_INFO('Benchmarking Preprocessing:'))
        
        # Create test image
        test_image = np.random.rand(*data_size).astype(np.float32)
        
        gpu_preprocessor = GPUImagePreprocessor()
        
        operations = [
            ('gaussian', {'sigma': 1.0}),
            ('median', {'size': 3}),
            ('histogram_eq', {})
        ]
        
        for op_name, params in operations:
            self.stdout.write(f'  {op_name.title()} Filter:')
            
            gpu_times = []
            cpu_times = []
            
            for i in range(iterations):
                # GPU benchmark
                if gpu_preprocessor.cupy_available:
                    start_time = time.time()
                    try:
                        if op_name == 'gaussian':
                            gpu_result = gpu_preprocessor.gaussian_filter_gpu(test_image, **params)
                        elif op_name == 'median':
                            gpu_result = gpu_preprocessor.median_filter_gpu(test_image, **params)
                        elif op_name == 'histogram_eq':
                            gpu_result = gpu_preprocessor.histogram_equalization_gpu(test_image)
                        
                        gpu_time = time.time() - start_time
                        gpu_times.append(gpu_time)
                    except Exception as e:
                        self.stdout.write(f'    GPU failed: {str(e)[:50]}')
                
                # CPU benchmark
                start_time = time.time()
                try:
                    from skimage import filters, exposure
                    if op_name == 'gaussian':
                        cpu_result = filters.gaussian(test_image, **params)
                    elif op_name == 'median':
                        cpu_result = filters.median(test_image)
                    elif op_name == 'histogram_eq':
                        cpu_result = exposure.equalize_hist(test_image)
                    
                    cpu_time = time.time() - start_time
                    cpu_times.append(cpu_time)
                except Exception as e:
                    self.stdout.write(f'    CPU failed: {str(e)[:50]}')
            
            # Show results
            if gpu_times and cpu_times:
                avg_gpu = np.mean(gpu_times)
                avg_cpu = np.mean(cpu_times)
                speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
                
                self.stdout.write(f'    GPU: {avg_gpu:.3f}s | CPU: {avg_cpu:.3f}s | Speedup: {speedup:.2f}x')
            elif cpu_times:
                self.stdout.write(f'    CPU only: {np.mean(cpu_times):.3f}s')
    
    def _run_full_benchmark_suite(self, benchmark, data_size, iterations):
        """Run comprehensive benchmark suite."""
        self.stdout.write('\n' + self.style.HTTP_INFO('Running Full Benchmark Suite:'))
        
        # Test multiple data sizes
        test_sizes = [
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024)
        ]
        
        for size in test_sizes:
            if size[0] * size[1] > data_size[0] * data_size[1] * 4:
                continue  # Skip very large sizes
            
            self.stdout.write(f'\n  Testing {size[0]}x{size[1]}:')
            
            # Memory check
            memory_status = get_gpu_memory_status()
            self.stdout.write(f'    Memory before: {memory_status["current_memory"]["usage_ratio"]:.1%}')
            
            # Run morphometrics benchmark
            test_masks = self._create_test_masks(size)
            
            if is_gpu_available():
                try:
                    start_time = time.time()
                    result = calculate_morphometrics_gpu(test_masks)
                    gpu_time = time.time() - start_time
                    
                    memory_status = get_gpu_memory_status()
                    self.stdout.write(f'    GPU morphometrics: {gpu_time:.3f}s')
                    self.stdout.write(f'    Memory after: {memory_status["current_memory"]["usage_ratio"]:.1%}')
                    
                except Exception as e:
                    self.stdout.write(f'    GPU morphometrics failed: {str(e)[:50]}')
    
    def _create_test_masks(self, data_size):
        """Create test mask data for benchmarking."""
        height, width = data_size
        masks = np.zeros((height, width), dtype=np.int32)
        
        # Add several circular masks
        num_cells = min(20, (height * width) // 1000)  # Scale with image size
        
        for i in range(1, num_cells + 1):
            # Random position and size
            center_x = np.random.randint(20, width - 20)
            center_y = np.random.randint(20, height - 20)
            radius = np.random.randint(5, 15)
            
            # Create circular mask
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Avoid overlap
            if not np.any(masks[mask] > 0):
                masks[mask] = i
        
        return masks
    
    def _show_results(self, benchmark):
        """Show benchmark results and recommendations."""
        self.stdout.write('\n' + self.style.HTTP_INFO('Benchmark Summary:'))
        
        # Statistics
        stats = benchmark.get_statistics()
        self.stdout.write(f'  Total benchmarks run: {stats["total_benchmarks"]}')
        self.stdout.write(f'  Success rate: {stats["success_rate"]:.1%}')
        
        if stats['avg_gpu_speedup'] > 0:
            self.stdout.write(f'  Average GPU speedup: {stats["avg_gpu_speedup"]:.2f}x')
        
        # Memory status
        memory_status = get_gpu_memory_status()
        self.stdout.write(f'  Final GPU memory usage: {memory_status["current_memory"]["usage_ratio"]:.1%}')
        
        # Recommendations
        recommendations = get_performance_recommendations()
        if recommendations:
            self.stdout.write('\n' + self.style.HTTP_INFO('Recommendations:'))
            for rec in recommendations:
                self.stdout.write(f'  - {rec}')
        
        # Final status
        if is_gpu_available() and stats.get('avg_gpu_speedup', 0) > 1.5:
            self.stdout.write('\n' + self.style.SUCCESS('GPU acceleration is working well!'))
        elif is_gpu_available():
            self.stdout.write('\n' + self.style.WARNING('GPU acceleration has limited benefit'))
        else:
            self.stdout.write('\n' + self.style.WARNING('GPU acceleration not available - using CPU only'))