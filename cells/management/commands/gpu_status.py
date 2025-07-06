"""
Django management command to check GPU status for morphometric analysis.
Usage: python manage.py gpu_status
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import torch


class Command(BaseCommand):
    help = 'Check GPU status and configuration for morphometric analysis'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test',
            action='store_true',
            help='Run a quick Cellpose test to verify GPU usage',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=== GPU STATUS CHECK ==='))
        
        # Basic GPU info
        if torch.cuda.is_available():
            self.stdout.write(f'✅ CUDA Available: True')
            self.stdout.write(f'✅ GPU Device: {torch.cuda.get_device_name(0)}')
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.stdout.write(f'✅ GPU Memory: {memory_gb:.1f} GB')
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.stdout.write(f'📊 Memory Allocated: {allocated:.1f} MB')
            self.stdout.write(f'📊 Memory Reserved: {reserved:.1f} MB')
        else:
            self.stdout.write(self.style.ERROR('❌ CUDA Not Available'))
            return
        
        # Django settings
        self.stdout.write('\n=== DJANGO SETTINGS ===')
        self.stdout.write(f'CELLPOSE_USE_GPU: {settings.CELLPOSE_USE_GPU}')
        self.stdout.write(f'GPU_MEMORY_FRACTION: {settings.GPU_MEMORY_FRACTION}')
        self.stdout.write(f'ENABLE_GPU_PREPROCESSING: {settings.ENABLE_GPU_PREPROCESSING}')
        
        # Test if requested
        if options['test']:
            self.run_gpu_test()
    
    def run_gpu_test(self):
        """Run a quick GPU test with Cellpose"""
        self.stdout.write('\n=== RUNNING GPU TEST ===')
        
        try:
            from cellpose import models
            import numpy as np
            import time
            
            # Create simple test image
            test_image = np.zeros((200, 200), dtype=np.uint8)
            from skimage import draw
            
            # Add a test circle
            rr, cc = draw.disk((100, 100), 30, shape=test_image.shape)
            test_image[rr, cc] = 200
            
            self.stdout.write('🔧 Created test image...')
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # Initialize model
            self.stdout.write('🔧 Initializing Cellpose model...')
            model = models.CellposeModel(gpu=True, model_type='cyto')
            
            model_memory = torch.cuda.memory_allocated() / 1024**2
            memory_used = model_memory - initial_memory
            
            # Check device
            device = next(model.net.parameters()).device
            if device.type == 'cuda':
                self.stdout.write(self.style.SUCCESS(f'✅ Model on GPU: {device}'))
            else:
                self.stdout.write(self.style.ERROR(f'❌ Model on CPU: {device}'))
                return
            
            # Run segmentation
            self.stdout.write('🔧 Running segmentation...')
            start_time = time.time()
            masks, flows, styles = model.eval(test_image, diameter=60, channels=[0,0])
            end_time = time.time()
            
            # Results
            cells_detected = len(np.unique(masks)) - 1
            processing_time = end_time - start_time
            final_memory = torch.cuda.memory_allocated() / 1024**2
            
            self.stdout.write(self.style.SUCCESS('✅ GPU TEST SUCCESSFUL!'))
            self.stdout.write(f'📊 Processing time: {processing_time:.2f}s')
            self.stdout.write(f'📊 Cells detected: {cells_detected}')
            self.stdout.write(f'📊 GPU memory used: {memory_used:.1f} MB')
            self.stdout.write(f'📊 Peak memory: {final_memory:.1f} MB')
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ GPU test failed: {str(e)}'))
            import traceback
            traceback.print_exc()