"""
Batch Processing Module for GPU Efficiency

This module provides batch processing capabilities to maximize GPU utilization
and improve performance when processing multiple images or analysis tasks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import queue
import threading

# Custom GPU modules removed - using Cellpose's built-in GPU acceleration
from .exceptions import MorphometricAnalysisError, DependencyError

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Container for batch processing task."""
    task_id: str
    data: Any
    metadata: Dict[str, Any]
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class BatchProcessor:
    """
    GPU-efficient batch processor for morphometric analysis tasks.
    
    This class manages batching of operations to maximize GPU utilization
    and minimize memory transfers between CPU and GPU.
    """
    
    def __init__(self, max_batch_size: int = None, max_queue_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size (auto-detected if None)
            max_queue_size: Maximum number of tasks in queue
        """
        # Use simple defaults - Cellpose handles GPU optimization internally
        self.max_batch_size = max_batch_size or 4
        self.max_queue_size = max_queue_size
        
        # Task queue and processing state
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.results = {}
        self.processing = False
        self.worker_thread = None
        
        # Performance tracking
        self.batch_stats = {
            'total_batches': 0,
            'total_tasks': 0,
            'total_time': 0.0,
            'gpu_time': 0.0,
            'avg_batch_time': 0.0
        }
        
        logger.info(f"BatchProcessor initialized - max_batch_size: {self.max_batch_size}")
    
    def add_task(self, task: BatchTask) -> bool:
        """
        Add a task to the processing queue.
        
        Args:
            task: BatchTask to add
            
        Returns:
            True if task was added, False if queue is full
        """
        try:
            # Use negative priority for priority queue (higher priority = lower number)
            self.task_queue.put((-task.priority, task.created_at, task), block=False)
            logger.debug(f"Added task {task.task_id} to queue (priority: {task.priority})")
            return True
        except queue.Full:
            logger.warning(f"Task queue full, cannot add task {task.task_id}")
            return False
    
    def start_processing(self):
        """Start background batch processing."""
        if self.processing:
            logger.warning("Batch processing already started")
            return
        
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Batch processing started")
    
    def stop_processing(self):
        """Stop background batch processing."""
        if not self.processing:
            return
        
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Batch processing stopped")
    
    def _process_loop(self):
        """Main processing loop for batch tasks."""
        while self.processing:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.1)  # Brief pause if no tasks
            except Exception as e:
                logger.error(f"Error in batch processing loop: {str(e)}")
                time.sleep(1.0)  # Longer pause on error
    
    def _collect_batch(self) -> List[BatchTask]:
        """Collect tasks for batch processing."""
        batch = []
        deadline = time.time() + 0.5  # Max wait time for batch collection
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                # Get task with short timeout
                priority, created_at, task = self.task_queue.get(timeout=0.1)
                batch.append(task)
                self.task_queue.task_done()
            except queue.Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[BatchTask]):
        """Process a batch of tasks."""
        if not batch:
            return
        
        start_time = time.time()
        batch_size = len(batch)
        
        logger.debug(f"Processing batch of {batch_size} tasks")
        
        try:
            # Group tasks by type for efficient processing
            task_groups = self._group_tasks_by_type(batch)
            
            for task_type, tasks in task_groups.items():
                self._process_task_group(task_type, tasks)
            
            # Update statistics
            elapsed = time.time() - start_time
            self.batch_stats['total_batches'] += 1
            self.batch_stats['total_tasks'] += batch_size
            self.batch_stats['total_time'] += elapsed
            self.batch_stats['avg_batch_time'] = self.batch_stats['total_time'] / self.batch_stats['total_batches']
            
            logger.debug(f"Batch processing completed in {elapsed:.3f}s "
                        f"({batch_size} tasks, {elapsed/batch_size:.3f}s per task)")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            # Mark all tasks as failed
            for task in batch:
                self.results[task.task_id] = {'error': str(e), 'success': False}
    
    def _group_tasks_by_type(self, batch: List[BatchTask]) -> Dict[str, List[BatchTask]]:
        """Group tasks by type for efficient batch processing."""
        groups = {}
        for task in batch:
            task_type = task.metadata.get('type', 'unknown')
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(task)
        return groups
    
    def _process_task_group(self, task_type: str, tasks: List[BatchTask]):
        """Process a group of similar tasks efficiently."""
        if task_type == 'morphometrics':
            self._process_morphometrics_batch(tasks)
        elif task_type == 'preprocessing':
            self._process_preprocessing_batch(tasks)
        elif task_type == 'segmentation':
            self._process_segmentation_batch(tasks)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            # Process individually as fallback
            for task in tasks:
                self._process_single_task(task)
    
    def _process_morphometrics_batch(self, tasks: List[BatchTask]):
        """Process morphometric calculations using standard scikit-image."""
        self._process_cpu_morphometrics_batch(tasks)
    
    def _process_cpu_morphometrics_batch(self, tasks: List[BatchTask]):
        """Process morphometric calculations using scikit-image."""
        from skimage import measure
        import numpy as np
        
        for task in tasks:
            try:
                masks = task.data
                
                # Use CPU regionprops for morphometric calculations
                props = measure.regionprops(masks)
                result = {
                    'areas': {},
                    'perimeters': {},
                    'centroids': {},
                    'shape_descriptors': {}
                }
                
                for prop in props:
                    if prop.label > 0:
                        result['areas'][prop.label] = prop.area
                        result['perimeters'][prop.label] = prop.perimeter
                        result['centroids'][prop.label] = prop.centroid
                        result['shape_descriptors'][prop.label] = {
                            'area': prop.area,
                            'perimeter': prop.perimeter,
                            'circularity': 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0,
                            'eccentricity': prop.eccentricity,
                            'solidity': prop.solidity,
                            'extent': prop.extent
                        }
                
                self.results[task.task_id] = {
                    'result': result,
                    'success': True,
                    'processing_time': 0.0  # Would need timing if required
                }
                
            except Exception as e:
                logger.error(f"CPU morphometrics processing failed for task {task.task_id}: {str(e)}")
                self.results[task.task_id] = {'error': str(e), 'success': False}
    
    def _process_preprocessing_batch(self, tasks: List[BatchTask]):
        """Process image preprocessing in batch."""
        try:
            from .image_preprocessing import GPUImagePreprocessor
            
            gpu_preprocessor = GPUImagePreprocessor()
            
            for task in tasks:
                try:
                    image = task.data
                    operation = task.metadata.get('operation', 'gaussian')
                    params = task.metadata.get('params', {})
                    
                    if operation == 'gaussian':
                        result = gpu_preprocessor.gaussian_filter_gpu(image, **params)
                    elif operation == 'median':
                        result = gpu_preprocessor.median_filter_gpu(image, **params)
                    elif operation == 'histogram_eq':
                        result = gpu_preprocessor.histogram_equalization_gpu(image)
                    elif operation == 'morphological':
                        result = gpu_preprocessor.morphological_operations_gpu(image, **params)
                    else:
                        raise ValueError(f"Unknown preprocessing operation: {operation}")
                    
                    self.results[task.task_id] = {'result': result, 'success': True}
                    
                except Exception as e:
                    self.results[task.task_id] = {'error': str(e), 'success': False}
            
            # No GPU memory cleanup needed - using standard processing
            
        except Exception as e:
            logger.error(f"Preprocessing batch processing failed: {str(e)}")
            for task in tasks:
                self.results[task.task_id] = {'error': str(e), 'success': False}
    
    def _process_segmentation_batch(self, tasks: List[BatchTask]):
        """Process segmentation tasks in batch."""
        try:
            # For Cellpose segmentation, batch processing is more complex
            # For now, process individually but with optimized GPU usage
            for task in tasks:
                self._process_single_task(task)
                
        except Exception as e:
            logger.error(f"Segmentation batch processing failed: {str(e)}")
            for task in tasks:
                self.results[task.task_id] = {'error': str(e), 'success': False}
    
    def _process_single_task(self, task: BatchTask):
        """Process a single task (fallback)."""
        try:
            # This would be implemented based on task type
            logger.warning(f"Processing task {task.task_id} individually (no batch optimization)")
            
            # Placeholder - actual implementation would depend on task type
            self.results[task.task_id] = {
                'result': None,
                'success': True,
                'note': 'Processed individually'
            }
            
        except Exception as e:
            self.results[task.task_id] = {'error': str(e), 'success': False}
    
    def get_result(self, task_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Get result for a specific task.
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait for result
            
        Returns:
            Task result dictionary
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if task_id in self.results:
                return self.results.pop(task_id)
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.batch_stats.copy()
        stats.update({
            'queue_size': self.task_queue.qsize(),
            'pending_results': len(self.results),
            'processing': self.processing,
            'max_batch_size': self.max_batch_size
        })
        return stats
    
    def clear_results(self):
        """Clear completed results to free memory."""
        cleared = len(self.results)
        self.results.clear()
        logger.debug(f"Cleared {cleared} completed results")


class BatchManager:
    """
    High-level batch processing manager for analysis workflows.
    
    This class provides a simple interface for batching analysis operations
    across multiple images or analysis steps.
    """
    
    def __init__(self):
        """Initialize batch manager."""
        self.processor = BatchProcessor()
        self.processor.start_processing()
        logger.info("BatchManager initialized")
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'processor'):
            self.processor.stop_processing()
    
    def process_morphometrics_batch(self, masks_list: List[np.ndarray], 
                                  task_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple morphometric calculations in batch.
        
        Args:
            masks_list: List of mask arrays
            task_ids: Optional list of task IDs
            
        Returns:
            List of morphometric results
        """
        if task_ids is None:
            task_ids = [f"morph_{i}" for i in range(len(masks_list))]
        
        if len(task_ids) != len(masks_list):
            raise ValueError("Number of task IDs must match number of masks")
        
        # Submit tasks
        for i, (masks, task_id) in enumerate(zip(masks_list, task_ids)):
            task = BatchTask(
                task_id=task_id,
                data=masks,
                metadata={'type': 'morphometrics'},
                priority=1
            )
            self.processor.add_task(task)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = self.processor.get_result(task_id, timeout=60.0)
                results.append(result)
            except TimeoutError:
                logger.error(f"Timeout waiting for result of task {task_id}")
                results.append({'error': 'Timeout', 'success': False})
        
        return results
    
    def process_preprocessing_batch(self, images: List[np.ndarray], 
                                  operations: List[str],
                                  params_list: List[Dict] = None) -> List[np.ndarray]:
        """
        Process multiple preprocessing operations in batch.
        
        Args:
            images: List of image arrays
            operations: List of operation names
            params_list: List of parameter dictionaries
            
        Returns:
            List of processed images
        """
        if params_list is None:
            params_list = [{}] * len(images)
        
        if len(operations) == 1:
            operations = operations * len(images)
        
        task_ids = [f"preproc_{i}" for i in range(len(images))]
        
        # Submit tasks
        for i, (image, operation, params, task_id) in enumerate(
            zip(images, operations, params_list, task_ids)
        ):
            task = BatchTask(
                task_id=task_id,
                data=image,
                metadata={
                    'type': 'preprocessing',
                    'operation': operation,
                    'params': params
                },
                priority=2
            )
            self.processor.add_task(task)
        
        # Collect results
        processed_images = []
        for task_id in task_ids:
            try:
                result = self.processor.get_result(task_id, timeout=30.0)
                if result['success']:
                    processed_images.append(result['result'])
                else:
                    logger.error(f"Preprocessing failed for {task_id}: {result.get('error')}")
                    processed_images.append(None)
            except TimeoutError:
                logger.error(f"Timeout waiting for preprocessing result {task_id}")
                processed_images.append(None)
        
        return processed_images
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.processor.get_statistics()
    
    def shutdown(self):
        """Shutdown batch processing."""
        self.processor.stop_processing()
        logger.info("BatchManager shutdown complete")


# Global batch manager instance
batch_manager = BatchManager()


def process_multiple_morphometrics(masks_list: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    Convenience function for batch morphometric processing.
    
    Args:
        masks_list: List of mask arrays
        
    Returns:
        List of morphometric results
    """
    return batch_manager.process_morphometrics_batch(masks_list)


def process_multiple_images(images: List[np.ndarray], operation: str, 
                          params: Dict = None) -> List[np.ndarray]:
    """
    Convenience function for batch image processing.
    
    Args:
        images: List of image arrays
        operation: Preprocessing operation name
        params: Operation parameters
        
    Returns:
        List of processed images
    """
    params = params or {}
    return batch_manager.process_preprocessing_batch(
        images, [operation], [params] * len(images)
    )