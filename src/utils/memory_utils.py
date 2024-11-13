import torch
import psutil
import os
from typing import Optional

class MemoryManager:
    @staticmethod
    def get_available_memory():
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    @staticmethod
    def get_gpu_memory():
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(gpu)
            total_memory = gpu_properties.total_memory / (1024 * 1024 * 1024)
            allocated_memory = torch.cuda.memory_allocated(gpu) / (1024 * 1024 * 1024)
            return total_memory - allocated_memory
        return 0
    
    @staticmethod
    def select_device() -> str:
        """Select the best available device based on memory"""
        if torch.cuda.is_available():
            gpu_memory = MemoryManager.get_gpu_memory()
            if gpu_memory > 2.0:  # If more than 2GB GPU memory is available
                return 'cuda'
        return 'cpu'
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        import gc
        gc.collect()

class BatchProcessor:
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.device = MemoryManager.select_device()
    
    def process_in_batches(self, items: list, process_func):
        """Process items in batches to manage memory"""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            if i % (self.batch_size * 4) == 0:
                MemoryManager.optimize_memory()
        
        return results