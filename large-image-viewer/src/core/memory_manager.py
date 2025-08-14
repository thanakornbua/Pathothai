"""
Memory manager for efficient handling of large images.
"""

import psutil
import gc
import numpy as np
from typing import Optional, Dict, Any
import threading
import time

class MemoryManager:
    """
    Manages memory usage for large image processing operations.
    """
    
    def __init__(self, max_memory_percent: float = 80.0):
        """
        Initialize memory manager.
        
        Args:
            max_memory_percent: Maximum percentage of system memory to use
        """
        self.max_memory_percent = max_memory_percent
        self.total_memory = psutil.virtual_memory().total
        self.max_memory_bytes = self.total_memory * (max_memory_percent / 100.0)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.monitor_thread = None
        self.monitoring = False
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
    
    def is_memory_available(self, required_bytes: int) -> bool:
        """Check if enough memory is available for an operation."""
        current_memory = psutil.virtual_memory()
        return current_memory.available >= required_bytes
    
    def estimate_image_memory(self, height: int, width: int, channels: int = 3, dtype=np.uint8) -> int:
        """
        Estimate memory required for an image.
        
        Args:
            height, width, channels: Image dimensions
            dtype: Data type of the image
            
        Returns:
            Estimated memory in bytes
        """
        if dtype == np.uint8:
            bytes_per_pixel = 1
        elif dtype == np.uint16:
            bytes_per_pixel = 2
        elif dtype == np.float32:
            bytes_per_pixel = 4
        elif dtype == np.float64:
            bytes_per_pixel = 8
        else:
            bytes_per_pixel = 4  # Default assumption
        
        return height * width * channels * bytes_per_pixel
    
    def suggest_chunk_size(self, image_height: int, image_width: int, channels: int = 3) -> tuple:
        """
        Suggest optimal chunk size based on available memory.
        
        Args:
            image_height, image_width, channels: Image dimensions
            
        Returns:
            Suggested (chunk_height, chunk_width)
        """
        available_memory = psutil.virtual_memory().available
        safe_memory = available_memory * 0.5  # Use only 50% of available memory
        
        # Calculate bytes per pixel
        bytes_per_pixel = channels  # Assuming uint8
        
        # Calculate maximum pixels that fit in safe memory
        max_pixels = safe_memory // bytes_per_pixel
        
        # Calculate chunk dimensions (square chunks preferred)
        chunk_pixels = min(max_pixels, image_height * image_width)
        chunk_side = int(np.sqrt(chunk_pixels))
        
        # Ensure chunks don't exceed image dimensions
        chunk_height = min(chunk_side, image_height)
        chunk_width = min(chunk_side, image_width)
        
        # Minimum chunk size
        chunk_height = max(chunk_height, 256)
        chunk_width = max(chunk_width, 256)
        
        return chunk_height, chunk_width
    
    def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring in a separate thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_memory(self, interval: float):
        """Internal memory monitoring function."""
        while self.monitoring:
            memory_info = self.get_memory_info()
            
            # If memory usage is too high, trigger cleanup
            if memory_info['used_percent'] > 90:
                print(f"High memory usage detected: {memory_info['used_percent']:.1f}%")
                self.cleanup_cache()
                gc.collect()
            
            time.sleep(interval)
    
    def add_to_cache(self, key: str, data: Any, max_cache_size: int = 5):
        """
        Add data to memory cache with size limiting.
        
        Args:
            key: Cache key
            data: Data to cache
            max_cache_size: Maximum number of items in cache
        """
        with self.cache_lock:
            # Remove oldest items if cache is full
            if len(self.cache) >= max_cache_size:
                # Remove the first item (oldest)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = data
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        with self.cache_lock:
            return self.cache.get(key)
    
    def remove_from_cache(self, key: str):
        """Remove specific item from cache."""
        with self.cache_lock:
            if key in self.cache:
                del self.cache[key]
    
    def cleanup_cache(self):
        """Clear all cached data."""
        with self.cache_lock:
            self.cache.clear()
        gc.collect()
    
    def optimize_for_large_image(self):
        """Optimize system for large image processing."""
        # Force garbage collection
        gc.collect()
        
        # Clear cache
        self.cleanup_cache()
        
        # Get memory info
        memory_info = self.get_memory_info()
        print(f"Memory optimization complete. Available: {memory_info['available_gb']:.1f} GB")
        
        return memory_info
    
    def get_safe_array_size(self, dtype=np.uint8) -> int:
        """
        Get the maximum safe array size for the given data type.
        
        Args:
            dtype: NumPy data type
            
        Returns:
            Maximum number of elements
        """
        available_memory = psutil.virtual_memory().available
        safe_memory = available_memory * 0.7  # Use 70% of available memory
        
        if dtype == np.uint8:
            element_size = 1
        elif dtype == np.uint16:
            element_size = 2
        elif dtype == np.float32:
            element_size = 4
        elif dtype == np.float64:
            element_size = 8
        else:
            element_size = 4  # Default
        
        return int(safe_memory // element_size)
    
    def check_memory_threshold(self, threshold_percent: float = 85.0) -> bool:
        """
        Check if memory usage exceeds threshold.
        
        Args:
            threshold_percent: Memory usage threshold
            
        Returns:
            True if memory usage is below threshold
        """
        memory = psutil.virtual_memory()
        return memory.percent < threshold_percent
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        self.cleanup_cache()
