"""
Core image loader module for handling large images efficiently.
Supports chunked loading, memory mapping, and various formats.
"""

import os
import numpy as np
from PIL import Image, ImageFile
import cv2
import psutil
import gc
from typing import Tuple, Optional, Union
import h5py
import zarr
import dask.array as da
from skimage import io

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LargeImageLoader:
    """
    Efficient loader for very large images (1GB+) with memory management.
    """
    
    def __init__(self, chunk_size: int = 1024, max_memory_gb: float = 4.0):
        """
        Initialize the image loader.
        
        Args:
            chunk_size: Size of image chunks for processing
            max_memory_gb: Maximum memory to use for image loading
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.current_image = None
        self.image_path = None
        self.image_shape = None
        self.channels = None
        self.dtype = None
        self.is_loaded = False
        
    def load_image(self, file_path: str) -> bool:
        """
        Load an image file efficiently.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            self.image_path = file_path
            file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
            
            print(f"Loading image: {os.path.basename(file_path)}")
            print(f"File size: {file_size:.2f} GB")
            
            # Choose loading strategy based on file size
            if file_size > self.max_memory_gb:
                return self._load_large_image_chunked(file_path)
            else:
                return self._load_image_memory(file_path)
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def _load_image_memory(self, file_path: str) -> bool:
        """Load image entirely into memory."""
        try:
            # Try different loading methods
            if file_path.lower().endswith(('.tif', '.tiff')):
                # Use scikit-image for TIFF files
                self.current_image = io.imread(file_path)
            else:
                # Use OpenCV for other formats
                self.current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if self.current_image is None:
                    # Fallback to PIL
                    pil_image = Image.open(file_path)
                    self.current_image = np.array(pil_image)
            
            if self.current_image is None:
                return False
                
            self._analyze_image()
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error in memory loading: {e}")
            return False
    
    def _load_large_image_chunked(self, file_path: str) -> bool:
        """Load very large images using chunked/memory-mapped approach."""
        try:
            # Create a memory-mapped array for the image
            temp_dir = os.path.join(os.path.dirname(file_path), 'temp_mmap')
            os.makedirs(temp_dir, exist_ok=True)
            
            # First, get image dimensions without loading
            with Image.open(file_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                mode = img.mode
            
            # Create HDF5 file for efficient access
            h5_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}.h5")
            
            if not os.path.exists(h5_path):
                print("Creating chunked representation...")
                self._create_chunked_file(file_path, h5_path, width, height, channels)
            
            # Load as dask array for efficient chunked access
            with h5py.File(h5_path, 'r') as f:
                self.current_image = da.from_array(f['image'], chunks=(self.chunk_size, self.chunk_size, channels))
            
            self.image_shape = (height, width, channels)
            self.channels = channels
            self.dtype = self.current_image.dtype
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error in chunked loading: {e}")
            return False
    
    def _create_chunked_file(self, source_path: str, h5_path: str, width: int, height: int, channels: int):
        """Create an HDF5 chunked representation of the large image."""
        try:
            with h5py.File(h5_path, 'w') as f:
                # Create chunked dataset
                dataset = f.create_dataset(
                    'image', 
                    shape=(height, width, channels),
                    dtype=np.uint8,
                    chunks=(self.chunk_size, self.chunk_size, channels),
                    compression='gzip'
                )
                
                # Load image in chunks
                with Image.open(source_path) as img:
                    for y in range(0, height, self.chunk_size):
                        for x in range(0, width, self.chunk_size):
                            # Calculate chunk boundaries
                            x_end = min(x + self.chunk_size, width)
                            y_end = min(y + self.chunk_size, height)
                            
                            # Extract chunk
                            box = (x, y, x_end, y_end)
                            chunk = img.crop(box)
                            chunk_array = np.array(chunk)
                            
                            # Handle different channel configurations
                            if len(chunk_array.shape) == 2:
                                chunk_array = np.expand_dims(chunk_array, axis=2)
                            if chunk_array.shape[2] < channels:
                                # Pad channels if necessary
                                padding = np.zeros((chunk_array.shape[0], chunk_array.shape[1], channels - chunk_array.shape[2]))
                                chunk_array = np.concatenate([chunk_array, padding], axis=2)
                            
                            # Store chunk
                            dataset[y:y_end, x:x_end, :] = chunk_array
                            
                            # Force garbage collection to manage memory
                            gc.collect()
                            
                        print(f"Processed row {y//self.chunk_size + 1}/{height//self.chunk_size + 1}")
                        
        except Exception as e:
            print(f"Error creating chunked file: {e}")
            raise
    
    def _analyze_image(self):
        """Analyze the loaded image to extract metadata."""
        if self.current_image is None:
            return
            
        if hasattr(self.current_image, 'shape'):
            self.image_shape = self.current_image.shape
            if len(self.image_shape) == 3:
                self.channels = self.image_shape[2]
            else:
                self.channels = 1
            self.dtype = self.current_image.dtype
        else:
            # Dask array
            self.image_shape = self.current_image.shape
            self.channels = self.image_shape[2] if len(self.image_shape) == 3 else 1
            self.dtype = self.current_image.dtype
    
    def get_chunk(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Get a specific chunk of the image.
        
        Args:
            x, y: Top-left coordinates
            width, height: Chunk dimensions
            
        Returns:
            numpy array of the requested chunk
        """
        if not self.is_loaded:
            return None
            
        try:
            if hasattr(self.current_image, 'compute'):
                # Dask array
                chunk = self.current_image[y:y+height, x:x+width].compute()
            else:
                # Regular numpy array
                chunk = self.current_image[y:y+height, x:x+width]
            
            return chunk
            
        except Exception as e:
            print(f"Error getting chunk: {e}")
            return None
    
    def get_thumbnail(self, max_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
        """
        Get a thumbnail of the image.
        
        Args:
            max_size: Maximum dimensions for the thumbnail
            
        Returns:
            Thumbnail as numpy array
        """
        if not self.is_loaded:
            return None
            
        try:
            height, width = self.image_shape[:2]
            
            # Calculate scaling factor
            scale_x = max_size[0] / width
            scale_y = max_size[1] / height
            scale = min(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if hasattr(self.current_image, 'compute'):
                # For dask arrays, sample the image
                step_x = max(1, width // new_width)
                step_y = max(1, height // new_height)
                thumbnail = self.current_image[::step_y, ::step_x].compute()
            else:
                # For regular arrays, use OpenCV resize
                thumbnail = cv2.resize(self.current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return thumbnail
            
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    def get_channel(self, channel_index: int) -> Optional[np.ndarray]:
        """
        Get a specific channel of the image.
        
        Args:
            channel_index: Index of the channel to extract
            
        Returns:
            Channel data as numpy array
        """
        if not self.is_loaded or channel_index >= self.channels:
            return None
            
        try:
            if hasattr(self.current_image, 'compute'):
                # Dask array
                if self.channels == 1:
                    return self.current_image.compute()
                else:
                    return self.current_image[:, :, channel_index].compute()
            else:
                # Regular numpy array
                if self.channels == 1:
                    return self.current_image
                else:
                    return self.current_image[:, :, channel_index]
                    
        except Exception as e:
            print(f"Error getting channel: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        self.current_image = None
        self.is_loaded = False
        gc.collect()
        
        # Clean up temporary files
        if self.image_path:
            temp_dir = os.path.join(os.path.dirname(self.image_path), 'temp_mmap')
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
