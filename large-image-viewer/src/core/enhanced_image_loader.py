"""
Enhanced large image loader for handling very large image files (up to 5GB+) efficiently.
Supports standard formats (PNG, JPEG, TIFF) and medical formats (DICOM).
"""

import os
import mmap
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import threading
import time
from pathlib import Path

try:
    import h5py
    import zarr
    import dask.array as da
    from dask.array import Array as DaskArray
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image, ImageFile
    # Enable loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import pydicom
    import SimpleITK as sitk
    import nibabel as nib
    HAS_DICOM = True
except ImportError:
    HAS_DICOM = False

try:
    import lz4.frame
    import blosc
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

try:
    import large_image
    HAS_LARGE_IMAGE = True
except ImportError:
    HAS_LARGE_IMAGE = False

import psutil
import gc

class RegionSelector:
    """Helper class for selecting and extracting image regions."""
    
    def __init__(self, image_shape: Tuple[int, ...]):
        self.image_shape = image_shape
        self.selected_regions = []
    
    def add_region(self, x: int, y: int, width: int, height: int, name: str = None):
        """Add a region for extraction."""
        # Clamp to image bounds
        x = max(0, min(x, self.image_shape[1] - 1))
        y = max(0, min(y, self.image_shape[0] - 1))
        width = min(width, self.image_shape[1] - x)
        height = min(height, self.image_shape[0] - y)
        
        region = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'name': name or f"Region_{len(self.selected_regions) + 1}"
        }
        self.selected_regions.append(region)
        return len(self.selected_regions) - 1
    
    def get_region_bounds(self, region_id: int) -> Tuple[int, int, int, int]:
        """Get region bounds as (x, y, width, height)."""
        if 0 <= region_id < len(self.selected_regions):
            region = self.selected_regions[region_id]
            return region['x'], region['y'], region['width'], region['height']
        return None
    
    def clear_regions(self):
        """Clear all selected regions."""
        self.selected_regions.clear()

class EnhancedLargeImageLoader:
    """
    Enhanced loader for very large images (up to 5GB+) with DICOM support.
    """
    
    SUPPORTED_FORMATS = {
        '.png': {'format': 'png', 'type': 'standard'},
        '.jpg': {'format': 'jpeg', 'type': 'standard'}, 
        '.jpeg': {'format': 'jpeg', 'type': 'standard'},
        '.tif': {'format': 'tiff', 'type': 'standard'}, 
        '.tiff': {'format': 'tiff', 'type': 'standard'},
        '.bmp': {'format': 'bmp', 'type': 'standard'},
        '.dcm': {'format': 'dicom', 'type': 'medical'}, 
        '.dicom': {'format': 'dicom', 'type': 'medical'},
        '.nii': {'format': 'nifti', 'type': 'medical'}, 
        '.nii.gz': {'format': 'nifti', 'type': 'medical'},
        '.hdr': {'format': 'analyze', 'type': 'medical'}, 
        '.img': {'format': 'analyze', 'type': 'medical'},
        '.svs': {'format': 'svs', 'type': 'wsi'},  # Aperio ScanScope Virtual Slide
        '.ndpi': {'format': 'ndpi', 'type': 'wsi'},  # Hamamatsu NanoZoomer
        '.vms': {'format': 'vms', 'type': 'wsi'}, 
        '.vmu': {'format': 'vmu', 'type': 'wsi'},  # Hamamatsu VMU/VMS
        '.scn': {'format': 'scn', 'type': 'wsi'},  # Leica SCN
        '.mrxs': {'format': 'mrxs', 'type': 'wsi'},  # MIRAX
        '.bif': {'format': 'bif', 'type': 'wsi'},  # Ventana BIF
    }
    
    def __init__(self, chunk_size: int = 512, max_memory_gb: float = 8.0, 
                 cache_size_gb: float = 2.0, enable_compression: bool = True):
        """
        Initialize the enhanced image loader.
        
        Args:
            chunk_size: Size of image chunks for processing (reduced for 5GB+ files)
            max_memory_gb: Maximum memory to use for image loading
            cache_size_gb: Size of the tile cache
            enable_compression: Enable memory compression for large files
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.cache_size_gb = cache_size_gb
        self.enable_compression = enable_compression and HAS_COMPRESSION
        
        # Image data
        self.current_image = None
        self.image_path = None
        self.image_shape = None
        self.channels = None
        self.dtype = None
        self.is_loaded = False
        self.file_format = None
        self.metadata = {}
        
        # Memory mapping for very large files
        self.memory_mapped_file = None
        self.dask_array = None
        
        # WSI and SVS-specific attributes
        self.slide_properties = {}
        self.zoom_levels = []
        self.openslide_handle = None
        self.large_image_source = None
        
        # Region selection
        self.region_selector = None
        
        # Tile cache for efficient navigation
        self.tile_cache = {}
        self.cache_access_times = {}
        
        # Performance monitoring
        self.load_time = 0
        self.memory_usage = 0
        
    def detect_format(self, file_path: str) -> str:
        """Detect the image format from file extension and content."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[ext]['format']
        
        # Try to detect from content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            # PNG signature
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png'
            # JPEG signature
            elif header.startswith(b'\xff\xd8\xff'):
                return 'jpeg'
            # TIFF signatures
            elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
                return 'tiff'
            # DICOM signature
            elif b'DICM' in header or self._is_dicom_file(file_path):
                return 'dicom'
            # Check for SVS/WSI files by extension since they're TIFF-based
            elif ext in ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']:
                return ext[1:]  # Return format without dot
                
        except Exception:
            pass
        
        return 'unknown'
    
    def _is_dicom_file(self, file_path: str) -> bool:
        """Check if file is a DICOM file."""
        if not HAS_DICOM:
            return False
        
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except Exception:
            return False
    
    def _is_wsi_file(self, file_path: str) -> bool:
        """Check if file is a Whole Slide Image file."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            wsi_extensions = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']
            return ext in wsi_extensions
        except Exception:
            return False
    
    def load_image(self, file_path: str) -> bool:
        """
        Load an image file efficiently with enhanced support for large files.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if loading successful
        """
        start_time = time.time()
        
        try:
            # Clean up previous image
            self.cleanup()
            
            # Check file size and available memory
            file_size = os.path.getsize(file_path)
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            print(f"Loading file: {file_path}")
            print(f"File size: {file_size / (1024**3):.2f} GB")
            print(f"Available memory: {available_memory:.2f} GB")
            
            # Detect format
            self.file_format = self.detect_format(file_path)
            print(f"Detected format: {self.file_format}")
            
            # Choose loading strategy based on file format and size
            if self.file_format in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'bif']:
                # Always use WSI loader for WSI files regardless of size
                success = self._load_wsi_file(file_path)
            elif self.file_format == 'dicom':
                if file_size > 2 * 1024**3:  # > 2GB
                    success = self._load_dicom_large(file_path)
                else:
                    success = self._load_standard_file(file_path)
            elif self.file_format == 'tiff' and HAS_TIFFFILE:
                if file_size > 2 * 1024**3:  # > 2GB
                    success = self._load_tiff_large(file_path)
                else:
                    success = self._load_standard_file(file_path)
            elif self.file_format in ['png', 'jpeg']:
                if file_size > 1 * 1024**3:  # > 1GB for PNG/JPEG
                    success = self._load_standard_large(file_path)
                else:
                    success = self._load_standard_file(file_path)
            else:
                success = self._load_standard_file(file_path)
            
            if success:
                self.image_path = file_path
                self.is_loaded = True
                self.region_selector = RegionSelector(self.image_shape)
                self.load_time = time.time() - start_time
                self.memory_usage = self._get_memory_usage()
                
                print(f"Load time: {self.load_time:.2f} seconds")
                print(f"Memory usage: {self.memory_usage:.2f} GB")
                
            return success
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def _load_large_file(self, file_path: str, file_size: int) -> bool:
        """Load very large files (2GB+) using memory mapping and chunked access."""
        try:
            if self.file_format == 'dicom':
                return self._load_dicom_large(file_path)
            elif self.file_format in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'bif']:
                return self._load_wsi_file(file_path)
            elif self.file_format == 'tiff' and HAS_TIFFFILE:
                return self._load_tiff_large(file_path)
            elif self.file_format in ['png', 'jpeg']:
                return self._load_standard_large(file_path)
            else:
                return self._load_standard_file(file_path)
                
        except Exception as e:
            print(f"Error in large file loading: {e}")
            return False
    
    def _load_wsi_file(self, file_path: str) -> bool:
        """Load whole slide imaging files (SVS, NDPI, etc.) efficiently."""
        print(f"Attempting to load WSI file: {file_path}")
        print(f"OpenSlide available: {HAS_OPENSLIDE}")
        print(f"Large-image available: {HAS_LARGE_IMAGE}")
        
        if not HAS_OPENSLIDE and not HAS_LARGE_IMAGE:
            print("Error: No WSI libraries available")
            raise ImportError("WSI support not available. Install openslide-python and/or large-image.")
        
        try:
            # Try OpenSlide first since we have it available
            if HAS_OPENSLIDE:
                print("Trying OpenSlide...")
                return self._load_with_openslide(file_path)
            # Fallback to large-image
            elif HAS_LARGE_IMAGE:
                print("Trying large-image...")
                return self._load_with_large_image(file_path)
            else:
                print("No WSI loader available")
                return False
                
        except Exception as e:
            print(f"Error loading WSI file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_with_large_image(self, file_path: str) -> bool:
        """Load WSI using large-image library."""
        try:
            # Open with large-image
            self.large_image_source = large_image.open(file_path)
            
            # Get basic properties
            metadata = self.large_image_source.getMetadata()
            
            self.image_shape = (metadata['sizeY'], metadata['sizeX'])
            self.channels = metadata.get('bandCount', 3)
            self.dtype = np.uint8  # Most WSI are 8-bit
            
            # Get available zoom levels
            self.zoom_levels = []
            for level in range(metadata['levels']):
                level_info = self.large_image_source.getMetadata(level)
                self.zoom_levels.append({
                    'level': level,
                    'width': level_info['sizeX'],
                    'height': level_info['sizeY'],
                    'downsample': metadata['magnification'] / level_info.get('magnification', metadata['magnification'])
                })
            
            # Load a thumbnail for initial display
            thumbnail_level = min(len(self.zoom_levels) - 1, 3)  # Use level 3 or highest available
            tile_info = self.large_image_source.getMetadata(thumbnail_level)
            
            # Get thumbnail region
            thumbnail_data, _ = self.large_image_source.getRegion(
                region=dict(left=0, top=0, width=tile_info['sizeX'], height=tile_info['sizeY']),
                format=large_image.constants.TILE_FORMAT_NUMPY,
                level=thumbnail_level
            )
            
            self.current_image = thumbnail_data
            
            # Store slide properties
            self.slide_properties = {
                'vendor': metadata.get('vendor', 'Unknown'),
                'magnification': metadata.get('magnification'),
                'mpp_x': metadata.get('mm_x'),
                'mpp_y': metadata.get('mm_y'),
                'levels': metadata['levels'],
                'tile_width': metadata.get('tileWidth', 256),
                'tile_height': metadata.get('tileHeight', 256)
            }
            
            return True
            
        except Exception as e:
            print(f"Error loading with large-image: {e}")
            return False
    
    def _load_with_openslide(self, file_path: str) -> bool:
        """Load WSI using OpenSlide library."""
        try:
            print(f"Opening {file_path} with OpenSlide...")
            
            # Open with OpenSlide
            self.openslide_handle = openslide.OpenSlide(file_path)
            print(f"Successfully opened with OpenSlide")
            
            # Get dimensions
            level_0_dims = self.openslide_handle.level_dimensions[0]
            self.image_shape = (level_0_dims[1], level_0_dims[0])  # height, width
            self.channels = 3  # Most WSI are RGB
            self.dtype = np.uint8
            
            print(f"Image dimensions: {self.image_shape}")
            print(f"Available levels: {len(self.openslide_handle.level_dimensions)}")
            
            # Get zoom levels
            self.zoom_levels = []
            for level, dims in enumerate(self.openslide_handle.level_dimensions):
                downsample = self.openslide_handle.level_downsamples[level]
                self.zoom_levels.append({
                    'level': level,
                    'width': dims[0],
                    'height': dims[1],
                    'downsample': downsample
                })
                print(f"Level {level}: {dims[0]}x{dims[1]}, downsample: {downsample}")
            
            # Load thumbnail from appropriate level for initial display
            thumbnail_level = min(len(self.zoom_levels) - 1, 3)
            if thumbnail_level >= len(self.openslide_handle.level_dimensions):
                thumbnail_level = len(self.openslide_handle.level_dimensions) - 1
                
            thumbnail_size = self.openslide_handle.level_dimensions[thumbnail_level]
            print(f"Loading thumbnail from level {thumbnail_level}, size: {thumbnail_size}")
            
            # Read thumbnail region - limit size to prevent memory issues
            max_thumbnail_size = 1024
            if thumbnail_size[0] > max_thumbnail_size or thumbnail_size[1] > max_thumbnail_size:
                # Calculate smaller thumbnail size
                scale = min(max_thumbnail_size / thumbnail_size[0], max_thumbnail_size / thumbnail_size[1])
                new_width = int(thumbnail_size[0] * scale)
                new_height = int(thumbnail_size[1] * scale)
                
                # Read from level 0 and resize
                thumbnail_pil = self.openslide_handle.read_region(
                    (0, 0), 0, (int(thumbnail_size[0] / self.openslide_handle.level_downsamples[thumbnail_level]),
                               int(thumbnail_size[1] / self.openslide_handle.level_downsamples[thumbnail_level]))
                )
                thumbnail_pil = thumbnail_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Read thumbnail region
                thumbnail_pil = self.openslide_handle.read_region(
                    (0, 0), thumbnail_level, thumbnail_size
                )
            
            print(f"Thumbnail size: {thumbnail_pil.size}")
            
            # Convert to RGB (remove alpha if present)
            if thumbnail_pil.mode == 'RGBA':
                thumbnail_pil = thumbnail_pil.convert('RGB')
            
            self.current_image = np.array(thumbnail_pil)
            print(f"Thumbnail converted to numpy array: {self.current_image.shape}")
            
            # Store slide properties
            properties = self.openslide_handle.properties
            self.slide_properties = {
                'vendor': properties.get('openslide.vendor', 'Unknown'),
                'magnification': properties.get('openslide.objective-power'),
                'mpp_x': properties.get('openslide.mpp-x'),
                'mpp_y': properties.get('openslide.mpp-y'),
                'levels': len(self.openslide_handle.level_dimensions),
                'comment': properties.get('openslide.comment', '')
            }
            
            print("WSI loading completed successfully")
            return True
            
        except Exception as e:
            print(f"Error loading with OpenSlide: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_dicom_large(self, file_path: str) -> bool:
        """Load large DICOM files efficiently."""
        if not HAS_DICOM:
            raise ImportError("DICOM support not available. Install pydicom and SimpleITK.")
        
        try:
            # Try SimpleITK for large DICOM files
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            
            # Convert to numpy array
            if array.ndim == 2:
                self.current_image = array
                self.image_shape = array.shape
                self.channels = 1
            elif array.ndim == 3:
                # Handle 3D DICOM data
                if array.shape[0] == 1:
                    # Single slice
                    self.current_image = array[0]
                    self.image_shape = array[0].shape
                    self.channels = 1
                else:
                    # Multiple slices - take middle slice for preview
                    middle_slice = array.shape[0] // 2
                    self.current_image = array[middle_slice]
                    self.image_shape = array[middle_slice].shape
                    self.channels = array.shape[0]  # Number of slices
            
            self.dtype = array.dtype
            
            # Store metadata
            self.metadata = {
                'spacing': image.GetSpacing(),
                'origin': image.GetOrigin(),
                'direction': image.GetDirection(),
                'size': image.GetSize()
            }
            
            # Create dask array for large files
            if HAS_DASK and array.nbytes > 1024**3:  # > 1GB
                chunk_shape = (min(self.chunk_size, array.shape[0]),) + array.shape[1:]
                self.dask_array = da.from_array(array, chunks=chunk_shape)
            
            return True
            
        except Exception as e:
            print(f"Error loading DICOM with SimpleITK: {e}")
            
            # Fallback to pydicom
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'pixel_array'):
                    array = ds.pixel_array
                    self.current_image = array
                    self.image_shape = array.shape
                    self.channels = 1 if array.ndim == 2 else array.shape[-1]
                    self.dtype = array.dtype
                    
                    # Store DICOM metadata
                    self.metadata = {
                        'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                        'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                        'modality': getattr(ds, 'Modality', 'Unknown'),
                        'pixel_spacing': getattr(ds, 'PixelSpacing', None)
                    }
                    
                    return True
                    
            except Exception as e2:
                print(f"Error loading DICOM with pydicom: {e2}")
                return False
    
    def _load_tiff_large(self, file_path: str) -> bool:
        """Load large TIFF files using tifffile."""
        if not HAS_TIFFFILE:
            raise ImportError("TIFF support not available. Install tifffile.")
        
        try:
            # Use tifffile for better large TIFF support
            with tifffile.TiffFile(file_path) as tif:
                # Get basic info without loading full image
                series = tif.series[0]
                self.image_shape = series.shape
                self.dtype = series.dtype
                
                # Handle different TIFF structures
                if len(series.shape) == 2:
                    self.channels = 1
                elif len(series.shape) == 3:
                    if series.shape[-1] <= 4:  # Likely RGB/RGBA
                        self.channels = series.shape[-1]
                    else:
                        self.channels = series.shape[0]  # Multi-page TIFF
                
                # For very large files, use memory mapping
                if series.nbytes > 2 * 1024**3:  # > 2GB
                    # Create memory-mapped array
                    self.current_image = tif.asarray(out='memmap')
                else:
                    self.current_image = tif.asarray()
                
                # Create dask array for chunked processing
                if HAS_DASK and series.nbytes > 1024**3:  # > 1GB
                    if len(series.shape) == 2:
                        chunk_shape = (self.chunk_size, self.chunk_size)
                    else:
                        chunk_shape = (1,) + (self.chunk_size, self.chunk_size)
                    
                    self.dask_array = da.from_array(self.current_image, chunks=chunk_shape)
                
                return True
                
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            return False
    
    def _load_standard_large(self, file_path: str) -> bool:
        """Load large standard format files using chunked reading."""
        try:
            print(f"Loading large standard file: {file_path}")
            
            # For PNG/JPEG files larger than 1GB, try chunked loading
            if HAS_PIL:
                # Temporarily disable PIL's decompression bomb protection
                original_max_pixels = Image.MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = None
                
                try:
                    with Image.open(file_path) as img:
                        # Get basic info
                        self.image_shape = (img.height, img.width)
                        self.channels = len(img.getbands())
                        
                        print(f"Image info: {self.image_shape}, channels: {self.channels}")
                        
                        # Convert to appropriate mode
                        if img.mode in ['RGB', 'RGBA']:
                            self.channels = len(img.mode)
                        elif img.mode == 'L':
                            self.channels = 1
                        
                        # For very large images, check available memory
                        image_size_bytes = img.width * img.height * self.channels
                        available_memory = psutil.virtual_memory().available
                        
                        print(f"Image size: {image_size_bytes / (1024**3):.2f} GB")
                        print(f"Available memory: {available_memory / (1024**3):.2f} GB")
                        
                        if image_size_bytes > available_memory * 0.4:  # Use max 40% of available memory
                            print("Image too large for memory, loading as thumbnail")
                            # Load a smaller version
                            max_size = int(np.sqrt(available_memory * 0.3 / self.channels))  # Use 30% of memory
                            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            self.image_shape = (img.height, img.width)
                            print(f"Thumbnail size: {self.image_shape}")
                        
                        # Load the image
                        try:
                            self.current_image = np.array(img)
                            print(f"Successfully loaded image: {self.current_image.shape}")
                            self.dtype = self.current_image.dtype
                            return True
                        except MemoryError:
                            print("Memory error, trying smaller thumbnail")
                            # Try even smaller thumbnail
                            max_size = int(np.sqrt(available_memory * 0.1 / self.channels))
                            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            self.current_image = np.array(img)
                            self.image_shape = (img.height, img.width)
                            self.dtype = self.current_image.dtype
                            print(f"Loaded smaller thumbnail: {self.current_image.shape}")
                            return True
                            
                finally:
                    # Restore original limit
                    Image.MAX_IMAGE_PIXELS = original_max_pixels
            
            return False
            
        except Exception as e:
            print(f"Error loading standard large file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_image_chunked(self, pil_image):
        """Load a PIL image in chunks to manage memory."""
        width, height = pil_image.size
        channels = len(pil_image.getbands())
        
        # Calculate chunk size based on available memory
        available_memory = psutil.virtual_memory().available
        max_chunk_pixels = min(
            self.chunk_size * self.chunk_size,
            available_memory // (channels * 4)  # Assume 4 bytes per pixel
        )
        
        chunk_height = min(height, int(np.sqrt(max_chunk_pixels / width)))
        chunk_height = max(1, chunk_height)
        
        # Pre-allocate result array
        if channels == 1:
            result = np.zeros((height, width), dtype=np.uint8)
        else:
            result = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Load in chunks
        for y in range(0, height, chunk_height):
            chunk_h = min(chunk_height, height - y)
            box = (0, y, width, y + chunk_h)
            
            chunk = pil_image.crop(box)
            chunk_array = np.array(chunk)
            
            if channels == 1:
                result[y:y+chunk_h, :] = chunk_array
            else:
                result[y:y+chunk_h, :, :] = chunk_array
            
            # Force garbage collection
            del chunk, chunk_array
            gc.collect()
        
        return result
    
    def _load_standard_file(self, file_path: str) -> bool:
        """Load standard size files using conventional methods."""
        try:
            if self.file_format == 'dicom':
                return self._load_dicom_standard(file_path)
            elif HAS_PIL:
                return self._load_pil_standard(file_path)
            elif HAS_OPENCV:
                return self._load_opencv_standard(file_path)
            else:
                return False
                
        except Exception as e:
            print(f"Error loading standard file: {e}")
            return False
    
    def _load_dicom_standard(self, file_path: str) -> bool:
        """Load standard DICOM files."""
        if not HAS_DICOM:
            return False
        
        try:
            ds = pydicom.dcmread(file_path)
            if hasattr(ds, 'pixel_array'):
                array = ds.pixel_array
                self.current_image = array
                self.image_shape = array.shape
                self.channels = 1 if array.ndim == 2 else array.shape[-1]
                self.dtype = array.dtype
                
                # Store DICOM metadata
                self.metadata = {
                    'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                    'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                    'modality': getattr(ds, 'Modality', 'Unknown'),
                    'pixel_spacing': getattr(ds, 'PixelSpacing', None)
                }
                
                return True
        except Exception:
            return False
    
    def _load_pil_standard(self, file_path: str) -> bool:
        """Load using PIL."""
        if not HAS_PIL:
            print("PIL not available")
            return False
        
        try:
            print(f"Opening {file_path} with PIL...")
            
            # Temporarily increase PIL's decompression bomb limit for large images
            original_max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
            
            try:
                with Image.open(file_path) as img:
                    print(f"Image mode: {img.mode}, size: {img.size}")
                    
                    # Check if image is too large for memory
                    image_size_bytes = img.width * img.height * len(img.getbands())
                    available_memory = psutil.virtual_memory().available
                    
                    print(f"Image memory requirement: {image_size_bytes / (1024**3):.2f} GB")
                    print(f"Available memory: {available_memory / (1024**3):.2f} GB")
                    
                    # If image is too large, load a thumbnail instead
                    if image_size_bytes > available_memory * 0.5:  # Use max 50% of available memory
                        print("Image too large for memory, creating thumbnail...")
                        max_size = int(np.sqrt(available_memory * 0.3 / len(img.getbands())))
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        print(f"Thumbnail size: {img.size}")
                    
                    # Convert to numpy array
                    print("Converting to numpy array...")
                    self.current_image = np.array(img)
                    print(f"Numpy array shape: {self.current_image.shape}")
                    
                    self.image_shape = self.current_image.shape[:2]
                    self.channels = 1 if len(self.current_image.shape) == 2 else self.current_image.shape[2]
                    self.dtype = self.current_image.dtype
                    
                    print(f"Final image info - Shape: {self.image_shape}, Channels: {self.channels}, Dtype: {self.dtype}")
                    return True
                    
            finally:
                # Restore original limit
                Image.MAX_IMAGE_PIXELS = original_max_pixels
                
        except Exception as e:
            print(f"Error in PIL loading: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_opencv_standard(self, file_path: str) -> bool:
        """Load using OpenCV."""
        if not HAS_OPENCV:
            print("OpenCV not available")
            return False
        
        try:
            print(f"Opening {file_path} with OpenCV...")
            
            # Try color first, then grayscale
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                print("Color loading failed, trying grayscale...")
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                print(f"OpenCV image shape: {img.shape}")
                
                if len(img.shape) == 3:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.channels = 3
                else:
                    self.channels = 1
                
                self.current_image = img
                self.image_shape = img.shape[:2]
                self.dtype = img.dtype
                
                print(f"OpenCV loading successful - Shape: {self.image_shape}, Channels: {self.channels}")
                return True
            else:
                print("OpenCV failed to load image")
                return False
                
        except Exception as e:
            print(f"Error in OpenCV loading: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        Extract a specific region from the image.
        
        Args:
            x, y: Top-left coordinates
            width, height: Region dimensions
            
        Returns:
            numpy array of the region or None if invalid
        """
        if not self.is_loaded:
            return None
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, self.image_shape[1] - 1))
        y = max(0, min(y, self.image_shape[0] - 1))
        width = min(width, self.image_shape[1] - x)
        height = min(height, self.image_shape[0] - y)
        
        try:
            if self.dask_array is not None:
                # Use dask for efficient region extraction
                if len(self.image_shape) == 2:
                    region = self.dask_array[y:y+height, x:x+width].compute()
                else:
                    region = self.dask_array[y:y+height, x:x+width, :].compute()
            else:
                # Direct array slicing
                if len(self.image_shape) == 2:
                    region = self.current_image[y:y+height, x:x+width]
                else:
                    region = self.current_image[y:y+height, x:x+width, :]
            
            return region
            
        except Exception as e:
            print(f"Error extracting region: {e}")
            return None
    
    def save_region(self, x: int, y: int, width: int, height: int, 
                   output_path: str, format: str = None) -> bool:
        """
        Save a specific region of the image to file.
        
        Args:
            x, y: Top-left coordinates
            width, height: Region dimensions
            output_path: Output file path
            format: Output format (auto-detected if None)
            
        Returns:
            bool: True if save successful
        """
        region = self.get_region(x, y, width, height)
        if region is None:
            return False
        
        try:
            # Detect output format
            if format is None:
                ext = Path(output_path).suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    format = 'JPEG'
                elif ext == '.png':
                    format = 'PNG'
                elif ext in ['.tif', '.tiff']:
                    format = 'TIFF'
                elif ext == '.bmp':
                    format = 'BMP'
                else:
                    format = 'PNG'  # Default
            
            # Convert to PIL Image and save
            if HAS_PIL:
                if len(region.shape) == 2:
                    # Grayscale
                    img = Image.fromarray(region, mode='L')
                elif region.shape[2] == 3:
                    # RGB
                    img = Image.fromarray(region, mode='RGB')
                elif region.shape[2] == 4:
                    # RGBA
                    img = Image.fromarray(region, mode='RGBA')
                else:
                    # Multi-channel - save as grayscale of first channel
                    img = Image.fromarray(region[:, :, 0], mode='L')
                
                img.save(output_path, format=format)
                return True
            
            # Fallback to OpenCV
            elif HAS_OPENCV:
                if len(region.shape) == 3 and region.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                
                return cv2.imwrite(output_path, region)
            
            return False
            
        except Exception as e:
            print(f"Error saving region: {e}")
            return False
    
    def get_thumbnail(self, size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
        """
        Get a thumbnail of the image.
        
        Args:
            size: Thumbnail size as (width, height)
            
        Returns:
            numpy array thumbnail or None if failed
        """
        if not self.is_loaded:
            return None
        
        try:
            # For WSI files, return the already loaded thumbnail
            if hasattr(self, 'openslide_handle') or hasattr(self, 'large_image_source'):
                if self.current_image is not None:
                    return self.current_image
                else:
                    # Generate thumbnail for WSI
                    return self.get_wsi_thumbnail(size)
            
            # For regular images
            if self.current_image is None:
                return None
            
            # Calculate scaling factor
            scale_x = size[0] / self.image_shape[1]
            scale_y = size[1] / self.image_shape[0]
            scale = min(scale_x, scale_y)
            
            new_width = int(self.image_shape[1] * scale)
            new_height = int(self.image_shape[0] * scale)
            
            if self.dask_array is not None and HAS_DASK:
                # Use dask for efficient downsampling
                if len(self.image_shape) == 2:
                    # Simple downsampling for large arrays
                    step_y = max(1, self.image_shape[0] // new_height)
                    step_x = max(1, self.image_shape[1] // new_width)
                    thumbnail = self.dask_array[::step_y, ::step_x].compute()
                else:
                    step_y = max(1, self.image_shape[0] // new_height)
                    step_x = max(1, self.image_shape[1] // new_width)
                    thumbnail = self.dask_array[::step_y, ::step_x, :].compute()
                    step_x = max(1, self.image_shape[1] // new_width)
                    thumbnail = self.dask_array[::step_y, ::step_x, :].compute()
            else:
                # Use current image for thumbnail
                if HAS_OPENCV and len(self.current_image.shape) in [2, 3]:
                    thumbnail = cv2.resize(self.current_image, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
                elif HAS_PIL:
                    if len(self.current_image.shape) == 2:
                        img = Image.fromarray(self.current_image, mode='L')
                    else:
                        img = Image.fromarray(self.current_image)
                    
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    thumbnail = np.array(img_resized)
                else:
                    # Simple numpy-based downsampling
                    step_y = max(1, self.image_shape[0] // new_height)
                    step_x = max(1, self.image_shape[1] // new_width)
                    if len(self.current_image.shape) == 2:
                        thumbnail = self.current_image[::step_y, ::step_x]
                    else:
                        thumbnail = self.current_image[::step_y, ::step_x, :]
            
            return thumbnail
            
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    def get_channel(self, channel_index: int) -> Optional[np.ndarray]:
        """
        Get a specific channel from the image.
        
        Args:
            channel_index: Index of the channel to extract
            
        Returns:
            numpy array of the channel or None if invalid
        """
        if not self.is_loaded:
            return None
        
        try:
            if len(self.current_image.shape) == 2:
                # Grayscale image
                if channel_index == 0:
                    return self.current_image
                else:
                    return None
            elif len(self.current_image.shape) == 3:
                # Multi-channel image
                if 0 <= channel_index < self.current_image.shape[2]:
                    return self.current_image[:, :, channel_index]
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting channel: {e}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get image metadata."""
        base_metadata = {
            'file_path': self.image_path,
            'format': self.file_format,
            'shape': self.image_shape,
            'channels': self.channels,
            'dtype': str(self.dtype),
            'load_time': self.load_time,
            'memory_usage_gb': self.memory_usage
        }
        
        # Add format-specific metadata
        base_metadata.update(self.metadata)
        
        return base_metadata
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def get_wsi_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """Get a specific region from WSI at specified zoom level."""
        if not hasattr(self, 'openslide_handle') and not hasattr(self, 'large_image_source'):
            raise ValueError("No WSI file loaded")
        
        try:
            if hasattr(self, 'large_image_source') and self.large_image_source is not None:
                # Use large-image
                region_data, _ = self.large_image_source.getRegion(
                    region=dict(left=x, top=y, width=width, height=height),
                    format=large_image.constants.TILE_FORMAT_NUMPY,
                    level=level
                )
                return region_data
                
            elif hasattr(self, 'openslide_handle') and self.openslide_handle is not None:
                # Use OpenSlide
                pil_img = self.openslide_handle.read_region((x, y), level, (width, height))
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                return np.array(pil_img)
                
        except Exception as e:
            print(f"Error getting WSI region: {e}")
            return None
    
    def get_tile(self, level: int, tile_x: int, tile_y: int, tile_size: int = 256) -> Optional[np.ndarray]:
        """Get a specific tile from the image at the given level."""
        try:
            if hasattr(self, 'large_image_source') and self.large_image_source is not None:
                # Use large-image for WSI files
                metadata = self.large_image_source.getMetadata()
                if level >= metadata['levels']:
                    return None
                
                # Calculate tile position in image coordinates
                x = tile_x * tile_size
                y = tile_y * tile_size
                
                # Get level metadata
                level_meta = self.large_image_source.getMetadata(level)
                
                # Clamp tile to image bounds
                width = min(tile_size, level_meta['sizeX'] - x)
                height = min(tile_size, level_meta['sizeY'] - y)
                
                if width <= 0 or height <= 0:
                    return None
                
                # Get tile data
                tile_data, _ = self.large_image_source.getRegion(
                    region=dict(left=x, top=y, width=width, height=height),
                    format=large_image.constants.TILE_FORMAT_NUMPY,
                    level=level
                )
                return tile_data
                
            elif hasattr(self, 'openslide_handle') and self.openslide_handle is not None:
                # Use OpenSlide for WSI files
                if level >= len(self.openslide_handle.level_dimensions):
                    return None
                
                # Calculate tile position
                x = tile_x * tile_size
                y = tile_y * tile_size
                
                # Get level dimensions
                level_dims = self.openslide_handle.level_dimensions[level]
                
                # Clamp tile to image bounds
                width = min(tile_size, level_dims[0] - x)
                height = min(tile_size, level_dims[1] - y)
                
                if width <= 0 or height <= 0:
                    return None
                
                try:
                    # Read tile - OpenSlide returns PIL Image in RGBA format
                    pil_img = self.openslide_handle.read_region((x, y), level, (width, height))
                    
                    # Convert RGBA to RGB (remove alpha channel)
                    if pil_img.mode == 'RGBA':
                        # Create white background
                        rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                        rgb_img.paste(pil_img, mask=pil_img.split()[3])  # Use alpha as mask
                        pil_img = rgb_img
                    elif pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    # Convert to numpy array
                    tile_array = np.array(pil_img)
                    
                    # Ensure we have the right shape and data type
                    if len(tile_array.shape) == 3 and tile_array.shape[2] == 3:
                        return tile_array.astype(np.uint8)
                    else:
                        print(f"Unexpected tile shape: {tile_array.shape}")
                        return None
                        
                except Exception as e:
                    print(f"Error reading OpenSlide tile: {e}")
                    return None
                
            elif self.current_image is not None:
                # For regular images, simulate tiling by extracting regions
                if level > 0:
                    # For higher levels, downsample the image
                    downsample = 2 ** level
                    downsampled_shape = (
                        self.image_shape[0] // downsample,
                        self.image_shape[1] // downsample
                    )
                    
                    if len(self.current_image.shape) == 3:
                        downsampled_shape = downsampled_shape + (self.current_image.shape[2],)
                    
                    # Simple downsampling (could be improved with proper filtering)
                    if len(self.current_image.shape) == 2:
                        downsampled = self.current_image[::downsample, ::downsample]
                    else:
                        downsampled = self.current_image[::downsample, ::downsample, :]
                else:
                    downsampled = self.current_image
                    downsampled_shape = self.image_shape
                
                # Calculate tile position
                x = tile_x * tile_size
                y = tile_y * tile_size
                
                # Clamp to image bounds
                height, width = downsampled_shape[:2]
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                if tile_width <= 0 or tile_height <= 0:
                    return None
                
                # Extract tile
                if len(downsampled.shape) == 2:
                    tile = downsampled[y:y+tile_height, x:x+tile_width]
                else:
                    tile = downsampled[y:y+tile_height, x:x+tile_width, :]
                
                return tile
                
        except Exception as e:
            print(f"Error getting tile ({level}, {tile_x}, {tile_y}): {e}")
            return None
        
        return None
    
    def get_wsi_thumbnail(self, max_size: tuple = (1024, 1024)) -> np.ndarray:
        """Get WSI thumbnail that fits within max_size."""
        if not hasattr(self, 'openslide_handle') and not hasattr(self, 'large_image_source'):
            raise ValueError("No WSI file loaded")
        
        try:
            if hasattr(self, 'large_image_source') and self.large_image_source is not None:
                # Use large-image
                metadata = self.large_image_source.getMetadata()
                
                # Calculate appropriate level
                scale = min(max_size[0] / metadata['sizeX'], max_size[1] / metadata['sizeY'])
                target_level = 0
                for level in range(metadata['levels']):
                    level_meta = self.large_image_source.getMetadata(level)
                    level_scale = level_meta['sizeX'] / metadata['sizeX']
                    if level_scale <= scale:
                        target_level = level
                        break
                
                # Get thumbnail
                level_meta = self.large_image_source.getMetadata(target_level)
                thumbnail_data, _ = self.large_image_source.getRegion(
                    region=dict(left=0, top=0, width=level_meta['sizeX'], height=level_meta['sizeY']),
                    format=large_image.constants.TILE_FORMAT_NUMPY,
                    level=target_level
                )
                return thumbnail_data
                
            elif hasattr(self, 'openslide_handle') and self.openslide_handle is not None:
                # Use OpenSlide
                # Find appropriate level
                dims = self.openslide_handle.level_dimensions[0]
                scale = min(max_size[0] / dims[0], max_size[1] / dims[1])
                
                target_level = 0
                for level, level_dims in enumerate(self.openslide_handle.level_dimensions):
                    level_scale = level_dims[0] / dims[0]
                    if level_scale <= scale:
                        target_level = level
                        break
                
                # Get thumbnail
                level_dims = self.openslide_handle.level_dimensions[target_level]
                pil_img = self.openslide_handle.read_region((0, 0), target_level, level_dims)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                return np.array(pil_img)
                
        except Exception as e:
            print(f"Error getting WSI thumbnail: {e}")
            return None
    
    def get_wsi_properties(self) -> dict:
        """Get WSI slide properties and metadata."""
        if hasattr(self, 'slide_properties'):
            return self.slide_properties.copy()
        return {}

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.memory_mapped_file:
                self.memory_mapped_file.close()
                self.memory_mapped_file = None
            
            # Clean up WSI resources
            if hasattr(self, 'openslide_handle') and self.openslide_handle:
                self.openslide_handle.close()
                self.openslide_handle = None
                
            if hasattr(self, 'large_image_source') and self.large_image_source:
                # large-image sources should be closed
                try:
                    self.large_image_source = None
                except:
                    pass
                    
            # Clear WSI-specific attributes
            if hasattr(self, 'slide_properties'):
                self.slide_properties = {}
            if hasattr(self, 'zoom_levels'):
                self.zoom_levels = []
            
            if self.dask_array is not None:
                del self.dask_array
                self.dask_array = None
            
            if self.current_image is not None:
                del self.current_image
                self.current_image = None
            
            # Clear tile cache
            self.tile_cache.clear()
            self.cache_access_times.clear()
            
            # Force garbage collection
            gc.collect()
            
            self.is_loaded = False
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Backward compatibility
LargeImageLoader = EnhancedLargeImageLoader

# Export SUPPORTED_FORMATS for external use
SUPPORTED_FORMATS = EnhancedLargeImageLoader.SUPPORTED_FORMATS
__all__ = ['EnhancedLargeImageLoader', 'LargeImageLoader', 'SUPPORTED_FORMATS']
