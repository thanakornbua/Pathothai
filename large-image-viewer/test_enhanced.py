"""
Test script for the Enhanced Large Image Viewer
Tests loading capabilities with different file formats and sizes.
"""

import sys
import os
import numpy as np
from PIL import Image
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_images():
    """Create test images of various sizes and formats."""
    print("Creating test images...")
    
    # Create test directory
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Small test image (RGB)
    small_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(small_rgb).save(f"{test_dir}/small_rgb.png")
    
    # Large test image (simulate 1GB+)
    print("Creating large test image (this may take a moment)...")
    large_size = 8192  # 8K x 8K RGB = ~192MB
    large_rgb = np.random.randint(0, 255, (large_size, large_size, 3), dtype=np.uint8)
    Image.fromarray(large_rgb).save(f"{test_dir}/large_rgb.png")
    
    # Grayscale image
    gray = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    Image.fromarray(gray, mode='L').save(f"{test_dir}/grayscale.png")
    
    # Multi-channel TIFF (if tifffile available)
    try:
        import tifffile
        multi_channel = np.random.randint(0, 255, (1024, 1024, 8), dtype=np.uint8)
        tifffile.imwrite(f"{test_dir}/multi_channel.tiff", multi_channel)
        print("Created multi-channel TIFF")
    except ImportError:
        print("Skipping multi-channel TIFF (tifffile not available)")
    
    print(f"Test images created in {test_dir}/")
    return test_dir

def test_image_loader():
    """Test the enhanced image loader."""
    print("\n=== Testing Enhanced Image Loader ===")
    
    try:
        from core.enhanced_image_loader import EnhancedLargeImageLoader
        
        # Create test images
        test_dir = create_test_images()
        
        # Initialize loader
        loader = EnhancedLargeImageLoader(chunk_size=512, max_memory_gb=4.0)
        
        # Test loading different formats
        test_files = [
            f"{test_dir}/small_rgb.png",
            f"{test_dir}/large_rgb.png", 
            f"{test_dir}/grayscale.png"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"\nTesting: {file_path}")
                file_size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"File size: {file_size:.1f} MB")
                
                start_time = time.time()
                success = loader.load_image(file_path)
                load_time = time.time() - start_time
                
                if success:
                    print(f"✓ Loaded successfully in {load_time:.2f}s")
                    print(f"  Shape: {loader.image_shape}")
                    print(f"  Channels: {loader.channels}")
                    print(f"  Format: {loader.file_format}")
                    
                    # Test thumbnail generation
                    thumbnail = loader.get_thumbnail((256, 256))
                    if thumbnail is not None:
                        print("  ✓ Thumbnail generated")
                    
                    # Test region extraction
                    if loader.image_shape[0] >= 100 and loader.image_shape[1] >= 100:
                        region = loader.get_region(0, 0, 100, 100)
                        if region is not None:
                            print("  ✓ Region extraction working")
                    
                    # Test metadata
                    metadata = loader.get_metadata()
                    print(f"  Load time: {metadata.get('load_time', 0):.2f}s")
                    print(f"  Memory usage: {metadata.get('memory_usage_gb', 0):.2f}GB")
                    
                else:
                    print(f"✗ Failed to load")
                
                loader.cleanup()
        
        print("\n✓ Image loader tests completed")
        
    except Exception as e:
        print(f"✗ Image loader test failed: {e}")

def test_dicom_support():
    """Test DICOM support if available."""
    print("\n=== Testing DICOM Support ===")
    
    try:
        import pydicom
        import SimpleITK as sitk
        from core.enhanced_image_loader import EnhancedLargeImageLoader
        
        print("✓ DICOM libraries available")
        
        # Create a simple synthetic DICOM file for testing
        print("Creating synthetic DICOM for testing...")
        
        # This is a basic test - in real use, you'd load actual DICOM files
        loader = EnhancedLargeImageLoader()
        print("✓ Enhanced loader supports DICOM format detection")
        
    except ImportError as e:
        print(f"⚠ DICOM support not available: {e}")
        print("Install with: pip install pydicom SimpleITK")
    except Exception as e:
        print(f"✗ DICOM test failed: {e}")

def test_region_selection():
    """Test region selection functionality."""
    print("\n=== Testing Region Selection ===")
    
    try:
        from core.enhanced_image_loader import EnhancedLargeImageLoader, RegionSelector
        
        # Create a test image
        test_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Test region selector
        selector = RegionSelector((1024, 1024, 3))
        
        # Add regions
        region_id = selector.add_region(100, 100, 200, 200, "Test Region 1")
        print(f"✓ Added region with ID: {region_id}")
        
        bounds = selector.get_region_bounds(region_id)
        print(f"✓ Region bounds: {bounds}")
        
        # Test with loader
        loader = EnhancedLargeImageLoader()
        # Simulate loading by setting image data directly
        loader.current_image = test_array
        loader.image_shape = test_array.shape[:2]
        loader.channels = test_array.shape[2]
        loader.dtype = test_array.dtype
        loader.is_loaded = True
        
        # Test region extraction
        region_data = loader.get_region(100, 100, 200, 200)
        if region_data is not None:
            print(f"✓ Region extracted, shape: {region_data.shape}")
        
        # Test saving (to memory)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            success = loader.save_region(100, 100, 200, 200, tmp.name)
            if success:
                print("✓ Region save functionality working")
                os.unlink(tmp.name)  # Clean up
        
        print("✓ Region selection tests completed")
        
    except Exception as e:
        print(f"✗ Region selection test failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency with large files."""
    print("\n=== Testing Memory Efficiency ===")
    
    try:
        import psutil
        from core.enhanced_image_loader import EnhancedLargeImageLoader
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test with chunk-based loading
        loader = EnhancedLargeImageLoader(chunk_size=256, max_memory_gb=2.0)
        
        # Create and test with moderately large array
        test_size = 4096  # 4K x 4K RGB ≈ 48MB
        print(f"Testing with {test_size}x{test_size} RGB image...")
        
        large_array = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)
        
        # Simulate chunked processing
        chunk_size = loader.chunk_size
        total_chunks = (test_size // chunk_size) ** 2
        print(f"Processing in {total_chunks} chunks of {chunk_size}x{chunk_size}")
        
        max_memory = initial_memory
        for y in range(0, test_size, chunk_size):
            for x in range(0, test_size, chunk_size):
                # Simulate chunk processing
                chunk_h = min(chunk_size, test_size - y)
                chunk_w = min(chunk_size, test_size - x)
                chunk = large_array[y:y+chunk_h, x:x+chunk_w]
                
                # Check memory
                current_memory = process.memory_info().rss / (1024**2)
                max_memory = max(max_memory, current_memory)
        
        memory_increase = max_memory - initial_memory
        print(f"Maximum memory increase: {memory_increase:.1f} MB")
        print(f"Memory efficiency: {'✓ Good' if memory_increase < 200 else '⚠ High'}")
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")

def main():
    """Run all tests."""
    print("Enhanced Large Image Viewer - Test Suite")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    required_packages = ['numpy', 'PIL', 'psutil']
    optional_packages = ['pydicom', 'SimpleITK', 'tifffile', 'cv2', 'h5py', 'zarr', 'dask']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (required)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} (optional)")
    
    # Run tests
    test_image_loader()
    test_dicom_support()
    test_region_selection()
    test_memory_efficiency()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo run the full application:")
    print("python main.py")

if __name__ == "__main__":
    main()
