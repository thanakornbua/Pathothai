#!/usr/bin/env python3
"""
Check dependencies for SVS and large file support.
"""

def check_dependencies():
    print("Checking dependencies for large file and SVS support...")
    print("=" * 50)
    
    # Check OpenSlide
    try:
        import openslide
        print("✅ OpenSlide available - SVS support enabled")
        openslide_version = openslide.__version__ if hasattr(openslide, '__version__') else "unknown"
        print(f"   Version: {openslide_version}")
    except ImportError:
        print("❌ OpenSlide not available - SVS files will not work")
        print("   Install with: pip install openslide-python")
    
    # Check large-image
    try:
        import large_image
        print("✅ large-image available - Enhanced WSI support")
        print(f"   Version: {large_image.__version__}")
    except ImportError:
        print("❌ large-image not available - Limited WSI support")
        print("   Install with: pip install large-image[all]")
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        print("✅ Pillow available - Standard image support")
        print(f"   Version: {Image.__version__}")
    except ImportError:
        print("❌ Pillow not available - Standard images will not work")
        print("   Install with: pip install Pillow")
    
    # Check tifffile
    try:
        import tifffile
        print("✅ tifffile available - Enhanced TIFF support")
        print(f"   Version: {tifffile.__version__}")
    except ImportError:
        print("❌ tifffile not available - Limited TIFF support")
        print("   Install with: pip install tifffile")
    
    # Check numpy
    try:
        import numpy as np
        print("✅ NumPy available")
        print(f"   Version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available - Critical dependency missing")
    
    print("=" * 50)
    print("Dependency check complete.")

if __name__ == "__main__":
    check_dependencies()
