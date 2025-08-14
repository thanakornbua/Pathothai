#!/usr/bin/env python3
"""
Debug script to test image loading functionality step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from core.enhanced_image_loader import EnhancedLargeImageLoader

def test_image_loader():
    """Test the image loader with debug information."""
    print("=" * 60)
    print("DEBUG: Testing Enhanced Image Loader")
    print("=" * 60)
    
    # Create loader
    loader = EnhancedLargeImageLoader()
    print(f"✅ Loader created successfully")
    
    # Test files (update these paths to actual files you have)
    test_files = [
        "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs",
        "C:/Users/tanth/Desktop/Pathothai/Her2Neg_Case_05.png",
        "C:/Users/tanth/Desktop/Pathothai/output_dsfalse/Her2Pos_Case_45.png"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"\n🔍 Testing file: {os.path.basename(file_path)}")
        print(f"   Path: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"   Size: {file_size / (1024**3):.3f} GB")
        
        # Detect format
        detected_format = loader.detect_format(file_path)
        print(f"   Format: {detected_format}")
        
        # Try to load
        try:
            print("   📥 Attempting to load...")
            success = loader.load_image(file_path)
            
            if success:
                print(f"   ✅ Load successful!")
                print(f"   📐 Image shape: {loader.image_shape}")
                print(f"   🎨 Channels: {loader.channels}")
                print(f"   📊 Data type: {loader.dtype}")
                
                # Try to get thumbnail
                try:
                    thumbnail = loader.get_thumbnail((256, 256))
                    if thumbnail is not None:
                        print(f"   🖼️  Thumbnail: {thumbnail.shape}")
                        print(f"   📈 Thumbnail range: {thumbnail.min()} - {thumbnail.max()}")
                    else:
                        print(f"   ❌ Thumbnail generation failed")
                except Exception as e:
                    print(f"   ❌ Thumbnail error: {e}")
                
            else:
                print(f"   ❌ Load failed")
                
        except Exception as e:
            print(f"   💥 Exception during loading: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        loader.cleanup()
        print(f"   🧹 Cleanup completed")

def test_simple_image():
    """Test with a simple generated image."""
    print(f"\n🎨 Testing with generated image...")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"   Generated image: {test_image.shape}")
    
    # Test the canvas with this image
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.image_canvas import ImageCanvas
        
        app = QApplication([])
        
        canvas = ImageCanvas()
        print(f"   ✅ Canvas created")
        
        canvas.set_image(test_image)
        print(f"   ✅ Image set on canvas")
        
        app.quit()
        print(f"   ✅ Test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Canvas test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_loader()
    test_simple_image()
    print("\n" + "=" * 60)
    print("DEBUG TEST COMPLETED")
    print("=" * 60)
