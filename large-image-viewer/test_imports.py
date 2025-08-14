"""
Simple test to check if our imports work correctly.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports to identify any issues."""
    print("Testing imports...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5.QtWidgets")
    except ImportError as e:
        print(f"✗ PyQt5.QtWidgets: {e}")
        return False
    
    try:
        from gui.main_window import MainWindow
        print("✓ gui.main_window.MainWindow")
    except ImportError as e:
        print(f"✗ gui.main_window.MainWindow: {e}")
        return False
    
    try:
        from gui.image_canvas import ImageCanvas
        print("✓ gui.image_canvas.ImageCanvas")
    except ImportError as e:
        print(f"✗ gui.image_canvas.ImageCanvas: {e}")
        return False
    
    try:
        from gui.channel_controls import ChannelControls
        print("✓ gui.channel_controls.ChannelControls")
    except ImportError as e:
        print(f"✗ gui.channel_controls.ChannelControls: {e}")
        return False
    
    try:
        from gui.region_selector import RegionSelectionWidget
        print("✓ gui.region_selector.RegionSelectionWidget")
    except ImportError as e:
        print(f"✗ gui.region_selector.RegionSelectionWidget: {e}")
        return False
    
    try:
        from core.enhanced_image_loader import EnhancedLargeImageLoader
        print("✓ core.enhanced_image_loader.EnhancedLargeImageLoader")
    except ImportError as e:
        print(f"✗ core.enhanced_image_loader.EnhancedLargeImageLoader: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Large Image Viewer - Import Test")
    print("=" * 40)
    
    if test_imports():
        print("\n✓ All imports successful!")
        print("You can now run: python main.py")
    else:
        print("\n✗ Some imports failed!")
        print("Please install missing dependencies.")
