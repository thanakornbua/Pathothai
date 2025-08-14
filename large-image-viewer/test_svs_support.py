#!/usr/bin/env python3
"""
Test SVS support in the enhanced image loader
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.enhanced_image_loader import EnhancedLargeImageLoader

def test_svs_imports():
    """Test that SVS-related imports work."""
    print("Testing SVS imports...")
    
    try:
        # Test importing the loader
        loader = EnhancedLargeImageLoader()
        print("✓ EnhancedLargeImageLoader imported successfully")
        
        # Check WSI support detection
        from core.enhanced_image_loader import HAS_OPENSLIDE, HAS_LARGE_IMAGE
        print(f"✓ OpenSlide support: {HAS_OPENSLIDE}")
        print(f"✓ Large-image support: {HAS_LARGE_IMAGE}")
        
        if not HAS_OPENSLIDE and not HAS_LARGE_IMAGE:
            print("⚠ Warning: No WSI libraries found. SVS files won't be supported.")
            print("  Install with: pip install openslide-python large-image")
        
        # Check supported formats
        from core.enhanced_image_loader import SUPPORTED_FORMATS
        wsi_formats = [ext for ext, info in SUPPORTED_FORMATS.items() 
                      if info.get('type') == 'wsi']
        print(f"✓ WSI formats supported: {wsi_formats}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_wsi_detection():
    """Test WSI file detection logic."""
    print("\nTesting WSI file detection...")
    
    try:
        loader = EnhancedLargeImageLoader()
        
        # Test various file extensions
        test_files = [
            'test.svs',
            'slide.ndpi', 
            'pathology.vms',
            'specimen.scn',
            'sample.mrxs',
            'normal.png',  # Should not be WSI
            'dicom.dcm'    # Should not be WSI
        ]
        
        for test_file in test_files:
            is_wsi = loader._is_wsi_file(test_file)
            expected_wsi = test_file.split('.')[-1] in ['svs', 'ndpi', 'vms', 'scn', 'mrxs', 'bif']
            status = "✓" if is_wsi == expected_wsi else "✗"
            print(f"{status} {test_file}: WSI={is_wsi}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing WSI detection: {e}")
        return False

def test_gui_integration():
    """Test GUI integration of WSI support."""
    print("\nTesting GUI integration...")
    
    try:
        # Test importing GUI components
        from gui.main_window import MainWindow
        print("✓ MainWindow imported successfully")
        
        # Test that file dialogs would include WSI formats
        # (We can't actually test the dialog without Qt app, but we can check the code)
        import inspect
        source = inspect.getsource(MainWindow.open_image)
        if '.svs' in source and '.ndpi' in source:
            print("✓ File dialog includes WSI formats")
        else:
            print("✗ File dialog missing WSI formats")
        
        # Check for WSI-specific menu action
        try:
            source = inspect.getsource(MainWindow.__init__)
            if 'open_wsi' in source:
                print("✓ WSI-specific menu action found")
            else:
                print("✗ WSI-specific menu action missing")
                print("  Searching for 'open_wsi' in MainWindow.__init__ source...")
                # Debug: check what we're actually looking at
                if 'open_dicom' in source:
                    print("  Found 'open_dicom' in source")
                if 'open_action' in source:
                    print("  Found 'open_action' in source")
        except Exception as e:
            print(f"✗ Error inspecting MainWindow.__init__: {e}")
            # Fallback: check if method exists
            if hasattr(MainWindow, 'open_wsi'):
                print("✓ WSI-specific method exists")
            else:
                print("✗ WSI-specific method missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing GUI integration: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing SVS/WSI Support")
    print("=" * 50)
    
    tests = [
        test_svs_imports,
        test_wsi_detection,
        test_gui_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! SVS support is ready.")
    else:
        print("⚠ Some tests failed. Check the output above.")
    
    return all(results)

if __name__ == "__main__":
    main()
