#!/usr/bin/env python3
"""
Test script for region export with 6-digit position support
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from src.core.enhanced_image_loader import EnhancedLargeImageLoader
from src.gui.region_selector import RegionSelectionWidget

def test_region_selector():
    """Test the region selector with 6-digit position support."""
    app = QApplication(sys.argv)
    
    # Create image loader
    loader = EnhancedLargeImageLoader()
    
    # Create region selector widget
    widget = RegionSelectionWidget(loader)
    
    # Test adding regions with large coordinates
    test_regions = [
        (12345, 67890, 512, 512, "Test Region 1"),
        (123456, 654321, 256, 256, "Test Region 2"),
        (40000, 15000, 1024, 1024, "Large Coordinates")
    ]
    
    print("Testing region selector with large coordinates:")
    for x, y, w, h, name in test_regions:
        widget.add_region(x, y, w, h, name)
        print(f"Added region: {name} at ({x}, {y}) size {w}x{h}")
    
    # Test different naming formats
    print("\nTesting naming formats:")
    for format_name in ['Region Name', '6-Digit Position', 'Simple Coordinates', 'Detailed Position']:
        widget.naming_combo.setCurrentText(format_name)
        widget.update_naming_preview()
        preview_text = widget.naming_preview.text()
        print(f"{format_name}: {preview_text}")
    
    print("\nRegion selector test completed successfully!")
    
    # Show the widget
    widget.show()
    
    return app.exec_()

if __name__ == "__main__":
    test_region_selector()
