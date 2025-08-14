#!/usr/bin/env python3
"""
Test SVS file loading in the actual image viewer to debug gray display issue.
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from core.enhanced_image_loader import EnhancedLargeImageLoader
from gui.image_canvas import ImageCanvas
import numpy as np

def test_svs_in_viewer():
    """Test loading SVS file in the actual viewer."""
    
    # Find an SVS file
    svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    
    if not os.path.exists(svs_file):
        print(f"SVS file not found: {svs_file}")
        return
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("SVS Viewer Test")
    window.setGeometry(100, 100, 1200, 800)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add status label
    status_label = QLabel("Loading SVS file...")
    layout.addWidget(status_label)
    
    # Create image canvas
    image_canvas = ImageCanvas()
    layout.addWidget(image_canvas)
    
    # Load SVS file
    print(f"Loading SVS file: {svs_file}")
    loader = EnhancedLargeImageLoader()
    success = loader.load_image(svs_file)
    
    if success:
        print("✅ SVS loaded successfully")
        print(f"Image shape: {loader.image_shape}")
        print(f"Zoom levels: {len(loader.zoom_levels)}")
        
        # Set the image loader in the canvas
        image_canvas.set_image_loader(loader)
        
        # Test getting a few tiles and analyze their appearance
        print("\n🔍 Analyzing tile data:")
        for level in [0, 1]:
            if level < len(loader.zoom_levels):
                tile_data = loader.get_tile(level, 0, 0)
                if tile_data is not None:
                    print(f"Level {level}, Tile (0,0):")
                    print(f"  Shape: {tile_data.shape}")
                    print(f"  Data type: {tile_data.dtype}")
                    print(f"  Min/Max: {tile_data.min()}/{tile_data.max()}")
                    print(f"  Mean: {tile_data.mean():.1f}")
                    print(f"  Standard deviation: {tile_data.std():.1f}")
                    
                    # Check if data looks like it should be normalized
                    if tile_data.max() > 200:
                        print(f"  ⚠️  High values detected - tiles may appear washed out")
                    
                    # Check individual channels
                    if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
                        for i, channel in enumerate(['R', 'G', 'B']):
                            ch_data = tile_data[:, :, i]
                            print(f"  {channel} channel: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.1f}")
        
        status_label.setText(f"SVS loaded: {loader.image_shape} - Use mouse to pan/zoom")
        
    else:
        print("❌ Failed to load SVS file")
        status_label.setText("Failed to load SVS file")
    
    # Show window
    window.show()
    
    # Start event loop
    app.exec_()

if __name__ == "__main__":
    test_svs_in_viewer()
