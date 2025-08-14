#!/usr/bin/env python3
"""
Debug SVS tile loading with detailed diagnostics.
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

def debug_svs_tiles():
    """Debug SVS tile loading with detailed output."""
    
    # Find an SVS file
    svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    
    if not os.path.exists(svs_file):
        print(f"SVS file not found: {svs_file}")
        return
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("SVS Tile Debug")
    window.setGeometry(100, 100, 800, 600)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    status_label = QLabel("Debugging SVS tile loading...")
    layout.addWidget(status_label)
    
    # Create image canvas
    image_canvas = ImageCanvas()
    layout.addWidget(image_canvas)
    
    # Load SVS file
    print(f"🔍 Loading SVS file: {svs_file}")
    loader = EnhancedLargeImageLoader()
    success = loader.load_image(svs_file)
    
    if success:
        print("✅ SVS loaded successfully")
        print(f"Image shape: {loader.image_shape}")
        print(f"Zoom levels: {len(loader.zoom_levels)}")
        
        # Set the image loader in the canvas
        print("\n🎨 Setting up image canvas...")
        image_canvas.set_image_loader(loader)
        
        status_label.setText(f"SVS loaded: {loader.image_shape} - Watch console for tile loading debug info")
        
        print("\n📍 The TileLoader will now start loading tiles.")
        print("Watch the console output for detailed tile information.")
        print("Look for:")
        print("- Tile data ranges (min/max/mean)")
        print("- Contrast enhancement messages")
        print("- Pixmap creation success/failure")
        
    else:
        print("❌ Failed to load SVS file")
        status_label.setText("Failed to load SVS file")
    
    # Show window
    window.show()
    
    # Start event loop
    app.exec_()

if __name__ == "__main__":
    debug_svs_tiles()
