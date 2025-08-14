#!/usr/bin/env python3
"""
Test SVS automatic tissue detection and navigation.
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from core.enhanced_image_loader import EnhancedLargeImageLoader
from gui.image_canvas import ImageCanvas

def test_svs_tissue_navigation():
    """Test SVS file with automatic tissue navigation."""
    
    # Find an SVS file
    svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    
    if not os.path.exists(svs_file):
        print(f"SVS file not found: {svs_file}")
        return
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("SVS Tissue Navigation Test")
    window.setGeometry(100, 100, 1200, 800)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add control buttons
    controls_layout = QHBoxLayout()
    
    # Status label
    status_label = QLabel("Loading SVS file...")
    controls_layout.addWidget(status_label)
    
    # Navigation buttons
    zoom_in_btn = QPushButton("Zoom In")
    zoom_out_btn = QPushButton("Zoom Out")
    fit_btn = QPushButton("Fit to Window")
    tissue_btn = QPushButton("Find Tissue")
    
    controls_layout.addWidget(zoom_in_btn)
    controls_layout.addWidget(zoom_out_btn)
    controls_layout.addWidget(fit_btn)
    controls_layout.addWidget(tissue_btn)
    controls_layout.addStretch()
    
    layout.addLayout(controls_layout)
    
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
        
        # Set the image loader in the canvas (this will trigger tissue detection)
        image_canvas.set_image_loader(loader)
        
        status_label.setText(f"SVS loaded: {loader.image_shape} - Use mouse to pan/zoom, or click 'Find Tissue'")
        
        # Connect buttons
        zoom_in_btn.clicked.connect(image_canvas.zoom_in)
        zoom_out_btn.clicked.connect(image_canvas.zoom_out)
        fit_btn.clicked.connect(image_canvas.fit_to_window)
        
        def find_and_navigate_to_tissue():
            """Find tissue and navigate to it."""
            tissue_center = image_canvas.find_tissue_center()
            if tissue_center:
                print(f"🔬 Navigating to tissue at: {tissue_center}")
                image_canvas.pan_to_tissue_area(tissue_center)
                image_canvas.update_display()
                status_label.setText(f"Navigated to tissue at {tissue_center}")
            else:
                print("❌ No tissue found")
                status_label.setText("No tissue areas detected")
        
        tissue_btn.clicked.connect(find_and_navigate_to_tissue)
        
    else:
        print("❌ Failed to load SVS file")
        status_label.setText("Failed to load SVS file")
    
    # Show window
    window.show()
    
    print("\n🎮 Controls:")
    print("   - Mouse: Pan and zoom (Ctrl+wheel)")
    print("   - Buttons: Zoom In/Out, Fit to Window, Find Tissue")
    print("   - The viewer should automatically navigate to tissue on startup")
    
    # Start event loop
    app.exec_()

if __name__ == "__main__":
    test_svs_tissue_navigation()
