#!/usr/bin/env python3
"""
Test script for the enhanced large image viewer with improved tiling and region selection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtCore import Qt
from gui.image_canvas import ImageCanvas
from core.enhanced_image_loader import EnhancedLargeImageLoader
import numpy as np

class TestViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Image Viewer Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        
        # Create controls
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        
        # Test image button
        self.load_test_btn = QPushButton("Load Test Image")
        self.load_test_btn.clicked.connect(self.load_test_image)
        controls_layout.addWidget(self.load_test_btn)
        
        # Region selection toggle
        self.region_btn = QPushButton("Toggle Region Selection")
        self.region_btn.clicked.connect(self.toggle_region_selection)
        controls_layout.addWidget(self.region_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls)
        
        # Create image canvas
        self.image_canvas = ImageCanvas()
        self.image_canvas.zoom_changed.connect(self.on_zoom_changed)
        self.image_canvas.coordinates_changed.connect(self.on_coordinates_changed)
        self.image_canvas.connect_region_selected(self.on_region_selected)
        
        layout.addWidget(self.image_canvas)
        
        # Image loader
        self.image_loader = EnhancedLargeImageLoader()
        
        self.region_selection_enabled = False
    
    def load_test_image(self):
        """Load a test image."""
        # Create a test image
        test_image = self.create_test_image()
        self.image_canvas.set_image(test_image)
        self.status_label.setText("Test image loaded")
    
    def create_test_image(self):
        """Create a test image with patterns."""
        width, height = 2048, 1536
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a checkerboard pattern
        tile_size = 64
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                color = 255 if ((x // tile_size) + (y // tile_size)) % 2 == 0 else 128
                
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                # Different colors for different regions
                if x < width // 3:
                    image[y:y_end, x:x_end, 0] = color  # Red channel
                elif x < 2 * width // 3:
                    image[y:y_end, x:x_end, 1] = color  # Green channel
                else:
                    image[y:y_end, x:x_end, 2] = color  # Blue channel
        
        # Add some text-like patterns
        for i in range(10):
            y = height // 2 + i * 20
            x_start = width // 4
            x_end = 3 * width // 4
            if y < height:
                image[y:y+10, x_start:x_end, :] = 255
        
        return image
    
    def toggle_region_selection(self):
        """Toggle region selection mode."""
        self.region_selection_enabled = not self.region_selection_enabled
        self.image_canvas.enable_region_selection(self.region_selection_enabled)
        
        if self.region_selection_enabled:
            self.status_label.setText("Region selection enabled - Click and drag to select")
            self.region_btn.setText("Disable Region Selection")
        else:
            self.status_label.setText("Region selection disabled")
            self.region_btn.setText("Enable Region Selection")
    
    def on_zoom_changed(self, zoom_level):
        """Handle zoom changes."""
        self.status_label.setText(f"Zoom: {zoom_level:.1f}%")
    
    def on_coordinates_changed(self, pos):
        """Handle coordinate changes."""
        pass  # Could update a separate coordinate display
    
    def on_region_selected(self, rect):
        """Handle region selection."""
        self.status_label.setText(
            f"Selected region: {rect.width()}x{rect.height()} at ({rect.x()}, {rect.y()})"
        )

def main():
    app = QApplication(sys.argv)
    
    viewer = TestViewer()
    viewer.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
