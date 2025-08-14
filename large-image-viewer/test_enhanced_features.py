#!/usr/bin/env python3
"""
Test script for enhanced image viewer features.
Tests the improved tiling, zooming, and region selection functionality.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.image_canvas import ImageCanvas, ImageDisplayLabel
from core.enhanced_image_loader import EnhancedLargeImageLoader

class TestWindow(QMainWindow):
    """Test window for enhanced features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Image Viewer Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        self.load_test_btn = QPushButton("Load Test Image")
        self.load_test_btn.clicked.connect(self.load_test_image)
        controls_layout.addWidget(self.load_test_btn)
        
        self.region_btn = QPushButton("Toggle Region Selection")
        self.region_btn.clicked.connect(self.toggle_region_selection)
        controls_layout.addWidget(self.region_btn)
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.zoom_out_btn)
        
        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.clicked.connect(self.fit_to_window)
        controls_layout.addWidget(self.fit_btn)
        
        controls_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        layout.addLayout(controls_layout)
        
        # Create image canvas
        self.image_canvas = ImageCanvas()
        self.image_canvas.zoom_changed.connect(self.on_zoom_changed)
        self.image_canvas.coordinates_changed.connect(self.on_coordinates_changed)
        self.image_canvas.connect_region_selected(self.on_region_selected)
        
        layout.addWidget(self.image_canvas)
        
        # Create image loader
        self.image_loader = EnhancedLargeImageLoader(chunk_size=256, max_memory_gb=4.0)
        
        # State
        self.region_selection_enabled = False
    
    def load_test_image(self):
        """Load a test image."""
        try:
            # Create a test image (large synthetic image)
            print("Creating test image...")
            width, height = 2048, 2048
            
            # Create a colorful test pattern
            test_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a gradient pattern
            for y in range(height):
                for x in range(width):
                    # Create repeating pattern
                    pattern_x = x % 256
                    pattern_y = y % 256
                    
                    test_image[y, x, 0] = pattern_x  # Red channel
                    test_image[y, x, 1] = pattern_y  # Green channel
                    test_image[y, x, 2] = (pattern_x + pattern_y) // 2  # Blue channel
            
            # Add some shapes
            center_x, center_y = width // 2, height // 2
            for i in range(0, min(width, height) // 4, 50):
                y_start = max(0, center_y - i)
                y_end = min(height, center_y + i)
                x_start = max(0, center_x - i)
                x_end = min(width, center_x + i)
                
                color = [(255 - i) % 255, i % 255, (i * 2) % 255]
                test_image[y_start:y_start+2, x_start:x_end] = color  # Top edge
                test_image[y_end-2:y_end, x_start:x_end] = color      # Bottom edge
                test_image[y_start:y_end, x_start:x_start+2] = color  # Left edge
                test_image[y_start:y_end, x_end-2:x_end] = color      # Right edge
            
            print(f"Test image created: {width}x{height}")
            
            # Set up the image loader with test data
            self.image_loader.current_image = test_image
            self.image_loader.image_shape = (height, width)
            self.image_loader.channels = 3
            self.image_loader.dtype = np.uint8
            self.image_loader.is_loaded = True
            
            # Create zoom levels for testing
            self.image_loader.zoom_levels = []
            for level in range(4):  # 4 zoom levels
                downsample = 2 ** level
                level_width = width // downsample
                level_height = height // downsample
                
                self.image_loader.zoom_levels.append({
                    'level': level,
                    'width': level_width,
                    'height': level_height,
                    'downsample': downsample
                })
            
            # Connect image loader to canvas
            self.image_canvas.set_image_loader(self.image_loader)
            
            # Set initial image display (use a downsampled version for performance)
            display_image = test_image[::4, ::4, :]  # Downsample by 4x for initial display
            self.image_canvas.set_image(display_image)
            
            self.status_label.setText(f"Test image loaded: {width}x{height}")
            print("Test image loaded successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error loading test image: {e}")
    
    def toggle_region_selection(self):
        """Toggle region selection mode."""
        self.region_selection_enabled = not self.region_selection_enabled
        self.image_canvas.enable_region_selection(self.region_selection_enabled)
        
        if self.region_selection_enabled:
            self.region_btn.setText("Disable Region Selection")
            self.status_label.setText("Region selection enabled - Click and drag to select")
        else:
            self.region_btn.setText("Enable Region Selection")
            self.status_label.setText("Region selection disabled")
    
    def zoom_in(self):
        """Zoom in."""
        self.image_canvas.zoom_in()
    
    def zoom_out(self):
        """Zoom out."""
        self.image_canvas.zoom_out()
    
    def fit_to_window(self):
        """Fit image to window."""
        self.image_canvas.fit_to_window()
    
    def on_zoom_changed(self, zoom_level):
        """Handle zoom change."""
        self.status_label.setText(f"Zoom: {zoom_level:.1f}%")
    
    def on_coordinates_changed(self, pos):
        """Handle coordinate change."""
        current_text = self.status_label.text()
        if "Position:" not in current_text:
            self.status_label.setText(f"{current_text} | Position: ({pos.x()}, {pos.y()})")
    
    def on_region_selected(self, rect):
        """Handle region selection."""
        self.status_label.setText(
            f"Region selected: ({rect.x()}, {rect.y()}) {rect.width()}x{rect.height()}"
        )
        print(f"Region selected: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}")

def main():
    """Main test function."""
    app = QApplication(sys.argv)
    
    print("Enhanced Image Viewer Feature Test")
    print("===================================")
    print()
    print("Features to test:")
    print("- Load Test Image: Creates a synthetic 2048x2048 test image")
    print("- Zoom In/Out: Test accurate zooming with viewport management")
    print("- Region Selection: Click and drag to select regions like screenshots")
    print("- Mouse wheel + Ctrl: Zoom around mouse cursor")
    print("- Mouse drag: Pan the image")
    print("- Double-click: Zoom in at cursor position")
    print("- Ctrl+Double-click: Fit to window")
    print("- Keyboard shortcuts:")
    print("  - +/-: Zoom in/out")
    print("  - 0: Reset to 100%")
    print("  - F: Fit to window")
    print("  - Arrow keys: Pan")
    print("  - Escape: Cancel selection or fit to window")
    print()
    
    window = TestWindow()
    window.show()
    
    # Auto-load test image after a short delay
    QTimer.singleShot(1000, window.load_test_image)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
