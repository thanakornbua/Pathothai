#!/usr/bin/env python3
"""
Test the new simple SVS loader directly to verify it works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from core.simple_svs_loader import SimpleSVSLoader

class SimpleSVSViewer(QMainWindow):
    """Simple viewer to test the SVS loader."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple SVS Tile Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout(central_widget)
        
        # Label to display tile
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: white;")
        self.image_label.setMinimumSize(600, 600)
        layout.addWidget(self.image_label)
        
        # Load and test SVS
        self.test_svs_loading()
    
    def test_svs_loading(self):
        """Test loading SVS tiles."""
        print("🔍 Testing Simple SVS Loader...")
        
        # Initialize loader
        loader = SimpleSVSLoader()
        
        # Load SVS file
        svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
        
        if not loader.load_svs_file(svs_file):
            print("❌ Failed to load SVS file")
            return
        
        print("✅ SVS file loaded successfully")
        
        # Create a composite image from multiple tiles to show more detail
        print("🎯 Loading 2x2 grid of tiles from level 0...")
        
        # Create composite pixmap
        composite_pixmap = QPixmap(512, 512)  # 2x2 tiles
        composite_pixmap.fill(Qt.white)
        
        from PyQt5.QtGui import QPainter
        painter = QPainter(composite_pixmap)
        
        tiles_loaded = 0
        for tile_y in range(2):
            for tile_x in range(2):
                print(f"🔍 Loading tile ({tile_x},{tile_y})...")
                pixmap = loader.get_tile_with_enhancement(0, tile_x, tile_y)
                
                if pixmap and not pixmap.isNull():
                    # Draw tile at correct position
                    painter.drawPixmap(tile_x * 256, tile_y * 256, pixmap)
                    tiles_loaded += 1
                    
                    # Check color of this tile
                    test_img = pixmap.toImage()
                    if not test_img.isNull():
                        center_color = test_img.pixelColor(test_img.width()//2, test_img.height()//2)
                        print(f"   🎨 Tile ({tile_x},{tile_y}) center: R={center_color.red()}, G={center_color.green()}, B={center_color.blue()}")
        
        painter.end()
        
        if tiles_loaded > 0:
            print(f"✅ Loaded {tiles_loaded}/4 tiles successfully")
            
            # Scale composite for display (maintain aspect ratio)
            display_size = 600
            scaled_composite = composite_pixmap.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_composite)
            
            print(f"🖼️  Displaying {scaled_composite.width()}x{scaled_composite.height()} composite image")
        else:
            print("❌ No tiles loaded successfully")
        
        # Clean up
        loader.cleanup()

def main():
    app = QApplication(sys.argv)
    
    viewer = SimpleSVSViewer()
    viewer.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
