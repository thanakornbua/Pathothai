#!/usr/bin/env python3
"""
Direct SVS viewer using only the working SimpleSVSLoader.
No complex inheritance, just straightforward SVS display.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QPixmap, QWheelEvent, QMouseEvent, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, QThread, pyqtSignal
from core.simple_svs_loader import SimpleSVSLoader

class SimpleTileLoader(QThread):
    """Simple tile loader using only SimpleSVSLoader."""
    
    tile_ready = pyqtSignal(int, int, int, QPixmap)  # level, x, y, pixmap
    
    def __init__(self, svs_loader):
        super().__init__()
        self.svs_loader = svs_loader
        self.tile_requests = []
        self.running = False
    
    def request_tile(self, level, tile_x, tile_y):
        """Request a tile to be loaded."""
        request = (level, tile_x, tile_y)
        if request not in self.tile_requests:
            self.tile_requests.append(request)
            if not self.isRunning():
                self.start()
    
    def run(self):
        """Load requested tiles."""
        self.running = True
        while self.running and self.tile_requests:
            level, tile_x, tile_y = self.tile_requests.pop(0)
            
            pixmap = self.svs_loader.get_tile_with_enhancement(level, tile_x, tile_y)
            if pixmap and not pixmap.isNull():
                self.tile_ready.emit(level, tile_x, tile_y, pixmap)
        
        self.running = False

class DirectSVSViewer(QWidget):
    """Direct SVS viewer with simple tile display."""
    
    def __init__(self):
        super().__init__()
        self.svs_loader = None
        self.tile_loader = None
        self.tiles_cache = {}
        
        # Display properties
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_point = QPoint()
        self.tile_size = 256
        self.current_level = 0
        
        # SVS properties
        self.image_width = 0
        self.image_height = 0
        self.level_count = 0
        
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
    def load_svs(self, file_path):
        """Load SVS file directly."""
        print(f"🔍 Loading SVS: {file_path}")
        
        self.svs_loader = SimpleSVSLoader()
        if not self.svs_loader.load_svs_file(file_path):
            print("❌ Failed to load SVS")
            return False
        
        # Get SVS properties
        self.level_count = self.svs_loader.get_level_count()
        if self.level_count > 0:
            self.image_width, self.image_height = self.svs_loader.get_level_dimensions(0)
            print(f"✅ SVS loaded: {self.image_width}x{self.image_height}, {self.level_count} levels")
            
            # Initialize tile loader
            self.tile_loader = SimpleTileLoader(self.svs_loader)
            self.tile_loader.tile_ready.connect(self.on_tile_ready)
            
            # Calculate initial zoom to fit
            self.fit_to_window()
            self.update()
            return True
        
        return False
    
    def fit_to_window(self):
        """Fit image to window."""
        if self.image_width > 0 and self.image_height > 0:
            widget_w = self.width()
            widget_h = self.height()
            
            zoom_x = widget_w / self.image_width
            zoom_y = widget_h / self.image_height
            self.zoom_factor = min(zoom_x, zoom_y) * 0.9  # 90% of fit
            
            # Center the image
            self.pan_offset = QPoint(
                (widget_w - int(self.image_width * self.zoom_factor)) // 2,
                (widget_h - int(self.image_height * self.zoom_factor)) // 2
            )
            
            print(f"🔍 Fit to window: zoom={self.zoom_factor:.3f}, offset=({self.pan_offset.x()}, {self.pan_offset.y()})")
    
    def on_tile_ready(self, level, tile_x, tile_y, pixmap):
        """Handle tile loaded."""
        key = (level, tile_x, tile_y)
        self.tiles_cache[key] = pixmap
        print(f"✅ Tile cached: ({level}, {tile_x}, {tile_y}) - {pixmap.width()}x{pixmap.height()}")
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the SVS image using tiles."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        
        if not self.svs_loader:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No SVS file loaded")
            return
        
        # Calculate which tiles to display
        widget_w = self.width()
        widget_h = self.height()
        
        # Convert widget coordinates to image coordinates
        if self.zoom_factor <= 0:
            return
        
        # Calculate visible image area
        left = max(0, int(-self.pan_offset.x() / self.zoom_factor))
        top = max(0, int(-self.pan_offset.y() / self.zoom_factor))
        right = min(self.image_width, int((widget_w - self.pan_offset.x()) / self.zoom_factor))
        bottom = min(self.image_height, int((widget_h - self.pan_offset.y()) / self.zoom_factor))
        
        # Calculate tile range
        tile_left = left // self.tile_size
        tile_top = top // self.tile_size
        tile_right = (right + self.tile_size - 1) // self.tile_size
        tile_bottom = (bottom + self.tile_size - 1) // self.tile_size
        
        print(f"🎨 Paint: visible area ({left},{top}) to ({right},{bottom})")
        print(f"🧩 Tiles needed: ({tile_left},{tile_top}) to ({tile_right},{tile_bottom})")
        
        tiles_drawn = 0
        tiles_requested = 0
        
        # Draw tiles
        for tile_y in range(tile_top, tile_bottom):
            for tile_x in range(tile_left, tile_right):
                # Calculate tile position in widget coordinates
                tile_image_x = tile_x * self.tile_size
                tile_image_y = tile_y * self.tile_size
                
                widget_x = tile_image_x * self.zoom_factor + self.pan_offset.x()
                widget_y = tile_image_y * self.zoom_factor + self.pan_offset.y()
                
                tile_key = (self.current_level, tile_x, tile_y)
                
                if tile_key in self.tiles_cache:
                    # Draw cached tile
                    pixmap = self.tiles_cache[tile_key]
                    scaled_size = int(self.tile_size * self.zoom_factor)
                    scaled_pixmap = pixmap.scaled(scaled_size, scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    painter.drawPixmap(int(widget_x), int(widget_y), scaled_pixmap)
                    tiles_drawn += 1
                    
                    print(f"🖼️  Drew tile ({tile_x},{tile_y}) at ({int(widget_x)},{int(widget_y)}) size {scaled_size}")
                else:
                    # Request tile
                    self.tile_loader.request_tile(self.current_level, tile_x, tile_y)
                    tiles_requested += 1
                    
                    # Draw placeholder
                    scaled_size = int(self.tile_size * self.zoom_factor)
                    painter.fillRect(int(widget_x), int(widget_y), scaled_size, scaled_size, Qt.gray)
                    painter.setPen(QPen(Qt.white, 1))
                    painter.drawRect(int(widget_x), int(widget_y), scaled_size, scaled_size)
        
        print(f"📊 Paint summary: {tiles_drawn} drawn, {tiles_requested} requested")
        
        # Draw info
        painter.setPen(Qt.white)
        info_text = f"Zoom: {self.zoom_factor:.3f} | Level: {self.current_level} | Tiles: {len(self.tiles_cache)}"
        painter.drawText(10, 20, info_text)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.2
        else:
            self.zoom_factor /= 1.2
        
        self.zoom_factor = max(0.01, min(10.0, self.zoom_factor))
        print(f"🔍 Zoom: {self.zoom_factor:.3f}")
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Start panning."""
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_pan_point = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle panning."""
        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning."""
        if event.button() == Qt.LeftButton:
            self.is_panning = False

class DirectSVSMainWindow(QMainWindow):
    """Main window for direct SVS viewing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Direct SVS Viewer - Enhanced")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create viewer widget
        self.viewer = DirectSVSViewer()
        self.setCentralWidget(self.viewer)
        
        # Load SVS file
        svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
        if self.viewer.load_svs(svs_file):
            print("✅ SVS viewer ready!")
        else:
            print("❌ Failed to initialize SVS viewer")

def main():
    app = QApplication(sys.argv)
    
    window = DirectSVSMainWindow()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
