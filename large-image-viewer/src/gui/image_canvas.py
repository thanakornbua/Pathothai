"""
Simple, traditional image canvas widget.
No tiling, no threading, just straightforward image display.
"""

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy, QRubberBand
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter


class ImageDisplayLabel(QLabel):
    """Simple, traditional image display with basic zoom and pan."""
    
    # Signals
    mouse_moved = pyqtSignal(QPoint)
    zoom_changed = pyqtSignal(float)
    region_selected = pyqtSignal(QRect)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Simple image display
        self.image_loader = None
        self.display_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.01
        self.max_zoom = 100.0
        
        # Pan variables
        self.last_pan_point = QPoint()
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        
        # Region selection
        self.is_selecting = False
        self.selection_start = QPoint()
        self.selection_current = QPoint()
        self.rubber_band = None
        self.selection_enabled = False
        
        print("✅ Traditional image display initialized")
    
    def set_image_loader(self, image_loader):
        """Set the image loader - load high quality but manageable size."""
        print(f"🔗 Setting image loader: {type(image_loader).__name__}")
        self.image_loader = image_loader
        
        # Load high quality but manageable size
        if image_loader:
            try:
                print("📸 Loading high quality image...")
                
                # Get full size dimensions
                height, width = image_loader.image_shape
                print(f"📏 Full image dimensions: {width}x{height}")
                
                # Choose level 1 for good quality but manageable size
                # Level 0 = full size (too big), Level 1 = 1/4 size (good balance)
                level = 1
                level_width = image_loader.zoom_levels[level]['width']
                level_height = image_loader.zoom_levels[level]['height']
                print(f"📐 Loading level {level} image: {level_width}x{level_height}")
                
                # Load level 1 (1/4 resolution but still high quality)
                image_data = image_loader.get_wsi_region(0, 0, level_width, level_height, level)
                
                if image_data is not None:
                    # Convert to pixmap
                    pixmap = self._convert_to_pixmap(image_data)
                    if pixmap:
                        self.display_pixmap = pixmap
                        self.reset_zoom()  # Fit image to window
                        print(f"✅ High quality image loaded successfully: {pixmap.width()}x{pixmap.height()}")
                    else:
                        print("❌ Failed to convert image data to pixmap")
                else:
                    print("❌ No image data returned")
                    
            except Exception as e:
                print(f"❌ Error loading high quality image: {e}")
                import traceback
                traceback.print_exc()
    
    def _convert_to_pixmap(self, image_data):
        """Convert numpy array to QPixmap - traditional approach."""
        try:
            # Ensure contiguous array
            if not image_data.flags['C_CONTIGUOUS']:
                image_data = np.ascontiguousarray(image_data)
            
            if len(image_data.shape) == 2:
                # Grayscale
                h, w = image_data.shape
                bytes_per_line = w
                if image_data.dtype != np.uint8:
                    image_data = ((image_data - image_data.min()) / 
                                 (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                image_bytes = image_data.tobytes()
                q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            elif len(image_data.shape) == 3:
                # Color
                h, w, c = image_data.shape
                if c == 3:
                    bytes_per_line = 3 * w
                    if image_data.dtype != np.uint8:
                        image_data = ((image_data - image_data.min()) / 
                                     (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                    image_bytes = image_data.tobytes()
                    q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
                elif c == 4:
                    bytes_per_line = 4 * w
                    if image_data.dtype != np.uint8:
                        image_data = ((image_data - image_data.min()) / 
                                     (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                    image_bytes = image_data.tobytes()
                    q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    return None
            else:
                return None
            
            return QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"❌ Error in traditional image conversion: {e}")
            return None
    
    def _update_display(self):
        """Update the displayed image with zoom and pan."""
        if not self.display_pixmap:
            return
        
        # Apply zoom
        scaled_pixmap = self.display_pixmap.scaled(
            int(self.display_pixmap.width() * self.zoom_factor),
            int(self.display_pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Create a new pixmap with pan offset
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
            
        final_pixmap = QPixmap(widget_size)
        final_pixmap.fill(Qt.black)
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self.pan_offset, scaled_pixmap)
        painter.end()
        
        self.setPixmap(final_pixmap)
        self.zoom_changed.emit(self.zoom_factor)
    
    def zoom_in(self):
        """Zoom in."""
        self.zoom_factor = min(self.zoom_factor * 1.2, self.max_zoom)
        self._update_display()
    
    def zoom_out(self):
        """Zoom out."""
        self.zoom_factor = max(self.zoom_factor / 1.2, self.min_zoom)
        self._update_display()
    
    def reset_zoom(self):
        """Reset zoom to fit image."""
        if self.display_pixmap:
            widget_size = self.size()
            if widget_size.width() <= 0 or widget_size.height() <= 0:
                return
                
            image_size = self.display_pixmap.size()
            
            scale_x = widget_size.width() / image_size.width()
            scale_y = widget_size.height() / image_size.height()
            self.zoom_factor = min(scale_x, scale_y, 1.0)  # Don't zoom in beyond 100%
            
            # Center the image
            self.pan_offset = QPoint(
                (widget_size.width() - int(image_size.width() * self.zoom_factor)) // 2,
                (widget_size.height() - int(image_size.height() * self.zoom_factor)) // 2
            )
            
            self._update_display()
    
    # Mouse event handling for pan and region selection
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            if self.selection_enabled:
                # Start region selection
                self.is_selecting = True
                self.selection_start = event.pos()
                self.selection_current = event.pos()
                if not self.rubber_band:
                    self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self.rubber_band.setGeometry(QRect(self.selection_start, self.selection_current).normalized())
                self.rubber_band.show()
            else:
                # Start panning
                self.is_panning = True
                self.last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.is_selecting:
            # Update selection rectangle
            self.selection_current = event.pos()
            if self.rubber_band:
                self.rubber_band.setGeometry(QRect(self.selection_start, self.selection_current).normalized())
        elif self.is_panning:
            # Pan the image
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self._update_display()
        
        self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.is_selecting:
                # Finish region selection
                self.is_selecting = False
                if self.rubber_band:
                    selection_rect = QRect(self.selection_start, self.selection_current).normalized()
                    self.region_selected.emit(selection_rect)
                    self.rubber_band.hide()
            elif self.is_panning:
                # Finish panning
                self.is_panning = False
                self.setCursor(Qt.ArrowCursor)
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.display_pixmap:
            # Get mouse position relative to image
            mouse_pos = event.pos()
            
            # Zoom
            zoom_delta = 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
            old_zoom = self.zoom_factor
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * zoom_delta))
            
            # Adjust pan to zoom towards mouse position
            if old_zoom != self.zoom_factor:
                zoom_change = self.zoom_factor / old_zoom
                self.pan_offset = QPoint(
                    int(mouse_pos.x() - (mouse_pos.x() - self.pan_offset.x()) * zoom_change),
                    int(mouse_pos.y() - (mouse_pos.y() - self.pan_offset.y()) * zoom_change)
                )
                self._update_display()
        
        super().wheelEvent(event)
    
    def enable_region_selection(self, enabled):
        """Enable or disable region selection mode."""
        self.selection_enabled = enabled
        if not enabled and self.rubber_band:
            self.rubber_band.hide()
    
    def get_region(self, x, y, width, height):
        """Get a region from the image for export."""
        if not self.image_loader:
            return None
        
        try:
            # Use the image loader to get the region
            return self.image_loader.get_region(x, y, width, height)
        except Exception as e:
            print(f"❌ Error getting region: {e}")
            return None
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        if self.display_pixmap:
            self._update_display()


class ImageCanvas(QLabel):
    """Main image canvas widget for compatibility."""
    
    # Signals for compatibility
    mouse_moved = pyqtSignal(QPoint)
    zoom_changed = pyqtSignal(float)
    region_selected = pyqtSignal(QRect)
    coordinates_changed = pyqtSignal(QPoint)  # Added missing signal
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Simple image display
        self.image_loader = None
        self.display_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.01
        self.max_zoom = 100.0
        
        # Pan variables
        self.last_pan_point = QPoint()
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        
        # Region selection
        self.is_selecting = False
        self.selection_start = QPoint()
        self.selection_current = QPoint()
        self.rubber_band = None
        self.selection_enabled = False
        
        print("✅ Traditional ImageCanvas initialized")
    
    def set_image_loader(self, image_loader):
        """Set the image loader - load high quality but manageable size."""
        print(f"🔗 Setting image loader: {type(image_loader).__name__}")
        self.image_loader = image_loader
        
        # Load high quality but manageable size
        if image_loader:
            try:
                print("📸 Loading high quality image...")
                
                # Get full size dimensions
                height, width = image_loader.image_shape
                print(f"📏 Full image dimensions: {width}x{height}")
                
                # Choose level 1 for good quality but manageable size
                # Level 0 = full size (too big), Level 1 = 1/4 size (good balance)
                level = 1
                level_width = image_loader.zoom_levels[level]['width']
                level_height = image_loader.zoom_levels[level]['height']
                print(f"📐 Loading level {level} image: {level_width}x{level_height}")
                
                # Load level 1 (1/4 resolution but still high quality)
                image_data = image_loader.get_wsi_region(0, 0, level_width, level_height, level)
                
                if image_data is not None:
                    # Convert to pixmap
                    pixmap = self._convert_to_pixmap(image_data)
                    if pixmap:
                        self.display_pixmap = pixmap
                        self.reset_zoom()  # Fit image to window
                        print(f"✅ High quality image loaded successfully: {pixmap.width()}x{pixmap.height()}")
                    else:
                        print("❌ Failed to convert image data to pixmap")
                else:
                    print("❌ No image data returned")
                    
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                import traceback
                traceback.print_exc()
    
    def _convert_to_pixmap(self, image_data):
        """Convert numpy array to QPixmap - traditional approach."""
        try:
            # Ensure contiguous array
            if not image_data.flags['C_CONTIGUOUS']:
                image_data = np.ascontiguousarray(image_data)
            
            if len(image_data.shape) == 2:
                # Grayscale
                h, w = image_data.shape
                bytes_per_line = w
                if image_data.dtype != np.uint8:
                    image_data = ((image_data - image_data.min()) / 
                                 (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                image_bytes = image_data.tobytes()
                q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            elif len(image_data.shape) == 3:
                # Color
                h, w, c = image_data.shape
                if c == 3:
                    bytes_per_line = 3 * w
                    if image_data.dtype != np.uint8:
                        image_data = ((image_data - image_data.min()) / 
                                     (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                    image_bytes = image_data.tobytes()
                    q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
                elif c == 4:
                    bytes_per_line = 4 * w
                    if image_data.dtype != np.uint8:
                        image_data = ((image_data - image_data.min()) / 
                                     (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                    image_bytes = image_data.tobytes()
                    q_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    return None
            else:
                return None
            
            return QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"❌ Error in traditional image conversion: {e}")
            return None
    
    def _update_display(self):
        """Update the displayed image with zoom and pan."""
        if not self.display_pixmap:
            return
        
        # Apply zoom
        scaled_pixmap = self.display_pixmap.scaled(
            int(self.display_pixmap.width() * self.zoom_factor),
            int(self.display_pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Create a new pixmap with pan offset
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
            
        final_pixmap = QPixmap(widget_size)
        final_pixmap.fill(Qt.black)
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self.pan_offset, scaled_pixmap)
        painter.end()
        
        self.setPixmap(final_pixmap)
        self.zoom_changed.emit(self.zoom_factor)
    
    def zoom_in(self):
        """Zoom in."""
        self.zoom_factor = min(self.zoom_factor * 1.2, self.max_zoom)
        self._update_display()
    
    def zoom_out(self):
        """Zoom out."""
        self.zoom_factor = max(self.zoom_factor / 1.2, self.min_zoom)
        self._update_display()
    
    def reset_zoom(self):
        """Reset zoom to fit image."""
        if self.display_pixmap:
            widget_size = self.size()
            if widget_size.width() <= 0 or widget_size.height() <= 0:
                return
                
            image_size = self.display_pixmap.size()
            
            scale_x = widget_size.width() / image_size.width()
            scale_y = widget_size.height() / image_size.height()
            self.zoom_factor = min(scale_x, scale_y, 1.0)  # Don't zoom in beyond 100%
            
            # Center the image
            self.pan_offset = QPoint(
                (widget_size.width() - int(image_size.width() * self.zoom_factor)) // 2,
                (widget_size.height() - int(image_size.height() * self.zoom_factor)) // 2
            )
            
            self._update_display()
    
    # Mouse event handling for pan and region selection
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            if self.selection_enabled:
                # Start region selection
                self.is_selecting = True
                self.selection_start = event.pos()
                self.selection_current = event.pos()
                if not self.rubber_band:
                    self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self.rubber_band.setGeometry(QRect(self.selection_start, self.selection_current).normalized())
                self.rubber_band.show()
            else:
                # Start panning
                self.is_panning = True
                self.last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.is_selecting:
            # Update selection rectangle
            self.selection_current = event.pos()
            if self.rubber_band:
                self.rubber_band.setGeometry(QRect(self.selection_start, self.selection_current).normalized())
        elif self.is_panning:
            # Pan the image
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self._update_display()
        
        self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.is_selecting:
                # Finish region selection
                self.is_selecting = False
                if self.rubber_band:
                    selection_rect = QRect(self.selection_start, self.selection_current).normalized()
                    self.region_selected.emit(selection_rect)
                    self.rubber_band.hide()
            elif self.is_panning:
                # Finish panning
                self.is_panning = False
                self.setCursor(Qt.ArrowCursor)
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.display_pixmap:
            # Get mouse position relative to image
            mouse_pos = event.pos()
            
            # Zoom
            zoom_delta = 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
            old_zoom = self.zoom_factor
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * zoom_delta))
            
            # Adjust pan to zoom towards mouse position
            if old_zoom != self.zoom_factor:
                zoom_change = self.zoom_factor / old_zoom
                self.pan_offset = QPoint(
                    int(mouse_pos.x() - (mouse_pos.x() - self.pan_offset.x()) * zoom_change),
                    int(mouse_pos.y() - (mouse_pos.y() - self.pan_offset.y()) * zoom_change)
                )
                self._update_display()
        
        super().wheelEvent(event)
    
    def enable_region_selection(self, enabled):
        """Enable or disable region selection mode."""
        self.selection_enabled = enabled
        if not enabled and self.rubber_band:
            self.rubber_band.hide()
    
    def get_region(self, x, y, width, height):
        """Get a region from the image for export."""
        if not self.image_loader:
            return None
        
        try:
            # Use the image loader to get the region
            return self.image_loader.get_region(x, y, width, height)
        except Exception as e:
            print(f"❌ Error getting region: {e}")
            return None
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        if self.display_pixmap:
            self._update_display()
    # Additional compatibility methods expected by main window
    def connect_region_selected(self, callback):
        """Connect region selected signal - compatibility method."""
        self.region_selected.connect(callback)
    
    def set_image(self, image_array):
        """Set image from numpy array - compatibility method."""
        if image_array is None:
            self.display_pixmap = None
            self.clear()
            return
        
        pixmap = self._convert_to_pixmap(image_array)
        if pixmap:
            self.display_pixmap = pixmap
            self.reset_zoom()
    
    def get_current_image(self):
        """Get current image as numpy array - compatibility method."""
        # This is a simplified version - would need more complex implementation
        # for full compatibility
        return None
    
    def fit_to_window(self):
        """Fit image to window - compatibility alias."""
        self.reset_zoom()
    
    def actual_size(self):
        """Show image at actual size - compatibility method."""
        self.zoom_factor = 1.0
        self._update_display()
    
    def get_zoom_level(self):
        """Get current zoom level - compatibility method."""
        return self.zoom_factor
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        if self.display_pixmap:
            self._update_display()
