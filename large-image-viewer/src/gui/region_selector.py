"""
Region selection widget for selecting and saving specific parts of large images.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QLabel, QSpinBox, QListWidget, QListWidgetItem,
                            QLineEdit, QComboBox, QProgressBar, QMessageBox,
                            QFileDialog, QCheckBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

class RegionExportThread(QThread):
    """Background thread for exporting regions."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, image_loader, regions, output_dir, format_type, naming_format="name"):
        super().__init__()
        self.image_loader = image_loader
        self.regions = regions
        self.output_dir = output_dir
        self.format_type = format_type
        self.naming_format = naming_format  # "name", "position_6digit", "custom"
    
    def run(self):
        try:
            total_regions = len(self.regions)
            
            for i, region in enumerate(self.regions):
                # Extract region
                x, y, width, height = region['x'], region['y'], region['width'], region['height']
                name = region['name']
                
                # Create output filename based on naming format
                if self.naming_format == "position_6digit":
                    # Format: x######_y######_w####_h####.ext
                    filename = f"x{x:06d}_y{y:06d}_w{width:04d}_h{height:04d}.{self.format_type.lower()}"
                elif self.naming_format == "position_coordinates":
                    # Format: region_x_y_w_h.ext
                    filename = f"region_{x}_{y}_{width}_{height}.{self.format_type.lower()}"
                elif self.naming_format == "position_detailed":
                    # Format: position_x000000_y000000_size_000x000.ext
                    filename = f"position_x{x:06d}_y{y:06d}_size_{width:03d}x{height:03d}.{self.format_type.lower()}"
                else:
                    # Default: use region name
                    filename = f"{name}.{self.format_type.lower()}"
                
                output_path = f"{self.output_dir}/{filename}"
                
                # Save region
                success = self.image_loader.save_region(x, y, width, height, output_path, self.format_type)
                
                if not success:
                    self.finished.emit(False, f"Failed to save region: {name}")
                    return
                
                # Update progress
                progress = int(((i + 1) / total_regions) * 100)
                self.progress.emit(progress)
            
            self.finished.emit(True, f"Successfully exported {total_regions} regions")
            
        except Exception as e:
            self.finished.emit(False, str(e))

class RegionSelectionOverlay(QWidget):
    """Overlay widget for drawing selection rectangles on the image canvas."""
    
    region_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: transparent;")
        
        # Selection state
        self.is_selecting = False
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.current_selection = QRect()
        
        # Existing regions
        self.regions = []
        
        # Selection mode
        self.selection_enabled = False
    
    def enable_selection(self, enabled):
        """Enable or disable region selection."""
        self.selection_enabled = enabled
        if not enabled:
            self.is_selecting = False
            self.current_selection = QRect()
        self.update()
    
    def add_region(self, x, y, width, height, name):
        """Add a region to display."""
        region = {
            'rect': QRect(x, y, width, height),
            'name': name,
            'color': QColor(255, 0, 0, 100)  # Semi-transparent red
        }
        self.regions.append(region)
        self.update()
    
    def clear_regions(self):
        """Clear all regions."""
        self.regions.clear()
        self.update()
    
    def mousePressEvent(self, event):
        """Start region selection."""
        if self.selection_enabled and event.button() == Qt.LeftButton:
            self.is_selecting = True
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            self.current_selection = QRect(self.selection_start, self.selection_end)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Update selection during drag."""
        if self.is_selecting:
            self.selection_end = event.pos()
            self.current_selection = QRect(self.selection_start, self.selection_end).normalized()
            self.update()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Complete region selection."""
        if self.is_selecting and event.button() == Qt.LeftButton:
            self.is_selecting = False
            
            # Emit selection if it's large enough
            if self.current_selection.width() > 10 and self.current_selection.height() > 10:
                self.region_selected.emit(
                    self.current_selection.x(),
                    self.current_selection.y(),
                    self.current_selection.width(),
                    self.current_selection.height()
                )
            
            self.current_selection = QRect()
            self.update()
        super().mouseReleaseEvent(event)
    
    def paintEvent(self, event):
        """Draw regions and current selection."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing regions
        for region in self.regions:
            # Fill
            painter.fillRect(region['rect'], QBrush(region['color']))
            
            # Border
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(region['rect'])
            
            # Label
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(region['rect'].topLeft() + QPoint(5, 15), region['name'])
        
        # Draw current selection
        if self.is_selecting and not self.current_selection.isEmpty():
            # Fill
            painter.fillRect(self.current_selection, QBrush(QColor(0, 255, 0, 50)))
            
            # Border
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_selection)

class RegionSelectionWidget(QWidget):
    """Widget for managing region selection and export."""
    
    def __init__(self, image_loader=None):
        super().__init__()
        self.image_loader = image_loader
        self.regions = []
        self.export_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        
        # Selection controls
        selection_group = QGroupBox("Region Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        # Selection mode toggle
        self.selection_checkbox = QCheckBox("Enable Region Selection")
        self.selection_checkbox.toggled.connect(self.on_selection_toggled)
        selection_layout.addWidget(self.selection_checkbox)
        
        # Manual region input
        manual_frame = QFrame()
        manual_layout = QHBoxLayout(manual_frame)
        
        manual_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(0, 999999)  # Increased to support large images
        self.x_spinbox.setMinimumWidth(100)  # Make wider to show more digits
        manual_layout.addWidget(self.x_spinbox)
        
        manual_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(0, 999999)  # Increased to support large images
        self.y_spinbox.setMinimumWidth(100)  # Make wider to show more digits
        manual_layout.addWidget(self.y_spinbox)
        
        manual_layout.addWidget(QLabel("W:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 999999)  # Increased to support large images
        self.width_spinbox.setMinimumWidth(100)  # Make wider to show more digits
        self.width_spinbox.setValue(256)
        manual_layout.addWidget(self.width_spinbox)
        
        manual_layout.addWidget(QLabel("H:"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 999999)  # Increased to support large images
        self.height_spinbox.setMinimumWidth(100)  # Make wider to show more digits
        self.height_spinbox.setValue(256)
        manual_layout.addWidget(self.height_spinbox)
        
        selection_layout.addWidget(manual_frame)
        
        # Region name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Region name...")
        name_layout.addWidget(self.name_input)
        
        add_region_btn = QPushButton("Add Region")
        add_region_btn.clicked.connect(self.add_manual_region)
        name_layout.addWidget(add_region_btn)
        
        selection_layout.addLayout(name_layout)
        
        layout.addWidget(selection_group)
        
        # Regions list
        regions_group = QGroupBox("Selected Regions")
        regions_layout = QVBoxLayout(regions_group)
        
        self.regions_list = QListWidget()
        self.regions_list.setSelectionMode(QListWidget.MultiSelection)
        regions_layout.addWidget(self.regions_list)
        
        # Region controls
        controls_layout = QHBoxLayout()
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_regions)
        controls_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_regions)
        controls_layout.addWidget(clear_btn)
        
        regions_layout.addLayout(controls_layout)
        
        layout.addWidget(regions_group)
        
        # Export controls
        export_group = QGroupBox("Export Regions")
        export_layout = QVBoxLayout(export_group)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['PNG', 'JPEG', 'TIFF', 'BMP'])
        self.format_combo.currentTextChanged.connect(self.update_naming_preview)
        format_layout.addWidget(self.format_combo)
        
        export_layout.addLayout(format_layout)
        
        # Naming format selection
        naming_layout = QHBoxLayout()
        naming_layout.addWidget(QLabel("Naming:"))
        
        self.naming_combo = QComboBox()
        self.naming_combo.addItems([
            'Region Name', 
            '6-Digit Position', 
            'Simple Coordinates',
            'Detailed Position'
        ])
        self.naming_combo.setCurrentText('Region Name')
        self.naming_combo.currentTextChanged.connect(self.on_naming_format_changed)
        naming_layout.addWidget(self.naming_combo)
        
        export_layout.addLayout(naming_layout)
        
        # Naming format preview
        self.naming_preview = QLabel("Preview: Region_1.png")
        self.naming_preview.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        export_layout.addWidget(self.naming_preview)
        
        # Export buttons
        export_buttons_layout = QHBoxLayout()
        
        export_selected_btn = QPushButton("Export Selected")
        export_selected_btn.clicked.connect(self.export_selected_regions)
        export_buttons_layout.addWidget(export_selected_btn)
        
        export_all_btn = QPushButton("Export All")
        export_all_btn.clicked.connect(self.export_all_regions)
        export_buttons_layout.addWidget(export_all_btn)
        
        export_layout.addLayout(export_buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        export_layout.addWidget(self.progress_bar)
        
        layout.addWidget(export_group)
        
        # Preview group
        preview_group = QGroupBox("Region Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Select a region to see preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        layout.addStretch()
        
        # Connect signals
        self.regions_list.itemSelectionChanged.connect(self.on_region_selection_changed)
        
        # Initialize naming preview
        self.update_naming_preview()
    
    def set_image_loader(self, image_loader):
        """Set the image loader instance."""
        self.image_loader = image_loader
        
        # Update spinbox ranges based on image size
        if image_loader and image_loader.is_loaded:
            height, width = image_loader.image_shape[:2]
            self.x_spinbox.setRange(0, width - 1)
            self.y_spinbox.setRange(0, height - 1)
            self.width_spinbox.setRange(1, width)
            self.height_spinbox.setRange(1, height)
    
    def on_selection_toggled(self, enabled):
        """Handle selection mode toggle."""
        # This would be connected to the overlay widget
        pass
    
    def add_region(self, x, y, width, height, name=None):
        """Add a region to the list."""
        if name is None:
            name = f"Region_{len(self.regions) + 1}"
        
        region = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'name': name
        }
        
        self.regions.append(region)
        
        # Add to list widget
        item_text = f"{name} ({x}, {y}, {width}x{height})"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, len(self.regions) - 1)  # Store region index
        self.regions_list.addItem(item)
        
        return len(self.regions) - 1
    
    def add_manual_region(self):
        """Add a manually specified region."""
        x = self.x_spinbox.value()
        y = self.y_spinbox.value()
        width = self.width_spinbox.value()
        height = self.height_spinbox.value()
        name = self.name_input.text().strip() or f"Region_{len(self.regions) + 1}"
        
        self.add_region(x, y, width, height, name)
        
        # Clear name input
        self.name_input.clear()
    
    def on_naming_format_changed(self):
        """Update filename preview when naming format changes."""
        self.update_naming_preview()
    
    def update_naming_preview(self):
        """Update the naming format preview."""
        naming_format = self.naming_combo.currentText()
        file_format = self.format_combo.currentText().lower()
        
        # Sample coordinates for preview - use larger values to show 6-digit formatting
        sample_x, sample_y, sample_w, sample_h = 123456, 567890, 256, 256
        
        if naming_format == '6-Digit Position':
            preview = f"x{sample_x:06d}_y{sample_y:06d}_w{sample_w:04d}_h{sample_h:04d}.{file_format}"
        elif naming_format == 'Simple Coordinates':
            preview = f"region_{sample_x}_{sample_y}_{sample_w}_{sample_h}.{file_format}"
        elif naming_format == 'Detailed Position':
            preview = f"position_x{sample_x:06d}_y{sample_y:06d}_size_{sample_w:03d}x{sample_h:03d}.{file_format}"
        else:  # Region Name
            preview = f"Region_1.{file_format}"
        
        self.naming_preview.setText(f"Preview: {preview}")
    
    def get_naming_format_key(self):
        """Get the internal key for the selected naming format."""
        naming_format = self.naming_combo.currentText()
        
        format_map = {
            'Region Name': 'name',
            '6-Digit Position': 'position_6digit',
            'Simple Coordinates': 'position_coordinates',
            'Detailed Position': 'position_detailed'
        }
        
        return format_map.get(naming_format, 'name')
    
    def remove_selected_regions(self):
        """Remove selected regions from the list."""
        selected_items = self.regions_list.selectedItems()
        
        # Get region indices to remove (in reverse order)
        indices_to_remove = []
        for item in selected_items:
            index = item.data(Qt.UserRole)
            indices_to_remove.append(index)
        
        indices_to_remove.sort(reverse=True)
        
        # Remove regions and items
        for index in indices_to_remove:
            if 0 <= index < len(self.regions):
                del self.regions[index]
            
            # Remove from list widget
            self.regions_list.takeItem(self.regions_list.row(selected_items[indices_to_remove.index(index)]))
        
        # Update remaining item indices
        for i in range(self.regions_list.count()):
            item = self.regions_list.item(i)
            item.setData(Qt.UserRole, i)
    
    def clear_all_regions(self):
        """Clear all regions."""
        self.regions.clear()
        self.regions_list.clear()
        self.preview_label.setText("Select a region to see preview")
    
    def on_region_selection_changed(self):
        """Handle region selection change for preview."""
        selected_items = self.regions_list.selectedItems()
        
        if selected_items and self.image_loader and self.image_loader.is_loaded:
            # Show preview of first selected region
            item = selected_items[0]
            region_index = item.data(Qt.UserRole)
            
            if 0 <= region_index < len(self.regions):
                region = self.regions[region_index]
                self.show_region_preview(region)
        else:
            self.preview_label.setText("Select a region to see preview")
    
    def show_region_preview(self, region):
        """Show a preview of the selected region."""
        if not self.image_loader or not self.image_loader.is_loaded:
            return
        
        try:
            # Extract region
            region_data = self.image_loader.get_region(
                region['x'], region['y'], region['width'], region['height']
            )
            
            if region_data is not None:
                # Convert to QPixmap for display
                from PyQt5.QtGui import QPixmap, QImage
                
                if len(region_data.shape) == 2:
                    # Grayscale
                    height, width = region_data.shape
                    bytes_per_line = width
                    
                    # Normalize to 0-255
                    if region_data.dtype != np.uint8:
                        region_data = ((region_data - region_data.min()) / 
                                     (region_data.max() - region_data.min()) * 255).astype(np.uint8)
                    
                    q_image = QImage(region_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                
                elif len(region_data.shape) == 3:
                    # Color
                    height, width, channels = region_data.shape
                    
                    if channels == 3:
                        bytes_per_line = 3 * width
                        
                        # Normalize to 0-255
                        if region_data.dtype != np.uint8:
                            region_data = ((region_data - region_data.min()) / 
                                         (region_data.max() - region_data.min()) * 255).astype(np.uint8)
                        
                        q_image = QImage(region_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    else:
                        # Use first channel for preview
                        channel_data = region_data[:, :, 0]
                        height, width = channel_data.shape
                        bytes_per_line = width
                        
                        if channel_data.dtype != np.uint8:
                            channel_data = ((channel_data - channel_data.min()) / 
                                          (channel_data.max() - channel_data.min()) * 255).astype(np.uint8)
                        
                        q_image = QImage(channel_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                
                # Scale to fit preview
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.preview_label.setText(f"Error loading preview: {e}")
    
    def export_selected_regions(self):
        """Export selected regions."""
        selected_items = self.regions_list.selectedItems()
        
        if not selected_items:
            QMessageBox.information(self, "Info", "No regions selected for export.")
            return
        
        # Get selected regions
        regions_to_export = []
        for item in selected_items:
            region_index = item.data(Qt.UserRole)
            if 0 <= region_index < len(self.regions):
                regions_to_export.append(self.regions[region_index])
        
        self._export_regions(regions_to_export)
    
    def export_all_regions(self):
        """Export all regions."""
        if not self.regions:
            QMessageBox.information(self, "Info", "No regions to export.")
            return
        
        self._export_regions(self.regions)
    
    def _export_regions(self, regions):
        """Export the specified regions."""
        if not self.image_loader or not self.image_loader.is_loaded:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
        
        # Get format and naming format
        format_type = self.format_combo.currentText()
        naming_format = self.get_naming_format_key()
        
        # Start export thread
        self.export_thread = RegionExportThread(
            self.image_loader, regions, output_dir, format_type, naming_format
        )
        self.export_thread.progress.connect(self.progress_bar.setValue)
        self.export_thread.finished.connect(self.on_export_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.export_thread.start()
    
    def on_export_finished(self, success, message):
        """Handle export completion."""
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Export Complete", message)
        else:
            QMessageBox.critical(self, "Export Failed", message)
    
    def get_regions(self):
        """Get all regions."""
        return self.regions.copy()
    
    def load_regions_from_file(self, file_path):
        """Load regions from a JSON file."""
        try:
            import json
            with open(file_path, 'r') as f:
                regions_data = json.load(f)
            
            self.clear_all_regions()
            for region_data in regions_data:
                self.add_region(
                    region_data['x'], region_data['y'],
                    region_data['width'], region_data['height'],
                    region_data['name']
                )
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load regions: {e}")
            return False
    
    def save_regions_to_file(self, file_path):
        """Save regions to a JSON file."""
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(self.regions, f, indent=2)
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save regions: {e}")
            return False
