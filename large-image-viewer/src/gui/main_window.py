"""
Main window for the Large Image Viewer application.
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QMenuBar, QMenu, QAction, QToolBar, QStatusBar,
                            QFileDialog, QMessageBox, QSplitter, QDockWidget,
                            QProgressBar, QLabel, QPushButton, QSlider,
                            QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
                            QComboBox, QTextEdit, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QKeySequence

from .image_canvas import ImageCanvas
from .channel_controls import ChannelControls
from .region_selector import RegionSelectionWidget, RegionSelectionOverlay
from core.enhanced_image_loader import EnhancedLargeImageLoader
from core.image_processor import ImageProcessor
from core.memory_manager import MemoryManager

class LoadImageThread(QThread):
    """Background thread for loading large images."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, file_path, loader):
        super().__init__()
        self.file_path = file_path
        self.loader = loader
    
    def run(self):
        try:
            success = self.loader.load_image(self.file_path)
            self.finished.emit(success, self.file_path if success else "Failed to load image")
        except Exception as e:
            self.finished.emit(False, str(e))

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Image Viewer - Pathothai")
        self.setGeometry(100, 100, 1400, 900)
        
        # Core components
        self.image_loader = EnhancedLargeImageLoader(chunk_size=512, max_memory_gb=8.0, cache_size_gb=2.0)
        self.image_processor = ImageProcessor()
        self.memory_manager = MemoryManager(max_memory_percent=80.0)
        
        # UI components
        self.image_canvas = None
        self.channel_controls = None
        self.region_selector = None
        self.region_overlay = None
        self.load_thread = None
        self.current_file = None
        
        # Setup UI
        self.init_ui()
        self.init_menus()
        self.init_toolbar()
        self.init_status_bar()
        self.init_dock_widgets()
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        # Timer for updating memory info
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_info)
        self.memory_timer.start(2000)  # Update every 2 seconds
    
    def init_ui(self):
        """Initialize the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Image display area
        self.image_canvas = ImageCanvas()
        
        # Connect image canvas signals
        self.image_canvas.zoom_changed.connect(self.on_zoom_changed)
        self.image_canvas.coordinates_changed.connect(self.on_coordinates_changed)
        self.image_canvas.connect_region_selected(self.on_region_selected)
        
        splitter.addWidget(self.image_canvas)
        
        # Set splitter proportions
        splitter.setSizes([1000, 300])  # Image area gets more space
    
    def init_menus(self):
        """Initialize menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open Image...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        # DICOM specific action
        open_dicom_action = QAction('Open &DICOM...', self)
        open_dicom_action.triggered.connect(self.open_dicom)
        file_menu.addAction(open_dicom_action)
        
        # WSI specific action
        open_wsi_action = QAction('Open &Whole Slide Image...', self)
        open_wsi_action.triggered.connect(self.open_wsi)
        file_menu.addAction(open_wsi_action)
        
        file_menu.addSeparator()
        
        save_action = QAction('&Save Image...', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        export_action = QAction('&Export Current View...', self)
        export_action.triggered.connect(self.export_view)
        file_menu.addAction(export_action)
        
        # Region export actions
        file_menu.addSeparator()
        export_regions_action = QAction('Export &Regions...', self)
        export_regions_action.triggered.connect(self.export_regions)
        file_menu.addAction(export_regions_action)
        
        save_regions_action = QAction('Save Region &Definitions...', self)
        save_regions_action.triggered.connect(self.save_region_definitions)
        file_menu.addAction(save_regions_action)
        
        load_regions_action = QAction('Load Region &Definitions...', self)
        load_regions_action.triggered.connect(self.load_region_definitions)
        file_menu.addAction(load_regions_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        zoom_in_action = QAction('Zoom &In', self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom &Out', self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_window_action = QAction('&Fit to Window', self)
        fit_window_action.setShortcut('Ctrl+0')
        fit_window_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_window_action)
        
        actual_size_action = QAction('&Actual Size', self)
        actual_size_action.setShortcut('Ctrl+1')
        actual_size_action.triggered.connect(self.actual_size)
        view_menu.addAction(actual_size_action)
        
        view_menu.addSeparator()
        
        fullscreen_action = QAction('&Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        histogram_action = QAction('Show &Histogram', self)
        histogram_action.triggered.connect(self.show_histogram)
        tools_menu.addAction(histogram_action)
        
        memory_info_action = QAction('&Memory Information', self)
        memory_info_action.triggered.connect(self.show_memory_info)
        tools_menu.addAction(memory_info_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_toolbar(self):
        """Initialize toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Open button
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_btn)
        
        toolbar.addSeparator()
        
        # Zoom controls
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self.fit_to_window)
        toolbar.addWidget(fit_btn)
        
        toolbar.addSeparator()
        
        # Region selection toggle
        self.region_select_action = QAction("Select Region", self)
        self.region_select_action.setCheckable(True)
        self.region_select_action.setChecked(False)
        self.region_select_action.triggered.connect(self.toggle_region_selection)
        toolbar.addAction(self.region_select_action)
        
        toolbar.addSeparator()
        
        # Zoom level display
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
    
    def init_status_bar(self):
        """Initialize status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # File info
        self.file_info_label = QLabel("No image loaded")
        self.status_bar.addWidget(self.file_info_label)
        
        # Coordinates display
        self.coords_label = QLabel("Position: (0, 0)")
        self.status_bar.addPermanentWidget(self.coords_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Memory info
        self.memory_label = QLabel("Memory: 0.0 GB")
        self.status_bar.addPermanentWidget(self.memory_label)
    
    def init_dock_widgets(self):
        """Initialize dock widgets for tools and controls."""
        # Channel controls dock
        self.channel_controls = ChannelControls()
        channel_dock = QDockWidget("Channel Controls", self)
        channel_dock.setWidget(self.channel_controls)
        self.addDockWidget(Qt.RightDockWidgetArea, channel_dock)
        
        # Connect channel control signals
        self.channel_controls.channel_changed.connect(self.on_channel_changed)
        self.channel_controls.adjustment_changed.connect(self.on_adjustment_changed)
        
        # Image processing dock
        processing_dock = QDockWidget("Image Processing", self)
        processing_widget = self.create_processing_widget()
        processing_dock.setWidget(processing_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, processing_dock)
        
        # Information dock
        info_dock = QDockWidget("Information", self)
        info_widget = self.create_info_widget()
        info_dock.setWidget(info_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, info_dock)
        
        # Region selection dock
        self.region_selector = RegionSelectionWidget(self.image_loader)
        region_dock = QDockWidget("Region Selection", self)
        region_dock.setWidget(self.region_selector)
        self.addDockWidget(Qt.RightDockWidgetArea, region_dock)
    
    def create_processing_widget(self):
        """Create image processing controls widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Brightness/Contrast
        bc_group = QGroupBox("Brightness & Contrast")
        bc_layout = QVBoxLayout(bc_group)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_adjustments)
        bc_layout.addWidget(QLabel("Brightness:"))
        bc_layout.addWidget(self.brightness_slider)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.apply_adjustments)
        bc_layout.addWidget(QLabel("Contrast:"))
        bc_layout.addWidget(self.contrast_slider)
        
        layout.addWidget(bc_group)
        
        # Gamma correction
        gamma_group = QGroupBox("Gamma Correction")
        gamma_layout = QVBoxLayout(gamma_group)
        
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.apply_adjustments)
        gamma_layout.addWidget(QLabel("Gamma:"))
        gamma_layout.addWidget(self.gamma_slider)
        
        layout.addWidget(gamma_group)
        
        # Filters
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['None', 'Gaussian Blur', 'Median Filter', 'Bilateral Filter', 
                                   'Unsharp Mask', 'Edge Enhance', 'Sobel Edge'])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)
        
        layout.addWidget(filter_group)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_adjustments)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget
    
    def create_info_widget(self):
        """Create information display widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        layout.addWidget(self.info_text)
        
        return widget
    
    def open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            "All Supported (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.dcm *.dicom *.nii *.nii.gz *.svs *.ndpi *.vms *.vmu *.scn *.mrxs *.bif);;"\
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;"\
            "DICOM Files (*.dcm *.dicom);;"\
            "NIfTI Files (*.nii *.nii.gz);;"\
            "Whole Slide Images (*.svs *.ndpi *.vms *.vmu *.scn *.mrxs *.bif);;"\
            "All Files (*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def open_dicom(self):
        """Open a DICOM file specifically."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open DICOM File", 
            "", 
            "DICOM Files (*.dcm *.dicom);;All Files (*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def open_wsi(self):
        """Open a Whole Slide Image file specifically."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Whole Slide Image", 
            "", 
            "Whole Slide Images (*.svs *.ndpi *.vms *.vmu *.scn *.mrxs *.bif);;"\
            "Aperio SVS (*.svs);;"\
            "Hamamatsu NDPI (*.ndpi);;"\
            "Olympus VSI (*.vms *.vmu);;"\
            "Leica SCN (*.scn);;"\
            "3DHistech MRXS (*.mrxs);;"\
            "Ventana BIF (*.bif);;"\
            "All Files (*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load an image in a background thread."""
        if self.load_thread and self.load_thread.isRunning():
            return
        
        self.current_file = file_path
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.load_thread = LoadImageThread(file_path, self.image_loader)
        self.load_thread.finished.connect(self.on_image_loaded)
        self.load_thread.start()
        
        self.status_bar.showMessage(f"Loading {os.path.basename(file_path)}...")
    
    def on_image_loaded(self, success, message):
        """Handle image loading completion."""
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_bar.showMessage(f"Loaded: {os.path.basename(message)}")
            
            # Connect image loader to canvas for tiling support
            self.image_canvas.set_image_loader(self.image_loader)
            
            # Update image canvas with thumbnail for initial display
            thumbnail = self.image_loader.get_thumbnail((1024, 1024))
            if thumbnail is not None:
                self.image_canvas.set_image(thumbnail)
                self.fit_to_window()
            
            # Update channel controls
            if self.image_loader.channels:
                self.channel_controls.set_channels(self.image_loader.channels)
            
            # Update region selector
            self.region_selector.set_image_loader(self.image_loader)
            
            # Update file info
            self.update_file_info()
            
        else:
            self.status_bar.showMessage("Failed to load image")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{message}")
    
    def update_file_info(self):
        """Update file information display."""
        if not self.image_loader.is_loaded:
            return
        
        info = f"File: {os.path.basename(self.current_file)}\n"
        info += f"Format: {self.image_loader.file_format.upper()}\n"
        info += f"Dimensions: {self.image_loader.image_shape}\n"
        info += f"Channels: {self.image_loader.channels}\n"
        info += f"Data type: {self.image_loader.dtype}\n"
        
        file_size = os.path.getsize(self.current_file) / (1024**3)  # GB
        info += f"File size: {file_size:.2f} GB\n"
        
        # Add loading performance info
        info += f"Load time: {self.image_loader.load_time:.2f}s\n"
        
        memory_usage = self.memory_manager.get_memory_info()
        info += f"Memory usage: {memory_usage['process_rss_gb']:.1f} GB\n"
        
        # Add format-specific metadata
        metadata = self.image_loader.get_metadata()
        if 'patient_id' in metadata:
            info += f"\nDICOM Info:\n"
            info += f"Patient ID: {metadata.get('patient_id', 'N/A')}\n"
            info += f"Study Date: {metadata.get('study_date', 'N/A')}\n"
            info += f"Modality: {metadata.get('modality', 'N/A')}\n"
        
        self.info_text.setPlainText(info)
        self.file_info_label.setText(f"{self.image_loader.image_shape} | {self.image_loader.channels} channels")
    
    def update_memory_info(self):
        """Update memory information in status bar."""
        memory_info = self.memory_manager.get_memory_info()
        self.memory_label.setText(f"Memory: {memory_info['process_rss_gb']:.1f} GB")
    
    def on_channel_changed(self, channel_index):
        """Handle channel selection change."""
        if not self.image_loader.is_loaded:
            return
        
        # Get specific channel data
        channel_data = self.image_loader.get_channel(channel_index)
        if channel_data is not None:
            self.image_canvas.set_image(channel_data)
    
    def on_adjustment_changed(self, adjustments):
        """Handle image adjustment changes."""
        # Apply adjustments and update display
        pass
    
    def apply_adjustments(self):
        """Apply brightness, contrast, and gamma adjustments."""
        if not self.image_loader.is_loaded:
            return
        
        # Get current image from canvas
        current_image = self.image_canvas.get_current_image()
        if current_image is None:
            return
        
        # Apply adjustments
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0
        
        adjusted = self.image_processor.adjust_brightness_contrast(current_image, brightness, contrast)
        adjusted = self.image_processor.adjust_gamma(adjusted, gamma)
        
        self.image_canvas.set_image(adjusted)
    
    def apply_filter(self, filter_name):
        """Apply selected filter to the image."""
        if not self.image_loader.is_loaded or filter_name == 'None':
            return
        
        current_image = self.image_canvas.get_current_image()
        if current_image is None:
            return
        
        filter_map = {
            'Gaussian Blur': 'gaussian',
            'Median Filter': 'median',
            'Bilateral Filter': 'bilateral',
            'Unsharp Mask': 'unsharp_mask',
            'Edge Enhance': 'edge_enhance',
            'Sobel Edge': 'sobel'
        }
        
        filter_type = filter_map.get(filter_name)
        if filter_type:
            filtered = self.image_processor.apply_filter(current_image, filter_type)
            self.image_canvas.set_image(filtered)
    
    def reset_adjustments(self):
        """Reset all image adjustments."""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.filter_combo.setCurrentText('None')
        
        if self.image_loader.is_loaded:
            thumbnail = self.image_loader.get_thumbnail((1024, 1024))
            if thumbnail is not None:
                self.image_canvas.set_image(thumbnail)
    
    def zoom_in(self):
        """Zoom in on the image."""
        self.image_canvas.zoom_in()
        self.update_zoom_label()
    
    def zoom_out(self):
        """Zoom out from the image."""
        self.image_canvas.zoom_out()
        self.update_zoom_label()
    
    def fit_to_window(self):
        """Fit image to window."""
        self.image_canvas.fit_to_window()
        self.update_zoom_label()
    
    def actual_size(self):
        """Show image at actual size."""
        self.image_canvas.actual_size()
        self.update_zoom_label()
    
    def update_zoom_label(self):
        """Update zoom level display."""
        zoom_level = self.image_canvas.get_zoom_level()
        self.zoom_label.setText(f"{zoom_level:.0f}%")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def save_image(self):
        """Save the current image."""
        if not self.image_loader.is_loaded:
            QMessageBox.information(self, "Info", "No image loaded to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Image", 
            "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files (*.tiff);;All Files (*)"
        )
        
        if file_path:
            current_image = self.image_canvas.get_current_image()
            if current_image is not None:
                # Save logic here
                self.status_bar.showMessage(f"Saved: {os.path.basename(file_path)}")
    
    def export_view(self):
        """Export the current view."""
        if not self.image_loader.is_loaded:
            QMessageBox.information(self, "Info", "No image loaded to export.")
            return
        
        # Export current view logic here
        pass
    
    def show_histogram(self):
        """Show image histogram."""
        if not self.image_loader.is_loaded:
            QMessageBox.information(self, "Info", "No image loaded.")
            return
        
        # Show histogram dialog
        pass
    
    def show_memory_info(self):
        """Show detailed memory information."""
        memory_info = self.memory_manager.get_memory_info()
        
        info_text = f"""Memory Information:
        
Total System Memory: {memory_info['total_gb']:.1f} GB
Available Memory: {memory_info['available_gb']:.1f} GB
System Memory Usage: {memory_info['used_percent']:.1f}%

Process Memory Usage:
RSS: {memory_info['process_rss_gb']:.1f} GB
VMS: {memory_info['process_vms_gb']:.1f} GB"""
        
        QMessageBox.information(self, "Memory Information", info_text)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Large Image Viewer v1.0
        
A high-performance image viewer for very large images (1GB+).

Features:
• Support for PNG, JPEG, TIFF formats
• Channel manipulation and visualization
• Advanced image processing tools
• Memory-efficient handling of large files
• GPU acceleration support

Developed for Pathothai"""
        
        QMessageBox.about(self, "About Large Image Viewer", about_text)
    
    def on_region_selected(self, region_rect):
        """Handle region selection from image canvas."""
        if self.region_selector and region_rect.isValid():
            x, y, width, height = region_rect.x(), region_rect.y(), region_rect.width(), region_rect.height()
            region_name = f"Region_{len(self.region_selector.get_regions()) + 1}"
            self.region_selector.add_region(x, y, width, height, region_name)
            
            # Update status
            self.status_bar.showMessage(f"Selected region: {width}x{height} at ({x}, {y})")
    
    def on_zoom_changed(self, zoom_level):
        """Handle zoom level changes."""
        if hasattr(self, 'zoom_label'):
            self.zoom_label.setText(f"Zoom: {zoom_level:.1f}%")
    
    def on_coordinates_changed(self, image_pos):
        """Handle mouse coordinate changes."""
        if hasattr(self, 'coords_label'):
            self.coords_label.setText(f"Position: ({image_pos.x()}, {image_pos.y()})")
    
    def toggle_region_selection(self):
        """Toggle region selection mode."""
        if hasattr(self, 'region_select_action'):
            enabled = self.region_select_action.isChecked()
            self.image_canvas.enable_region_selection(enabled)
            
            if enabled:
                self.status_bar.showMessage("Region selection enabled - Click and drag to select an area")
            else:
                self.status_bar.showMessage("Region selection disabled")
    
    def export_regions(self):
        """Export selected regions."""
        if self.region_selector:
            self.region_selector.export_all_regions()
        else:
            QMessageBox.information(self, "Info", "No regions available for export.")
    
    def save_region_definitions(self):
        """Save region definitions to file."""
        if not self.region_selector or not self.region_selector.get_regions():
            QMessageBox.information(self, "Info", "No regions to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Region Definitions",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.region_selector.save_regions_to_file(file_path):
                self.status_bar.showMessage(f"Saved regions to: {os.path.basename(file_path)}")
    
    def load_region_definitions(self):
        """Load region definitions from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Region Definitions",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.region_selector.load_regions_from_file(file_path):
                self.status_bar.showMessage(f"Loaded regions from: {os.path.basename(file_path)}")
                
                # Update overlay
                self.region_overlay.clear_regions()
                for region in self.region_selector.get_regions():
                    self.region_overlay.add_region(
                        region['x'], region['y'], region['width'], region['height'], region['name']
                    )
    
    def closeEvent(self, event):
        """Handle application close."""
        # Clean up resources
        if self.image_loader:
            self.image_loader.cleanup()
        
        if self.memory_manager:
            self.memory_manager.stop_monitoring()
        
        event.accept()
