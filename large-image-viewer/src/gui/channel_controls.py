"""
Channel controls widget for managing image channels and their properties.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QCheckBox, QSlider, QLabel, QPushButton, QComboBox,
                            QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
                            QColorDialog, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPalette, QPixmap, QPainter

class ColorButton(QPushButton):
    """Custom button for color selection."""
    
    color_changed = pyqtSignal(QColor)
    
    def __init__(self, color=QColor(255, 255, 255)):
        super().__init__()
        self.setFixedSize(30, 30)
        self.current_color = color
        self.update_color_display()
        self.clicked.connect(self.select_color)
    
    def update_color_display(self):
        """Update the button appearance to show current color."""
        pixmap = QPixmap(28, 28)
        pixmap.fill(self.current_color)
        self.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: {self.current_color.name()};
            }}
            QPushButton:hover {{
                border: 2px solid #666;
            }}
        """)
    
    def select_color(self):
        """Open color dialog and update color."""
        color = QColorDialog.getColor(self.current_color, self, "Select Color")
        if color.isValid():
            self.current_color = color
            self.update_color_display()
            self.color_changed.emit(color)
    
    def set_color(self, color):
        """Set the color programmatically."""
        self.current_color = color
        self.update_color_display()

class ChannelItem(QWidget):
    """Widget representing a single channel with controls."""
    
    # Signals
    visibility_changed = pyqtSignal(int, bool)
    color_changed = pyqtSignal(int, QColor)
    opacity_changed = pyqtSignal(int, float)
    brightness_changed = pyqtSignal(int, float)
    contrast_changed = pyqtSignal(int, float)
    
    def __init__(self, channel_index, channel_name, default_color=None):
        super().__init__()
        self.channel_index = channel_index
        self.channel_name = channel_name
        
        # Default colors for common channels
        default_colors = {
            0: QColor(255, 0, 0),    # Red
            1: QColor(0, 255, 0),    # Green
            2: QColor(0, 0, 255),    # Blue
            3: QColor(255, 255, 0),  # Yellow
            4: QColor(255, 0, 255),  # Magenta
            5: QColor(0, 255, 255),  # Cyan
        }
        
        if default_color is None:
            default_color = default_colors.get(channel_index, QColor(255, 255, 255))
        
        self.init_ui(default_color)
    
    def init_ui(self, default_color):
        """Initialize the channel control UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Header with checkbox and color
        header_layout = QHBoxLayout()
        
        # Visibility checkbox
        self.visibility_checkbox = QCheckBox(self.channel_name)
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.toggled.connect(
            lambda checked: self.visibility_changed.emit(self.channel_index, checked)
        )
        header_layout.addWidget(self.visibility_checkbox)
        
        header_layout.addStretch()
        
        # Color button
        self.color_button = ColorButton(default_color)
        self.color_button.color_changed.connect(
            lambda color: self.color_changed.emit(self.channel_index, color)
        )
        header_layout.addWidget(self.color_button)
        
        layout.addLayout(header_layout)
        
        # Opacity control
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_changed.emit(self.channel_index, value / 100.0)
        )
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("100%")
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_label.setText(f"{value}%")
        )
        opacity_layout.addWidget(self.opacity_label)
        
        layout.addLayout(opacity_layout)
        
        # Brightness control
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(
            lambda value: self.brightness_changed.emit(self.channel_index, value)
        )
        brightness_layout.addWidget(self.brightness_slider)
        
        self.brightness_label = QLabel("0")
        self.brightness_slider.valueChanged.connect(
            lambda value: self.brightness_label.setText(str(value))
        )
        brightness_layout.addWidget(self.brightness_label)
        
        layout.addLayout(brightness_layout)
        
        # Contrast control
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(
            lambda value: self.contrast_changed.emit(self.channel_index, value / 100.0)
        )
        contrast_layout.addWidget(self.contrast_slider)
        
        self.contrast_label = QLabel("100%")
        self.contrast_slider.valueChanged.connect(
            lambda value: self.contrast_label.setText(f"{value}%")
        )
        contrast_layout.addWidget(self.contrast_label)
        
        layout.addLayout(contrast_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
    
    def get_settings(self):
        """Get current channel settings."""
        return {
            'visible': self.visibility_checkbox.isChecked(),
            'color': self.color_button.current_color,
            'opacity': self.opacity_slider.value() / 100.0,
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value() / 100.0
        }
    
    def set_settings(self, settings):
        """Apply channel settings."""
        if 'visible' in settings:
            self.visibility_checkbox.setChecked(settings['visible'])
        if 'color' in settings:
            self.color_button.set_color(settings['color'])
        if 'opacity' in settings:
            self.opacity_slider.setValue(int(settings['opacity'] * 100))
        if 'brightness' in settings:
            self.brightness_slider.setValue(settings['brightness'])
        if 'contrast' in settings:
            self.contrast_slider.setValue(int(settings['contrast'] * 100))
    
    def reset_to_defaults(self):
        """Reset channel to default settings."""
        self.visibility_checkbox.setChecked(True)
        self.opacity_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)

class ChannelControls(QWidget):
    """Main channel controls widget."""
    
    # Signals
    channel_changed = pyqtSignal(int)  # Selected channel index
    adjustment_changed = pyqtSignal(dict)  # All adjustments
    blend_mode_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.channel_items = []
        self.current_channels = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Channel selection group
        selection_group = QGroupBox("Channel Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        # View mode selection
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View Mode:"))
        
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(['Composite', 'Single Channel', 'Channels Separate'])
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        view_layout.addWidget(self.view_mode_combo)
        
        selection_layout.addLayout(view_layout)
        
        # Single channel selection (hidden by default)
        self.single_channel_widget = QWidget()
        self.single_channel_layout = QHBoxLayout(self.single_channel_widget)
        self.single_channel_layout.addWidget(QLabel("Channel:"))
        
        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self.on_channel_selected)
        self.single_channel_layout.addWidget(self.channel_selector)
        
        selection_layout.addWidget(self.single_channel_widget)
        self.single_channel_widget.setVisible(False)
        
        layout.addWidget(selection_group)
        
        # Blend mode group
        blend_group = QGroupBox("Blend Mode")
        blend_layout = QVBoxLayout(blend_group)
        
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(['Normal', 'Add', 'Multiply', 'Screen', 'Overlay'])
        self.blend_mode_combo.currentTextChanged.connect(self.blend_mode_changed.emit)
        blend_layout.addWidget(self.blend_mode_combo)
        
        layout.addWidget(blend_group)
        
        # Individual channel controls
        self.channels_group = QGroupBox("Channel Controls")
        self.channels_layout = QVBoxLayout(self.channels_group)
        
        # Scroll area for channels
        from PyQt5.QtWidgets import QScrollArea
        self.channels_scroll = QScrollArea()
        self.channels_scroll.setWidgetResizable(True)
        self.channels_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.channels_widget = QWidget()
        self.channels_widget_layout = QVBoxLayout(self.channels_widget)
        self.channels_widget_layout.setContentsMargins(0, 0, 0, 0)
        
        self.channels_scroll.setWidget(self.channels_widget)
        self.channels_layout.addWidget(self.channels_scroll)
        
        layout.addWidget(self.channels_group)
        
        # Global controls
        global_group = QGroupBox("Global Controls")
        global_layout = QVBoxLayout(global_group)
        
        # Reset all button
        reset_all_btn = QPushButton("Reset All Channels")
        reset_all_btn.clicked.connect(self.reset_all_channels)
        global_layout.addWidget(reset_all_btn)
        
        # Auto adjust button
        auto_adjust_btn = QPushButton("Auto Adjust")
        auto_adjust_btn.clicked.connect(self.auto_adjust_channels)
        global_layout.addWidget(auto_adjust_btn)
        
        layout.addWidget(global_group)
        
        layout.addStretch()
    
    def set_channels(self, channel_count):
        """Set up controls for the specified number of channels."""
        # Clear existing channel items
        for item in self.channel_items:
            item.setParent(None)
        self.channel_items.clear()
        
        # Clear channel selector
        self.channel_selector.clear()
        
        # Create new channel items
        for i in range(channel_count):
            channel_name = f"Channel {i + 1}"
            
            # Add to selector
            self.channel_selector.addItem(channel_name)
            
            # Create channel control item
            channel_item = ChannelItem(i, channel_name)
            
            # Connect signals
            channel_item.visibility_changed.connect(self.on_channel_visibility_changed)
            channel_item.color_changed.connect(self.on_channel_color_changed)
            channel_item.opacity_changed.connect(self.on_channel_opacity_changed)
            channel_item.brightness_changed.connect(self.on_channel_brightness_changed)
            channel_item.contrast_changed.connect(self.on_channel_contrast_changed)
            
            self.channels_widget_layout.addWidget(channel_item)
            self.channel_items.append(channel_item)
        
        self.current_channels = list(range(channel_count))
        
        # Update view
        self.update_adjustment_signal()
    
    def on_view_mode_changed(self, mode):
        """Handle view mode change."""
        if mode == 'Single Channel':
            self.single_channel_widget.setVisible(True)
            self.channels_group.setVisible(False)
        else:
            self.single_channel_widget.setVisible(False)
            self.channels_group.setVisible(True)
        
        if mode == 'Single Channel' and self.channel_selector.count() > 0:
            self.channel_changed.emit(self.channel_selector.currentIndex())
        else:
            self.channel_changed.emit(-1)  # Composite mode
    
    def on_channel_selected(self, index):
        """Handle single channel selection."""
        if self.view_mode_combo.currentText() == 'Single Channel':
            self.channel_changed.emit(index)
    
    def on_channel_visibility_changed(self, channel_index, visible):
        """Handle channel visibility change."""
        self.update_adjustment_signal()
    
    def on_channel_color_changed(self, channel_index, color):
        """Handle channel color change."""
        self.update_adjustment_signal()
    
    def on_channel_opacity_changed(self, channel_index, opacity):
        """Handle channel opacity change."""
        self.update_adjustment_signal()
    
    def on_channel_brightness_changed(self, channel_index, brightness):
        """Handle channel brightness change."""
        self.update_adjustment_signal()
    
    def on_channel_contrast_changed(self, channel_index, contrast):
        """Handle channel contrast change."""
        self.update_adjustment_signal()
    
    def update_adjustment_signal(self):
        """Emit adjustment changed signal with current settings."""
        adjustments = {
            'view_mode': self.view_mode_combo.currentText(),
            'blend_mode': self.blend_mode_combo.currentText(),
            'channels': {}
        }
        
        for i, item in enumerate(self.channel_items):
            adjustments['channels'][i] = item.get_settings()
        
        self.adjustment_changed.emit(adjustments)
    
    def reset_all_channels(self):
        """Reset all channels to default settings."""
        for item in self.channel_items:
            item.reset_to_defaults()
        
        self.view_mode_combo.setCurrentText('Composite')
        self.blend_mode_combo.setCurrentText('Normal')
        
        self.update_adjustment_signal()
    
    def auto_adjust_channels(self):
        """Auto-adjust channel settings based on image statistics."""
        # This would analyze the current image and automatically adjust
        # brightness, contrast, etc. for optimal viewing
        # For now, just emit a signal that could be handled by the main window
        self.update_adjustment_signal()
    
    def get_current_adjustments(self):
        """Get current adjustment settings."""
        adjustments = {
            'view_mode': self.view_mode_combo.currentText(),
            'blend_mode': self.blend_mode_combo.currentText(),
            'channels': {}
        }
        
        for i, item in enumerate(self.channel_items):
            adjustments['channels'][i] = item.get_settings()
        
        return adjustments
    
    def apply_adjustments(self, adjustments):
        """Apply adjustment settings."""
        if 'view_mode' in adjustments:
            self.view_mode_combo.setCurrentText(adjustments['view_mode'])
        
        if 'blend_mode' in adjustments:
            self.blend_mode_combo.setCurrentText(adjustments['blend_mode'])
        
        if 'channels' in adjustments:
            for channel_index, settings in adjustments['channels'].items():
                if channel_index < len(self.channel_items):
                    self.channel_items[channel_index].set_settings(settings)
    
    def export_settings(self):
        """Export current settings to a dictionary."""
        return self.get_current_adjustments()
    
    def import_settings(self, settings):
        """Import settings from a dictionary."""
        self.apply_adjustments(settings)
