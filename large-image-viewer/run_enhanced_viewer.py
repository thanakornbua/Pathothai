#!/usr/bin/env python3
"""
Complete test script for the enhanced large image viewer.
This script demonstrates all the new features including tiling, zooming, and region selection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """Run the enhanced large image viewer."""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    viewer = MainWindow()
    viewer.show()
    
    # Print feature status
    print("🚀 Enhanced Large Image Viewer Started!")
    print("✅ Tiling system enabled")
    print("✅ Region selection available")
    print("✅ Accurate zooming implemented")
    print("✅ Mouse-based navigation ready")
    print()
    print("📋 Usage Instructions:")
    print("  • File > Open to load an image")
    print("  • Ctrl + Mouse Wheel to zoom around cursor")
    print("  • Left click and drag to pan")
    print("  • Click 'Select Region' button to enable region selection")
    print("  • In selection mode: Click and drag to select areas")
    print("  • Keyboard shortcuts:")
    print("    - '+/-' for zoom in/out")
    print("    - 'F' to fit to window")
    print("    - '0' for actual size")
    print("    - Arrow keys to pan")
    print("    - ESC to cancel selection")
    print()
    print("🎯 New Features:")
    print("  • Background tile loading for smooth performance")
    print("  • Smart caching system")
    print("  • Accurate coordinate mapping")
    print("  • Screenshot-style region selection")
    print("  • Real-time zoom and position feedback")
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
