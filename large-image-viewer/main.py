"""
Large Image Viewer Application
A high-performance image viewer for very large images (1GB+).
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the application."""
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        from gui.main_window import MainWindow
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Large Image Viewer")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("Pathothai")
        
        # Set application style
        app.setStyle('Fusion')  # Modern cross-platform style
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        print("✅ Traditional large image viewer ready!")
        
        # Handle command line arguments
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.exists(file_path):
                main_window.load_image(file_path)
        
        # Start event loop
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Error: Required dependencies not installed - {e}")
        print("\nTo install required dependencies, run:")
        print("pip install PyQt5 numpy opencv-python pillow scikit-image h5py zarr dask")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
