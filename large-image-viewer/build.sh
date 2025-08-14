#!/bin/bash
# Build script for Large Image Viewer executable

echo "Large Image Viewer - Build Script"
echo "=================================="

# Check if PyInstaller is installed
python3 -c "import PyInstaller" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyInstaller"
        exit 1
    fi
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import PyQt5, numpy, cv2, PIL, skimage, h5py, zarr, dask, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    pip3 install PyQt5 numpy opencv-python pillow scikit-image h5py zarr dask psutil
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf dist build

# Build executable
echo "Building executable..."
pyinstaller build_spec.spec

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo "Executable location: dist/LargeImageViewer"
echo ""

# Test the executable
echo "Testing executable..."
cd dist
./LargeImageViewer --help 2>/dev/null

echo ""
echo "Build script completed."
