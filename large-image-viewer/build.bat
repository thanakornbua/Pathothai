@echo off
REM Build script for Large Image Viewer executable

echo Large Image Viewer - Build Script
echo ==================================

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo Error: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PyQt5, numpy, cv2, PIL, skimage, h5py, zarr, dask, psutil" 2>nul
if errorlevel 1 (
    echo Installing required dependencies...
    pip install PyQt5 numpy opencv-python pillow scikit-image h5py zarr dask psutil
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Clean previous build
echo Cleaning previous build...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build

REM Build executable
echo Building executable...
pyinstaller build_spec.spec

if errorlevel 1 (
    echo Error: Build failed
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable location: dist\LargeImageViewer.exe
echo.

REM Test the executable
echo Testing executable...
cd dist
LargeImageViewer.exe --help 2>nul

echo.
echo Build script completed.
pause
