# Large Image Viewer

A high-performance image viewer specifically designed for handling very large images (up to 5GB+ file sizes) with advanced channel manipulation capabilities and medical imaging support.

## Features

### Core Capabilities
- **Massive File Support**: Efficiently handles PNG, JPEG, TIFF, and DICOM files up to 5GB+
- **Medical Imaging**: Full DICOM support with metadata preservation
- **Memory Management**: Intelligent chunked loading and memory-mapped file access
- **GPU Acceleration**: CUDA support for enhanced processing performance
- **Multi-threading**: Optimized parallel processing for better responsiveness

### Image Viewing
- **Zoom & Pan**: Smooth zooming up to 5000% with precise pan controls
- **Fit to Window**: Automatic image scaling to fit viewport
- **Actual Size**: Quick access to 100% zoom level
- **Fullscreen Mode**: Distraction-free viewing experience

### Advanced Features
- **Region Selection**: Interactive selection and export of specific image regions
- **Selective Export**: Save only portions of large images to reduce file sizes
- **DICOM Metadata**: Display patient information, study details, and technical parameters
- **Multi-format Support**: Seamless handling of scientific and medical image formats

### Channel Management
- **Individual Channel Control**: View and manipulate each image channel separately
- **Channel Visualization**: Color-coded channel display with customizable colors
- **Opacity Control**: Adjust transparency for each channel independently
- **Blend Modes**: Multiple blending options (Normal, Add, Multiply, Screen, Overlay)

### Image Processing
- **Real-time Adjustments**: Brightness, contrast, and gamma correction
- **Filtering**: Gaussian blur, median filter, bilateral filter, edge enhancement
- **Histogram Display**: Statistical analysis of image data
- **Export Options**: Save processed images or current view

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **RAM**: 16 GB (32 GB recommended for 5GB+ files)
- **Storage**: 200 MB free space
- **Python**: 3.8+ (for source installation)

### Recommended Requirements
- **RAM**: 64 GB for optimal performance with 5GB+ files
- **GPU**: CUDA-compatible GPU for acceleration
- **SSD**: NVMe SSD for improved large file loading

## Installation

### Option 1: Executable (Recommended)
1. Download the latest release from the releases page
2. Extract the archive
3. Run `LargeImageViewer.exe` (Windows) or `LargeImageViewer` (Linux/Mac)

### Option 2: From Source
```bash
# Clone the repository
git clone https://github.com/pathothai/large-image-viewer.git
cd large-image-viewer

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Option 3: Build Your Own Executable
```bash
# Windows
build.bat

# Linux/Mac
chmod +x build.sh
./build.sh
```

## Usage

### Opening Images
- **File Menu**: Use `File > Open Image` or press `Ctrl+O`
- **DICOM Files**: Use `File > Open DICOM` for medical images
- **Command Line**: `LargeImageViewer image.png` or `LargeImageViewer scan.dcm`
- **Drag & Drop**: Drag image files directly into the application

### Region Selection & Export
1. **Enable Selection**: Check "Enable Region Selection" in the Region Selection panel
2. **Draw Regions**: Click and drag on the image to select areas
3. **Manual Input**: Enter exact coordinates in the input fields
4. **Export Options**:
   - **Export Selected**: Save only chosen regions
   - **Export All**: Save all defined regions
   - **Format Options**: PNG, JPEG, TIFF, BMP
5. **Save/Load Definitions**: Save region coordinates for reuse

### Navigation
- **Zoom In**: `Ctrl++` or mouse wheel up (with Ctrl)
- **Zoom Out**: `Ctrl+-` or mouse wheel down (with Ctrl)
- **Pan**: Click and drag with left mouse button
- **Fit to Window**: `Ctrl+0`
- **Actual Size**: `Ctrl+1`
- **Fullscreen**: `F11`

### Channel Controls
1. **View Modes**:
   - **Composite**: View all channels combined
   - **Single Channel**: View one channel at a time
   - **Channels Separate**: Individual channel windows

2. **Channel Adjustments**:
   - Toggle channel visibility with checkboxes
   - Adjust colors using color picker buttons
   - Control opacity, brightness, and contrast with sliders

3. **Global Controls**:
   - **Reset All**: Return all channels to default settings
   - **Auto Adjust**: Automatically optimize channel settings

### Image Processing
- **Brightness/Contrast**: Real-time adjustment sliders
- **Gamma Correction**: Gamma curve adjustment
- **Filters**: Apply various image filters from dropdown menu
- **Reset**: Return to original image state

## File Format Support

### Supported Formats
- **PNG**: All variants including 16-bit and multi-channel
- **JPEG**: Standard JPEG and JPEG 2000
- **TIFF**: Uncompressed and compressed TIFF files, including scientific formats
- **DICOM**: Medical imaging format with full metadata support
- **NIfTI**: Neuroimaging format (.nii, .nii.gz)
- **Multi-channel Images**: Support for scientific and medical imaging formats

### Large File Optimization (5GB+)
- **Chunked Loading**: Processes images in manageable chunks (512x512 default)
- **Memory Mapping**: Direct file access without full loading into RAM
- **Progressive Display**: Show image while loading in background
- **Cache Management**: Intelligent caching for frequently accessed regions
- **Compressed Storage**: Optional compression for memory efficiency

## Performance Tips

### For Very Large Files (5GB+)
1. **Increase Available RAM**: Close other applications, consider 64GB+ RAM
2. **Use NVMe SSD Storage**: Fastest possible file access is crucial
3. **Enable GPU Acceleration**: Ensure CUDA drivers are installed
4. **Adjust Chunk Size**: Smaller chunks (256-512) for extreme files
5. **Region-based Workflow**: Use region selection to work with manageable portions

### Memory Management
- Monitor memory usage in the status bar
- Use `Tools > Memory Information` for detailed statistics
- The application automatically manages memory to prevent crashes

## Troubleshooting

### Common Issues

**Application Won't Start**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.7+ required)
- Verify PyQt5 installation: `python -c "import PyQt5"`

**Large File Loading Errors**
- Check available RAM (8GB minimum for 1GB+ files)
- Ensure sufficient disk space for temporary files
- Try reducing image size or converting to supported format

**Performance Issues**
- Close other memory-intensive applications
- Reduce zoom level for smoother navigation
- Enable GPU acceleration if available
- Adjust chunk size in settings

**Display Problems**
- Update graphics drivers
- Try different Qt styles: `app.setStyle('Fusion')`
- Check system display scaling settings

### Error Reporting
If you encounter issues:
1. Check the console output for error messages
2. Note your system specifications
3. Include the problematic image file format and size
4. Report issues with detailed reproduction steps

## Development

### Project Structure
```
large-image-viewer/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── build_spec.spec        # PyInstaller configuration
├── build.bat/.sh          # Build scripts
├── src/
│   ├── core/              # Core processing modules
│   │   ├── image_loader.py    # Large image loading
│   │   ├── image_processor.py # Image processing
│   │   └── memory_manager.py  # Memory management
│   └── gui/               # User interface
│       ├── main_window.py     # Main application window
│       ├── image_canvas.py    # Image display widget
│       └── channel_controls.py # Channel control widgets
└── dist/                  # Built executables (after build)
```

### Building from Source
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pyinstaller

# Run tests (if available)
python -m pytest tests/

# Build executable
pyinstaller build_spec.spec
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyQt5 for cross-platform GUI
- Uses OpenCV, PIL, and scikit-image for image processing
- Leverages H5PY, Zarr, and Dask for large file handling
- GPU acceleration powered by PyTorch/CUDA

## Version History

### v1.0.0
- Initial release
- Basic large image viewing capabilities
- Channel manipulation and processing tools
- Cross-platform executable support

---

**Developed for Pathothai**  
For support and updates, visit: [Project Repository](https://github.com/pathothai/large-image-viewer)
