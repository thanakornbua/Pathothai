# Large Image Viewer

This project is a large image viewing application that can handle PNG and JPG files up to 1GB in size. It provides a graphical user interface (GUI) for viewing and manipulating image channels, making it suitable for users who need to work with high-resolution images.

## Features

- View large PNG and JPG images.
- Manipulate individual image channels (e.g., RGB).
- Zoom and pan functionality for detailed inspection.
- Efficient memory management for handling large files.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd large-image-viewer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage

- Open the application and use the file menu to load an image.
- Use the channel controls to view and manipulate the individual channels.
- Zoom in and out using the mouse wheel or toolbar buttons.
- Pan the image by clicking and dragging.

## Building Executable

To create an executable version of the application, run the following command:
```
python build_exe.py
```

This will generate a standalone executable that can be run on any compatible Windows machine.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.