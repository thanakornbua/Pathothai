# C Image Viewer (Windows)

Features:
- Supports .png, .jpg/.jpeg, .svs (via OpenSlide), .dcm (via GDCM)
- Handles very large images by reading regions and limiting GPU texture size
- Channel view: press 0 for RGB, 1=R, 2=G, 3=B

## Build (x64, MSVC)
Requires: CMake, vcpkg (for SDL2, GDCM, OpenSlide) or preinstalled libs.

### Example with vcpkg
1. Install vcpkg and integrate with MSVC
2. Install libs:
   - `vcpkg install sdl2:x64-windows gdcm:x64-windows openslide:x64-windows`
3. Configure and build:
   - `cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake" -DWITH_OPENSLIDE=ON -DWITH_DICOM=ON`
   - `cmake --build build --config Release`

Run:
- `build/bin/c_image_viewer.exe <path-to-image>`

## MSI packaging (WiX)
- Generate an MSI using WiX Toolset (heat.exe + candle.exe + light.exe). A simple Product.wxs can be added later.

## Notes
- SVS requires OpenSlide.
- DICOM uses GDCM; supports 8-bit uncompressed images in this minimal example.
- Very large textures are center-cropped to max 8192x8192 for display stability.
