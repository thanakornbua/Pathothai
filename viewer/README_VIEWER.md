# Universal Image Viewer

A simple PyQt5 GUI to open and view DICOM, PNG, JPG, and SVS images. Supports single-channel viewing, extracting a region by coordinates (up to 8-digit values), and saving the extracted region.

Features
- Open images: .dcm, .png, .jpg/.jpeg, .svs
- View a single channel (R/G/B or gray)
- Enter region as x,y,width,height and display it
- Save last extracted region as PNG/JPG
- Efficient SVS handling: overview from lowest level, on-demand region reads

Requirements
- Python 3.9+
- PyQt5
- numpy
- Pillow
- pydicom
- openslide-python (and OpenSlide runtime)

Install dependencies (Windows PowerShell)

Optional: Create/activate a venv first.

Usage
Run the app from the repo root:

python c:\Users\tanth\Desktop\Pathothai\universal_image_viewer.py

Packaging to MSI
We use cx_Freeze to build an MSI installer.

1) Install build tools:
- pip install cx-Freeze

2) Build MSI:
- python setup_cxfreeze.py bdist_msi

The MSI will be in build\exe.win-amd64-* or dist/ depending on cx_Freeze version. If OpenSlide DLLs are needed, ensure they are discoverable at runtime (System PATH) or include them as build includes.

Notes
- For SVS, large region requests can be slow; prefer smaller areas.
- DICOM normalization scales the full frame min..max to 0..255 for display only.
