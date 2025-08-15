import sys
import os
from typing import List

# Pylance: cx_Freeze is only required for building, ignore if missing in dev env
from cx_Freeze import setup, Executable  # type: ignore

base = None
if sys.platform == "win32":
    base = "Win32GUI"

app_script = os.path.join(os.path.dirname(__file__), "universal_image_viewer.py")

include_files: List[str] = []

# Optionally include OpenSlide DLLs if OPENSLIDE_PATH is set to its bin folder
openslide_path = os.environ.get("OPENSLIDE_PATH")
if openslide_path and os.path.isdir(openslide_path):
    # Include all DLLs from the provided bin directory
    for name in os.listdir(openslide_path):
        if name.lower().endswith('.dll'):
            include_files.append(os.path.join(openslide_path, name))

build_options = {
    "includes": [
        "numpy",
        "PIL.Image",
        "pydicom",
        "openslide",
        "PyQt5",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "PyQt5.QtCore",
    ],
    "include_files": include_files,
    # Exclude large or irrelevant modules
    "excludes": ["tkinter", "matplotlib", "scipy"],
    "optimize": 1,
}

executables = [
    Executable(app_script, base=base, target_name="UniversalImageViewer.exe", icon=None),
]

setup(
    name="Universal Image Viewer",
    version="0.1.0",
    description="Viewer for DICOM/PNG/JPG/SVS with channel and region support",
    options={"build_exe": build_options},
    executables=executables,
)
