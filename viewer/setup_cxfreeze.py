import sys
import os
from typing import List

# Pylance: cx_Freeze is only required for building, ignore if missing in dev env
from cx_Freeze import setup, Executable  # type: ignore

base = None
if sys.platform == "win32":
    base = "Win32GUI"

app_script = os.path.join(os.path.dirname(__file__), "universal_image_viewer.py")

include_files: List = []

# Optionally include OpenSlide DLLs if OPENSLIDE_PATH is set to its bin folder
openslide_path = os.environ.get("OPENSLIDE_PATH")
if openslide_path and os.path.isdir(openslide_path):
    # Include all DLLs from the provided bin directory
    for name in os.listdir(openslide_path):
        if name.lower().endswith('.dll'):
            include_files.append(os.path.join(openslide_path, name))

# Manually include only the Qt 'platforms' plugin dir; exclude other plugin groups
platforms_dir_env = os.path.join(sys.prefix, "Library", "plugins", "platforms")
if os.path.isdir(platforms_dir_env):
    include_files.append((platforms_dir_env, "platforms"))
    # Generate a minimal qt.conf so Qt finds plugins relative to the exe
    try:
        qt_conf_path = os.path.join(os.path.dirname(__file__), "qt.conf")
        with open(qt_conf_path, "w", encoding="utf-8") as f:
            f.write("[Paths]\nPlugins=platforms\n")
        include_files.append((qt_conf_path, "qt.conf"))
    except Exception:
        pass

# Include key Qt/OpenGL DLLs required by the qwindows platform plugin
bin_dir = os.path.join(sys.prefix, "Library", "bin")
for dll_name in [
    "libEGL.dll",
    "libGLESv2.dll",
    "D3DCompiler_47.dll",
    "opengl32sw.dll",
]:
    dll_path = os.path.join(bin_dir, dll_name)
    if os.path.isfile(dll_path):
        include_files.append(dll_path)

# Include core Qt5 DLLs commonly required by the qwindows plugin
for dll_name in [
    "Qt5Core.dll",
    "Qt5Gui.dll",
    "Qt5Widgets.dll",
    "Qt5Network.dll",
    "Qt5Svg.dll",
]:
    dll_path = os.path.join(bin_dir, dll_name)
    if os.path.isfile(dll_path):
        include_files.append(dll_path)

# Include ICU libraries if present (needed by Qt for text/locale)
if os.path.isdir(bin_dir):
    for name in os.listdir(bin_dir):
        if name.lower().startswith("icu") and name.lower().endswith(".dll"):
            include_files.append(os.path.join(bin_dir, name))

build_options = {
    # Keep the module graph minimal and explicit
    "packages": [
        "numpy",
        "PIL",
        "pydicom",
    "openslide",
    ],
    "includes": [
        # Explicit submodules actually imported by the app
        "PIL.Image",
        "PIL.ImageQt",
        "numpy",
        "pydicom",
        "openslide",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "PyQt5.QtCore",
    ],
    "include_files": include_files,
    # Exclude large or irrelevant modules and those known to cause deep hook scans
    "excludes": [
        "tkinter",
        "matplotlib",
        "scipy",
        "IPython",
        "jedi",
        "notebook",
        "numba",
        "sympy",
        "skimage",
        "sklearn",
        "setuptools",
        "pip",
        "wheel",
        "pkg_resources",
        "importlib_metadata",
        "PyQt5.QtWebEngineWidgets",
        "PyQt5.QtWebEngineCore",
        "PyQtWebEngine",
    # Exclude Qt QML/Quick to avoid qthooks trying to bundle QML paths we don't use
    "PyQt5.QtQml",
    "PyQt5.QtQuick",
    "PyQt5.QtQmlModels",
    "PyQt5.QtQuickWidgets",
        "OpenGL",
        "pygments",
        "pytest",
        "pluggy",
        "packaging",
    ],
    # Avoid zipping packages to keep things simple for Qt and hooks
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    # Optimize bytecode
    "optimize": 1,
    # Exclude Qt plugins root to prevent hooks from copying unwanted groups (designer, etc.)
    "bin_path_excludes": [
        os.path.join(sys.prefix, "Library", "plugins"),
        os.path.join(sys.prefix, "plugins"),
    ],
    # Ensure the Qt bin directory is on PATH at runtime so qwindows.dll can find its deps
    "bin_path_includes": [
        os.path.join(sys.prefix, "Library", "bin"),
    ],
    # Include MSVC runtimes
    "include_msvcr": True,
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
