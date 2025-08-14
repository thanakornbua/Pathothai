# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Add src directory to path
src_path = os.path.join(os.path.dirname(SPECPATH), 'src')
sys.path.insert(0, src_path)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/core', 'core'),
        ('src/gui', 'gui'),
    ],
    hiddenimports=[
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'numpy',
        'cv2',
        'PIL',
        'skimage',
        'h5py',
        'zarr',
        'dask',
        'psutil',
        'pydicom',
        'SimpleITK',
        'nibabel',
        'tifffile',
        'lz4',
        'blosc',
        'scipy',
        'numba',
        'gui.main_window',
        'gui.image_canvas',
        'gui.channel_controls',
        'gui.region_selector',
        'core.enhanced_image_loader',
        'core.image_processor',
        'core.memory_manager'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LargeImageViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Add icon file if available
)
