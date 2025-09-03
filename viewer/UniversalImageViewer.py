import sys
import os
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QComboBox,
    QLineEdit,
    QHBoxLayout,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QRubberBand,
    QCheckBox,
    QGraphicsPathItem,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
)
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPen, QPainterPath, QColor, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QRect, QPoint, QPointF, QSize, QObject, QThread
import numpy as np

try:
    import openslide
except ImportError:
    openslide = None
try:
    import pydicom
except ImportError:
    pydicom = None
from PIL import Image, ImageFile
# Disable Pillow DOS protection for very large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_FORMATS = ['.svs', '.dcm', '.png', '.jpg', '.jpeg']


# ------------- Utility functions (cached) -------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    if a_max <= a_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - a_min) / (a_max - a_min)
    return (scaled * 255.0).astype(np.uint8)


@lru_cache(maxsize=2048)
def _clip_polygon_to_rect_cached(points: Tuple[Tuple[float, float], ...], rect: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], ...]:
    # Sutherland–Hodgman polygon clipping for axis-aligned rect
    if not points:
        return tuple()
    rx, ry, rw, rh = rect
    left = rx
    right = rx + rw
    top = ry
    bottom = ry + rh

    def inside_left(p):
        return p[0] >= left

    def inside_right(p):
        return p[0] <= right

    def inside_top(p):
        return p[1] >= top

    def inside_bottom(p):
        return p[1] <= bottom

    def intersect(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2 and y1 == y2:
            return (x1, y1)
        if edge == 'left':
            x = left
            t = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
            y = y1 + t * (y2 - y1)
            return (x, y)
        if edge == 'right':
            x = right
            t = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
            y = y1 + t * (y2 - y1)
            return (x, y)
        if edge == 'top':
            y = top
            t = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            x = x1 + t * (x2 - x1)
            return (x, y)
        if edge == 'bottom':
            y = bottom
            t = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            x = x1 + t * (x2 - x1)
            return (x, y)
        return p2

    def clip_edge(points_in, inside_fn, edge_name):
        if not points_in:
            return []
        out = []
        s = points_in[-1]
        for e in points_in:
            if inside_fn(e):
                if inside_fn(s):
                    out.append(e)
                else:
                    out.append(intersect(s, e, edge_name))
                    out.append(e)
            else:
                if inside_fn(s):
                    out.append(intersect(s, e, edge_name))
            s = e
        return out

    out = list(points)
    out = clip_edge(out, inside_left, 'left')
    if not out:
        return tuple()
    out = clip_edge(out, inside_right, 'right')
    if not out:
        return tuple()
    out = clip_edge(out, inside_top, 'top')
    if not out:
        return tuple()
    out = clip_edge(out, inside_bottom, 'bottom')
    return tuple((float(x), float(y)) for (x, y) in out)


def _parse_qupath_xml(xml_path: str) -> List[Dict]:
    # Robust parser for common annotation XML (Aperio/QuPath style)
    import xml.etree.ElementTree as ET
    regions: List[Dict] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Determine annotation label from Attributes if present
    for annotation in root.findall('.//Annotation'):
        annotation_label: Optional[str] = None
        try:
            attr = annotation.find('.//Attributes/Attribute')
            if attr is not None:
                annotation_label = attr.attrib.get('Value') or attr.attrib.get('Name') or ''
        except Exception:
            pass
        if not annotation_label:
            annotation_label = annotation.attrib.get('Name', '')
        for region in annotation.findall('.//Regions/Region'):
            label = region.attrib.get('Text') or annotation_label or ''
            coords: List[Tuple[float, float]] = []
            for vertex in region.findall('.//Vertices/Vertex'):
                try:
                    x = float(vertex.attrib['X'])
                    y = float(vertex.attrib['Y'])
                    coords.append((x, y))
                except Exception:
                    continue
            if coords:
                regions.append({'label': label, 'coords': coords})
    # Fallback for generic schema
    if not regions:
        for region in root.findall('.//Region'):
            label = region.attrib.get('Text') or ''
            coords = []
            for vertex in region.findall('.//Vertex'):
                try:
                    x = float(vertex.attrib['X'])
                    y = float(vertex.attrib['Y'])
                    coords.append((x, y))
                except Exception:
                    continue
            if coords:
                regions.append({'label': label, 'coords': coords})
    return regions


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Universal Image Viewer')
        self.image = None  # numpy array for current display
        self.current_channel = 0
        self.file_path = None
        self.file_ext = None
        self.svs_slide = None  # openslide.OpenSlide when viewing SVS
        self.svs_level_used = None  # which openslide level the current display image comes from
        self.last_region = None  # (x,y,w,h) of last extracted region
        self.last_pin = None  # (x,y) pin in level-0 or image coords
        self.pin_item = None  # QGraphics item for pin marker
        # Annotation state
        self.overlay_regions = []  # list of dicts: {'label': str, 'coords': [(x,y), ...]}
        self.overlay_items = []  # QGraphicsPathItem items drawn as overlays
        self._annotation_cache = {}
        # Keep a reference to export threads to prevent GC
        self._active_export_threads = []
        self._active_export_workers = []
        self._init_graphics()
        self.init_ui()

    def _init_graphics(self):
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        # Enable mouse tracking for live position updates
        self.view.setMouseTracking(True)
        try:
            self.view.viewport().setMouseTracking(True)
        except Exception:
            pass
        self.pixmap_item = None
        # Connect mouse-drag region selection and pin click
        self.view.regionSelected.connect(self._on_region_selected)
        self.view.pinClicked.connect(self._on_pin_clicked)
        self.view.mouseMoved.connect(self._on_mouse_moved)

    def init_ui(self):
        open_btn = QPushButton('Open Image')
        open_btn.clicked.connect(self.open_image)
        fullres_btn = QPushButton('Full Resolution')
        fullres_btn.setToolTip('For SVS slides, attempt to load level 0 at full resolution (may use a lot of memory).')
        fullres_btn.clicked.connect(self.load_full_resolution)

        self.channel_box = QComboBox()
        self.channel_box.currentIndexChanged.connect(self.change_channel)

        self.region_input = QLineEdit()
        self.region_input.setPlaceholderText('x,y,width,height (up to 8 digits each)')

        self.pin_label = QLabel('Pin: -')
        self.pos_label = QLabel('Pos: -')
        # Filename displayer
        self.file_label = QLabel('File: -')
        self.file_label.setToolTip('')

        region_btn = QPushButton('Extract Region')
        region_btn.clicked.connect(self.extract_region)
        save_btn = QPushButton('Save Region')
        save_btn.clicked.connect(self.save_region)
        export_btn = QPushButton('Export CSV')
        export_btn.clicked.connect(self.export_region_csv)
        # Annotation controls
        load_xml_btn = QPushButton('Load Annotations')
        load_xml_btn.setToolTip('Load an XML annotation file to overlay regions (SVS only).')
        load_xml_btn.clicked.connect(self.load_annotations)
        self.show_annotations_cb = QCheckBox('Show Annotations')
        self.show_annotations_cb.setChecked(True)
        self.show_annotations_cb.stateChanged.connect(lambda _: self._draw_overlays())

        top_layout = QHBoxLayout()
        top_layout.addWidget(open_btn)
        top_layout.addWidget(fullres_btn)
        top_layout.addWidget(self.file_label)
        top_layout.addWidget(self.channel_box)
        top_layout.addWidget(self.pos_label)
        top_layout.addWidget(self.pin_label)
        top_layout.addWidget(self.region_input)
        top_layout.addWidget(region_btn)
        top_layout.addWidget(save_btn)
        top_layout.addWidget(export_btn)
        top_layout.addWidget(load_xml_btn)
        top_layout.addWidget(self.show_annotations_cb)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.view)
        # Show placeholder text in scene initially
        self.scene.addText('No image loaded')

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def resizeEvent(self, event):
        # Re-render when size changes to keep view updated
        super().resizeEvent(event)
        self.display_image(region=self.last_region if self.last_region is not None else None)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.svs *.dcm *.png *.jpg *.jpeg)')
        if not file_path:
            return
        self.open_from_path(file_path)

    def open_from_path(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            QMessageBox.warning(self, 'Unsupported', f'File type {ext} not supported.')
            return
        self.file_path = file_path
        self.file_ext = ext
        self.svs_slide = None
        # Reset annotations when opening a new file
        self.overlay_regions = []
        self._clear_overlay_items()
        try:
            if ext == '.svs':
                if not openslide:
                    raise RuntimeError('openslide-python is not installed. SVS cannot be opened.')
                slide = openslide.OpenSlide(file_path)
                self.svs_slide = slide
                # Use the smallest level for overview to avoid huge memory use
                level = slide.level_count - 1
                level_dim = slide.level_dimensions[level]
                img = slide.read_region((0, 0), level, level_dim).convert('RGB')
                self.image = np.array(img)
                self.svs_level_used = level
            elif ext == '.dcm':
                if not pydicom:
                    raise RuntimeError('pydicom is not installed. DICOM cannot be opened.')
                ds = pydicom.dcmread(file_path)
                arr = ds.pixel_array
                # Handle 16-bit grayscale or multichannel
                if arr.ndim == 2:
                    arr8 = _to_uint8(arr)
                    arr = np.stack([arr8] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    # If 16-bit per channel, normalize to 8-bit
                    if arr.dtype != np.uint8:
                        arr = _to_uint8(arr)
                else:
                    # Unknown shape, try to squeeze to 2D then stack
                    arr = np.squeeze(arr)
                    if arr.ndim == 2:
                        arr8 = _to_uint8(arr)
                        arr = np.stack([arr8] * 3, axis=-1)
                    else:
                        raise RuntimeError(f'Unsupported DICOM pixel array shape: {arr.shape}')
                self.image = arr
                self.svs_level_used = None
            else:
                img = Image.open(file_path).convert('RGB')
                self.image = np.array(img)
                self.svs_level_used = None
        except Exception as e:
            QMessageBox.critical(self, 'Open Image Failed', str(e))
            self.image = None
            self.file_path = None
            self.file_ext = None
            self.svs_slide = None
            # Reset filename label on failure
            try:
                self.file_label.setText('File: -')
                self.file_label.setToolTip('')
            except Exception:
                pass
            return
        # Update filename label on success
        try:
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            self.file_label.setToolTip(file_path)
        except Exception:
            pass
        self.update_channel_box()
        self.display_image()

    def _on_region_selected(self, rectf: QRectF):
        # Convert a selection rect in scene coords to image pixel coords, handling SVS scaling
        if self.pixmap_item is None or self.image is None:
            return
        # Map scene rect to pixmap/item coordinates
        p1 = self.pixmap_item.mapFromScene(rectf.topLeft())
        p2 = self.pixmap_item.mapFromScene(rectf.bottomRight())
        x0 = int(np.floor(min(p1.x(), p2.x())))
        y0 = int(np.floor(min(p1.y(), p2.y())))
        x1 = int(np.ceil(max(p1.x(), p2.x())))
        y1 = int(np.ceil(max(p1.y(), p2.y())))
        # Clip to current image bounds
        H, W = self.image.shape[0], self.image.shape[1]
        x0 = max(0, min(x0, W))
        y0 = max(0, min(y0, H))
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        if x1 <= x0 or y1 <= y0:
            return
        sel_w = x1 - x0
        sel_h = y1 - y0
        # Scale to level-0 coordinates for SVS if needed
        if self.file_ext == '.svs' and self.svs_slide is not None and self.svs_level_used is not None:
            down = float(self.svs_slide.level_downsamples[self.svs_level_used])
            X = int(round(x0 * down))
            Y = int(round(y0 * down))
            W0 = int(round(sel_w * down))
            H0 = int(round(sel_h * down))
        else:
            X, Y, W0, H0 = x0, y0, sel_w, sel_h
        # Enforce 8-digit limit
        if any(len(str(abs(v))) > 8 for v in (X, Y, W0, H0)):
            QMessageBox.warning(self, 'Region Too Large', 'Selected region exceeds 8-digit coordinate limits.')
            return
        self.last_region = (X, Y, W0, H0)
        # Update the text field and display region
        self.region_input.setText(f"{X},{Y},{W0},{H0}")
        self.display_image(region=self.last_region)

    def _on_pin_clicked(self, scene_pt: QPointF):
        # Map scene point to pixmap/item coordinates, then scale to level-0 if needed
        if self.pixmap_item is None or self.image is None:
            return
        p_item = self.pixmap_item.mapFromScene(scene_pt)
        x = int(round(p_item.x()))
        y = int(round(p_item.y()))
        H, W = self.image.shape[0], self.image.shape[1]
        if x < 0 or y < 0 or x >= W or y >= H:
            return
        if self.file_ext == '.svs' and self.svs_slide is not None and self.svs_level_used is not None:
            down = float(self.svs_slide.level_downsamples[self.svs_level_used])
            X = int(round(x * down))
            Y = int(round(y * down))
        else:
            X, Y = x, y
        self.last_pin = (X, Y)
        self.pin_label.setText(f"Pin: {X},{Y}")
        # Draw marker as a small red circle on the pixmap item
        if self.pin_item is not None:
            try:
                self.scene.removeItem(self.pin_item)
            except Exception:
                pass
            self.pin_item = None
        r = 6
        # Create in scene coords positioned at scene_pt for stability
        pen = QPen(Qt.red)
        pen.setWidth(2)
        self.pin_item = self.scene.addEllipse(scene_pt.x() - r, scene_pt.y() - r, 2 * r, 2 * r, pen)

    def _on_mouse_moved(self, scene_pt: QPointF):
        # Update live mouse position label in image or level-0 coordinates
        if self.pixmap_item is None or self.image is None:
            self.pos_label.setText('Pos: -')
            return
        p_item = self.pixmap_item.mapFromScene(scene_pt)
        x = int(round(p_item.x()))
        y = int(round(p_item.y()))
        H, W = self.image.shape[0], self.image.shape[1]
        if x < 0 or y < 0 or x >= W or y >= H:
            self.pos_label.setText('Pos: -')
            return
        if self.file_ext == '.svs' and self.svs_slide is not None and self.svs_level_used is not None:
            down = float(self.svs_slide.level_downsamples[self.svs_level_used])
            X = int(round(x * down))
            Y = int(round(y * down))
        else:
            X, Y = x, y
        self.pos_label.setText(f'Pos: {X},{Y}')

    def update_channel_box(self):
        if self.image is None:
            self.channel_box.clear()
            return
        channels = self.image.shape[2] if self.image.ndim == 3 else 1
        self.channel_box.clear()
        if channels == 1:
            self.channel_box.addItem('Gray')
        elif channels >= 3:
            # Provide an 'All' option to view all channels (RGB) together
            self.channel_box.addItem('All')
            self.channel_box.addItem('R')
            self.channel_box.addItem('G')
            self.channel_box.addItem('B')
            if channels > 3:
                for i in range(3, channels):
                    self.channel_box.addItem(f'Ch{i+1}')
        else:
            for i in range(channels):
                self.channel_box.addItem(f'Ch{i+1}')
        self.current_channel = 0

    def change_channel(self, idx):
        self.current_channel = idx
        self.display_image()

    def display_image(self, region=None):
        if self.image is None:
            self.scene.clear()
            self.pixmap_item = None
            self.scene.addText('No image loaded')
            return
        # Clear pin overlay when switching regions to avoid confusion
        if region is not None and self.pin_item is not None:
            try:
                self.scene.removeItem(self.pin_item)
            except Exception:
                pass
            self.pin_item = None
        # Always clear annotation overlay items before redrawing
        self._clear_overlay_items()
        img = self.image
        # If region is requested and we have an SVS slide, read from slide to avoid memory issues
        if region is not None:
            x, y, w, h = region
            if self.file_ext == '.svs' and self.svs_slide is not None:
                try:
                    tile = self.svs_slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert('RGB')
                    img = np.array(tile)
                except Exception as e:
                    QMessageBox.warning(self, 'Region Error', f'Could not extract region from SVS: {e}')
                    img = self.image
            else:
                # For non-SVS, just slice the loaded image with clipping
                H, W = img.shape[0], img.shape[1]
                x0 = max(0, int(x)); y0 = max(0, int(y))
                x1 = min(W, x0 + int(w)); y1 = min(H, y0 + int(h))
                if x0 >= x1 or y0 >= y1:
                    QMessageBox.warning(self, 'Region Error', 'Region is out of bounds or empty.')
                else:
                    img = img[y0:y1, x0:x1]
        # Channel selection: support 'All' (RGB) or single-channel view
        if img.ndim == 3 and img.shape[2] >= 1:
            label = self.channel_box.currentText() if self.channel_box.count() else ''
            if label == 'All':
                # Keep as RGB if >=3 channels; clamp to first 3 if more
                if img.shape[2] >= 3:
                    img = img[..., :3]
                else:
                    # For 2-channel uncommon case, fall back to first channel replicated
                    ch0 = img[..., 0]
                    img = np.stack([ch0] * 3, axis=-1)
            elif label in ('R', 'G', 'B') or label.startswith('Ch'):
                if label in ('R', 'G', 'B'):
                    ch = {'R': 0, 'G': 1, 'B': 2}[label]
                else:
                    try:
                        ch = int(label[2:]) - 1
                    except Exception:
                        ch = 0
                ch = max(0, min(ch, img.shape[2] - 1))
                img = img[..., ch]
                img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        # Ensure uint8 RGB
        if img.dtype != np.uint8:
            img = _to_uint8(img)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
        # Ensure contiguous memory for QImage
        img = np.ascontiguousarray(img)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if self.pixmap_item is None:
            self.scene.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.view.reset_view()
        else:
            self.pixmap_item.setPixmap(pixmap)
        # Draw annotations if any
        self._draw_overlays(region)

    def extract_region(self):
        text = self.region_input.text()
        try:
            x, y, w, h = map(lambda s: int(s.strip()), text.split(','))
            # 8-digit limit per requirement
            if any(len(str(abs(val))) > 8 for val in (x, y, w, h)):
                raise ValueError('Coordinates must be up to 8 digits.')
            if w <= 0 or h <= 0:
                raise ValueError('Width and height must be positive.')
            self.last_region = (x, y, w, h)
            self.display_image(region=self.last_region)
        except Exception as e:
            QMessageBox.warning(self, 'Invalid Input', f'Error: {e}')

    def save_region(self):
        if self.last_region is None:
            QMessageBox.information(self, 'Save Region', 'No region extracted yet. Use "Extract Region" first.')
            return
        x, y, w, h = self.last_region
        # Prompt export settings
        dlg = ExportSettingsDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        settings = dlg.get_settings()
        # Kick off threaded export
        worker = ExportWorker(
            file_path=self.file_path,
            file_ext=self.file_ext,
            image=np.copy(self.image) if self.image is not None else None,
            region=(x, y, w, h),
            channel_label=self.channel_box.currentText() if self.channel_box.count() else '',
            overlay_regions=self.overlay_regions if settings['include_overlay'] else [],
            style=settings,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Connect to ImageViewer slots to ensure UI-thread operations
        worker.error.connect(self._on_export_error)
        worker.saved.connect(self._on_export_saved)
        worker.readyImage.connect(self._on_export_ready_image)
        worker.finished.connect(lambda: None)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._active_export_threads.append(thread)
        self._active_export_workers.append(worker)
        thread.start()

    def _cleanup_export_thread(self, worker_obj: QObject):
        try:
            thread = worker_obj.thread()
            if thread is not None:
                thread.quit(); thread.wait()
                try:
                    self._active_export_threads.remove(thread)
                except Exception:
                    pass
            try:
                self._active_export_workers.remove(worker_obj)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _prepare_image_for_save(arr: np.ndarray) -> np.ndarray:
        # Ensure uint8, 3-channel, contiguous
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.stack([arr[..., 0]] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        return np.ascontiguousarray(arr)

    def _on_export_error(self, msg: str):
        # Runs on UI thread; sender() is the worker
        try:
            QMessageBox.critical(self, 'Export Failed', msg)
        finally:
            worker = self.sender()
            if isinstance(worker, QObject):
                self._cleanup_export_thread(worker)

    def _on_export_saved(self, path: str):
        try:
            QMessageBox.information(self, 'Export', f'Exported to {path}')
        finally:
            worker = self.sender()
            if isinstance(worker, QObject):
                self._cleanup_export_thread(worker)

    def _on_export_ready_image(self, img_obj):
        # Ask for save path on UI thread
        worker = self.sender()
        try:
            settings = getattr(worker, 'style', {}) if worker else {}
            region = getattr(worker, 'region', (0, 0, 0, 0)) if worker else (0, 0, 0, 0)
            x, y, w, h = region
            default_name = f"region_{x}_{y}_{w}_{h}.{(settings.get('format') or 'png').lower()}"
            save_path, _ = QFileDialog.getSaveFileName(self, 'Save Region As', default_name, 'PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff)')
            if save_path:
                try:
                    arr = img_obj if isinstance(img_obj, np.ndarray) else np.array(img_obj)
                    arr = self._prepare_image_for_save(arr)
                    fmt = os.path.splitext(save_path)[1].lower()
                    pil = Image.fromarray(arr)
                    if fmt in ('.jpg', '.jpeg'):
                        pil.save(save_path, quality=int(settings.get('quality', 90)))
                    else:
                        pil.save(save_path)
                    QMessageBox.information(self, 'Export', f'Exported to {save_path}')
                except Exception as e:
                    shape = getattr(arr, 'shape', None)
                    dtype = getattr(arr, 'dtype', None)
                    QMessageBox.critical(self, 'Export Failed', f'{e}\nShape: {shape} dtype: {dtype}\nPath: {save_path}')
        finally:
            if isinstance(worker, QObject):
                self._cleanup_export_thread(worker)

    def load_full_resolution(self):
        # Attempt to load the entire image at full resolution. For SVS, this may require large memory.
        if not self.file_path or not self.file_ext:
            QMessageBox.information(self, 'Full Resolution', 'No image loaded.')
            return
        # Non-SVS images are already fully loaded
        if self.file_ext != '.svs' or self.svs_slide is None:
            QMessageBox.information(self, 'Full Resolution', 'This image is already at full resolution.')
            return
        try:
            w, h = self.svs_slide.level_dimensions[0]
            mp = (w * h) / 1e6
            est_mb = (w * h * 3) / (1024 * 1024)
            reply = QMessageBox.question(
                self,
                'Load Full Resolution',
                f'This slide is {w} x {h} (~{mp:.1f} MP).\nEstimated memory to display: ~{est_mb:.0f} MB.\nProceed to load full resolution?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            img = self.svs_slide.read_region((0, 0), 0, (int(w), int(h))).convert('RGB')
            self.image = np.array(img)
            self.svs_level_used = 0
            self.last_region = None
            self.update_channel_box()
            self.display_image()
        except Exception as e:
            QMessageBox.critical(self, 'Full Resolution Failed', f'Could not load full resolution: {e}')

    def export_region_csv(self):
        if self.last_region is None:
            QMessageBox.information(self, 'Export CSV', 'No region extracted yet. Use "Extract Region" first.')
            return
        save_path, _ = QFileDialog.getSaveFileName(self, 'Export Region CSV', 'regions.csv', 'CSV Files (*.csv)')
        if not save_path:
            return
        x, y, w, h = self.last_region
        line = f'"{self.file_path or ""}",{x},{y},{w},{h},{self.channel_box.currentText() or ""}\r\n'
        try:
            with open(save_path, 'a', encoding='utf-8', newline='') as f:
                f.write(line)
            QMessageBox.information(self, 'Export CSV', f'Appended region to {save_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Export Failed', f'Could not write CSV: {e}')

    def load_annotations(self):
        # Load an XML annotation file and parse regions
        xml_path, _ = QFileDialog.getOpenFileName(self, 'Open Annotation XML', 'Annotations', 'XML Files (*.xml)')
        if not xml_path:
            return
        if self.file_ext != '.svs' or self.svs_slide is None:
            QMessageBox.information(self, 'Annotations', 'Annotation overlay is currently supported for SVS slides only.')
        try:
            # Cache parsed annotations by file path
            if xml_path in self._annotation_cache:
                regions = self._annotation_cache[xml_path]
            else:
                regions = _parse_qupath_xml(xml_path)
                self._annotation_cache[xml_path] = regions
            if not regions:
                QMessageBox.information(self, 'Annotations', 'No regions found in the XML file.')
            self.overlay_regions = regions
            # Redraw overlays for current view
            self._draw_overlays()
        except Exception as e:
            QMessageBox.critical(self, 'Annotations', f'Failed to load annotations: {e}')

    # Rendering overlays onto an image buffer using current style
    def _render_overlay_on_region_image(self, region_img: np.ndarray, region_rect, style: Dict) -> np.ndarray:
        if region_img.ndim == 2:
            region_img = np.stack([region_img] * 3, axis=-1)
        if region_img.dtype != np.uint8:
            region_img = _to_uint8(region_img)
        x, y, w, h = map(float, region_rect)
        clip_rect = (x, y, w, h)
        img = np.ascontiguousarray(region_img)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        painter = None
        try:
            from PyQt5.QtGui import QPainter
            painter = QPainter(qimg)
            painter.setRenderHint(QPainter.Antialiasing, True)
            outline = QColor(style.get('outline_color', '#FFD700'))
            fill = QColor(style.get('fill_color', '#FFFF00'))
            fill.setAlpha(int(style.get('fill_alpha', 40)))
            pen = QPen(outline)
            pen.setWidth(int(style.get('stroke_width', 2)))
            brush = QBrush(fill)
            draw_labels = bool(style.get('draw_labels', False))
            label_size = int(style.get('label_font_size', 10))
            for reg in self.overlay_regions:
                pts = reg.get('coords', [])
                if len(pts) < 2:
                    continue
                clipped = _clip_polygon_to_rect_cached(tuple((float(a), float(b)) for (a, b) in pts), tuple(clip_rect))
                if not clipped:
                    continue
                path = QPainterPath()
                first = True
                cx = 0.0; cy = 0.0; n = 0
                for (X, Y) in clipped:
                    px = float(X) - x
                    py = float(Y) - y
                    cx += px; cy += py; n += 1
                    if first:
                        path.moveTo(px, py)
                        first = False
                    else:
                        path.lineTo(px, py)
                path.closeSubpath()
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPath(path)
                if draw_labels and n > 0:
                    try:
                        painter.setPen(QPen(QColor('#000000')))
                        painter.setBrush(Qt.NoBrush)
                        painter.setFont(painter.font())
                        f = painter.font()
                        f.setPointSize(label_size)
                        painter.setFont(f)
                        cx /= n; cy /= n
                        painter.drawText(QPointF(cx, cy), str(reg.get('label', '')))
                    except Exception:
                        pass
        finally:
            if painter is not None:
                painter.end()
        return np.array(img)

    def _clear_overlay_items(self):
        # Remove any existing overlay graphics from the scene
        if self.overlay_items:
            for item in self.overlay_items:
                try:
                    if item.scene() is self.scene:
                        self.scene.removeItem(item)
                    else:
                        # If child of pixmap_item, deleting reference is enough
                        item.setParentItem(None)
                        try:
                            self.scene.removeItem(item)
                        except Exception:
                            pass
                except Exception:
                    pass
            self.overlay_items = []

    def _draw_overlays(self, region=None):
        # Draw annotation overlays scaled to current view
        self._clear_overlay_items()
        if (
            not self.show_annotations_cb.isChecked()
            or not self.overlay_regions
            or self.pixmap_item is None
        ):
            return
        try:
            # Determine mapping from level-0 coords to pixmap coordinates
            origin_x = 0.0
            origin_y = 0.0
            scale = 1.0
            clip_rect = None  # Optional clip in level-0 coords: (x, y, w, h)
            if self.file_ext == '.svs' and self.svs_slide is not None:
                if region is not None:
                    # Current pixmap shows a level-0 tile at region (x,y,w,h)
                    origin_x, origin_y, rw, rh = region
                    scale = 1.0
                    clip_rect = (float(origin_x), float(origin_y), float(rw), float(rh))
                else:
                    # Overview or full-resolution whole slide
                    if self.svs_level_used is not None:
                        down = float(self.svs_slide.level_downsamples[self.svs_level_used])
                        origin_x = 0.0
                        origin_y = 0.0
                        scale = 1.0 / down
                    else:
                        # Fallback assume level-0
                        scale = 1.0
            else:
                # Non-SVS overlays are not supported (coordinates likely in level-0 WSI space)
                return

            pen = QPen(QColor(255, 215, 0))  # golden outline
            pen.setWidth(2)
            brush = QBrush(QColor(255, 255, 0, 40))  # translucent fill

            for reg in self.overlay_regions:
                pts = reg.get('coords', [])
                if len(pts) < 2:
                    continue
                # Optionally clip to region rect in level-0 coordinates
                if clip_rect is not None:
                    pts_to_draw = _clip_polygon_to_rect_cached(tuple((float(a), float(b)) for (a, b) in pts), tuple(clip_rect))
                    if not pts_to_draw:
                        continue
                else:
                    pts_to_draw = pts
                path = QPainterPath()
                first = True
                for (X, Y) in pts_to_draw:
                    px = (float(X) - origin_x) * scale
                    py = (float(Y) - origin_y) * scale
                    if first:
                        path.moveTo(px, py)
                        first = False
                    else:
                        path.lineTo(px, py)
                path.closeSubpath()
                item = QGraphicsPathItem(path, parent=self.pixmap_item)
                item.setPen(pen)
                item.setBrush(brush)
                self.overlay_items.append(item)
        except Exception:
            # Fail silently for overlay drawing; don't block primary viewing
            pass
class ExportSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Settings')
        v = QVBoxLayout(self)
        # Basics
        self.include_overlay_cb = QCheckBox('Include annotations overlay')
        self.include_overlay_cb.setChecked(True)
        v.addWidget(self.include_overlay_cb)
        form = QFormLayout()
        # Output format and quality
        self.format_box = QComboBox()
        self.format_box.addItems(['PNG', 'JPEG', 'TIFF'])
        form.addRow('Format', self.format_box)
        self.quality_sb = QSpinBox()
        self.quality_sb.setRange(1, 100)
        self.quality_sb.setValue(90)
        form.addRow('JPEG quality', self.quality_sb)
        # Scale
        self.scale_sb = QDoubleSpinBox()
        self.scale_sb.setRange(0.1, 10.0)
        self.scale_sb.setSingleStep(0.1)
        self.scale_sb.setValue(1.0)
        form.addRow('Scale factor', self.scale_sb)
        # Overlay style
        self.outline_color_le = QLineEdit('#FFD700')
        form.addRow('Outline color (#RRGGBB)', self.outline_color_le)
        self.fill_color_le = QLineEdit('#FFFF00')
        form.addRow('Fill color (#RRGGBB)', self.fill_color_le)
        self.fill_alpha_sb = QSpinBox()
        self.fill_alpha_sb.setRange(0, 255)
        self.fill_alpha_sb.setValue(40)
        form.addRow('Fill alpha (0-255)', self.fill_alpha_sb)
        self.stroke_width_sb = QSpinBox()
        self.stroke_width_sb.setRange(1, 12)
        self.stroke_width_sb.setValue(2)
        form.addRow('Stroke width (px)', self.stroke_width_sb)
        self.draw_labels_cb = QCheckBox('Draw labels')
        self.draw_labels_cb.setChecked(False)
        form.addRow(self.draw_labels_cb)
        self.label_font_size_sb = QSpinBox()
        self.label_font_size_sb.setRange(6, 48)
        self.label_font_size_sb.setValue(10)
        form.addRow('Label font size', self.label_font_size_sb)
        v.addLayout(form)
        # Auto-save
        self.autosave_cb = QCheckBox('Auto-save without asking')
        self.autosave_cb.setChecked(False)
        v.addWidget(self.autosave_cb)
        autosave_form = QFormLayout()
        self.dest_dir_le = QLineEdit('')
        browse_btn = QPushButton('Browse...')
        def browse():
            d = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
            if d:
                self.dest_dir_le.setText(d)
        browse_btn.clicked.connect(browse)
        h = QHBoxLayout(); h.addWidget(self.dest_dir_le); h.addWidget(browse_btn)
        autosave_form.addRow('Destination folder', h)
        self.filename_pattern_le = QLineEdit('region_{x}_{y}_{w}_{h}')
        autosave_form.addRow('Filename pattern', self.filename_pattern_le)
        v.addLayout(autosave_form)
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def get_settings(self) -> Dict:
        return {
            'include_overlay': self.include_overlay_cb.isChecked(),
            'format': self.format_box.currentText(),
            'quality': int(self.quality_sb.value()),
            'scale': float(self.scale_sb.value()),
            'outline_color': self.outline_color_le.text().strip() or '#FFD700',
            'fill_color': self.fill_color_le.text().strip() or '#FFFF00',
            'fill_alpha': int(self.fill_alpha_sb.value()),
            'stroke_width': int(self.stroke_width_sb.value()),
            'draw_labels': self.draw_labels_cb.isChecked(),
            'label_font_size': int(self.label_font_size_sb.value()),
            'autosave': self.autosave_cb.isChecked(),
            'dest_dir': self.dest_dir_le.text().strip(),
            'filename_pattern': self.filename_pattern_le.text().strip() or 'region_{x}_{y}_{w}_{h}',
        }


class ExportWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    saved = pyqtSignal(str)
    readyImage = pyqtSignal(object)  # emits numpy array

    def __init__(self, file_path: Optional[str], file_ext: Optional[str], image: Optional[np.ndarray], region: Tuple[int, int, int, int], channel_label: str, overlay_regions: List[Dict], style: Dict):
        super().__init__()
        self.file_path = file_path
        self.file_ext = (file_ext or '').lower()
        self.image = image
        self.region = region
        self.channel_label = channel_label
        self.overlay_regions = overlay_regions
        self.style = style

    def _apply_channel(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[2] > 1:
            label = self.channel_label or ''
            if label == 'All':
                arr = arr[..., :3]
            else:
                if label in ('R', 'G', 'B'):
                    ch = {'R': 0, 'G': 1, 'B': 2}[label]
                elif label.startswith('Ch'):
                    try:
                        ch = int(label[2:]) - 1
                    except Exception:
                        ch = 0
                else:
                    ch = 0
                ch = max(0, min(ch, arr.shape[2] - 1))
                arr = arr[..., ch]
                arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        return arr

    def _render_overlay(self, arr: np.ndarray) -> np.ndarray:
        if not self.overlay_regions:
            return arr
        # Local renderer mirroring the viewer implementation, using cached clipping
        x, y, w, h = map(float, self.region)
        clip_rect = (x, y, w, h)
        img = np.ascontiguousarray(arr if arr.dtype == np.uint8 else _to_uint8(arr))
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        painter = None
        try:
            from PyQt5.QtGui import QPainter
            painter = QPainter(qimg)
            painter.setRenderHint(QPainter.Antialiasing, True)
            outline = QColor(self.style.get('outline_color', '#FFD700'))
            fill = QColor(self.style.get('fill_color', '#FFFF00'))
            fill.setAlpha(int(self.style.get('fill_alpha', 40)))
            pen = QPen(outline)
            pen.setWidth(int(self.style.get('stroke_width', 2)))
            brush = QBrush(fill)
            draw_labels = bool(self.style.get('draw_labels', False))
            label_size = int(self.style.get('label_font_size', 10))
            for reg in self.overlay_regions:
                pts = reg.get('coords', [])
                if len(pts) < 2:
                    continue
                clipped = _clip_polygon_to_rect_cached(tuple((float(a), float(b)) for (a, b) in pts), tuple(clip_rect))
                if not clipped:
                    continue
                path = QPainterPath()
                first = True
                cx = 0.0; cy = 0.0; n = 0
                for (X, Y) in clipped:
                    px = float(X) - x
                    py = float(Y) - y
                    cx += px; cy += py; n += 1
                    if first:
                        path.moveTo(px, py)
                        first = False
                    else:
                        path.lineTo(px, py)
                path.closeSubpath()
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPath(path)
                if draw_labels and n > 0:
                    try:
                        painter.setPen(QPen(QColor('#000000')))
                        painter.setBrush(Qt.NoBrush)
                        f = painter.font(); f.setPointSize(label_size)
                        painter.setFont(f)
                        cx /= n; cy /= n
                        painter.drawText(QPointF(cx, cy), str(reg.get('label', '')))
                    except Exception:
                        pass
        finally:
            if painter is not None:
                painter.end()
        return np.array(img)

    def run(self):
        try:
            x, y, w, h = self.region
            # Read region
            if self.file_ext == '.svs' and self.file_path:
                if openslide is None:
                    raise RuntimeError('openslide-python not available')
                slide = openslide.OpenSlide(self.file_path)
                tile = slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert('RGB')
                arr = np.array(tile)
            else:
                if self.image is None:
                    raise RuntimeError('No image loaded')
                H, W = self.image.shape[0], self.image.shape[1]
                x0 = max(0, int(x)); y0 = max(0, int(y))
                x1 = min(W, x0 + int(w)); y1 = min(H, y0 + int(h))
                if x0 >= x1 or y0 >= y1:
                    raise RuntimeError('Region is out of bounds or empty')
                arr = self.image[y0:y1, x0:x1]
            # Channel selection
            arr = self._apply_channel(arr)
            # Overlay
            if self.overlay_regions:
                arr = self._render_overlay(arr)
            # Scale
            scale = float(self.style.get('scale', 1.0))
            if abs(scale - 1.0) > 1e-6:
                pil = Image.fromarray(arr)
                new_size = (max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale))))
                pil = pil.resize(new_size, Image.BILINEAR)
                arr = np.array(pil)
            # Save or emit
            if self.style.get('autosave'):
                dest_dir = self.style.get('dest_dir') or ''
                if not dest_dir:
                    raise RuntimeError('Destination folder is empty')
                os.makedirs(dest_dir, exist_ok=True)
                fmt = (self.style.get('format') or 'PNG').upper()
                pattern = self.style.get('filename_pattern') or 'region_{x}_{y}_{w}_{h}'
                name = pattern.format(x=x, y=y, w=w, h=h)
                ext = '.png' if fmt == 'PNG' else ('.jpg' if fmt == 'JPEG' else '.tif')
                out_path = os.path.join(dest_dir, name + ext)
                pil = Image.fromarray(arr)
                if fmt == 'JPEG':
                    pil.save(out_path, quality=int(self.style.get('quality', 90)))
                else:
                    pil.save(out_path)
                self.saved.emit(out_path)
            else:
                self.readyImage.emit(arr)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class ZoomableGraphicsView(QGraphicsView):
    # Emits a QRectF in scene coordinates when user Shift+drags a region
    regionSelected = pyqtSignal(QRectF)
    # Emits a QPointF in scene coordinates when user Ctrl+Clicks to drop a pin
    pinClicked = pyqtSignal(QPointF)
    # Emits a QPointF in scene coordinates on mouse move
    mouseMoved = pyqtSignal(QPointF)
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 0
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = None

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() == 0:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom += 1 if factor > 1 else -1
        if self._zoom < -20:
            self._zoom = -20
            return
        if self._zoom > 50:
            self._zoom = 50
            return
        self.scale(factor, factor)

    def reset_view(self):
        self._zoom = 0
        self.resetTransform()
        # Fit to view if there is content
        if self.scene() and not self.scene().items() == []:
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
            self._origin = event.pos()
            self._rubber_band.setGeometry(QRect(self._origin, QSize()))
            self._rubber_band.show()
            event.accept()
            return
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
            pt_scene = self.mapToScene(event.pos())
            self.pinClicked.emit(pt_scene)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Always emit the current scene position
        try:
            self.mouseMoved.emit(self.mapToScene(event.pos()))
        except Exception:
            pass
        if self._rubber_band.isVisible() and self._origin is not None:
            rect = QRect(self._origin, event.pos()).normalized()
            self._rubber_band.setGeometry(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber_band.isVisible() and self._origin is not None:
            self._rubber_band.hide()
            rect = QRect(self._origin, event.pos()).normalized()
            self._origin = None
            if rect.width() > 3 and rect.height() > 3:
                tl = self.mapToScene(rect.topLeft())
                br = self.mapToScene(rect.bottomRight())
                self.regionSelected.emit(QRectF(tl, br))
            event.accept()
            return
        super().mouseReleaseEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    # If a path is provided as CLI arg, open it
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        viewer.open_from_path(sys.argv[1])
    viewer.show()
    sys.exit(app.exec_())
