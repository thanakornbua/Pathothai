import sys
import os
import warnings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox, QLineEdit, QHBoxLayout, QMessageBox, QGraphicsView, QGraphicsScene, QRubberBand)
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QRect, QPoint, QPointF, QSize
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

        region_btn = QPushButton('Extract Region')
        region_btn.clicked.connect(self.extract_region)
        save_btn = QPushButton('Save Region')
        save_btn.clicked.connect(self.save_region)
        export_btn = QPushButton('Export CSV')
        export_btn.clicked.connect(self.export_region_csv)

        top_layout = QHBoxLayout()
        top_layout.addWidget(open_btn)
        top_layout.addWidget(fullres_btn)
        top_layout.addWidget(self.channel_box)
        top_layout.addWidget(self.pos_label)
        top_layout.addWidget(self.pin_label)
        top_layout.addWidget(self.region_input)
        top_layout.addWidget(region_btn)
        top_layout.addWidget(save_btn)
        top_layout.addWidget(export_btn)

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
                    arr8 = self._to_uint8(arr)
                    arr = np.stack([arr8] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    # If 16-bit per channel, normalize to 8-bit
                    if arr.dtype != np.uint8:
                        arr = self._to_uint8(arr)
                else:
                    # Unknown shape, try to squeeze to 2D then stack
                    arr = np.squeeze(arr)
                    if arr.ndim == 2:
                        arr8 = self._to_uint8(arr)
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
            return
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
            img = self._to_uint8(img)
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
        # Extract region from source depending on file type
        try:
            if self.file_ext == '.svs' and self.svs_slide is not None:
                tile = self.svs_slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert('RGB')
                region_img = np.array(tile)
            else:
                if self.image is None:
                    QMessageBox.warning(self, 'Save Region', 'No image loaded.')
                    return
                img = self.image
                H, W = img.shape[0], img.shape[1]
                x0 = max(0, int(x)); y0 = max(0, int(y))
                x1 = min(W, x0 + int(w)); y1 = min(H, y0 + int(h))
                if x0 >= x1 or y0 >= y1:
                    QMessageBox.warning(self, 'Save Region', 'Region is out of bounds or empty.')
                    return
                region_img = img[y0:y1, x0:x1]
            # Respect channel selection when saving
            if region_img.ndim == 3 and region_img.shape[2] > 1:
                label = self.channel_box.currentText() if self.channel_box.count() else ''
                if label == 'All':
                    # Keep RGB (first 3 channels)
                    region_img = region_img[..., :3]
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
                    ch = max(0, min(ch, region_img.shape[2] - 1))
                    region_img = region_img[..., ch]
                    region_img = np.stack([region_img] * 3, axis=-1)
            elif region_img.ndim == 2:
                region_img = np.stack([region_img] * 3, axis=-1)
            if region_img.dtype != np.uint8:
                region_img = self._to_uint8(region_img)
            save_path, _ = QFileDialog.getSaveFileName(self, 'Save Region As', 'region.png', 'PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)')
            if save_path:
                Image.fromarray(region_img).save(save_path)
        except Exception as e:
            QMessageBox.critical(self, 'Save Failed', f'Could not save region: {e}')

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

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        # Min-max normalize to 0-255 per array for display
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            return arr
        a_min = float(np.min(arr))
        a_max = float(np.max(arr))
        if a_max <= a_min:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - a_min) / (a_max - a_min)
        return (scaled * 255.0).astype(np.uint8)


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
