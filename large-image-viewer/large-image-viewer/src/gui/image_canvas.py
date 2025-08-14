from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Image Viewer")
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)
        self.image_item = None

    def load_image(self, file_path):
        image = QImage(file_path)
        if image.isNull():
            raise ValueError("Failed to load image.")
        self.display_image(image)

    def display_image(self, image):
        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.view.setSceneRect(self.image_item.boundingRect())
        self.view.fitInView(self.image_item, Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(0.8, 0.8)

    def pan(self, dx, dy):
        self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() + dx)
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() + dy)

    def clear_image(self):
        if self.image_item:
            self.scene.removeItem(self.image_item)
            self.image_item = None
            self.view.setSceneRect(0, 0, 0, 0)