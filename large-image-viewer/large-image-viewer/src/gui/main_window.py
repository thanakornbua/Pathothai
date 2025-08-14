from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QToolBar, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from .image_canvas import ImageCanvas
from .channel_controls import ChannelControls

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_canvas = ImageCanvas()
        self.channel_controls = ChannelControls()

        self.init_ui()

    def init_ui(self):
        self.setCentralWidget(self.image_canvas)

        self.create_menu()
        self.create_toolbar()

        layout = QVBoxLayout()
        layout.addWidget(self.channel_controls)
        layout.addWidget(self.image_canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction(QIcon(None), "Open", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        exit_action = QAction(QIcon(None), "Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def create_toolbar(self):
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        open_action = QAction(QIcon(None), "Open", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_canvas.load_image(file_name)