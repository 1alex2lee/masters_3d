
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import MeshData, GLMeshItem, GLScatterPlotItem
from pyqtgraph import ColorBarItem, ColorMap, AxisItem

from stl import mesh
import numpy as np

from python import load, model_control, prediction
from python.windows import developer

class MainWindow(QMainWindow):
# class MainWindow (QUiLoader):
# class MainWindow(uiclass, baseclass):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi('ui/main.ui', self)

        model = QFileSystemModel()
        model.setRootPath(QDir.currentPath())

        self.process_dropdown.addItems(load.processes())

        self.load_mesh_button.pressed.connect(lambda: self.load_mesh())

        self.process_dropdown.setCurrentIndex(-1)

        self.process_dropdown.currentTextChanged.connect(lambda: self.material_dropdown.addItems(load.materials(self.process_dropdown.currentText())))
        self.material_dropdown.currentTextChanged.connect(lambda: self.model_dropdown.addItems(load.models(self.process_dropdown.currentText(), self.material_dropdown.currentText())))
        self.model_dropdown.currentTextChanged.connect(lambda: self.select_model())

        self.direction_dropdown.addItems(["X","Y","Z","Total"])
        self.direction_dropdown.currentTextChanged.connect(lambda: self.change_direction())

        self.action_developer.triggered.connect(lambda: self.open_developer_window())

        # self.main_view.pan(385, 468, 198)

        # p1 = self.GraphicsLayoutWidget.addPlot(row=0, col=0)
        # p2 = self.GraphicsLayoutWidget.addPlot(row=0, col=1)
        # v = self.GraphicsLayoutWidget.addViewBox(row=1, col=0, colspan=2)

    def select_model (self):
        if self.model_dropdown.currentText() == "Displacement":
            self.direction_dropdown.setEnabled(True)
        else:
            self.direction_dropdown.setEnabled(False)
        self.load_mesh_button.setEnabled(True)
        prediction.change_model(self)


    def change_direction (self):
        prediction.plot_displacement(self)

    def load_mesh (self):
        file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")
        prediction.load_mesh(file[0], self)

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.developer_window.show()