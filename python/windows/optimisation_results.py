
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import MeshData, GLMeshItem, GLScatterPlotItem
from pyqtgraph import ColorBarItem, ColorMap, AxisItem

from stl import mesh
import numpy as np
import os

from python import load, optimisation
from python.windows import developer, optimisation_setup

class Window(QMainWindow):
# class MainWindow (QUiLoader):
# class MainWindow(uiclass, baseclass):

    def __init__(self, component=""):
        super(Window, self).__init__()

        #Load the UI Page
        uic.loadUi('ui/optimisation_results.ui', self)

        model = QFileSystemModel()
        model.setRootPath(QDir.currentPath())

        self.load_results_button.pressed.connect(lambda: self.load_result())
        
        self.action_developer.triggered.connect(lambda: self.open_developer_window())

        self.num_design_slider.valueChanged[int].connect(self.change_num_design)
        # self.num_design_slider.sliderReleased[int].connect(self.change_num_design)

        self.direction_dropdown.addItems(["X","Y","Z","Total"])

        if component != "":
            self.file = f"temp/optimisation/OptimisationOutputs/{component}/LatentVectorsForPlotting.pkl"
            optimisation.load_result(self)

        self.indicator_dropdown.currentTextChanged.connect(lambda: optimisation.load_result(self))
        self.direction_dropdown.currentTextChanged.connect(lambda: optimisation.load_result(self))

        # self.main_view.pan(385, 468, 198)

        # p1 = self.GraphicsLayoutWidget.addPlot(row=0, col=0)
        # p2 = self.GraphicsLayoutWidget.addPlot(row=0, col=1)
        # v = self.GraphicsLayoutWidget.addViewBox(row=1, col=0, colspan=2)

    def load_result (self):
        self.file = QFileDialog.getOpenFileName(self, "Import Optimisation Result File", filter="PKL file (*.pkl)")[0]
        component = self.file.split("/")[-2]
        self.component_label.setText(f"Optimisation results for {component} loaded")
        if component == "bulkhead":
            self.indicator_dropdown.clear()
            self.indicator_dropdown.addItems(["Thinning"])
        if component == "u-bending":
            self.indicator_dropdown.clear()
            self.indicator_dropdown.addItems(["Thinning","Displacement"])
        self.num_design_slider.setValue(100)
        optimisation.load_result(self)

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.developer_window.show()

    def change_num_design (self, num):
        optimisation.load_result(self, num)
        