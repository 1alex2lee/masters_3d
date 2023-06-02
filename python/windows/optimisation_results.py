
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import MeshData, GLMeshItem, GLScatterPlotItem
from pyqtgraph import ColorBarItem, ColorMap, AxisItem

from stl import mesh
import numpy as np
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from python import load, optimisation
from python.windows import developer, optimisation_setup, prediction, sensitivity

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)
        self.axes = fig.subplots(subplot_kw={"projection": "3d"})

        super(MplCanvas, self).__init__(fig)

class Window(QMainWindow):
# class MainWindow (QUiLoader):
# class MainWindow(uiclass, baseclass):

    def __init__(self, component=""):
        super(Window, self).__init__()

        #Load the UI Page
        uic.loadUi('ui/optimisation_results.ui', self)

        model = QFileSystemModel()
        model.setRootPath(QDir.currentPath())

        self.num_designs = 1

        self.load_results_button.pressed.connect(self.prompt_file)
        
        self.action_developer.triggered.connect(self.open_developer_window)

        self.num_design_slider.valueChanged.connect(self.update_text)
        self.num_design_slider.sliderReleased.connect(lambda: optimisation.load_result(self))

        self.direction_dropdown.addItems(["X","Y","Z","Total"])

        self.indicator_dropdown.currentTextChanged.connect(lambda: optimisation.load_result(self))
        self.direction_dropdown.currentTextChanged.connect(lambda: optimisation.load_result(self))

        # self.refresh_button.pressed.connect(lambda: optimisation.load_result(self))

        if component != "":
            print(component)
            self.file = f"temp/optimisation/OptimisationOutputs/{component}/LatentVectorsForPlotting.pkl"
            self.load_result()

        self.indicator_dropdown.currentTextChanged.connect(self.change_indicator)
        # self.direction_dropdown.currentTextChanged.connect(lambda: optimisation.load_result(self))

        # self.main_view.pan(385, 468, 198)

        # p1 = self.GraphicsLayoutWidget.addPlot(row=0, col=0)
        # p2 = self.GraphicsLayoutWidget.addPlot(row=0, col=1)
        # v = self.GraphicsLayoutWidget.addViewBox(row=1, col=0, colspan=2)

        # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.grid.addWidget(self.canvas)

        self.action_prediction.triggered.connect(self.open_prediction_window)
        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_sensitivity.triggered.connect(self.open_sensitivity_window)
        self.action_developer.triggered.connect(self.open_developer_window)

    def prompt_file (self):
        self.file = QFileDialog.getOpenFileName(self, "Import Optimisation Result File", filter="PKL file (*.pkl)")[0]
        if self.file != "":
            self.load_result()

    def load_result (self):
        # component = self.file.split("/")[-2] 
        component = os.path.split(os.path.split(self.file)[0])[1]
        self.component_label.setText(f"Optimised geometry for {component} loaded")
        if "bulkhead" in component.lower():
            self.indicator_dropdown.clear()
            self.indicator_dropdown.addItems(["Thinning (%)"])
        if "u-bending" in component.lower():
            self.indicator_dropdown.clear()
            self.indicator_dropdown.addItems(["Thinning (%)", "Displacement (mm)"])
        # self.num_design_slider.setValue(100)
        # optimisation.load_result(self)

        self.num_design_label.setEnabled(True)
        self.num_design_slider.setEnabled(True)
        self.indicator_label.setEnabled(True)
        self.indicator_dropdown.setEnabled(True)
        self.direction_label.setEnabled(True)
        # self.direction_dropdown.setEnabled(True)
        # self.refresh_button.setEnabled(True)


    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.developer_window.show()

    # def change_num_design (self, num):
    #     optimisation.load_result(self, num)
        
    def change_indicator (self):
        indicator = self.indicator_dropdown.currentText().lower()
        if "displacement" in indicator.lower():
            self.direction_dropdown.setEnabled(True)
        else:
            self.direction_dropdown.setEnabled(False)

    def update_text (self):
        num = self.num_design_slider.value()
        num_designs = self.num_designs
        self.num_design_label.setText(f"Showing design iteration {num} of {num_designs}")
            
    def open_prediction_window (self):
        self.prediction_window = prediction.Window()
        self.close()
        self.prediction_window.show()   

    def new_optimisaiton_window (self):
        self.optimisation_window = optimisation_setup.Window()
        self.close()
        self.optimisation_window.show()   

    def open_sensitivity_window (self):
        self.sensitivity_window = sensitivity.Window()
        self.close()
        self.sensitivity_window.show()

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.close()
        self.developer_window.show()
