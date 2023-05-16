
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import MeshData, GLMeshItem, GLScatterPlotItem
from pyqtgraph import ColorBarItem, ColorMap, AxisItem

from stl import mesh
import numpy as np

from python import load, model_control, prediction
from python.windows import developer, optimisation_results, optimisation_setup, sensitivity
from python.optimisation_funcs import surface_points_normals, autodecoder, single_prediction

class Window(QMainWindow):
# class MainWindow (QUiLoader):
# class MainWindow(uiclass, baseclass):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi('ui/main.ui', self)

        model = QFileSystemModel()
        # model.setRootPath(QDir.currentPath())

        self.file = ""

        self.component_dropdown.addItems(load.components())

        self.load_mesh_button.pressed.connect(self.prompt_file)

        self.process_dropdown.setCurrentIndex(-1)

        self.component_dropdown.currentTextChanged.connect(self.update_process_dropdown)
        self.process_dropdown.currentTextChanged.connect(self.update_material_dropdown)
        self.material_dropdown.currentTextChanged.connect(self.update_indicator_dropdown)
        # self.indicator_dropdown.currentTextChanged.connect(self.select_model)
        self.indicator_dropdown.currentTextChanged.connect(self.load_mesh)

        self.direction_dropdown.addItems(["X","Y","Z","Total"])
        # self.direction_dropdown.currentTextChanged.connect(self.change_direction)
        self.direction_dropdown.currentTextChanged.connect(self.load_mesh)

        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_sensitivity.triggered.connect(self.open_sensitivity_window)
        self.action_developer.triggered.connect(self.open_developer_window)

        self.update_process_dropdown()

        # self.main_view.pan(385, 468, 198)

        # p1 = self.GraphicsLayoutWidget.addPlot(row=0, col=0)
        # p2 = self.GraphicsLayoutWidget.addPlot(row=0, col=1)
        # v = self.GraphicsLayoutWidget.addViewBox(row=1, col=0, colspan=2)
    
    def update_process_dropdown (self):
        self.process_dropdown.clear()
        self.process_dropdown.addItems(load.processes(self.component_dropdown.currentText()))

    def update_material_dropdown (self):
        self.material_dropdown.clear()
        self.material_dropdown.addItems(load.materials(self.component_dropdown.currentText(), self.process_dropdown.currentText()))

    def update_indicator_dropdown (self):
        self.indicator_dropdown.clear()
        self.indicator_dropdown.addItems(load.indicators(self.component_dropdown.currentText(), self.process_dropdown.currentText(), self.material_dropdown.currentText()))

    def select_model (self):
        if self.indicator_dropdown.currentText() == "Displacement":
            self.direction_label.setEnabled(True)
            self.direction_dropdown.setEnabled(True)
        else:
            self.direction_dropdown.setEnabled(False)
        self.load_mesh_button.setEnabled(True)
        component = self.component_dropdown.currentText().lower()
        if component == "car door panel":
            prediction.change_model(self)

    def change_direction (self):
        prediction.plot_displacement(self)

    def prompt_file (self):
        self.file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")
        component = self.component_dropdown.currentText().lower()
        if component == "bulkhead" or component == "u-bending":
            self.points, self.normals, self.offsurface_points = surface_points_normals.generate(self.file[0], self)
            self.best_latent_vector = autodecoder.get_latent_vector(self.points, self.normals, self.offsurface_points, self, component)
            self.verts, self.faces = autodecoder.get_verts_faces(self.best_latent_vector, self, component)
        self.load_mesh()

    def load_mesh (self):
        self.select_model()
        file = self.file
        component = self.component_dropdown.currentText().lower()
        indicator = self.indicator_dropdown.currentText().lower()
        if component == "car door panel":
            prediction.load_mesh(file[0], self)
        if component == "bulkhead":
            single_prediction.bulkhead_thinning(self.verts, self.faces, self)
        if component == "u-bending":
            # points, normals, offsurface_points = surface_points_normals.generate(file[0], self)
            # best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, self, component)
            # verts, faces = autodecoder.get_verts_faces(best_latent_vector, self, component)
            if indicator == "thinning":
                single_prediction.ubending_thinning(self.verts, self.faces, self)
            if indicator == "displacement":
                 single_prediction.ubending_displacement(self.verts, self.faces, self)
            
    def new_optimisaiton_window (self):
        self.optimisation_window = optimisation_setup.Window()
        self.close()
        self.optimisation_window.show()   

    def open_optimisaiton_window (self):
        self.optimisation_window = optimisation_results.Window()
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