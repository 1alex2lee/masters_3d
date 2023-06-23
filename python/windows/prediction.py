
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir
from PyQt6.QtGui import QPixmap

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import GLAxisItem
from pyqtgraph import ColorBarItem, ColorMap, AxisItem

from stl import mesh
import numpy as np
import time
import trimesh

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

        # self.component_dropdown.addItems(load.components())
        self.component_dropdown.addItems(["U-bending", "Car Door Panel"])

        self.load_mesh_button.pressed.connect(self.prompt_file)

        self.process_dropdown.setCurrentIndex(-1)

        self.component_dropdown.currentTextChanged.connect(self.update_process_dropdown)
        self.process_dropdown.currentTextChanged.connect(self.update_material_dropdown)
        self.material_dropdown.currentTextChanged.connect(self.update_indicator_dropdown)
        # self.indicator_dropdown.currentTextChanged.connect(self.select_model)
        self.indicator_dropdown.currentTextChanged.connect(self.load_mesh)

        self.slider1.sliderReleased.connect(self.load_mesh)
        self.slider2.sliderReleased.connect(self.load_mesh)
        self.slider3.sliderReleased.connect(self.load_mesh)
        self.slider4.sliderReleased.connect(self.load_mesh)

        self.slider1.valueChanged.connect(self.update_sliders)
        self.slider2.valueChanged.connect(self.update_sliders)
        self.slider3.valueChanged.connect(self.update_sliders)
        self.slider4.valueChanged.connect(self.update_sliders)

        self.direction_dropdown.addItems(["X","Y","Z","Total"])
        # self.direction_dropdown.currentTextChanged.connect(self.change_direction)
        self.direction_dropdown.currentTextChanged.connect(self.load_mesh)

        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_sensitivity.triggered.connect(self.open_sensitivity_window)
        self.action_developer.triggered.connect(self.open_developer_window)

        self.update_process_dropdown()

        # self.main_view.pan(385, 468, 198)
        axis = GLAxisItem()
        self.main_view.addItem(axis)

        # p1 = self.GraphicsLayoutWidget.addPlot(row=0, col=0)
        # p2 = self.GraphicsLayoutWidget.addPlot(row=0, col=1)
        # v = self.GraphicsLayoutWidget.addViewBox(row=1, col=0, colspan=2)
    
    def update_process_dropdown (self):
        self.file = ""
        component = self.component_dropdown.currentText()
        if "car door panel" in component.lower():
            self.label_heading.setPixmap(QPixmap("ui/cardoorpanel_blank_explanation.png"))
            self.label_description.setText("Use the sliders below to adjust the dimensions of the blank shape as denoted in the picture.")

            minL1, maxL1 = 740, 800
            minL2, maxL2 = 680, 740
            midL1 = minL1 + (maxL1-minL1)/2
            midL2 = minL2 + (maxL2-minL2)/2

            self.slider1.setMinimum(minL1)
            self.slider1.setMaximum(maxL1)
            self.slider1.setValue(midL1)
            self.slider2.setMinimum(minL2)
            self.slider2.setMaximum(maxL2)
            self.slider2.setValue(midL2)

            self.label1.setText(f"L1: {self.slider1.value()} mm")
            self.label2.setText(f"L2: {self.slider2.value()} mm")
            self.label3.setText("")
            self.label4.setText("")

            self.label3.setEnabled(False)
            self.label4.setEnabled(False)
            self.slider3.setEnabled(False)
            self.slider4.setEnabled(False)
        if "u-bending" in component.lower():
            self.label_heading.setText("U-bending Processing Parameters")
            self.label_description.setText("The blank shape is held constant for this model.")

            maxBHF = 59
            maxFriction = 0.20
            maxClearance = 1.49
            maxThickness = 2.99

            minBHF = 5.2
            minFriction = 0.1
            minClearance = 1.1
            minThickness = 0.51

            midBHF = minBHF + (maxBHF-minBHF)/2
            midFriction = minFriction + (maxFriction-minFriction)/2
            midClearance = minClearance + (maxClearance-minClearance)/2
            midThickness = minThickness + (maxThickness-minThickness)/2

            self.slider1.setMinimum(minBHF * 100)
            self.slider1.setMaximum(maxBHF * 100)
            self.slider1.setValue(midBHF * 100)
            self.slider2.setMinimum(minFriction * 100)
            self.slider2.setMaximum(maxFriction * 100)
            self.slider2.setValue(midFriction * 100)
            self.slider3.setMinimum(minClearance * 100)
            self.slider3.setMaximum(maxClearance * 100)
            self.slider3.setValue(midClearance * 100)
            self.slider4.setMinimum(minThickness * 100)
            self.slider4.setMaximum(maxThickness * 100)
            self.slider4.setValue(midThickness * 100)

            self.label1.setText(f"Blank Holding Force: {self.slider1.value() / 100} kN")
            self.label2.setText(f"Friction Coefficient: {self.slider2.value() / 100}")
            self.label3.setText(f"Clearance: {self.slider3.value() / 100} %")
            self.label4.setText(f"Thickness: {self.slider4.value() / 100} mm")

            self.label3.setEnabled(True)
            self.label4.setEnabled(True)
            self.slider3.setEnabled(True)
            self.slider4.setEnabled(True)
        self.process_dropdown.clear()
        self.process_dropdown.addItems(load.processes(self.component_dropdown.currentText()))

    def update_material_dropdown (self):
        self.material_dropdown.clear()
        self.material_dropdown.addItems(load.materials(self.component_dropdown.currentText(), self.process_dropdown.currentText()))

    def update_indicator_dropdown (self):
        self.indicator_dropdown.clear()
        self.indicator_dropdown.addItems(load.indicators(self.component_dropdown.currentText(), self.process_dropdown.currentText(), self.material_dropdown.currentText()))

    def select_model (self):
        if "displacement" in self.indicator_dropdown.currentText().lower():
            self.direction_label.setEnabled(True)
            self.direction_dropdown.setEnabled(True)
        else:
            self.direction_dropdown.setEnabled(False)
        self.load_mesh_button.setEnabled(True)
        component = self.component_dropdown.currentText()
        if "car door panel" in component.lower():
            prediction.change_model(self)

    def change_direction (self):
        prediction.plot_displacement(self)

    def prompt_file (self):
        self.file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")[0]
        if self.file != "":
            component = self.component_dropdown.currentText()
            if "bulkhead" in component.lower() or "u-bending" in component.lower():
                # tic = time.perf_counter()
                # self.points, self.normals, self.offsurface_points = surface_points_normals.generate(self.file, self)
                # toc = time.perf_counter()
                # print(f"generate points and normals took {toc-tic:0.4f} seconds")
                # tic = time.perf_counter()
                # self.best_latent_vector = autodecoder.get_latent_vector(self.points, self.normals, self.offsurface_points, self, component)
                # toc = time.perf_counter()
                # print(f"autodecoder get latent vector took {toc-tic:0.4f} seconds")
                # tic = time.perf_counter()
                # self.verts, self.faces = autodecoder.get_verts_faces(self.best_latent_vector, self, component)
                # toc = time.perf_counter()
                # print(f"autodecoder get verts faces took {toc-tic:0.4f} seconds")
                tic = time.perf_counter()
                # load mesh (STL CAD File)
                mesh = trimesh.load_mesh(self.file, force='mesh')
                # fix mesh if normals are not pointing "up"
                trimesh.repair.fix_inversion(mesh)
                self.verts, self.faces = mesh.vertices, mesh.faces
                toc = time.perf_counter()
                print(f"load STL took {toc-tic:0.4f} seconds")
            self.load_mesh()

    def load_mesh (self):
        self.select_model()
        file = self.file
        if file != "":
            component = self.component_dropdown.currentText()
            indicator = self.indicator_dropdown.currentText()
            if "car door panel" in component.lower():
                L1 = self.slider1.value()
                L2 = self.slider2.value()
                self.label1.setText(f"L1: {L1} mm")
                self.label2.setText(f"L2: {L2} mm")
                prediction.load_mesh(file, self, L1, L2)
            if "bulkhead" in component.lower():
                single_prediction.bulkhead_thinning(self.verts, self.faces, self)
            if "u-bending" in component.lower():
                # points, normals, offsurface_points = surface_points_normals.generate(file[0], self)
                # best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, self, component)
                # verts, faces = autodecoder.get_verts_faces(best_latent_vector, self, component)
                bhf = self.slider1.value() / 100
                friction = self.slider2.value() / 100
                clearance = self.slider3.value() / 100
                thickness = self.slider4.value() / 100
                self.label1.setText(f"Blank Holding Force: {bhf} kN")
                self.label2.setText(f"Friction Coefficient: {friction}")
                self.label3.setText(f"Clearance: {clearance} %")
                self.label4.setText(f"Thickness: {thickness} mm")
                if "thinning" in indicator.lower():
                    single_prediction.ubending_thinning(self.verts, self.faces, self, bhf, friction, clearance, thickness)
                if "displacement" in indicator.lower():
                    single_prediction.ubending_displacement(self.verts, self.faces, self, bhf, friction, clearance, thickness)
        else:
            self.main_view.clear()

    def update_sliders (self):
        component = self.component_dropdown.currentText()
        if "car door panel" in component.lower():
            L1 = self.slider1.value()
            L2 = self.slider2.value()
            self.label1.setText(f"L1: {L1} mm")
            self.label2.setText(f"L2: {L2} mm")
        if "u-bending" in component.lower():
            bhf = self.slider1.value() / 100
            friction = self.slider2.value() / 100
            clearance = self.slider3.value() / 100
            thickness = self.slider4.value() / 100
            self.label1.setText(f"Blank Holding Force: {bhf} kN")
            self.label2.setText(f"Friction Coefficient: {friction}")
            self.label3.setText(f"Clearance: {clearance} %")
            self.label4.setText(f"Thickness: {thickness} mm")

            
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