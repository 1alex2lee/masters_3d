
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QDir

from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import MeshData, GLMeshItem

from stl import mesh
import numpy as np

from python import load, model_control, prediction
from python.windows import developer

class MainWindow(QMainWindow):
# class MainWindow (QUiLoader):
# class MainWindow(uiclass, baseclass):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # self.setupUi(self)

        #Load the UI Page
        uic.loadUi('main.ui', self)

        # ui_file = QFile("mainwindow.ui")
        # ui_file.open(QFile.ReadOnly)
        # self.load(QFile("main.ui"))

        # self.window = QUiLoader().load(QFile("main.ui"))

        self.thinning_button.pressed.connect(lambda: self.change_text(self.thinning_button.text()))
        self.springback_button.pressed.connect(lambda: self.change_text(self.springback_button.text()))
        self.strain_button.pressed.connect(lambda: self.change_text(self.strain_button.text()))

        model = QFileSystemModel()
        model.setRootPath(QDir.currentPath())
        # self.dir_tree.setModel(model)

        self.process_dropdown.addItems(load.processes())
        self.material_dropdown.addItems(load.materials())

        self.load_mesh_button.pressed.connect(lambda: self.load_mesh())

        self.process_dropdown.setCurrentIndex(-1)
        self.material_dropdown.setCurrentIndex(-1)

        self.process_dropdown.currentTextChanged.connect(lambda: model_control.select_process_material(self.process_dropdown.currentText(),self.material_dropdown.currentText()))
        self.material_dropdown.currentTextChanged.connect(lambda: model_control.select_process_material(self.process_dropdown.currentText(),self.material_dropdown.currentText()))

        self.action_developer.triggered.connect(lambda: self.open_developer_window())

    def change_text (self, text):
        self.selected_label.setText(text+" selected")
        model_control.select_model_type(text)

    def load_mesh (self):
        file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="*.stl")
        points, faces = prediction.load_mesh(file[0])

        # mesh_data = MeshData(vertexes=points, faces=faces)
        # m = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        # self.main_view.addItem(m)

        # stl_mesh = mesh.Mesh.from_file('temp/b_pillar.stl')
        # points = stl_mesh.points.reshape(-1, 3)
        # faces = np.arange(points.shape[0]).reshape(-1, 3)

        mesh_data = MeshData(vertexes=points, faces=faces)
        m = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.main_view.addItem(m)

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.developer_window.show()