# import sys

# import numpy as np
# from PySide6.QtWidgets import QApplication
# from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem
# from stl import mesh

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     view = GLViewWidget()
#     # https://ozeki.hu/attachments/116/Eiffel_tower_sample.STL
#     stl_mesh = mesh.Mesh.from_file('temp/b_pillar.stl')

#     points = stl_mesh.points.reshape(-1, 3)
#     faces = np.arange(points.shape[0]).reshape(-1, 3)

#     mesh_data = MeshData(vertexes=points, faces=faces)
#     mesh = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
#     view.addItem(mesh)

#     view.show()
#     app.exec()

from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication
from PyQt6.QtCore import QDir

# from PySide6.QtCore import QFile, QDir, QAbstractItemModel, QModelIndex, Qt
from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem

import sys
import pyqtgraph as pg
from stl import mesh
import numpy as np

from python import load, model_control, prediction

# from mainui import Ui_MainWindow

# uiclass, baseclass = pg.Qt.loadUiType("main.ui")

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

    def change_text (self, text):
        self.selected_label.setText(text+" selected")
        model_control.select_model_type(text)

    def load_mesh (self):
        file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="*.stl")
        points, faces = prediction.load_mesh(file[0], self)

        mesh_data = MeshData(vertexes=points, faces=faces)
        m = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.main_view.addItem(m)


if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv)
    app = QApplication(sys.argv)
    main = MainWindow()
    # main.load(QFile("main.ui"))

    # stl_mesh = mesh.Mesh.from_file('temp/b_pillar.stl')
    # points = stl_mesh.points.reshape(-1, 3)
    # faces = np.arange(points.shape[0]).reshape(-1, 3)
    # mesh_data = MeshData(vertexes=points, faces=faces)
    # m = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
    # main.main_view.addItem(m)

    # input = np.load("temp/input.npy")
    # output = np.load("temp/output.npy")
    # points = np.zeros((256*384,3))
    # points[:,0] = np.tile(np.arange(256),384)

    # thinning = np.multiply(input[0],1-output[0])

    # for i in range(384):
    #     points[256*i:256*(i+1),1] = i

    # for i in range(384):
    #     for j in range(256):
    #         points[(i*256)+j,2] = thinning[j,i]

    # faces = np.arange(points.shape[0]).reshape(-1, 3)
    # mesh_data = MeshData(vertexes=points, faces=faces)
    # m = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
    # main.main_view.addItem(m)

    main.show()
    sys.exit(app.exec())