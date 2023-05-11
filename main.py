
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox, QProgressDialog
from PyQt6.QtCore import QDir, QAbstractTableModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileSystemModel
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem

import sys, os
import pyqtgraph as pg
from stl import mesh
import numpy as np

from python import load, model_control, prediction
from python.windows import main
from python.windows import developer
from python.windows import optimisation
from python.windows import optimisation_setup
from python.windows import optimisation_setup2
from python.windows import optimisation_results

if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv)
    app = QApplication(sys.argv)
    # main = main.Window()
    # main = optimisation_setup.Window()
    # main = optimisation_setup2.Window()
    main = optimisation_results.Window()
    # main = developer.DeveloperWindow()
    # main = developer.TrainNewWindow()
    
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